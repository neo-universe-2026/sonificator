//! Low-latency, musical video sonification.
//!
//! Feed RGBA frames to [`Sonificator::push_frame`] and drain interleaved
//! stereo PCM with [`Sonificator::render`]. The core has no audio-device
//! dependency, so it can be used from a native callback or WASM/WebAudio.

use tunes::consts::C4_MAJOR_PENTATONIC_SCALE;

const TAU: f32 = std::f32::consts::TAU;

/// Normalized image statistics extracted from one video frame.
///
/// All fields are in 0..=1 except [`FrameFeatures::warmth`], which is -1..=1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameFeatures {
    /// Auto-detected visual pulse strength, independent of any particular color.
    /// Blends frame brightness with chromatic salience according to
    /// [`AnalysisOptions::color_weight`].
    pub pulse: f32,
    /// Horizontal centroid of salient visual activity, normalized to 0..=1.
    pub pulse_x: f32,
    /// Vertical centroid of salient visual activity, normalized to 0..=1.
    pub pulse_y: f32,
    /// Spatial spread of salient visual activity, where 0 is clustered and 1 is dispersed.
    pub pulse_dispersion: f32,
    /// Mean perceived brightness (Rec. 709 luma) across the frame.
    pub brightness: f32,
    /// Luma contrast: population standard deviation of luma, scaled to roughly 0..=1.
    pub contrast: f32,
    /// Mean chroma (saturation proxy) across the frame.
    pub saturation: f32,
    /// Salience-weighted dominant hue in 0..=1 (0 red, 1/3 green, 2/3 blue).
    /// Falls back to 0.5 when the frame is achromatic.
    pub hue: f32,
    /// Warm vs cool color balance: +1 warm (red) .. -1 cool (blue). 0 when achromatic.
    pub warmth: f32,
    /// Inter-frame luma change on a coarse 16x16 grid, 0..=1. Zero on the first frame.
    pub motion: f32,
}

const LUMA_GRID_SIZE: usize = 16;

/// Tunable frame-analysis behaviour.
///
/// The defaults analyze any video without assuming a particular palette: pulse
/// strength comes primarily from frame brightness, with a small chromatic
/// contribution so dim-but-vivid content (neon particles on black, for example)
/// remains audible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalysisOptions {
    /// Blend between brightness-based and color-based pulse salience.
    /// 0.0 uses perceived brightness only, 1.0 uses chromatic salience only
    /// (the legacy behaviour). Defaults to 0.25.
    pub color_weight: f32,
    /// Optional hue band to emphasize, as `(center, half_width)` in 0..=1 hue
    /// units (for example `(0.08, 0.1)` targets orange). `None` treats every
    /// hue equally.
    pub hue_focus: Option<(f32, f32)>,
    /// How strongly hues outside `hue_focus` are suppressed, 0.0 (ignore the
    /// focus) to 1.0 (only the focused band counts). Achromatic pixels are
    /// only suppressed at half strength.
    pub hue_focus_strength: f32,
    /// Pulse sensitivity multiplier applied before normalization,
    /// clamped to 0.25..=4.0. Defaults to 1.0.
    pub sensitivity: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            color_weight: 0.25,
            hue_focus: None,
            hue_focus_strength: 1.0,
            sensitivity: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RichFrameAnalysis {
    features: FrameFeatures,
    hue_norm: f32,
    warmth: f32,
    has_chroma: bool,
    luma_grid: [f32; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
    luma_counts: [u32; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
}

impl Default for RichFrameAnalysis {
    fn default() -> Self {
        Self {
            features: FrameFeatures::default(),
            hue_norm: 0.0,
            warmth: 0.0,
            has_chroma: false,
            luma_grid: [0.0; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
            luma_counts: [0; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
        }
    }
}

fn rgb_to_hue_norm(r: f32, g: f32, b: f32, max_c: f32, chroma: f32) -> f32 {
    if chroma <= 1e-5 {
        return 0.0;
    }
    let hue_sector = if (max_c - r).abs() <= 1e-6 {
        ((g - b) / chroma).rem_euclid(6.0)
    } else if (max_c - g).abs() <= 1e-6 {
        (b - r) / chroma + 2.0
    } else {
        (r - g) / chroma + 4.0
    };
    (hue_sector / 6.0).clamp(0.0, 1.0)
}

/// Circular distance between two hues in 0..=1 units.
fn hue_distance(a: f32, b: f32) -> f32 {
    let delta = (a - b).abs() % 1.0;
    delta.min(1.0 - delta)
}

fn hue_focus_factor(pixel_hue: f32, chroma: f32, focus: (f32, f32), strength: f32) -> f32 {
    let (center, half_width) = focus;
    let closeness = (1.0 - hue_distance(pixel_hue, center) / half_width.max(1e-4)).clamp(0.0, 1.0);
    // Achromatic pixels carry no hue evidence, so they are only suppressed at
    // half strength instead of vanishing entirely.
    let evidence = if chroma > 0.04 { 1.0 } else { 0.5 };
    1.0 - strength * evidence * (1.0 - closeness)
}

fn analyze_frame_rich(
    rgba: &[u8],
    width: u32,
    height: u32,
    options: &AnalysisOptions,
) -> RichFrameAnalysis {
    let Some(pixel_count) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return RichFrameAnalysis::default();
    };
    if width == 0 || height == 0 || rgba.len() < pixel_count {
        return RichFrameAnalysis::default();
    }

    let stride_x = (width / 32).max(1);
    let stride_y = (height / 32).max(1);
    let color_weight = options.color_weight.clamp(0.0, 1.0);
    let brightness_weight = 1.0 - color_weight;
    let focus_strength = options.hue_focus_strength.clamp(0.0, 1.0);

    let mut count = 0.0f32;
    let mut luma_sum = 0.0f32;
    let mut luma_sq_sum = 0.0f32;
    let mut chroma_sum = 0.0f32;
    let mut salience_sum = 0.0f32;
    let mut chroma_salience_sum = 0.0f32;
    let mut weighted_x = 0.0f32;
    let mut weighted_y = 0.0f32;
    let mut weighted_x_squared = 0.0f32;
    let mut weighted_y_squared = 0.0f32;
    let mut weighted_hue = 0.0f32;
    let mut weighted_warmth = 0.0f32;

    let mut luma_grid = [0.0f32; LUMA_GRID_SIZE * LUMA_GRID_SIZE];
    let mut luma_counts = [0u32; LUMA_GRID_SIZE * LUMA_GRID_SIZE];

    for y in (0..height).step_by(stride_y as usize) {
        for x in (0..width).step_by(stride_x as usize) {
            let i = (y as usize * width as usize + x as usize) * 4;
            let r = rgba[i] as f32 / 255.0;
            let g = rgba[i + 1] as f32 / 255.0;
            let b = rgba[i + 2] as f32 / 255.0;

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let gx = ((x as usize * LUMA_GRID_SIZE) / width as usize).min(LUMA_GRID_SIZE - 1);
            let gy = ((y as usize * LUMA_GRID_SIZE) / height as usize).min(LUMA_GRID_SIZE - 1);
            let cell = gy * LUMA_GRID_SIZE + gx;
            luma_grid[cell] += luma;
            luma_counts[cell] += 1;

            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let chroma = (max_c - min_c).max(0.0);

            luma_sum += luma;
            luma_sq_sum += luma * luma;
            chroma_sum += chroma;

            // Color-independent brightness component plus a hue-agnostic chromatic
            // component that picks up dim-but-vivid content (orange particles, blue
            // oceans, emerald vegetation, violet aurora, ...).
            let chromatic_salience = if chroma > 0.04 && max_c > 0.08 {
                chroma * chroma * (0.35 + 0.65 * max_c.max(luma))
            } else {
                0.0
            };
            let mixed = brightness_weight * luma + color_weight * chromatic_salience;
            let salience = if let Some(focus) = options.hue_focus {
                let pixel_hue = rgb_to_hue_norm(r, g, b, max_c, chroma);
                mixed * hue_focus_factor(pixel_hue, chroma, focus, focus_strength)
            } else {
                mixed
            };

            let normalized_x = x as f32 / width.saturating_sub(1).max(1) as f32;
            let normalized_y = y as f32 / height.saturating_sub(1).max(1) as f32;
            salience_sum += salience;
            weighted_x += salience * normalized_x;
            weighted_y += salience * normalized_y;
            weighted_x_squared += salience * normalized_x * normalized_x;
            weighted_y_squared += salience * normalized_y * normalized_y;

            if chromatic_salience > 0.0 {
                chroma_salience_sum += chromatic_salience;
                weighted_hue += chromatic_salience * rgb_to_hue_norm(r, g, b, max_c, chroma);
                let pixel_warmth = ((r - b) / (chroma + 1e-4)).clamp(-1.0, 1.0);
                weighted_warmth += chromatic_salience * pixel_warmth;
            }
            count += 1.0;
        }
    }

    for (sum, &n) in luma_grid.iter_mut().zip(luma_counts.iter()) {
        if n > 0 {
            *sum /= n as f32;
        }
    }

    if count == 0.0 {
        return RichFrameAnalysis {
            luma_grid,
            ..RichFrameAnalysis::default()
        };
    }

    let mean_luma = luma_sum / count;
    let luma_variance = (luma_sq_sum / count - mean_luma * mean_luma).max(0.0);
    let sensitivity = options.sensitivity.clamp(0.25, 4.0);
    // Soft-knee normalization: a uniformly bright frame settles well below 1.0
    // while genuine bursts of activity still saturate the pulse.
    let effective_salience = (salience_sum / count) * sensitivity;
    let features = FrameFeatures {
        pulse: if effective_salience > 0.0 {
            (effective_salience / (effective_salience + 0.45)).clamp(0.0, 1.0)
        } else {
            0.0
        },
        pulse_x: if salience_sum > 0.0 {
            (weighted_x / salience_sum).clamp(0.0, 1.0)
        } else {
            0.5
        },
        pulse_y: if salience_sum > 0.0 {
            (weighted_y / salience_sum).clamp(0.0, 1.0)
        } else {
            0.5
        },
        pulse_dispersion: if salience_sum > 0.0 {
            let mean_x = weighted_x / salience_sum;
            let mean_y = weighted_y / salience_sum;
            let variance_x = (weighted_x_squared / salience_sum - mean_x * mean_x).max(0.0);
            let variance_y = (weighted_y_squared / salience_sum - mean_y * mean_y).max(0.0);
            ((variance_x + variance_y) / 0.5).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        },
        brightness: mean_luma.clamp(0.0, 1.0),
        contrast: (luma_variance.sqrt() * 2.0).clamp(0.0, 1.0),
        saturation: (chroma_sum / count).clamp(0.0, 1.0),
        hue: if chroma_salience_sum > 0.0 {
            (weighted_hue / chroma_salience_sum).clamp(0.0, 1.0)
        } else {
            0.5
        },
        warmth: if chroma_salience_sum > 0.0 {
            (weighted_warmth / chroma_salience_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        },
        motion: 0.0,
    };

    RichFrameAnalysis {
        features,
        hue_norm: features.hue,
        warmth: features.warmth,
        has_chroma: chroma_salience_sum > 0.0,
        luma_grid,
        luma_counts,
    }
}

/// Analyze an 8-bit RGBA frame with default analysis options, using at most
/// roughly 32x32 samples.
pub fn analyze_frame(rgba: &[u8], width: u32, height: u32) -> FrameFeatures {
    analyze_frame_with(rgba, width, height, &AnalysisOptions::default())
}

/// Analyze an 8-bit RGBA frame with custom analysis options.
pub fn analyze_frame_with(
    rgba: &[u8],
    width: u32,
    height: u32,
    options: &AnalysisOptions,
) -> FrameFeatures {
    analyze_frame_rich(rgba, width, height, options).features
}

/// Dynamic musical groove state driven by frame activity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrooveState {
    SparseDrift,
    FlowingGroove,
    IntenseClimax,
}

impl GrooveState {
    fn from_activity(activity: f32) -> Self {
        if activity < 0.28 {
            Self::SparseDrift
        } else if activity < 0.65 {
            Self::FlowingGroove
        } else {
            Self::IntenseClimax
        }
    }
}

/// Stateful frame-to-music renderer.
#[derive(Debug, Clone)]
pub struct Sonificator {
    sample_rate: f32,
    analysis_options: AnalysisOptions,
    features: FrameFeatures,
    target: FrameFeatures,
    phase: [f32; 12],
    beat: f32,
    key_ratio: f32,
    progression: [usize; 4],
    fingerprinted: bool,
    pulse_attack: f32,
    pulse_level: f32,
    prev_luma_grid: [f32; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
    prev_luma_counts: [u32; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
    has_prev_frame: bool,
    target_motion: f32,
    motion_energy: f32,
    target_warmth: f32,
    warmth: f32,

    last_step_16th: usize,
    latched_pulse_x: f32,
    latched_pulse_y: f32,
    latched_pulse_dispersion: f32,
    latched_warmth: f32,
    latched_saturation: f32,
    latched_activity: f32,
    latched_melody_freq: f32,
    latched_melody_gate: bool,
    latched_bass_freq: f32,
    latched_bass_gate: bool,
    latched_chord_gate: f32,
    latched_arp_freq: f32,
    latched_arp_gate: bool,
    latched_timbre_h2: f32,
    latched_timbre_h3: f32,
}

impl Sonificator {
    pub fn new(sample_rate: u32) -> Self {
        Self::new_with_options(sample_rate, AnalysisOptions::default())
    }

    pub fn new_with_options(sample_rate: u32, analysis_options: AnalysisOptions) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000) as f32,
            analysis_options,
            features: FrameFeatures::default(),
            target: FrameFeatures::default(),
            phase: [0.0; 12],
            beat: 0.0,
            key_ratio: 1.0,
            progression: [0, 3, 4, 0],
            fingerprinted: false,
            pulse_attack: 0.0,
            pulse_level: 0.0,
            prev_luma_grid: [0.0; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
            prev_luma_counts: [0; LUMA_GRID_SIZE * LUMA_GRID_SIZE],
            has_prev_frame: false,
            target_motion: 0.0,
            motion_energy: 0.0,
            target_warmth: 0.0,
            warmth: 0.0,

            last_step_16th: usize::MAX,
            latched_pulse_x: 0.5,
            latched_pulse_y: 0.5,
            latched_pulse_dispersion: 0.0,
            latched_warmth: 0.0,
            latched_saturation: 0.0,
            latched_activity: 0.0,
            latched_melody_freq: 261.63,
            latched_melody_gate: false,
            latched_bass_freq: 130.81,
            latched_bass_gate: false,
            latched_chord_gate: 0.2,
            latched_arp_freq: 523.25,
            latched_arp_gate: false,
            latched_timbre_h2: 0.25,
            latched_timbre_h3: 0.10,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate as u32
    }

    pub fn analysis_options(&self) -> AnalysisOptions {
        self.analysis_options
    }

    /// Replace the frame-analysis options. Takes effect from the next pushed frame.
    pub fn set_analysis_options(&mut self, analysis_options: AnalysisOptions) {
        self.analysis_options = analysis_options;
    }

    pub fn features(&self) -> FrameFeatures {
        self.features
    }

    pub fn push_frame(&mut self, rgba: &[u8], width: u32, height: u32) {
        let rich = analyze_frame_rich(rgba, width, height, &self.analysis_options);
        let mut next_target = rich.features;

        // Track a coarse 16x16 spatial luma grid across consecutive frames so motion
        // drives tempo, and so dim monochrome footage still gains pulse energy and a
        // spatial centroid / spread.
        let mut motion_sum = 0.0f32;
        let mut motion_x = 0.0f32;
        let mut motion_y = 0.0f32;
        let mut motion_x2 = 0.0f32;
        let mut motion_y2 = 0.0f32;
        if self.has_prev_frame {
            let mut motion_cells = 0.0f32;
            for (idx, (&curr, &prev)) in rich
                .luma_grid
                .iter()
                .zip(self.prev_luma_grid.iter())
                .enumerate()
            {
                let delta = (curr - prev).abs();
                let nx = (idx % LUMA_GRID_SIZE) as f32 / (LUMA_GRID_SIZE - 1) as f32;
                let ny = (idx / LUMA_GRID_SIZE) as f32 / (LUMA_GRID_SIZE - 1) as f32;
                motion_sum += delta;
                motion_x += delta * nx;
                motion_y += delta * ny;
                motion_x2 += delta * nx * nx;
                motion_y2 += delta * ny * ny;
                if rich.luma_counts[idx] > 0 || self.prev_luma_counts[idx] > 0 {
                    motion_cells += 1.0;
                }
            }
            // Average over cells this video actually samples so small or
            // low-resolution frames are not diluted by empty grid cells.
            let mean_delta = motion_sum / motion_cells.max(1.0);
            self.target_motion = (mean_delta * 6.0).clamp(0.0, 1.0);

            // If a video frame is essentially black or frozen (no salient content),
            // inter-frame motion still supplies pulse energy and spatial features so
            // greyscale or dark footage remains expressive.
            if next_target.pulse <= 0.002 && motion_sum > 0.02 {
                let mx = (motion_x / motion_sum).clamp(0.0, 1.0);
                let my = (motion_y / motion_sum).clamp(0.0, 1.0);
                let vx = (motion_x2 / motion_sum - mx * mx).max(0.0);
                let vy = (motion_y2 / motion_sum - my * my).max(0.0);
                next_target.pulse = self.target_motion;
                next_target.pulse_x = mx;
                next_target.pulse_y = my;
                next_target.pulse_dispersion = ((vx + vy) / 0.5).sqrt().clamp(0.0, 1.0);
            }
        } else {
            self.target_motion = 0.0;
        }
        next_target.motion = self.target_motion;
        self.prev_luma_grid = rich.luma_grid;
        self.prev_luma_counts = rich.luma_counts;
        self.has_prev_frame = true;

        let pulse_rise = next_target.pulse - self.target.pulse;
        self.pulse_attack = self
            .pulse_attack
            .max((pulse_rise * 3.5 + self.target_motion * 0.6).clamp(0.0, 1.0));
        self.target = next_target;
        self.target_warmth = rich.warmth;

        if !self.fingerprinted && (self.target.pulse > 0.002 || self.target_motion > 0.02) {
            let color_seed = if rich.has_chroma { rich.hue_norm } else { 0.5 };
            let fingerprint = (self.target.pulse_x * 0.45 + color_seed * 0.55).clamp(0.0, 1.0);
            let semitone = (fingerprint * 11.0).round() as i32 % 12;
            self.key_ratio = 2.0f32.powf(semitone as f32 / 12.0);
            self.progression = match (fingerprint * 3.0) as usize % 3 {
                0 => [0, 3, 4, 0],
                1 => [0, 4, 5, 3],
                _ => [0, 2, 5, 3],
            };
            self.fingerprinted = true;
        }
    }

    /// Render `frames` stereo frames as left/right interleaved PCM samples.
    pub fn render(&mut self, frames: usize) -> Vec<f32> {
        let mut output = vec![0.0; frames.saturating_mul(2)];
        self.render_into(&mut output);
        output
    }

    /// Allocation-free variant for realtime callbacks.
    pub fn render_into(&mut self, output: &mut [f32]) {
        const DENSITY_RANK: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
        let (stereo_frames, _) = output.as_chunks_mut::<2>();
        for pair in stereo_frames {
            self.features = smooth(self.features, self.target, 0.0025);
            self.motion_energy += (self.target_motion - self.motion_energy) * 0.0015;
            self.warmth += (self.target_warmth - self.warmth) * 0.0015;
            let pulse_response = if self.target.pulse > self.pulse_level {
                0.0008
            } else {
                0.00015
            };
            self.pulse_level += (self.target.pulse - self.pulse_level) * pulse_response;
            let pulse_gate = ((self.pulse_level - 0.18) / 0.52).clamp(0.0, 1.0).powf(0.7);

            // Governs transitions between Sparse Drift (< 0.28), Flowing Groove (0.28..0.65), and Intense Climax (>= 0.65)
            let base_activity = pulse_gate * 0.72 + self.features.pulse_dispersion * 0.28;
            let activity = (base_activity + self.motion_energy * 0.22).clamp(0.0, 1.0);
            let tempo = 68.0 + activity * 28.0;
            let beats_per_sample = tempo / 60.0 / self.sample_rate;

            let step_16th = ((self.beat * 4.0) as usize) % 64;
            let step_fract = (self.beat * 4.0).fract();

            // 16th-note step latch: quantizes note triggers and prevents continuous mid-note microtonal bending
            if step_16th != self.last_step_16th {
                self.last_step_16th = step_16th;
                self.latched_pulse_x = self.features.pulse_x;
                self.latched_pulse_y = self.features.pulse_y;
                self.latched_pulse_dispersion = self.features.pulse_dispersion;
                self.latched_warmth = self.warmth;
                self.latched_saturation = self.features.saturation;
                self.latched_activity = activity;
                let groove_state = GrooveState::from_activity(self.latched_activity);

                let bar = (step_16th / 16) % self.progression.len();
                let chord = self.progression[bar];
                let step_in_bar = step_16th % 16;
                let scale = C4_MAJOR_PENTATONIC_SCALE;

                let root = scale[chord.min(scale.len() - 1)] * self.key_ratio * 0.5;
                let fifth_harmonic = root * 1.5;

                // Syncopated Bass on Root/Fifth
                match groove_state {
                    GrooveState::SparseDrift => {
                        if step_in_bar == 0 {
                            self.latched_bass_freq = root;
                            self.latched_bass_gate = true;
                        } else if step_in_bar == 8 {
                            self.latched_bass_freq = fifth_harmonic;
                            self.latched_bass_gate = true;
                        } else {
                            self.latched_bass_gate = false;
                        }
                    }
                    GrooveState::FlowingGroove => match step_in_bar {
                        0 | 10 => {
                            self.latched_bass_freq = root;
                            self.latched_bass_gate = true;
                        }
                        6 | 14 => {
                            self.latched_bass_freq = fifth_harmonic;
                            self.latched_bass_gate = true;
                        }
                        _ => {
                            self.latched_bass_gate = false;
                        }
                    },
                    GrooveState::IntenseClimax => match step_in_bar {
                        0 | 6 | 8 => {
                            self.latched_bass_freq = root;
                            self.latched_bass_gate = true;
                        }
                        3 | 11 | 14 => {
                            self.latched_bass_freq = fifth_harmonic;
                            self.latched_bass_gate = true;
                        }
                        _ => {
                            self.latched_bass_gate = false;
                        }
                    },
                }

                // Chord Gating
                self.latched_chord_gate = match groove_state {
                    GrooveState::SparseDrift => 0.35 + self.latched_activity * 0.2,
                    GrooveState::FlowingGroove => {
                        if matches!(step_in_bar, 0 | 3 | 6 | 8 | 11 | 14) {
                            0.65
                        } else {
                            0.15
                        }
                    }
                    GrooveState::IntenseClimax => {
                        if matches!(step_in_bar, 0 | 3 | 6 | 8 | 10 | 12 | 14) {
                            0.85
                        } else {
                            0.05
                        }
                    }
                };

                // Melody: pulse_y shifts register (octave/fifth lifts) and overtone brightness,
                // while spectral warmth blends warm 2nd harmonic vs cool crystalline 3rd harmonic.
                let register_lift = if self.latched_pulse_y >= 0.65 {
                    2.0 // Octave lift
                } else if self.latched_pulse_y >= 0.35 {
                    1.5 // Fifth lift
                } else {
                    1.0 // Base register
                };

                let warmth_bias = (self.latched_warmth * 0.08).clamp(-0.08, 0.08);
                self.latched_timbre_h2 =
                    (0.18 + self.latched_pulse_y * 0.35 + warmth_bias).clamp(0.05, 0.65);
                // Saturated frames sound brighter: the crystalline 3rd harmonic rises
                // with mean chroma while warm frames lean into the 2nd harmonic.
                self.latched_timbre_h3 =
                    (0.06 + self.latched_pulse_y * 0.30 + self.latched_saturation * 0.15
                        - warmth_bias)
                        .clamp(0.02, 0.55);

                // Quantized pentatonic call-and-response motifs selected via step-latched pulse_x
                // (never produces out-of-scale microtonal semitones against the chord pad).
                const MOTIF_BANK: [[usize; 8]; 3] = [
                    [0, 2, 1, 2, 4, 2, 1, 0],
                    [0, 1, 3, 4, 3, 2, 1, 0],
                    [2, 4, 3, 2, 1, 2, 4, 2],
                ];
                let motif_variant = ((self.latched_pulse_x * 2.99) as usize).min(2);
                let phrase = MOTIF_BANK[motif_variant];
                let melody_trigger = match groove_state {
                    GrooveState::SparseDrift => matches!(step_in_bar, 0 | 6 | 10),
                    GrooveState::FlowingGroove => matches!(step_in_bar, 0 | 3 | 6 | 8 | 11 | 14),
                    GrooveState::IntenseClimax => {
                        matches!(step_in_bar, 0 | 2 | 3 | 6 | 8 | 10 | 11 | 14)
                    }
                };

                if melody_trigger {
                    let step_idx = (step_in_bar / 2 + bar) % phrase.len();
                    let note_deg = (phrase[step_idx] + chord) % scale.len();
                    self.latched_melody_freq = scale[note_deg] * self.key_ratio * register_lift;
                    self.latched_melody_gate = true;
                } else {
                    self.latched_melody_gate = false;
                }

                // Dispersion-Modulated Arpeggio Groove Density
                let density_slots = (self.latched_pulse_dispersion * 16.0).floor() as usize;
                let arp_trigger = DENSITY_RANK[step_in_bar] <= density_slots;
                if arp_trigger {
                    let arp_phrase = [0usize, 2, 4, 2, 1, 3, 4, 2];
                    let arp_idx = (step_16th + chord + bar) % arp_phrase.len();
                    let arp_deg = (arp_phrase[arp_idx] + chord) % scale.len();
                    let octave_shift = 2.0
                        * if self.latched_pulse_dispersion > 0.55 && step_in_bar % 2 == 1 {
                            2.0
                        } else {
                            1.0
                        };
                    self.latched_arp_freq = scale[arp_deg] * self.key_ratio * octave_shift;
                    self.latched_arp_gate = true;
                } else {
                    self.latched_arp_gate = false;
                }
            }

            let bar = (step_16th / 16) % self.progression.len();
            let chord = self.progression[bar];
            let scale = C4_MAJOR_PENTATONIC_SCALE;
            let root = scale[chord.min(scale.len() - 1)] * self.key_ratio * 0.5;
            let third = scale[(chord + 2).min(scale.len() - 1)] * self.key_ratio * 0.5;
            let fifth = scale[(chord + 4).min(scale.len() - 1)] * self.key_ratio * 0.5;

            let drone = sine(&mut self.phase[0], root * 0.5, self.sample_rate) * 0.045;

            let bass_env = if self.latched_bass_gate {
                (-step_fract * 4.5).exp()
            } else {
                0.0
            };
            let bass = (sine(
                &mut self.phase[1],
                self.latched_bass_freq * 0.5,
                self.sample_rate,
            ) + sine(
                &mut self.phase[10],
                self.latched_bass_freq,
                self.sample_rate,
            ) * 0.35)
                * bass_env
                * (0.14 + pulse_gate * 0.34);

            let pad_envelope = self.latched_chord_gate * (1.0 - step_fract * 0.3);
            let pad = (sine(&mut self.phase[2], root * 2.0, self.sample_rate)
                + sine(&mut self.phase[3], third * 2.0, self.sample_rate) * 0.65
                + sine(&mut self.phase[4], fifth * 2.0, self.sample_rate) * 0.45)
                * pad_envelope
                * (0.05 + pulse_gate * 0.16);

            let melody_env = if self.latched_melody_gate {
                (-step_fract * 3.5).exp()
            } else {
                0.0
            };
            let melody = (sine(
                &mut self.phase[5],
                self.latched_melody_freq,
                self.sample_rate,
            ) + sine(
                &mut self.phase[6],
                self.latched_melody_freq * 2.0,
                self.sample_rate,
            ) * self.latched_timbre_h2
                + sine(
                    &mut self.phase[7],
                    self.latched_melody_freq * 3.0,
                    self.sample_rate,
                ) * self.latched_timbre_h3)
                * melody_env
                * (0.09 + pulse_gate * 0.34);

            let arp_env = if self.latched_arp_gate {
                (-step_fract * 6.0).exp()
            } else {
                0.0
            };
            let arpeggio = (sine(&mut self.phase[8], self.latched_arp_freq, self.sample_rate)
                + sine(
                    &mut self.phase[9],
                    self.latched_arp_freq * 2.0,
                    self.sample_rate,
                ) * 0.25)
                * arp_env
                * (0.035 + pulse_gate * 0.24);

            let pulse_accent = self
                .features
                .pulse
                .max(pulse_gate)
                .max(self.pulse_attack)
                .clamp(0.0, 1.0);
            // Overall frame brightness lifts the bed of the mix slightly, while motion
            // adds transient shimmer through pulse_attack above.
            let visual_bloom = (0.30 + pulse_gate * 0.70)
                * (0.7 + pulse_accent * 0.3)
                * (0.85 + self.features.brightness * 0.3);
            let mix = (drone + bass + pad + melody + arpeggio) * visual_bloom;
            // Master boost with a soft limiter: quiet passages gain linearly while
            // peaks fold back gracefully instead of hard-clipping.
            let sample = (mix * 4.5).tanh();
            let pan = (self.features.pulse_x * 2.0 - 1.0) * 0.18;
            pair[0] = (sample * (1.0 - pan)).clamp(-1.0, 1.0);
            pair[1] = (sample * (1.0 + pan)).clamp(-1.0, 1.0);

            self.beat += beats_per_sample;
            if self.beat >= 16.0 {
                self.beat -= 16.0;
            }
            self.pulse_attack *= 0.99992;
        }
    }
}

pub fn smooth(current: FrameFeatures, target: FrameFeatures, amount: f32) -> FrameFeatures {
    FrameFeatures {
        pulse: current.pulse + (target.pulse - current.pulse) * amount,
        pulse_x: current.pulse_x + (target.pulse_x - current.pulse_x) * amount,
        pulse_y: current.pulse_y + (target.pulse_y - current.pulse_y) * amount,
        pulse_dispersion: current.pulse_dispersion
            + (target.pulse_dispersion - current.pulse_dispersion) * amount,
        brightness: current.brightness + (target.brightness - current.brightness) * amount,
        contrast: current.contrast + (target.contrast - current.contrast) * amount,
        saturation: current.saturation + (target.saturation - current.saturation) * amount,
        hue: current.hue + (target.hue - current.hue) * amount,
        warmth: current.warmth + (target.warmth - current.warmth) * amount,
        motion: current.motion + (target.motion - current.motion) * amount,
    }
}

fn sine(phase: &mut f32, frequency: f32, sample_rate: f32) -> f32 {
    let value = (*phase * TAU).sin();
    *phase = (*phase + frequency / sample_rate).fract();
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brightness_drives_pulse_regardless_of_color() {
        // White on black: no chroma at all, yet the pulse must register.
        let frame = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let features = analyze_frame(&frame, 2, 1);
        // Even after the default 25% chromatic weighting, a pure brightness
        // flash registers well.
        assert!(features.pulse > 0.4);
        assert_eq!(features.saturation, 0.0);
        assert_eq!(features.hue, 0.5);
        assert_eq!(features.warmth, 0.0);
    }

    #[test]
    fn reports_basic_image_stats() {
        assert_eq!(analyze_frame(&[255u8; 4], 1, 1).brightness, 1.0);
        assert_eq!(analyze_frame(&[0u8; 4], 1, 1).brightness, 0.0);

        let grey = analyze_frame(&[128u8, 128, 128, 255], 1, 1);
        assert!((grey.brightness - 128.0 / 255.0).abs() < 1e-3);
        assert_eq!(grey.contrast, 0.0);

        let black_and_white = [255u8; 4]
            .iter()
            .copied()
            .chain([0u8; 4])
            .collect::<Vec<_>>();
        let contrasty = analyze_frame(&black_and_white, 2, 1);
        assert!(
            contrasty.contrast > 0.9,
            "half black / half white is maximal contrast"
        );
    }

    #[test]
    fn reports_saturation_and_hue_per_palette() {
        let grey = analyze_frame(&[128u8, 128, 128, 255], 1, 1);
        assert_eq!(grey.saturation, 0.0);

        let orange = analyze_frame(&[240u8, 80, 20, 255], 1, 1);
        assert!(orange.saturation > 0.5);
        assert!(
            orange.hue < 0.1,
            "orange hue should sit near red: {}",
            orange.hue
        );
        assert!(orange.warmth > 0.5);

        let blue = analyze_frame(&[20u8, 120, 240, 255], 1, 1);
        assert!(
            blue.hue > 0.55 && blue.hue < 0.65,
            "blue hue expected near 2/3: {}",
            blue.hue
        );
        assert!(blue.warmth < -0.5);
    }

    #[test]
    fn color_weight_shifts_pulse_from_color_to_brightness() {
        let grey = [128u8, 128, 128, 255];
        let brightness_only = analyze_frame_with(
            &grey,
            1,
            1,
            &AnalysisOptions {
                color_weight: 0.0,
                ..AnalysisOptions::default()
            },
        );
        let color_only = analyze_frame_with(
            &grey,
            1,
            1,
            &AnalysisOptions {
                color_weight: 1.0,
                ..AnalysisOptions::default()
            },
        );
        assert!(brightness_only.pulse > 0.5);
        assert_eq!(
            color_only.pulse, 0.0,
            "grey frames have no chromatic salience"
        );
    }

    #[test]
    fn hue_focus_suppresses_other_hues() {
        let options = AnalysisOptions {
            hue_focus: Some((0.045, 0.1)),
            hue_focus_strength: 1.0,
            ..AnalysisOptions::default()
        };
        let orange = analyze_frame_with(&[240u8, 80, 20, 255], 1, 1, &options);
        let blue = analyze_frame_with(&[20u8, 120, 240, 255], 1, 1, &options);
        assert!(orange.pulse > 0.4);
        assert!(
            blue.pulse < 0.1,
            "blue should be filtered out by orange focus"
        );
    }

    #[test]
    fn sensitivity_scales_pulse() {
        let frame = [64u8, 64, 64, 255];
        let quiet = analyze_frame_with(
            &frame,
            1,
            1,
            &AnalysisOptions {
                sensitivity: 0.25,
                ..AnalysisOptions::default()
            },
        );
        let normal = analyze_frame(&frame, 1, 1);
        let loud = analyze_frame_with(
            &frame,
            1,
            1,
            &AnalysisOptions {
                sensitivity: 4.0,
                ..AnalysisOptions::default()
            },
        );
        assert!(quiet.pulse < normal.pulse);
        assert!(normal.pulse < loud.pulse);
    }

    #[test]
    fn sonificator_exposes_analysis_options() {
        let mut sonificator = Sonificator::new(48_000);
        assert_eq!(sonificator.analysis_options(), AnalysisOptions::default());
        sonificator.set_analysis_options(AnalysisOptions {
            sensitivity: 2.0,
            hue_focus: Some((0.3, 0.2)),
            ..AnalysisOptions::default()
        });
        assert_eq!(sonificator.analysis_options().sensitivity, 2.0);
        assert_eq!(sonificator.analysis_options().hue_focus, Some((0.3, 0.2)));
    }

    #[test]
    fn locates_pulse_on_the_x_axis() {
        let left = [240u8, 80, 20, 255, 0, 0, 0, 255];
        let right = [0u8, 0, 0, 255, 240, 80, 20, 255];
        assert!(analyze_frame(&left, 2, 1).pulse_x < 0.1);
        assert!(analyze_frame(&right, 2, 1).pulse_x > 0.9);
    }

    #[test]
    fn locates_brightness_pulse_on_the_y_axis() {
        let top = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let bottom = [0u8, 0, 0, 255, 255, 255, 255, 255];
        assert!(analyze_frame(&top, 1, 2).pulse_y < 0.1);
        assert!(analyze_frame(&bottom, 1, 2).pulse_y > 0.9);
    }

    #[test]
    fn measures_pulse_dispersion() {
        let clustered = [
            240u8, 80, 20, 255, 240, 80, 20, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        ];
        let dispersed = [
            240u8, 80, 20, 255, 0, 0, 0, 255, 0, 0, 0, 255, 240, 80, 20, 255,
        ];
        let clustered_dispersion = analyze_frame(&clustered, 2, 2).pulse_dispersion;
        let dispersed_dispersion = analyze_frame(&dispersed, 2, 2).pulse_dispersion;
        assert!(clustered_dispersion < dispersed_dispersion);
    }

    #[test]
    fn renders_bounded_stereo_audio() {
        let mut sonificator = Sonificator::new(48_000);
        sonificator.push_frame(&[90, 150, 220, 255].repeat(4), 2, 2);
        let audio = sonificator.render(512);
        assert_eq!(audio.len(), 1024);
        assert!(
            audio
                .iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        );
        assert!(audio.iter().any(|sample| sample.abs() > 0.001));
    }

    #[test]
    fn visual_fingerprint_changes_the_phrase() {
        let mut warm = Sonificator::new(48_000);
        let mut cool = Sonificator::new(48_000);
        warm.push_frame(&[240, 80, 20, 255].repeat(4), 2, 2);
        cool.push_frame(&[20, 80, 240, 255].repeat(4), 2, 2);
        let warm_audio = warm.render(2048);
        let cool_audio = cool.render(2048);
        let difference: f32 = warm_audio
            .iter()
            .zip(cool_audio)
            .map(|(left, right)| (left - right).abs())
            .sum();
        assert!(difference > 0.1);
    }

    #[test]
    fn pulse_level_controls_loudness() {
        let mut bright = Sonificator::new(48_000);
        let mut dark = Sonificator::new(48_000);
        bright.push_frame(&[240, 80, 20, 255].repeat(4), 2, 2);
        dark.push_frame(&[2, 2, 2, 255].repeat(4), 2, 2);
        let bright_audio = bright.render(4096);
        let dark_audio = dark.render(4096);
        let bright_energy: f32 = bright_audio.iter().map(|sample| sample.abs()).sum();
        let dark_energy: f32 = dark_audio.iter().map(|sample| sample.abs()).sum();
        assert!(bright_energy > dark_energy * 1.2);
    }

    #[test]
    fn pulse_y_shifts_melodic_register_and_timbre() {
        let mut son_low = Sonificator::new(48_000);
        let mut son_high = Sonificator::new(48_000);

        let frame_low = FrameFeatures {
            pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.1,
            pulse_dispersion: 0.3,
            ..FrameFeatures::default()
        };
        let frame_high = FrameFeatures {
            pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.9,
            pulse_dispersion: 0.3,
            ..FrameFeatures::default()
        };

        son_low.target = frame_low;
        son_low.features = frame_low;
        son_high.target = frame_high;
        son_high.features = frame_high;

        let audio_low = son_low.render(48_000);
        let audio_high = son_high.render(48_000);

        assert_ne!(audio_low, audio_high);

        let diff_low: f32 = audio_low.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let diff_high: f32 = audio_high.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        assert!(
            diff_high > diff_low * 1.3,
            "higher pulse_y should produce higher frequency register and richer timbre: diff_high={diff_high}, diff_low={diff_low}"
        );
        let zcr_low = audio_low
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zcr_high = audio_high
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        assert!(
            zcr_high > zcr_low,
            "higher pulse_y should produce higher zero-crossing rate: zcr_high={zcr_high}, zcr_low={zcr_low}"
        );
    }

    #[test]
    fn pulse_dispersion_modulates_groove_density() {
        let mut son_sparse = Sonificator::new(48_000);
        let mut son_dense = Sonificator::new(48_000);

        let frame_sparse = FrameFeatures {
            pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.05,
            ..FrameFeatures::default()
        };
        let frame_dense = FrameFeatures {
            pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.95,
            ..FrameFeatures::default()
        };

        son_sparse.target = frame_sparse;
        son_sparse.features = frame_sparse;
        son_dense.target = frame_dense;
        son_dense.features = frame_dense;

        let audio_sparse = son_sparse.render(48_000);
        let audio_dense = son_dense.render(48_000);

        assert_ne!(audio_sparse, audio_dense);

        let energy_sparse: f32 = audio_sparse.iter().map(|s| s * s).sum();
        let energy_dense: f32 = audio_dense.iter().map(|s| s * s).sum();
        assert!(
            energy_dense > energy_sparse,
            "dense groove should have higher energy/note density than sparse groove: dense={energy_dense}, sparse={energy_sparse}"
        );
    }

    #[test]
    fn rhythmic_groove_creates_metric_contrast() {
        let mut son = Sonificator::new(48_000);
        let frame = FrameFeatures {
            pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.5,
            ..FrameFeatures::default()
        };
        son.target = frame;
        son.features = frame;

        let audio = son.render(48_000 * 2);

        let chunk_size = 48_000 / 8;
        let mut chunk_energies = Vec::new();
        for chunk in audio.chunks(chunk_size) {
            let e: f32 = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
            chunk_energies.push(e);
        }
        let max_e = chunk_energies.iter().copied().fold(0.0f32, f32::max);
        let min_e = chunk_energies.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(max_e > 0.0);
        assert!(
            max_e > min_e * 1.5,
            "groove must create metric contrast across beats and subdivisions: max={max_e}, min={min_e}"
        );
    }

    #[test]
    fn motion_drives_activity_on_monochrome_video() {
        let mut son = Sonificator::new(48_000);
        let dark_frame = [16u8, 16, 16, 255].repeat(4 * 4);
        let bright_gray_frame = [210u8, 210, 210, 255].repeat(4 * 4);

        son.push_frame(&dark_frame, 4, 4);
        assert!(
            son.target.pulse < 0.15,
            "dark static frame has little pulse"
        );

        son.push_frame(&bright_gray_frame, 4, 4);
        assert!(son.target.pulse > 0.5);
        assert!(son.target_motion > 0.5);

        let audio = son.render(4096);
        let energy: f32 = audio.iter().map(|s| s.abs()).sum();
        assert!(energy > 1.0);
    }

    #[test]
    fn motion_fallback_drives_a_black_video() {
        let mut son = Sonificator::new(48_000);
        let black = [0u8; 4 * 4 * 4];
        let mut moving_dot = [0u8; 4 * 4 * 4];
        moving_dot[12] = 220;
        moving_dot[13] = 220;
        moving_dot[14] = 220;
        // Dark content with a single lit pixel is detected directly by brightness.
        son.push_frame(&black, 4, 4);
        son.push_frame(&moving_dot, 4, 4);
        assert!(son.target.pulse > 0.05, "a lit dot on black must register");
        assert!(son.target.pulse_x > 0.7, "the dot is on the right side");

        // Content vanishing into pure black keeps a pulse through inter-frame motion.
        son.push_frame(&black, 4, 4);
        assert!(
            son.target.pulse > 0.01,
            "fading to black keeps the pulse alive"
        );
        assert!(
            son.target.pulse_x > 0.7,
            "the motion centroid stays on the right"
        );
    }
}
