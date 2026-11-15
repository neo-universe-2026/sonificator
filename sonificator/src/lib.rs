//! Low-latency, musical video sonification.
//!
//! Feed RGBA frames to [`Sonificator::push_frame`] and drain interleaved
//! stereo PCM with [`Sonificator::render`]. The core has no audio-device
//! dependency, so it can be used from a native callback or WASM/WebAudio.

use tunes::consts::C4_MAJOR_PENTATONIC_SCALE;

const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameFeatures {
    /// Calibrated warm/orange emission level for the CO2 globe video.
    pub orange_pulse: f32,
    /// Horizontal position of the orange pulse, normalized to 0..=1.
    pub pulse_x: f32,
    /// Vertical position of the orange pulse, normalized to 0..=1.
    pub pulse_y: f32,
    /// Spatial spread of orange pulses, where 0 is clustered and 1 is dispersed.
    pub pulse_dispersion: f32,
}

/// Analyze an 8-bit RGBA frame using at most roughly 32x32 samples.
pub fn analyze_frame(rgba: &[u8], width: u32, height: u32) -> FrameFeatures {
    let Some(pixel_count) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return FrameFeatures::default();
    };
    if width == 0 || height == 0 || rgba.len() < pixel_count {
        return FrameFeatures::default();
    }

    let stride_x = (width / 32).max(1);
    let stride_y = (height / 32).max(1);
    let mut count = 0.0;
    let mut orange_energy = 0.0;
    let mut orange_weighted_x = 0.0;
    let mut orange_weighted_y = 0.0;
    let mut orange_weighted_x_squared = 0.0;
    let mut orange_weighted_y_squared = 0.0;

    for y in (0..height).step_by(stride_y as usize) {
        for x in (0..width).step_by(stride_x as usize) {
            let i = (y as usize * width as usize + x as usize) * 4;
            let r = rgba[i] as f32 / 255.0;
            let g = rgba[i + 1] as f32 / 255.0;
            let b = rgba[i + 2] as f32 / 255.0;
            let orange = if r > 0.08 && r > g * 1.08 && g > b * 1.05 {
                (r - b).max(0.0) * (r - g).max(0.0) * r
            } else {
                0.0
            };
            orange_energy += orange;
            let normalized_x = x as f32 / width.saturating_sub(1).max(1) as f32;
            let normalized_y = y as f32 / height.saturating_sub(1).max(1) as f32;
            orange_weighted_x += orange * normalized_x;
            orange_weighted_y += orange * normalized_y;
            orange_weighted_x_squared += orange * normalized_x * normalized_x;
            orange_weighted_y_squared += orange * normalized_y * normalized_y;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return FrameFeatures::default();
    }
    FrameFeatures {
        // The supplied animation is nearly black except for sparse orange
        // particles. This gamma lift makes those pulses musically useful.
        orange_pulse: ((orange_energy / count / 0.0015).clamp(0.0, 1.0)).sqrt(),
        pulse_x: if orange_energy > 0.0 {
            (orange_weighted_x / orange_energy).clamp(0.0, 1.0)
        } else {
            0.5
        },
        pulse_y: if orange_energy > 0.0 {
            (orange_weighted_y / orange_energy).clamp(0.0, 1.0)
        } else {
            0.5
        },
        pulse_dispersion: if orange_energy > 0.0 {
            let mean_x = orange_weighted_x / orange_energy;
            let mean_y = orange_weighted_y / orange_energy;
            let variance_x = (orange_weighted_x_squared / orange_energy - mean_x * mean_x).max(0.0);
            let variance_y = (orange_weighted_y_squared / orange_energy - mean_y * mean_y).max(0.0);
            ((variance_x + variance_y) / 0.5).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        },
    }
}

/// Dynamic musical groove state driven by frame activity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrooveState {
    SparseDrift,
    FlowingGroove,
    IntenseClimax,
}

/// Stateful frame-to-music renderer.
#[derive(Debug, Clone)]
pub struct Sonificator {
    sample_rate: f32,
    features: FrameFeatures,
    target: FrameFeatures,
    phase: [f32; 12],
    beat: f32,
    key_ratio: f32,
    progression: [usize; 4],
    fingerprinted: bool,
    pulse_attack: f32,
    pulse_level: f32,

    last_step_16th: usize,
    latched_pulse_x: f32,
    latched_pulse_y: f32,
    latched_pulse_dispersion: f32,
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
        Self {
            sample_rate: sample_rate.max(8_000) as f32,
            features: FrameFeatures::default(),
            target: FrameFeatures::default(),
            phase: [0.0; 12],
            beat: 0.0,
            key_ratio: 1.0,
            progression: [0, 3, 4, 0],
            fingerprinted: false,
            pulse_attack: 0.0,
            pulse_level: 0.0,

            last_step_16th: usize::MAX,
            latched_pulse_x: 0.5,
            latched_pulse_y: 0.5,
            latched_pulse_dispersion: 0.0,
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

    pub fn features(&self) -> FrameFeatures {
        self.features
    }

    pub fn push_frame(&mut self, rgba: &[u8], width: u32, height: u32) {
        let next_target = analyze_frame(rgba, width, height);
        let pulse_rise = next_target.orange_pulse - self.target.orange_pulse;
        self.pulse_attack = self.pulse_attack.max((pulse_rise * 3.5).clamp(0.0, 1.0));
        self.target = next_target;
        if !self.fingerprinted && self.target.orange_pulse > 0.002 {
            let semitone = (self.target.pulse_x * 11.0).round() as i32 % 12;
            self.key_ratio = 2.0f32.powf(semitone as f32 / 12.0);
            self.progression = match (self.target.pulse_x * 3.0) as usize % 3 {
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
            let pulse_response = if self.target.orange_pulse > self.pulse_level {
                0.0008
            } else {
                0.00015
            };
            self.pulse_level += (self.target.orange_pulse - self.pulse_level) * pulse_response;
            let pulse_gate = ((self.pulse_level - 0.18) / 0.52).clamp(0.0, 1.0).powf(0.7);

            // Governs transitions between Sparse Drift (< 0.28), Flowing Groove (0.28..0.65), and Intense Climax (>= 0.65)
            let activity = (pulse_gate * 0.72 + self.features.pulse_dispersion * 0.28).clamp(0.0, 1.0);
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
                self.latched_activity = activity;

                let bar = (step_16th / 16) % self.progression.len();
                let chord = self.progression[bar];
                let step_in_bar = step_16th % 16;
                let scale = C4_MAJOR_PENTATONIC_SCALE;

                let root = scale[chord.min(scale.len() - 1)] * self.key_ratio * 0.5;
                let fifth_harmonic = root * 1.5;

                // Syncopated Bass on Root/Fifth
                if self.latched_activity < 0.28 {
                    // Sparse Drift: downbeat root plucks
                    if step_in_bar == 0 {
                        self.latched_bass_freq = root;
                        self.latched_bass_gate = true;
                    } else if step_in_bar == 8 {
                        self.latched_bass_freq = fifth_harmonic;
                        self.latched_bass_gate = true;
                    } else {
                        self.latched_bass_gate = false;
                    }
                } else if self.latched_activity < 0.65 {
                    // Flowing Groove: syncopated root and fifth offbeats
                    match step_in_bar {
                        0 => {
                            self.latched_bass_freq = root;
                            self.latched_bass_gate = true;
                        }
                        6 => {
                            self.latched_bass_freq = fifth_harmonic;
                            self.latched_bass_gate = true;
                        }
                        10 => {
                            self.latched_bass_freq = root;
                            self.latched_bass_gate = true;
                        }
                        14 => {
                            self.latched_bass_freq = fifth_harmonic;
                            self.latched_bass_gate = true;
                        }
                        _ => {
                            self.latched_bass_gate = false;
                        }
                    }
                } else {
                    // Intense Climax: driving syncopated groove
                    match step_in_bar {
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
                    }
                }

                // Chord Gating
                if self.latched_activity < 0.28 {
                    self.latched_chord_gate = 0.35 + self.latched_activity * 0.2;
                } else if self.latched_activity < 0.65 {
                    if matches!(step_in_bar, 0 | 3 | 6 | 8 | 11 | 14) {
                        self.latched_chord_gate = 0.65;
                    } else {
                        self.latched_chord_gate = 0.15;
                    }
                } else {
                    if matches!(step_in_bar, 0 | 3 | 6 | 8 | 10 | 12 | 14) {
                        self.latched_chord_gate = 0.85;
                    } else {
                        self.latched_chord_gate = 0.05;
                    }
                }

                // Melody: pulse_y shifts register (octave/fifth lifts) and timbre
                let register_lift = if self.latched_pulse_y >= 0.65 {
                    2.0 // Octave lift
                } else if self.latched_pulse_y >= 0.35 {
                    1.5 // Fifth lift
                } else {
                    1.0 // Base register
                };

                self.latched_timbre_h2 = 0.18 + self.latched_pulse_y * 0.35;
                self.latched_timbre_h3 = 0.06 + self.latched_pulse_y * 0.30;

                // Step-latched pitch quantization prevents mid-note bending
                let semitone_shift = ((self.latched_pulse_x - 0.5) * 4.0).round();
                let pulse_pitch = 2.0f32.powf(semitone_shift / 12.0);

                let phrase = [0usize, 2, 1, 2, 4, 2, 1, 0];
                let melody_trigger = if self.latched_activity < 0.28 {
                    matches!(step_in_bar, 0 | 6 | 10)
                } else if self.latched_activity < 0.65 {
                    matches!(step_in_bar, 0 | 3 | 6 | 8 | 11 | 14)
                } else {
                    matches!(step_in_bar, 0 | 2 | 3 | 6 | 8 | 10 | 11 | 14)
                };

                if melody_trigger {
                    let step_idx = (step_in_bar / 2 + bar) % phrase.len();
                    let note_deg = (phrase[step_idx] + chord) % scale.len();
                    self.latched_melody_freq =
                        scale[note_deg] * self.key_ratio * register_lift * pulse_pitch;
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

            let drone = sine(&mut self.phase[0], root * 0.5, self.sample_rate) * 0.02;

            let bass_env = if self.latched_bass_gate {
                (-step_fract * 4.5).exp()
            } else {
                0.0
            };
            let bass = (sine(&mut self.phase[1], self.latched_bass_freq * 0.5, self.sample_rate)
                + sine(&mut self.phase[10], self.latched_bass_freq, self.sample_rate) * 0.35)
                * bass_env
                * (0.08 + pulse_gate * 0.12);

            let pad_envelope = self.latched_chord_gate * (1.0 - step_fract * 0.3);
            let pad = (sine(&mut self.phase[2], root * 2.0, self.sample_rate)
                + sine(&mut self.phase[3], third * 2.0, self.sample_rate) * 0.65
                + sine(&mut self.phase[4], fifth * 2.0, self.sample_rate) * 0.45)
                * pad_envelope
                * (0.03 + pulse_gate * 0.06);

            let melody_env = if self.latched_melody_gate {
                (-step_fract * 3.5).exp()
            } else {
                0.0
            };
            let melody = (sine(&mut self.phase[5], self.latched_melody_freq, self.sample_rate)
                + sine(&mut self.phase[6], self.latched_melody_freq * 2.0, self.sample_rate)
                    * self.latched_timbre_h2
                + sine(&mut self.phase[7], self.latched_melody_freq * 3.0, self.sample_rate)
                    * self.latched_timbre_h3)
                * melody_env
                * (0.06 + pulse_gate * 0.14);

            let arp_env = if self.latched_arp_gate {
                (-step_fract * 6.0).exp()
            } else {
                0.0
            };
            let arpeggio = (sine(&mut self.phase[8], self.latched_arp_freq, self.sample_rate)
                + sine(&mut self.phase[9], self.latched_arp_freq * 2.0, self.sample_rate) * 0.25)
                * arp_env
                * (0.02 + pulse_gate * 0.10);

            let pulse_accent = self
                .features
                .orange_pulse
                .max(pulse_gate)
                .max(self.pulse_attack)
                .clamp(0.0, 1.0);
            let visual_bloom = (0.16 + pulse_gate * 0.84) * (0.7 + pulse_accent * 0.3);
            let sample = (drone + bass + pad + melody + arpeggio) * visual_bloom;
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
        orange_pulse: current.orange_pulse + (target.orange_pulse - current.orange_pulse) * amount,
        pulse_x: current.pulse_x + (target.pulse_x - current.pulse_x) * amount,
        pulse_y: current.pulse_y + (target.pulse_y - current.pulse_y) * amount,
        pulse_dispersion: current.pulse_dispersion
            + (target.pulse_dispersion - current.pulse_dispersion) * amount,
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
    fn analyzes_frame_features() {
        let frame = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let features = analyze_frame(&frame, 2, 1);
        assert_eq!(features.orange_pulse, 0.0);
    }

    #[test]
    fn locates_orange_pulse_on_the_x_axis() {
        let left = [240u8, 80, 20, 255, 0, 0, 0, 255];
        let right = [0u8, 0, 0, 255, 240, 80, 20, 255];
        assert!(analyze_frame(&left, 2, 1).pulse_x < 0.1);
        assert!(analyze_frame(&right, 2, 1).pulse_x > 0.9);
    }

    #[test]
    fn locates_orange_pulse_on_the_y_axis() {
        let top = [240u8, 80, 20, 255, 0, 0, 0, 255];
        let bottom = [0u8, 0, 0, 255, 240, 80, 20, 255];
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
    fn orange_level_controls_loudness() {
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
            orange_pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.1,
            pulse_dispersion: 0.3,
        };
        let frame_high = FrameFeatures {
            orange_pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.9,
            pulse_dispersion: 0.3,
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
            orange_pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.05,
        };
        let frame_dense = FrameFeatures {
            orange_pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.95,
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
            orange_pulse: 0.6,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.5,
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
}
