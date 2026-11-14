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
            let tempo = 68.0 + pulse_gate * 24.0;
            let beats_per_sample = tempo / 60.0 / self.sample_rate;
            let beat = self.beat;
            let bar = (beat / 4.0) as usize;
            let chord = self.progression[bar % self.progression.len()];
            let scale = C4_MAJOR_PENTATONIC_SCALE;
            let root = scale[chord.min(scale.len() - 1)] * self.key_ratio * 0.5;
            let third = scale[(chord + 2).min(scale.len() - 1)] * self.key_ratio * 0.5;
            let fifth = scale[(chord + 4).min(scale.len() - 1)] * self.key_ratio * 0.5;
            let phrase = [0usize, 2, 1, 2, 4, 2, 1, 0];
            let step = ((beat * 2.0) as usize) % phrase.len();
            let pulse_pitch = 2.0f32.powf((self.features.pulse_x - 0.5) * 2.0 / 12.0);
            let melody = scale[(phrase[step] + chord) % scale.len()] * self.key_ratio * pulse_pitch;
            let eighth = (beat * 2.0).fract();
            let sixteenth = (beat * 4.0).fract();
            let note_envelope = (-eighth * 8.0).exp();
            let shimmer_envelope = (-sixteenth * 14.0).exp();
            let pulse_accent = self
                .features
                .orange_pulse
                .max(pulse_gate)
                .max(self.pulse_attack)
                .clamp(0.0, 1.0);
            let drone = sine(&mut self.phase[0], root * 0.5, self.sample_rate) * 0.10;
            let bass = sine(&mut self.phase[1], root, self.sample_rate)
                * (0.08 + pulse_gate * 0.10)
                * (1.0 - (beat % 4.0) / 4.0 * 0.35);
            let pad = (sine(&mut self.phase[2], root * 2.0, self.sample_rate)
                + sine(&mut self.phase[3], third * 2.0, self.sample_rate) * 0.65
                + sine(&mut self.phase[4], fifth * 2.0, self.sample_rate) * 0.45)
                * (0.035 + pulse_gate * 0.07);
            let piano = (sine(&mut self.phase[5], melody, self.sample_rate)
                + sine(&mut self.phase[6], melody * 2.0, self.sample_rate) * 0.32
                + sine(&mut self.phase[7], melody * 3.0, self.sample_rate) * 0.12)
                * note_envelope
                * (0.025 + pulse_gate * 0.16);
            let arpeggio_note = scale
                [(phrase[(step + 2) % phrase.len()] + chord + bar) % scale.len()]
                * self.key_ratio;
            let arpeggio = (sine(&mut self.phase[8], arpeggio_note * 2.0, self.sample_rate)
                + sine(&mut self.phase[9], arpeggio_note * 4.0, self.sample_rate) * 0.2)
                * shimmer_envelope
                * (0.008 + pulse_gate * 0.12)
                * 0.25;
            let visual_bloom = (0.16 + pulse_gate * 0.84) * (0.7 + pulse_accent * 0.3);
            let sample = (drone + bass + pad + piano + arpeggio) * visual_bloom;
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

fn smooth(current: FrameFeatures, target: FrameFeatures, amount: f32) -> FrameFeatures {
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
}
