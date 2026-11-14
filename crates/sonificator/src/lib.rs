//! Low-latency, musical video sonification.
//!
//! Feed RGBA frames to [`Sonificator::push_frame`] and drain interleaved
//! stereo PCM with [`Sonificator::render`]. The core has no audio-device
//! dependency, so it can be used from a native callback or WASM/WebAudio.

use tunes::consts::{C4_MAJOR_PENTATONIC_SCALE, C4_MAJOR_SCALE};

const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameFeatures {
    pub brightness: f32,
    pub saturation: f32,
    /// Calibrated warm/orange emission level for the CO2 globe video.
    pub orange_pulse: f32,
    pub motion: f32,
    pub edge_density: f32,
    pub centroid_x: f32,
    pub centroid_y: f32,
}

/// Analyze an 8-bit RGBA frame using at most roughly 32x32 samples.
pub fn analyze_frame(
    rgba: &[u8],
    width: u32,
    height: u32,
    previous: Option<&[u8]>,
) -> FrameFeatures {
    if width == 0 || height == 0 || rgba.len() < width as usize * height as usize * 4 {
        return FrameFeatures::default();
    }

    let stride_x = (width / 32).max(1);
    let stride_y = (height / 32).max(1);
    let mut count = 0.0;
    let mut brightness = 0.0;
    let mut saturation = 0.0;
    let mut orange_energy = 0.0;
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    let mut motion = 0.0;
    let mut edges = 0.0;

    for y in (0..height).step_by(stride_y as usize) {
        for x in (0..width).step_by(stride_x as usize) {
            let i = (y as usize * width as usize + x as usize) * 4;
            let r = rgba[i] as f32 / 255.0;
            let g = rgba[i + 1] as f32 / 255.0;
            let b = rgba[i + 2] as f32 / 255.0;
            let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            brightness += luma;
            saturation += if max > 0.0 { (max - min) / max } else { 0.0 };
            let orange = if r > 0.08 && r > g * 1.08 && g > b * 1.05 {
                (r - b).max(0.0) * (r - g).max(0.0) * r
            } else {
                0.0
            };
            orange_energy += orange;
            weighted_x += luma * x as f32 / width.saturating_sub(1).max(1) as f32;
            weighted_y += luma * y as f32 / height.saturating_sub(1).max(1) as f32;

            if let Some(previous) = previous {
                if previous.len() >= rgba.len() {
                    let old = (0.2126 * previous[i] as f32
                        + 0.7152 * previous[i + 1] as f32
                        + 0.0722 * previous[i + 2] as f32)
                        / 255.0;
                    motion += (luma - old).abs();
                }
            }
            if x + stride_x < width {
                let right = (y as usize * width as usize + (x + stride_x) as usize) * 4;
                let right_luma = (0.2126 * rgba[right] as f32
                    + 0.7152 * rgba[right + 1] as f32
                    + 0.0722 * rgba[right + 2] as f32)
                    / 255.0;
                edges += (luma - right_luma).abs();
            }
            count += 1.0;
        }
    }
    if count == 0.0 {
        return FrameFeatures::default();
    }
    FrameFeatures {
        brightness: (brightness / count).clamp(0.0, 1.0),
        saturation: (saturation / count).clamp(0.0, 1.0),
        // The supplied animation is nearly black except for sparse orange
        // particles. This gamma lift makes those pulses musically useful.
        orange_pulse: ((orange_energy / count / 0.0015).clamp(0.0, 1.0)).sqrt(),
        motion: (motion / count).clamp(0.0, 1.0),
        edge_density: (edges / count).clamp(0.0, 1.0),
        centroid_x: (weighted_x / brightness.max(0.001)).clamp(0.0, 1.0),
        centroid_y: (weighted_y / brightness.max(0.001)).clamp(0.0, 1.0),
    }
}

/// Stateful frame-to-music renderer.
#[derive(Debug, Clone)]
pub struct Sonificator {
    sample_rate: f32,
    features: FrameFeatures,
    target: FrameFeatures,
    previous_frame: Vec<u8>,
    phase: [f32; 8],
    beat: f32,
}

impl Sonificator {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000) as f32,
            features: FrameFeatures::default(),
            target: FrameFeatures::default(),
            previous_frame: Vec::new(),
            phase: [0.0; 8],
            beat: 0.0,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate as u32
    }

    pub fn features(&self) -> FrameFeatures {
        self.features
    }

    /// Submit a frame. The frame is copied for motion analysis on the next call.
    pub fn push_frame(&mut self, rgba: &[u8], width: u32, height: u32) {
        let previous = (!self.previous_frame.is_empty()).then_some(self.previous_frame.as_slice());
        self.target = analyze_frame(rgba, width, height, previous);
        self.previous_frame.clear();
        self.previous_frame.extend_from_slice(rgba);
    }

    /// Render `frames` stereo frames as left/right interleaved PCM samples.
    pub fn render(&mut self, frames: usize) -> Vec<f32> {
        let mut output = vec![0.0; frames.saturating_mul(2)];
        self.render_into(&mut output);
        output
    }

    /// Allocation-free variant for realtime callbacks.
    pub fn render_into(&mut self, output: &mut [f32]) {
        for pair in output.chunks_exact_mut(2) {
            self.features = smooth(self.features, self.target, 0.0025);
            let tempo = 82.0 + self.features.orange_pulse * 14.0 + self.features.motion * 8.0;
            let beats_per_sample = tempo / 60.0 / self.sample_rate;
            let beat = self.beat;
            let chord = [0usize, 3, 4, 0][((beat / 4.0) as usize) % 4];
            let scale = if self.features.orange_pulse > 0.35 {
                C4_MAJOR_PENTATONIC_SCALE
            } else {
                C4_MAJOR_SCALE
            };
            let root = scale[chord.min(scale.len() - 1)];
            let third = scale[(chord + 2).min(scale.len() - 1)] * 0.5;
            let fifth = scale[(chord + 4).min(scale.len() - 1)] * 0.5;
            let melody_index = ((beat * (1.0 + self.features.edge_density * 0.5)
                + self.features.centroid_x * 2.0) as usize)
                % scale.len();
            let melody = scale[melody_index];
            let pulse = (beat * 0.5).fract();
            let envelope = (1.0 - pulse * 1.8).clamp(0.0, 1.0);
            let pulse_accent = ((self.features.orange_pulse - 0.25) / 0.75).clamp(0.0, 1.0);
            let bass = sine(&mut self.phase[0], third, self.sample_rate) * 0.24;
            let pad = (sine(&mut self.phase[1], root, self.sample_rate)
                + sine(&mut self.phase[2], fifth, self.sample_rate) * 0.55
                + sine(&mut self.phase[3], root * 2.0, self.sample_rate) * 0.25)
                * (0.10 + self.features.brightness * 0.05 + pulse_accent * 0.07);
            let lead = sine(&mut self.phase[4], melody, self.sample_rate)
                * envelope
                * (0.035 + pulse_accent * 0.10);
            let sparkle = sine(&mut self.phase[5], melody * 2.0, self.sample_rate)
                * envelope
                * self.features.edge_density
                * 0.025;
            let kick = if (beat * 2.0).fract() < 0.08 {
                (1.0 - (beat * 2.0).fract() * 12.0).max(0.0) * (0.025 + pulse_accent * 0.10)
            } else {
                0.0
            };
            let sample = (bass + pad + lead + sparkle + kick)
                * (0.52 + self.features.brightness * 0.20 + pulse_accent * 0.18);
            let pan = (self.features.centroid_x * 2.0 - 1.0) * 0.18;
            pair[0] = (sample * (1.0 - pan)).clamp(-1.0, 1.0);
            pair[1] = (sample * (1.0 + pan)).clamp(-1.0, 1.0);
            self.beat += beats_per_sample;
            if self.beat >= 16.0 {
                self.beat -= 16.0;
            }
        }
    }
}

fn smooth(current: FrameFeatures, target: FrameFeatures, amount: f32) -> FrameFeatures {
    FrameFeatures {
        brightness: current.brightness + (target.brightness - current.brightness) * amount,
        saturation: current.saturation + (target.saturation - current.saturation) * amount,
        orange_pulse: current.orange_pulse + (target.orange_pulse - current.orange_pulse) * amount,
        motion: current.motion + (target.motion - current.motion) * amount,
        edge_density: current.edge_density + (target.edge_density - current.edge_density) * amount,
        centroid_x: current.centroid_x + (target.centroid_x - current.centroid_x) * amount,
        centroid_y: current.centroid_y + (target.centroid_y - current.centroid_y) * amount,
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
    fn analyzes_brightness_and_centroid() {
        let frame = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let features = analyze_frame(&frame, 2, 1, None);
        assert!(features.brightness > 0.45 && features.brightness < 0.55);
        assert!(features.centroid_x < 0.1);
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
}
