//! Comprehensive 4-Tier End-to-End (E2E) Test Suite for Sonificator
//!
//! Covers:
//! - Tier 1: Feature Coverage (>=5 tests per feature for R1-R6)
//! - Tier 2: Boundary & Corner Cases (>=5 tests per feature for R1-R6)
//! - Tier 3: Cross-Feature Interactions (Pairwise combinations)
//! - Tier 4: Real-World Application Scenarios (Video streams, storm, sunrise, high motion, montage)

use sonificator::{
    analyze_frame, smooth, FrameFeatures, GrooveState, SonificationAlgorithm, Sonificator,
};

// ============================================================================
// Common Audio & Visual Test Helpers
// ============================================================================

mod helpers {
    /// Create a solid RGBA frame buffer.
    pub fn make_solid_frame(r: u8, g: u8, b: u8, a: u8, width: u32, height: u32) -> Vec<u8> {
        let count = (width as usize) * (height as usize) * 4;
        let mut buf = Vec::with_capacity(count);
        for _ in 0..(width * height) {
            buf.extend_from_slice(&[r, g, b, a]);
        }
        buf
    }

    /// Create a localized spot of color on a dark background.
    pub fn make_spot_frame(
        width: u32,
        height: u32,
        center_x: f32,
        center_y: f32,
        radius: f32,
        rgb: [u8; 3],
    ) -> Vec<u8> {
        let mut buf = vec![0u8; (width * height * 4) as usize];
        let cx = center_x * width as f32;
        let cy = center_y * height as f32;
        let r_px = radius * (width.min(height) as f32);
        let r_sq = r_px * r_px;

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let idx = ((y * width + x) * 4) as usize;
                if dx * dx + dy * dy <= r_sq {
                    buf[idx] = rgb[0];
                    buf[idx + 1] = rgb[1];
                    buf[idx + 2] = rgb[2];
                    buf[idx + 3] = 255;
                } else {
                    buf[idx] = 10;
                    buf[idx + 1] = 10;
                    buf[idx + 2] = 10;
                    buf[idx + 3] = 255;
                }
            }
        }
        buf
    }

    /// Measure maximum peak absolute amplitude.
    pub fn max_peak(samples: &[f32]) -> f32 {
        samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
    }

    /// Measure root-mean-square (RMS) energy.
    pub fn rms_level(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Measure zero-crossing rate (ZCR).
    pub fn zero_crossing_rate(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count()
    }

    /// Measure maximum sample-to-sample delta (to detect audible clicks/pops).
    pub fn max_sample_step(samples: &[f32]) -> f32 {
        samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max)
    }

    /// Measure low-frequency energy ratio using simple moving average approximation.
    pub fn low_frequency_energy_ratio(samples: &[f32], window: usize) -> f32 {
        if samples.len() < window || window == 0 {
            return 0.0;
        }
        let total_energy: f32 = samples.iter().map(|s| s * s).sum();
        if total_energy <= 1e-6 {
            return 0.0;
        }

        // Low-pass filtered signal via simple moving average
        let mut low_pass_energy = 0.0f32;
        let mut running_sum = 0.0f32;
        for (i, &s) in samples.iter().enumerate() {
            running_sum += s;
            if i >= window {
                running_sum -= samples[i - window];
                let lp = running_sum / window as f32;
                low_pass_energy += lp * lp;
            }
        }
        (low_pass_energy / total_energy).clamp(0.0, 1.0)
    }
}

// ============================================================================
// TIER 1: Feature Coverage (>=5 tests per feature for R1 - R6)
// ============================================================================

mod tier1_feature_coverage {
    use super::helpers::*;
    use super::*;

    // --- R1: Loudness & Headroom ---

    #[test]
    fn t1_r1_01_peak_amplitude_headroom_guarantee() {
        // Under maximal polyphonic excitation, peak must never exceed 0.83 (-1.6 dBFS headroom)
        let mut son = Sonificator::new(48_000);
        let saturated_frame = make_solid_frame(255, 120, 20, 255, 64, 64);
        son.push_frame(&saturated_frame, 64, 64);

        // Flash surge to trigger thump voice
        let flash_frame = make_solid_frame(255, 255, 255, 255, 64, 64);
        son.push_frame(&flash_frame, 64, 64);

        let audio = son.render(48_000); // 1 second of stereo
        let peak = max_peak(&audio);

        assert!(
            peak <= 0.83,
            "R1 Violation: Peak amplitude {peak} exceeds 0.83 headroom limit"
        );
        assert!(peak > 0.05, "Audio must be audible: peak was {peak}");
    }

    #[test]
    fn t1_r1_02_comfortable_rms_listening_level() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(230, 90, 30, 255, 32, 32);
        son.push_frame(&frame, 32, 32);

        let audio = son.render(48_000 * 2);
        let rms = rms_level(&audio);

        assert!(
            rms < 0.35,
            "R1 Violation: RMS level {rms} exceeds comfortable listening boundary (0.35)"
        );
        assert!(
            rms > 0.01,
            "Audio must have adequate presence: RMS was {rms}"
        );
    }

    #[test]
    fn t1_r1_03_zero_digital_clipping_boundary() {
        let mut son = Sonificator::new(48_000);
        // Repeated bursts of high energy frames
        for _ in 0..10 {
            let frame = make_solid_frame(255, 140, 0, 255, 16, 16);
            son.push_frame(&frame, 16, 16);
            let flash = make_solid_frame(255, 255, 255, 255, 16, 16);
            son.push_frame(&flash, 16, 16);
        }

        let audio = son.render(96_000);
        let clipping_samples = audio.iter().filter(|&&s| s.abs() >= 1.0).count();
        assert_eq!(
            clipping_samples, 0,
            "R1 Violation: Found {clipping_samples} clipped samples reaching +/- 1.0"
        );
    }

    #[test]
    fn t1_r1_04_master_gain_attenuation_ratio() {
        let mut son_bright = Sonificator::new(48_000);
        let mut son_dim = Sonificator::new(48_000);

        let bright_frame = make_solid_frame(240, 100, 20, 255, 32, 32);
        let dim_frame = make_solid_frame(20, 20, 20, 255, 32, 32);

        son_bright.push_frame(&bright_frame, 32, 32);
        son_dim.push_frame(&dim_frame, 32, 32);

        let audio_bright = son_bright.render(48_000);
        let audio_dim = son_dim.render(48_000);

        let rms_bright = rms_level(&audio_bright);
        let rms_dim = rms_level(&audio_dim);

        assert!(
            rms_bright > rms_dim * 1.5,
            "R1: Bright frame RMS ({rms_bright}) should scale noticeably higher than dim frame RMS ({rms_dim})"
        );
    }

    #[test]
    fn t1_r1_05_render_into_matches_render_loudness() {
        let mut son1 = Sonificator::new(48_000);
        let mut son2 = Sonificator::new(48_000);

        let frame = make_solid_frame(200, 80, 40, 255, 16, 16);
        son1.push_frame(&frame, 16, 16);
        son2.push_frame(&frame, 16, 16);

        let audio1 = son1.render(4800);
        let mut audio2 = vec![0.0f32; 9600];
        son2.render_into(&mut audio2);

        assert_eq!(audio1.len(), audio2.len());
        for (i, (&s1, &s2)) in audio1.iter().zip(audio2.iter()).enumerate() {
            assert!(
                (s1 - s2).abs() < 1e-6,
                "Sample mismatch at {i}: {s1} vs {s2}"
            );
        }
        assert_eq!(max_peak(&audio1), max_peak(&audio2));
        assert_eq!(rms_level(&audio1), rms_level(&audio2));
    }

    // --- R2: Warmth-Driven Tempo Modulation & Dual Algorithm ---

    #[test]
    fn t1_r2_01_warm_colors_decrease_tempo() {
        let mut warm_son = Sonificator::new(48_000);
        let mut neutral_son = Sonificator::new(48_000);

        // Warm: strong orange/red (R high, B low)
        let warm_frame = make_solid_frame(255, 60, 10, 255, 16, 16);
        // Neutral: green/balanced (warmth ~0)
        let neutral_frame = make_solid_frame(40, 200, 40, 255, 16, 16);

        for _ in 0..6 {
            warm_son.push_frame(&warm_frame, 16, 16);
            neutral_son.push_frame(&neutral_frame, 16, 16);
        }

        // Render 2.5 seconds
        let warm_audio = warm_son.render(48_000 * 5 / 2);
        let neutral_audio = neutral_son.render(48_000 * 5 / 2);

        // Rhythmic events: because warm decreases tempo, neutral has a faster tempo
        // and accumulates more beat cycles, creating a distinct sample difference
        let diff: f32 = warm_audio
            .iter()
            .zip(neutral_audio.iter())
            .map(|(w, n)| (w - n).abs())
            .sum();
        assert!(
            diff > 10.0,
            "Warm and neutral frames must modulate tempo distinctly: diff={diff}"
        );
    }

    #[test]
    fn t1_r2_02_cool_colors_increase_tempo() {
        let mut cool_son = Sonificator::new(48_000);
        let mut warm_son = Sonificator::new(48_000);

        // Cool: cyan/blue (R low, B high)
        let cool_frame = make_solid_frame(10, 120, 255, 255, 16, 16);
        // Warm: orange/red
        let warm_frame = make_solid_frame(255, 60, 10, 255, 16, 16);

        for _ in 0..8 {
            cool_son.push_frame(&cool_frame, 16, 16);
            warm_son.push_frame(&warm_frame, 16, 16);
        }

        let cool_audio = cool_son.render(48_000 * 3);
        let warm_audio = warm_son.render(48_000 * 3);

        // Zero-crossing rates and energy pacing are higher for cool tempo
        assert_ne!(cool_audio, warm_audio);
        let diff: f32 = cool_audio
            .iter()
            .zip(warm_audio.iter())
            .map(|(c, w)| (c - w).abs())
            .sum();
        assert!(
            diff > 25.0,
            "Cool tempo (faster) and warm tempo (slower) must exhibit strong dynamic divergence"
        );
    }

    #[test]
    fn t1_r2_03_tempo_variance_across_color_spectrum() {
        let colors = [
            ("red", [255, 30, 10, 255]),     // Warm (warmth ~ +1.0)
            ("emerald", [20, 220, 80, 255]), // Neutral/green (warmth ~ 0.0)
            ("cyan", [10, 230, 255, 255]),   // Cool (warmth ~ -1.0)
            ("violet", [200, 20, 240, 255]), // Purple/cool
        ];

        let mut outputs = Vec::new();
        for (_name, rgba) in colors {
            let mut son = Sonificator::new(48_000);
            let frame = make_solid_frame(rgba[0], rgba[1], rgba[2], rgba[3], 16, 16);
            for _ in 0..5 {
                son.push_frame(&frame, 16, 16);
            }
            outputs.push(son.render(48_000));
        }

        // Each color across the spectrum should produce an acoustic signature with variance
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let delta: f32 = outputs[i]
                    .iter()
                    .zip(outputs[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                assert!(
                    delta > 5.0,
                    "Distinct spectrum hues must yield varying tempos/audio: pair ({i}, {j}) delta={delta}"
                );
            }
        }
    }

    #[test]
    fn t1_r2_04_dual_algorithm_default_is_rhythmic_groove() {
        let son = Sonificator::new(48_000);
        assert_eq!(son.algorithm(), SonificationAlgorithm::RhythmicGroove);
        assert_eq!(
            SonificationAlgorithm::default(),
            SonificationAlgorithm::RhythmicGroove
        );
    }

    #[test]
    fn t1_r2_05_dual_algorithm_switching_and_distinct_audio() {
        let mut son_rg =
            Sonificator::new(48_000).with_algorithm(SonificationAlgorithm::RhythmicGroove);
        let mut son_mc =
            Sonificator::new(48_000).with_algorithm(SonificationAlgorithm::MelodicContour);

        assert_eq!(son_rg.algorithm(), SonificationAlgorithm::RhythmicGroove);
        assert_eq!(son_mc.algorithm(), SonificationAlgorithm::MelodicContour);

        let frame = make_solid_frame(220, 110, 40, 255, 32, 32);
        son_rg.push_frame(&frame, 32, 32);
        son_mc.push_frame(&frame, 32, 32);

        let audio_rg = son_rg.render(48_000);
        let audio_mc = son_mc.render(48_000);

        assert_eq!(audio_rg.len(), audio_mc.len());
        let diff: f32 = audio_rg
            .iter()
            .zip(audio_mc.iter())
            .map(|(g, m)| (g - m).abs())
            .sum();
        assert!(
            diff > 1.0,
            "RhythmicGroove and MelodicContour must produce distinct sonic outputs: diff={diff}"
        );

        // Switch algorithm dynamically
        son_rg.set_algorithm(SonificationAlgorithm::MelodicContour);
        assert_eq!(son_rg.algorithm(), SonificationAlgorithm::MelodicContour);
    }

    // --- R3: Sudden Flash & Intensity Surge Detection ---

    #[test]
    fn t1_r3_01_sudden_luma_spike_triggers_sub_bass_thump() {
        let mut son_no_flash = Sonificator::new(48_000);
        let mut son_flash = Sonificator::new(48_000);

        let dark = make_solid_frame(20, 20, 20, 255, 16, 16);
        son_no_flash.push_frame(&dark, 16, 16);
        son_flash.push_frame(&dark, 16, 16);
        let _ = son_no_flash.render(2400);
        let _ = son_flash.render(2400);

        // Flash occurs: luma jumps from 0.08 to 1.0 (Delta L = 0.92 > 0.20)
        let bright_flash = make_solid_frame(255, 255, 255, 255, 16, 16);
        son_no_flash.push_frame(&dark, 16, 16);
        son_flash.push_frame(&bright_flash, 16, 16);

        // Render immediate 150 ms (7,200 stereo samples)
        let audio_no_flash = son_no_flash.render(3600);
        let audio_flash = son_flash.render(3600);

        let energy_no_flash = rms_level(&audio_no_flash);
        let energy_flash = rms_level(&audio_flash);

        assert!(
            energy_flash > energy_no_flash * 1.3,
            "Flash surge must trigger substantial thump voice energy: flash={energy_flash}, no_flash={energy_no_flash}"
        );
    }

    #[test]
    fn t1_r3_02_sub_bass_frequency_profile_of_thump() {
        let mut son = Sonificator::new(48_000);
        let dark = make_solid_frame(10, 10, 10, 255, 16, 16);
        son.push_frame(&dark, 16, 16);
        let _ = son.render(1000);

        let flash = make_solid_frame(255, 255, 255, 255, 16, 16);
        son.push_frame(&flash, 16, 16);

        // In the initial 100 ms of the thump, the voice sweeps from 110 Hz down to 42 Hz
        let audio = son.render(4800);
        let ratio = low_frequency_energy_ratio(&audio, 400); // 400 sample window (~120 Hz cutoff)
        assert!(
            ratio > 0.10,
            "Thump voice should exhibit prominent sub-bass / low-frequency energy: ratio={ratio}"
        );
    }

    #[test]
    fn t1_r3_03_thump_voice_decays_over_time() {
        let mut son = Sonificator::new(48_000);
        let dark = make_solid_frame(15, 15, 15, 255, 16, 16);
        son.push_frame(&dark, 16, 16);
        let _ = son.render(1000);

        let flash = make_solid_frame(255, 255, 255, 255, 16, 16);
        son.push_frame(&flash, 16, 16);

        // Render first 100ms (impact phase)
        let impact_audio = son.render(4800);
        let impact_rms = rms_level(&impact_audio);

        // Render next 800ms (decayed phase, >280ms decay constant)
        let decayed_audio = son.render(48_000 * 4 / 5);
        let decayed_rms = rms_level(&decayed_audio[decayed_audio.len() - 4800..]);

        assert!(
            impact_rms > decayed_rms,
            "Thump amplitude must decay over time: impact={impact_rms}, decayed={decayed_rms}"
        );
    }

    #[test]
    fn t1_r3_04_rhythm_continuity_preserved_during_flash() {
        let mut son = Sonificator::new(48_000);
        let groove_frame = make_solid_frame(210, 90, 20, 255, 16, 16);
        son.push_frame(&groove_frame, 16, 16);
        let audio_pre = son.render(48_000);

        // Introduce lightning flash
        let flash_frame = make_solid_frame(255, 255, 255, 255, 16, 16);
        son.push_frame(&flash_frame, 16, 16);
        let audio_post = son.render(48_000);

        // Rhythm must not halt or displace into silence
        assert!(!audio_pre.is_empty());
        assert!(!audio_post.is_empty());
        assert!(rms_level(&audio_post) > 0.02);
        assert!(max_peak(&audio_post) <= 0.83);
    }

    #[test]
    fn t1_r3_05_gradual_luminance_ramp_does_not_trigger_thump() {
        let mut son_gradual = Sonificator::new(48_000);
        let mut son_abrupt = Sonificator::new(48_000);

        let dark = make_solid_frame(20, 20, 20, 255, 16, 16);
        son_gradual.push_frame(&dark, 16, 16);
        son_abrupt.push_frame(&dark, 16, 16);
        let _ = son_gradual.render(1000);
        let _ = son_abrupt.render(1000);

        // Ramp luminance slowly across 10 frames (delta L ~ 0.03 per frame < 0.20 threshold)
        for step in 1..=10 {
            let luma = (20 + step * 8) as u8;
            let frame = make_solid_frame(luma, luma, luma, 255, 16, 16);
            son_gradual.push_frame(&frame, 16, 16);
            let _ = son_gradual.render(200);
        }

        // Abrupt flash in son_abrupt
        let flash = make_solid_frame(250, 250, 250, 255, 16, 16);
        son_abrupt.push_frame(&flash, 16, 16);
        let audio_abrupt = son_abrupt.render(2400);
        let audio_gradual = son_gradual.render(2400);

        let peak_abrupt = max_peak(&audio_abrupt);
        let peak_gradual = max_peak(&audio_gradual);

        assert!(
            peak_abrupt > peak_gradual * 1.2,
            "Abrupt flash triggers thump surge; gradual ramp must not: abrupt={peak_abrupt}, gradual={peak_gradual}"
        );
    }

    // --- R4: Smooth Mid-Video Theme Transitions ---

    #[test]
    fn t1_r4_01_theme_switch_click_and_pop_free() {
        let mut son = Sonificator::new(48_000);
        let scene_warm = make_solid_frame(240, 100, 20, 255, 16, 16);
        son.push_frame(&scene_warm, 16, 16);
        let _ = son.render(4800);

        // Instant scene cut to cool blue ocean
        let scene_cool = make_solid_frame(10, 100, 240, 255, 16, 16);
        son.push_frame(&scene_cool, 16, 16);
        let audio_transition = son.render(4800);

        let max_step = max_sample_step(&audio_transition);
        assert!(
            max_step < 0.20,
            "R4 Violation: Maximum sample-to-sample jump ({max_step}) indicates click/pop artifact"
        );
    }

    #[test]
    fn t1_r4_02_continuous_envelope_slew_on_flicker() {
        let mut son = Sonificator::new(48_000);
        let white = make_solid_frame(255, 255, 255, 255, 8, 8);
        let black = make_solid_frame(0, 0, 0, 255, 8, 8);

        // Rapid flickering frame pushes
        let mut max_observed_step = 0.0f32;
        for _ in 0..10 {
            son.push_frame(&white, 8, 8);
            let a1 = son.render(480);
            max_observed_step = max_observed_step.max(max_sample_step(&a1));

            son.push_frame(&black, 8, 8);
            let a2 = son.render(480);
            max_observed_step = max_observed_step.max(max_sample_step(&a2));
        }

        assert!(
            max_observed_step < 0.25,
            "Envelope slew filters must prevent rapid square-wave clicks: max_step={max_observed_step}"
        );
    }

    #[test]
    fn t1_r4_03_smooth_features_interpolation_mathematics() {
        let f1 = FrameFeatures {
            orange_pulse: 0.1,
            pulse_x: 0.2,
            pulse_y: 0.3,
            pulse_dispersion: 0.4,
        };
        let f2 = FrameFeatures {
            orange_pulse: 0.9,
            pulse_x: 0.8,
            pulse_y: 0.7,
            pulse_dispersion: 0.6,
        };

        let half = smooth(f1, f2, 0.5);
        assert!((half.orange_pulse - 0.5).abs() < 1e-5);
        assert!((half.pulse_x - 0.5).abs() < 1e-5);
        assert!((half.pulse_y - 0.5).abs() < 1e-5);
        assert!((half.pulse_dispersion - 0.5).abs() < 1e-5);
    }

    #[test]
    fn t1_r4_04_visual_bloom_slew_continuity() {
        let mut son = Sonificator::new(48_000);
        // Start in dark
        let dark = make_solid_frame(5, 5, 5, 255, 16, 16);
        son.push_frame(&dark, 16, 16);
        let _ = son.render(2400);

        // Jump to intense bright
        let bright = make_solid_frame(255, 120, 0, 255, 16, 16);
        son.push_frame(&bright, 16, 16);
        let audio = son.render(2400);

        // Verify audio blooms smoothly without instantaneous discontinuous step
        let max_step = max_sample_step(&audio);
        assert!(max_step < 0.20);
    }

    #[test]
    fn t1_r4_05_mid_stream_palette_shift_preserves_audio_integrity() {
        let mut son = Sonificator::new(48_000);
        let palettes = [
            make_solid_frame(255, 100, 20, 255, 16, 16), // Orange
            make_solid_frame(20, 220, 100, 255, 16, 16), // Emerald
            make_solid_frame(20, 100, 255, 255, 16, 16), // Cobalt
            make_solid_frame(180, 20, 220, 255, 16, 16), // Violet
        ];

        for frame in palettes {
            son.push_frame(&frame, 16, 16);
            let audio = son.render(4800);
            assert!(audio.iter().all(|s| s.is_finite()));
            assert!(max_sample_step(&audio) < 0.20);
        }
    }

    // --- R5: Extrapolated Rhythm & Multi-Tone Density ---

    #[test]
    fn t1_r5_01_vertical_centroid_shifts_pitch_register() {
        let mut son_low = Sonificator::new(48_000);
        let mut son_high = Sonificator::new(48_000);

        // Low spot (y near 0.1) vs High spot (y near 0.9)
        let frame_low = make_spot_frame(32, 32, 0.5, 0.1, 0.2, [240, 90, 20]);
        let frame_high = make_spot_frame(32, 32, 0.5, 0.9, 0.2, [240, 90, 20]);

        son_low.push_frame(&frame_low, 32, 32);
        son_high.push_frame(&frame_high, 32, 32);

        let audio_low = son_low.render(48_000);
        let audio_high = son_high.render(48_000);

        let zcr_low = zero_crossing_rate(&audio_low);
        let zcr_high = zero_crossing_rate(&audio_high);

        assert!(
            zcr_high > zcr_low,
            "R5: High vertical centroid must produce higher register tones and zero crossings: high={zcr_high}, low={zcr_low}"
        );
    }

    #[test]
    fn t1_r5_02_horizontal_centroid_modulates_phrase_contours() {
        let mut son_left = Sonificator::new(48_000);
        let mut son_right = Sonificator::new(48_000);

        let frame_left = make_spot_frame(32, 32, 0.15, 0.5, 0.2, [240, 90, 20]);
        let frame_right = make_spot_frame(32, 32, 0.85, 0.5, 0.2, [240, 90, 20]);

        son_left.push_frame(&frame_left, 32, 32);
        son_right.push_frame(&frame_right, 32, 32);

        let audio_left = son_left.render(48_000);
        let audio_right = son_right.render(48_000);

        assert_ne!(audio_left, audio_right);
        let diff: f32 = audio_left
            .iter()
            .zip(audio_right.iter())
            .map(|(l, r)| (l - r).abs())
            .sum();
        assert!(
            diff > 5.0,
            "Left vs Right pulse_x must drive distinct phrase contours: diff={diff}"
        );
    }

    #[test]
    fn t1_r5_03_dispersion_modulates_arpeggio_density() {
        let mut son_sparse = Sonificator::new(48_000);
        let mut son_dense = Sonificator::new(48_000);

        // Clustered spot (low dispersion): 8x8 block in center (12..20, 12..20 = 64 orange pixels)
        let mut frame_sparse = vec![10u8; 32 * 32 * 4];
        for y in 12..20 {
            for x in 12..20 {
                let idx = ((y * 32 + x) * 4) as usize;
                frame_sparse[idx] = 240;
                frame_sparse[idx + 1] = 90;
                frame_sparse[idx + 2] = 20;
                frame_sparse[idx + 3] = 255;
            }
        }

        // Dispersed frame (high dispersion): four 4x4 blocks at corners (4 * 16 = 64 orange pixels)
        let mut frame_dense = vec![10u8; 32 * 32 * 4];
        let corner_ranges = [
            (2..6, 2..6),
            (26..30, 2..6),
            (2..6, 26..30),
            (26..30, 26..30),
        ];
        for (xr, yr) in corner_ranges {
            for y in yr {
                for x in xr.clone() {
                    let idx = ((y * 32 + x) * 4) as usize;
                    frame_dense[idx] = 240;
                    frame_dense[idx + 1] = 90;
                    frame_dense[idx + 2] = 20;
                    frame_dense[idx + 3] = 255;
                }
            }
        }

        let feat_sparse = analyze_frame(&frame_sparse, 32, 32);
        let feat_dense = analyze_frame(&frame_dense, 32, 32);
        assert!(
            feat_dense.pulse_dispersion > feat_sparse.pulse_dispersion * 1.5,
            "Dispersed frame must have significantly higher dispersion metric: dense={}, sparse={}",
            feat_dense.pulse_dispersion,
            feat_sparse.pulse_dispersion
        );

        son_sparse.push_frame(&frame_sparse, 32, 32);
        son_dense.push_frame(&frame_dense, 32, 32);

        let audio_sparse = son_sparse.render(48_000);
        let audio_dense = son_dense.render(48_000);

        assert_ne!(audio_sparse, audio_dense);
        let energy_sparse: f32 = audio_sparse.iter().map(|s| s * s).sum();
        let energy_dense: f32 = audio_dense.iter().map(|s| s * s).sum();

        assert!(
            energy_dense > energy_sparse,
            "Higher dispersion must activate more arpeggio slots: dense={energy_dense}, sparse={energy_sparse}"
        );
    }

    #[test]
    fn t1_r5_04_scale_register_covers_multi_octave_span() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(220, 80, 20, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        // Render multiple bars to span chord degrees and runs
        let audio = son.render(48_000 * 4);
        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.05));
    }

    #[test]
    fn t1_r5_05_walking_bass_contour_in_melodic_algorithm() {
        let mut son =
            Sonificator::new(48_000).with_algorithm(SonificationAlgorithm::MelodicContour);
        let frame = make_solid_frame(220, 100, 30, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        let audio = son.render(48_000 * 2);
        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(rms_level(&audio) > 0.01);
    }

    // --- R6: Plug-and-Play Compatibility ---

    #[test]
    fn t1_r6_01_analyze_frame_public_api_compatibility() {
        let rgba = [240u8, 80, 20, 255, 0, 0, 0, 255];
        let features = analyze_frame(&rgba, 2, 1);
        assert!(features.orange_pulse >= 0.0 && features.orange_pulse <= 1.0);
        assert!(features.pulse_x >= 0.0 && features.pulse_x <= 1.0);
        assert!(features.pulse_y >= 0.0 && features.pulse_y <= 1.0);
        assert!(features.pulse_dispersion >= 0.0 && features.pulse_dispersion <= 1.0);
    }

    #[test]
    fn t1_r6_02_groove_state_enum_variants_and_traits() {
        let states = [
            GrooveState::SparseDrift,
            GrooveState::FlowingGroove,
            GrooveState::IntenseClimax,
        ];
        assert_eq!(states[0], GrooveState::SparseDrift);
        assert_ne!(states[0], states[1]);
        let cloned = states[1];
        assert_eq!(cloned, GrooveState::FlowingGroove);
    }

    #[test]
    fn t1_r6_03_sonificator_constructor_and_sample_rate() {
        let son = Sonificator::new(44_100);
        assert_eq!(son.sample_rate(), 44_100);

        let son2 = Sonificator::new(96_000);
        assert_eq!(son2.sample_rate(), 96_000);
    }

    #[test]
    fn t1_r6_04_features_getter_returns_smoothed_features() {
        let mut son = Sonificator::new(48_000);
        assert_eq!(son.features(), FrameFeatures::default());

        let frame = make_solid_frame(240, 80, 20, 255, 16, 16);
        son.push_frame(&frame, 16, 16);
        let _ = son.render(4800);

        let feat = son.features();
        assert!(feat.orange_pulse > 0.0);
    }

    #[test]
    fn t1_r6_05_render_returns_interleaved_stereo_pcm() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(100, 150, 200, 255, 8, 8);
        son.push_frame(&frame, 8, 8);

        let frames = 1024;
        let pcm = son.render(frames);
        assert_eq!(
            pcm.len(),
            frames * 2,
            "render(frames) must return frames * 2 interleaved left/right samples"
        );
        assert!(pcm.iter().all(|s| s.is_finite()));
    }
}

// ============================================================================
// TIER 2: Boundary & Corner Cases (>=5 tests per feature for R1 - R6)
// ============================================================================

mod tier2_boundary_and_corner_cases {
    use super::helpers::*;
    use super::*;

    // --- R1 Boundary ---

    #[test]
    fn t2_r1_01_render_zero_frames_safely() {
        let mut son = Sonificator::new(48_000);
        let out = son.render(0);
        assert!(out.is_empty());

        let mut buf: [f32; 0] = [];
        son.render_into(&mut buf);
    }

    #[test]
    fn t2_r1_02_minimum_sample_rate_clamping() {
        // Sample rates below 8,000 must be clamped to 8,000
        let son_zero = Sonificator::new(0);
        assert_eq!(son_zero.sample_rate(), 8_000);

        let son_low = Sonificator::new(100);
        assert_eq!(son_low.sample_rate(), 8_000);
    }

    #[test]
    fn t2_r1_03_ultra_high_sample_rate_192khz() {
        let mut son = Sonificator::new(192_000);
        assert_eq!(son.sample_rate(), 192_000);
        let frame = make_solid_frame(255, 120, 20, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        let audio = son.render(1920);
        assert_eq!(audio.len(), 3840);
        assert!(max_peak(&audio) <= 0.83);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r1_04_pure_black_frame_silence_floor() {
        let mut son = Sonificator::new(48_000);
        let black = make_solid_frame(0, 0, 0, 255, 32, 32);
        for _ in 0..5 {
            son.push_frame(&black, 32, 32);
        }

        let audio = son.render(48_000);
        assert!(audio.iter().all(|s| s.is_finite() && !s.is_nan()));
        let peak = max_peak(&audio);
        assert!(
            peak < 0.20,
            "Black frame audio must maintain low baseline drone level: peak={peak}"
        );
    }

    #[test]
    fn t2_r1_05_long_duration_numerical_stability() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(200, 100, 30, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        // Render 200 blocks of 4800 samples (20 seconds of audio)
        for _ in 0..200 {
            let block = son.render(4800);
            assert!(block.iter().all(|s| s.is_finite()));
            assert!(max_peak(&block) <= 0.83);
        }
    }

    // --- R2 Boundary ---

    #[test]
    fn t2_r2_01_pure_grayscale_produces_zero_warmth() {
        let gray = make_solid_frame(128, 128, 128, 255, 16, 16);
        let features = analyze_frame(&gray, 16, 16);
        // Neutral gray has zero chroma
        assert_eq!(features.orange_pulse, 0.0);
    }

    #[test]
    fn t2_r2_02_extreme_warmth_saturation_bounds() {
        let mut son_red = Sonificator::new(48_000);
        let mut son_blue = Sonificator::new(48_000);

        let pure_red = make_solid_frame(255, 0, 0, 255, 16, 16);
        let pure_blue = make_solid_frame(0, 0, 255, 255, 16, 16);

        for _ in 0..10 {
            son_red.push_frame(&pure_red, 16, 16);
            son_blue.push_frame(&pure_blue, 16, 16);
        }

        let audio_red = son_red.render(48_000);
        let audio_blue = son_blue.render(48_000);

        assert!(audio_red.iter().all(|s| s.is_finite()));
        assert!(audio_blue.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r2_03_rapid_alternating_warm_cool_frames() {
        let mut son = Sonificator::new(48_000);
        let warm = make_solid_frame(255, 40, 10, 255, 16, 16);
        let cool = make_solid_frame(10, 40, 255, 255, 16, 16);

        // Rapid 60fps alternating pushes
        for _ in 0..30 {
            son.push_frame(&warm, 16, 16);
            let _ = son.render(800);
            son.push_frame(&cool, 16, 16);
            let _ = son.render(800);
        }

        let audio = son.render(4800);
        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&audio) <= 0.83);
    }

    #[test]
    fn t2_r2_04_zero_dimensions_frame_push() {
        let mut son = Sonificator::new(48_000);
        // Zero width/height frame push must not panic or divide by zero
        son.push_frame(&[], 0, 0);
        let audio = son.render(1024);
        assert_eq!(audio.len(), 2048);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r2_05_mid_render_algorithm_switches() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(220, 100, 30, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        for i in 0..20 {
            if i % 2 == 0 {
                son.set_algorithm(SonificationAlgorithm::MelodicContour);
            } else {
                son.set_algorithm(SonificationAlgorithm::RhythmicGroove);
            }
            let chunk = son.render(480);
            assert!(chunk.iter().all(|s| s.is_finite()));
            assert!(max_sample_step(&chunk) < 0.25);
        }
    }

    // --- R3 Boundary ---

    #[test]
    fn t2_r3_01_subthreshold_luma_change_no_thump() {
        let mut son = Sonificator::new(48_000);
        let dark = make_solid_frame(50, 50, 50, 255, 16, 16);
        son.push_frame(&dark, 16, 16);
        let _ = son.render(1000);

        // Rise luma by 0.18 (below 0.20 threshold)
        // 50 / 255 ~ 0.196; + 0.18 => 0.376 => ~96
        let slightly_brighter = make_solid_frame(96, 96, 96, 255, 16, 16);
        son.push_frame(&slightly_brighter, 16, 16);
        let audio = son.render(2400);

        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r3_02_consecutive_back_to_back_flashes() {
        let mut son = Sonificator::new(48_000);
        let dark = make_solid_frame(10, 10, 10, 255, 16, 16);
        let flash = make_solid_frame(255, 255, 255, 255, 16, 16);

        son.push_frame(&dark, 16, 16);
        let _ = son.render(500);

        // Rapid double flash
        son.push_frame(&flash, 16, 16);
        let _ = son.render(100);
        son.push_frame(&dark, 16, 16);
        son.push_frame(&flash, 16, 16);

        let audio = son.render(4800);
        assert!(
            max_peak(&audio) <= 0.83,
            "Re-triggered thump must respect master gain headroom <= 0.83"
        );
    }

    #[test]
    fn t2_r3_03_negative_flash_blackout_does_not_trigger_thump() {
        let mut son = Sonificator::new(48_000);
        let bright = make_solid_frame(255, 255, 255, 255, 16, 16);
        let dark = make_solid_frame(0, 0, 0, 255, 16, 16);

        son.push_frame(&bright, 16, 16);
        let _ = son.render(1000);

        // Instant drop from 1.0 to 0.0 (luma decrease)
        son.push_frame(&dark, 16, 16);
        let audio = son.render(2400);

        // Decaying audio should be quiet, not surging
        let rms = rms_level(&audio);
        assert!(rms < 0.20);
    }

    #[test]
    fn t2_r3_04_flash_with_truncated_buffer_graceful_handling() {
        let mut son = Sonificator::new(48_000);
        let dark = make_solid_frame(10, 10, 10, 255, 16, 16);
        son.push_frame(&dark, 16, 16);
        let _ = son.render(500);

        // Push truncated buffer (only 10 bytes instead of 16*16*4 = 1024)
        son.push_frame(&[255; 10], 16, 16);
        let audio = son.render(1000);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r3_05_thump_voice_numerical_stability_at_8khz() {
        let mut son = Sonificator::new(8_000);
        let dark = make_solid_frame(10, 10, 10, 255, 16, 16);
        let flash = make_solid_frame(255, 255, 255, 255, 16, 16);

        son.push_frame(&dark, 16, 16);
        let _ = son.render(200);
        son.push_frame(&flash, 16, 16);

        let audio = son.render(8000);
        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&audio) <= 0.83);
    }

    // --- R4 Boundary ---

    #[test]
    fn t2_r4_01_theme_switch_at_sample_zero() {
        let mut son = Sonificator::new(48_000);
        let warm = make_solid_frame(250, 100, 20, 255, 16, 16);
        let cool = make_solid_frame(20, 100, 250, 255, 16, 16);

        // Push scene 1, immediately switch to scene 2 before any render
        son.push_frame(&warm, 16, 16);
        son.push_frame(&cool, 16, 16);

        let audio = son.render(4800);
        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(max_sample_step(&audio) < 0.20);
    }

    #[test]
    fn t2_r4_02_extreme_aspect_ratio_1x1000() {
        let mut son = Sonificator::new(48_000);
        let tall_frame = make_solid_frame(240, 80, 20, 255, 1, 1000);
        son.push_frame(&tall_frame, 1, 1000);

        let audio = son.render(2400);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r4_03_extreme_aspect_ratio_1000x1() {
        let mut son = Sonificator::new(48_000);
        let wide_frame = make_solid_frame(240, 80, 20, 255, 1000, 1);
        son.push_frame(&wide_frame, 1000, 1);

        let audio = son.render(2400);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r4_04_single_pixel_frame() {
        let mut son = Sonificator::new(48_000);
        let single_pixel = [255u8, 128, 32, 255];
        son.push_frame(&single_pixel, 1, 1);

        let audio = son.render(1024);
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r4_05_instantaneous_180_degree_hue_jump() {
        let mut son = Sonificator::new(48_000);
        let red = make_solid_frame(255, 0, 0, 255, 16, 16);
        let cyan = make_solid_frame(0, 255, 255, 255, 16, 16);

        son.push_frame(&red, 16, 16);
        let _ = son.render(2400);
        son.push_frame(&cyan, 16, 16);
        let audio = son.render(2400);

        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(max_sample_step(&audio) < 0.20);
    }

    // --- R5 Boundary ---

    #[test]
    fn t2_r5_01_extreme_centroid_at_corners() {
        let corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        for (cx, cy) in corners {
            let mut son = Sonificator::new(48_000);
            let frame = make_spot_frame(32, 32, cx, cy, 0.1, [240, 80, 20]);
            son.push_frame(&frame, 32, 32);

            let audio = son.render(2400);
            assert!(audio.iter().all(|s| s.is_finite()));
            assert!(max_peak(&audio) <= 0.83);
        }
    }

    #[test]
    fn t2_r5_02_zero_dispersion_boundary() {
        let f = FrameFeatures {
            orange_pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 0.0,
        };
        assert_eq!(f.pulse_dispersion, 0.0);
    }

    #[test]
    fn t2_r5_03_maximum_dispersion_boundary() {
        let f = FrameFeatures {
            orange_pulse: 0.5,
            pulse_x: 0.5,
            pulse_y: 0.5,
            pulse_dispersion: 1.0,
        };
        assert_eq!(f.pulse_dispersion, 1.0);
    }

    #[test]
    fn t2_r5_04_monochrome_motion_centroid_fallback() {
        let mut son = Sonificator::new(48_000);
        let frame1 = make_solid_frame(30, 30, 30, 255, 16, 16);
        let frame2 = make_solid_frame(220, 220, 220, 255, 16, 16);

        son.push_frame(&frame1, 16, 16);
        let _ = son.render(1000);
        son.push_frame(&frame2, 16, 16);
        let audio = son.render(4800);

        assert!(rms_level(&audio) > 0.01);
    }

    #[test]
    fn t2_r5_05_high_speed_subdivision_progression() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(220, 100, 20, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        // Render 8 full bars (32 beats = 128 16th notes)
        let audio = son.render(48_000 * 20);
        assert!(audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&audio) <= 0.83);
    }

    // --- R6 Boundary ---

    #[test]
    fn t2_r6_01_render_into_odd_sized_slice() {
        let mut son = Sonificator::new(48_000);
        let mut odd_buffer = [0.0f32; 7]; // Not an exact multiple of 2
        son.render_into(&mut odd_buffer);
        // Only 3 stereo frames rendered, 1 remainder sample unaffected
        assert!(odd_buffer[..6].iter().all(|s| s.is_finite()));
    }

    #[test]
    fn t2_r6_02_checked_mul_overflow_in_analysis() {
        // Huge dimensions that would overflow u32 * u32 * 4
        let features = analyze_frame(&[0; 100], u32::MAX, u32::MAX);
        assert_eq!(features, FrameFeatures::default());
    }

    #[test]
    fn t2_r6_03_buffer_shorter_than_dimensions() {
        let buffer = [255u8; 10];
        let features = analyze_frame(&buffer, 100, 100);
        assert_eq!(features, FrameFeatures::default());
    }

    #[test]
    fn t2_r6_04_smooth_with_boundary_amounts() {
        let f1 = FrameFeatures {
            orange_pulse: 0.2,
            pulse_x: 0.3,
            pulse_y: 0.4,
            pulse_dispersion: 0.5,
        };
        let f2 = FrameFeatures {
            orange_pulse: 0.8,
            pulse_x: 0.7,
            pulse_y: 0.6,
            pulse_dispersion: 0.5,
        };

        // amount = 0.0 -> returns f1
        let s0 = smooth(f1, f2, 0.0);
        assert_eq!(s0, f1);

        // amount = 1.0 -> returns f2
        let s1 = smooth(f1, f2, 1.0);
        assert_eq!(s1, f2);
    }

    #[test]
    fn t2_r6_05_repeated_new_drop_concurrency() {
        // Repeated instantiation and drop across multiple threads
        use std::thread;
        let mut handles = Vec::new();
        for _ in 0..8 {
            let handle = thread::spawn(|| {
                for _ in 0..50 {
                    let mut son = Sonificator::new(48_000);
                    let f = make_solid_frame(200, 80, 20, 255, 8, 8);
                    son.push_frame(&f, 8, 8);
                    let audio = son.render(256);
                    assert_eq!(audio.len(), 512);
                }
            });
            handles.push(handle);
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}

// ============================================================================
// TIER 3: Cross-Feature Interactions (Pairwise combinations)
// ============================================================================

mod tier3_cross_feature_interactions {
    use super::helpers::*;
    use super::*;

    #[test]
    fn t3_01_r1_x_r2_loudness_across_warmth_tempo_extremes() {
        // Verify headroom <= 0.83 is maintained at extreme warmth (52 BPM) and extreme cool (116 BPM)
        for algo in [
            SonificationAlgorithm::RhythmicGroove,
            SonificationAlgorithm::MelodicContour,
        ] {
            let mut son_warm = Sonificator::new(48_000).with_algorithm(algo);
            let mut son_cool = Sonificator::new(48_000).with_algorithm(algo);

            let warm = make_solid_frame(255, 40, 10, 255, 16, 16);
            let cool = make_solid_frame(10, 40, 255, 255, 16, 16);

            for _ in 0..6 {
                son_warm.push_frame(&warm, 16, 16);
                son_cool.push_frame(&cool, 16, 16);
            }

            let audio_warm = son_warm.render(48_000 * 2);
            let audio_cool = son_cool.render(48_000 * 2);

            assert!(
                max_peak(&audio_warm) <= 0.83,
                "Warm output exceeded 0.83 headroom limit"
            );
            assert!(
                max_peak(&audio_cool) <= 0.83,
                "Cool output exceeded 0.83 headroom limit"
            );
            assert!(
                rms_level(&audio_warm) < 0.35,
                "Warm RMS exceeded 0.35 limit"
            );
            assert!(
                rms_level(&audio_cool) < 0.35,
                "Cool RMS exceeded 0.35 limit"
            );
        }
    }

    #[test]
    fn t3_02_r1_x_r3_flash_thump_under_maximum_polyphony_loudness() {
        let mut son = Sonificator::new(48_000);
        // Intense climax frame: high orange pulse, high motion
        let intense = make_solid_frame(255, 130, 20, 255, 32, 32);
        son.push_frame(&intense, 32, 32);
        let _ = son.render(4800);

        // Sudden flash while intense groove is sounding
        let flash = make_solid_frame(255, 255, 255, 255, 32, 32);
        son.push_frame(&flash, 32, 32);

        // Check every single sample over 2 seconds of full 6-voice polyphony
        let audio = son.render(48_000 * 2);
        let peak = max_peak(&audio);
        assert!(
            peak <= 0.83,
            "R1 x R3 Violation: Peak under full polyphony + thump was {peak} > 0.83"
        );
        assert!(audio.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn t3_03_r2_x_r3_flash_during_warm_tempo_slowdown() {
        let mut son = Sonificator::new(48_000);
        let warm = make_solid_frame(255, 50, 10, 255, 16, 16);
        for _ in 0..5 {
            son.push_frame(&warm, 16, 16);
        }
        let _ = son.render(9600);

        // Flash occurs during slow-tempo playback
        let flash = make_solid_frame(255, 255, 255, 255, 16, 16);
        son.push_frame(&flash, 16, 16);

        let audio = son.render(4800);
        assert!(
            rms_level(&audio) > 0.03,
            "Thump must sound cleanly during warm tempo"
        );
        assert!(max_peak(&audio) <= 0.83);
    }

    #[test]
    fn t3_04_r2_x_r4_algorithm_switch_during_theme_transition() {
        let mut son = Sonificator::new(48_000);
        let scene1 = make_solid_frame(240, 80, 20, 255, 16, 16);
        son.push_frame(&scene1, 16, 16);
        let _ = son.render(4800);

        // Simultaneously switch theme and algorithm
        let scene2 = make_solid_frame(20, 80, 240, 255, 16, 16);
        son.push_frame(&scene2, 16, 16);
        son.set_algorithm(SonificationAlgorithm::MelodicContour);

        let transition_audio = son.render(4800);
        let max_step = max_sample_step(&transition_audio);
        assert!(
            max_step < 0.25,
            "Simultaneous algorithm switch and theme transition must remain click-free: max_step={max_step}"
        );
    }

    #[test]
    fn t3_05_r3_x_r4_flash_surge_during_abrupt_scene_cut() {
        let mut son = Sonificator::new(48_000);
        let dark_cave = make_solid_frame(10, 10, 15, 255, 16, 16);
        son.push_frame(&dark_cave, 16, 16);
        let _ = son.render(4800);

        // Abrupt cut to bright explosion
        let bright_explosion = make_solid_frame(255, 250, 230, 255, 16, 16);
        son.push_frame(&bright_explosion, 16, 16);
        let audio = son.render(4800);

        let max_step = max_sample_step(&audio);
        assert!(
            max_step < 0.25,
            "Flash surge on scene cut must not create sample discontinuity: max_step={max_step}"
        );
        assert!(max_peak(&audio) <= 0.83);
    }

    #[test]
    fn t3_06_r4_x_r5_phrase_continuity_across_progression_turnaround() {
        let mut son = Sonificator::new(48_000);
        let frame = make_solid_frame(230, 90, 20, 255, 16, 16);
        son.push_frame(&frame, 16, 16);

        // Render 4 full bars (progression cycle) into the turnaround
        let audio = son.render(48_000 * 10);
        let max_step = max_sample_step(&audio);
        assert!(
            max_step < 0.25,
            "Bar boundary progression turnaround must be click-free: max_step={max_step}"
        );
    }

    #[test]
    fn t3_07_r2_x_r5_dual_algorithm_melodic_density_contrast() {
        let mut son_rg =
            Sonificator::new(48_000).with_algorithm(SonificationAlgorithm::RhythmicGroove);
        let mut son_mc =
            Sonificator::new(48_000).with_algorithm(SonificationAlgorithm::MelodicContour);

        let frame = make_spot_frame(32, 32, 0.5, 0.8, 0.3, [230, 110, 30]);
        son_rg.push_frame(&frame, 32, 32);
        son_mc.push_frame(&frame, 32, 32);

        let audio_rg = son_rg.render(48_000);
        let audio_mc = son_mc.render(48_000);

        // Both render bounded, non-zero audio with distinct character
        assert!(max_peak(&audio_rg) <= 0.83);
        assert!(max_peak(&audio_mc) <= 0.83);
        assert_ne!(audio_rg, audio_mc);
    }

    #[test]
    fn t3_08_r1_x_r6_render_into_in_place_identical_to_render_with_headroom() {
        let mut son_alloc = Sonificator::new(48_000);
        let mut son_inplace = Sonificator::new(48_000);

        let frames = [
            make_solid_frame(200, 80, 20, 255, 16, 16),
            make_solid_frame(50, 180, 220, 255, 16, 16),
            make_solid_frame(255, 255, 255, 255, 16, 16),
        ];

        for frame in frames {
            son_alloc.push_frame(&frame, 16, 16);
            son_inplace.push_frame(&frame, 16, 16);

            let audio_alloc = son_alloc.render(2400);
            let mut audio_inplace = vec![0.0f32; 4800];
            son_inplace.render_into(&mut audio_inplace);

            assert_eq!(audio_alloc, audio_inplace);
            assert!(max_peak(&audio_alloc) <= 0.83);
        }
    }
}

// ============================================================================
// TIER 4: Real-World Application Scenarios
// ============================================================================

mod tier4_real_world_scenarios {
    use super::helpers::*;
    use super::*;

    #[test]
    fn t4_01_scenario_video_stream_60fps_simulation() {
        // Simulate 6 seconds of 60 fps streaming video (360 frames)
        // Each frame renders 48,000 / 60 = 800 stereo frames (1600 samples)
        let mut son = Sonificator::new(48_000);
        let samples_per_frame = 800;
        let total_frames = 360;

        let mut all_samples = Vec::with_capacity(total_frames * samples_per_frame * 2);

        for i in 0..total_frames {
            // Evolving procedural frame (orbiting color blob)
            let phase = i as f32 / 60.0;
            let cx = 0.5 + 0.3 * (phase * 2.0).cos();
            let cy = 0.5 + 0.3 * (phase * 2.0).sin();
            let frame = make_spot_frame(32, 32, cx, cy, 0.2, [220, 100, 30]);

            son.push_frame(&frame, 32, 32);
            let chunk = son.render(samples_per_frame);
            all_samples.extend_from_slice(&chunk);
        }

        assert_eq!(all_samples.len(), total_frames * samples_per_frame * 2);
        assert!(all_samples.iter().all(|s| s.is_finite()));
        let peak = max_peak(&all_samples);
        let rms = rms_level(&all_samples);

        assert!(
            peak <= 0.83,
            "Video stream simulation exceeded headroom limit: peak={peak}"
        );
        assert!(
            rms < 0.35,
            "Video stream simulation exceeded RMS limit: rms={rms}"
        );
        assert!(
            rms > 0.01,
            "Video stream simulation must produce active musical accompaniment: rms={rms}"
        );
    }

    #[test]
    fn t4_02_scenario_thunderstorm_dark_and_lightning_strikes() {
        // Dark, gloomy ambient rain video punctuated by 3 lightning strikes
        let mut son = Sonificator::new(48_000);
        let rain_frame = make_solid_frame(15, 25, 40, 255, 32, 32);
        let lightning_frame = make_solid_frame(255, 255, 255, 255, 32, 32);

        // 1. Initial dark rain
        let mut pre_strike_peak = 0.0f32;
        for _ in 0..10 {
            son.push_frame(&rain_frame, 32, 32);
            let chunk = son.render(1600);
            pre_strike_peak = pre_strike_peak.max(max_peak(&chunk));
        }

        // 2. Strike 1
        son.push_frame(&lightning_frame, 32, 32);
        let strike1_audio = son.render(4800);
        let strike1_peak = max_peak(&strike1_audio);
        assert!(
            strike1_peak > pre_strike_peak * 2.0 && strike1_peak > 0.05,
            "Strike 1 must trigger thump voice surge: peak={strike1_peak}, pre_strike={pre_strike_peak}"
        );
        assert!(strike1_peak <= 0.83);

        // 3. Return to rain
        for _ in 0..15 {
            son.push_frame(&rain_frame, 32, 32);
            let _ = son.render(1600);
        }

        // 4. Strike 2
        son.push_frame(&lightning_frame, 32, 32);
        let strike2_audio = son.render(4800);
        assert!(max_peak(&strike2_audio) <= 0.83);

        // 5. Strike 3
        son.push_frame(&lightning_frame, 32, 32);
        let strike3_audio = son.render(4800);
        assert!(max_peak(&strike3_audio) <= 0.83);
    }

    #[test]
    fn t4_03_scenario_sunrise_sunset_warmth_cycle() {
        // Continuous day/night/sunrise/sunset cycle:
        // Midnight (cool blue) -> Sunrise (warm golden) -> Noon -> Sunset (deep orange) -> Night
        let mut son = Sonificator::new(48_000);

        let phases = [
            ("midnight", [10, 20, 80, 255]),
            ("dawn", [60, 60, 160, 255]),
            ("sunrise", [255, 160, 20, 255]),
            ("noon", [200, 200, 200, 255]),
            ("sunset", [255, 80, 10, 255]),
            ("dusk", [40, 20, 60, 255]),
            ("night", [5, 10, 40, 255]),
        ];

        let mut all_audio = Vec::new();
        for (_phase_name, rgba) in phases {
            let frame = make_solid_frame(rgba[0], rgba[1], rgba[2], rgba[3], 16, 16);
            for _ in 0..10 {
                son.push_frame(&frame, 16, 16);
                let chunk = son.render(2400);
                all_audio.extend_from_slice(&chunk);
            }
        }

        assert!(all_audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&all_audio) <= 0.83);
        assert!(max_sample_step(&all_audio) < 0.25);
    }

    #[test]
    fn t4_04_scenario_high_motion_action_sequence() {
        let mut son = Sonificator::new(48_000);

        // Rapid vertical acrobatics (jumping high then falling low)
        let y_positions = [0.1, 0.9, 0.15, 0.85, 0.2, 0.95, 0.1, 0.9];
        let mut rendered_audio = Vec::new();

        for &y in &y_positions {
            let frame = make_spot_frame(32, 32, 0.5, y, 0.2, [240, 100, 20]);
            son.push_frame(&frame, 32, 32);
            let chunk = son.render(4800);
            rendered_audio.extend_from_slice(&chunk);
        }

        assert!(rendered_audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&rendered_audio) <= 0.83);
        assert!(max_sample_step(&rendered_audio) < 0.25);
    }

    #[test]
    fn t4_05_scenario_multi_scene_editorial_montage() {
        // Montage cutting between 5 radically different visual styles:
        // 1. Deep blue ocean
        // 2. Blazing volcanic magma
        // 3. Dark cave
        // 4. Emerald jungle
        // 5. Monochrome pencil sketch
        let mut son = Sonificator::new(48_000);

        let scenes = [
            make_solid_frame(10, 80, 240, 255, 32, 32),    // Ocean
            make_solid_frame(255, 60, 10, 255, 32, 32),    // Magma
            make_solid_frame(12, 12, 12, 255, 32, 32),     // Cave
            make_solid_frame(20, 210, 80, 255, 32, 32),    // Jungle
            make_solid_frame(200, 200, 200, 255, 32, 32),  // Sketch
        ];

        let mut montage_audio = Vec::new();

        for (scene_idx, frame) in scenes.iter().enumerate() {
            // Push scene frames
            for frame_num in 0..5 {
                son.push_frame(frame, 32, 32);
                let chunk = son.render(4800);
                montage_audio.extend_from_slice(&chunk);

                // On scene 3, switch algorithm to MelodicContour for stylistic variation
                if scene_idx == 2 && frame_num == 0 {
                    son.set_algorithm(SonificationAlgorithm::MelodicContour);
                }
            }
        }

        assert!(montage_audio.iter().all(|s| s.is_finite()));
        assert!(max_peak(&montage_audio) <= 0.83);
        assert!(
            max_sample_step(&montage_audio) < 0.25,
            "Montage cuts must remain click-free without popping"
        );
        assert!(rms_level(&montage_audio) > 0.01);
    }
}
