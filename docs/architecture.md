# Architecture

## Purpose

The Sonificator crate separates visual-feature analysis from audio-device playback. Its public interface accepts video frames and produces blocks of stereo PCM samples. This keeps the core reusable by a command-line renderer, a native application or a WebAssembly host.

## Processing pipeline

    RGBA video frame
          |
          v
    analyze_frame()
          |
          v
    FrameFeatures
          |
          v
    Sonificator::push_frame()
          |
          v
    smoothed feature state
          |
          v
    Sonificator::render() / render_into()
          |
          v
    interleaved stereo f32 PCM

## Frame input and analysis

The input frame is an 8-bit RGBA byte slice with a declared width and height. The analyser validates the dimensions and buffer length, then samples the image on a grid designed to keep analysis lightweight (approximately 32 samples across each dimension).

Per sampled pixel it computes Rec. 709 luma, chroma and hue, then forms a salience value that blends brightness-based and chromatic components according to `AnalysisOptions::color_weight`. Pulse strength is a soft-knee normalization of the mean salience; brightness, contrast, saturation, hue and warmth are accumulated as plain averaging statistics, and position / dispersion as salience-weighted moments. `AnalysisOptions::hue_focus` optionally re-weights the mixed salience toward a hue band, and `sensitivity` scales the pulse before normalization.

A 16x16 coarse luma grid is kept per frame; `Sonificator::push_frame` diffs it against the previous frame to derive motion energy and a motion-based spatial centroid/spread. When a frame has no salient content (pure black etc.) but the grid changed, motion supplies the pulse energy instead, keeping dark or greyscale footage expressive. The motion average only counts cells this video actually samples, so small frames are not diluted.

## FrameFeatures

The analyser returns these normalized features:

| Feature | Meaning |
| --- | --- |
| pulse | Auto-detected visual pulse strength, brightness-driven by default with a blendable chromatic component |
| pulse_x | Horizontal centre of salient activity, from 0 (left) to 1 (right) |
| pulse_y | Vertical centre, from 0 (top) to 1 (bottom) |
| pulse_dispersion | Estimated spatial spread of salient activity |
| brightness | Mean perceived luma of the frame |
| contrast | Luma standard deviation, scaled to roughly 0..=1 |
| saturation | Mean chroma across the frame |
| hue | Salience-weighted dominant hue in 0..=1; 0.5 when achromatic |
| warmth | Warm vs cool balance, -1 (cool) ..= 1 (warm) |
| motion | Mean inter-frame luma change on the coarse grid |

When there is no salient content, the position defaults to the frame centre and the pulse / dispersion values are zero. Invalid or undersized input returns default features.

These are image-derived descriptors, not physical measurements.

## Stateful synthesis

Sonificator stores the latest target features, smoothed features, oscillator phases and musical timing state. push_frame updates the target state. render advances the synthesiser and returns the requested number of stereo frames.

The synthesis currently combines a drone, bass, chord pad, melodic line and arpeggiated layer. It uses a C-major pentatonic scale and a progression selected from an initial visual fingerprint. Feature smoothing reduces abrupt changes between video frames.

## Audio output

The library returns floating-point samples in left/right interleaved order. It does not open an audio device or choose a playback API. The caller owns playback, buffering and synchronisation.

## Command-line renderer

The sonificator-render binary uses FFprobe to read video dimensions and frame rate, then uses FFmpeg to decode video frames to RGBA. It feeds each frame to the library and requests the corresponding number of audio frames at 48 kHz.

- Default mode writes a stereo WAV file.
- --merge creates a temporary WAV and muxes it with the original video into an MP4.
- FFmpeg and FFprobe must be installed and accessible on PATH.

## WebAssembly integration

The separate [Sonificator Studio](https://github.com/neo-universe-2026/sonificator-website) repository wraps the same Rust library using wasm-bindgen. Its JavaScript application samples a playing video into a canvas, passes RGBA frames to the WASM wrapper and requests audio blocks from a Web Audio processing callback.

## Extension points

A future multi-dataset design should separate dataset-specific feature extraction from the synthesis engine. `AnalysisOptions` already covers palette tuning (brightness/color blend, hue focus, sensitivity); datasets needing deeper domain extraction can contribute features on top of `FrameFeatures` while reusing the same synthesis engine.
