# Sonificator

**Sonificator** is a Rust audio engine that turns visual features extracted from video frames into continuously generated stereo music. It is the audio-generation core of the Neo Universe / Earth Information Jukebox project.

The current implementation is a focused prototype for a CO₂ globe visualisation whose emissions appear as orange particles. It detects those visual patterns and maps selected features to musical behaviour. It is not yet a general-purpose scientific sonification framework.

## What it does

1. Accepts 8-bit RGBA video frames.
2. Samples each frame to estimate the level, centre position and spatial spread of orange regions.
3. Smooths those features over time.
4. Generates interleaved stereo PCM audio using synthesised tones and a pentatonic musical vocabulary.

The engine is independent of an audio device. Applications can request audio samples and send them to their own playback system, including a native audio callback or a Web Audio / WebAssembly integration.

## Repository layout

| Path | Purpose |
| --- | --- |
| sonificator/ | Reusable Rust library: frame analysis and audio synthesis |
| sonificator-render/ | Command-line renderer that reads video with FFmpeg and writes audio/video output |
| docs/ | Design notes and the current feature-to-sound mapping |

## Requirements

- Rust toolchain with Cargo (the workspace uses Rust edition 2024).
- FFmpeg and FFprobe available on PATH to use the command-line renderer.
- A video file readable by FFmpeg for CLI rendering.

## Build and test

Run these commands from the repository root:

    cargo build --workspace
    cargo test --workspace

## Render a video

Create a WAV soundtrack:

    cargo run -p sonificator-render -- path/to/input.mp4

Choose an output filename:

    cargo run -p sonificator-render -- path/to/input.mp4 output.wav

Create a new MP4 containing the original video and generated audio:

    cargo run -p sonificator-render -- --merge path/to/input.mp4 output-with-audio.mp4

If no output path is supplied, the renderer writes sonificator-render.wav, or sonificator-render.mp4 when --merge is used. The renderer currently outputs 48 kHz stereo audio.

## Library quick start

    use sonificator::Sonificator;

    let mut engine = Sonificator::new(48_000);

    // rgba contains width * height * 4 bytes in RGBA order.
    engine.push_frame(&rgba, width, height);

    // Request 512 stereo frames. Samples are interleaved L, R, L, R...
    let samples: Vec<f32> = engine.render(512);

Each call to render(n) returns 2 * n floating-point samples. Values are bounded to the normalised audio range. For a realtime callback, render_into provides an allocation-free rendering path when the caller supplies an appropriately sized output buffer.

See [the audio mapping notes](docs/audio-mapping.md) and [architecture notes](docs/architecture.md) for implementation details and limitations.

## Current scope and limitations

- The frame analyser is deliberately tuned to orange emissions in a particular CO₂ visualisation style. It should not be assumed to interpret arbitrary NASA datasets correctly.
- pulse_x currently influences musical fingerprinting, melody pitch and stereo placement. Orange pulse intensity influences rhythmic energy, tempo and dynamics.
- Vertical position and spatial dispersion are measured and exposed as features, but are not yet mapped to distinct musical controls in the synthesiser.
- The engine describes visual patterns in sound; it does not calculate or validate atmospheric CO₂ concentrations.
- Audio generation is implemented here. Browser playback is demonstrated separately in [Sonificator Studio](https://github.com/neo-universe-2026/sonificator-website).

## Related project

- [Sonificator Studio](https://github.com/neo-universe-2026/sonificator-website) — browser demonstration using WebAssembly and Web Audio.
- [Earth Information Jukebox website](https://github.com/neo-universe-2026/Website) — accessible globe-based exploration of NASA visualisations.

## Contributing

Please open an issue before making substantial changes so the intended behaviour can be discussed. Keep changes focused, run cargo fmt --all and cargo test --workspace, and update the documentation whenever the public API or feature-to-sound mapping changes.

## Project status

This is an evolving competition prototype. The library and command-line renderer are implemented; broader dataset-specific feature extraction and integration into the main globe website remain future work.
