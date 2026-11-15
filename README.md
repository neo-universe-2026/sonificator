# Sonificator

**Sonificator** is a Rust audio engine that turns visual features extracted from video frames into continuously generated stereo music. It is the audio-generation core of the Neo Universe / Earth Information Jukebox project.

The current implementation is video-independent: it auto-detects visual pulses from general frame brightness regardless of color, and exposes extra image statistics such as hue, saturation, contrast and motion. Tunables in `AnalysisOptions` let you blend in chromatic salience or focus on a hue band when a dataset calls for it. It is still a prototype rather than a general-purpose scientific sonification framework.

## What it does

1. Accepts 8-bit RGBA video frames.
2. Samples each frame to estimate pulse strength (brightness-driven by default, color-agnostic), centre position, spatial spread, plus image stats: brightness, contrast, saturation, hue, warmth and inter-frame motion.
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

    use sonificator::{Sonificator, AnalysisOptions};

    // Defaults are color-agnostic: pulses come from general frame brightness.
    let mut engine = Sonificator::new(48_000);

    // rgba contains width * height * 4 bytes in RGBA order.
    engine.push_frame(&rgba, width, height);

    // Request 512 stereo frames. Samples are interleaved L, R, L, R...
    let samples: Vec<f32> = engine.render(512);

Each call to render(n) returns 2 * n floating-point samples. Values are bounded to the normalised audio range. For a realtime callback, render_into provides an allocation-free rendering path when the caller supplies an appropriately sized output buffer.

### Analysis options

`AnalysisOptions` tunes how frames become features. Pass it to `Sonificator::new_with_options` or `set_analysis_options`:

| Option | Default | Effect |
| --- | --- | --- |
| `color_weight` | 0.25 | Blend of brightness vs chromatic salience in the pulse. 0.0 = brightness only, 1.0 = legacy color-only behaviour. |
| `hue_focus` | `None` | Optional `(center, half_width)` hue band in 0..=1 to emphasize (e.g. `Some((0.08, 0.1))` targets orange). |
| `hue_focus_strength` | 1.0 | How strongly non-focused hues are suppressed, 0.0 to 1.0. |
| `sensitivity` | 1.0 | Pulse gain before normalization, clamped to 0.25..=4.0 — useful for very dim or very flat videos. |

`analyze_frame` / `analyze_frame_with` expose the same analysis without generating audio.

See [the audio mapping notes](docs/audio-mapping.md) and [architecture notes](docs/architecture.md) for implementation details and limitations.

## Current scope and limitations

- The analyser is palette-agnostic by default: it responds to general brightness and motion on any video, not only to a single hue. It does not interpret data semantics; a bright region means visual brightness, not necessarily a higher concentration of anything.
- `pulse_x` influences musical fingerprinting, melody pitch and stereo placement. Pulse strength influences rhythmic energy, tempo and dynamics; brightness and saturation shape the mix and timbre; hue seeds the musical fingerprint; motion drives tempo.
- The engine describes visual patterns in sound; it does not calculate or validate atmospheric CO₂ concentrations.
- Audio generation is implemented here. Browser playback is demonstrated separately in [Sonificator Studio](https://github.com/neo-universe-2026/sonificator-website).

## Related project

- [Sonificator Studio](https://github.com/neo-universe-2026/sonificator-website) — browser demonstration using WebAssembly and Web Audio.
- [Earth Information Jukebox website](https://github.com/neo-universe-2026/Website) — accessible globe-based exploration of NASA visualisations.

## Contributing

Please open an issue before making substantial changes so the intended behaviour can be discussed. Keep changes focused, run cargo fmt --all and cargo test --workspace, and update the documentation whenever the public API or feature-to-sound mapping changes.

## Project status

This is an evolving competition prototype. The library and command-line renderer are implemented; broader dataset-specific feature extraction and integration into the main globe website remain future work.
