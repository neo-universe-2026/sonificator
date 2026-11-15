# Audio mapping

This document describes the behaviour implemented in the current Sonificator source. The analyser is video-independent: it auto-detects visual pulses from general frame brightness instead of looking for one particular color.

## Visual features

`FrameFeatures` carries these normalized per-frame statistics:

- **Pulse:** overall visual pulse strength. By default this is dominated by frame brightness (Rec. 709 luma), blended with a hue-agnostic chromatic component so dim-but-vivid content (neon particles on black, for example) still registers. Soft-knee normalization keeps uniformly bright frames below 1.0.
- **Horizontal position:** salience-weighted centre from left to right.
- **Vertical position:** salience-weighted centre from top to bottom.
- **Dispersion:** a normalised estimate of how spread out the salient pixels are.
- **Brightness:** mean perceived luma of the frame.
- **Contrast:** population standard deviation of luma, scaled to roughly 0–1.
- **Saturation:** mean chroma across the frame.
- **Hue:** salience-weighted dominant hue in 0..=1 (0 red, 1/3 green, 2/3 blue); 0.5 when achromatic.
- **Warmth:** warm vs cool balance in -1..=1 (+1 red, -1 blue); 0 when achromatic.
- **Motion:** mean absolute luma change on a coarse 16x16 grid between consecutive frames.

Frame-to-frame, if a frame is essentially dark or frozen (no salient content) but the luma grid changed, inter-frame motion supplies pulse energy and a motion-based centroid and spread, so dark or greyscale footage stays expressive.

Thresholds and scaling constants are heuristics, not calibrated detectors. Re-tune them (via `AnalysisOptions::sensitivity`) per dataset as needed.

## Analysis options

`AnalysisOptions` tunes the extraction:

| Option | Default | Effect |
| --- | --- | --- |
| `color_weight` | 0.25 | Blend of brightness vs chromatic salience in the pulse. 0.0 = brightness only; 1.0 = chromatic only (legacy behaviour). |
| `hue_focus` | `None` | Optional `(center, half_width)` hue band in 0..=1 to emphasize; pixels closer to the band count more. |
| `hue_focus_strength` | 1.0 | How strongly out-of-band hues are suppressed. Achromatic pixels are suppressed at half strength. |
| `sensitivity` | 1.0 | Pulse gain before normalization (clamped 0.25..=4.0). |

## Feature-to-sound mapping

| Visual feature | Current musical effect |
| --- | --- |
| Pulse strength | Drives a smoothed pulse level; stronger activity raises the pulse gate, influencing tempo, note density and the energy of bass, pad, melody and arpeggio layers |
| Horizontal position | Selects part of the initial musical fingerprint (key ratio and chord progression), shifts melody motif choice, and changes stereo balance |
| Vertical position | Shifts melody register (octave/fifth lifts) and overtone brightness |
| Spatial dispersion | Modulates arpeggio groove density |
| Brightness | Slightly lifts the overall mix level; contributes to initial fingerprinting and pulse strength |
| Saturation | Raises the crystalline 3rd harmonic of the melody timbre |
| Hue | Seeds the initial musical fingerprint (key and progression) alongside horizontal position |
| Warmth | Blends warm 2nd harmonic vs cool 3rd harmonic in the melody timbre |
| Motion | Adds to groove activity/tempo and drives pulse attacks between frames |

The generated texture combines low drone, bass, chord pad, melody and arpeggio components. Tempo varies within the implementation's current range as the pulse gate and motion change.

## How to interpret the sound

The sound is an artistic encoding of visual patterns. A listener may hear changes in activity, position, density or timbre as changes in musical energy, pitch or stereo placement. It should not be interpreted as a direct reading of any scientific quantity, a calibrated scientific alert, or a substitute for the source dataset's legend and metadata.

## Design principles for future mappings

For each new dataset, document:

1. The scientific variable being represented and its units.
2. The visual encoding used in the source visualisation (colour scale, geometry, animation, time axis).
3. The feature extraction method (and any `AnalysisOptions` tuning) with its known failure modes.
4. The mapping from each feature to pitch, rhythm, timbre, loudness or spatial audio.
5. Whether the mapping is monotonic and what a listener should infer.
6. A validation method with representative data and user testing.

A good sonification should be learnable and consistent. The same sound should mean the same thing within a dataset, and the documentation should explain when comparisons across datasets are not valid.
