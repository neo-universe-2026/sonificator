# Current audio mapping

This document describes the behaviour implemented in the current Sonificator source. It is specific to the orange-emission CO₂ visualisation prototype.

## Visual features

The frame analyser estimates:

- **Orange intensity:** the average weighted amount of qualifying orange colour in the sampled frame, scaled and gamma-adjusted into a 0–1 range.
- **Horizontal position:** the orange-weighted centre from left to right.
- **Vertical position:** the orange-weighted centre from top to bottom.
- **Dispersion:** a normalised estimate of how spread out the orange pixels are.

The thresholds and scaling constants are heuristic choices for the target visual style. They are not a calibrated detector and should be re-evaluated for other videos.

## Feature-to-sound mapping

| Visual feature | Current musical effect |
| --- | --- |
| Orange intensity | Drives a smoothed pulse level; stronger activity increases the pulse gate, which influences tempo and the energy of the musical layers |
| Horizontal position | Selects an initial musical fingerprint (key ratio and chord progression), slightly shifts melody pitch, and changes stereo balance |
| Vertical position | Measured and exposed through the API; no distinct synthesis mapping yet |
| Spatial dispersion | Measured and exposed through the API; no distinct synthesis mapping yet |

The generated texture combines low drone, bass, chord pad, melody and arpeggio components. Tempo is varied within the implementation's current range as the pulse gate changes.

## How to interpret the sound

The sound is an artistic encoding of visual patterns. A listener may hear changes in activity, position and density as changes in musical energy, pitch or stereo placement. It should not be interpreted as a direct reading of CO₂ concentration, a calibrated scientific alert, or a substitute for the source dataset's legend and metadata.

## Design principles for future mappings

For each new dataset, document:

1. The scientific variable being represented and its units.
2. The visual encoding used in the source visualisation (colour scale, geometry, animation, time axis).
3. The feature extraction method and its known failure modes.
4. The mapping from each feature to pitch, rhythm, timbre, loudness or spatial audio.
5. Whether the mapping is monotonic and what a listener should infer.
6. A validation method with representative data and user testing.

A good sonification should be learnable and consistent. The same sound should mean the same thing within a dataset, and the documentation should explain when comparisons across datasets are not valid.
