# IVOCT Segmentation Beamer Slides

This is a XeLaTeX Beamer slide project based on the provided `presentation-slide` template.

Compile entry:

- `main.tex`

Recommended compiler:

- XeLaTeX

Package contents:

- `main.tex`: slide source.
- `bitbeamer.cls`, `bithesis.cls`: template class files.
- `latexmkrc`: template build config.
- `images/`: BIT logo assets.
- `figures/`: fold visualization images used in the slides.
- `data/`: result JSON and decision notes referenced by the slides.

Current result:

- Strategy: MAE encoder + Conv decoder reproduced baseline.
- Mean Dice: `0.5212 ± 0.1424`.
- Global threshold: `0.3`.
