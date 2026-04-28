# IVOCT Segmentation Result Package

This package is prepared for sharing the current stage result with the advisor.

Compile `main.tex` with XeLaTeX on TeXPage or another LaTeX platform that supports Chinese.

Package layout:

- `main.tex`: detailed report source.
- `figures/`: all figures referenced by the report.
- `data/`: all JSON/Markdown result files referenced by the report.
- `code/`: key configuration, dataset, model, loss, metric, and training files.
- `pdf/`: placeholder directory for the compiled PDF.

Current main result:

- Strategy: MAE encoder + Conv decoder, LOPO-CV.
- Result file: `data/results_lopo_20260424_004401.json`.
- Mean Dice: `0.5212 ± 0.1424`.
- Global threshold: `0.3`.
