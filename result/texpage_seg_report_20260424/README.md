# IVOCT Segmentation Baseline Report

Upload this directory to TeXPage and compile `main.tex`.

Recommended compiler:

- XeLaTeX

Project contents:

- `main.tex`: report source
- `figures/`: fold visualization images used by the report
- `data/`: JSON/Markdown evidence files referenced by the report

Current best result:

- Strategy: reproduced baseline, MAE encoder + Conv decoder, LOPO-CV
- Result file: `/root/CN_seg_baseline/seven/seg/logs/results_lopo_20260424_004401.json`
- Mean Dice: `0.5212 ± 0.1424`
- Global threshold: `0.3`
