# Stage 1 build report

- atlas: **CC200**  (T = 78, R = 200)
- samples: **877**
- cache tensor `X`  : (877, 78, 200) float32  (temporal input, truncated to T)
- cache tensor `FC` : (877, 200, 200) float32  (graph input, FULL-length series)
- standardisation is deliberately NOT baked in (fitted per training fold)

## Site x class

| site | n | ASD | control | raw length min/median/max |
|---|---|---|---|---|
| CALTECH | 37 | 19 | 18 | 146/146/146 |
| CMU | 5 | 3 | 2 | 236/316/316 |
| KKI | 38 | 12 | 26 | 124/152/152 |
| LEUVEN_1 | 29 | 14 | 15 | 246/246/246 |
| LEUVEN_2 | 32 | 13 | 19 | 246/246/246 |
| MAX_MUN | 42 | 18 | 24 | 116/116/196 |
| NYU | 171 | 73 | 98 | 176/176/176 |
| OHSU | 20 | 12 | 8 | 78/78/78 |
| OLIN | 25 | 14 | 11 | 206/206/206 |
| PITT | 44 | 21 | 23 | 196/196/196 |
| SBL | 26 | 14 | 12 | 196/196/196 |
| SDSU | 33 | 12 | 21 | 176/176/176 |
| STANFORD | 36 | 17 | 19 | 176/235/236 |
| TRINITY | 44 | 21 | 23 | 146/146/146 |
| UCLA_1 | 55 | 28 | 27 | 116/116/116 |
| UCLA_2 | 20 | 8 | 12 | 116/116/116 |
| UM_1 | 80 | 36 | 44 | 296/296/296 |
| UM_2 | 31 | 12 | 19 | 296/296/296 |
| USM | 61 | 38 | 23 | 232/236/236 |
| YALE | 48 | 22 | 26 | 196/196/196 |

## Totals

- ASD: 407  control: 470
- sites: 20  folds per seed: 20  seeds: 5
- test evaluations: 5 seeds x 20 folds = 100 site-folds

Each fold: train = all other sites, val = label-stratified slice of those
training sites, test = the held-out site (never used for model selection).