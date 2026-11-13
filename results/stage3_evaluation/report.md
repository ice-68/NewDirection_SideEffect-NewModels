# Stage 3 Site-Level Evaluation

Protocol: `stage3_uncertainty_v1`

## Site-Macro AUC Uncertainty

| model | estimate | ci_low | ci_high | n_sites | n_seeds |
| --- | --- | --- | --- | --- | --- |
| B0 | 0.7044 | 0.6460 | 0.7578 | 20 | 5 |
| B1 | 0.6362 | 0.5911 | 0.6813 | 20 | 5 |
| B2 | 0.5775 | 0.5353 | 0.6156 | 20 | 5 |
| B3 | 0.6148 | 0.5741 | 0.6508 | 20 | 5 |

The 95% intervals use 10,000 site-cluster bootstrap replicates. All
five seed results for a sampled site are retained together.

## Site By Model AUC

| test_site | n_subjects | B0 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- |
| CALTECH | 37 | 0.6988 | 0.4889 | 0.4725 | 0.5608 |
| CMU | 5 | 0.3333 | 0.5333 | 0.3333 | 0.3667 |
| KKI | 38 | 0.6603 | 0.6737 | 0.5526 | 0.6128 |
| LEUVEN_1 | 29 | 0.6429 | 0.6067 | 0.5610 | 0.6543 |
| LEUVEN_2 | 32 | 0.7336 | 0.6381 | 0.6275 | 0.6324 |
| MAX_MUN | 42 | 0.5694 | 0.4472 | 0.5537 | 0.5639 |
| NYU | 171 | 0.7822 | 0.7187 | 0.5831 | 0.6498 |
| OHSU | 20 | 0.6250 | 0.5000 | 0.5479 | 0.6229 |
| OLIN | 25 | 0.7143 | 0.5805 | 0.6494 | 0.7299 |
| PITT | 44 | 0.7598 | 0.6720 | 0.6683 | 0.7288 |
| SBL | 26 | 0.6071 | 0.5250 | 0.6190 | 0.6833 |
| SDSU | 33 | 0.6889 | 0.6913 | 0.5103 | 0.5095 |
| STANFORD | 36 | 0.6656 | 0.5771 | 0.4539 | 0.4786 |
| TRINITY | 44 | 0.6352 | 0.6766 | 0.5027 | 0.5847 |
| UCLA_1 | 55 | 0.6974 | 0.6074 | 0.6794 | 0.6593 |
| UCLA_2 | 20 | 0.8854 | 0.8479 | 0.7083 | 0.6562 |
| UM_1 | 80 | 0.7475 | 0.6609 | 0.5540 | 0.5869 |
| UM_2 | 31 | 0.9254 | 0.7825 | 0.6842 | 0.6588 |
| USM | 61 | 0.8478 | 0.7140 | 0.6032 | 0.6181 |
| YALE | 48 | 0.8689 | 0.7832 | 0.6860 | 0.7392 |

Each cell is the mean AUC across seeds 0-4 for that held-out site.

## Interpretation Boundary

These intervals describe between-site heterogeneity conditional on the
frozen predictions. They do not include model refitting or subject
resampling within sites. Any comparative or scientific interpretation
requires human review. [待人工复核]
