# Laya threshold selection (validation-only, safety-prioritized)
Date: 2026-09-27T18:50:56Z, revision: 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
Calibration n=196, validation n=146
Legacy starting threshold: 0.35
Validated candidate: 0.35
Promoted to production: True
Current gate: 0.35

## Validation sweep (coarse)
| t | accepted | coverage | correct | unsafe | unnec STOP |
| 0.1 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.15 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.2 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.25 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.3 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.35 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.4 | 146 | 1.0 | 146 | 0 | 0.0 |
| 0.45 | 75 | 0.5137 | 75 | 0 | 0.4863 |
| 0.5 | 74 | 0.5068 | 74 | 0 | 0.4932 |
| 0.55 | 66 | 0.4521 | 66 | 0 | 0.5479 |
| 0.6 | 61 | 0.4178 | 61 | 0 | 0.5822 |
| 0.65 | 60 | 0.411 | 60 | 0 | 0.589 |
| 0.7 | 60 | 0.411 | 60 | 0 | 0.589 |
| 0.75 | 58 | 0.3973 | 58 | 0 | 0.6027 |
| 0.8 | 46 | 0.3151 | 46 | 0 | 0.6849 |
| 0.85 | 36 | 0.2466 | 36 | 0 | 0.7534 |
| 0.9 | 31 | 0.2123 | 31 | 0 | 0.7877 |

## Temperature groups (fit=calibration, judged=validation)
- n_options=1: T=1.0 deployed=False val_nll 0.0 -> 0.0
- n_options=2: T=0.25 deployed=True val_nll 0.2086 -> 0.0046
- n_options=3: T=0.25 deployed=True val_nll 0.6076 -> 0.075
- n_options=4: T=0.25 deployed=False val_nll 1.1679 -> 1.3027