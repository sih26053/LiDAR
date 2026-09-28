# Laya baseline (measured, current checkpoint, pre-change)
Generated: 2026-09-27T17:34:29Z
Replay run: benchmark-eval-1790525081, rows: 1631

No good/bad verdict is given; numbers only.

## validation:constrained_laya (n=146)
accuracy=1.0 balanced=1.0 macroF1=1.0 forward_rate=0.589 unsafe=0.0 emer_fwd=0.0 ece=0.3611 brier=0.1857 lat_p50=10167.485 lat_p95=11937.47
## validation:unconstrained_laya (n=75)
accuracy=0.2 balanced=0.25 macroF1=0.0833 forward_rate=1.0 unsafe=0.8 emer_fwd=1.0 ece=0.1959 brier=0.1964 lat_p50=9006.76 lat_p95=11701.587
## test:constrained_laya (n=86)
accuracy=1.0 balanced=1.0 macroF1=1.0 forward_rate=0.5349 unsafe=0.0 emer_fwd=0.0 ece=0.3397 brier=0.1722 lat_p50=9265.245 lat_p95=10314.32
## test:unconstrained_laya (n=50)
accuracy=0.2 balanced=0.25 macroF1=0.0833 forward_rate=1.0 unsafe=0.8 emer_fwd=1.0 ece=0.1996 brier=0.2012 lat_p50=9464.03 lat_p95=11812.046999999999
## ood_test:constrained_laya (n=200)
accuracy=0.995 balanced=0.6667 macroF1=0.7 forward_rate=0.78 unsafe=0.0 emer_fwd=0.0 ece=0.4446 brier=0.2536 lat_p50=9116.98 lat_p95=11691.753999999999
## ood_test:unconstrained_laya (n=200)
accuracy=0.78 balanced=0.25 macroF1=0.2191 forward_rate=1.0 unsafe=0.22 emer_fwd=1.0 ece=0.3448 brier=0.2907 lat_p50=9199.564999999999 lat_p95=13022.865999999998