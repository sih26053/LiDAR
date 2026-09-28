# Laya OOD test (held-out families, frozen config)
Date: 2026-09-27T18:49:42Z
Families: ood_big, ood_dense, ood_dropout, ood_narrow, ood_noisy, ood_pose, ood_sparse, ood_wide

## ood_test:constrained_laya: acc=0.995 bal=0.6667 F1=0.7 unsafe=0.0 ece=0.4446
## ood_test:unconstrained_laya: acc=0.78 bal=0.25 F1=0.2191 unsafe=0.22 ece=0.3448

Constrained failures: 1
- ood_wide D_BOTH_SIDES_AVAILABLE: oracle=right proposed=left (n=1)