# EGR V1 evidence channel under ICR-V3 (three agents), MedQA

Retained set: 120 questions, 720 directed records per condition. Baselines read from the published artifact; EGR run in an isolated root.

| Condition | CR | PR | SI | Acc_ret | adopts wrong sender |
|---|---:|---:|---:|---:|---:|
| No Message | 8/116 = 6.90% | 114/116 = 98.28% | 52.59% | 185/720 = 25.69% | 0.9% |
| Full Text | 93/116 = 80.17% | 49/116 = 42.24% | 61.21% | 196/720 = 27.22% | 57.8% |
| StateBridge | 67/116 = 57.76% | 87/116 = 75.00% | 66.38% | 208/720 = 28.89% | 25.0% |
| LatentMAS | 81/116 = 69.83% | 38/116 = 32.76% | 51.29% | 178/720 = 24.72% | 67.2% |
| **EGR evidence** | 96/116 = 82.76% | 47/116 = 40.52% | 61.64% | 199/720 = 27.64% | 59.5% |
| CTRL other-evidence | 0/3 = 0.00% | 2/2 = 100.00% | 50.00% | 4/26 = 15.38% | 0.0% |

## Paired, EGR evidence against each baseline

### vs Full Text: 658/720 identical answers (91.4%), net +3 records

| Stratum | EGR right, baseline wrong | baseline right, EGR wrong | net |
|---|---:|---:|---:|
| correction_opportunity | 9 | 6 | +3 |
| destruction_risk | 11 | 13 | -2 |
| both_wrong | 3 | 1 | +2 |

### vs StateBridge: 592/720 identical answers (82.2%), net -9 records

| Stratum | EGR right, baseline wrong | baseline right, EGR wrong | net |
|---|---:|---:|---:|
| correction_opportunity | 33 | 4 | +29 |
| destruction_risk | 3 | 43 | -40 |
| both_wrong | 4 | 2 | +2 |

