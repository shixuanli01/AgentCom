# Experiments

Experiment configs and launchers for the new AgentCom method live here. Do not
write outputs into this directory; generated runs will use an ignored
`artifacts/` tree once the first protocol is frozen.

## Current Experiment

- [EGR V1 method freeze (Chinese)](EGR_V1_METHOD_FREEZE_ZH.md)
- [EGR V1 cross-benchmark report (Chinese)](EGR_CROSS_BENCHMARK_REPORT_ZH.md)
- [Cross-benchmark first-wave protocol (Chinese)](CROSS_BENCHMARK_FIRST_WAVE_ZH.md)
- [EGR MedQA300 report (Chinese)](EGR_MEDQA300_REPORT_ZH.md)
- [Structured Latent Relay V1 (Chinese)](SLR_V1_PROTOCOL_ZH.md)
- [Trajectory Memory Relay V1 (Chinese)](TMR_V1_PROTOCOL_ZH.md)
- [Turning-point hidden-state selection](TURNING_POINT_SELECTION.md)
- [Single-path MedQA communication sweep (Chinese)](SINGLE_PATH_MEDQA_SWEEP_ZH.md)

The current communication audit is EGR V1 over the frozen ICR prebeliefs. EGR
is explicitly a receiver-side evidence-integration protocol, not a latent
transport replacement; transport comparisons and receiver-policy comparisons
are reported separately. SLR is the newest controlled latent method. It separates a 48-slot unordered
exploratory memory from a 16-slot ordered predictive decision trace. TMR remains
an active baseline: `tmr_last64` isolates native external memory transport,
while `tmr_coverage64` changes only source-position selection.

## Direct Predecessors

- [MP-StateBridge V1 experimental plan (Chinese)](MP_STATEBRIDGE_V1_PLAN_ZH.md)
- [MP-StateBridge V1 experimental plan (English)](MP_STATEBRIDGE_V1_PLAN_EN.md)
- [Multi-path oracle-miss audit (Chinese)](MP_STATEBRIDGE_ORACLE_MISS_AUDIT_ZH.md)
- [Reliable Oracle-path mechanism audit (Chinese)](MP_STATEBRIDGE_SOLID_ORACLE_MECHANISM_ZH.md)
- [Manual evidence-decision causal pilot (Chinese)](MANUAL_EVIDENCE_CAUSAL_PILOT_ZH.md)
- [Manual evidence causal study, 20-item expansion (Chinese)](MANUAL_EVIDENCE_CAUSAL_20_ZH.md)
- [Refiner causal smoke (Chinese)](MP_STATEBRIDGE_REFINER_CAUSAL_SMOKE_ZH.md)
- [Concurrency benchmark](MP_STATEBRIDGE_CONCURRENCY_BENCHMARK.md)
