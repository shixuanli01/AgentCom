# Experiments

Experiment configs and launchers for the new AgentCom method live here. Do not
write outputs into this directory; generated runs will use an ignored
`artifacts/` tree once the first protocol is frozen.

## Current Experiment

- [Trajectory Memory Relay V1 (Chinese)](TMR_V1_PROTOCOL_ZH.md)
- [Turning-point hidden-state selection](TURNING_POINT_SELECTION.md)
- [Single-path MedQA communication sweep (Chinese)](SINGLE_PATH_MEDQA_SWEEP_ZH.md)

TMR is the active controlled comparison. `tmr_last64` isolates native external
memory transport from StateBridge's Procrustes pseudo-token transport.
`tmr_coverage64` then changes only source-position selection.

## Direct Predecessors

- [MP-StateBridge V1 experimental plan (Chinese)](MP_STATEBRIDGE_V1_PLAN_ZH.md)
- [MP-StateBridge V1 experimental plan (English)](MP_STATEBRIDGE_V1_PLAN_EN.md)
- [Multi-path oracle-miss audit (Chinese)](MP_STATEBRIDGE_ORACLE_MISS_AUDIT_ZH.md)
- [Reliable Oracle-path mechanism audit (Chinese)](MP_STATEBRIDGE_SOLID_ORACLE_MECHANISM_ZH.md)
- [Manual evidence-decision causal pilot (Chinese)](MANUAL_EVIDENCE_CAUSAL_PILOT_ZH.md)
- [Manual evidence causal study, 20-item expansion (Chinese)](MANUAL_EVIDENCE_CAUSAL_20_ZH.md)
- [Refiner causal smoke (Chinese)](MP_STATEBRIDGE_REFINER_CAUSAL_SMOKE_ZH.md)
- [Concurrency benchmark](MP_STATEBRIDGE_CONCURRENCY_BENCHMARK.md)
