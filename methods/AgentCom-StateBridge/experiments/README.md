# Experiments

Experiment configs and launchers for the new AgentCom method live here. Do not
write outputs into this directory; generated runs will use an ignored
`artifacts/` tree once the first protocol is frozen.

## Current Experiment

- [Turning-point hidden-state selection](TURNING_POINT_SELECTION.md)

This is the current controlled comparison against StateBridge `last_k`. It
keeps the message budget and alignment fixed and changes only which positions
are selected.

## Direct Predecessors

- [MP-StateBridge V1 experimental plan (Chinese)](MP_STATEBRIDGE_V1_PLAN_ZH.md)
- [MP-StateBridge V1 experimental plan (English)](MP_STATEBRIDGE_V1_PLAN_EN.md)
- [Multi-path oracle-miss audit (Chinese)](MP_STATEBRIDGE_ORACLE_MISS_AUDIT_ZH.md)
- [Refiner causal smoke (Chinese)](MP_STATEBRIDGE_REFINER_CAUSAL_SMOKE_ZH.md)
- [Concurrency benchmark](MP_STATEBRIDGE_CONCURRENCY_BENCHMARK.md)
