# Experiments

Experiment protocols and reports live here. Do not write run outputs into this
directory; generated runs use the ignored `artifacts/` tree.

## Current

- [ICR-V3 protocol (Chinese)](ICR_V3_PROTOCOL_ZH.md) — the frozen prompt,
  output-contract, and scoring specification for the full-set baseline sweep of
  `none`, `true_text`, `true_statebridge`, and `true_latentmas`.

V3 exists because V2 compared channels under visibly different prompts. It
fixes one five-block revision template with a single condition-varying slot,
adds a per-dataset output contract, and repairs five scoring defects. It
introduces no new communication method.

## Earlier tracks

### EGR — receiver-side evidence policy

- [EGR V1 method freeze (Chinese)](EGR_V1_METHOD_FREEZE_ZH.md)
- [EGR V1 cross-benchmark report (Chinese)](EGR_CROSS_BENCHMARK_REPORT_ZH.md)
- [EGR MedQA300 report (Chinese)](EGR_MEDQA300_REPORT_ZH.md)
- [EGR M1 protocol (Chinese)](EGR_M1_PROTOCOL_ZH.md)
- [Cross-benchmark first-wave protocol (Chinese)](CROSS_BENCHMARK_FIRST_WAVE_ZH.md)

EGR is a receiver-side evidence-integration protocol, not a latent transport.
Transport comparisons and receiver-policy comparisons are reported separately.

### TMR — training-free latent transport

- [Trajectory Memory Relay V1 (Chinese)](TMR_V1_PROTOCOL_ZH.md)
- [Turning-point hidden-state selection](TURNING_POINT_SELECTION.md)

`tmr_last64` isolates native external-memory transport; `tmr_coverage64`
changes only source-position selection. Implemented and smoke-tested; no full
accuracy result was produced.

### Multi-path predecessors

- [MP-StateBridge V1 plan (Chinese)](MP_STATEBRIDGE_V1_PLAN_ZH.md)
- [MP-StateBridge V1 plan (English)](MP_STATEBRIDGE_V1_PLAN_EN.md)
- [Multi-path oracle-miss audit (Chinese)](MP_STATEBRIDGE_ORACLE_MISS_AUDIT_ZH.md)
- [Refiner causal smoke (Chinese)](MP_STATEBRIDGE_REFINER_CAUSAL_SMOKE_ZH.md)
- [Concurrency benchmark](MP_STATEBRIDGE_CONCURRENCY_BENCHMARK.md)
- [LatentMAS channel audit](ICR_LATENTMAS_AUDIT.md)
