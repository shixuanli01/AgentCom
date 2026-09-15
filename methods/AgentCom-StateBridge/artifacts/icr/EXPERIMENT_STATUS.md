# ICR experiment status

Updated: 2026-09-15 UTC

The original five-seed plan was intentionally reduced by the experiment owner to three complete Text/StateBridge seed pairs and two complete LatentMAS seed pairs. Historical partial `seed_pair_03` and `seed_pair_04` artifacts were preserved and are not included in the formal pooled estimates.

| Step | Status | Main artifact | Notes |
| --- | --- | --- | --- |
| Step 1 | complete | `artifacts/icr_medqa300/seed_pair_00/analysis_v2/` | Frozen seed00 reanalyzed with 10,000 item-cluster bootstrap samples. |
| Step 2 | complete (3-seed agreed scope) | `artifacts/icr_medqa300/pooled_3seed/` | seed00/01/02; 12,600 directional-condition records; seven conditions; every condition has 600 records per seed. |
| Step 3 | complete (2-seed agreed scope) | `artifacts/icr_medqa300/pooled_2seed_with_latentmas/` | seed00/01; 12,000 directional-condition records; ten conditions; 10,000 bootstrap samples; directional classification `B_directional_signature`. |
| Step 4A | pending | — | Cross-benchmark core-condition validation has not started. |
| Step 4B | pending | — | Cross-benchmark causal controls have not started. |

## Validation summary

- Complete raw record counts: seed00 = 6,000, seed01 = 6,000, seed02 = 4,200.
- Every included condition contains exactly 600 records per included seed; all record keys are unique and all JSON records parse.
- Cache audit passes for the frozen seed00 source artifact and for seed01/02.
- Test suite: 50 passed.
- Text/StateBridge pooled across three seeds: Text accuracy 72.72%, StateBridge accuracy 71.06%; difference 1.67 percentage points (95% CI 0.83 to 2.50). StateBridge CR exceeds PR in every seed, and Text PR exceeds StateBridge PR in every seed.
- Two-seed channel comparison: Text accuracy 72.83%, StateBridge 71.42%, LatentMAS 71.17%. StateBridge CR−PR is +74.63 percentage points; LatentMAS CR−PR is −35.82 points, yielding the predeclared directional Pattern-B signature.

## Resume point

Review the Step-2 and Step-3 summaries above. If the agreed next phase is approved, continue with Step 4A cross-benchmark core conditions without changing the frozen ICR protocol or prompts.
