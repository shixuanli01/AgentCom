# MedQA StateBridge — replication seed_pair_01

Produced 2026-09-22. This is the second sampling of the revision pass on MedQA.
The phase-1 beliefs and the StateBridge prefixes are the frozen artifacts,
reached by symlink and unchanged; only `replication_id` differs, which changes
`revision_seed` and therefore the receiver's sampling.

## Provenance

| Field | Value |
|---|---|
| Artifact root | `runs/medqa/replication_01` |
| Beliefs and prefixes | symlink to `artifacts/icr_v3/medqa_full_seed42` (unmodified) |
| replication_id | `seed_pair_01` |
| global_seed | 42 |
| Model | Qwen/Qwen3-4B |
| Generation | temperature 0.6, top_p 0.95, max_new_tokens 16384 |
| Revision prompt | `icr_v3_mid_injection` |
| Answer parser | `icr.parsing_v3@ICR-V3` |
| Dataset sha256 | `ddee8dc64d3b2a1e5061c409c68d626109237640009168838a038db29efe802c` |
| Config fingerprint | `355c5073cd75bb6fba78bad536a8081cca7855c7263452fe59411417cc2b8f52` |
| Scope | 120 retained questions x 6 directed pairs = 720 records |
| Failures | 0 workers, 0 records with status != complete, 0 non-terminating |

## Headline metrics

95% intervals are 10,000 item-cluster bootstrap resamples, analysis seed 20260922.

| Metric | Count | Value | 95% CI |
|---|---:|---:|---:|
| CR | 63/116 | 54.31% | [42.59%, 65.83%] |
| PR | 83/116 | 71.55% | [60.83%, 81.45%] |
| SR | 2/436 | 0.46% | [0.00%, 1.21%] |
| SCR | 52/52 | 100.00% | [100.00%, 100.00%] |
| Acc_ret | 200/720 | 27.78% | [21.94%, 33.89%] |
| SI | — | 62.93% | [55.45%, 70.24%] |

## Behaviour

| Quantity | Value |
|---|---:|
| Answer changed | 132/720 = 18.33% |
| Kept receiver's prior answer | 588/720 |
| Adopted sender's answer | 128/720 |
| Moved to a third answer | 4/720 |
| Unparsed | 0/720 |
| Adopted sender in the CR stratum | 63/116 = 54.31% |
| Adopted a wrong sender in the PR stratum | 33/116 = 28.45% |

## Cost

| Quantity | Mean |
|---|---:|
| Receiver prompt tokens | 983 |
| Receiver output tokens | 905 |
| Payload | 64 states x 2560 dims, 327,680 bytes |
| Revision seconds per record | 20.18 |

Payload bytes are not convertible to an equivalent token count. The alignment
cost is recorded on the phase-1 belief, not per message, so per-message
construction cost is MISSING.

## Files

| File | Contents |
|---|---|
| `statebridge_seed01_summary.csv` | the headline table with counts and CIs |
| `statebridge_seed01_by_direction.csv` | all four subsets split by the six directed pairs |
| `statebridge_seed01_records.csv` | all 720 records, one row each |

## What this run does and does not establish

It establishes that the MedQA StateBridge rates are stable under resampling. It
does not replace seed_pair_00: the two are samples from the same distribution,
neither is more correct, and the other four datasets and the other three
channels have only seed_pair_00. Quoting seed_pair_01 for StateBridge alongside
seed_pair_00 for Full Text, LatentMAS and No Message would make MedQA's paired
comparisons cross two different revision samplings, which is what those
comparisons are designed to hold fixed.
