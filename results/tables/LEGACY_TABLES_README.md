# Table index — source and evaluation scope

Every table below is derived from the per-record JSON artifacts under
`methods/AgentCom-StateBridge/artifacts/icr_v3/<task>_full_seed42/`. Percentages
are computed from integer counts; no percentage is aggregated from a rounded
percentage.

Scope vocabulary:
- **retained** — the (item, direction, condition) records in `revisions/merged.jsonl`: items where the three agents did **not** all answer correctly, × 6 ordered directions.
- **skipped sample** — records in `revisions/all_correct_sample.jsonl`: a sample of the all-three-correct items that phase 2 otherwise skips.
- **all items** — the full item set × 6 directions, only partly measured.

| Table | Source | Scope | Status |
|---|---|---|---|
| `sample_statistics.csv` | `prebeliefs/rank*/records/*.json`, `config.json` | all items | VERIFIED (GPQA row PENDING) |
| `count_identity_check.csv` | `prebeliefs/rank*/records/*.json` | all items and retained, separately | VERIFIED |
| `integrity_checks.csv` | raw `revisions/*/records/*.json` + `prebeliefs/` | all raw records on disk | VERIFIED |
| `main_results.csv` | `revisions/merged.jsonl` | retained | VERIFIED |
| `cr_pr_plot_data.csv` | `revisions/merged.jsonl` | retained | VERIFIED |
| `paired_vs_no_message.csv` | `revisions/merged.jsonl` | retained, and stratified by pair class | VERIFIED |
| `sr_scr.csv` | `revisions/merged.jsonl` + `all_correct_sample.jsonl` | retained (SR, SCR) and skipped sample (SCR) | VERIFIED |
| `answer_transitions.csv` | `revisions/merged.jsonl` | retained | VERIFIED; code-task rows are string identity only |
| `accuracy_scopes.csv` | `prebeliefs/merged.jsonl`, `revisions/merged.jsonl` | five scopes, one per column group | VERIFIED |
| `six_direction_detail.csv` | `revisions/merged.jsonl` | retained, split by the six ordered directions | VERIFIED |
| `cost.csv` | `revisions/merged.jsonl` `communication_payload`, `prompt_tokens`, `generation_length`, `generation_seconds` | retained | PARTIAL — construction cost MISSING for 3 of 4 channels |
| `truncation_sensitivity.csv` | `prebeliefs/` `hit_eos` + `revisions/merged.jsonl` | retained, recomputed with non-terminating items dropped whole | VERIFIED — sensitivity only, primary analysis is retention |
| `truncation_contamination.csv` | `prebeliefs/` `hit_eos` joined to `revisions/merged.jsonl` | retained | VERIFIED |
| `exclusions_and_reuse.csv` | `config.json`, shard paths, `prebeliefs/` | all items | VERIFIED |
| `full_set_accuracy_estimated.csv` | `icr/full_set_accuracy.py` over both merged files | all items, partly imputed | **ESTIMATED, NOT MEASURED** |
| `humanevalplus_degenerate_loops.csv` | HumanEval+ `prebeliefs/` and `revisions/` with `hit_eos == false` | all HumanEval+ records | VERIFIED |

Bootstrap columns in `main_results.csv`: item-cluster paired bootstrap, 2,000
resamples, analysis seed 20260921, percentile 2.5/97.5 interval. A resample whose
denominator is zero is dropped and counted in `bootstrap_undefined_*`. The
published `analysis_v2` artifacts use 10,000 resamples and seed 20260920; the
difference is recorded in `manifest.json`.

All five datasets are present in every table. GPQA-Diamond's phase 2 completed at
10:47 UTC on 2026-09-21 with 0 worker failures; its all-correct sample is 124
records per condition (25.8% of that population), because the run that would have
covered it in full was cut short deliberately to save wall-clock.
