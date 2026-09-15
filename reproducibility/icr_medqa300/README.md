# ICR MedQA300 recovery capsule

This directory is the small, Git-native recovery layer. Large raw artifacts remain attached to GitHub Release `icr-medqa300-20260915`.

## Restore the repository

Normal GitHub path:

```bash
git clone https://github.com/shixuanli01/AgentCom.git
cd AgentCom
git checkout main
```

If using the separately uploaded Git bundle:

```bash
git clone AgentCom-icr-preservation-20260915.bundle AgentCom
cd AgentCom
git checkout main
```

## Restore artifacts

Download these Release assets into one directory:

```text
icr-artifacts-20260915.tar.zst
icr-artifacts-20260915.tar.zst.sha256
files.sha256
symlinks.txt
```

Then run:

```bash
sha256sum -c icr-artifacts-20260915.tar.zst.sha256
tar -I zstd -xf icr-artifacts-20260915.tar.zst
sha256sum -c files.sha256
```

The archive paths are repository-relative and should be extracted at the repository root.

## Restore the Python environment

The formal manifests are primary for experiment-critical versions. The saved `pip-freeze-postrun.txt` is a post-run snapshot and includes `python-docx`/`lxml`, which were added later only to create the Chinese report.

```bash
python3.10 -m venv .venv-agentcom
source .venv-agentcom/bin/activate
pip install -r methods/AgentCom-StateBridge/requirements.txt
```

Before inference, confirm at least:

```text
Python 3.10.21
PyTorch 2.7.1+cu128
Transformers 4.51.3
CUDA runtime 12.8
Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c
```

## Re-run analyses without model inference

From `methods/AgentCom-StateBridge`:

```bash
../../.venv-agentcom/bin/python -m icr.pooled_analysis \
  --base-root artifacts/icr_medqa300 \
  --replications seed_pair_00 seed_pair_01 seed_pair_02 \
  --num-bootstrap 10000 --bootstrap-seed 20260915

../../.venv-agentcom/bin/python -m icr.channel_analysis \
  --base-root artifacts/icr_medqa300 \
  --replications seed_pair_00 seed_pair_01 \
  --output-name pooled_2seed_with_latentmas \
  --num-bootstrap 10000 --bootstrap-seed 20260916
```

## Validate

```bash
cd methods/AgentCom-StateBridge
../../.venv-agentcom/bin/python -m pytest -q
```

Expected result at freeze time: `50 passed`.

See also:

- `docs/ICR_EXPERIMENT_LEDGER.md`
- `docs/ICR_PROTOCOL_AND_PROMPTS.md`
- `reports/ICR_MedQA300_主要结果与审计报告_2026-09-15.md`
- `methods/AgentCom-StateBridge/artifacts/icr/EXPERIMENT_STATUS.md`
