# RunPod pilot

Use one on-demand RTX 4090 Pod with the official CUDA-enabled PyTorch template,
Python 3.11 or newer, SSH access, and approximately 30 GB of workspace storage.
Check the live hourly quote before deployment. The authorized initial budget is
$20 total; storage and idle time also count. The first pilot completed on 2026-10-10; its artifacts are saved locally under
`benchmarks/results/runpod_session_20261010`.

Upload the prepared local archive; a fresh GitHub clone omits the evaluation
corrections and CUDA changes. Extract it into `/workspace`, then run:

```bash
cd /workspace/rgtta
bash runpod/setup.sh
bash runpod/pilot.sh
```

The setup preserves the template's CUDA-enabled PyTorch and records the resolved
environment. The pilot compares TTA and RG-TTA on compact iTransformer and
PatchTST models, one synthetic recurring-regime dataset, one seed, and H=96.
This measures feasibility before selecting larger models or launching a full
study. It is not the AE's requested scalability experiment.

Results are under `benchmarks/results/runpod_gpu_pilot`. Download that directory
and `runpod/environment-resolved.txt` before terminating the Pod. The one-hour
process timeout does **not** stop the Pod or its billing. Configure and verify
the Pod stop mechanism before leaving an instance unattended. Do not launch a
larger batch until pilot runtime, peak memory, and the remaining budget are
checked. CUDA timing is synchronized; CPU multiprocessing remains supported,
but CUDA runs require `--workers 1`.

For agent-managed creation, run `python3 runpod/save_api_key.py` in your Mac's
interactive terminal. It prompts without displaying the key and writes it to
`/private/tmp/rgtta-runpod-api-key` with owner-only permissions, outside the
repository and deployment bundle. Return to chat and say `ready`; do not paste
the key. Existing files are not overwritten. Delete the file and revoke the key
when remote management is no longer needed.

Verification: 24 targeted tests passed both locally and on an RTX 4090 Pod.
The CUDA pilot completed with PyTorch 2.8.0+cu128 and Python 3.12.3. Shell syntax
checks pass, and the credential helper was checked with a dummy token. Explicit
`cuda:0` selection is required by the tested cloud PyTorch build. The pilot took
11.2 seconds across two compact architectures and two policies; this is not a
runtime estimate for the full study. The downloaded environment and source
manifest identify the exact executed configuration.

## Full matrix

`runpod/run_matrix.py` saves one JSON checkpoint per model/dataset/horizon/seed,
validates finite errors and evaluated batches, and refuses to resume if source,
data, or settings changed. The default matrix has 672 experiments and six
adaptation policies (4,032 policy results). `runpod/full.sh` then runs a separate
672-experiment retraining matrix. TAFAS variants are unavailable in this checkout.
Failures are recorded explicitly; an incomplete matrix exits nonzero.

The October 10 full session uses a $0.89/hour RTX 4090, a bounded remote run,
periodic locally verified result backups, and a local eight-hour stop watchdog.
Keep the controlling computer awake and online. These local controls are not a
RunPod billing cap. Final backup verification precedes Pod termination. Model
checkpoint binaries are excluded from the result archives; metrics, provenance,
resolved dependencies, and logs are retained. Results are not submission-ready
until completeness and scientific validity have been reviewed.

## Completed revision artifacts (2026-10-12)

The GPU queue is finished and the Pod terminated. Its final archive is
`benchmarks/results/runpod_resume_session_20261010/results.tar.gz`.
Unlike the early pilot archive, the final archive retains upstream `.pth`
checkpoints for the released-code comparisons. Historical notes above describe
individual earlier sessions, not the final archival policy.

Run `python3 runpod/finalize_results.py` to validate all three archived sources
and merge the corrected RG and DLinear retraining replacements. Run
`python3 runpod/render_paper_results.py` to generate tables and the figure, then
`bash paper/build_revision.sh` to compile the revised six-page draft.
See `paper/revision_notes.txt` for measured results and unresolved scientific
comparisons. Do not resume historical training directly from the current source:
its hashes have changed; the archived deployments preserve each executed version.
