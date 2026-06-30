# WCA Control Manifests

Formal V21+ experiments must start from a control manifest, not from a hand-written queue JSON.

Allowed formal workflow:

```bash
python3 scripts/wca_exp_control.py plan configs/control/v21/example.json
python3 scripts/wca_exp_control.py validate configs/control/v21/example.json
python3 scripts/wca_exp_control.py submit configs/control/v21/example.json -- --remote root@example.com
python3 scripts/wca_exp_control.py status configs/control/v21/example.json -- --remote root@example.com
python3 scripts/wca_exp_control.py fetch configs/control/v21/example.json -- --remote root@example.com
python3 scripts/wca_exp_control.py report configs/control/v21/example.json
```

The generated queue JSON under `artifacts/control/<experiment_id>/generated_queue.json` is an execution artifact. It is not the source of scientific truth.

Current policy:

- V20a/V20b/V20c are frozen historical queues.
- V20d is the first strict PDEBench formal entrypoint.
- V21+ strict queues must contain `generated_from_manifest`, `control_manifest_path`, and `control_manifest_sha256`.
- `allow_submit=false` means the manifest is a draft and must not be submitted.
- The control plane must fetch raw artifacts before analysis: queue logs, run summaries, train logs, eval plans, per-sample rows, reports, and provenance manifests.
- Do not deploy or rsync new source code into the remote workspace while a formal queue is running from that workspace. A mid-queue deploy makes later jobs execute a different source tree than the queue's deploy manifest, invalidating source traceability. Wait for the queue to finish, fetch artifacts, then deploy the next committed code version.
- Strict fetch defaults to a single remote artifact archive. Per-file rsync is available only as a fallback/debug mode because many small SSH transfers are more likely to fail and can leave incomplete evidence bundles.
