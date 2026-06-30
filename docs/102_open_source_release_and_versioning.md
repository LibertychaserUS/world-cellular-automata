# WCA Open-Source Release and Versioning Protocol

Status: active release protocol  
Owner: main agent  
Scope: public GitHub, Hugging Face, and Zenodo-facing WCA releases

## 1. Purpose

The WCA project needs a release chain that lets external readers reproduce the
research without exposing local operating details. Public release is not a dump
of the working tree. It is a curated product with explicit evidence boundaries,
versioned artifacts, and machine-checkable manifests.

The public structure is split into three repositories:

| Repository | Purpose | Contents | Exclusions |
| --- | --- | --- | --- |
| `world-cellular-automata` | Main code and reproducible experiment entrypoints | `src/`, selected `scripts/`, selected configs, tests, control-plane tools, public docs | raw VPS logs, remote host configs, caches, checkpoints, incomplete branches, raw artifacts |
| `world-cellular-automata-technical-report` | Human-readable research record | PDFs, LaTeX, figures, tables, release manifest, citation files | private paths, internal queue traces, raw experiment logs |
| `world-cellular-automata-artifacts` | Optional heavy evidence package | result tables, prediction panels, compact run manifests, Zenodo handoff metadata | secrets, SSH commands, full raw datasets unless license allows, uncurated cache trees |

The code repository should be enough to run smoke tests, inspect the model, and
reproduce selected public experiments given the documented external datasets and
hardware. Heavy results belong in release assets, Hugging Face datasets, or
Zenodo records, not in the main code tree.

## 2. Version Model

Use coordinated, evidence-aware versions.

| Component | Version prefix | Example | Meaning |
| --- | --- | --- | --- |
| Code package | `vMAJOR.MINOR.PATCH` | `v0.1.0` | Source/runtime API and reproducibility entrypoints |
| Report release | `report-vMAJOR.MINOR` | `report-v0.1` | Narrative, figures, tables, and claim state |
| Artifact release | `artifacts-vMAJOR.MINOR` | `artifacts-v0.1` | Tables, prediction samples, compact logs, checksums |
| Experiment protocol | `protocol-vN` | `protocol-v28b` | Specific manifest/eval contract family |

Version rules:

1. Patch version changes may fix tests, docs, packaging, or reporting without
   changing a scientific claim.
2. Minor version changes may add experiments, figures, or public APIs.
3. Major version changes may alter the WCA core contract, the strict-eval
   protocol, or public claim boundaries.
4. A report release must declare the exact code commit, report commit, and
   artifact manifest hash it references.
5. An artifact release must never rely on a private path as the only locator.

## 3. Evidence States

Every public claim must carry one evidence state:

| State | Meaning | Allowed in main claims? |
| --- | --- | --- |
| `formal` | Matched contract, clean split, strict report gate, no hard sentinel failure | yes |
| `bounded` | Valid under a narrower contract or diagnostic scope | yes, with scope |
| `exploratory` | Useful signal but not strict enough for a formal result | no, appendix or roadmap |
| `frozen-invalid` | Known contract or protocol defect | no, incident log only |

Raw-field anchors and token-level models must not be mixed in one formal table
unless the metric space has been explicitly matched. If the same plot includes
both, the figure caption must label the comparison as contextual or exploratory.

## 4. Required Public Release Artifacts

Each release must contain:

1. `README.md`: public scope, quick start, what is and is not claimed.
2. `LICENSE`: source license.
3. `CITATION.cff`: citation metadata.
4. `release_manifest.json`: file paths, byte sizes, SHA256 hashes, generation
   time, source commit, and evidence-state table.
5. `REPRODUCIBILITY.md`: setup, smoke tests, data acquisition, expected outputs.
6. `SECURITY_AND_PRIVACY.md`: excluded data classes and secret handling.
7. `docs/`: public architecture, experiment protocol, and evidence-tree docs.

The release manifest is the authority. If a file is not in the manifest, it is
not part of the public release.

## 5. Never Publish

The release gate must block:

- SSH commands, ports, passwords, private keys, or known-host files.
- API tokens, GitHub/Hugging Face tokens, cloud credentials.
- VPS hostnames, private remote roots, raw queue logs with operational metadata.
- `.DS_Store`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.venv`.
- raw checkpoints unless explicitly licensed and intentionally released.
- full raw datasets unless redistribution rights are checked.
- internal chat transcripts or notes containing credentials.
- half-finished experiment manifests that were not reviewed for public contract
  status.

## 6. Gate Sequence

Before publishing a code release:

1. Build a clean export with `scripts/build_open_source_export.py`.
2. Run secret scan on the export.
3. Run Python smoke tests from the export.
4. Run TS control-plane tests from the export when TS files are included.
5. Verify `release_manifest.json` hashes.
6. Confirm no artifact path points outside the exported root.
7. Create a GitHub release from the clean export tag.
8. Upload large report/artifact packages to Hugging Face or Zenodo only after
   the manifest is generated.

Before publishing a report release:

1. Rebuild PDFs from LaTeX.
2. Regenerate all figures and tables from reviewed `paper/tables`.
3. Remove local metadata files.
4. Recompute `release_manifest.json`.
5. Zip and test the archive.
6. Upload to GitHub release and Hugging Face dataset repo.

Before publishing an artifact release:

1. Compress only curated tables, prediction panels, and compact reports.
2. Include raw data only when license-compatible and necessary.
3. Include `artifact_manifest.json` with SHA256 and source run IDs.
4. Include `DATA_CARD.md` describing provenance and limitations.
5. Upload to Hugging Face dataset repo; mint or update Zenodo DOI from the same
   package when stable.

## 7. Repository Naming

Recommended public names:

- `world-cellular-automata`
- `world-cellular-automata-technical-report`
- `world-cellular-automata-artifacts`

Avoid names that hide the architecture (`wca-report`) or overclaim the result
(`wca-sota-world-model`). The name should communicate the architecture first.

## 8. Current Release Plan

Current immediate release targets:

1. Keep `world-cellular-automata-technical-report` as the report repository.
2. Create `world-cellular-automata` from a clean allowlisted export.
3. Upload the report release package to Hugging Face under the authenticated
   user namespace.
4. Defer `world-cellular-automata-artifacts` until large artifacts have a
   finalized data card and manifest.

## 9. Acceptance Criteria

A public release is complete only when:

- the exported repository is buildable from a clean checkout;
- all release files are present in the manifest;
- tests listed in `REPRODUCIBILITY.md` pass;
- no blocked evidence state is presented as a main result;
- no private host, key, token, or local-only path is present;
- GitHub and Hugging Face upload URLs are recorded in the release manifest.

