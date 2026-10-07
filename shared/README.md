# Sharing training results and flight corpora

Use Git LFS for selected training results and datasets in `shared/`. Both developers
get the same files at the same repository paths. Git records small pointers; the Git
host's LFS service stores the contents. No separate storage account is needed if your
Git host provides LFS and the repository has sufficient storage and download quota.

The root `.gitattributes` sends everything under `shared/` to LFS except Markdown,
YAML, and the small JSON files named `manifest.json`, `config.json`, `metrics.json`,
`summary.json`, and `label_map.json`. Keep those exceptions small. Bulk JSON/CSV logs,
model weights, Parquet, NumPy arrays, ACMI replays, and videos use LFS automatically.

## Published populations

- [Notebook 05 combat population, 2026-10-06](models/notebook05-combat-20261006T140858_362365Z/README.md):
  10 selected PPO pilots, each trained for 150 epochs with 75 training scenarios.
  Includes original configs/manifests and complete epoch metrics. Checkpoints and
  logs total 5.68 MiB in LFS; the release README includes a selective download
  command, loading example, validation results, and source-provenance limitations.

## Large-file transfer test

`lfs-test/payload-128mib.bin` is a 128 MiB (134,217,728 byte) synthetic binary payload
for testing an LFS commit, upload, and download. It contains pseudorandom bytes, not
flight data or a trained model. After pulling it, verify the contents from the
repository root:

```powershell
Get-FileHash -Algorithm SHA256 shared/lfs-test/payload-128mib.bin
```

Expected SHA-256:
`2ffa0ac2ed377277367f89b23b6d9ecf147b57b89f5cf52c1745774bcaee7e40`.

## Layout

```text
shared/
  README.md
  models/
    heading-pilot-v001/
      policy.pt
      config.yaml
      manifest.yaml
  runs/
    2026-10-05-alice-heading-001/
      training_metrics.jsonl
      metrics.json
      manifest.yaml
  corpora/
    heading-pilot-v001-batch-0001/
      manifest.yaml
      label_map.json
      episodes.parquet
      trajectories/
        shard_000.parquet
        shard_001.parquet
      splits.parquet
```

These are suggested paths, not generated artifacts. Use unique run/batch IDs so both
developers can publish without overwriting one another. Keep the canonical dataset
layout produced by the notebooks: `manifest.yaml`, `label_map.json`,
`episodes.parquet`, and `trajectories/`. Save actual train/validation/test assignments
alongside it when available.

Training and generation continue to write to ignored local output folders such as
`artifacts/`. Copy completed, selected outputs into `shared/` for publication. The
existing ignore rules also exclude directories named `datasets`, `checkpoints`, and
`mlruns` anywhere in the tree; use the layout above instead of those names inside
`shared/`. LFS attributes do not override Git ignore rules.

## Setup on each developer's machine

Install [Git LFS](https://git-lfs.com/) if `git lfs version` is unavailable. From the
repository root, run:

```powershell
git lfs install --local
git lfs pull
```

The local install sets up this checkout's filters and upload hook. Each developer
needs it; hooks are not shared through commits. The `.gitattributes` file is shared
through Git. If an existing pre-push hook conflicts, use `git lfs install --manual`
and integrate its instructions into that hook.

For a fresh checkout after installing LFS:

```powershell
git clone <repository-url>
cd Trajectory_Classification
git lfs install --local
git lfs pull
```

## Publish a completed result

First copy the chosen checkpoint, configuration, and manifest into a new
`shared/models/<agent-version>/` directory. Copy the completed flight dataset into a
new `shared/corpora/<agent-version>-batch-<number>/` directory. Include selected run
metrics under `shared/runs/<run-id>/` if useful. Do this once writers have finished so
each publication is a consistent snapshot.

For example, after preparing the `heading-pilot-v001` paths above:

```powershell
git add .gitattributes shared/README.md shared/models/heading-pilot-v001 shared/corpora/heading-pilot-v001-batch-0001
git lfs status
git lfs ls-files
git diff --cached --stat
git commit -m "Share heading pilot v001 and its first flight batch"
git push
```

Check that the weights and Parquet files appear in `git lfs ls-files` before
committing. The pre-push hook uploads LFS contents during `git push`. Files added
before these attributes existed need to be added again under the new rules; these
rules do not convert old Git history.

The other developer downloads the published version with:

```powershell
git pull
git lfs pull
```

Point the consuming notebook or script at the chosen `shared/` dataset directory.
A fresh clone with LFS enabled normally downloads the current checkout's LFS files
automatically. To opt into selective downloads in an existing checkout:

```powershell
git lfs install --local --skip-smudge
git pull
git lfs pull --include="shared/models/heading-pilot-v001/**,shared/corpora/heading-pilot-v001-batch-0001/**" --exclude=""
```

This avoids automatic downloads on subsequent checkouts/pulls; it does not remove
files already downloaded. Omitted LFS files remain pointers until pulled. Restore
automatic downloads with `git lfs install --local --force`.

## Identify the exact agent behind each corpus

Freeze one checkpoint for each corpus version. Record its SHA-256, its repository
path, the code commit, the simulator revision, generation settings, scenario seeds,
and observation/action definitions. A filename such as `best_model.pt` alone does
not identify the agent, because training can overwrite it.

Keep the generator's existing manifest fields and add a `provenance` section. This
example contains placeholders to replace with actual values:

```yaml
provenance:
  code_commit: "<full Git commit used for generation>"
  code_dirty: false
  policy_path: shared/models/heading-pilot-v001/policy.pt
  policy_sha256: "<SHA-256 of that exact checkpoint>"
  training_run: "2026-10-05-alice-heading-001"
  simulator_commit: db4f95657c081d120442022a339bed62d340c93b
  policy_config_path: shared/models/heading-pilot-v001/config.yaml
  generation_config_path: config.yaml
  action_selection: deterministic
  seed_start: 100000
  episode_count: 1000
  sample_hz: 10
  feature_schema_version: "1.0.0"
  label_schema_version: "1.0.0"
```

The checkpoint hash can be obtained in PowerShell with:

```powershell
Get-FileHash -Algorithm SHA256 shared/models/heading-pilot-v001/policy.pt
git rev-parse HEAD
git status --short
```

Save the actual policy and generation configurations at the recorded paths; include
observation ordering/normalization, action mapping, simulator settings, and dependency
versions. If generation used uncommitted code, preserve that code in a commit or save
the patch and record it rather than claiming a clean source revision. These manifest
fields are a publication convention; the current generators do not populate them
automatically. A weights-only checkpoint may support evaluation without supporting
training resumption; record which capability the published model provides.

## Keep growth manageable

- Publish selected checkpoints and completed batches. Give each developer a distinct
  batch ID and non-overlapping scenario seed range within a corpus.
- Keep published shards unchanged. Add another batch as the corpus grows. A changed
  LFS file is stored as another complete object, so repeatedly rewriting a single
  multi-GB archive or trajectory file increases storage quickly.
- As an initial target, use compressed Parquet shards around 100-250 MB and tune from
  actual output sizes. Episode counts alone do not guarantee a byte-size bound. Keep
  whole episodes in one split, including both aircraft perspectives; persist splits
  across batches so repeated seeds/scenarios do not leak between train and test.
- Keep large replay/video collections optional. Most consumers only need the model,
  episode metadata, and trajectory shards.
- A corpus from one trained agent measures that agent's behaviour over the chosen
  scenarios. Testing generalization to other agents needs additional policies.
- Deleting a file from the current branch does not remove its historical LFS objects
  or reclaim the host's storage quota automatically.

For GitHub, the documentation checked on 2026-10-07 lists **10 GiB storage and 10 GiB
monthly download bandwidth** for Free/Pro, with a **2 GB per-file limit**. Team has
different allowances. Storage includes historical versions; downloads by collaborators
and CI count against the repository owner's allowance. For example, two fresh downloads
of a 4 GiB corpus use approximately 8 GiB of download bandwidth. Other Git hosts have
their own limits. Recheck the repository owner's plan before the first large push.

If recurring downloads or retained versions outgrow the chosen LFS plan, the same
immutable Parquet layout and provenance can later be used with DVC and object storage.

Sources: [GitHub LFS configuration](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage),
[GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs),
[GitHub per-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
[selective LFS pull](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-pull.adoc).
