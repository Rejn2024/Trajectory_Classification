# Tactical checkpoint replays

Twenty Tacview recordings from the completed tactical population's validation-selected checkpoints: one clean win and one contested encounter per pilot. These are fresh inference recordings, not intermediate training-epoch demonstrations.

[Download the compressed replay archive](tactical_checkpoint_replays.zip). Extract it and open any `replays/pilot_*/*.txt.acmi` file in Tacview. The archive includes the encounter index, individual summaries and checksums.

```powershell
git lfs pull --include="shared/replays/tactical-checkpoints-v1/**" --exclude=""
Expand-Archive shared/replays/tactical-checkpoints-v1/tactical_checkpoint_replays.zip -DestinationPath artifacts/tacview
```

The learned observer is **A01**, shown as **Red** by the native simulator. The scripted tactical opponent is **B01**, shown as **Blue**. Both are F-16s. The RL adapter's blue/red terminology is reversed, so use these aircraft IDs.

These encounters were selected for visual interest from an earlier assessment. They are separate from the 3,000 fresh forecasting episodes and do not form an unbiased performance sample. All 20 reproduced their archived outcome category; trajectories can differ slightly with inference hardware and batching.

The ACMI 2.1 headers, timestamps, finite coordinates, aircraft identities, checksums and ZIP integrity have been checked. Three selected encounters have also been rendered and inspected in native Tacview; their [MP4 demonstrations](../../demonstrators/tactical-checkpoints-v1/README.md) can be played without Tacview. This does not constitute visual inspection of all twenty recordings.

| Pilot | Selection | Seed | Duration s | Outcome |
| --- | --- | ---: | ---: | --- |
| pilot_001 | clean_win | 1120018 | 95.5 | surviving_elimination |
| pilot_001 | contested | 1120200 | 120.0 | survived_without_elimination |
| pilot_002 | clean_win | 1120175 | 110.4 | surviving_elimination |
| pilot_002 | contested | 1120172 | 120.0 | survived_without_elimination |
| pilot_003 | clean_win | 1120259 | 116.8 | surviving_elimination |
| pilot_003 | contested | 1120286 | 120.0 | survived_without_elimination |
| pilot_004 | clean_win | 1120071 | 120.0 | surviving_elimination |
| pilot_004 | contested | 1120085 | 120.0 | survived_without_elimination |
| pilot_005 | clean_win | 1120142 | 118.3 | surviving_elimination |
| pilot_005 | contested | 1120278 | 120.0 | survived_without_elimination |
| pilot_006 | clean_win | 1120140 | 115.0 | surviving_elimination |
| pilot_006 | contested | 1120065 | 120.0 | survived_without_elimination |
| pilot_007 | clean_win | 1120071 | 119.1 | surviving_elimination |
| pilot_007 | contested | 1120142 | 120.0 | survived_without_elimination |
| pilot_008 | clean_win | 1120018 | 102.4 | surviving_elimination |
| pilot_008 | contested | 1120145 | 120.0 | survived_without_elimination |
| pilot_009 | clean_win | 1120196 | 102.7 | surviving_elimination |
| pilot_009 | contested | 1120145 | 120.0 | survived_without_elimination |
| pilot_010 | clean_win | 1120142 | 113.2 | surviving_elimination |
| pilot_010 | contested | 1120057 | 120.0 | survived_without_elimination |

[Manifest and checkpoint hashes](manifest.json)
