# Ten-flight validation sample

This is the end-to-end validation sample for notebook 11: **10 physical flights / 20 perspectives**, not the requested 10,000-flight training campaign. Its complete flag refers only to the ten-flight validation configuration. The full campaign is prepared for the user's training machine.

# NN versus NN collection

Two recorded perspectives per physical engagement, using frozen notebook-07 tactical pilots.
manifest.json reports the actual flight count; complete is true only at the requested count.
Both perspectives share scenario_group. Keep them in the same future split.
Observations are idealised 51-feature controller inputs at 10 Hz. Target decisions are privileged.
Decisions occur at 1 Hz before the next physics step. Terminal samples have no new decision.
No forward filling, fitted scaling, splits, or forecast windows are stored.
The native forced-throttle setting is preserved; catalogue speed labels may not change throttle.
A01/ and B01/ each contain observations, tracks and target_decisions, sharded by 100 flights by default.
episodes.parquet has one row per perspective (twice the physical flight count).
Use scripts/generate_nn_duel_corpus.py load_forecasting_view for a 09b-compatible episode.
The existing 09b loader is fixed to its original corpus; it does not auto-discover this dataset.
See notebooks/11_nn_vs_nn_training_data.ipynb for collection, verification, skill gaps and publication.
