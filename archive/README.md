# archive/

`MDP_seizure_101_11.ipynb` is the original Google Colab development notebook,
copied here unchanged. It contains the complete history of the project: the
data pipeline, the classifier training, the 96-state model that was later
reduced to 72 states, every calibration sweep, every experiment that was
tried and rejected (velocity/volatility states, adaptive quantile bins,
per-patient refractory lengths, ...), and the cells that saved the session
caches.

Read it as a lab notebook, not as a program:

* **Values change across cells.** `CONFIG`, `BASE` and the level bins are
  mutated in place as the work progressed (for example `w_fa[Hard]` goes from
  5.0 to 2.0 in cell 66 and the upper level bin from 0.85 to 0.75 later).
  A cell's printed output reflects the globals in force when it ran, which
  are not necessarily the values defined above it.
* **Do not run cells selectively.** Many cells depend on globals set by
  earlier cells and several re-define functions with the same name.
* **The adopted configuration is `configs/final.yaml`** and the code of
  record is `src/seizure_mdp/`. Anything that disagrees with those two is
  superseded.
* Some comments are in Arabic; they are working notes, not documentation.
