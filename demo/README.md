# Demo model

`model.npz` holds the numbers of the selected model (Milestone 3: `word_svc` with isotonic
calibration): the 200,000-term vocabulary, idf weights, the six LinearSVC weight vectors and
intercepts, and the six isotonic breakpoint lists. `rules.json` holds the certified abstention
thresholds (results/m4/selective.json) and conformal quantiles (results/m5/conformal.json),
unchanged.

They are produced by `python scripts/export_demo_model.py`, which refits the isotonic maps
exactly as the Milestone 3 script and stops unless they reproduce the committed run's cal_test
probabilities. `sensclf.portable` recomputes the model's scores with NumPy only (for the
in-browser demo, where scikit-learn pickles cannot be loaded); tests compare it with
scikit-learn.

The model was trained on synthetic data from Nemotron-PII (NVIDIA Corporation, CC BY 4.0) and
Gretel synthetic PII finance (Gretel.ai, Apache-2.0); see docs/DATA.md. It contains no real
personal data.
