---
title: Sensitive Text Classifier
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Classifier that abstains when unsure, with certified risk
---

# A sensitive-text classifier that can say "I don't know": demo

Paste a document: for six categories of personal data the page shows the calibrated probability,
the answer under a certified error-rate target (yes / no / not sure) and the conformal
prediction set. It runs in the browser with Pyodide and NumPy, using the study's selected model
and the thresholds and quantiles of its committed results.

Code and results: https://github.com/echchakarhaitam-cyber/sensitivity-classifier (this Space's
files are kept in its `space/` folder; the page loads `src/sensclf/portable.py`, `demo/model.npz`
and `demo/rules.json` from a pinned commit).
