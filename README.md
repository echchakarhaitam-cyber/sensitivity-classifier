# sensclf

A public, independent research project on document-level sensitive-content classification with
calibrated probabilities, selective prediction (abstention) and conformal guarantees, evaluated
in-domain and under domain shift.

Work in progress. No results yet. The task is defined in [docs/TASK.md](docs/TASK.md) and the
data in [docs/DATA.md](docs/DATA.md).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
```
