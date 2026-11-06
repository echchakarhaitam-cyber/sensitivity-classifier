# Calibration, abstention and conformal sets: the background of Milestones 3-5

All three milestones work on one category at a time, with a score s(x) from a Milestone 2
model and a binary label y. They answer three different questions about the same score:
*does p = 0.8 mean 80%?* (calibration), *when may the model decide alone?* (abstention) and
*which labels cannot be excluded?* (conformal sets).

## Calibration (Milestone 3, `sensclf.calibration`)

A probability p is **calibrated** if, among all documents that receive p, a share p is
positive: P(y = 1 | p(x) = p) = p. A good ranking (high PR-AUC) does not imply it: any
increasing transform of a score keeps the ranking and changes the probabilities.

**Measuring it.** The *expected calibration error* (ECE) puts the predictions into 15
equal-width bins and averages |mean p - positive rate| over bins, weighted by bin size. It is
easy to read but biased upwards: with few documents per bin the absolute differences are noisy
and never cancel, which is why its bootstrap interval can lie above the point value. The
**Brier score** mean((p - y)^2) and the **log loss** -mean(y log p + (1 - y) log(1 - p)) are
proper scoring rules: they are minimised in expectation by the true conditional probability, so
they reward calibration and sharpness together and have no binning choice. The calibrator is
therefore selected by Brier score on dev.

**Post-hoc calibrators**, all fitted on held-out data (cal_fit) and never on the training set,
where the model is over-confident:

* *temperature scaling*: p = sigma(z / T), one parameter; keeps the ranking, only fixes over- or
  under-confidence (Guo et al., 2017);
* *Platt scaling*: p = sigma(a z + b), two parameters (Platt, 1999); turns SVM margins into
  probabilities;
* *isotonic regression*: the best non-decreasing step function from z to p
  (Zadrozny & Elkan, 2002); most flexible, needs more data and produces ties (flat steps).

Here z is the logit of a probabilistic model or the raw SVM margin.

**Why it can fail under shift.** A calibrator learns the map score -> frequency on the
calibration distribution. If the positive rate or the meaning of a score changes (a new
generator, other templates), the map is wrong. A more flexible map (isotonic) fits the old
distribution more closely and can be further off on the new one. Milestone 3 measures this:
isotonic is the best calibrator in-domain and the worst on test_ood.

## Abstention, or selective prediction (Milestone 4, `sensclf.selective`)

The model decides yhat = 1[p >= 0.5] with confidence c = max(p, 1 - p), and accepts a document
only if c >= lambda; otherwise it abstains (a human reviews it). **Coverage** is the accepted
share, **selective risk** the error rate among accepted documents. Moving lambda traces the
**risk-coverage curve**; its area (AURC, Geifman & El-Yaniv, 2017) summarises how well the
confidence ranks errors last, and the *oracle* AURC (all errors ranked last) is the floor.

**Certifying a threshold.** For a target risk r*, we want a lambda whose true selective risk is
<= r* with probability >= 1 - delta. On held-out data (cal_test), each candidate threshold
accepts n documents with k errors; if the true risk were above r*, seeing only k errors would
be unlikely, with probability at most the binomial tail P(Bin(n, r*) <= k). Testing K
candidates fixed in advance (coverages 5%, 10%, ..., 100%) at level delta / K each
(Bonferroni) and keeping the loosest rejected one gives the guarantee for whichever threshold is
returned, by the union bound. This is the "learn then test" view of risk control
(Angelopoulos et al., 2021). The guarantee assumes cal_test and future documents are
exchangeable; it says nothing under shift.

**Documents vs categories.** Accepting a document only when all six categories are accepted
does not give a 6-category guarantee: errors add up across categories. A union bound (r* / 6
per category) would, at a large cost in coverage; Milestone 4 reports the uncorrected
document-level error to make the gap visible.

## Conformal prediction sets (Milestone 5, `sensclf.conformal`)

Instead of one decision, output a **set** of labels that contains the truth with probability
>= 1 - alpha. With nonconformity a(y) = 1 - p(y), take the n held-out scores a_i(y_i) and
their ceil((n + 1)(1 - alpha))-th smallest value q. The set is {y : a(y) <= q}. If the new
document is exchangeable with the n held-out ones, its score is equally likely to fall at any
rank among the n + 1, which gives P(y in C(x)) >= 1 - alpha for any model, any n and any
distribution (Vovk et al., 2005; Angelopoulos & Bates, 2021). Without ties, coverage is also
<= 1 - alpha + 1 / (n + 1); with ties (isotonic steps) it can be higher.

For a binary label the set is {0}, {1} (a decision), {0, 1} (cannot exclude either: an
abstention) or empty (neither is plausible; it happens when alpha is large).

* **Marginal vs Mondrian.** The marginal guarantee averages over positives and negatives; with
  few positives the positives can be under-covered. Mondrian conformal computes one quantile
  per true class (Vovk et al., 2005), so coverage holds for positives and negatives separately.
  That matters here because a missed sensitive document is the costly error.
* **Joint sets.** Using alpha / 6 per category, the union bound gives P(all six true labels in
  their sets) >= 1 - alpha.
* **Groups.** Two renderings of the same Nemotron persona are not exchangeable with each
  other, so the quantile uses one document per group, which makes the guarantee hold over
  groups.

The guarantee is **marginal over the draw of the calibration documents**: one particular draw
can give a coverage slightly below 1 - alpha. Milestone 5 shows this by redrawing the documents
100 times.

## References

* Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. arXiv:2107.07511.
* Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn then
  test: calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
* Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks.
  NeurIPS.
* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
  networks. ICML.
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods. Advances in Large Margin Classifiers.
* Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
  Springer.
* Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass
  probability estimates. KDD.
