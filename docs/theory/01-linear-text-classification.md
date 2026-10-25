# Linear text classification: the background of Milestone 2

**TF-IDF.** A document becomes a sparse vector with one coordinate per n-gram (word 1–2 grams,
or character 3–5 grams inside word boundaries). The weight of term t in document d is
`tf(t, d) · idf(t)`. With `sublinear_tf`, `tf = 1 + log(count)`, so ten repetitions do not
count ten times as much as one. `idf(t) = log((1 + N) / (1 + df(t))) + 1` down-weights terms
found in many documents. Each vector is then scaled to unit L2 norm, so long and short
documents are comparable. Word n-grams capture field names and phrases ("blood type", "api
key"). Character n-grams also see formats (digit groups, `@`, separators), spelling variants
and words cut by punctuation, at the price of many more features.

**Logistic regression as a probabilistic model.** For one category,
`P(y = 1 | x) = σ(wᵀx + b)` with `σ(z) = 1 / (1 + e^{-z})`. Fitting minimises the negative
log-likelihood (log loss) plus an L2 penalty `‖w‖² / (2C)`: a small C means strong
regularisation. Log loss is a proper scoring rule, so in the limit of a well-specified model
and enough data the outputs approach true conditional probabilities. In practice the penalty,
the separability of high-dimensional text and any shift between training and test data all
distort them. Whether 0.8 means "80% of such documents are positive" is an empirical question
(calibration, Milestone 3), not a property guaranteed by the model.

**Linear SVM: margins, not probabilities.** LinearSVC minimises the hinge loss
`max(0, 1 − y·(wᵀx + b))` (y ∈ {−1, +1}) plus the same L2 penalty. The decision value
`wᵀx + b` is a signed distance-like margin. Its sign gives the class, and its order ranks
documents, so ROC-AUC and average precision are meaningful. Its scale is not a probability:
documents beyond the margin contribute no loss, so nothing ties a value of 2 to any frequency.
Brier score and log loss are therefore not computed for the SVM, and its default threshold is
0 rather than 0.5. Mapping margins to probabilities (Platt scaling, isotonic regression) needs
held-out data, which is what the reserved calibration split is for.

**One-vs-rest.** Each of the six categories gets its own binary model on shared features.
This ignores label correlations (they show up in the co-occurrence matrix in docs/DATA.md),
but it keeps every category's score and threshold separately interpretable.

**Operating points.** A score becomes a decision with a threshold. The default (0.5 on
probabilities, 0 on margins) assumes calibrated scores and equal error costs. The dev-F1
threshold is the one that maximises F1 on dev. It is chosen on dev only: choosing it on a test
set would make the test F1 an optimistic, in-sample number.

**PR-AUC vs ROC-AUC under class imbalance.** ROC-AUC is the probability that a random
positive is ranked above a random negative. It does not depend on prevalence, so it can stay
high while most of the top-ranked documents are false positives when positives are rare.
Average precision (the area under the precision–recall curve, computed as the step-wise sum
`Σ (R_k − R_{k−1}) P_k`) weights errors by how they affect the precision of the retrieved
set. Its no-skill value equals the prevalence, so it must always be read next to it. That is
why model selection here uses macro average precision, and why PR-AUC values are not
comparable across sets with different prevalences (e.g. test_ood).

**Why a cluster bootstrap.** The bootstrap estimates the sampling variability of a statistic
by recomputing it on sets resampled with replacement. It assumes the resampled units are
independent. Here they are not: each Nemotron persona appears twice (the `us` and `intl`
renderings share person, document type and often the annotated values). Resampling
documents would treat a pair as two independent observations and give intervals that are too
narrow. Resampling **groups** (all documents sharing a `group_id`, one draw = the whole group)
keeps the dependence inside each draw. Weighting documents by the number of times their group
is drawn makes this cheap for any statistic, ranking metrics included. Using the same group
draws for every model makes the intervals **paired**. A difference between two models then
gets its own interval, which is far narrower than comparing two overlapping marginal intervals.
Percentile intervals from 1000 resamples are approximate. They describe uncertainty from the
choice of documents, not from training randomness or from the label definition.
