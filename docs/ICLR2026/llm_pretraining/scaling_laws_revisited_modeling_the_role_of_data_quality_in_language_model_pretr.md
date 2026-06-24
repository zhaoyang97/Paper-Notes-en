---
title: >-
  [Paper Note] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining
description: >-
  [ICLR 2026][LLM Pretraining][Data Quality] This paper introduces a dimensionless data quality parameter $Q \in (0,1]$ into the classic Chinchilla scaling law, obtaining $L(N,D,Q)=A/N^\alpha + B/(D^\beta Q^\gamma) + E$. Through systematic controlled experiments involving noise injection in machine translation and causal language modeling, the authors demonstrate that loss decreases predictably with improved data quality, and high-quality data can compensate for smaller model s…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Data Quality"
  - "Scaling Laws"
  - "Effective Sample Size"
  - "Chinchilla"
  - "Pretraining"
date: 2026-05-08
content_hash: f2684f4a57bc4ed8
---

# Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=x54wwB6QvL](https://openreview.net/forum?id=x54wwB6QvL)  
**Area**: LLM Pretraining / Scaling Law  
**Keywords**: Data Quality, Scaling Laws, Effective Sample Size, Chinchilla, Pretraining

## TL;DR
This paper introduces a dimensionless data quality parameter $Q \in (0,1]$ into the classic Chinchilla scaling law, obtaining $L(N,D,Q)=A/N^\alpha + B/(D^\beta Q^\gamma) + E$. Through systematic controlled experiments involving noise injection in machine translation and causal language modeling, the authors demonstrate that loss decreases predictably with improved data quality, and high-quality data can compensate for smaller model sizes and lower computational costs.

## Background & Motivation

**Background**: Scaling laws, represented by Kaplan and Chinchilla, have accurately characterized how loss changes with model scale $N$ and data volume $D$, serving as a benchmark for guided compute allocation in large-scale training. However, these laws typically assume a "fixed quality" for training data, modeling only along the $N$ and $D$ axes.

**Limitations of Prior Work**: While the intuition that "cleaner data yields better models" is common, it has not been quantified within a scaling law. In practice, practitioners improve data via filtering, deduplication, and denoising, observing that "gains from filtering are comparable to increasing compute," yet no formula exists to predict how much loss reduction or parameter savings a specific quality improvement $X$ can yield. In specialized domains like medicine or commerce, where corpora are naturally scarce and vary in quality, the lack of a quantitative framework makes it impossible to trade off between data cleaning efforts and model scaling.

**Key Challenge**: Data quality is a multi-dimensional and fuzzy concept (often decomposed into accuracy, completeness, consistency, timeliness, uniqueness, and validity). Directly inserting these dimensions into scaling laws is neither solvable nor estimable. Conversely, ignoring quality makes the laws decoupled from reality. A sufficiently simple, smoothly degradable, and estimable abstraction of quality is required.

**Goal**: (1) Provide a single scalar definition for data quality; (2) Embed it into the Chinchilla form in a theoretically consistent manner; (3) Validate the predictive power of this law through controlled experiments and provide practical guidance on "trading quality for scale."

**Key Insight**: The authors borrow from the perspectives of "effective sample size" and information theory—inferior data essentially reduces the proportion of "usable information" in a dataset, equivalent to shrinking $D$ samples into $D \cdot g(Q)$ effective samples. This perspective is supported by classic results in PAC learning, Fisher information, and channel capacity, naturally leading to a multiplicative $Q^\gamma$ correction term.

**Core Idea**: A dimensionless scalar $Q \in (0,1]$ is used to characterize the usable information of a corpus ($Q=1$ for perfectly clean, smaller values for noisier data). This is used as a multiplier for the effective sample size in the data term of the scaling law, resulting in $B/(D^\beta Q^\gamma)$.

## Method

### Overall Architecture
Ours does not propose a new model but rather a **quality-aware scaling law** along with its theoretical derivation and estimation methods. The logic follows three steps: first, providing an operational scalar definition of data quality $Q$ (two estimators); second, justifying why $Q$ should enter the scaling law in the multiplicative form $Q^\gamma$ via "effective sample size" and "information theory" paths; and finally, fitting $B,\beta,\gamma,E$ through controlled experiments with synthetic noise to verify predictive power.

The final quality-aware scaling law is:

$$L(N, D, Q) = \frac{A}{N^\alpha} + \frac{B}{D^\beta Q^\gamma} + E$$

where $N$ is the number of parameters, $D$ is the number of training tokens, $Q$ is the data quality, and $E$ is the irreducible loss. When $Q=1$ (highest quality), it reduces to the standard Chinchilla law $L=A/N^\alpha + B/D^\beta + E$. Smaller $Q$ leads to larger loss; conversely, larger $Q$ requires less $D$ to achieve the same loss, mathematically expressing the trade-off between quality and data volume/compute.

### Key Designs

**1. Two Estimators for Scalar Quality Q: Compressing Fuzzy "Quality" into a Smoothly Degradable Number**

The pain point is that quality is multi-dimensional. The authors require $Q$ only to "degrade smoothly with contamination and proxy the proportion of usable information." Two complementary estimators are provided. The first is **corruption rate ($CR$)**: if the dataset corruption rate is $CR$ (assuming $0\le CR<1$), then $Q(\omega)=1-CR$—where 10% contaminated tokens correspond to $Q=0.9$, estimable via standard sampling. The second is a more general **deficiency** $\Delta(\omega)$, satisfying positivity, continuity, and additivity, leading to $Q(\omega)=\exp(-\Delta(\omega))$. Deficiency can be decomposed into noise, coverage/diversity, redundancy, and synthetic data:

$$\Delta(\omega) = \mu_1 E + \mu_2 \frac{1}{F} + \mu_3 G + \mu_4 H$$

The elegance of this decomposition is that by choosing specific forms for each term, it can **repropose various quality scaling laws in literature** (e.g., clustering density from Chen et al., diversity + synthesis from Chang et al., diminishing returns of repeated epochs from Goyal et al.), proving $Q$ is a unified container.

**2. Effective Sample Size Factorization: Using Fisher Information / PAC Theory to Justify Q^γ**

This is the theoretical foundation. The core assumption (Assumption 1) is the existence of a monotonic link function $g:[0,1]\to\mathbb{R}^+$ with $g(1)=1$, such that the loss for a given parameter size satisfies:

$$L_N(D,Q) \approx \frac{B}{D_{\text{eff}}^\beta} = \frac{B}{(D\cdot g(Q))^\beta}$$

Meaning "$D$ dirty samples = $D \cdot g(Q)$ clean samples." The authors anchor the exponent of $g(Q)\approx Q^\gamma$ using two classic results: in additive Gaussian noise regression (Lemma 1), Fisher information per observation is proportional to $1/\sigma^2$, total info $I_D\propto D/\sigma^2$. Defining $\Delta=\ln(\sigma^2/\sigma_0^2)$ and $Q=e^{-\Delta}=\sigma_0^2/\sigma^2$ results in $D_{\text{eff}}=D\cdot Q^\gamma$ with $\gamma=1$, recovering the classic "$D \cdot$ SNR" scaling. In symmetric label noise (Lemma 2, flip rate $\eta$), the corrected loss variance scales by $(1-2\eta)^{-2}$, and the effective sample size shrinks to $D\cdot(1-2\eta)^2=D\cdot(2Q-1)^2$. In the high-quality range $Q\in(1/2,1]$, this is locally approximated by $Q^\gamma$ with $\gamma\approx 2$. Both paths support $D_{\text{eff}}=D\cdot g(Q)$, theoretically grounding the $Q^\gamma$ form.

**3. Information Theoretical Evidence: Multiplicative Decay of Mutual Information**

To ensure the law does not rely on a single assumption, a second derivation is provided (Proposition 1). Viewing dirty tokens $\tilde{X}$ as the output of clean tokens $X$ passing through a memoryless channel $C_Q$, the authors assume contamination causes a multiplicative decay in usable mutual information: $I(\tilde{X};Z)=\rho(Q)\,I(X;Z)$, where $\rho(1)=1$, $\rho(Q)$ is monotonic, and $\rho(Q)\approx cQ^\gamma$ near $Q\to 1$. Combined with information-theoretic generalization bounds $L_D\propto 1/(D\cdot I(\tilde{X};Z))^\beta$, regrouping constants yields $L(N,D,Q)\approx A/N^\alpha + B/(D^\beta Q^\gamma)+E$. For binary symmetric channels, $\gamma\approx 2$; for Gaussian noise, $\gamma\approx 1$. These converging views are central to the argument's strength.

### Loss & Training
Experiments optimize context-averaged cross-entropy loss, reporting the same loss on a held-out test set to measure the law's predictive power on out-of-sample data. Law parameters $B,\beta,\gamma,E$ are estimated using the parameterization fitting process of Hoffmann et al., employing both Least Squares and Huber regression (Huber is more robust to outliers).

## Key Experimental Results

Experiments used decoder-only models for two tasks: Neural Machine Translation (NMT, En-De, Paracrawl v8, ~133M params 8-layer GPT-Neo) and Causal Language Modeling (CLM, C4-en, 8-layer Llama-3). Each task involved 3 data sizes × 7 quality levels, run across 3 seeds (63 runs total). Quality was controlled by injecting synthetic noise: for NMT, 50% of non-special tokens in selected samples were replaced with pad; for CLM, 50% were replaced with random vocabulary tokens. Noise sample ratios $\eta=\{0,10,20,25,30,40,50\}\%$ correspond to $Q=\{1.0,0.9,0.8,0.75,0.7,0.6,0.5\}$. Data volumes for NMT were 0.5M/1M/2M sentence pairs; for CLM, 0.1B/1B/10B tokens.

### Main Results: Fitted Parameters of the Quality-Aware Scaling Law

| Task | Fitting Method | $B$ | $\beta$ | $\gamma$ | $E$ |
|------|---------|------|---------|----------|------|
| NMT | Least Squares | 166.57 | 0.263 | 0.185 | 0.147 |
| NMT | Huber | 139.60 | 0.250 | 0.173 | 0.067 |
| CLM | Least Squares | 1428.23 | 0.395 | 0.389 | 3.440 |
| CLM | Huber | 1441.51 | 0.396 | 0.401 | 3.439 |

Loss decreases predictably as data volume $D$ and quality $Q$ increase. Notably, the estimated quality exponent $\hat\gamma$ is **significantly less than 1** (NMT $\approx 0.173$, CLM $\approx 0.401$), implying that effective data volume decays **sub-linearly** with quality—models are more robust to moderate contamination than PAC learning/information theory (which predict $\gamma\ge 1$) suggest.

### Ablation Study / Out-of-Distribution Validation (CLM)

| Setting | Method | $B$ | $\beta$ | $\gamma$ | $E$ |
|------|------|------|---------|----------|------|
| CLM (in-dist) | Least Squares | 1428.23 | 0.395 | 0.389 | 3.440 |
| CLM (unseen) | Least Squares | 1589.07 | 0.397 | 0.332 | 4.552 |
| CLM (in-dist) | Huber | 1441.51 | 0.396 | 0.401 | 3.439 |
| CLM (unseen) | Huber | 1427.30 | 0.391 | 0.337 | 4.540 |

Evaluating the trained model on unseen data shows fitted $\beta,\gamma$ values very close to in-distribution ones, demonstrating scaling generalization.

### Key Findings
- **Sub-linear decay ($\gamma<1$) is a core conclusion**: Natural language contains redundancy. Even partially contaminated samples carry context (syntax, alignment, co-occurrence), so loss grows slower than linearly as $Q$ decreases—$Q$ must drop significantly for loss to spike. This provides quantitative support for the utility of "not-so-clean" data.
- **Autoregressive tasks are more noise-sensitive**: The CLM $\gamma\approx 0.40$ is significantly larger than NMT's $\gamma\approx 0.17$. Explanation: CLM token swapping destroys local dependencies and increases entropy; in NMT, even if half the tokens are padded, cross-sequence alignment and context leak sufficient information. $\gamma$ serves as a "Robustness Index"—smaller is more noise-resistant.
- **Quality effects can be cleanly isolated**: Defining $\Delta L(Q)=L(N,D,Q)-L(N,D,1)\approx \hat B D^{-\hat\beta}(Q^{-\hat\gamma}-1)$, plotting $\Delta L(Q)$ against $Q^{-\hat\gamma}-1$ yields a stable line through the origin across different data volumes. This confirms that $A/N^\alpha+E$ terms do not change with $Q$.
- **Noise proxy validity**: As synthetic noise increases, embedding similarity strictly decreases monotonically, proving synthetic noise is a reasonable proxy for semantic degradation.

## Highlights & Insights
- **Compressing "Quality" into a scalar for the Law**: The most ingenious move is avoiding the multi-dimensional taxonomy of quality, instead requiring $Q$ only to satisfy "smooth degradation + usable info proxy." This makes the framework operational and backward compatible with Chinchilla.
- **Dual-perspective derivation**: Converging from both effective sample size (Fisher/PAC) and information theory (Channel Mutual Information) to $B/(D^\beta Q^\gamma)$ makes the law more credible than simple empirical fitting.
- **$\gamma$ as a transferable "Robustness Index"**: Interpreting the quality exponent as the degree of noise resistance for a task-model pair allows this to be transferred to any scenario assessing if data cleaning is "worth it."
- **Deficiency decomposition unifies literature**: The formula $\Delta=\mu_1 E+\mu_2/F+\mu_3 G+\mu_4 H$ incorporates several quality-related scaling works, positioning this as a meta-framework.

## Limitations & Future Work
- **Reliance on synthetic noise injection**: Experiments use $pad$ and token swaps to control $Q$. While embedding similarity decreases, real web corpora contamination (factual errors, domain shift, low-quality paraphrasing) differs from i.i.d. synthetic noise.
- **Small model scale and single epoch**: NMT ~133M, CLM 8-layer with 10B tokens total. The $A/N^\alpha$ term was not fully explored, and the trade-off conclusion relies on extrapolation.
- **Practical estimation of Q remains open**: The law's utility depends on estimating $Q$ for real corpora. Identifying "bad" samples for corruption rates or calibrating $\mu_i$ in deficiency remains difficult in practice.

## Related Work & Insights
- **vs Chinchilla (Hoffmann et al. 2022)**: Adds a $Q^\gamma$ term to the data component, making Chinchilla a special case ($Q=1$).
- **vs Bansal et al. 2022 (NMT Noise Scaling)**: They found noise shifts the curve without changing exponents; ours explicitly parameterizes quality as $Q$ with its own exponent $\gamma$, upgrading from observation to a predictive law.
- **vs Goyal et al. 2024 (Data filtering is not compute-invariant)**: They noted the value of high-quality data depends on the budget; ours uses $D_{\text{eff}}=D\cdot Q^\gamma$ to formulate these "iso-loss" contours mathematically.
- **vs Chen et al. 2025 / Chang et al. 2024**: These use clustering density or compression diversity/synthesis; ours unifies them via the deficiency decomposition $\Delta$.

## Rating
- Novelty: ⭐⭐⭐⭐ Formally embedding data quality into Chinchilla with theoretical consistency.
- Experimental Thoroughness: ⭐⭐⭐ Clean controlled experiments, but small scale and reliant on synthetic noise.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation and complete chain from theory to estimation.
- Value: ⭐⭐⭐⭐ Provides calculated guidance for "clean data vs scale" trade-offs, especially for domain-specific models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[ICLR 2026\] Learned Meta-Tokens for Language Modeling](learned_meta-tokens_for_language_modeling.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[ICLR 2026\] Learned Meta-Tokens for Language Modeling](learned_meta-tokens_for_language_modeling.md)
- [\[ICLR 2026\] Reformulation for Pretraining Data Augmentation](reformulation_for_pretraining_data_augmentation.md)

</div>

<!-- RELATED:END -->
