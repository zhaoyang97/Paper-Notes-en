---
title: >-
  [Paper Note] Reliable Algorithm Selection for Machine Learning-Guided Design
description: >-
  [ICML 2025][Computational Biology][Algorithm Selection] Proposed a design algorithm selection method that formulates the success determination of candidate design algorithm configurations as a multiple hypothesis testing problem. By incorporating Prediction-Powered Inference (PPI) techniques to correct prediction errors, the method guarantees with high probability the selection of algorithm configurations that satisfy user-defined success criteria on unlabelled design distrib…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Algorithm Selection"
  - "ML-Guided Design"
  - "Prediction-Powered Inference"
  - "Multiple Hypothesis Testing"
  - "Protein Design"
date: 2026-05-08
content_hash: 939794cd360fe474
---

# Reliable Algorithm Selection for Machine Learning-Guided Design

**Conference**: ICML 2025  
**arXiv**: [2503.20767](https://arxiv.org/abs/2503.20767)  
**Code**: [GitHub](https://github.com/clarafy/design-algorithm-selection)  
**Area**: Biological Sequence Design / Statistical Inference  
**Keywords**: Algorithm Selection, ML-Guided Design, Prediction-Powered Inference, Multiple Hypothesis Testing, Protein Design

## TL;DR

Proposed a design algorithm selection method that formulates the success determination of candidate design algorithm configurations as a multiple hypothesis testing problem. By incorporating Prediction-Powered Inference (PPI) techniques to correct prediction errors, the method guarantees with high probability the selection of algorithm configurations that satisfy user-defined success criteria on unlabelled design distributions.

## Background & Motivation

- **Practical Challenges of ML-Guided Design**: In protein/RNA design, researchers must select design algorithms (such as AdaLead, CbAS, etc.) along with their hyperparameters and prediction models. These choices directly dictate the design performance.
- **Unreliable Predictions**: Sequences produced by design algorithms usually deviate from the training distribution, where the prediction models can exhibit large out-of-distribution errors—high predicted values do not equate to high ground-truth labels.
- **Expensive Annotation**: Wet-lab validation of designed sequences is highly costly (synthesis + measurement), necessitating reliable algorithm selection prior to annotation.
- **Limitations of Prior Work**:
    - Relying solely on predictions: Misled by prediction errors.
    - Calibrating predictions: Focuses on the uncertainty of individual designs, rather than directly serving algorithm selection decisions.
    - Bayesian Optimization: Aims to approach global optima iteratively, rather than guaranteeing that the current round's designs meet the criteria.

**Core Problem**: How to select design algorithm configurations that satisfy user-defined success criteria without acquiring design labels?

## Method

### Overall Architecture

**Formulating algorithm selection as multiple hypothesis testing** (Algorithm 1):

1. For each configuration $\lambda$ in the menu $\Lambda$, generate $N$ designs and obtain their predictions using its prediction model.
2. Define the null hypothesis $H_\lambda$: configuration $\lambda$ is unsuccessful, i.e., $\theta_\lambda := \mathbb{E}_{Y \sim P_{Y;\lambda}}[g(Y)] < \tau$.
3. Calculate the p-value for each null hypothesis (based on PPI).
4. Apply Bonferroni correction and output $\hat{\Lambda} = \{\lambda \in \Lambda: p_\lambda \le \alpha/|\Lambda|\}$.

**High-Probability Guarantee**: $\mathbb{P}(\theta_\lambda \ge \tau, \forall \lambda \in \hat{\Lambda}) \ge 1-\alpha$.

### Key Designs

**1. Flexible Definition of Success Criteria**

$$\theta_\lambda = \mathbb{E}_{Y \sim P_{Y;\lambda}}[g(Y)] \ge \tau$$

- Mean design label: $g(y)=y$
- Exceedance rate: $g(y)=\mathbf{1}[y \ge \gamma]$ (e.g., at least 10% of design labels exceed the wild-type)
- Users can customize $g$ and $\tau$ according to practical requirements

**2. Prediction-Powered p-values** (Algorithm 2)

Core Idea: Correct estimation bias based solely on predictions utilizing hold-out annotated data.

$$\hat{\theta} = \underbrace{\frac{1}{N}\sum_{i=1}^N g(\hat{y}_i^\lambda)}_{\text{预测部分 }\hat{\mu}} + \underbrace{\frac{1}{n}\sum_{j=1}^n w_j(g(y_j) - g(\hat{y}_j))}_{\text{偏差校正 }\hat{\Delta}}$$

where $w_j = p_{X;\lambda}(x_j)/p_{\text{lab}}(x_j)$ is the density ratio between the design distribution and the annotated data distribution.

Standard error: $\sigma^2 = \frac{\hat{\sigma}^2_{\text{pred}}}{N} + \frac{\hat{\sigma}^2_{\text{err}}}{n}$

p-value: $P = 1 - \Phi\left(\frac{\hat{\theta}-\tau}{\sigma}\right)$

**3. The Role of Density Ratio**

- Hold-out annotated data must be reweighted using the density ratio, because covariate shift exists between the design distribution and the annotated data distribution.
- **Known Density Ratio**: e.g., sequences originate from a known combinatorial library (NNK library) or an autoregressive generative model.
- **Unknown Density Ratio**: Estimate the density ratio using multi-class logistic regression (MDRE).

**4. Finite-sample Guarantees** (Algorithm 3 + Theorem 3.1)

By replacing the normal approximation with Hoeffding's inequality, non-asymptotically valid p-values are obtained, guaranteeing that Theorem 3.1 holds.

### Loss & Training

- Prediction models: Ridge regression, fully-connected NN ensembles, CNN ensembles (depending on the task)
- Training data: 5k-10k annotated sequences
- Hold-out data: 5k annotated sequences for PPI correction
- Design sequences: $N=50k$-$1M$ samples per configuration

## Key Experimental Results

### Main Results

**Experiment 1: Protein GB1 Binding Affinity Design**

- Design space: 4 sites $\times$ 20 amino acids = $20^4=160,000$ variants (fully annotated data available)
- Menu: 101 temperature hyperparameters $\lambda \in [0.2, 0.7]$
- Success criterion: Mean design label $\ge \tau$

| Method | Error Rate Control | Selection Rate |
|------|-----------|--------|
| Prediction-only | 100% Error Rate | High |
| CalibratedForecasts | 100% Error Rate | High |
| GMMForecasts(q=0) | 0% Error Rate | Too conservative, unselected under many $\tau$ |
| **Ours** | **<\alpha=10%** | **100% when \tau\in[0,1]** |

- Ours maintains an error rate $< \alpha$ across all $\tau$, while retaining high selection rates across a wide range of success criteria.
- The Prediction-only method is severely misled by prediction errors, where the true mean of selected configurations falls far below $\tau$.

**Experiment 2: RNA Binding Energy Design**

- Menu: 78 configurations (5 design algorithms $\times$ multiple hyperparameters $\times$ 3 prediction models)
- Density ratio requires estimation (MDRE)

| Method | Error Rate when \tau < 0.32 | Selection Rate |
|------|---------------|--------|
| Prediction-only | 100% | High |
| **Ours** | **\approx 0%** | **Reasonable** |
| GMMForecasts(q=0) | 0% | Extremely conservative |

- Even when the density ratio must be estimated, the error rate of Ours remains far lower than alternative methods.
- Even in occasional misselections, the ground-truth labels of unsuccessful configurations remain close to $\tau$ (yielding mild consequences).

### Ablation Study

- **Known vs. Estimated Density Ratio**: When known, the theoretical guarantees are strictly satisfied; when estimated, the error rate is slightly higher than $\alpha$ but still significantly outperforms the baselines.
- **Menu Size**: Increasing from 78 to 249 configurations, the Bonferroni correction leads to a drop in the selection rate of approximately 10-20%.
- **Different Success Criteria**: Results remain consistent under the exceedance rate criterion $g(y)=\mathbf{1}[y\ge 1]$.

### Key Findings

- Methods relying solely on predictions almost always make incorrect selections (100% error rate) because the design distribution deviates from the training distribution.
- Setting aside annotated data for PPI correction (even if it halves the training data) is more valuable than utilizing all data solely for training.
- The essence of the method is to answer "in which regions of the design space do we lack statistical evidence?"—if the design distribution deviates too far from the annotated distribution, returning an empty set is a reasonable behavior.

## Highlights & Insights

1. **Precise Problem Formulation**: Rigorously formulates the practical question of "which design algorithm to select" as a statistical hypothesis testing problem, providing probabilistic guarantees.
2. **Ingenious Application of PPI**: Corrects prediction bias using a small amount of annotated data instead of attempting to calibrate individual predictions—directly targeting downstream decision-making.
3. **Dual Role of Density Ratio**: Simultaneously corrects covariate shift and naturally characterizes "how far is too far"—a large variance in density ratio signifies insufficient evidence.
4. **Practicality-Oriented**: The user-defined success criteria are flexible and practical (mean, exceedance rate, etc.), closely aligned with biological design workflows.
5. **Combination of Theory and Practice**: Features finite-sample guarantees (Theorem 3.1) coupled with empirical validation in real-world RNA and protein design tasks.

## Limitations & Future Work

- Bonferroni correction can be overly conservative when the menu is very large; hierarchical or correlation-aware multiple testing corrections could be considered.
- The quality of density ratio estimation directly impacts the reliability of the guarantees, which remains challenging in high-dimensional sequence spaces.
- Only single-round design is considered, without extending to multi-round iterative designs (though the framework can in principle be applied round-by-round).
- Holding out annotated data reduces the volume of training data, requiring a trade-off between model training quality and correction capability.
- Auxiliary objectives such as design diversity are not considered—in practice, further screening among multiple successful configurations may be required.
- Conformal prediction methods as baselines were shown to be overly conservative (never selecting any configuration), indicating a need for research into more powerful alternatives.

## Related Work & Insights

- **Angelopoulos et al. (2023)**: Prediction-Powered Inference, the technical foundation of Ours.
- **Zhu, Brookes & Busia et al. (2024)**: Protein library design methodologies (the design algorithm in the first experiment of Ours).
- **Wheelock et al. (2022)**: Design label distribution modeling based on mixture density networks/forecast mixtures (the GMMForecasts baseline).
- **Sinai et al. (2020)**: AdaLead design algorithm.
- **Brookes et al. (2019)**: CbAS conditional sampling design algorithm.
- **Angelopoulos et al. (2021)**: Learn Then Test framework (multiple testing guarantees).

**Insights**: In ML-guided design, the quantification of predictive uncertainty should serve specific **downstream decisions** (selecting which algorithm), rather than solely pursuing generic uncertainty estimation. The PPI framework provides an elegant paradigm: "prediction + minor annotation → reliable decisions".

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Formulates the algorithm selection problem for ML-guided design with probabilistic guarantees for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two biological design tasks (protein and RNA) across both known and estimated density ratio scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definitions, precise methodology descriptions, with Figure 1 providing excellent intuitive guidance.
- Value: ⭐⭐⭐⭐⭐ — Addresses core practical pain points in the biological design field, showing broad potential for generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Multivariate Conformal Selection](multivariate_conformal_selection.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/computational_biology/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[ICLR 2026\] PepBenchmark: A Standardized Benchmark for Peptide Machine Learning](../../ICLR2026/computational_biology/pepbenchmark_a_standardized_benchmark_for_peptide_machine_learning.md)
- [\[ICML 2026\] Active Timepoint Selection for Learning Measure-Valued Trajectories](../../ICML2026/computational_biology/active_timepoint_selection_for_learning_measure-valued_trajectories.md)

</div>

<!-- RELATED:END -->
