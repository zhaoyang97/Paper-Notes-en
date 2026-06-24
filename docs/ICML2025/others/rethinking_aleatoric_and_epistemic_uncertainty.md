---
title: >-
  [Paper Note] Rethinking Aleatoric and Epistemic Uncertainty
description: >-
  [ICML 2025][aleatoric uncertainty] This paper points out that the aleatoric/epistemic uncertainty dichotomy in machine learning suffers from fundamental conceptual confusion. It proposes a decision-theoretic alternative framework that unifies predictive uncertainty, reducible/irreducible decomposition, predictive performance, and data dispersion within a coherent theoretical system, and reveals the limitations of BALD as an epistemic uncertainty estimator.
tags:
  - "ICML 2025"
  - "aleatoric uncertainty"
  - "epistemic uncertainty"
  - "decision theory"
  - "BALD"
  - "uncertainty decomposition"
date: 2026-05-08
content_hash: b5bfc524037ac8a9
---

# Rethinking Aleatoric and Epistemic Uncertainty

**Conference**: ICML 2025  
**arXiv**: [2412.20892](https://arxiv.org/abs/2412.20892)  
**Code**: None  
**Area**: Others  
**Keywords**: aleatoric uncertainty, epistemic uncertainty, decision theory, BALD, uncertainty decomposition

## TL;DR

This paper points out that the aleatoric/epistemic uncertainty dichotomy in machine learning suffers from fundamental conceptual confusion. It proposes a decision-theoretic alternative framework that unifies predictive uncertainty, reducible/irreducible decomposition, predictive performance, and data dispersion within a coherent theoretical system, and reveals the limitations of BALD as an epistemic uncertainty estimator.

## Background & Motivation

**Background**: In machine learning, researchers widely use aleatoric (associated with data noise) and epistemic (associated with the model's knowledge state) uncertainty to reason about the probabilistic predictions of models. Since Gal (2016) and Kendall & Gal (2017), this dichotomy has become the mainstream paradigm, utilized in various scenarios such as active learning, model selection, and OOD detection.

**Limitations of Prior Work**: Discussions regarding these two concepts in literature exhibit severe inconsistencies. Specifically:
   - The boundary between model prediction and the data generation process is repeatedly blurred (Amini et al., 2020; van Amersfoort et al., 2020)
   - Unreasonable assumptions are made regarding how uncertainty decomposes on unseen data
   - Misleading connections are established between uncertainty and predictive accuracy
   - The same concept (e.g., epistemic uncertainty) is defined using completely different mathematical quantities such as density, information gain, or variance

**Key Challenge**: The aleatoric-epistemic dichotomy can, by design, accommodate only two concepts. However, the quantities that researchers actually need to distinguish far exceed these two—including predictive uncertainty, reducible/irreducible decomposition of uncertainty, predictive performance, and statistical data dispersion. Forcing these distinct concepts under two labels leads to conceptual overloading, conflating quantities that should otherwise be distinguished.

**Goal**:
   - Reveal specific contradictions in the existing aleatoric-epistemic framework
   - Establish an alternative theoretical framework that can coherently express all related quantities
   - Analyze the estimation quality of BALD as an uncertainty metric

**Key Insight**: The authors start from decision theory, using a final action and its loss function as the starting point to formalize uncertainty via subjective expected loss, thereby avoiding direct reliance on specific metrics such as Shannon entropy or variance.

**Core Idea**: Replace the aleatoric-epistemic dichotomy with a decision-theoretic framework, using a loss-function-driven definition of uncertainty to coherently unify predictive uncertainty, reducibility decomposition, predictive performance, and data dispersion.

## Method

### Overall Architecture

This work is not a specific algorithm, but rather a **reconstruction of the theoretical framework**. The overall approach is as follows:

- **Input**: A decision problem $(a \in \mathcal{A}, z \in \mathcal{Z}, \ell(a,z))$ + training data $y_{1:n}$ + predictive model $p_n(z)$
- **Output**: Rigorous definitions and relationships of four concepts: predictive uncertainty, reducibility of uncertainty, predictive performance, and data dispersion
- **Key Intermediate Steps**:
  1. Define subjective uncertainty based on Bayes-optimal actions
  2. Derive reducible/irreducible decompositions by reasoning about the impact of new data
  3. Connect decomposition formulas in classical statistics and information theory
  4. Analyze the positioning of BALD under this framework

### Key Designs

1. **Decision-Theoretic Foundation**:

    - **Function**: Defines uncertainty starting from the action-loss pair $(a, \ell)$
    - **Mechanism**: Uncertainty is defined as the subjective expected loss of the Bayes-optimal action under the belief $p_n(z)$, i.e., $U_n = \min_{a \in \mathcal{A}} \mathbb{E}_{p_n(z)}[\ell(a, z)]$. This boils down to Shannon entropy when $\ell$ is log loss, and variance when $\ell$ is squared error
    - **Design Motivation**: Avoids a priori choices of entropy or variance as uncertainty metrics—letting the loss function determine what is "uncertain," thereby providing unified and task-relevant uncertainty definitions for different tasks

2. **Reducible vs Irreducible Decomposition**:

    - **Function**: Decomposes predictive uncertainty into two parts: one that can be eliminated as data grows, and one that cannot
    - **Mechanism**: Defines $U_\infty = \lim_{n \to \infty} U_n$ as the irreducible uncertainty when the amount of data approaches infinity; the reducible part is then $U_n - U_\infty$. By considering the expected change in uncertainty after acquiring new data, the degree of reducibility can be estimated without relying on the limit of infinite data
    - **Design Motivation**: This decomposition replaces the traditional simplified mapping of aleatoric = irreducible and epistemic = reducible, while clarifying that "reducibility" depends on the model class, update method, and data acquisition strategy, rather than being determined solely by data "noise"
    - **Differences from Prior Methods**: Traditional approaches directly equate $\mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$ with aleatoric/irreducible uncertainty, but this is merely a finite-sample estimator that can have substantial bias

3. **Performance vs Dispersion**:

    - **Function**: Rigorously distinguishes among (a) model uncertainty regarding its own beliefs, (b) model performance relative to external ground truth, and (c) the statistical dispersion of the data source itself
    - **Mechanism**: Predictive performance is measured using the expected loss under a reference distribution $p_{\text{eval}}(z)$, i.e., $R_n = \mathbb{E}_{p_{\text{eval}}(z)}[\ell(a_n^*, z)]$; data dispersion is a property of the data generation process $p_{\text{train}}$ itself. While related, these three are fundamentally distinct
    - **Design Motivation**: Under the classical aleatoric-epistemic framework, "aleatoric uncertainty" is simultaneously used to refer to three distinct quantities: $H[p_\infty(z)]$, $H[p_{\text{train}}]$, and $H[p_{\text{eval}}(z)]$—which in general are not equal

4. **Reinterpreting BALD**:

    - **Function**: Analyzes the quality of the BALD score $\text{EIG}_\theta = H[p_n(z)] - \mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$ as an estimator of reducible predictive uncertainty
    - **Mechanism**: BALD measures the information gain about parameters $\theta$ after observing $z$, rather than the long-term reducible predictive uncertainty. Because the mapping from parameters to predictions is typically non-invertible, information gain in parameter space does not necessarily translate into a reduction of uncertainty in prediction space
    - **Key Findings**: BALD should be better understood as an approximate measure of **short-term** parameter uncertainty reduction rather than long-term reducible predictive uncertainty. Although it is effective in practice, it remains suboptimal in prediction-oriented scenarios

### Theoretical Analysis

One of the core contributions of this work is pointing out the three main sources of confusion in the classical formula Eq.(1):

- **Confusion 1**: Equating $\mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$ (the expectation of conditional predictive entropy) with the irreducible uncertainty $H[p_\infty(z)]$—the former is merely an estimator of the latter under finite $n$ and can be highly inaccurate.
- **Confusion 2**: Conflating the model's subjective predictive belief with the actual data generation process—$p_n(z)$ is generally not equal to $p_{\text{train}}$ or $p_{\text{eval}}$.
- **Confusion 3**: Equating information gain in parameter space with uncertainty reduction in prediction space—the two differ when the mapping from parameters to predictions is non-invertible.

## Key Experimental Results

This work is a theoretical/conceptual study and does not include standard benchmark experiments. Instead, it supports its conclusions through conceptual analysis and mathematical demonstration. Below are the core conceptual comparisons and analytical results.

### Confusion Analysis of the Classical Aleatoric-Epistemic Framework

| Conceptual Label | Imputed Meaning | Actual Mathematical Quantity | Issues |
|---------|------------|----------------|---------|
| "Aleatoric" Meaning 1 | Inherent noise of data generation | $H[p_{\text{train}}(y_{1:n} \mid \pi)]$ | Training data distribution $\neq$ predictive distribution |
| "Aleatoric" Meaning 2 | Uncertainty of evaluation data | $H[p_{\text{eval}}(z)]$ | Evaluation distribution $\neq$ predictive distribution |
| "Aleatoric" Meaning 3 | Expectation of conditional predictive entropy | $\mathbb{E}_{p_n(\theta)}[H[p_n(z \mid \theta)]]$ | Merely a finite-sample estimator of $H[p_\infty(z)]$ |
| "Epistemic" Meaning 1 | Uncertainty of parameters | $H[p_n(\theta)]$ | Parameter-to-prediction mapping is non-invertible |
| "Epistemic" Meaning 2 | Parameter information gain (BALD) | $H[p_n(z)] - \mathbb{E}_{p_n(\theta)}[H[p_n(z \mid \theta)]]$ | Highly biased when estimating long-term reducible uncertainty |

### Decision-Theoretic Framework vs. Classical Framework Comparison

| Dimension | Classical Aleatoric-Epistemic Framework | Proposed Decision-Theoretic Framework |
|---------|---------------------------|-------------|
| Conceptual Capacity | Only 2 labels (aleatoric / epistemic) | 4+ independent concepts |
| Uncertainty Definition | Fixed to Shannon entropy or variance | Flexibly determined by loss function $\ell(a,z)$ |
| Reducible/Irreducible Decomposition | Approximated via Eq.(1) (can be inaccurate) | Rigorously defined through $U_n - U_\infty$ |
| Predictive Performance | Conflated with uncertainty | Independently defined as expected loss under reference distribution |
| Data Dispersion | Equated with aleatoric uncertainty | Independent concept, a property of $p_{\text{train}}$ |
| Model Assumptions | Requires stochastic parameters $\theta$ | Applicable to any learning method |
| Scope of Application | Bayesian models + information-theoretic quantities | Any ML method (including deep learning, ensembles, etc.) |

### Key Findings

- **BALD is not a direct measure of reducible uncertainty**: It measures parameter information gain rather than the long-term reducible portion of predictive uncertainty. The discrepancy between the two can be significant under finite data.
- **The equation "aleatoric = irreducible" does not hold**: Irreducible uncertainty $H[p_\infty(z)]$ depends not only on data "noise" but also on the model class and the update method. The same data source yields different irreducible uncertainties for different models.
- **Conditional predictive entropy $\mathbb{E}[H[p_n(z|\theta)]]$ is a biased estimator of $H[p_\infty(z)]$**: Under scarce data, the bias of this estimator can be massive, leading to flawed assessments of aleatoric uncertainty.
- **Origin of BALD's practical effectiveness**: Its utility in the real world stems from its approximation of *short-term* parameter uncertainty reduction. In iterative scenarios like active learning, short-term reduction is sufficiently informative on its own.

## Highlights & Insights

- **Precise diagnosis of conceptual overloading**: The authors systematically tease out how aleatoric/epistemic uncertainties are assigned multiple, incompatible definitions (illustrated clearly in Figure 2). They indicate that this is not merely a semantic issue but a substantial barrier affecting method design and evaluation. This "conceptual auditing" methodology is highly valuable in itself.

- **Loss-driven definition of uncertainty**: Defining uncertainty as the expected loss of the Bayes-optimal action elegantly unifies special cases like variance and Shannon entropy, and naturally binds uncertainty to practical tasks. It is highly inspiring for scenarios requiring task-specific uncertainty measures (e.g., medical diagnoses where false negatives cost more than false positives).

- **The triad of Predictive vs. Performance vs. Dispersion**: Distinctly separating subjective belief ($p_n(z)$), external evaluation ($p_{\text{eval}}$), and data generation ($p_{\text{train}}$) prevents the chronic mistake of conflating "model uncertainty" with "problem noise." This separation can be applied to any task involving model evaluation.

- **Repositioning of BALD**: Instead of dismissing the practical value of BALD, the paper pinpointed its theoretical limitation—unreliable as a long-term estimator but effective as a short-term approximation. This provides clear directions for improving active learning objective functions.

## Limitations & Future Work

- **Purely theoretical contribution without empirical validation**: The core of the paper is the reconstruction of a conceptual framework; it does not validate on real-world datasets whether this new framework leads to better method design or more accurate uncertainty estimation. Future experiments are needed to demonstrate the actual gains of the proposed framework over the old paradigm.
- **Computational difficulty of $U_\infty$**: Although $U_n - U_\infty$ is a rigorous definition of reducible uncertainty in theory, $U_\infty$ is usually incomputable in practice and still relies on finite-sample estimators. The paper lacks discussion on how to approximate $U_\infty$ practically.
- **Choice of loss function**: The framework requires pre-specifying a loss function $\ell$, which may not be clear in exploratory scenarios (e.g., representation learning, unsupervised learning). The applicability of this framework is compromised in these settings.
- **Vague alternative solutions for existing methods**: While criticizing existing metrics like BALD, the paper does not propose specific alternative algorithms or computation schemes, rendering limited guidance for practitioners.
- **Cost of model-agnostic formulation**: To remain applicable to arbitrary ML methods, the framework relinquishes the structural advantages of Bayesian inference (such as the interpretability of parameter posteriors), which may be suboptimal compared to tailored theories in specific scenarios.

## Related Work & Insights

- **vs. Kendall & Gal (2017)**: They formalized $total = aleatoric + epistemic$ as the information-theoretic decomposition Eq.(1). The current paper points out that this decomposition is only approximate under finite data and that its terms are assigned contradictory meanings. While the proposed framework excels in conceptual coherence, Kendall & Gal provides simple, practically computable metrics.

- **vs. Wimmer et al. (2023)**: Wimmer objects to Shannon entropy as a predictive uncertainty metric. This paper goes a step further, demonstrating that Shannon entropy is merely a special case of the decision-theoretic framework under log loss, thereby incorporating this criticism into a more general theory.

- **vs. Sale et al. (2023b, 2024b)**: Sale et al. proposed using class-wise variance to define uncertainty. The proposed framework can understand this as the Bayes risk under a specific loss function, providing a more unified perspective.

- **vs. Hofman et al. (2024b)**: Hofman et al. defined uncertainty based on proper scoring rules. This is aligned with the loss-driven approach in this paper but differs in focus; the two can be viewed as complementary.

- **Relationship with active learning research**: The analysis of BALD directly inspires improvements in active learning objectives—one should directly optimize the reduction of predictive uncertainty instead of parameter information gain (Bickford Smith et al., 2023, 2024).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Systematically re-evaluates the most fundamental uncertainty concepts in ML. Although the decision-theoretic perspective is not strictly new, its application in this context is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐ As a theoretical paper, it lacks standard benchmark experiments; however, the absence of empirical validation remains a notable drawback.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The arguments unfold progressively, conceptual diagrams (Figures 1 and 2) are clear, and every contradiction is backed by specific literature references and mathematical quantities.
- **Value**: ⭐⭐⭐⭐ Significantly contributes to understanding the conceptual foundations of uncertainty, though its practical impact remains to be validated by subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](../../CVPR2025/others/rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](../../ICML2026/others/on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)

</div>

<!-- RELATED:END -->
