---
title: >-
  [Paper Note] When Machine Learning Gets Personal: Evaluating Prediction and Explanation
description: >-
  [ICLR 2026][Personalized Models] This paper proposes a unified framework to quantify the impact of model personalization on both prediction accuracy and explanation quality. It proves that these two dimensions can be decoupled (explanations may improve or degrade while predictions remain unchanged), derives finite-sample lower bounds on hypothesis testing error probabilities based on dataset statistics, and reveals that in many practical settings the benefit of personalization is statistically untestable in principle.
tags:
  - ICLR 2026
  - Personalized Models
  - Explainability
  - Benefit of Personalization
  - Hypothesis Testing
  - Finite-Sample Lower Bounds
  - Sufficiency
  - Incomprehensiveness
date: 2026-05-08
content_hash: 04fe5a544dcb3bac
---

# When Machine Learning Gets Personal: Evaluating Prediction and Explanation

**Conference**: ICLR 2026
**arXiv**: [2502.02786](https://arxiv.org/abs/2502.02786)
**Code**: None (UCSB)
**Area**: Interpretability
**Keywords**: Personalized Models, Explainability, Benefit of Personalization, Hypothesis Testing, Finite-Sample Lower Bounds, Sufficiency, Incomprehensiveness

## TL;DR
This paper proposes a unified framework to quantify the impact of model personalization on both prediction accuracy and explanation quality. It proves that these two dimensions can be decoupled (explanations may improve or degrade while predictions remain unchanged), derives finite-sample lower bounds on hypothesis testing error probabilities based on dataset statistics, and reveals that in many practical settings the benefit of personalization is statistically untestable in principle.

## Background & Motivation

**State of the Field**: In high-stakes domains such as healthcare, ML models are increasingly personalized by incorporating individual attributes (e.g., gender, race). Users may reasonably expect that sharing personal information will yield more accurate diagnoses and clearer explanations. Yet this assumption has rarely been rigorously validated.

**Limitations of Prior Work**: (1) The effects of personalization on prediction and explanation may be inconsistent — improved predictions do not necessarily imply improved explanations; (2) sensitive attributes may amplify biases (e.g., the racially biased health algorithm identified by Obermeyer et al.); (3) existing theoretical frameworks (the BoP framework of Monteiro Paes et al., 2022) are restricted to binary classification with binary costs, and do not cover regression or explanation quality.

**Root Cause**: The benefit of personalization must be validated at the group level (no group should be harmed), yet the statistical validity of such validation is fundamentally limited by finite sample sizes — the more personal attributes (and thus groups) there are, the fewer samples per group and the less reliable the tests become.

**Paper Goals**: (1) How are the effects of personalization on prediction and explanation related or separable? (2) Under a given dataset, when can the benefit of personalization be reliably tested and when can it not? (3) How large must group sample sizes be to detect an effect of a given magnitude?

**Starting Point**: The BoP framework is generalized to arbitrary cost functions (including continuous regression losses and explanation quality metrics), minimax hypothesis testing lower bounds are derived, and actionable guidelines for experimental design are provided.

**Core Idea**: The personalization gains for prediction and explanation can be decoupled, and in many practical scenarios certain gains are statistically untestable — a finding that fundamentally limits the practical utility of personalization.

## Method

### Overall Architecture
The framework builds upon the Benefit of Personalization (BoP) paradigm. A generic model $h_0: \mathcal{X} \to \mathcal{Y}$ does not use group attributes, while a personalized model $h_p: \mathcal{X} \times \mathcal{S} \to \mathcal{Y}$ does. The benefit of personalization is quantified via group-level cost differences (G-BoP) and the minimum group-wise gain (BoP $\gamma$).

### Key Designs

1. **Prediction–Explanation Separation Theorems**:

    - **Theorem 4.1**: There exist data distributions under which the Bayes-optimal classifier satisfies $\gamma_P = 0$ (no prediction gain) yet $\gamma_X > 0$ (positive explanation gain).
    - Intuition: Adding a personal feature that is highly correlated with existing features does not alter predictions, but the explainer may reassign importance to this more direct feature.
    - **Theorem 4.2**: There exist distributions under which $\gamma_P = 0$ but $\gamma_X < 0$ (explanation degrades).
    - Intuition: Additional features may diffuse importance assignments, rendering explanations less interpretable.
    - **Theorem 4.3**: Personalization can produce opposing explanation effects across different groups.
    - **Theorem 4.4** (Partial Converse): Under additive linear models, if the BoP for both sufficiency and incomprehensiveness is zero, then the prediction BoP is also zero.

2. **Hypothesis Testing Validity Analysis**:

    - Null hypothesis $H_0: \gamma \leq 0$ (at least one group does not benefit).
    - Alternative hypothesis $H_1: \gamma \geq \epsilon$ (all groups benefit by at least $\epsilon$).
    - Decision rule: reject $H_0$ if $\hat{\gamma} \geq \epsilon$.
    - **Theorem 5.1**: Derives a minimax lower bound on the error probability of any test:

$$\min_\Psi \max P_e \geq \frac{1}{2}\left(1 - \frac{1}{2\sqrt{d}}\left[\frac{1}{d}\sum_{j=1}^d \left(\mathbb{E}_{p^\epsilon}\left[\frac{p^\epsilon(\mathbf{B})}{p(\mathbf{B})}\right]\right)^{m_j} - 1\right]^{1/2}\right)$$

    - where $d = 2^k$ groups ($k$ binary attributes) and $m_j$ is the sample size of group $j$.
    - **Corollary 5.3**: Provides the minimum group size $m_{\min}$ required to guarantee $P_e \leq v$.

3. **Classification vs. Regression**:

    - Classification: individual BoP is categorical ($\{-1, 0, 1\}$); the lower bounds for prediction and explanation are **identical**.
    - Regression: individual BoP is continuous (Gaussian/Laplace); the lower bounds for prediction and explanation **can differ** — one may be testable while the other is not.

### Cost Function Taxonomy

| Type | Classification | Regression |
|------|---------------|------------|
| Loss | $\Pr(h(\tilde{\mathbf{X}}) \neq \mathbf{Y} \mid \mathbf{S}=s)$ | $\mathbb{E}[\|h(\tilde{\mathbf{X}}) - \mathbf{Y}\|^2 \mid \mathbf{S}=s]$ |
| Evaluation Metric | $-\text{AUC}$ | $-R^2$ |
| Sufficiency | $\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_J) \mid \mathbf{S}=s)$ | $\mathbb{E}[\|h(\tilde{\mathbf{X}}) - h(\tilde{\mathbf{X}}_J)\|^2 \mid \mathbf{S}=s]$ |
| Incomprehensiveness | $-\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_{\backslash J}) \mid \mathbf{S}=s)$ | $-\mathbb{E}[\|h(\tilde{\mathbf{X}}) - h(\tilde{\mathbf{X}}_{\backslash J})\|^2 \mid \mathbf{S}=s]$ |

## Key Experimental Results

### MIMIC-III Clinical Setting

**Personalization attributes**: Age × Race ({18–45, 45+} × {White, NonWhite}), yielding 4 groups.

| Cost Type | Group | G-BoP (Prediction) | G-BoP (Explanation – Sufficiency) |
|-----------|-------|-------------------|----------------------------------|
| Classification | Per group | Some positive, some negative | Direction may differ from prediction |
| Regression | Per group | Some positive, some negative | Direction may differ from prediction |

### Testability Analysis

| Setting | $\epsilon = 0.002$ | Conclusion |
|---------|------------------|------------|
| Classification (N = hundreds) | $P_e \geq 40\%$ | **Untestable** |
| Regression – Sufficiency | $P_e \geq 40\%$ | **Untestable** |
| Regression – Prediction | Depends on $\sigma^2$ | Potentially testable |

### Key Findings
- Prediction and explanation personalization effects indeed decouple on real data — certain groups exhibit improved predictions but degraded explanations.
- For typical clinical dataset sizes ($N = 100$–$10{,}000$), even with only one or two personal attributes, the error lower bounds for testing personalization benefit are already high.
- In classification tasks, the testability of prediction and explanation gains is equivalent; in regression tasks the two can diverge.

## Highlights & Insights
- **Conceptual Contribution**: This work provides the first formal proof that prediction gain and explanation gain can be decoupled — challenging the intuition that a better model necessarily yields better explanations, with foundational implications for the XAI community.
- **Value of Negative Results**: The paper proves that certain personalization effects are **untestable in principle** — not a limitation addressable by better algorithms, but a fundamental information-theoretic constraint. This serves as an important cautionary note against overclaiming in personalized medicine.
- **Experimental Design Tool**: Corollary 5.3 directly answers "how many samples are needed," "how large an effect can be detected," and "how many attributes can be used" — providing a practical tool for practitioners.
- **Generality**: The framework extends to arbitrary cost functions and distribution families, without restriction to binary classification.

## Limitations & Future Work
- The theoretical analysis assumes i.i.d. data and fully randomized group assignment; selection bias may be present in practice.
- The explanation methods employed (Integrated Gradients, DeepLIFT, Shapley values) are post-hoc; applicability to inherently interpretable models remains to be examined.
- Whether the partial converse under additive models (Theorem 4.4) holds for nonlinear models is an open question.
- The combinatorial explosion of groups as attributes increase ($d = 2^k$ grows exponentially) is the primary practical constraint.

## Related Work & Insights
- **vs. Monteiro Paes et al. (2022)**: Extends the BoP framework from binary costs to arbitrary costs, and from binary classification to regression and explanation quality.
- **vs. Balagopalan et al. (2022) / Dai et al. (2022)**: Those works identify group disparities in explanation quality but do not study the causal effect of personalization itself.
- **vs. Fairness Literature**: Rather than requiring equal performance, this work examines a weaker condition — that no group is systematically harmed by the personalized system.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The formal proof of prediction–explanation separation and the testability lower bounds are both first-of-their-kind contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Theory-centric with MIMIC-III empirical validation covering both classification and regression.
- Writing Quality: ⭐⭐⭐⭐⭐ — The alternating presentation of theorems, examples, and intuitions is excellent, and the figures are well designed.
- Value: ⭐⭐⭐⭐⭐ — Far-reaching implications for personalized ML and XAI; the negative results (untestability) carry equally significant practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICCV 2025\] Minerva: Evaluating Complex Video Reasoning](../../ICCV2025/interpretability/minerva_evaluating_complex_video_reasoning.md)

<!-- RELATED:END -->
