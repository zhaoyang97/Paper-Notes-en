---
title: >-
  [Paper Note] When Machine Learning Gets Personal: Evaluating Prediction and Explanation
description: >-
  [ICLR 2026][Interpretability][Benefit of Personalization] This paper proposes a unified framework to quantify the impact of model personalization on prediction accuracy and explanation quality. It proves that the two can be decoupled (prediction remains unchanged while explanation improves or degrades) and derives finite-sample lower bounds for hypothesis testing error probab
tags:
  - ICLR 2026
  - Interpretability
  - Benefit of Personalization
date: 2026-05-08
content_hash: ed43b1cf709b1dca
---
# When Machine Learning Gets Personal: Evaluating Prediction and Explanation

**Conference**: ICLR 2026  
**arXiv**: [2502.02786](https://arxiv.org/abs/2502.02786)  
**Code**: None (UCSB)  
**Area**: Interpretability  
**Keywords**: Personalized models, Interpretability, Benefit of Personalization, Hypothesis testing, Finite-sample lower bounds, Sufficiency, Incompleteness

## TL;DR
This paper proposes a unified framework to quantify the impact of model personalization on prediction accuracy and explanation quality. It proves that the two can be decoupled (prediction remains unchanged while explanation improves or degrades) and derives finite-sample lower bounds for hypothesis testing error probabilities based on dataset statistics, revealing that personalization effects are statistically untestable in many practical scenarios.

## Background & Motivation

**Background**: In high-risk domains such as healthcare, ML models are increasingly personalized by incorporating individual attributes (gender, race, etc.). User expectations imply that providing personal information leads to more accurate diagnoses and clearer explanations. However, this assumption has rarely been rigorously verified.

**Limitations of Prior Work**: (1) The impact of personalization on prediction and explanation may be inconsistent—improved prediction does not necessarily imply improved explanation; (2) Sensitive attributes may amplify bias (e.g., racial bias in health algorithms discovered by Obermeyer et al.); (3) Existing theoretical frameworks (e.g., BoP by Monteiro Paes et al., 2022) are limited to binary costs for binary classification, failing to cover regression or explanation quality.

**Key Challenge**: The benefit of personalization must be verified at the group-level (ensuring no group is harmed), but the statistical validity of such verification in finite samples is fundamentally limited—the more groups (more personal attributes), the fewer samples per group, making tests increasingly unreliable.

**Goal**: (1) How are the impacts of personalization on prediction and explanation related or decoupled? (2) When can personalization effects be reliably tested given a dataset? (3) What group sample size is required to detect an effect of a given magnitude?

**Key Insight**: Generalize the BoP framework to arbitrary cost functions (including continuous regression loss and explanation quality metrics), derive minimax hypothesis testing lower bounds, and provide actionable guidelines for experimental design.

**Core Idea**: Prediction and explanation gains from personalization can be decoupled, and in many practical scenarios, certain gains are statistically untestable—fundamentally limiting the utility of personalization.

## Method

### Overall Architecture
This paper does not train new models but establishes a theoretical framework to "evaluate whether personalization is worthwhile." The approach proceeds in four steps. First, it abstracts prediction and explanation quality into a unified cost framework: comparing a generic model $h_0: \mathcal{X} \to \mathcal{Y}$ (no group attributes) with a personalized model $h_p: \mathcal{X} \times \mathcal{S} \to \mathcal{Y}$ using group-level cost differences (G-BoP) and the minimum gain across all groups (BoP $\gamma$). Second, it uses constructive theorems to prove that "prediction gain" and "explanation gain" can improve or degrade independently. Third, it addresses practice: if personalization is beneficial, can it be reliably detected in finite samples? This is formulated as a hypothesis test to derive finite-sample error lower bounds. Fourth, it applies these bounds to specific tasks, revealing fundamental differences in testability between classification and regression.

Since the paper focuses on theory and statistical inference (theorem construction + minimax bounds + hypothesis testing), no multi-stage data pipeline or architecture diagram is provided; the four key designs follow the logic of "unified metric → decoupling theorems → testing bounds → task differences."

### Key Designs

**1. Unified Cost Function System: Measuring Prediction and Explanation in one Framework**

To compare prediction and explanation quality, they must share a cost abstraction. The authors define four types of metrics as "expected costs conditioned on groups," providing forms for both classification and regression. Prediction uses Loss (error rate $\Pr(h(\tilde{\mathbf{X}}) \neq \mathbf{Y} \mid \mathbf{S}=s)$ for classification, MSE $\mathbb{E}[\|h(\tilde{\mathbf{X}}) - \mathbf{Y}\|^2 \mid \mathbf{S}=s]$ for regression) or negative metrics ($-\text{AUC}$, $-R^2$). Explanation uses sufficiency (whether feature subset $J$ reproduces prediction, $\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_J) \mid \mathbf{S}=s)$) and incompleteness (prediction should change after removing $J$, $-\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_{\backslash J}) \mid \mathbf{S}=s)$).

| Type | Classification | Regression |
|------|------|------|
| Loss | $\Pr(h(\tilde{\mathbf{X}}) \neq \mathbf{Y} \mid \mathbf{S}=s)$ | $\mathbb{E}[\|h(\tilde{\mathbf{X}}) - \mathbf{Y}\|^2 \mid \mathbf{S}=s]$ |
| Metric | $-\text{AUC}$ | $-R^2$ |
| Sufficiency | $\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_J) \mid \mathbf{S}=s)$ | $\mathbb{E}[\|h(\tilde{\mathbf{X}}) - h(\tilde{\mathbf{X}}_J)\|^2 \mid \mathbf{S}=s]$ |
| Incompleteness | $-\Pr(h(\tilde{\mathbf{X}}) \neq h(\tilde{\mathbf{X}}_{\backslash J}) \mid \mathbf{S}=s)$ | $-\mathbb{E}[\|h(\tilde{\mathbf{X}}) - h(\tilde{\mathbf{X}}_{\backslash J})\|^2 \mid \mathbf{S}=s]$ |

**2. Prediction-Explanation Decoupling Theorem: Evidence of Independent Variation**

The intuition that "better prediction implies better explanation" is dismantled via constructive theorems. Theorem 4.1 constructs a distribution where the Bayes optimal classifier has zero prediction gain $\gamma_P = 0$ but positive explanation gain $\gamma_X > 0$—when personal features correlate highly with existing ones, the explainer shifts importance to the more direct feature, clarifying the explanation without changing the prediction. Theorem 4.2 constructs cases where $\gamma_P = 0$ but $\gamma_X < 0$: additional features diffuse importance, making explanations fuzzier. Theorem 4.4 notes that only in restricted settings (like additive linear models) can prediction BoP be inferred from explanation BoP.

**3. Hypothesis Testing and Minimax Error Lower Bounds: Judging Reliability**

The authors formalize the detection of personalization benefits as a one-sided test: null hypothesis $H_0: \gamma \leq 0$ (at least one group does not benefit) vs. alternative $H_1: \gamma \geq \epsilon$ (all groups benefit by at least $\epsilon$). Theorem 5.1 provides a minimax error probability lower bound for any test:

$$\min_\Psi \max P_e \geq \frac{1}{2}\left(1 - \frac{1}{2\sqrt{d}}\left[\frac{1}{d}\sum_{j=1}^d \left(\mathbb{E}_{p^\epsilon}\left[\frac{p^\epsilon(\mathbf{B})}{p(\mathbf{B})}\right]\right)^{m_j} - 1\right]^{1/2}\right),$$

where $d = 2^k$ is the number of groups defined by $k$ binary attributes, and $m_j$ is the sample size of the $j$-th group. As $d$ grows exponentially with attributes or $m_j$ decreases, the lower bound approaches random guessing ($1/2$).

**4. Testability Differences between Classification and Regression**

Classification individual BoP is categorical $\{-1, 0, 1\}$, leading to identical testing lower bounds for prediction and explanation. In regression, individual BoP is continuous (e.g., Gaussian or Laplace), and the lower bounds for prediction and explanation depend on their respective variances. This leads to scenarios where prediction gains are testable on a dataset while explanation gains are obscured by noise.

## Key Experimental Results

### MIMIC-III Healthcare Scenario

**Personalized Attributes**: Age × Race ({18-45, 45+} × {White, NonWhite}), 4 groups.

| Cost Type | Groups | G-BoP (Prediction) | G-BoP (Explanation-Sufficiency) |
|----------|------|-------------|------------------------|
| Classification | All groups | Mix of Pos/Neg | Direction potentially differs from prediction |
| Regression | All groups | Mix of Pos/Neg | Direction potentially differs from prediction |

### Detectability Analysis

| Setting | $\epsilon = 0.002$ | Conclusion |
|------|------------------|------|
| Classification (N=Hundreds) | $P_e \geq 40\%$ | **Untestable** |
| Regression Sufficiency | $P_e \geq 40\%$ | **Untestable** |
| Regression Prediction | Dep. on $\sigma^2$ | Potentially Testable |

### Key Findings
- Personalization effects for prediction and explanation decouple in real-world data—some groups see better predictions but worse explanations.
- At typical healthcare dataset sizes (N=100-10,000), error lower bounds for testing personalization effects are very high even with only 1-2 personal attributes.
- Testability is equivalent for prediction and explanation in classification but can decouple in regression.

## Highlights & Insights
- **Novelty**: First formal proof that prediction and explanation gains can decouple, challenging the intuition that good models necessarily yield good explanations.
- **Value**: Demonstrates that certain personalization effects are **untestable in principle** due to information-theoretic limits, rather than algorithmic flaws. This serves as a cautionary note for "personalized medicine."
- **Function**: Corollary 5.3 provides a practical tool for practitioners to determine required sample sizes and detectable effect magnitudes.
- **Experimental Thoroughness**: Generalized framework applicable to arbitrary cost functions and distribution families beyond binary classification.

## Limitations & Future Work
- Assumes IID data and random group assignment; real-world data may have selection bias.
- Current explanation methods (Integrated Gradients/DeepLIFT/Shapley) are post-hoc; applicability to inherently interpretable models needs verification.
- The partial converse (Theorem 4.4) for additive models remains an open question for non-linear models.
- Combinatorial explosion ($d=2^k$) is the primary constraint on utility when multiple attributes are crossed.

## Related Work & Insights
- **vs. Monteiro Paes et al. (2022)**: Extended the BoP framework from binary costs to arbitrary costs and from binary classification to regression/explanation quality.
- **vs. Balagopalan et al. (2022) / Dai et al. (2022)**: These identified group disparities in explanation quality but did not study the causal effect of personalization itself.
- **vs. Fairness Literature**: Does not require equal performance but investigates a weaker condition: that no group is systematically harmed by personalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formal proofs of decoupling and testability bounds are first-of-kind.
- Experimental Thoroughness: ⭐⭐⭐⭐ Primarily theoretical with validation on MIMIC-III across classification and regression.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent alternation between theorems, examples, and intuition.
- Value: ⭐⭐⭐⭐⭐ Significant impact on personalized ML and XAI; negative results on testability are highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](../../ICML2026/interpretability/position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICLR 2026\] FAME: Formal Abstract Minimal Explanation for Neural Networks](fame_formal_abstract_minimal_explanation_for_neural_networks.md)
- [\[ICML 2025\] Rethinking Explainable Machine Learning as Applied Statistics](../../ICML2025/interpretability/rethinking_explainable_machine_learning_as_applied_statistics.md)
- [\[ICLR 2026\] Evaluating SAE Interpretability Without Generating Explanations](evaluating_sae_interpretability_without_generating_explanations.md)

</div>

<!-- RELATED:END -->
