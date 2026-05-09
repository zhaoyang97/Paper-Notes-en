---
title: >-
  [Paper Note] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts
description: >-
  [NeurIPS 2025][Optimization][Mixture of Experts] This paper presents the first systematic theoretical analysis of contaminated MoE models with softmax gating. It introduces the distinguishability condition, proving that MLE achieves the minimax-optimal parametric rate $\widetilde{\mathcal{O}}(n^{-1/2})$ when the condition holds, and revealing the fundamental mechanism behind significantly degraded estimation rates when the condition fails (i.e., when prompt and pretrained model knowledge overlap).
tags:
  - NeurIPS 2025
  - Optimization
  - Mixture of Experts
  - Parameter Estimation
  - Minimax Optimality
  - Distinguishability
  - Fine-Tuning Theory
date: 2026-05-08
content_hash: f69ea5332e09976a
---

# On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts

**Conference**: NeurIPS 2025
**arXiv**: [2505.18455](https://arxiv.org/abs/2505.18455)
**Code**: None
**Area**: Optimization
**Keywords**: Mixture of Experts, Softmax Gating, Parameter Estimation, Minimax Optimality, Distinguishability

## TL;DR

This paper presents the first minimax parameter estimation analysis for contaminated Mixture of Experts (MoE) models with softmax gating. It introduces the concept of "distinguishability" to characterize the relationship between a pretrained model and a prompt, proving that MLE achieves the parametric rate $\tilde{O}(n^{-1/2})$ when distinguishability holds, while the rate degrades significantly otherwise.

## Background & Motivation

Mixture of Experts (MoE) models dynamically assign input-dependent weights to different experts via a gating network, and have been widely adopted in NLP (DeepSeek-V3, Mixtral), vision (M3ViT), and multimodal domains. The **contaminated MoE** serves as a theoretical model for parameter-efficient fine-tuning (e.g., prefix tuning): a frozen pretrained expert is combined with a trainable prompt expert.

Despite their practical prevalence, the theoretical properties of contaminated MoE models remain largely unexplored. Prior work (Yan et al., Nguyen et al.) studied only the **input-independent gating** setting (i.e., fixed constant weights), which differs substantially from the softmax gating used in practice.

**Core Problem**: Under softmax gating, how does the relationship between the pretrained model and the prompt affect parameter estimation? What happens when the prompt learns knowledge that overlaps with the pretrained model?

## Method

### Overall Architecture

The object of study is the conditional density of a softmax-contaminated MoE:

$$p_{G_*}(y|x) = \frac{1}{1+\exp(\beta^{*\top}x + \tau^*)} f_0(y|h_0(x,\eta_0), \nu_0) + \frac{\exp(\beta^{*\top}x + \tau^*)}{1+\exp(\beta^{*\top}x + \tau^*)} f(y|h(x,\eta^*), \nu^*)$$

where the first term is the fixed pretrained model (known) and the second is the trainable prompt model (unknown). The unknown parameters $G_* = (\beta^*, \tau^*, \eta^*, \nu^*)$ comprise gating and prompt expert parameters, and MLE is used for estimation.

### Key Designs

1. **Distinguishability Condition (Definition 1)**: An analytic condition defining when the pretrained model $f_0$ is "distinguishable" from the prompt model $f$—requiring that linear combinations formed by $f_0$, $f$, and their partial derivatives are linearly independent in function space. Intuitively, this means the pretrained model and prompt possess **different areas of expertise**. **Proposition 1** provides a concise criterion: if $f_0$ does not belong to the Gaussian density family, distinguishability holds automatically; when $f_0$ is Gaussian and shares the same expert function $h_0 = h$ with the prompt, the condition is violated.

2. **Parameter Estimation under Distinguishability (Theorem 1)**: The MLE for all parameters (gating $\beta, \tau$ and prompt $\eta, \nu$) converges at the parametric rate $\tilde{O}(n^{-1/2})$. Compared to input-independent gating, the convergence rate of prompt parameter estimation under softmax gating **no longer depends on how fast the gating parameter tends to zero**, yielding a faster rate—demonstrating that softmax gating is statistically more efficient.

3. **Parameter Estimation under Non-Distinguishability (Theorem 3)**: When the prompt and pretrained model share overlapping knowledge (i.e., $(\eta^*, \nu^*) \to (\eta_0, \nu_0)$), the estimation rates degrade significantly:

    - Gating parameters: $\tilde{O}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-2})$
    - Expert parameters: $\tilde{O}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-1})$

   For example, if prompt parameters approach pretrained parameters at rate $O(n^{-1/8})$, the MLE convergence rates degrade to $O(n^{-1/4})$ for gating parameters and $O(n^{-3/8})$ for expert parameters.

4. **Strong Identifiability Condition (Definition 2)**: Required to derive exact rates in the non-distinguishable setting, demanding that three groups of functions formed by partial derivatives of the expert function $h$ are linearly independent. This holds for tanh, sigmoid, and GELU, but fails for ReLU due to vanishing second-order derivatives.

### Loss & Training

- **Minimax Lower Bounds (Theorems 2 & 4)**: In both settings, the MLE rates are minimax optimal (matching lower bounds up to logarithmic factors).
- **Technical Innovation**: The true parameter $G_*$ is allowed to vary with sample size $n$; uniform convergence rates are established rather than pointwise ones, which is both more practically relevant and essential for minimax analysis.
- MLE is computed using the EM algorithm with the BFGS optimizer.

## Key Experimental Results

### Main Results

**Distinguishable Setting** ($f_0$: Laplace, $f$: Gaussian, $d=8$)

| Parameter | Theoretical Rate | Empirical Rate |
|-----------|-----------------|----------------|
| $\beta$ | $O(n^{-1/2})$ | $O(n^{-0.45})$ |
| $\tau$ | $O(n^{-1/2})$ | $O(n^{-0.52})$ |
| $\eta$ | $O(n^{-1/2})$ | $O(n^{-0.50})$ |
| $\nu$ | $O(n^{-1/2})$ | $O(n^{-0.54})$ |

### Ablation Study

**Non-Distinguishable Setting** ($f_0, f$: Gaussian, $\eta^*$ approaches $\eta_0$ at rate $O(n^{-1/8})$)

| Parameter | Theoretical Rate | Empirical Rate |
|-----------|-----------------|----------------|
| $\exp(\tau)$ | $O(n^{-1/4})$ | $O(n^{-0.23})$ |
| $\beta$ | $O(n^{-3/8})$ | $O(n^{-0.37})$ |
| $\eta$ | $O(n^{-3/8})$ | $O(n^{-0.39})$ |
| $\nu$ | $O(n^{-3/8})$ | $O(n^{-0.35})$ |

### Key Findings

- Empirical rates closely match theoretical predictions, validating the theoretical analysis.
- In the distinguishable setting, all parameters converge at rates close to $n^{-1/2}$.
- In the non-distinguishable setting, gating rates slow to approximately $n^{-1/4}$ and expert parameter rates to approximately $n^{-3/8}$.
- **Softmax gating outperforms input-independent gating**: it eliminates the dependence of prompt parameter estimation on the gating parameter.

## Highlights & Insights

- The **distinguishability concept** is a novel theoretical tool that precisely characterizes the fundamental issue of "knowledge overlap between prompt and pretrained model" in fine-tuning.
- **Two practical guidelines**: (1) softmax gating should be preferred over input-independent gating for greater sample efficiency; (2) prompt models should be designed to possess expertise distinct from that of the pretrained model.
- The analytical framework allowing parameters to vary with sample size sets a new standard for MoE theory.
- The theoretical results provide a statistical explanation for the parameter efficiency of prefix tuning.

## Limitations & Future Work

- The current analysis is restricted to a single prompt expert; extending to multiple prompts is an important future direction.
- The prompt is limited to the Gaussian family; extending to more general distributions would broaden applicability.
- Only synthetic experiments are conducted; validation in real fine-tuning scenarios is absent.
- The effect of high dimensionality $d$ on estimation difficulty is not addressed.
- The initialization sensitivity of specific optimization algorithms is not considered.

## Related Work & Insights

- **Relation to Ho et al. (2022)**: Generalizes from algebraic independence in general MoE to distinguishability in contaminated MoE.
- **Relation to Nguyen et al. (2023, 2024)**: Extends from softmax MoE and input-independent contaminated MoE to softmax contaminated MoE.
- **Connection to prefix tuning**: Prompts should learn knowledge complementary to the pretrained model.
- **Inspiration**: The distinguishability condition may be relevant to model merging and expert routing strategy design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Distinguishability definition is novel; first theoretical analysis of softmax contaminated MoE)
- Experimental Thoroughness: ⭐⭐⭐ (Synthetic experiments validate theory, but real-world scenarios are absent)
- Writing Quality: ⭐⭐⭐⭐ (Mathematically rigorous and well-structured, though notation-heavy)
- Value: ⭐⭐⭐⭐ (Provides a solid theoretical foundation for MoE fine-tuning with valuable practical guidance)

**Area**: Optimization / Statistical Learning Theory
**Keywords**: Mixture of Experts, Parameter Estimation, Minimax Optimality, Distinguishability, Fine-Tuning Theory

## TL;DR

This paper presents the first systematic theoretical analysis of contaminated MoE models with softmax gating. It introduces the distinguishability condition, proving that MLE achieves the minimax-optimal parametric rate $\widetilde{\mathcal{O}}(n^{-1/2})$ when the condition holds, and revealing the fundamental mechanism behind significantly degraded estimation rates when the condition fails (i.e., when prompt and pretrained model knowledge overlap).

## Background & Motivation

MoE models dynamically assign input-dependent weights to multiple sub-models via a gating network, and are widely used in NLP, CV, and multimodal domains. In parameter-efficient fine-tuning (e.g., prefix tuning), the model can be interpreted as a **contaminated MoE**: a frozen pretrained expert combined with a trainable prompt expert.

**Existing Theoretical Gaps**:

(1) **Limitations of input-independent gating**: Prior work by Yan et al. and Nguyen et al. studied only contaminated MoE with input-independent gating, where the mixture weight is a constant $\lambda^*$—highly unrealistic in practice. Real systems employ softmax gating $\frac{\exp(\beta^{\top}x + \tau)}{1 + \exp(\beta^{\top}x + \tau)}$, where weights depend on the input.

(2) **Parameters varying with sample size**: Prior MoE theory assumes the true parameter $G^*$ is fixed regardless of sample size $n$, yielding only pointwise convergence rates. This paper allows $G^*$ to depend on $n$, establishing **uniform** convergence rates that are more practically relevant and indispensable for minimax analysis.

(3) **Core Question**: What happens to parameter estimation when the prompt learns knowledge that highly overlaps with the pretrained model? Does softmax gating offer advantages over constant gating?

**Key Insight**: The paper introduces "distinguishability" as a novel analytical concept, partitioning the problem into distinguishable and non-distinguishable settings and establishing upper bounds with matching minimax lower bounds in each case.

## Method

### Overall Architecture

Consider the softmax contaminated MoE model:

$$p_{G_*}(y|x) = \frac{1}{1+\exp(\beta^{*\top}x + \tau^*)} f_0(y|h_0(x,\eta_0), \nu_0) + \frac{\exp(\beta^{*\top}x + \tau^*)}{1+\exp(\beta^{*\top}x + \tau^*)} f(y|h(x,\eta^*), \nu^*)$$

where $f_0$ is the frozen pretrained model (known), $f$ is the Gaussian prompt model (unknown), and $G_* = (\beta^*, \tau^*, \eta^*, \nu^*)$ are the parameters to be estimated. Maximum likelihood estimation (MLE) is used to estimate $G_*$.

### Key Designs

1. **Distinguishability Condition (Definition 1)**:
    - Function: Defines when the pretrained model $f_0$ is "distinguishable" from the prompt model $f$.
    - Mechanism: Requires that no non-trivial linear combination of $f_0$, $f$, and their first-order partial derivatives equals zero. Intuitively, this rules out the scenario where the prompt learns knowledge identical to that of the pretrained model.
    - Key Property (Proposition 1): **If the pretrained model $f_0$ does not belong to the Gaussian density family, it is automatically distinguishable from the prompt model $f$.** Conversely, if $f_0$ is also Gaussian and $h_0 = h$ (same expert structure), distinguishability is violated.

2. **Analysis under the Distinguishable Setting (Theorems 1 & 2)**:
    - Function: Establishes the convergence rate of MLE and the minimax lower bound under the distinguishability condition.
    - Core Result: All parameter estimators ($\hat{\beta}_n, \hat{\tau}_n, \hat{\eta}_n, \hat{\nu}_n$) converge at the parametric rate $\widetilde{\mathcal{O}}(n^{-1/2})$, matching the minimax lower bound up to logarithmic factors—i.e., **minimax optimal**.
    - Key Finding: Under softmax gating, the estimation rate for prompt parameters $\eta^*, \nu^*$ is $\widetilde{\mathcal{O}}(n^{-1/2})$, improving upon the input-independent gating rate of $\widetilde{\mathcal{O}}(n^{-1/2} (\lambda^*)^{-1})$, which depends on how fast the gating parameter tends to zero.

3. **Analysis under the Non-Distinguishable Setting (Theorems 3 & 4)**:
    - Function: Characterizes estimation rates when prompt parameters $(\eta^*, \nu^*)$ approach pretrained parameters $(\eta_0, \nu_0)$.
    - Core Result: Estimation rates degrade significantly, depending on the distance $\|(\Delta\eta^*, \Delta\nu^*)\|$ between prompt and pretrained parameters.
    - Specific Rates: The estimation error of $\exp(\hat{\tau}_n)$ is $\widetilde{\mathcal{O}}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-2})$; the estimation error of $(\hat{\beta}_n, \hat{\eta}_n, \hat{\nu}_n)$ is $\widetilde{\mathcal{O}}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-1})$.
    - Example: If $(\eta^*, \nu^*)$ approaches $(\eta_0, \nu_0)$ at rate $\mathcal{O}(n^{-1/8})$, the gating parameter estimation rate degrades to $\mathcal{O}(n^{-1/4})$ and the expert parameter rate to $\mathcal{O}(n^{-3/8})$.

### Strong Identifiability Condition

To handle Taylor expansions in the non-distinguishable setting, Definition 2 requires the first- and second-order partial derivatives of the expert function $h$ to satisfy three groups of linear independence conditions. Examples satisfying this condition include $h(x,\eta) = \text{GELU}(\eta^\top x)$, $\text{sigmoid}(\eta^\top x)$, and $\tanh(\eta^\top x)$; $\text{ReLU}(\eta^\top x)$ does not satisfy it, as its second-order derivatives vanish almost everywhere.

## Key Experimental Results

### Main Results (Synthetic Data Validation)

| Setting | Parameter | Theoretical Rate | Empirical Rate |
|---------|-----------|-----------------|----------------|
| Distinguishable ($f_0$=Laplace) | $\hat{\beta}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.45})$ |
| Distinguishable | $\hat{\tau}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.52})$ |
| Distinguishable | $\hat{\eta}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.50})$ |
| Distinguishable | $\hat{\nu}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.54})$ |

### Ablation Study (Non-Distinguishable Setting)

| Varying Parameter | Rate of $\exp(\hat{\tau}_n)$ | Theoretical Prediction | Rate of $(\hat{\beta}_n, \hat{\eta}_n, \hat{\nu}_n)$ | Theoretical Prediction |
|-------------------|------------------------------|------------------------|------------------------------------------------------|------------------------|
| $\eta^*$ approaches $\eta_0$ at $n^{-1/8}$ | $\mathcal{O}(n^{-0.23})$ | $\mathcal{O}(n^{-1/4})$ | $\mathcal{O}(n^{-0.37\sim0.39})$ | $\mathcal{O}(n^{-3/8})$ |
| $\nu^*$ approaches $\nu_0$ at $n^{-1/8}$ | $\mathcal{O}(n^{-0.22})$ | $\mathcal{O}(n^{-1/4})$ | $\mathcal{O}(n^{-0.37\sim0.39})$ | $\mathcal{O}(n^{-3/8})$ |

Empirical results closely match theoretical predictions, validating the accuracy of the theoretical analysis.

### Key Findings

- **Softmax gating outperforms constant gating**: In the distinguishable setting, softmax gating eliminates the dependence of prompt parameter estimation on the gating parameter, improving the rate from $\widetilde{\mathcal{O}}(n^{-1/2}(\lambda^*)^{-1})$ to $\widetilde{\mathcal{O}}(n^{-1/2})$.
- **Knowledge overlap is the fundamental challenge**: When the prompt learns knowledge overlapping with the pretrained model, gating parameter estimation error is amplified by $\|(\Delta\eta^*, \Delta\nu^*)\|^{-2}$ and expert parameter error by $\|(\Delta\eta^*, \Delta\nu^*)\|^{-1}$.
- **Density estimation is unaffected**: Regardless of distinguishability, the Hellinger distance for model density estimation always maintains the parametric rate $\widetilde{\mathcal{O}}(n^{-1/2})$.

## Highlights & Insights

1. **Clear problem motivation**: Formalizing parameter-efficient fine-tuning (prefix tuning) as a contaminated MoE model provides a new statistical perspective for fine-tuning theory.
2. **Elegant distinguishability concept**: A concise analytical condition precisely captures the practical concern of "whether the prompt learns knowledge already possessed by the pretrained model."
3. **Theoretical completeness**: Upper and lower bounds match (minimax optimal), with complementary analyses in both settings.
4. **Practical guidance**: The theory directly suggests (1) using softmax gating rather than constant gating; (2) designing prompt models to learn expertise distinct from that of the pretrained model.

## Limitations & Future Work

- Only the single prompt model case is considered; analysis for multiple prompts (multi-task fine-tuning) is left for future work.
- The prompt is restricted to the Gaussian density family; extension to more general distribution families (e.g., mixture distributions) remains to be studied.
- Experiments are conducted only on synthetic data; empirical support from real LLM fine-tuning scenarios is lacking.
- The strong identifiability condition (Definition 2) does not hold for ReLU networks, limiting direct applicability to mainstream architectures.
- Elimination of logarithmic factors in the theoretical rates requires more refined analysis.

## Related Work & Insights

- Builds upon the MoE theory research lineage: Ho et al. (2022) on Gaussian MoE → Nguyen et al. (2023) on softmax gating MoE → Yan et al. (2025) on input-independent contaminated MoE → this paper on softmax contaminated MoE.
- Provides theoretical insights for understanding fine-tuning methods such as prompt tuning and LoRA: prompts should learn knowledge **complementary** to, rather than overlapping with, the pretrained model.
- The distinguishability concept may generalize to broader model combination and ensemble scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Near-Exponential Savings for Mean Estimation with Active Learning](near-exponential_savings_for_mean_estimation_with_active_learning.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)

<!-- RELATED:END -->
