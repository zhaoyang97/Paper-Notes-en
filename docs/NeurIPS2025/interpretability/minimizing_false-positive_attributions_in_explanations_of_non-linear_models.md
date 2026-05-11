---
title: >-
  [Paper Note] Minimizing False-Positive Attributions in Explanations of Non-Linear Models
description: >-
  [NeurIPS 2025][Interpretability][XAI] This paper proposes PatternLocal to address false-positive attributions caused by suppressor variables in XAI explanations of non-linear models. The method converts local discriminat…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "XAI"
  - "suppressor variables"
  - "local explanations"
  - "generative explanation"
  - "LIME"
date: 2026-05-08
content_hash: 360347bf0da4c028
---

# Minimizing False-Positive Attributions in Explanations of Non-Linear Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.11210](https://arxiv.org/abs/2505.11210)
**Code**: [GitHub](https://github.com/gjoelbye/PatternLocal)
**Area**: Explainable AI / Interpretability
**Keywords**: XAI, suppressor variables, local explanations, generative explanation, LIME

## TL;DR
This paper proposes PatternLocal to address false-positive attributions caused by suppressor variables in XAI explanations of non-linear models. The method converts local discriminative surrogate weights into a generative representation, and significantly reduces false-positive feature attributions on three datasets: the XAI-TRIS benchmark, MRI artificial lesions, and EEG motor imagery.

## Background & Motivation
**Background**: XAI methods such as LIME, KernelSHAP, and gradient-based approaches are widely used to explain the decision-making process of black-box models, and are particularly critical in high-stakes domains such as healthcare and finance.

**Limitations of Prior Work**: Existing studies have demonstrated that mainstream XAI methods including LIME and SHAP assign importance weights to **suppressor variables** — variables that influence model predictions but have no direct statistical dependence on the target variable. For example, when a model predicts epilepsy by exploiting noise probes in irrelevant brain regions, XAI methods may erroneously flag those regions as important.

**Key Challenge**: The Pattern method for linear models can distinguish discriminative weights from generative activation patterns to eliminate suppressor variable effects, but this method and its deep network extensions (PatternNet/PatternAttribution) perform poorly in non-linear settings and cannot effectively handle suppressor variables in local non-linear explanations.

**Goal**: Extend suppressor variable suppression from globally linear models to **local** explanations of non-linear models, addressing instance-level false-positive attribution.

**Key Insight**: Local linear surrogate weights are first obtained via LIME, KernelSHAP, or gradient methods, and then converted from discriminative weights to a generative representation through a data-driven forward model.

**Core Idea**: Building on local linear surrogates produced by methods such as LIME, the discriminative weights are converted into generative activation patterns (Patterns) via kernel-weighted regression, thereby naturally eliminating the influence of suppressor variables.

## Method

### Overall Architecture
PatternLocal is a **two-stage**, model-agnostic XAI method:
1. **Stage 1 (Local Linear Surrogate)**: A local linear surrogate is constructed for the target sample $\mathbf{x}_\star$ using LIME, KernelSHAP, or gradient methods, yielding a discriminative weight vector $\mathbf{w}$.
2. **Stage 2 (Generative Conversion)**: Using training data within the neighborhood of $\mathbf{x}_\star$, the surrogate prediction $\tilde{y} = \mathbf{w}^\top \mathbf{h}(\mathbf{x})$ is regressed onto the simplified input space $\mathbf{h}(\mathbf{x})$, producing a generative activation pattern $\mathbf{a}$.

### Key Designs
1. **Suppressor Variable Elimination (Core Principle)**: The central idea of the Pattern method is that a discriminative model weight $\mathbf{W}$ corresponds to a unique forward model $\mathbf{A} = \Sigma_\mathbf{X} \mathbf{W} \Sigma_\mathbf{M}^{-1}$. The activation patterns of the forward model retain only features statistically related to the target, naturally eliminating suppressor variables. PatternLocal generalizes this principle to the **local** non-linear setting.
2. **Kernel-Weighted Local Regression**: The formal objective of PatternLocal is:
    $\mathbf{a} = \arg\min_\mathbf{u} \mathbb{E}_{\mathbf{x} \sim \mathbb{P}_\mathcal{X}} \left[ \Pi_{\mathbf{x}'_\star}(\mathbf{h}(\mathbf{x})) \| \mathbf{h}(\mathbf{x}) - \mathbf{u} \tilde{y} \|_2^2 \right] + \lambda Q(\mathbf{u})$
   where $\Pi$ is a local kernel function that enforces locality of the explanation, and $Q$ is a regularization term.
3. **Closed-Form Solution (Ridge Regression Form)**: When $Q(\mathbf{u}) = \|\mathbf{u}\|_2^2$, a closed-form solution exists:
    $\mathbf{a}_{\ell_2} = \frac{\text{Cov}_\Pi[\mathbf{h}(\mathbf{x}), \tilde{y}]}{\text{Var}_\Pi[\tilde{y}] + \lambda}$
   That is, the kernel-weighted covariance between simplified features and the surrogate response, divided by the regularized variance.
4. **Input Simplification Schemes**: Three input simplifications $\mathbf{h}$ are supported: (a) identity mapping (raw features); (b) superpixel representation; (c) low-rank approximation. Different schemes are applicable to different scenarios.

### Loss & Training
- Regularization supports either L1 (Lasso) or L2 (Ridge); L1 induces sparsity while L2 admits a closed-form solution.
- Hyperparameters are tuned via Bayesian optimization (TPE algorithm) on a validation set using the EMD metric.
- The local kernel function $\Pi$ ensures that explanations reflect only the behavior in the neighborhood of $\mathbf{x}_\star$.

## Key Experimental Results

### Main Results — XAI-TRIS Benchmark (MLP, Identity Mapping)

| Method | LIN-WHITE EMD↓ | XOR-CORR EMD↓ | RIGID-CORR EMD↓ | XOR-CORR IME↓ |
|------|----------------|----------------|------------------|----------------|
| **PatternLocal** | **Best** | **Significantly best** | **Comparable to filter methods** | **Significantly best** |
| LIME | Second | High | High | High |
| KernelSHAP | Second | High | High | High |
| Gradient | Moderate | High | High | High |
| IntegratedGrad | Moderate | Moderate | Moderate | Moderate |
| Sobel (filter) | Lower | Moderate | **Lower** | Moderate |
| Laplace (filter) | Lower | Moderate | **Lower** | Moderate |

### Toy Example Validation (XOR Problem, Mean Attribution Magnitude for Suppressor Variable x3)

| Method | Mean Attribution to x3 |
|------|---------------|
| LIME | ~0.18 (erroneous attribution) |
| KernelSHAP | ~0.17 (erroneous attribution) |
| Gradient | ~0.19 (erroneous attribution) |
| **PatternLocal** | **~0.01 (near zero)** |

### EEG Motor Imagery Dataset (Physiological Plausibility Evaluation)

| Method | Dipole Fit Goodness (mean±std) |
|------|----------------------|
| **PatternLocal** | **0.756 ± 0.090** |
| Raw instances | 0.738 ± 0.013 |
| LIME | 0.604 ± 0.013 |

### Key Findings
- PatternLocal significantly outperforms all other XAI methods in XOR and RIGID scenarios (non-linear problems with suppressor variables).
- In the RIGID-CORR scenario, Sobel/Laplace filters perform well due to the rigid edge structure of XAI-TRIS images, but this advantage does not generalize to complex backgrounds such as MRI.
- On the MRI artificial lesion dataset, PatternLocal explanations align with true lesion locations more accurately than LIME, while filter methods fail due to the absence of clear edges.
- In EEG experiments, PatternLocal explanations are physiologically plausible in both time-frequency and source analysis domains, localizing feature patterns to the expected motor cortex regions (contralateral activity).

## Highlights & Insights
- **Theoretical Elegance**: The unified perspective of discriminative-to-generative conversion is compelling; the toy example provides a mathematical proof that PatternLocal exactly eliminates the suppressor variable ($a_3=0$).
- **Model-Agnostic**: The method can serve as a plug-and-play post-processing module for any XAI method that produces local linear surrogates (LIME/SHAP/gradient families).
- **Closed-Form Availability**: The Ridge variant admits an analytic solution, enabling computational efficiency.
- **Cross-Modal Validation**: The method is validated on both images (synthetic and MRI) and EEG time-series signals, demonstrating generality beyond the visual domain.
- **Experimental Rigor**: Bayesian hyperparameter optimization ensures fair comparison; multiple input simplifications, regularization schemes, and model combinations are evaluated.

## Limitations & Future Work
- **Requires Access to Training Data**: Sufficient training samples within the neighborhood of the explained instance are required; this limits applicability in privacy-sensitive or data-scarce settings.
- **Spatial Alignment Assumption**: The method assumes a degree of consistency or alignment across the input space, which may not hold for natural images or user-generated content.
- **RIGID-CORR Scenario**: The method does not outperform simple edge filters in scenes with rigid edges, indicating that it does not dominate across all structural types.
- **Residual Misattribution**: As with other XAI methods, saliency maps should be treated as indicative rather than definitive.
- **Large-Scale Modern Architectures Untested**: Experiments focus on MLP/CNN/ShallowNet; whether more complex models such as Transformers benefit equally remains to be verified.

## Related Work & Insights
- **vs. Pattern/PatternNet**: The original Pattern method applies only to globally linear models; PatternNet extends it to deep networks but performs poorly on the non-linear suppressor variable benchmark. PatternLocal overcomes this limitation through local surrogates combined with kernel-weighted regression.
- **vs. LIME/KernelSHAP**: LIME and SHAP are inherently discriminative local linear surrogates. PatternLocal adds a generative conversion step on top of them, eliminating suppressor variables with virtually no additional assumptions.
- **vs. Filter Methods (Sobel/Laplace)**: Filters happen to perform well on synthetic images with clear edges but are unsuitable for complex backgrounds such as MRI and carry no theoretical guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of extending the Pattern method from globally linear to locally non-linear settings is concise and compelling, and the theoretical analysis via the toy example is convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets across different modalities (synthetic images / MRI / EEG), with extensive hyperparameter search and ablation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivation from linear to non-linear settings is clear, the toy example is intuitive, and the overall structure is rigorous.
- Value: ⭐⭐⭐⭐ Improving the reliability of XAI explanations has practical value in high-stakes domains such as healthcare, and the model-agnostic nature facilitates easy integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)
- [\[NeurIPS 2025\] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)

</div>

<!-- RELATED:END -->
