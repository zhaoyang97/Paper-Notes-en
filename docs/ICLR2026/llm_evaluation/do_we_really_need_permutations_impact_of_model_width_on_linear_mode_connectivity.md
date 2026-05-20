---
title: >-
  [Paper Note] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity
description: >-
  [ICLR 2026][LLM Evaluation][linear mode connectivity] This paper empirically demonstrates that linear mode connectivity (LMC) between independently trained models can be achieved by simply increasing model width…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "linear mode connectivity"
  - "model merging"
  - "permutation symmetry"
  - "model width"
  - "loss landscape"
date: 2026-05-08
content_hash: aff79b65f91a7d64
---

# Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity

**Conference**: ICLR 2026
**arXiv**: [2510.08023](https://arxiv.org/abs/2510.08023)  
**Code**: None  
**Area**: LLM Evaluation
**Keywords**: linear mode connectivity, model merging, permutation symmetry, model width, loss landscape

## TL;DR

This paper empirically demonstrates that linear mode connectivity (LMC) between independently trained models can be achieved by simply increasing model width, without any parameter permutation. It further proposes *Layer-wise Exponentially Weighted Connectivity* (LEWC) to explain the underlying mechanism.

## Background & Motivation

**Linear Mode Connectivity (LMC)** refers to the existence of a low-loss linear path between the parameters of two independently trained models, i.e., linear interpolation of parameters does not cause a significant increase in loss. LMC is critical for understanding loss landscape geometry and for applications such as federated learning and model merging.

### Prior Knowledge

Entezari et al. (2022) hypothesized that for sufficiently wide models, there always exists a permutation $\pi$ such that LMC holds. Ainsworth et al. (2023) empirically validated this hypothesis via Weight Matching (WM), but found that very large width multipliers are required (e.g., 32× for ResNet-20, 4× for VGG-16). The prevailing belief was:
- The role of width is to enlarge the space of candidate permutations, increasing the probability of finding a good one.
- Without permutation, LMC does not hold.

### Core Finding of This Paper

Even **without any permutation**, simply averaging the weights of two independently trained models achieves test accuracy comparable to the original models, as long as the models are sufficiently wide. This overturns the conventional assumption that permutation is a necessary condition for LMC.

## Method

### Overall Architecture

This is an **analytical paper** rather than a methods paper. The core contribution is the introduction of the LEWC concept and a sufficient-condition analysis explaining why wide models can achieve LMC without permutation.

### Key Designs

#### 1. Layer-wise Exponentially Weighted Connectivity (LEWC)

**Definition (LEWC)**: Two model parameters $\boldsymbol{\theta}_a$ and $\boldsymbol{\theta}_b$ satisfy LEWC if and only if, for every layer $\ell$ and every $\lambda \in [0,1]$:

$$f_\ell(\mathbf{x}; \lambda\boldsymbol{\theta}_a + (1-\lambda)\boldsymbol{\theta}_b) = \lambda^\ell f_\ell(\mathbf{x}; \boldsymbol{\theta}_a) + (1-\lambda)^\ell f_\ell(\mathbf{x}; \boldsymbol{\theta}_b)$$

**Interpretation**: The $\ell$-th layer output of the merged model is an **exponentially decaying weighted sum** of the corresponding layer outputs of the two original models. At the final layer, this implies that the merged model's output is equivalent to a weighted ensemble of the two originals. Since scaling logits does not change predicted labels, LEWC directly implies LMC (no accuracy degradation).

#### 2. Sufficient Conditions for LEWC

**Condition 1: Weak Additivity of ReLU**

$$\sigma(\lambda \tilde{\mathbf{z}}_\ell^{(a)} + (1-\lambda)\tilde{\mathbf{z}}_\ell^{(b)}) = \lambda\sigma(\tilde{\mathbf{z}}_\ell^{(a)}) + (1-\lambda)\sigma(\tilde{\mathbf{z}}_\ell^{(b)})$$

That is, ReLU behaves linearly along the interpolation path between the pre-activations of the two models. This holds because:
- **Curse-of-dimensionality effect**: In high dimensions, the cosine similarity of ReLU outputs of two Gaussian vectors tends toward 0.93.
- **Low-rank weights leading to non-overlapping activations**: Wide models have low-rank weight matrices, so the dominant second-moment subspaces of the two models do not overlap.

**Condition 2: Reciprocal Orthogonality**

$$\mathbf{W}_\ell^{(b)} \mathbf{z}_{\ell-1}^{(a)} = 0 \quad \text{and} \quad \mathbf{W}_\ell^{(a)} \mathbf{z}_{\ell-1}^{(b)} = 0$$

That is, the weight matrix of one model applied to the activations of the other yields zero — the two models are "mutually non-interfering" in feature space.

**Theorem 5.3 (Core)**: For bias-free models, if both weak additivity and reciprocal orthogonality hold simultaneously, then LEWC holds.

#### 3. Essential Distinction from LLFC

The LLFC (Layer-wise Linear Feature Connectivity) proposed by Zhou et al. (2023) requires **commutativity**, i.e., the weights of the two models are sufficiently close. LEWC, by contrast, requires **reciprocal orthogonality**, meaning the weights are highly dissimilar and orthogonal. The two conditions are mutually exclusive:
- LLFC explains LMC after permutation (WM aligns weights → weights become close).
- LEWC explains LMC without permutation (wide models → orthogonal weights).

#### 4. The Critical Role of Low-Rank Structure

Increased width → relatively lower rank of weight matrices → lower effective dimensionality of activation vectors → non-overlapping activation spaces of the two models → weak additivity and reciprocal orthogonality hold → LEWC holds → LMC holds.

### Loss & Training

Standard training (SGD + weight decay 0.003) is used; no new training method is proposed. A key observation is softmax temperature calibration: since LEWC causes exponential decay in logit norms, inverse temperature scaling is needed so that the calibrated loss barrier approaches zero.

## Key Experimental Results

### Main Results

**Table 1: Barrier values with/without permutation ($\lambda=1/2$)**

| Network | Dataset | No-perm Acc barrier | No-perm Loss barrier | WM-perm Acc barrier | WM-perm Loss barrier |
|---------|---------|:-:|:-:|:-:|:-:|
| MLP (16×) | MNIST | 0.519% | 0.013 | -0.027% | -0.003 |
| VGG-11 (16×) | CIFAR-10 | 1.308% | 0.066 | 7.000% | 0.177 |
| ResNet-20 (32×) | CIFAR-10 | 2.694% | 0.087 | 5.135% | 0.173 |

Sufficiently wide models achieve very small barriers without any permutation. In some cases, applying WM permutation actually yields a *larger* barrier (e.g., VGG-11 and ResNet-20).

**Random permutation experiment**: Applying random permutations before merging sufficiently wide models still preserves accuracy — confirming that permutation becomes irrelevant once the model is wide enough.

### Ablation Study

**Effect of weak weight decay ($10^{-4}$)**

| Condition | VGG-11 LEWC | VGG-11 Weak Additivity | VGG-11 Reciprocal Orthogonality |
|-----------|:-:|:-:|:-:|
| Standard WD (0.003) | ✓ (high cosine similarity) | ✓ | ✓ (low ratio) |
| Weak WD ($10^{-4}$) | ✗ (low cosine similarity) | ✗ | ✗ (high ratio) |

Weak weight decay → high-rank weights → both sufficient conditions for LEWC fail → LMC fails. This confirms that low-rank structure is the key driver of LEWC.

### Key Findings

1. **Width monotonically improves merging performance**: Increasing width causes merged model accuracy to rise monotonically until it matches the original models.
2. **Temperature calibration is necessary**: LEWC causes exponential decay in logit norms; softmax calibration is required for the loss barrier to approach zero.
3. **LEWC ≠ flatness**: Random perturbation experiments show that loss landscape flatness alone cannot explain LMC — LEWC is an independent mechanism.
4. A width multiplier of approximately 2× or more (e.g., 16× for VGG-11, 32× for ResNet-20) is sufficient to achieve permutation-free LMC.

## Highlights & Insights

1. **Disruptive finding**: Overturns the widely held assumption that permutation is a necessary condition for LMC, revealing that width itself matters more than the permutation search space.
2. **LEWC concept**: Elegantly interprets the merged model as an exponentially weighted ensemble of the original models, bridging model merging and ensemble learning.
3. **Reciprocal orthogonality vs. commutativity**: Clearly distinguishes two fundamentally different LMC mechanisms, deepening the understanding of neural network loss landscapes.
4. **Causal chain from low rank to LMC**: Low-rank weights → non-overlapping activations → weak additivity + reciprocal orthogonality → LEWC → LMC.

## Limitations & Future Work

1. Experiments are limited to simple datasets (MNIST, CIFAR-10), since permutation-free LMC requires large width multipliers.
2. Only standard architectures (MLP, VGG-11, ResNet-20) are considered; modern architectures such as Transformers are not validated.
3. As an analytical work, no practical model merging or federated learning method is proposed.
4. LEWC requires batch normalization recalibration and temperature scaling, increasing practical complexity.
5. The theoretical analysis provides sufficient conditions rather than necessary ones.
6. On more complex datasets (e.g., CIFAR-100, ImageNet), the width requirements for LMC may be prohibitively large.

## Related Work & Insights

- **Ainsworth et al. (2023)**: Weight Matching method; the primary comparative framework for this paper.
- **Zhou et al. (2023)**: Proposed LLFC; forms a complementary explanation to LEWC in this paper.
- **Entezari et al. (2022)**: Proposed the permutation invariance hypothesis; this paper effectively revises it.
- **Implications for federated learning**: If client models are sufficiently wide, simple FedAvg may be adequate without complex alignment strategies.
- **Practical implications for model merging**: Training wider models with appropriate weight decay may be the simplest effective merging strategy.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Technical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★★ |
| Value | ★★★☆☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](../../ACL2026/llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](../../ACL2026/llm_evaluation/do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)

</div>

<!-- RELATED:END -->
