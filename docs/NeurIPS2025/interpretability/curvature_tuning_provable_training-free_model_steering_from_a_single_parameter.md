---
title: >-
  [Paper Note] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter
description: >-
  [NeurIPS 2025][Interpretability][curvature tuning] This paper proposes Curvature Tuning (CT), which provably modulates the curvature of a model's decision boundary by injecting a single hyperparameter $\beta$ into the ac…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "curvature tuning"
  - "activation functions"
  - "spline theory"
  - "parameter-efficient fine-tuning"
  - "decision boundary"
date: 2026-05-08
content_hash: 3c95dfffe97db509
---

# Curvature Tuning: Provable Training-free Model Steering From a Single Parameter

**Conference**: NeurIPS 2025
**arXiv**: [2502.07783](https://arxiv.org/abs/2502.07783)  
**Code**: [GitHub](https://github.com/Leon-Leyang/curvature-tuning)  
**Area**: Interpretability
**Keywords**: curvature tuning, activation functions, spline theory, parameter-efficient fine-tuning, decision boundary

## TL;DR

This paper proposes Curvature Tuning (CT), which provably modulates the curvature of a model's decision boundary by injecting a single hyperparameter $\beta$ into the activation function. CT improves generalization and robustness without modifying weights, and as a fine-tuning method requires far fewer parameters than LoRA rank 1.

## Background & Motivation

Existing parameter-efficient fine-tuning (PEFT) methods (e.g., LoRA, Adapter) focus exclusively on **weight adaptation**—introducing or updating weight parameters. However, they generally lack interpretability and rely on heuristic hyperparameter choices (e.g., LoRA's rank, placement, and initialization) without theoretical guidance. A critically overlooked component is the **activation function**, which governs the nonlinearity and expressive capacity of a model.

The core observation of this paper: viewing deep networks as affine spline operators, existing fine-tuning methods adjust the slopes and breakpoints of the spline, whereas **tuning the activation function alters the underlying geometric structure** of the model (i.e., the curvature of the decision boundary).

## Method

### Overall Architecture

CT is grounded in the spline interpretation of deep networks: ReLU networks are equivalent to max-affine spline operators. CT smooths decision boundaries by replacing hard region selection (one-hot) with soft probabilistic selection, and is provided in two variants:
- **S-CT (Steering CT)**: introduces a single global hyperparameter $\beta$ with no training required.
- **T-CT (Trainable CT)**: assigns an independent trainable pair $(\beta, c)$ to each neuron.

### Key Designs

**CT Unit (CTU) activation function**:

$$\varphi_{\beta,c}(\mathbf{x}) = c \cdot \sigma\left(\frac{\beta \mathbf{x}}{1-\beta}\right) \cdot \mathbf{x} + (1-c) \cdot \ln\left[1 + \exp\left(\frac{\mathbf{x}}{1-\beta}\right)\right] \cdot (1-\beta)$$

where $\beta \in [0,1]$ controls curvature, $c \in [0,1]$ is a mixing coefficient, and $\sigma(\cdot)$ denotes the sigmoid. This is a convex combination of reparameterized SiLU and Softplus:

$$\text{SiLU}(\mathbf{x}) = \sigma(\eta \mathbf{x}) \cdot \mathbf{x}, \quad \eta = \frac{\beta}{1-\beta}$$

$$\text{Softplus}(\mathbf{x}) = \frac{1}{\gamma} \cdot \ln[1 + \exp(\gamma \mathbf{x})], \quad \gamma = \frac{1}{1-\beta}$$

CTU naturally subsumes SiLU ($c=1$), Softplus ($c=0$), and the GELU approximation ($c=1, \beta=0.64$).

**$\beta$-VQ inference framework**: replaces the hard selection of max-affine splines with entropy-regularized soft selection, whose optimal solution takes the softmax form:

$$\mathbf{t}_r^\beta = \frac{\exp\left(\frac{\beta(\langle \mathbf{A}_{r,\cdot}, \mathbf{x}\rangle + \mathbf{b}_r)}{1-\beta}\right)}{\sum_{i=1}^R \exp\left(\frac{\beta(\langle \mathbf{A}_{i,\cdot}, \mathbf{x}\rangle + \mathbf{b}_i)}{1-\beta}\right)}$$

### Loss & Training

S-CT involves no training loss (only a grid search over $\beta$). T-CT trains the per-layer $(\beta, c)$ parameters using standard cross-entropy loss while freezing all original weights.

**Theoretical guarantee** (Theorem 3.1): For a ReLU network $f$, replacing ReLU with CTU (with fixed $\beta \in [0,1)$) is equivalent to projecting $f$ onto a **smooth function space**, keeping gradients and curvature bounded while achieving higher local expressive capacity for the same weight parameters $\mathbf{W}$.

## Key Experimental Results

### Main Results

**Downstream transfer accuracy** (ImageNet-pretrained ResNet, average accuracy % over 12 datasets):

| Method | Trainable Params | ResNet-18 | ResNet-50 |
|--------|-----------------|-----------|-----------|
| Frozen (LP) | 0 | 73.96 | 76.24 |
| S-CT | 1 | 75.34 | 76.92 |
| LoRA (r=1) | 35K–79K | 73.64 | 78.68 |
| **T-CT** | 4K–45K | **78.26** | **81.31** |

T-CT improves over LP by 8.59%/8.34% (ResNet-50/152) and over LoRA (r=1) by 4.64%/1.70%, while using only 11%–59% of LoRA's parameter count.

**Adversarial robustness** (RobustBench $\ell_\infty$ attack):

| Model | Dataset | Frozen | S-CT | Optimal $\beta$ |
|-------|---------|--------|------|----------------|
| ResNet-18 | CIFAR-10 | 11.17% | 14.93% | 0.90 |
| ResNet-18 | CIFAR-100 | 4.47% | 6.90% | 0.92 |
| ResNet-18 | ImageNet | 0.00% | 7.00% | 0.89 |

S-CT significantly improves robustness without any adversarial training.

### Ablation Study

- The optimal $\beta$ for S-CT is close to 1 (ResNet-50: 0.94, ResNet-152: 0.96), allowing the search range to be narrowed.
- $c=0.5$ (CTU) outperforms pure SiLU ($c=1$) and pure Softplus ($c=0$).
- The $\beta$ values learned by T-CT follow a U-shaped distribution (concentrated near 0 and 1); $c$ values exhibit a similar pattern, with an effective mean close to the manually chosen S-CT value.
- Full comparison against LoRA rank 1/2/4: T-CT still outperforms all LoRA ranks on ResNet-18/50.

### Key Findings

1. Modulating activation function curvature and modulating weights (LoRA) are **orthogonal and complementary** dimensions of model improvement.
2. As $\beta \to 0$, the network degenerates to a linear map (zero curvature); as $\beta \to 1$, it recovers the original ReLU; intermediate values provide an optimal balance.
3. The robustness improvement is an **implicit bias** of CT, requiring no adversarial training objective.

## Highlights & Insights

1. **Theory-driven PEFT**: provides provable guarantees grounded in spline theory, in contrast to the heuristic designs of existing PEFT methods.
2. **Extreme parameter efficiency**: S-CT uses only 1 hyperparameter; T-CT uses fewer than 60% of the parameters of LoRA (r=1).
3. CT is complementary to, rather than a replacement for, LoRA: CT modulates the function space while LoRA modulates the feature space.
4. The CTU design is compatible with multiple activation functions, including ReLU, SiLU, GELU, and Softplus.

## Limitations & Future Work

- Theoretical guarantees hold strictly for piecewise-affine networks (ReLU/MaxPool); only partial guarantees apply to GELU/SiLU in Transformers.
- S-CT requires a grid search over $\beta$ (range 0.7–1.0, step 0.01), which, while low-cost, is not fully automatic.
- Robustness improvements are limited in certain settings (e.g., near-zero $\ell_2$ improvement on ResNet-152 with CIFAR).

## Related Work & Insights

- Unlike Srinivas et al., who learn low-curvature activation functions during pretraining, CT is the first to position activation function tuning as a PEFT paradigm.
- Building on the deep network spline interpretation of Balestriero & Baraniuk, the paper provides rigorous theoretical tools for nonlinearity modulation.
- An open question for future work: can CT be combined with LoRA for further gains? Preliminary experiments in the paper already suggest complementary effects.

## Rating

- ⭐ Novelty: 5/5 — A fundamentally new perspective on PEFT with tight theory–experiment integration.
- ⭐ Experimental Thoroughness: 5/5 — 6 models × 12 datasets × dual validation on generalization and robustness.
- ⭐ Writing Quality: 4/5 — Mathematical derivations are thorough, though notation is occasionally heavy.
- ⭐ Value: 5/5 — Opens a new paradigm of activation function tuning as PEFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](../../AAAI2026/interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](../../CVPR2026/interpretability/cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)

</div>

<!-- RELATED:END -->
