---
title: >-
  [Paper Note] Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection
description: >-
  [AAAI 2026][Model Compression][Weight Pruning] This paper presents the first systematic investigation of the interaction between weight pruning and coreset selection…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Weight Pruning"
  - "Coreset Selection"
  - "Synergistic Effects"
  - "Training Acceleration"
  - "State Preservation Mechanism"
date: 2026-05-08
content_hash: fe21faaa21b6a1cf
---

# Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection

**Conference**: AAAI 2026  
**arXiv**: [2511.09901](https://arxiv.org/abs/2511.09901)  
**Authors**: Weilin Wan, Fan Yi, Weizhong Zhang, Quan Zhou, Cheng Jin (Fudan University)
**Code**: Not available  
**Area**: Model Compression  
**Keywords**: Weight Pruning, Coreset Selection, Synergistic Effects, Training Acceleration, State Preservation Mechanism

## TL;DR

This paper presents the first systematic investigation of the interaction between weight pruning and coreset selection, proposing the SWaST mechanism to alternately perform both operations and establish synergistic effects, while introducing a state preservation mechanism to address the "dual loss" problem, achieving up to 17.83% accuracy improvement under 10%–90% FLOPs reduction.

## Background & Motivation

### State of the Field
Modern deep neural networks rely on massive model parameters and training samples, incurring enormous computational overhead. Weight pruning and coreset selection are two mainstream paradigms for improving training efficiency: the former removes redundant weights, while the latter identifies the most representative training samples.

### Limitations of Prior Work
- Weight pruning and coreset selection have **always been studied independently**, with their interaction overlooked.
- In classical machine learning (e.g., SVM), the synergistic effects of joint feature and sample selection have been well studied, with theoretical tools (KKT conditions) guaranteeing safe removal correctness.
- In deep learning, the highly non-convex training objective precludes analogous theoretical guarantees, leading to the separate development of the two paradigms.
- Performing pruning and sample selection simultaneously may trigger the **"dual loss"** phenomenon—critical weights and their supporting samples are erroneously removed together, causing nearly irreversible performance degradation.

### Root Cause
The paper aims to explore and exploit the synergistic effects between weight pruning and coreset selection in deep learning: redundant samples (especially noisy ones) cause weights to be over-tuned, increasing pruning difficulty; redundant weights tend to overfit noisy data, undermining the effectiveness of coreset selection. This bidirectional interference suggests the potential for joint optimization.

## Method

### Key Design 1: Alternating Optimization Mechanism SWaST

SWaST (Simultaneous Weight and Sample Tailoring) consists of a warm-up phase and an alternating optimization phase:

1. **Warm-up Phase**: Train for $\mathcal{K}$ epochs on the full dataset to establish a good initialization.
2. **Alternating Optimization Phase**: Execute coreset selection every $\mathcal{R}$ epochs to identify the most representative samples; subsequently train on the selected coreset and perform online pruning.

Two variants are distinguished by the aggressiveness of the pruning strategy:

- **SWaST-trim**: Prunes only fully connected layers, retaining most parameters to ensure training stability; efficiency gains primarily come from coreset selection.
- **SWaST-cut**: Applies global pruning across the entire network for larger efficiency gains, but may cause training instability (the "dual loss" problem).

The difficulty of coreset selection can be measured by:

$$\mathcal{I}(\mathcal{D}, \hat{\mathcal{D}}) = \sup_{\boldsymbol{\theta}} \frac{|\mathcal{L}(\boldsymbol{\theta}) - \hat{\mathcal{L}}(\boldsymbol{\theta})|}{\mathcal{L}(\boldsymbol{\theta})}$$

Experiments show that $\mathcal{I}(\mathcal{D}, \hat{\mathcal{D}})$ grows exponentially with polynomial degree (i.e., model parameter dimensionality), indicating that pruning to reduce model size can significantly lower coreset selection difficulty.

### Key Design 2: State Preservation Mechanism

To address the "dual loss" problem, a two-stage state preservation mechanism is designed:

**Stage 1 — State Recording** (executed every $\mathcal{R}$ epochs): After coreset update, model states are captured via forward propagation:

$$\tilde{\mathcal{D}} = \{(\mathbf{x}_i, \mathbf{z}_i) : \mathbf{z}_i = f_{\boldsymbol{\theta}_{\text{pre}}}(\mathbf{x}_i), \mathbf{x}_i \in \mathcal{X}\}$$

**Stage 2 — State-Constrained Training**: State consistency is enforced through a composite loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{(\mathbf{x}_i, \mathbf{z}_i) \in \tilde{\mathcal{D}}} \text{KL}(\sigma(\mathbf{z}_i) \| \sigma(f_{\boldsymbol{\theta}}(\mathbf{x}_i)))$$

where $\lambda=0.1$ balances the primary learning objective with state consistency, and $\sigma$ denotes the softmax function. The KL divergence term detects distributional shifts caused by erroneous pruning and corrects them in subsequent steps, thereby stabilizing the joint optimization process.

### Key Design 3: Theoretical Analysis of Synergistic Effects

A polynomial interpolation task is employed for transparent analysis, revealing two key observations:

1. **Redundant Samples Impede Pruning**: Excessive samples (especially noisy ones) cause polynomial coefficients to be over-tuned to fit all samples, making it difficult for magnitude-based pruning to identify irrelevant weights.
2. **Redundant Weights Impede Coreset Selection**: Under standard metrics, coreset selection difficulty grows exponentially with the number of redundant model weights, as $\mathcal{L}(\boldsymbol{\theta})$ rapidly approaches zero due to overfitting on noisy data.

## Key Experimental Results

### Experiment 1: SWaST-trim under Varying Pruning Ratios and Coreset Sizes (ResNet-18, CIFAR-10/100)

| Dataset | Coreset Size | Pruning 90% | Pruning 50% | Pruning 10% | Coreset Only |
|--------|-----------|----------|----------|----------|---------|
| CIFAR-10 | 10% | 92.58 (+0.38) | 92.40 (+0.20) | 92.29 (+0.09) | 92.20 |
| CIFAR-10 | 5% | 90.16 (+0.49) | 89.92 (+0.25) | 89.77 (+0.10) | 89.67 |
| CIFAR-10 | 1% | 78.60 (+2.97) | 77.14 (+1.61) | 76.01 (+0.48) | 75.53 |
| CIFAR-100 | 10% | 71.94 (+0.97) | 71.44 (+0.47) | 71.20 (+0.23) | 70.97 |
| CIFAR-100 | 5% | 66.19 (+1.40) | 65.56 (+0.77) | 65.09 (+0.30) | 64.79 |
| CIFAR-100 | 1% | 38.38 (+1.96) | 37.58 (+1.16) | 36.88 (+0.46) | 36.42 |

**Pattern**: The smaller the coreset, the more significant the gain from SWaST (largest improvement at 1% coreset), validating the enhancement effect of pruning on coreset selection.

### Experiment 2: SWaST-cut Global Pruning (ResNet-101, Multiple Datasets)

| Dataset | Coreset Size | Pruning 90% | Pruning 50% | Pruning 30% | Coreset Only |
|--------|-----------|----------|----------|----------|---------|
| CIFAR-10 | 1% | 64.57 (+14.85) | 66.92 (+17.20) | **67.55 (+17.83)** | 49.72 |
| CIFAR-100 | 5% | 59.27 (+5.39) | 62.87 (+8.99) | 62.97 (+9.09) | 53.88 |
| TinyImageNet | 5% | 38.65 (+3.55) | 42.93 (+7.83) | 41.94 (+6.84) | 35.10 |
| ImageNet-1K | 10% | 37.55 (+5.83) | 38.92 (+7.20) | 39.19 (+7.47) | 31.72 |
| ImageNet-1K | 5% | 31.63 (+1.35) | 32.71 (+2.43) | 34.34 (+4.06) | 30.28 |

**Key Findings**: On ResNet-101 + CIFAR-10 + 1% coreset, SWaST-cut achieves a maximum accuracy improvement of **17.83%** at 30% pruning ratio; moderate pruning ratios (30%–50%) are generally optimal, while aggressive pruning (90%) occasionally degrades performance.

### Experiment 3: Coreset Selection Improves Pruning Effectiveness

On ResNet-18/CIFAR-10, SWaST-cut achieves 93.15% accuracy at 90% pruning ratio, a 3.33% improvement over the pruning-only baseline. Under the same FLOPs budget, SWaST leads by up to 4.43% in accuracy (at relative FLOPs = 0.01).

### Experiment 4: Noise Resistance and Coreset Quality

- SWaST-cut reduces the proportion of noisy samples in the final coreset by **10.62%**.
- Higher pruning ratios lead to more pronounced reductions in overfitting (measured by test loss minus validation loss).
- Post-pruning models exhibit higher loss on noisy samples, indicating that pruning effectively prevents memorization of incorrect patterns.

## Key Findings

1. **Synergistic Effects Are Real**: Weight pruning and coreset selection yield significant bidirectional gains in deep learning—pruning reduces coreset selection difficulty, while coreset selection enhances pruning effectiveness.
2. **Smaller Coresets Benefit Most**: The smaller the coreset, the more substantial the accuracy improvement from pruning, as smaller coresets face greater selection challenges and overfitting risks.
3. **Dual Loss Phenomenon**: Simultaneously and aggressively removing both weights and samples may cause critical weights and their supporting samples to be erroneously discarded together, leading to irreversible degradation—a problem specific to deep learning.
4. **State Preservation Is Effective**: The KL divergence constraint detects erroneous pruning and corrects it in subsequent steps, significantly stabilizing joint optimization.

## Highlights & Insights

- **Pioneer Interaction Analysis**: This is the first work to systematically explore the interaction between weight pruning and coreset selection in deep learning, drawing on joint selection theory from classical ML.
- **Transparent Theoretical Insight**: A polynomial interpolation task provides interpretable analysis, revealing the bidirectional interference mechanism between redundant samples and weights.
- **Discovery and Mitigation of Dual Loss**: The paper identifies a joint optimization pitfall unique to deep learning and effectively mitigates it with a state preservation mechanism.
- **General Framework Design**: SWaST is compatible with any online pruning algorithm and coreset selection method, offering strong flexibility.
- **Significant Empirical Gains**: Up to 17.83% accuracy improvement with simultaneous 10%–90% FLOPs reduction.

## Limitations & Future Work

- **Limited to Unstructured Pruning**: Only unstructured methods such as RigL are employed; synergistic effects with structured pruning (e.g., channel/filter pruning) remain unverified.
- **Limited Coreset Selection Methods**: Only three methods—GradMatch, Moderate, and EL2N—are used, without coverage of more recent approaches.
- **State Preservation Overhead**: Additional forward passes are required to record logits and compute KL divergence, increasing training cost.
- **Lack of Theoretical Guarantees**: Synergistic effects are validated only empirically; no theoretical analysis framework for deep learning has been established.
- **Hyperparameter Sensitivity**: Systematic guidance for selecting $\lambda$, $\mathcal{R}$, and $\mathcal{K}$ is lacking.
- **Insufficient Large-Scale Validation**: ImageNet-1K experiments are limited in scale; validation on larger models (e.g., ViT, GPT) is absent.

## Related Work & Insights

- **Classical ML Joint Selection** (Shibagaki et al. 2016; Zhang et al. 2017): Leverages KKT conditions to safely remove features and samples in convex models such as SVMs; this paper extends the idea to non-convex deep learning.
- **GradMatch** (Killamsetty et al. 2021): Gradient-matching-based coreset selection, used as the default selection method in CIFAR experiments.
- **RigL** (Evci et al. 2020): Dynamic sparse training method that updates sparse topology during training, used as the default pruning algorithm.
- **EL2N** (Paul et al. 2021): Sample importance metric based on error L2-norm, used in ImageNet-1K and noise experiments.
- **Moderate** (Xia et al. 2022): Distance-based coreset selection, used in TinyImageNet experiments.
- **Knowledge Distillation**: The state preservation mechanism shares similarities with self-distillation, but serves a different purpose—stabilizing joint optimization rather than knowledge transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic joint exploration of synergistic effects between pruning and coreset selection; the discovery of the "dual loss" problem is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers CIFAR-10/100, TinyImageNet, and ImageNet-1K across multiple architectures and settings with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — The narrative arc from classical ML to deep learning is clear; polynomial experiments provide intuitive understanding.
- **Value**: ⭐⭐⭐⭐ — Reveals a neglected joint optimization direction in training efficiency; practically useful, though large-model validation is lacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](../../ICML2026/model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[AAAI 2026\] Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs](dont_start_over_a_cost-effective_framework_for_migrating_personalized_prompts_be.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)
- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)

</div>

<!-- RELATED:END -->
