---
title: >-
  [Paper Note] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs
description: >-
  [ACL 2025 (Oral)][Model Compression][Knowledge Distillation] Demonstrates that naive Top-K sparse knowledge distillation yields biased estimation and proposes Random Sampling Knowledge Distillation (RSKD) based on importance sampling. RSKD provides unbiased gradient estimation while requiring the storage of only extremely sparse logits. The training overhead is increased by less than 10% compared to cross-entropy, while maintaining performance on par with full knowledge disti…
tags:
  - "ACL 2025 (Oral)"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Sparse logits"
  - "Importance Sampling"
  - "LLM Pre-training"
  - "Unbiased Estimation"
date: 2026-05-08
content_hash: e56e938c6105f0c2
---

# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs

**Conference**: ACL 2025 (Oral)  
**arXiv**: [2503.16870](https://arxiv.org/abs/2503.16870)  
**Code**: [GitHub](https://github.com/SamsungLabs/RSKD)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Sparse logits, Importance Sampling, LLM Pre-training, Unbiased Estimation  

## TL;DR

Demonstrates that naive Top-K sparse knowledge distillation yields biased estimation and proposes Random Sampling Knowledge Distillation (RSKD) based on importance sampling. RSKD provides unbiased gradient estimation while requiring the storage of only extremely sparse logits. The training overhead is increased by less than 10% compared to cross-entropy, while maintaining performance on par with full knowledge distillation.

## Background & Motivation

**Background**: Knowledge distillation (KD) is a classic technique of transferring knowledge from a large model (teacher) to a smaller model (student). In the LLM era, KD is considered an important means of reducing deployment costs. A natural implementation path is to pre-compute and cache the teacher's output logits, and then load and use them when training the student, which avoids repeatedly running the teacher's forward pass.

**Limitations of Prior Work**: The vocabulary size of LLMs is typically very large (32k-128k), and fully caching the entire logit vector for each token requires massive storage space. An intuitive solution is to cache only Top-K logits (e.g., K=10 or 100). However, whether this practice is theoretically correct has not been rigorously analyzed before. Furthermore, applying sparse distillation to the LLM pre-training stage has been almost entirely unexplored.

**Key Challenge**: The trade-off between storage efficiency and distillation quality. Caching all logits is too storage-heavy, while Top-K caching, despite saving space, introduces bias—the truncated probability mass cannot be correctly normalized, causing the student to learn an incorrect probability distribution.

**Goal**: Design a sparse yet unbiased logit caching strategy to achieve an optimal balance between storage efficiency and performance in LLM knowledge distillation.

**Key Insight**: The authors approach the problem from a statistical perspective, first rigorously proving that the Top-K method yields biased gradient estimation, and then designing an unbiased alternative using importance sampling theory.

**Core Idea**: Use random sampling (instead of deterministic Top-K) to select the logit positions to be cached, and correct it through importance sampling weights to ensure unbiased gradient estimation, while only needing to store a highly sparse set of logits.

## Method

### Overall Architecture

The training process is divided into two phases: (1) Offline phase—runs the teacher model to perform inference on the training data, and randomly samples K logit indices and their values for storage at each token position; (2) Online phase—when training the student model, loads the sparse logits from the cache and constructs the distillation loss using importance sampling weights. The overall training pipeline is identical to standard KD, with differences only in the logit storage and loss calculation.

### Key Designs

1. **Theoretical Analysis of Top-K Bias**:

    - Function: Theoretically proves the issues of naive Top-K sparse KD.
    - Mechanism: When only keeping the Top-K tokens of the teacher's probability distribution, the remaining probability mass is discarded or uniformly distributed to the other tokens, presenting a distorted distribution to the student. Formally, the gradient of the KL divergence loss between $\hat{p}_{\text{Top-K}}$ and the ground-truth $p_{\text{teacher}}$ is inconsistent, rendering the student unable to learn tail distribution correctly.
    - Design Motivation: To establish a theoretical foundation for proposing the subsequent unbiased scheme, and to explain why directly using Top-K KD degrades calibration.

2. **Random Sampling Knowledge Distillation (RSKD)**:

    - Function: Provides an unbiased sparse knowledge distillation scheme.
    - Mechanism: Instead of choosing a fixed Top-K, K token positions are randomly sampled for caching proportionally based on the teacher's probability distribution. For each sampled token $i$, its logit value and sampling probability $q(i)$ are recorded. Then, an importance sampling estimator is constructed: $\hat{\mathcal{L}}_{\text{KD}} = -\sum_{i \in S} \frac{p_T(i)}{q(i)} \log p_S(i)$, where $S$ is the sampled set. This ensures that $\mathbb{E}[\nabla \hat{\mathcal{L}}] = \nabla \mathcal{L}_{\text{full KD}}$, indicating that the gradient is unbiased in expectation.
    - Design Motivation: Importance sampling is a classic method in Monte Carlo estimation. Introducing it into the knowledge distillation scenario drastically reduces the number of logits to be stored while maintaining unbiasedness.

3. **Adaptive Sampling Probability Design**:

    - Function: Optimizes sampling efficiency and reduces the variance of gradient estimation.
    - Mechanism: The choice of the sampling probability $q(i)$ directly affects the estimation variance. The authors design a sampling distribution related to the teacher's probability values and the student's learning signals, making higher-probability and high-information tokens more likely to be sampled, thereby reducing the variance. In practice, the teacher's softmax probability is adopted as the sampling distribution.
    - Design Motivation: Although uniform random sampling is also unbiased, its variance is large. Adaptive sampling behaves like "smart sample selection", obtaining more stable gradients under the same level of sparsity.

### Loss & Training

The final training loss is a weighted combination of standard cross-entropy (CE) loss and the RSKD distillation loss: $\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1-\alpha) \cdot \hat{\mathcal{L}}_{\text{RSKD}}$. The RSKD loss term ensures unbiasedness through importance weights, restricting the overall training overhead to less than an 10% increase compared to pure CE training.

## Key Experimental Results

### Main Results

| Model Size | Distillation Method | Perplexity (PPL) ↓ | Avg. Downstream Accuracy ↑ | Storage Overhead |
|----------|---------|---------------|----------------|---------|
| 300M | CE only (No Distillation) | Baseline | Baseline | 0 |
| 300M | Full KD | Best | Best | 100% |
| 300M | Top-K KD (K=10) | Suboptimal, but biased | Lower than Full KD | ~0.01% |
| 300M | RSKD (K=10) | Close to Full KD | Close to Full KD | ~0.01% |
| 1B | Full KD | Best | Best | 100% |
| 1B | RSKD (K=10) | Close to Full KD | Close to Full KD | ~0.01% |
| 3B | Full KD | Best | Best | 100% |
| 3B | RSKD (K=10) | Close to Full KD | Close to Full KD | ~0.01% |

### Ablation Study

| Configuration | PPL | Description |
|------|-----|------|
| RSKD (K=10) | Near-optimal | Only 10 logits need to be stored to get close to Full KD |
| RSKD (K=5) | Slightly decreased | Further sparsification, yet still outperforms Top-K |
| Top-K (K=100) | Moderate | Biased even when K is increased by 10x |
| Top-K (K=10) | Poor | Bias is most pronounced |
| Uniform Sampling (K=10) | High variance | Unbiased but unstable |
| RSKD + Adaptive Sampling | Best | Unbiased and low-variance |

### Key Findings

- Even with $K=100$, the Top-K method fails to eliminate bias, whereas RSKD approaches Full KD performance at $K=10$ with less storage than Top-K.
- RSKD significantly outperforms Top-K in terms of calibration (the reliability of probability estimation), validating the theoretical analysis regarding bias.
- The advantages of RSKD are consistently maintained from scales of 300M to 3B, demonstrating good scalability.
- The training time overhead is increased by less than 10%, primarily stemming from reading cached logits and calculating importance weights.

## Highlights & Insights

- **Theory-driven Method Design**: Rigorously proves that Top-K is biased first, and then proposes an unbiased scheme based on importance sampling theory, featuring a highly complete logical chain. This "theory first, practice later" research paradigm is exemplary.
- **Extreme Storage Efficiency**: Only about 0.01% of the logit information needs to be cached to achieve performance close to Full KD, making the "pre-computation + caching" KD paradigm truly viable in practice.
- **Direct Applicability of RSKD to LLM Pre-training Distillation**: This is one of the few works extending sparse KD to the pre-training stage, breaking the previous limitation where KD was primarily used for fine-tuning.

## Limitations & Future Work

- The experiment only scales up to a 3B student model; verification is needed to confirm if it remains effective at 7B/13B scales.
- The pre-computation phase of the teacher model incurs resource costs itself, which are not analyzed end-to-end in detail in the paper.
- Evaluation is only conducted on English pre-training; logit distribution differences across different languages in multilingual scenarios might affect sampling efficiency.
- Future work could explore integration with other compression techniques like quantization and pruning to further decrease deployment costs.
- Whether the sampling strategy can be adaptively adjusted during the training process (e.g., curriculum-learning style sampling) is an interesting direction.

## Related Work & Insights

- **vs MiniLLM**: MiniLLM employs the reverse form of KL divergence for distillation, requiring the teacher to run online. RSKD allows completely offline distillation, which is more practical.
- **vs TinyLLaMA**: TinyLLaMA trains a small model from scratch without distillation. RSKD demonstrates that even simple distillation can yield significant improvements.
- **vs DistilBERT**: A classic BERT distillation method but does not address the sparse logit challenge. RSKD solves the core efficiency bottleneck under large vocabulary scenarios.
- An ACL 2025 Oral paper, which serves as a paradigm of theoretical and experimental integration, and is highly recommended as a baseline reference for the KD domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Using importance sampling for KD logit caching is a fresh perspective with solid theoretical backing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple scales and K-values with comprehensive ablations, though the maximum model scale could be larger.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations; high oral-paper caliber.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a core practical bottleneck in LLM KD, yielding high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](../../ICCV2025/model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](../../ICML2025/model_compression/efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/model_compression/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ACL 2025\] Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition](compact_and_compressible_representations_for_llms_using_structured_sparse_decom.md)

</div>

<!-- RELATED:END -->
