---
title: >-
  [Paper Note] Pre-training Distillation for Large Language Models: A Design Space Exploration
description: >-
  [ACL 2025][Model Compression][pre-training distillation] This work systematically explores the design space of pre-training distillation (PD) for large language models across four dimensions: logits processing, loss function selection, scaling laws, and offline/online logits. Through extensive experiments, optimal configurations and valuable insights are provided.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "pre-training distillation"
  - "knowledge distillation"
  - "logits processing"
  - "scaling law"
  - "LLM"
date: 2026-05-08
content_hash: e37431db2981e706
---

# Pre-training Distillation for Large Language Models: A Design Space Exploration

**Conference**: ACL 2025  
**arXiv**: [2410.16215](https://arxiv.org/abs/2410.16215)  
**Code**: —  
**Area**: Model Compression / Knowledge Distillation / Large Language Model Pre-training  
**Keywords**: pre-training distillation, knowledge distillation, logits processing, scaling law, LLM  

## TL;DR

This work systematically explores the design space of pre-training distillation (PD) for large language models across four dimensions: logits processing, loss function selection, scaling laws, and offline/online logits. Through extensive experiments, optimal configurations and valuable insights are provided.

## Background & Motivation

### Background
Knowledge Distillation (KD) is a standard method for transferring knowledge from large models to smaller ones. In the LLM era, KD is primarily applied in the **post-training stage**, where student models learn directly from teacher-generated instruction-response pairs. However, the exploration of extending KD to the **pre-training stage** remains quite limited.

### Design Motivation
- Teacher model logits contain richer information than one-hot labels, serving as label smoothing to accelerate training and improve performance.
- Although pre-training distillation (PD) is intuitively advantageous, there is a lack of systematic research to guide its optimal implementation.
- This work aims to fill this gap by exploring the design space of pre-training distillation through a large number of controlled experiments.

### Basic Formula
$$\theta_S^* = \arg\min_{\theta_S} [(1-\alpha)\mathcal{L}_{\text{lm}} + \alpha\mathcal{L}_{\text{kd}}]$$

where $\mathcal{L}_{\text{lm}}$ is the conventional language modeling loss, $\mathcal{L}_{\text{kd}}$ represents the distillation loss, and $\alpha$ controls the trade-off between the two.

## Method

### Four Dimensions of the Design Space

**Dimension 1: Logits Processing**
- **Truncation Methods**: A **two-stage top-p-k truncation** is proposed—first applying top-p truncation (effective when the distribution is sharp), followed by top-k truncation (a secondary truncation when the distribution is uniform).
    - The vocabulary size of GLM-4-9B is approximately 150K, which requires 58.6 PB for full storage. After top-0.95-100 truncation, this is reduced to approximately **15 TB** (a 4000$\times$ compression).
- **Temperature Normalization**: $F(\mathbf{z}) = \text{softmax}(\text{Truncate}(\mathbf{z}) / \tau)$

**Dimension 2: Loss Function Selection**
- Selection of distillation loss $L$: NLL (negative log-likelihood), KLD (Kullback-Leibler divergence), MSE
- Scheduling strategy for $\alpha$: static, linear increase/decrease, cyclical, and WSD scheduling

**Dimension 3: Scaling Law**
- Student model sizes: 330M, 670M, 1.9B, 3.8B, 6.8B
- Teacher model sizes: 9B, 32B
- Pre-training corpus size: 100B, 500B tokens

**Dimension 4: Offline vs. Online Logits**
- Offline: Generating logits using a pre-trained teacher model.
- Online: Synchronously storing logits during the pre-training process of the teacher model.

### Preliminary Experiments
- Teacher model: GLM-4-9B
- Student model: 1.9B
- Training data: 100B tokens
- Under the preliminary configuration, PD brings an average performance gain of **1.6%**.

## Key Experimental Results

### Dimension 1: Logits Processing Experimental Results

**top-p-k Truncation**:
- Performance differences across different $p$ values (top-$p$-100) are negligible. **Smaller $p$ values further reduce storage**.
- For different $k$ values (top-0.95-$k$), $k=50$ achieves the best performance; $k=1$ also brings improvement (equivalent to LM training utilizing teacher labels, acting as implicit noise filtering).

**Temperature**:

| τ | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|------|-----|-----|-----|-----|-----|-----|------|
| Gain (%) | 1.6 | 2.1 | 2.5 | **2.7** | 1.6 | 2.5 | -0.1 | 1.0 |

- Performance is comparable when $\tau \le 2.0$, while the improvement is limited for $\tau \ge 5.0$ (excessively uniform distributions hinder student learning).
- Adaptive temperature (AdaKDH) yields the best outcome (+2.8%), but shows no significant additional gain compared to the optimal static configuration.

### Dimension 2: Loss Function Experimental Results

**Comparison of Distillation Losses**:

| Method | Average Gain |
|------|---------|
| NLL | +1.6% |
| KLD | **+2.6%** |
| MSE | **-7.6%** |

- Both KLD and NLL are effective, whereas MSE leads to a significant performance drop—contrary to the findings in computer vision (CV) where MSE is often optimal.
- **Optimal $\alpha$ Scheduling**: WSD-$\alpha$ + WSD-LR leads to an improvement of **+8.0%**.
    - Key Insight: Utilizing a higher KD loss during the peak learning rate stage significantly enhances performance.
    - A linearly decreasing $\alpha$ (KD first, then LM) outperforms a linearly increasing one (LM first, then KD).

### Dimension 3: Scaling Law Experimental Results

**Model Size**:
- **Larger student models configurations benefit more**: 6.8B > 3.8B > 1.9B > 670M > 330M.
- **Larger teacher models are not necessarily better**: The 9B teacher outperforms the 32B teacher on certain students, potentially due to the capacity gap.
- PD is effective when the student model size is at least $\sim 10\%$ of the teacher model size.

**Corpus Size** (500B tokens experiment):
- PD consistently brings improvement throughout the entire pre-training process.
- The gains initial increase and then tend to stabilize, yet remain significant in the end—PD not only improves training efficiency but also raises the performance upper bound.

### Dimension 4: Offline vs. Online Experimental Results

| Method | Average Gain |
|------|---------|
| LLM-Online-100B-L (early teacher logits) | -20.9% |
| LLM-Online-100B (late teacher logits) | -3.9% |
| LLM-Online-100B* ($\alpha=0.1$, tuned) | +0.5% |

- Even logits from an unconverged teacher can slightly assist student training.
- Online logits are less effective than offline ones, but they offer the advantage of **zero extra inference overhead**.
- Recommendation: If training a suite of LLMs of different sizes, online logits can be recorded during the pre-training of the largest model.

## Highlights & Insights

1. **Systematic Design Space Exploration**: This work is the first to conduct comprehensive controlled experiments across four key dimensions of LLM pre-training distillation.
2. **WSD-$\alpha$ + WSD-LR Combination** yields the greatest benefit (+8.0%), serving as key practical guidance.
3. **Failure of MSE Loss in LLM Distillation**: This is contrary to experience in CV, suggesting unique training dynamics in LLM pre-training distillation.
4. **"Capacity Gap" Effect**: Larger teachers are not always better, providing a reference for teacher selection in practical deployments.
5. **Feasibility of Online Logits**: This study is the first to validate the effectiveness of synchronously saving logits during teacher pre-training for subsequent distillation.
6. **Storage Efficiency**: The top-p-k truncation achieves a 4000$\times$ compression, making the storage of large-scale logits viable.

## Limitations & Future Work

1. Interaction effects among different factors are not explored (combinatorial experiments are too costly).
2. Pre-training scales do not reach the trillion-token level (the scale used by current state-of-the-art LLMs).
3. Weak-to-strong scenarios where the student model surpasses the teacher are not investigated.
4. The computational cost of the experiments is extremely high, with carbon emissions representing a potential ethical concern.

## Related Work & Insights

- **Pre-ChatGPT Era Distillation**: DistilBERT, TinyBERT, etc., are based on million-parameter models, and their configurations are not directly applicable to billion-parameter LLMs.
- **LLM Distillation**: Gemma 2, AFM, LokiLM, and Minitron employ pre-training distillation, though details remain limited.
- **Post-Training Distillation**: Alpaca, Vicuna, etc., learn from GPT-4 responses.
- **MiniLLM**: Distillation on top of pre-trained LLMs rather than training from scratch.

## Rating

⭐⭐⭐⭐ — The extensive experiments provide highly valuable references, and the conclusions are clear and practical (particularly findings regarding WSD scheduling, the failure of MSE, and the capacity gap). As a design space exploration, it is highly systematic, though it lacks cross-dimensional combinations and larger-scale validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)
- [\[ACL 2025\] DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)

</div>

<!-- RELATED:END -->
