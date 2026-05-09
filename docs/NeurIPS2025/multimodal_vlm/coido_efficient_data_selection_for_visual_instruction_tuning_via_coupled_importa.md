---
title: >-
  [Paper Note] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization
description: >-
  [NeurIPS 2025][Multimodal VLM][Data Selection] This paper proposes CoIDO, a bi-objective optimization framework for data selection that jointly optimizes data importance and diversity. By training a lightweight scorer on only 20% of randomly sampled data, CoIDO selects a 20% subset from LLaVA-665K that achieves 98.2% of the performance of full-data fine-tuning, while incurring the lowest computational overhead among all compared methods.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Data Selection
  - Instruction Tuning
  - Importance-Diversity Optimization
  - Lightweight Scorer
  - MLLM
date: 2026-05-08
content_hash: 043f1ca53c76202a
---

# CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.17847](https://arxiv.org/abs/2510.17847)
**Code**: [GitHub](https://github.com/SuDIS-ZJU/CoIDO)
**Area**: Multimodal VLM
**Keywords**: Data Selection, Instruction Tuning, Importance-Diversity Optimization, Lightweight Scorer, MLLM

## TL;DR

This paper proposes CoIDO, a bi-objective optimization framework for data selection that jointly optimizes data importance and diversity. By training a lightweight scorer on only 20% of randomly sampled data, CoIDO selects a 20% subset from LLaVA-665K that achieves 98.2% of the performance of full-data fine-tuning, while incurring the lowest computational overhead among all compared methods.

## Background & Motivation

Instruction tuning of multimodal large language models (MLLMs) relies on large-scale datasets such as LLaVA-665K, yet a single training epoch of LLaVA-1.5-7B requires over 20 hours on 8×A100 40GB GPUs. Data selection methods aim to identify high-quality subsets to reduce this cost, but existing approaches suffer from two critical limitations:

**Decoupled importance and diversity**: Data importance is assessed during training, while diversity is optimized independently during selection. Because the two objectives are never jointly optimized, the resulting trade-off is often suboptimal.

**Full-dataset traversal requirement**: Existing methods require the target MLLM to process the entire dataset to compute gradients or features, incurring computational costs comparable to full fine-tuning and thereby undermining the purpose of data selection.

The core idea of CoIDO is to train a lightweight scorer on a small fraction of the data, enabling it to learn the data distribution and subsequently infer selection scores for the full dataset.

## Method

### Overall Architecture

Given a large-scale visual instruction dataset $\mathcal{D} = \{z_j\}_{j=1}^N$, the goal is to select a subset $\mathcal{D}_h \subset \mathcal{D}$ such that $|\mathcal{D}_h| = \gamma N$ (e.g., $\gamma = 0.2$), while retaining at least 95% of the full-data performance.

The CoIDO pipeline proceeds as follows:

1. **Feature extraction**: Extract textual features (LLM Score), visual features (ImageReward Score), and image-text alignment features (CLIP Score).
2. **Spectral clustering**: Apply spectral clustering to the multimodal features to obtain $M$ clusters.
3. **Scorer training**: Train the CoIDO Scorer (a 4-layer MLP) jointly with the target MLLM using only $p\%$ (e.g., 20%) of randomly sampled data.
4. **Data selection**: Infer CoIDO scores for the full dataset and perform task-stratified selection of the top-$\gamma$ subset.

### Key Designs

**Importance loss $\mathcal{L}_I$**: The scorer outputs a CoIDO score $w_{ik}$ for each sample. After Softmax normalization, these scores weight the cross-entropy loss of the target MLLM:

$$\mathcal{L}_I = \sum_{i=1}^{m} \sum_{k=1}^{n_i} w_{ik} \cdot \text{CE}(y_{ik}, \hat{y}_{ik})$$

By the chain rule of backpropagation, samples with high cross-entropy (harder samples) receive lower $w_{ik}$, meaning a lower score corresponds to higher importance.

**Diversity loss $\mathcal{L}_D$**: The variance of the per-cluster mean weights is minimized to prevent any single cluster from dominating the selection:

$$\mathcal{L}_D = \text{Var}(\{\bar{w}_1, \bar{w}_2, \ldots, \bar{w}_m\}), \quad \bar{w}_i = \frac{1}{n_i} \sum_{k=1}^{n_i} w_{ik}$$

Note that $\mathcal{L}_D$ constrains only the inter-cluster means, preserving the intra-cluster ranking.

### Loss & Training

A homoscedastic uncertainty framework is adopted to dynamically balance the two objectives:

$$\mathcal{L}_{\text{total}} = \frac{1}{\sigma_I^2} \mathcal{L}_I + \frac{1}{2\sigma_D^2} \mathcal{L}_D + \log \sigma_I + \log \sigma_D$$

where $\sigma_I$ and $\sigma_D$ are learnable parameters that regulate the uncertainty of the importance and diversity objectives, respectively. At inference time, these two parameters are discarded and the trained scorer is used directly for scoring.

The derivation of the importance objective is grounded in a temperature-scaled Boltzmann distribution:

$$p(y_{ik} | x_{ik}, \theta, \sigma_I, w_{ik}) = \text{Softmax}\left(\frac{w_{ik}}{\sigma_I^2} f_\theta(x_{ik})\right)$$

A second-order Taylor expansion is applied to approximate the log-sum-exp term, and the first-order error term is neglected by exploiting the highly concentrated output distribution of LLMs (effective candidate tokens $T \approx 5\text{-}10$).

## Key Experimental Results

### Main Results

**LLaVA-1.5-7B-LoRA fine-tuned on a 20% subset selected from LLaVA-665K**:

| Method | VQAv2 | GQA | SQA-I | POPE | MME | MMBench(en) | Rel.(%) | Training Data | FLOPs |
|--------|-------|-----|-------|------|-----|-------------|---------|---------------|-------|
| Full Data | 79.1 | 63.0 | 68.4 | 86.4 | 1476.9 | 66.1 | 100 | — | 10.2E |
| Random | 75.9 | 59.3 | 68.6 | 85.9 | 1461.0 | 60.3 | 95.1 | — | — |
| ICONS | 77.0 | 60.4 | 70.4 | 86.1 | 1447.7 | 64.6 | 97.1 | 100+5+2.2% | 12.6E |
| COINCIDE | 76.5 | 59.8 | 69.2 | 86.1 | 1495.6 | 63.1 | 97.4 | 100% | 4.9E |
| **CoIDO** | **77.2** | **60.4** | 69.4 | 85.4 | 1450.2 | **63.8** | **98.2** | **20%** | **4.2E** |

### Ablation Study

**Comparison of optimization strategies**:

| Loss Function | VQAv2 | GQA | MMBench(en) | Rel.(%) |
|---------------|-------|-----|-------------|---------|
| $\mathcal{L}_I$ only | 77.9 | 48.9 | 51.1 | 89.0 |
| $\mathcal{L}_I + \mathcal{L}_D$ | 74.5 | 55.8 | 57.0 | 92.0 |
| $\lambda \mathcal{L}_I + (1-\lambda) \mathcal{L}_D$ | 76.1 | 59.4 | 60.5 | 95.9 |
| Homoscedastic Uncertainty (Ours) | **77.2** | **60.4** | **63.8** | **98.2** |

**Sensitivity to training data ratio**: Performance is generally lower when $p < 10\%$; it stabilizes once $p > 20\%$, indicating that 20% of the data is sufficient to capture the overall data distribution.

### Key Findings

1. CoIDO achieves 98.2% relative performance with only 20% training cost and 4.2 ExaFLOPs, excelling in both efficiency and accuracy.
2. Optimizing importance alone ($\mathcal{L}_I$) yields only 89% relative performance; incorporating diversity raises it to 98.2%, demonstrating the necessity of joint optimization.
3. The MLP scorer outperforms more complex Transformer-based scorers, owing to the already rich input features (CLIP Score, ImageReward, etc.).

## Highlights & Insights

- **Extreme efficiency**: The scorer requires only 20% of the data for training, substantially reducing overhead compared to ICONS, which requires 100%+5%+2.2% of the data.
- **Theoretical elegance**: The loss balancing mechanism is naturally derived from the MLE framework and homoscedastic uncertainty, eliminating the need for manual hyperparameter tuning.
- **Transferability**: Once trained, the scorer can be directly applied to new in-domain data without retraining.
- **No dedicated selection algorithm**: After training, data is selected by straightforward score-based ranking, entirely avoiding complex post-hoc selection procedures.

## Limitations & Future Work

- Experiments are conducted only on LLaVA-1.5-7B; generalization to larger-scale or more recent MLLM architectures remains unverified.
- The number of clusters $M$ in spectral clustering must be specified in advance, which may affect the quality of diversity modeling.
- Applicability to pre-training data selection has not been validated, as the study focuses exclusively on visual instruction tuning.
- Under the 20% selection ratio, POPE performance exhibits a slight decline (85.4 vs. 86.4), suggesting that hallucination detection benchmarks may be more sensitive to data composition.

## Related Work & Insights

- **TIVE / ICONS**: Gradient-based data selection methods that require full-dataset traversal; CoIDO avoids this overhead via the lightweight scorer.
- **COINCIDE**: A clustering-based method that also processes the full dataset; CoIDO achieves superior efficiency.
- **Kendall et al. (CVPR 2018)**: The homoscedastic uncertainty multi-task learning framework, which CoIDO innovatively adapts for importance-diversity balancing in data selection.
- Insight: The data selection problem can be formulated as a multi-objective optimization problem, and uncertainty weighting provides a natural and effective balancing mechanism.

## Rating

- ⭐ Novelty: 4/5 — The joint optimization framework and lightweight scorer design are novel, supported by rigorous theoretical derivation.
- ⭐ Experimental Thoroughness: 4/5 — Ablations are comprehensive, covering multiple scorer architectures and optimization strategies, though model scale is limited.
- ⭐ Writing Quality: 4/5 — Mathematical derivations are clear and the architecture diagram is intuitive.
- ⭐ Value: 4/5 — Provides an efficient and practical data selection tool for large-scale MLLM instruction tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](../../ICCV2025/multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)

</div>

<!-- RELATED:END -->
