---
title: >-
  [Paper Note] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models
description: >-
  [CVPR 2026][Self-Supervised Learning][model chain] This paper proposes Chain-of-Models Pre-Training (CoM-PT), which arranges vision foundation models in a size-ordered "model chain" and progressively accelerates training…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "model chain"
  - "pre-training acceleration"
  - "inverse knowledge transfer"
  - "CLIP"
  - "vision foundation models"
date: 2026-05-08
content_hash: 73701941f067130c
---

# Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2604.12391](https://arxiv.org/abs/2604.12391)  
**Code**: [https://github.com/deep-optimization/CoM-PT](https://github.com/deep-optimization/CoM-PT)  
**Area**: Self-Supervised Learning / Training Acceleration
**Keywords**: model chain, pre-training acceleration, inverse knowledge transfer, CLIP, vision foundation models

## TL;DR

This paper proposes Chain-of-Models Pre-Training (CoM-PT), which arranges vision foundation models in a size-ordered "model chain" and progressively accelerates training via inverse knowledge transfer (weight initialization + feature distillation) from smaller to larger models, achieving lossless training acceleration whose efficiency improves as the model family grows.

## Background & Motivation

**Background**: Pre-training vision foundation models (VFMs) is extremely costly (e.g., ViT-L/14 on LAION-2B requires $1.2 \times 10^5$ A100 GPU hours). Existing acceleration methods—mixed precision, masked modeling, data-efficient approaches, etc.—optimize within a single-model dimension.

**Limitations of Prior Work**: VFMs are typically pre-trained as model families (varying sizes to satisfy different deployment scenarios), yet the standard practice of independent training is highly redundant—models share the same optimization objective, dataset, and training protocol, causing common knowledge to be repeatedly learned.

**Key Challenge**: As model families continue to grow (more specialized model sizes and larger model ranges), the total cost of independent training scales linearly, creating a dilemma between bearing escalating pre-training costs and sacrificing deployment flexibility.

**Goal**: Achieve pre-training acceleration that scales efficiently with model family size.

**Key Insight**: At the micro level, the training cost of large models is the dominant source of overhead; at the macro level, the redundancy of independent training is the root cause of inefficiency. The key to addressing both bottlenecks simultaneously lies in enabling intra-family knowledge reuse from small to large models.

**Core Idea**: Arrange the model family in size order to form a model chain; the smallest model is trained from scratch, and each subsequent model is accelerated via inverse knowledge transfer (small → large).

## Method

### Overall Architecture

The model chain $C_M: m_1 \rightarrow m_2 \rightarrow \cdots \rightarrow m_n$ is ordered by increasing model size. $m_1$ is pre-trained independently from scratch; each subsequent model $m_{i+1}$ is accelerated via inverse knowledge transfer from $m_i$. Inverse knowledge transfer consists of two components: weight initialization in parameter space and feature distillation in feature space.

### Key Designs

1. **Inverse Weight Initialization**:

    - **Function**: Reuses the knowledge of smaller models to initialize larger models in parameter space.
    - **Mechanism**: (i) Width expansion: directly embeds the smaller teacher's parameters into the corresponding positions of the larger student, with remaining parameters randomly initialized; (ii) Depth expansion: copies each layer's weights as the successor layer. A straightforward function-preserving initialization strategy.
    - **Design Motivation**: Leverages the already-trained smaller model's knowledge to provide a better starting point, accelerating convergence of the larger model.

2. **Inverse Feature Distillation**:

    - **Function**: Reuses the dynamic knowledge of smaller models in feature space.
    - **Mechanism**: $\mathcal{L}_{IFD}(F^t, F^s) = \alpha \| F^t - \mathbf{T}(F^s) \|_2^2$, projecting student features into the teacher feature space via transformation $\mathbf{T}(\cdot)$. In CLIP, distillation is applied to both visual and text features: $\hat{\mathcal{L}}_{IFD} = (\mathcal{L}_{IFD}(v^t,v^s) + \mathcal{L}_{IFD}(t^t,t^s))/2$.
    - **Design Motivation**: Weight initialization transfers static knowledge, while feature distillation captures dynamic cross-sample knowledge; the two components work synergistically to ensure effective knowledge transfer relay.

3. **Three Principles for Model Chain Design**:

    - **Function**: Guides the construction of an optimal model chain.
    - **Mechanism**: (i) Optimal smallest model: selected based on data scale—small enough to maximize efficiency yet with sufficient capacity to fit the data distribution; (ii) Intermediate model variants: use expansion ratios of 2×–4×, with larger ratios optimizing cost and smaller ratios maximizing speedup; (iii) Training epoch allocation: decreases linearly along the model chain.
    - **Design Motivation**: A counterintuitive phenomenon emerges—the ViT-T→S→B→L chain trains two additional models compared to ViT-B→L yet incurs 20% lower total cost.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{task} + \hat{\mathcal{L}}_{IFD}$, where the task loss is the contrastive loss of LaCLIP (with text augmentation). It is enforced that $\mathcal{L}_{IFD} < \mathcal{L}_{task}$.

## Key Experimental Results

### Main Results

| Model Chain | ImageNet Top-1 | Training MACs | Speedup |
|-------------|---------------|---------------|---------|
| ViT-L (independent) | 38.2% | 100% | 1.0× |
| ViT-B→L | 38.0% | 48% | 2.1× |
| ViT-S→B→L | 38.1% | 36% | 2.8× |
| ViT-T→S→B→L | **38.3%** | **28%** | **3.6×** |

### Ablation Study

| Configuration | ImageNet Top-1 | Notes |
|---------------|---------------|-------|
| Full CoM-PT | 38.3% | Weight init + feature distillation |
| Weight init only | 37.8% | No distillation |
| Feature distillation only | 37.5% | Random initialization |
| Independent training | 38.2% | Baseline |

### Key Findings

- Counterintuitive phenomenon: training more models yields higher efficiency—extending the chain from 3 → 4 → 7 models raises the speedup from 4.13× to 5.68× and 7.09×.
- The model chain structure itself drives the primary efficiency gains; weight initialization and distillation each contribute modestly but synergize well.
- Performance is validated as lossless (<0.5% accuracy degradation) across 45 downstream datasets.

## Highlights & Insights

- "Training more models leads to higher efficiency" is a highly insightful finding: intermediate models in an extended chain converge rapidly thanks to their predecessors, such that the total overhead can be lower than directly training the large model alone.
- The method is agnostic to the pre-training paradigm and can be generalized to more compute-intensive settings such as LLM pre-training.
- Inverse knowledge transfer (small → large) forms a dual counterpart to conventional knowledge distillation (large → small), offering a novel perspective.

## Limitations & Future Work

- Validation is primarily conducted on CLIP; large-scale testing on LLM pre-training has not yet been performed.
- Model chain design still requires manual tuning; an automated search method is lacking.
- Width and depth expansion rely on simple copy/insertion strategies; more principled alternatives may exist.
- Cross-architecture model chains (e.g., ViT → Swin) remain unexplored.

## Related Work & Insights

- **vs. Net2Net**: Net2Net first proposed function-preserving transformations for model expansion; CoM-PT extends this idea into a systematic training pipeline.
- **vs. FLIP/DeCLIP**: These methods accelerate training along the single-model dimension, whereas CoM-PT accelerates along the model-family dimension—the two approaches are orthogonal and complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Model-family-level training acceleration represents an entirely new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 45 downstream datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Micro/macro-level analysis is thorough and well-structured.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for large-scale pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[ICML 2026\] NITP: Next Implicit Token Prediction for LLM Pre-training](../../ICML2026/self_supervised/nitp_next_implicit_token_prediction_for_llm_pre-training.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)

</div>

<!-- RELATED:END -->
