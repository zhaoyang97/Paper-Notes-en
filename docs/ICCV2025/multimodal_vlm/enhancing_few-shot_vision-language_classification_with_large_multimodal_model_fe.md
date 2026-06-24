---
title: >-
  [Paper Note] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features
description: >-
  [ICCV 2025][Multimodal VLM][Sparse attention vectors] This paper proposes Sparse Attention Vectors (SAVs) — a training-free method that extracts fewer than 5% of attention heads from frozen generative Large Multimodal Models (LMMs) as strong feature representations. With only approximately 20 labeled samples per class, SAVs achieve state-of-the-art performance on vision-language classification tasks, outperforming LoRA fine-tuning by an average of 7% on challenging benchmarks…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Sparse attention vectors"
  - "few-shot classification"
  - "feature extraction"
  - "training-free"
  - "vision-language classification"
date: 2026-05-08
content_hash: b54132cb6ee7794c
---

# Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features

**Conference**: ICCV 2025
**arXiv**: [2412.00142](https://arxiv.org/abs/2412.00142)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Sparse attention vectors, few-shot classification, feature extraction, training-free, vision-language classification

## TL;DR

This paper proposes Sparse Attention Vectors (SAVs) — a training-free method that extracts fewer than 5% of attention heads from frozen generative Large Multimodal Models (LMMs) as strong feature representations. With only approximately 20 labeled samples per class, SAVs achieve state-of-the-art performance on vision-language classification tasks, outperforming LoRA fine-tuning by an average of 7% on challenging benchmarks including BLINK, VLGuard, and NaturalBench.

## Background & Motivation

### Core Problem

Generative LMMs (e.g., LLaVA, Qwen2-VL) excel at open-ended vision-language tasks, yet underperform encoder-based models such as CLIP on vision-language classification tasks (where inputs are image-text pairs and outputs are discrete labels). The paper aims to extract multimodal features from frozen generative LMMs that can be applied to arbitrary downstream classification tasks without any fine-tuning.

### Limitations of Prior Work

**Generative LMMs perform poorly on classification tasks**: Billion-parameter LMMs underperform smaller models such as CLIP and SigLIP on image classification, because their generative outputs are ill-suited to discrete label prediction.

**Existing feature extraction approaches have inherent limitations**:
- Carefully engineered prompts (hard/soft prompting): fail to close the gap with encoder-based models.
- Fine-tuning (e.g., LoRA): requires training-scale data and computation for each new task, making it inefficient.
- Few-shot in-context learning (ICL): instruction tuning degrades the few-shot capability of LMMs, leading to inconsistent performance.

**CLIP features are unimodal**: CLIP extracts either visual or textual features independently and cannot handle interleaved image-text inputs (e.g., image + question in VQA).

### Core Motivation

**Key Insight**: Drawing on the neuroscientific concept of functional specificity — the observation that specific brain regions are responsible for specific functions — and on transformer interpretability research showing that individual attention heads correspond to specific tasks, the authors hypothesize that among the hundreds of attention heads in an LMM, a very small fraction (<5%) naturally develop feature representations well-suited to specific classification tasks. The attention vectors of these heads can be directly used as discriminative classifiers without any gradient updates.

## Method

### Overall Architecture

The SAVs method proceeds in three steps:
1. **Feature extraction**: Extract attention vectors from every attention head of the frozen LMM.
2. **Sparse head selection**: Select the $k$ heads with the highest classification accuracy on a small set of labeled samples.
3. **Majority-vote classification**: For each new sample, classify independently using the $k$ selected heads and take the majority vote.

### Key Designs

#### 1. **Attention Vector Extraction (Step 1)**

- **Function**: Extract feature vectors from the output of each attention head in the LMM.
- **Mechanism**:
  For an LMM with $L$ layers and $H$ attention heads per layer, given an input sequence $x = \{x_1, ..., x_T\}$, the attention vector of head $(l, m)$ is defined as the output at the last token position:

  $$\mathbf{h}_l^m(x_i^T) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_m}}\right)V$$

  where $d_m = d/H$. Attention vectors from all $L \times H$ heads are extracted as candidate features.

- **Design Motivation**: The attention vector at the last token position aggregates information from the entire input sequence, making it the most natural representation site in a generative model. Different heads may encode representations corresponding to different semantic dimensions.

#### 2. **Sparse Head Selection (Step 2)**

- **Function**: Select the $k$ heads most relevant to the target classification task from all $L \times H$ heads.
- **Mechanism**:
  Given a small labeled set $\{(x_i, y_i)\}_{i=1}^N$ (approximately 20 samples per class), for each head $(l, m)$:

  1. Compute the centroid vector for each class $c$:
  $$\mu_c^{l,m} = \frac{1}{|N_c|}\sum_{j:y_j=c}\mathbf{h}_l^m(x_j^T)$$

  2. Apply nearest-centroid classification using cosine similarity:
  $$s_{l,m}(x_i, c) = \frac{\mathbf{h}_l^m(x_i^T) \cdot \mu_c^{l,m}}{\|\mathbf{h}_l^m(x_i^T)\| \|\mu_c^{l,m}\|}$$

  3. Compute the classification score (accuracy on labeled samples) for each head:
  $$\text{score}(l, m) = \sum_{i=1}^N \mathbf{1}[\hat{y} = y_i]$$

  4. Select the top-$k$ scoring heads as SAVs:
  $$\mathcal{H}_{\text{SAV}} = \{(l,m) \mid \text{score}(l,m) \text{ is in the top } k\}$$

- **Design Motivation**: Directly evaluating each head's classification ability on labeled data is more precise than heuristic approaches (e.g., selecting the last few layers). Experiments demonstrate that only 20 heads (<5% of the total) suffice to capture task-relevant features.

#### 3. **Majority-Vote Classification (Step 3)**

- **Function**: Independently classify each new query using the $k$ selected heads, then aggregate via majority vote.
- **Mechanism**:
  For a query sequence $Q$, each head independently predicts:
  $$\hat{y}_{l,m} = \arg\max_{c \in \mathcal{C}} s_{l,m}(Q^T, c)$$

  The final prediction is determined by majority vote:
  $$\arg\max_{y \in \mathcal{C}} \sum_{(l,m) \in \mathcal{H}_{\text{SAV}}} \mathbf{1}[\hat{y}_{l,m} = y]$$

- **Design Motivation**: Majority voting is the simplest ensemble strategy; however, because the selected heads are already high-quality classifiers, simple voting achieves strong performance. This also avoids introducing any additional learnable parameters.

### Loss & Training

SAVs is a completely training-free method:
- The model is fully frozen; no gradient updates are performed at any stage.
- Approximately 20 labeled samples per class are required for head selection.
- The default number of selected heads is $k = 20$.
- Both LLaVA-OneVision-7B and Qwen2-VL-7B are supported.
- All experiments can be run on a single A100 GPU.

## Key Experimental Results

### Main Results

**SAVs vs. baselines on LLaVA-OneVision-7B (selected key tasks)**:

| Method | MHalu↑ | VLGuard↑ | BLINK↑ | NB-Group↑ | EuroSAT↑ | Pets↑ | ImageNet↑ |
|--------|--------|----------|--------|-----------|----------|-------|-----------|
| Zero-shot | 34.7 | 31.4 | 45.0 | 27.0 | 66.5 | 88.1 | 85.3 |
| 4-shot ICL | 25.0 | 35.0 | 38.9 | 22.2 | 47.1 | 63.9 | 60.6 |
| MTV | 37.3 | 32.9 | 44.5 | 30.7 | 65.5 | 88.5 | 85.6 |
| LoRA | 78.3 | 90.0 | 47.0 | 32.4 | 85.0 | 96.8 | 91.8 |
| **SAVs** | **80.8** | **94.3** | **51.8** | **35.1** | **86.7** | **97.0** | **99.6** |

**SAVs vs. baselines on Qwen2-VL-7B**:

| Method | MHalu↑ | VLGuard↑ | BLINK↑ | NB-Group↑ | EuroSAT↑ | Pets↑ |
|--------|--------|----------|--------|-----------|----------|-------|
| Zero-shot | 24.0 | 26.9 | 43.3 | 28.5 | 54.7 | 92.6 |
| LoRA | 84.8 | 87.7 | 46.3 | 28.8 | 72.9 | 98.4 |
| **SAVs** | **85.1** | **96.0** | **47.2** | **32.3** | **79.9** | **98.1** |

Average gain of SAVs over LoRA: +7% on LLaVA-OV; similar margins on Qwen2-VL.

### Ablation Study

**Comparison of classification strategies**:

| Classification Method | MHalu | NaturalBench | EuroSAT |
|----------------------|-------|--------------|---------|
| KNN | 71.9 | 28.2 | 84.2 |
| Linear probe (MLP) | 80.6 | 33.1 | 87.0 |
| **Centroid classification (Ours)** | **80.8** | **35.1** | **86.7** |

**Head-level vs. layer-level selection**:

| Feature Granularity | MHalu | NaturalBench | EuroSAT |
|--------------------|-------|--------------|---------|
| Select 2 layers | 68.3 | 31.2 | 83.1 |
| **Select 20 heads** | **80.8** | **35.1** | **86.7** |

**Scalability with sample count**: Performance increases continuously from 5 to 200 samples per class, with no signs of saturation.

**Number of attention heads**: Performance approaches optimality at 20 heads, within the range of 5 to 40 heads evaluated.

### Key Findings

1. **ICL degrades performance**: 4-shot ICL underperforms zero-shot on nearly all tasks, confirming that instruction tuning undermines the few-shot prompting capability of LMMs.
2. **SAVs surpass LoRA**: Without any gradient updates, SAVs outperform LoRA — which requires training — on the majority of tasks.
3. **Extreme sparsity suffices**: Only 20 heads (<5% of the total) capture task-relevant features, suggesting that classification-relevant information naturally concentrates in a small number of heads during pretraining.
4. **Heads outperform layers**: Selecting 20 sparse heads substantially outperforms selecting 2 complete layers, demonstrating that classification signals are distributed across a small number of heads throughout the model depth.
5. **High sample efficiency**: SOTA performance is achieved with only 20 samples per class, and performance continues to improve as the number of samples increases.

## Highlights & Insights

1. **Validation of the "functional specificity" hypothesis**: A phenomenon analogous to functional brain localization is identified in LMMs — specific attention heads naturally encode features relevant to specific tasks.
2. **Converting generative models into discriminative models**: Without modifying the model, selecting internal representations alone transforms a generative LMM into an effective classifier.
3. **Completely training-free**: No gradient computation is required, substantially lowering the barrier to deployment and reducing computational cost.
4. **Cross-task generalization**: SAVs are effective across highly heterogeneous tasks including safety detection, VQA, image-text retrieval, and image classification.
5. **Revealing a failure mode of ICL**: The conflict between instruction tuning and ICL is an important finding with broader implications.

## Limitations & Future Work

1. **Head selection must be repeated for each new task**: Although fine-tuning is not required, selecting SAVs still necessitates one forward-pass evaluation over labeled samples for every new task.
2. **Requires a small amount of labeled data**: While 20 labeled samples per class is modest, the method is not fully zero-shot.
3. **Limited scalability to large label spaces**: Evaluation is primarily conducted on tasks with 2–16 classes; the effectiveness of majority voting under much larger label spaces (e.g., 1,000 classes) remains unclear.
4. **Applicable only to selection-based classification**: SAVs are not directly applicable to tasks requiring generated responses (e.g., open-ended VQA).
5. **Insufficient theoretical analysis**: The mechanisms by which a small number of heads encode classification-relevant information, and the relationship between these heads and other model capabilities (e.g., generation), are not thoroughly analyzed.

## Related Work & Insights

- **Distinction from task vectors**: Task vectors are used to enhance generative capabilities, whereas SAVs directly employ attention vectors as classification features.
- **Complementarity with CLIP**: CLIP extracts unimodal features, while SAVs extract multimodal features that jointly encode image and text.
- **Broader implication**: Generative models may harbor a wealth of untapped discriminative features; systematically mining such features could represent an important research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Proposes a novel paradigm for extracting discriminative features from generative model attention heads; the approach is elegant and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 15+ datasets across four task categories (safety, VQA, retrieval, classification) with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear and contextual motivation is well-developed.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new pathway for leveraging generative LMMs on discriminative tasks, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](../../CVPR2025/multimodal_vlm/generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)

</div>

<!-- RELATED:END -->
