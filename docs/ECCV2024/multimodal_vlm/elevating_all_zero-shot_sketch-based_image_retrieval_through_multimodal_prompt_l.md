---
title: >-
  [Paper Note] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning
description: >-
  [ECCV2024][Multimodal VLM][CLIP] SpLIP is proposed, a bidirectional multimodal prompt learning framework based on frozen CLIP. By utilizing bidirectional knowledge exchange between vision and text encoders, an adaptive-margin triplet loss, and a conditional cross-modal jigsaw task, it achieves SOTA performance across three sketch retrieval settings: ZS-SBIR, GZS-SBIR, and FG-ZS-SBIR.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "CLIP"
  - "Sketch-Based Image Retrieval (SBIR)"
  - "Multimodal Prompt Learning"
  - "Zero-Shot Learning"
  - "Cross-Modal Alignment"
date: 2026-05-08
content_hash: 099f7d50d2b4085a
---

# Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning

**Conference**: ECCV2024  
**arXiv**: [2407.04207](https://arxiv.org/abs/2407.04207)  
**Code**: [Project Page](https://mainaksingha01.github.io/SpLIP/)  
**Area**: Multimodal VLM  
**Keywords**: CLIP, Sketch-Based Image Retrieval (SBIR), Multimodal Prompt Learning, Zero-Shot Learning, Cross-Modal Alignment

## TL;DR

SpLIP is proposed, a bidirectional multimodal prompt learning framework based on frozen CLIP. By utilizing bidirectional knowledge exchange between vision and text encoders, an adaptive-margin triplet loss, and a conditional cross-modal jigsaw task, it achieves SOTA performance across three sketch retrieval settings: ZS-SBIR, GZS-SBIR, and FG-ZS-SBIR.

## Background & Motivation

### Task Definition

Sketch-Based Image Retrieval (SBIR) aims to retrieve corresponding photos from a gallery based on hand-drawn sketches. The core challenge lies in the huge domain gap between sketches and photos. This task has three main settings:

- **ZS-SBIR**: The testing categories are completely unseen during training.
- **GZS-SBIR**: During inference, the gallery contains both seen and unseen category photos, which easily biases the model towards seen classes.
- **FG-ZS-SBIR**: Fine-grained matching is performed at the instance level rather than the category level.

### Limitations of Prior Work

**Unimodal Prompt Learning**: Existing CLIP-based SBIR methods (such as CLIP-AT, TLT) mainly use unimodal (visual) prompts, failing to fully exploit the collaborative capacity of CLIP's dual vision-text pathways.

**Limitations of Multimodal Prompts**: Existing multimodal prompt methods (like MaPLe, PromptSRC) only support unidirectional token sharing, making the text pathway insensitive to visual information and limiting semantic depth.

**Coarse Jigsaw Strategies**: The patch shuffling strategy in CLIP-AT uses the same or different permutations for positive and negative pairs to perform triplet learning, failing to effectively establish local-to-global correspondences.

### Motivation

A **bidirectional** cross-modal knowledge exchange mechanism is required to build tighter synergy between CLIP's vision and text encoders, while introducing a conditional cross-modal jigsaw task to enhance fine-grained alignment.

## Method

### Overall Architecture

SpLIP is based on frozen CLIP (ViT-B/32) and consists of the following core components:

1. **Vision-guided Textual Prompting**
2. **Text-guided Visual Prompting**
3. **Conditional Cross-modal Jigsaw Solver**
4. **Adaptive margin triplet loss**

During training, only the LayerNorm parameters, three mapping modules ($\mathcal{B}_t, \mathcal{B}_v, \mathcal{B}_{vt}$), and the jigsaw solver $\mathcal{F}_{js}$ are optimized, while the CLIP backbone remains completely frozen.

### Key Designs

#### Key Design 1: Bidirectional Prompt Sharing

**Vision $\rightarrow$ Text Direction**: The image patch embeddings $\mathbf{E}_0$ are converted into $m=4$ learnable text tokens $\mathbf{T}$ via the mapping block $\mathcal{B}_t$, which are injected into every layer of the text encoder $\mathcal{F}_t$. Distinct from the random initialization in methods like MaPLe, this textual prompt directly captures the visual distribution.

**Text $\rightarrow$ Vision Direction (Dual Channels)**:
- **Channel 1**: Maps textual tokens ("sketch/photo of a", excluding [CLS]) to $\mathcal{J}-1$ visual prompt tokens $\mathbf{V}^{\text{tg}}$ via $\mathcal{B}_v$, which remain identical across all layers.
- **Channel 2**: Converts the layer output $\mathbf{W}_l$ of the text encoder (containing tokens of **all training classes**) into $n=2$ **layer-wise varying** visual prompts $\mathbf{V}^{\text{ms}}$ via $\mathcal{B}_{vt}$.

Key Innovation: $\mathbf{V}^{\text{ms}}$ aggregates textual information across all training categories (category-agnostic). This cross-class knowledge sharing effectively bridges the semantic discrepancy within the embedding space.

#### Key Design 2: Conditional Cross-modal Jigsaw Task

Given an anchor sketch $s_a$ and its permuted version $s_a'$, a positive photo $p_a^+$ (same class/instance) and a negative photo $p_a^-$:

- Construct fused features: $r = [\mathcal{F}_v(s_a), \mathcal{F}_v(s_a')]$, $r^+ = [\mathcal{F}_v(p_a^+), \mathcal{F}_v(s_a')]$, $r^- = [\mathcal{F}_v(p_a^-), \mathcal{F}_v(s_a')]$
- The jigsaw solver $\mathcal{F}_{js}$ (consisting of a 2-layer Transformer encoder + classifier) is required to predict the permutation index.
- Core Idea: A positive photo should facilitate solving the sketch's jigsaw task more effectively than a negative photo, as positive photos share the same spatial layouts with the sketch.

Difference from CLIP-AT: CLIP-AT enforces positive pairs to share the same permutations and negative pairs to use different permutations, whereas SpLIP pairs the permuted sketch with the **unpermuted** positive photo, thereby capturing local-to-global correspondences more effectively.

#### Key Design 3: Adaptive Margin Triplet Loss

Traditional triplet loss uses a fixed margin, whereas SpLIP dynamically computes the margin leveraging class name embeddings from the CLIP text encoder:

$$\mu(c^+, c^-) = \cos(\mathcal{F}_t(c^+), \mathcal{F}_t(c^-))$$

When positive and negative classes are semantically close (high cosine similarity), the margin is larger, forcing the model to put more effort into distinguishing similar categories.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{triplet} + \alpha \cdot \mathcal{L}_{class} + \beta \cdot \mathcal{L}_{cjs}$$

- $\mathcal{L}_{triplet}$: Cross-modal triplet loss with adaptive margin
- $\mathcal{L}_{class}$: Image-text classification loss (cross-entropy) based on CLIP textual prompts
- $\mathcal{L}_{cjs}$: Conditional jigsaw loss = jigsaw cross-entropy + hinge loss (constraining the positive pair to exhibit lower jigsaw CE than the negative pair)

## Key Experimental Results

### Main Results: Category-level ZS-SBIR (Table 1)

| Method | Backbone | Sketchy-1 mAP | Sketchy-2 mAP@200 | TU-Berlin mAP | QuickDraw mAP |
|------|------|:---:|:---:|:---:|:---:|
| BDA | CNN | 43.7 | 55.6 | 37.4 | 15.4 |
| PSKD | ViT | 68.8 | 56.0 | 50.2 | 15.0 |
| ZSE-Ret | ViT | 73.6 | 50.4 | 56.9 | 14.2 |
| CLIP-AT | CLIP | - | 72.3 | 65.1 | 20.2 |
| TLT | CLIP | 77.9 | 66.1 | 61.5 | 27.8 |
| MARL | CLIP | - | 69.1 | 70.5 | 32.7 |
| **SpLIP** | **CLIP** | **80.2** | **76.4** | **73.1** | **34.2** |

### GZS-SBIR Results (Table 2)

| Method | Sketchy-2 mAP@200 | Sketchy-2 P@200 | TU-Berlin mAP | TU-Berlin P@100 |
|------|:---:|:---:|:---:|:---:|
| STL (AAAI'23) | 63.4 | 53.8 | 40.2 | 49.8 |
| CLIP-AT | 55.6 | 62.7 | 60.9 | 63.8 |
| MARL | 62.3 | 68.5 | 62.6 | 67.8 |
| **SpLIP** | **68.2** | **74.5** | **66.7** | **70.3** |

SpLIP outperforms the second-best method on GZS-SBIR by **+4.8%** (Sketchy) and **+4.1%** (TU-Berlin) in mAP.

### FG-ZS-SBIR Results (Table 3, Sketchy Dataset)

| Method | Acc@1 | Acc@5 |
|------|:---:|:---:|
| CLIP-AT | 28.68 | 62.34 |
| MARL | 29.96 | 58.53 |
| **SpLIP** | **33.45** | **66.71** |

SpLIP exceeds CLIP-AT by nearly **+5%** and MARL by **+3.5%** on Top-1 accuracy.

### Cross-Dataset Generalization (Table 4, Train on Sketchy-Ext, Test on Others)

| Method | TU-Berlin mAP | TU-Berlin P@100 | QuickDraw mAP | QuickDraw P@100 |
|------|:---:|:---:|:---:|:---:|
| CLIP-AT | 56.4 | 63.1 | 30.7 | 45.0 |
| **SpLIP** | **70.6** | **76.0** | **45.8** | **58.6** |

Significant improvements are observed in cross-dataset scenarios: **+14.2%** mAP on TU-Berlin and **+15.1%** mAP on QuickDraw.

### Ablation Study: Loss Functions (Table 5, Sketchy)

| Loss Combination | ZS mAP@200 | ZS P@200 | FG Acc@1 | FG Acc@5 |
|---------|:---:|:---:|:---:|:---:|
| $\mathcal{L}_{class}$ | 57.5 | 58.1 | 17.23 | 39.41 |
| $\mathcal{L}_{triplet}$ | 71.6 | 72.7 | 26.54 | 59.92 |
| $\mathcal{L}_{class} + \mathcal{L}_{triplet}$ | 73.1 | 73.9 | 30.07 | 62.95 |
| + $\mathcal{L}_{margin}$ | 74.5 | 75.1 | 31.23 | 63.54 |
| + $\mathcal{L}_{cjs}$ (Full) | **76.4** | **77.3** | **33.45** | **66.71** |

### Ablation Study: Learnable Modules (Table 6, Sketchy)

| Configuration | ZS mAP@200 | FG Acc@1 |
|------|:---:|:---:|
| w/o LayerNorm | 74.2 | 31.24 |
| w/o $\mathcal{B}_v$ (No text $\rightarrow$ vision prompt) | 71.9 | 29.76 |
| w/o $\mathcal{B}_t$ (No vision $\rightarrow$ text prompt) | 72.8 | 30.41 |
| w/o $\mathcal{B}_{vt}$ (No cross-layer text $\rightarrow$ vision) | 70.5 | 28.92 |
| w/o All visual prompts | 68.8 | 26.54 |
| w/o All prompt modules | 62.5 | 28.49 |
| **SpLIP (Full)** | **76.4** | **33.45** |

### Key Findings

1. **Bidirectional prompt sharing is crucial**: Removing prompt sharing in either direction results in a drop of over 4% in ZS-SBIR and over 3% in FG-SBIR.
2. **$\mathcal{B}_{vt}$ is the most critical module**: Removing it drops the mAP by 5.9%, as it aggregates textual information across all training categories to the visual encoder.
3. **Adaptive margin outperforms fixed margin**: Dynamic $\mu$ significantly surpasses the fixed $\mu=0.2$ across all tasks.
4. **Conditional jigsaw outperforms CLIP-AT's jigsaw strategy**: Performance degrades noticeably when replacing it with the CLIP-AT approach.
5. **Deep prompts (layer-wise injection) outperform shallow prompts**: The mAP continuously improves as the number of participating layers increases.
6. **Strong cross-dataset generalization ability**: Gains of 14-15% mAP are achieved on unseen datasets.

## Highlights & Insights

1. **First work to apply multimodal prompt learning to SBIR**: Previous CLIP-SBIR methods only used visual prompts or simple fine-tuning.
2. **Elegant design of bidirectional information flow**: Vision $\rightarrow$ text (via $\mathcal{B}_t$) makes textual prompts context-aware of visual contents; text $\rightarrow$ vision (via $\mathcal{B}_v + \mathcal{B}_{vt}$) integrates semantic knowledge into the visual encoder, creating a closed loop.
3. **Category-agnostic knowledge aggregation**: $\mathcal{B}_{vt}$ aggregates text features of all training categories instead of just the current class; it does not require seeing test class names during inference, naturally supporting zero-shot scenarios.
4. **Clever refinement of the conditional jigsaw task**: Changing “permuted sketch vs. permuted photo” to “permuted sketch vs. whole photo” establishes better local-global correspondences.
5. **Adaptive margin leverages CLIP’s semantic structure**: Categories that are semantically close require greater margins to differentiate, which aligns with human intuition.

## Limitations & Future Work

1. **Limited to ViT-B/32**: Larger CLIP backbones (like ViT-L/14) have not been explored and might offer further performance gains.
2. **Training overhead**: Although the CLIP backbone is frozen, the bidirectional prompt sharing and the jigsaw solver introduce extra computations (multiple mapping modules and a Transformer decoder).
3. **Hyperparameters of the jigsaw task**: The selection of the permutation size $|\mathcal{Y}^{\text{perm}}|$ is not thoroughly discussed, which might significantly affect the fine-grained performance.
4. **FG-ZS-SBIR evaluated only on Sketchy**: Results on other fine-grained datasets (such as QMUL-Shoe/Chair) are missing.
5. **Inference speed not reported**: Multimodal prompt generation (especially for $\mathcal{B}_{vt}$ which needs to process all training classes) might lead to inference latency.
6. **Grid search for loss weights $\alpha$ and $\beta$**: Analysis on sensitivity across different datasets is absent.

## Related Work & Insights

### Comparison with MaPLe (CVPR'23)
Like SpLIP, MaPLe is a multimodal prompt learning method, but it only supports unidirectional token sharing from vision to text (initializing visual prompts with text prompts) and is limited to specific layers. SpLIP implements a bidirectional, all-layer knowledge exchange, which is more effective for SBIR.

### Comparison with CLIP-AT (CVPR'23)
CLIP-AT utilizes visual prompts combined with a patch shuffling triplet objective. SpLIP upgrades this in three dimensions: multimodal prompts (vs. unimodal), conditional jigsaw (vs. naive permutation matching), and adaptive margin (vs. fixed margin). SpLIP outperforms CLIP-AT by **14-15% mAP** in cross-dataset evaluations.

### Insights
- The **paradigm of bidirectional prompt sharing** can be extended to other cross-modal tasks (e.g., text-to-image generation, VQA).
- The **adaptive margin concept** can be applied to any metric learning based on CLIP semantic space.
- The **conditional jigsaw task** idea can be generalized to other scenarios requiring fine-grained alignment (e.g., cross-domain ReID, medical image registration).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Bidirectional multimodal prompt sharing + conditional jigsaw + adaptive margin, with three innovations working collaboratively.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three ZS-SBIR settings × four datasets + cross-dataset generalization + detailed ablations, highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear, formula notation is complete, and the illustrations are intuitive.
- **Value**: ⭐⭐⭐⭐ — Re-establishes a new adapter paradigm for CLIP in the SBIR domain, with the bidirectional prompt mechanism showing solid generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/multimodal_vlm/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ECCV 2024\] CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](clap_isolating_content_from_style_through_contrastive_learning_with_augmented_pr.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](../../CVPR2025/multimodal_vlm/visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[ECCV 2024\] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)

</div>

<!-- RELATED:END -->
