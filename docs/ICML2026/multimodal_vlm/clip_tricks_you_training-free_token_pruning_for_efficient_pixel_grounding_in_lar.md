---
title: >-
  [Paper Note] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models
description: >-
  [ICML2026][Multimodal VLM][Visual token pruning] This study identifies a counter-intuitive "similarity reversal" phenomenon in CLIP, where visual tokens of referred regions exhibit low similarity with text [EOS] tokens.…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Visual token pruning"
  - "CLIP similarity reversal"
  - "Pixel-level grounding"
  - "Training-free inference acceleration"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: e42e053a1f4ec5d4
---

# CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models

**Conference**: ICML2026  
**arXiv**: [2605.13178](https://arxiv.org/abs/2605.13178)  
**Code**: https://github.com/sejong-rcv/LiteLVLM  
**Area**: Multimodal VLM  
**Keywords**: Visual token pruning, CLIP similarity reversal, Pixel-level grounding, Training-free inference acceleration, Large Vision-Language Models  

## TL;DR

This study identifies a counter-intuitive "similarity reversal" phenomenon in CLIP, where visual tokens of referred regions exhibit low similarity with text [EOS] tokens. Based on this, LiteLVLM is proposed—a training-free text-guided visual token pruning method. It retains 90.3% of original pixel grounding performance while removing 66.7% of tokens, achieving 22% inference acceleration and 2.3× memory savings.

## Background & Motivation

**Background**: The inference overhead of Large Vision-Language Models (LVLMs) primarily stems from visual tokens—LLaVA generates 576 tokens, while high-resolution versions like LLaVA-NeXT reach 2880. Thus, visual token pruning is critical for enhancing inference efficiency.

**Limitations of Prior Work**: Existing pruning methods (LLaVA-PruMerge, VisionZip, FastV) rely on text-agnostic global importance metrics. While effective for image understanding, they perform poorly in pixel-level grounding tasks. In grounding, token importance depends heavily on the input text (e.g., "player" vs. "ball" requires different spatial regions), which global metrics fail to capture.

**Key Challenge**: The only method utilizing text information, TRIM, retains tokens based on high CLIP vision-text similarity. However, the authors discovered a **similarity reversal**—visual tokens in referred regions actually have the lowest similarity with text [EOS] tokens. Consequently, TRIM retains irrelevant tokens, performing 10-20% worse than random pruning.

**Goal**: Design a text-guided, training-free token pruning strategy to significantly reduce visual tokens while maintaining segmentation accuracy in pixel-level grounding tasks.

**Key Insight**: The authors analyzed the CLIP [EOS] token attention distribution and discovered a **text attention sink** phenomenon—68% of the [EOS] token's attention is concentrated on the non-semantic [SOS] token, with only 14% assigned to the semantic-carrying referring word ([RES] token). This causes the [EOS] embedding to lack rich semantics, leading to abnormally low similarity with referred region visual tokens.

**Core Idea**: Reverse the CLIP vision-text similarity ranking—retain low-similarity tokens (covering referred regions) and recover high [CLS]-contribution tokens (providing background context) to achieve clear foreground-background separation.

## Method

### Overall Architecture

LiteLVLM is built upon the GLaMM architecture: an input image is encoded by CLIP ViT-L/14 into 576 visual tokens, and input text is processed by a CLIP text encoder to extract the [EOS] embedding. LiteLVLM performs pruning before visual tokens enter the LLM. Filtered tokens pass through an MLP projection layer into Vicuna-7B to generate a <SEG> grounding embedding, which is then used by a SAM-style pixel decoder to output a binary segmentation mask. Since pruning occurs before the LLM, it is fully compatible with efficient attention mechanisms like FlashAttention.

### Key Designs

1. **Similarity-aware Token Selection**:
    - **Function**: Retain foreground visual tokens spatially aligned with the referred object.
    - **Mechanism**: Compute the dot-product similarity $s_{i,j} = E_i^{\mathcal{T}} \cdot E_j^{v\top}$ between each visual token embedding $E_j^v$ and the [EOS] text embedding $E_i^{\mathcal{T}}$, then select the $k$ tokens with the **lowest** similarity. These tokens are weakly aligned with global semantics but retain rich local referral information.
    - **Design Motivation**: CLIP contrastive learning only aligns the global [CLS] and [EOS] tokens. Region tokens receive weaker gradient signals due to low attention weights, resulting in minimum similarity with [EOS]. Reversing the ranking turns this "defect" into a precise foreground localization tool.

2. **Context-aware Token Recovery**:
    - **Function**: Recover background and global context tokens to assist in foreground-background boundary determination.
    - **Mechanism**: For the set of unselected tokens $S$, compute the context contribution score $s_i' = \| \frac{\exp(Q^{\mathcal{I}} K_i^\top / \sqrt{d})}{\sum_{j \in S} \exp(Q^{\mathcal{I}} K_j^\top / \sqrt{d})} \cdot V_i \|_2$ for each token relative to [CLS]. Select tokens with the highest scores. This metric considers both attention weights and the information content (L2 norm) of the Value vector.
    - **Design Motivation**: Purely foreground tokens lack background references, leading to blurry segmentation boundaries. Context tokens provide global semantic and spatial references, enabling the pixel decoder to distinguish boundaries clearly.

3. **Adaptive Token Selection**:
    - **Function**: Dynamically adjust the ratio of foreground to background tokens based on input text.
    - **Mechanism**: Given a budget $\mathcal{B}$ and $N$ input text queries, first identify $\mathcal{B}$ low-similarity candidates $\mathcal{S}_i$ for each query. The intersection $\mathcal{S} = \bigcap_{i=1}^{N} \mathcal{S}_i$ is taken as the similarity-aware tokens; for $N=1$, it is empirically set to 50% of the budget. The remaining budget $\mathcal{B} - |\mathcal{S}|$ is filled via context-aware recovery.
    - **Design Motivation**: Fixed ratios (e.g., 75%/25%) perform inconsistently across scenarios. This adaptive strategy dynamically allocates based on text input intersection, outperforming the best fixed ratio by 2.0%.

## Key Experimental Results

### Main Results: Referring Expression Segmentation (Retaining 192/576 tokens, ↓66.7%)

| Method | RefCOCO-val | RefCOCO-testA | RefCOCO+-val | RefCOCOg-val | Avg. | Rel. |
|------|------------|---------------|-------------|-------------|------|------|
| GLaMM (Upper Bound) | 79.5 | 83.2 | 72.6 | 74.2 | 75.5 | 100% |
| TRIM | 55.5 | 58.5 | 40.6 | 45.2 | 47.3 | 62.6% |
| FastV | 69.5 | 73.9 | 57.9 | 61.5 | 62.6 | 82.9% |
| LLaVA-PruMerge | 68.8 | 74.6 | 58.2 | 61.2 | 63.1 | 83.5% |
| VisionZip | 71.1 | 76.4 | 59.7 | 64.7 | 65.5 | 86.7% |
| VisPruner | 72.4 | 75.5 | 61.5 | 65.4 | 66.0 | 87.4% |
| **LiteLVLM** | **74.4** | **78.7** | **64.1** | **66.0** | **68.1** | **90.3%** |

### Ablation Study (RefCOCO-val, Different Token Budgets)

| Config | 29 tokens | 58 tokens | 144 tokens | 288 tokens | Avg. | Note |
|------|----------|----------|-----------|-----------|------|------|
| Random Pruning | 57.2 | 62.7 | 68.2 | 72.0 | 65.0 | Baseline |
| Context-only Recovery | 54.5 | 59.9 | 67.4 | 71.6 | 63.3 | Worse than random |
| High Similarity Selection (↑) | 48.1 | 49.4 | 53.8 | 59.5 | 52.7 | Severe degradation |
| Low Similarity Selection (↓) | 59.8 | 63.6 | 68.4 | 72.8 | 66.1 | Verifies reversal |
| **LiteLVLM (↓ + Recovery)** | **62.5** | **65.2** | **72.9** | **75.2** | **68.9** | Complementary |

### Efficiency Analysis (192 tokens / 64 tokens)

| Method | FLOPs (TB) | Prefill (ms) | CUDA Time (ms) | Memory (GB) |
|------|-----------|-------------|----------------|----------|
| GLaMM (Upper Bound) | 4.66 | 166.25 | 340.89 | 0.81 |
| FastV (192) | 2.65 | 162.88 | 340.25 | 0.81 |
| VisPruner (192) | 2.17 | 75.65 | 276.09 | 0.37 |
| **LiteLVLM (192)** | **2.11** | **74.88** | **265.83** | **0.35** |
| **LiteLVLM (64)** | **1.27** | **54.02** | **237.35** | **0.21** |

### Key Findings

- TRIM (selecting by high similarity) is the worst performer under all budgets, confirming the universality of the CLIP similarity reversal phenomenon.
- Adaptive token selection performs 2.0% better than the best fixed ratio (50/50), indicating the importance of dynamic allocation for varied text inputs.
- In video grounding (Ref-DAVIS-17 / Refer-YouTube-VOS), retaining 196/576 tokens results in only a 0.5% performance loss, outperforming VisPruner by 4.9-6.7%.
- Pruning before the LLM reduces prefilling time from 166ms to 75ms and ensures compatibility with FlashAttention; FastV, which prunes inside the LLM, cannot benefit.

## Highlights & Insights

- **Discovery of CLIP similarity reversal**: Contrastive learning only aligns [CLS]/[EOS] global tokens, leaving local tokens anti-correlated with global representations due to weak gradient signals. This not only explains TRIM's failure but also provides a new perspective on CLIP's internal representations.
- **Text attention sink phenomenon**: [EOS] directs 68% of its attention to the non-semantic [SOS] position, similar to attention sinks in LLMs, suggesting a universal impact of positional bias on special tokens in Transformers.
- **"Reverse Thinking" methodology**: When intuition (high similarity = importance) fails, reversing the ranking converts a CLIP "flaw" into a precise localization tool. This approach could be transferred to other tasks relying on CLIP for spatial selection (e.g., regional selection in open-vocabulary detection).

## Limitations & Future Work

- The method relies heavily on LVLM architectures using CLIP as the vision encoder; whether similarity reversal holds for models using SigLIP or other non-contrastive encoders requires verification.
- Generalization has been tested on GLaMM/VideoGLaMM; further validation on other grounding architectures like LISA or PixelLM is needed.
- The empirical 50% ratio for $N=1$ in the adaptive strategy lacks theoretical grounding; more refined budget allocation strategies may further improve performance.
- Optimization is currently focused on the inference phase; the potential for token pruning during training remains unexplored.

## Related Work & Insights

- **LLaVA-PruMerge (ICCV25)**: Text-agnostic pruning based on [CLS] token similarity; effective for image understanding but fails at grounding.
- **TRIM (COLING25)**: The only prior text-guided method, but incorrectly selects based on high similarity, serving as a counter-example.
- **VisPruner (ICCV25)**: Current SOTA token pruning method; LiteLVLM surpasses it by 2.9-10.2% across all settings.
- **FastV (ECCV24)**: Performs pruning inside the LLM based on attention, hindering compatibility with FlashAttention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[CVPR 2026\] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](../../CVPR2026/multimodal_vlm/clip-free_label_free_unsupervised_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)

</div>

<!-- RELATED:END -->
