---
title: >-
  [Paper Note] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs
description: >-
  [Image Generation] FineLIP enables CLIP models to handle long text descriptions and perform fine-grained visual-textual matching through positional embedding stretching (77 to 248 tokens), an Adaptive Token Refinement Module (ATRM), and a Cross-modal Late Interaction Module (CLIM). It significantly outperforms existing methods such as Long-CLIP and TULIP on long-description retrieval tasks.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 28c812699f339234
---

# FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs

## TL;DR

FineLIP enables CLIP models to handle long text descriptions and perform fine-grained visual-textual matching through positional embedding stretching (77 to 248 tokens), an Adaptive Token Refinement Module (ATRM), and a Cross-modal Late Interaction Module (CLIM). It significantly outperforms existing methods such as Long-CLIP and TULIP on long-description retrieval tasks.

## Background & Motivation

- **Two Key Limitations of CLIP**:
  1. **Text length limitation**: The CLIP text encoder can process at most 77 tokens, preventing it from encoding detailed, rich, long descriptions.
  2. **Global feature alignment**: Traditional CLIP only aligns global visual and text features, neglecting local fine-grained information.
- **Demand for long descriptions**: With the advancement of LVLMs (e.g., GPT-4V, LLaVA), detailed image descriptions far exceeding 77 tokens can be generated, containing rich information like color, position, and spatial relations. Existing methods fail to fully utilize these details.
- **Limitations of Prior Work**:
    - **Long-CLIP**: Extends to 248 tokens but only uses global feature alignment, overlooking local details.
    - **TULIP**: Introduces relative positional encodings but is also focused on the global level.
    - **DreamLIP**: Decomposes long descriptions into multiple short descriptions without directly processing long text.
    - **FILIP/SPARC**: Incorporate fine-grained alignment but are only designed for short descriptions.
- **Key Insight**: Simultaneously addressing both long text encoding and fine-grained alignment is essential, as neither can be omitted.

## Method

### Overall Architecture

FineLIP enhances the pretrained CLIP model through three main steps:
1. Stretching positional embeddings to support long text inputs.
2. Adaptive Token Refinement Module (ATRM) to aggregate visual and textual tokens.
3. Cross-modal Late Interaction Module (CLIM) to achieve token-level fine-grained alignment.

### Key Designs

#### 1. Positional Embedding Stretching

- Preserves the first 20 positional embeddings (experiments show they are already fully trained during pretraining).
- Conducts $4\times$ adaptive interpolation stretching on the 21st to 77th positional embeddings.
- Final length: $20 + (77 - 20) \times 4 = 248$ tokens.
- Advantage: Preserves the cross-modal alignment capability of the pretrained weights, avoiding training from scratch.

#### 2. Adaptive Token Refinement Module (ATRM)

- **Design Motivation**: Local tokens in the last layer of the Transformer can be ambiguous; direct token-level alignment yields suboptimal results.
- **Strategy**: **Aggregation is superior to selection**—token selection loses information, while aggregation retains all information.
- **Mechanism**:
    - Input $N$ tokens $\rightarrow$ output $N'$ refined tokens ($N'/N = 0.2$, default aggregation ratio of 20%).
    - Transformation matrix $W_{ref} \in \mathbb{R}^{N' \times N}$, learned via a self-attention-like mechanism:
    $$W_{ref} = \text{SoftMax}\left(\frac{W_q \sigma(X W_k)^T}{\tau}\right)$$
    - $\tau$ is a learnable temperature parameter that encourages sparse attention.
- **Features**: Applies aggregation to both visual and textual branches (with independent parameters for each) rather than only refining the visual side.
- Preserves global [CLS] and [EOS] tokens, excluding them from aggregation.

#### 3. Cross-modal Late Interaction Module (CLIM)

- Computes the cosine similarity between the refined visual token $v'_i$ and text token $t'_j$.
- Bidirectional MaxSim pooling:
$$R(I,T) = \frac{1}{P'}\sum_{i=1}^{P'}\max_j S(v'_i, t'_j) + \frac{1}{M'}\sum_{j=1}^{M'}\max_i S(t'_i, v'_j)$$
- Retains global tokens ([CLS]/[EOS]) for alignment, enabling cross-granularity hybrid alignment.

### Loss & Training

Adopts **Triplet Marginal Loss** (rather than traditional contrastive loss) with a margin $\alpha = 0.2$:
$$\mathcal{L}_{triplet} = \mathcal{L}_{i2t} + \mathcal{L}_{t2i}$$
$$\mathcal{L}_{i2t} = \max(0, R(I_q, T^-) - R(I_q, T^+) + \alpha)$$

This ensures that the similarity of positive sample pairs exceeds that of negative sample pairs by at least a margin of $\alpha$.

## Key Experimental Results

### Main Results (Urban1k + DOCCI Retrieval)

**B/16 Model on Urban1k**:

| Method| I2T R@1 | I2T R@5 | T2I R@1 | T2I R@5 |
|------|---------|---------|---------|---------|
| Baseline | 0.859 | 0.969 | 0.866 | 0.963 |
| Long-CLIP | 0.789 | - | 0.795 | - |
| TULIP | 0.881 | - | 0.866 | - |
| SPARC | 0.854 | 0.963 | 0.853 | 0.957 |
| LAPS | 0.890 | 0.987 | 0.884 | 0.971 |
| **FineLIP** | **0.907** | **0.983** | **0.893** | **0.975** |
| **FineLIP*** | **0.912** | **0.985** | **0.900** | **0.977** |

The improvement is even more significant on the **L/14 model**, where FineLIP* achieves an I2T R@1 of 0.940.

### Ablation Study (Tab. 3)

Validates the necessity of the following components:
- A) Positional embedding stretching vs. training from scratch $\rightarrow$ stretching is significantly superior.
- B) ATRM aggregation ratio of 0.2 is optimal.
- C) Dual-branch aggregation (visual + textual) is superior to visual-only aggregation.
- D) Triplet Loss outperforms Contrastive Loss.

### Key Findings

1. FineLIP comprehensively outperforms Long-CLIP (+12% I2T R@1) and TULIP on all long-description retrieval tasks.
2. The contribution of token refinement in the textual branch is significant—only refining the visual side yields limited performance, while dual-branch refinement maximizes the gain.
3. An aggregation ratio of 0.2 ($5\times$ compression) achieves the optimal balance between performance and efficiency.
4. Triplet Loss is more suitable for fine-grained retrieval scenarios than Contrastive Loss.
5. In text-to-image generation tasks, the text encoder of FineLIP also demonstrates superior FID and CLIP-Score.

## Highlights & Insights

1. **Dual-branch token refinement**: Unlike methods that only focus on visual tokens (FILIP/SPARC), FineLIP also aggregates textual tokens to eliminate the ambiguity of the original raw text tokens.
2. **Cross-granularity hybrid alignment**: Retains the global [CLS]/[EOS] tokens in the refined set, allowing global-local information to interact within the same framework.
3. **Low overhead, high reward**: The parameter count of ATRM is very small (only the $W_q$ and $W_k$ projection matrices), and it reduces the number of subsequent tokens, thus improving efficiency.
4. **Universal enhancement**: FineLIP can be plugged into any CLIP variant (B/16, L/14), yielding consistent improvements.

## Limitations & Future Work

1. **248-token ceiling**: Although extended from 77 to 248, it may still be insufficient for extremely long (e.g., paragraph-level) descriptions.
2. **Training data requirements**: Requires long-description datasets (e.g., ShareGPT4V, DOCCI), which are expensive to construct.
3. **Zero-shot classification**: The paper mainly evaluates retrieval and generation tasks, and does not report zero-shot classification benchmarks.
4. **Aggregation strategy**: The current linear aggregation might lose spatial information in extremely fine-grained scenarios (e.g., small object detection).

## Related Work & Insights

- **Long-CLIP (2024)**: Pioneer of positional embedding stretching $\rightarrow$ integrated and extended in this work.
- **FILIP (NeurIPS'21)**: Token-level similarity alignment $\rightarrow$ performs better after incorporating token aggregation in this work.
- **ColBERT (Information Retrieval)**: Late interaction mechanism $\rightarrow$ inspires the MaxSim pooling design of CLIM.
- **Insights**: Improvements in vision-language models rely not only on scaling up, but more importantly, on refining the granularity of alignment and the way information is utilized.

## Rating

⭐⭐⭐⭐ — The method is simple and elegant; the combination of dual-branch aggregation and fine-grained alignment is highly practical. The performance improvement in long-description scenarios is significant, consistent, and broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)
- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

</div>

<!-- RELATED:END -->
