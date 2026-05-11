---
title: >-
  [Paper Note] Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens
description: >-
  [CVPR 2026][Multimodal VLM][discrete diffusion model] This paper proposes CubiD, the first model to perform discrete diffusion generation over high-dimensional representation tokens (768-dim). By conducting fine-grained…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "discrete diffusion model"
  - "high-dimensional representation token"
  - "visual generation"
  - "dimension-wise quantization"
  - "unified multimodal"
date: 2026-05-08
content_hash: 54f111f5b9b392fc
---

# Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens

**Conference**: CVPR 2026
**arXiv**: [2603.19232](https://arxiv.org/abs/2603.19232)
**Code**: [GitHub](https://github.com/YuqingWang1029/CubiD)
**Area**: Multimodal VLM
**Keywords**: discrete diffusion model, high-dimensional representation token, visual generation, dimension-wise quantization, unified multimodal

## TL;DR

This paper proposes CubiD, the first model to perform discrete diffusion generation over high-dimensional representation tokens (768-dim). By conducting fine-grained mask prediction over an $h \times w \times d$ cubic tensor, CubiD achieves high-quality image generation while preserving visual understanding capability.

## Background & Motivation

**Demand for unified multimodal modeling**: Language models naturally leverage semantic tokens for both understanding and generation. However, visual models remain fragmented—understanding relies on high-dimensional semantic features while generation relies on low-dimensional compressed tokens (8–32 dim), impeding unified architectures.

**Reconstruction advantage of high-dimensional representations**: Recent work (e.g., RAE) demonstrates that pretrained features of 768–1024 dimensions enable high-quality reconstruction, yet discrete generative modeling over such representations poses fundamental challenges.

**Failure of vector quantization in high-dimensional spaces**: Conventional VQ suffers from the curse of dimensionality in high-dimensional spaces—data sparsity renders clustering ineffective, codebook size must grow exponentially, and quantization-induced feature shift severely degrades semantic information.

**Feasibility of dimension-wise quantization**: Quantizing each dimension independently circumvents the difficulty of joint quantization. As a training-free method, it can be directly applied to frozen pretrained features; however, generative modeling remains a bottleneck.

**Limitations of existing generative approaches**: Autoregressive generation requires $O(hwd)$ steps, which is infeasible; standard discrete diffusion cannot model intra-position dimensional dependencies.

**Core insight**: The $h \times w \times d$ tensor has a natural multi-dimensional structure that can break the atomicity constraint of spatial positions, enabling flexible operations across the full three-dimensional space.

## Method

### Overall Architecture

CubiD consists of two stages: (1) high-dimensional token discretization—features are extracted using frozen pretrained encoders (DINOv2/SigLIP2) followed by dimension-wise quantization; (2) Cubic Discrete Diffusion—fine-grained mask modeling and iterative generation over a discrete $h \times w \times d$ tensor.

### Key Designs

#### Dimension-wise Quantization

Each continuous value is independently quantized into $L$ discrete levels: $q_{x,y,i} = \text{Quantize}(z_{x,y,i}; L)$. Using $L=8$ for DINOv2 and $L=16$ for SigLIP2 achieves reconstruction quality on par with continuous features. On LLaVA understanding benchmarks, dimension-wise quantization (DQ) incurs nearly no degradation (GQA: 63.1 vs. 63.2), whereas vector quantization (VQ) degrades severely (54.9).

#### Fine-grained Cubic Masking

Unlike MaskGIT, which masks entire spatial positions, CubiD independently masks arbitrary elements within the $h \times w \times d$ tensor. During training, the masking ratio is sampled from a truncated Gaussian distribution $r \sim \text{TruncNorm}(\mu=1.0, \sigma=0.10, [0,1])$; selected elements are replaced with a learnable [MASK] token. The model is trained with a cross-entropy loss to predict masked tokens:
$$\mathcal{L} = -\mathbb{E}\left[\sum_{i \in \mathbf{M}} \log p(q_i | \mathbf{q}_{\bar{\mathbf{M}}})\right]$$

#### Model Architecture

A standard bidirectional attention Transformer is adopted. For each spatial position, the $d$ tokens are dequantized and concatenated into a $d$-dimensional vector as a single token, keeping the sequence length fixed at $h \times w$ regardless of feature dimensionality. An MLP prediction head produces $d \times L$ logits as output.

### Loss & Training

Cross-entropy loss (Eq. 3) is computed over all masked positions. Inference employs iterative unmasking with a cosine schedule, completing generation in a fixed $T$ steps.

## Key Experimental Results

### Main Results: ImageNet 256×256 Generation

| Method | Dim | Params | gFID↓ (w/o cfg) | IS↑ | gFID↓ (w/ cfg) |
|--------|-----|--------|-----------------|-----|----------------|
| MaskGIT | 16 | 227M | 6.18 | 182.1 | 4.02 |
| CubiD-L (Ours) | 768 | 946M | 5.25 | — | — |
| CubiD-XXL (Ours) | 768 | 3.7B | 4.68 | — | 1.88 |

### Ablation Study

| Ablation | Setting | gFID↓ |
|----------|---------|-------|
| Masking strategy | Per-dim / Per-spatial / **Per-element** | 120.03 / 22.22 / **5.33** |
| Mask token | Fixed / Random / **Learned** | 5.56 / 56.38 / **5.33** |
| Model scale | 946M / 1.4B / **3.7B** | 5.25 / 4.91 / **4.68** |
| Inference steps | 64 / 256 / 512 | 9.14 / 5.33 / 5.25 |

### Key Findings

- Element-wise masking is critical: Per-dim masking completely fails (gFID=120) and Per-spatial masking produces blurry results (gFID=22), demonstrating that intra- and inter-position dependencies within high-dimensional tokens are inseparable.
- Dimension-wise quantization preserves understanding capability: DQ achieves near-identical performance to continuous features across four LLaVA benchmarks.
- The model exhibits favorable scaling behavior from 900M to 3.7B parameters.
- The approach generalizes across encoders: both DINOv2 (gFID=5.25) and SigLIP2 (gFID=5.87) are effective.

## Highlights & Insights

- **First** work to achieve discrete generation over high-dimensional representation tokens, bridging unified representations for understanding and generation.
- The fine-grained cubic masking design is elegant, transforming the intractable $O(hwd)$-step problem into parallel iterative generation completed in a fixed $T$ steps.
- Experiments verify that discretized high-dimensional tokens can simultaneously serve both understanding and generation tasks.
- Ablation studies thoroughly demonstrate the necessity of each design choice.

## Limitations & Future Work

- Validation is currently limited to class-conditional generation on ImageNet; text-guided generation has not been explored.
- An external decoder (from RAE) is required to reconstruct images from representations.
- Inference still requires hundreds of steps, leaving room for efficiency improvements.
- An in-depth comparison with state-of-the-art continuous diffusion models (e.g., DiT) in terms of FID is lacking.

## Related Work & Insights

- The key distinction from MaskGIT lies in masking granularity: CubiD operates at the dimension level rather than the spatial position level.
- CubiD is complementary to RAE: RAE generates high-dimensional representations via continuous diffusion, while CubiD uses discrete diffusion.
- The essential difference from low-dimensional discrete generation methods such as TiTok is that CubiD operates directly in the native dimensionality of pretrained features, preserving semantic integrity.
- This work lays the groundwork for unified multimodal architectures where a single set of discrete tokens serves both understanding and generation.
- The success of dimension-wise quantization offers an important insight for the VQ-VAE community—joint quantization is not necessary in high-dimensional spaces.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](../../ICCV2025/multimodal_vlm/musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[CVPR 2026\] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)

</div>

<!-- RELATED:END -->
