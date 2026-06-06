---
title: >-
  [Paper Note] Watermarking Autoregressive Image Generation
description: >-
  [NeurIPS 2025][LLM Safety][Autoregressive image generation] This paper is the first to adapt LLM watermarking (KGW green/red scheme) to the token level of autoregressive image generation models. It identifies and address…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Autoregressive image generation"
  - "watermarking"
  - "reverse cycle consistency"
  - "token-level watermarking"
  - "LLM watermark adaptation"
date: 2026-05-08
content_hash: d781d35259212db7
---

# Watermarking Autoregressive Image Generation

**Conference**: NeurIPS 2025
**arXiv**: [2506.16349](https://arxiv.org/abs/2506.16349)  
**Code**: [https://github.com/facebookresearch/wmar](https://github.com/facebookresearch/wmar)  
**Area**: Image Generation / AI Watermarking
**Keywords**: Autoregressive image generation, watermarking, reverse cycle consistency, token-level watermarking, LLM watermark adaptation

## TL;DR

This paper is the first to adapt LLM watermarking (KGW green/red scheme) to the token level of autoregressive image generation models. It identifies and addresses the key challenge of insufficient Reverse Cycle Consistency (RCC) through tokenizer–detokenizer fine-tuning and a watermark synchronization layer, achieving robust image watermark detection with theoretical guarantees.

## Background & Motivation

Autoregressive image generation models (DALL-E, Chameleon, RAR, etc.) discretize images into token sequences and generate them with Transformers, making them a significant alternative to diffusion models. However, no effective provenance tracking solution exists for their outputs.

**Limitations of existing watermarking schemes**:
- **Post-processing watermarks** (pixel modification): model-agnostic but vulnerable to adversarial attacks and lacking theoretical p-value guarantees.
- **Diffusion model watermarks**: designed specifically for diffusion-based generation and inapplicable to autoregressive models.
- **LLM watermarks** (KGW): effective on text tokens but never adapted to image tokens.

**Core challenge — Reverse Cycle Consistency (RCC)**: LLM watermark detection requires re-tokenizing generated content and checking the proportion of green tokens. For text, BPE tokenizers achieve very high RCC (token match ≈ 0.995). For **image VQ tokenizers**, however, the cycle of generated tokens → decoded image → re-encoded tokens changes approximately one-third of tokens (TM ≈ 0.66). This further degrades to 0.31 under JPEG compression and approaches zero under geometric transformations (flip, rotation). The root causes are: (1) VQ tokenizers are trained for forward cycle consistency (FCC), leaving decoded images off-manifold; and (2) spatial sensitivity causes even semantics-preserving edits to alter the majority of tokens.

## Method

### Overall Architecture

1. **Generation**: apply KGW watermarking directly to the autoregressive token sequence (adding $\delta$ to the logits of green tokens).
2. **Detection**: image → re-tokenize → count green tokens → compute p-value.
3. **Core improvements**: (a) fine-tune the detokenizer/encoder to improve RCC; (b) apply a watermark synchronization layer to handle geometric transformations.

### Key Designs

1. **RCC Fine-tuning** (Section 3.1):

    - The encoder $E$, quantizer $Q_C$, and codebook $C$ are kept frozen (to avoid retraining the autoregressive model).
    - Only the decoder $D$ and a separate encoder copy $E'$ (used exclusively at detection time) are fine-tuned.
    - **RCC loss**: $\mathcal{L}_{RCC}(s) = \mathbb{E}_{a \sim \mathcal{A}} \| \hat{z} - E'(a(D(\hat{z}))) \|_2^2$, targeting alignment of the soft latents after the decode–encode cycle with the original hard latents $\hat{z} = C_s$.
    - Data augmentations (JPEG, brightness, slight rotations, etc.) are sampled randomly during training to make RCC robust to valuemetric transformations.
    - **Regularization**: $\mathcal{L}_{reg} = \|D(\hat{z}) - D_0(\hat{z})\|_2^2 + \mathcal{L}_{LPIPS}$, preserving decoding quality.
    - Total loss: $\mathcal{L} = \mathcal{L}_{RCC} + \lambda \cdot \mathcal{L}_{reg}$

2. **Watermark Synchronization Layer** (Section 3.2):

    - Geometric transformations (flip, rotation) completely disrupt token correspondence and cannot be addressed by RCC fine-tuning alone.
    - Solution: leverage localized watermarking [Sander et al.] to embed four fixed 32-bit synchronization messages in the four image quadrants.
    - At detection time: search over a grid of rotation angles to find the optimal pair of orthogonal lines that best separates the four messages, thereby estimating and inverting the geometric transformation.
    - The token-level watermark detector is then applied to the rectified image to compute the p-value.

3. **Cross-modal Joint Detection**:

    - For mixed-modality outputs (e.g., interleaved text and images from Chameleon), scores $S^{(i)}$, $T^{(i)}$, and $h^{(i)}$ across samples are summed, deduplicated, and used to compute a unified p-value.
    - Joint detection across text and image tokens further improves detection confidence.

### Loss & Training

Training is performed on 50,000 ImageNet training images for 10 epochs. Training times: Taming: 22 h / 16 V100s; Chameleon: 2.5 h / 8 H200s; RAR-XL: 0.5 h / 8 H200s. Watermark parameters: $\delta=2$, $\gamma=0.25$.

## Key Experimental Results

### Main Results (TPR @ 1% FPR)

| Variant | No Transform | Valuemetric | Geometric | Adversarial | Neural Compression |
|---------|-------------|-------------|-----------|-------------|-------------------|
| Base | 0.99 | 0.26 | 0.01 | 0.43 | 0.48 |
| FT | 1.00 | 0.45 | 0.01 | 0.70 | 0.71 |
| FT+Augs | 1.00 | **0.92** | 0.01 | 0.70 | 0.79 |
| FT+Augs+Sync | 0.98 | 0.83 | **0.82** | 0.69 | 0.80 |

RCC fine-tuning improves valuemetric robustness from 0.26 to 0.92; the synchronization layer improves geometric robustness from 0.01 to 0.82.

### Ablation Study (Token Match and Generation Quality)

| Configuration | Token Match (original) | Token Match (JPEG Q=25) | FID |
|--------------|----------------------|------------------------|-----|
| Original tokenizer | 0.66 | 0.31 | 16.7 |
| FT | >0.80 | ~0.55 | ≤16.7 |
| FT+Augs | >0.80 | ~0.70 | ≤16.7 |
| FT+Augs+Sync | >0.80 | ~0.70 | 17.3 |

Fine-tuning substantially improves token match with negligible change in FID, confirming that watermarking does not degrade generation quality.

### Key Findings
- RCC is the central bottleneck for watermark robustness: the original VQ tokenizer achieves TM of only 0.66, which exceeds 0.80 after fine-tuning.
- Fine-tuning not only improves valuemetric robustness but also unexpectedly enhances robustness against neural compression and diffusion purification attacks.
- The synchronization layer resolves the fundamental challenge of geometric transformations, albeit with a slight trade-off in valuemetric robustness.
- Compared to post-processing methods (CIN, MBRS, Trustmark, WAM), the proposed method is more robust to diffusion purification and neural compression.
- Consistent conclusions across three models (Taming, Chameleon, RAR-XL) demonstrate the generality of the approach.

## Highlights & Insights
- **The identification and resolution of the RCC problem** constitutes the paper's most significant contribution: it precisely diagnoses the core obstacle to transferring LLM watermarking to image tokens.
- The fine-tuning scheme is extremely lightweight — only the decoder and an encoder copy are updated, with no need to retrain the autoregressive model.
- The p-value computation for cross-modal joint detection maintains theoretical rigor (binomial hypothesis test).
- The synchronization layer paradigm — using an auxiliary signal to estimate the transformation, invert it, and then detect the watermark — is broadly generalizable.

## Limitations & Future Work
- The synchronization layer assumes that cropping preserves at least one corner; handling arbitrary crops would require more sophisticated synchronization patterns.
- A trade-off exists between the synchronization layer and valuemetric robustness, as synchronization signal corruption can cause incorrect inversion.
- Only zero-bit watermarking (presence/absence detection) is studied; multi-bit message embedding is not explored.
- Applicability to non-standard autoregressive architectures such as VAR remains to be verified.

## Related Work & Insights
- **vs. KGW (LLM watermarking)**: directly adapted but the paper identifies and resolves the RCC challenge, enabling cross-modal extension of watermarking from text to image tokens.
- **vs. diffusion model watermarks (Tree-Ring, etc.)**: different paradigms — diffusion models inject watermarks in latent space, whereas this work injects them into token sequences.
- **vs. post-processing methods (Trustmark, WAM)**: post-processing methods offer stronger valuemetric robustness but are highly vulnerable to diffusion purification and neural compression.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First study of watermarking for autoregressive image generation; both the identification of the RCC problem and its solution are original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 models and multiple attack types (valuemetric / geometric / adversarial / compression), with comparisons against post-processing baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, challenge analysis is thorough, and experiments are comprehensive.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in watermark-based provenance tracking for the rapidly growing field of autoregressive image generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
