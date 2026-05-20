---
title: >-
  [Paper Note] Robustness in Both Domains: CLIP Needs a Robust Text Encoder
description: >-
  [NeurIPS 2025][Image Generation][CLIP] This paper proposes LEAF (Levenshtein Efficient Adversarial Finetuning), the first adversarial fine-tuning method targeting the CLIP text encoder. LEAF substantially improves robust…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "CLIP"
  - "text encoder robustness"
  - "adversarial fine-tuning"
  - "character-level attacks"
  - "Levenshtein distance"
date: 2026-05-08
content_hash: ea6a88c5ce601b9b
---

# Robustness in Both Domains: CLIP Needs a Robust Text Encoder

**Conference**: NeurIPS 2025
**arXiv**: [2506.03355](https://arxiv.org/abs/2506.03355)  
**Code**: Available (github.com/LIONS-EPFL/LEAF, huggingface.co/LEAF-CLIP)  
**Area**: Image Generation
**Keywords**: CLIP, text encoder robustness, adversarial fine-tuning, character-level attacks, Levenshtein distance

## TL;DR

This paper proposes LEAF (Levenshtein Efficient Adversarial Finetuning), the first adversarial fine-tuning method targeting the CLIP text encoder. LEAF substantially improves robustness under character-level text perturbations across zero-shot classification, text-image retrieval, and image generation, while preserving performance in the image domain.

## Background & Motivation

CLIP is widely adopted in downstream tasks such as retrieval, large multimodal models (LMMs), and text-to-image generation. However, adversarial attacks can cause significant shifts in CLIP embeddings:

**Progress in image-domain robustness**: TeCoA and FARE have addressed adversarial fine-tuning of the image encoder via supervised and unsupervised approaches, respectively.

**Gap in text-domain robustness**: The robustness of the text encoder remains entirely unexplored.

**Necessity of dual-domain defense**: Protecting only one modality is insufficient against real-world attack scenarios.

Core motivation: CLIP requires adversarial robustness **simultaneously in both the image and text domains**.

## Method

### Overall Architecture

LEAF extends the FARE objective to the text domain:

**TextFARE Objective**:
$$\min_{\theta} \sum_{i=1}^n \max_{S_i': d_{Lev}(S_i, S_i') \leq k \wedge S_i' \in \mathcal{C}(S_i)} \|f_{\theta^{CLIP}}(S_i) - f_{\theta}(S_i')\|_2^2$$

That is, the text encoder parameters $\theta$ are optimized so that, under perturbations within Levenshtein distance $\leq k$, the encoder output remains as close as possible to the encoding of the original text.

### Key Designs

**LEAF Attack Algorithm** (efficient training-time attack):
1. **Position selection**: Randomly sample $\rho$ positions, substitute test characters, and select the position yielding the highest loss.
2. **Character selection**: Randomly try $\rho$ characters at the selected position and select the substitution yielding the highest loss.

Key advantage: Only a constant $\rho$ perturbation evaluations per sentence are required (independent of sentence length), with support for batch-level parallelism.
- Charmer (baseline attack): requires $O(2|S|+1+n_{Charmer} \cdot |\Gamma|)$ evaluations.
- LEAF: requires only $2 \times B \times \rho$ evaluations ($B$ = batch size).

**Semantic Constraints**:
- Adopts the constraint from Chanakya et al. (2024): perturbations are not permitted to introduce new English words.
- Enforced via the NLTK lexicon.
- Constraints are critical for preserving image-domain performance.

**Decoupled Training**:
- The text encoder and image encoder are fine-tuned independently.
- FARE fine-tunes the image encoder; LEAF fine-tunes the text encoder.
- The two can be freely combined.

### Loss & Training

- Training is conducted on the first 80K samples of DataComp-small for 30 epochs.
- Batch size 128, AdamW optimizer, learning rate $10^{-5}$.
- $k=1$ (single-character perturbation), $\rho=50$.
- Training with semantic constraints.

## Key Experimental Results

### Main Results

Zero-shot classification (ImageNet + AG-News):

| Robust Encoder | ImageNet | ImageNet | AG-News | AG-News |
| Image / Text | Clean Acc. | Adv. Acc. | Clean Acc. | Adv. Acc. |
|-----------|----------|---------|---------|---------|
| ✗ / ✗ (CLIP-L/14) | 76.4 | 0.0 | 74.4 | 44.7 |
| ✓ / ✗ (FARE) | 74.7 | 47.6 | 78.7 | 44.5 |
| ✗ / ✓ (LEAF) | 73.4 | 0.0 | 73.9 | 60.1 |
| ✓ / ✓ (FARE+LEAF) | 72.6 | 46.0 | 78.0 | **63.2** |

OpenCLIP-ViT-H/14 results:

| Robust Encoder (Image/Text) | ImageNet Adv. | AG-News Adv. |
|-------------------|-------------|-------------|
| ✗ / ✗ | 0.0 | 37.6 |
| ✓ / ✗ | 48.4 | 37.5 |
| ✓ / ✓ | 46.3 | **53.3** |

### Ablation Study

Effect of training hyperparameters (ViT-L/14, $k=1$):

| $\rho$ | Constraint | ImageNet Clean | AG-News Adv. |
|--------|------|---------------|-------------|
| 1 (random) | ✓ | 74.7 | 54.4 (+9.9) |
| 10 | ✓ | 74.8 | 59.9 |
| 50 | ✓ | 72.6 | 63.2 (+18.7) |
| 50 | ✗ | 65.5 | 66.3 |

Training speed comparison:

| Attack Method | Time per batch (s) | AG-News Adv. |
|---------|-------------|-------------|
| Charmer-20 | 118.19 | baseline |
| Charmer-1 | 15.17 | slightly lower |
| LEAF ($\rho$=20) | 1.83 | comparable |
| LEAF ($\rho$=50) | 3.23 | comparable |

Text-to-image retrieval (MS-COCO, $k=2$, average over 3 targets):

| Model | Robust? | R@1 Clean | R@1 Adv. | R@5 Clean | R@5 Adv. |
|------|------|---------|---------|---------|---------|
| CLIP-L/14 | ✗ | 49.11 | 30.66 | 73.79 | 52.76 |
| CLIP-L/14 | ✓ | 48.71 | 40.22 | 73.71 | 65.09 |

### Key Findings

1. **LEAF achieves an order-of-magnitude speedup**: 1.83s vs. 118.19s per batch with negligible performance degradation.
2. **Dual-domain robustness is necessary**: Robustness in both domains is only achieved when both the image and text encoders are made robust.
3. **Semantic constraints are critical**: Training without constraints severely degrades image-domain clean accuracy (from 74.7 to 65.5).
4. **Robust models are more interpretable**: Embeddings from the robust text encoder are more amenable to inversion back to text via optimization.
5. **Generalizes to larger perturbation budgets**: Models trained at $k=1$ generalize effectively to perturbations at $k=5$.

## Highlights & Insights

1. **Fills a gap in the literature**: This is the first systematic study of adversarial robustness in the CLIP text encoder.
2. **Efficient and effective**: LEAF's batch-parallel design makes adversarial training in the text domain practically feasible.
3. **Plug-and-play compatibility**: The robust encoder can directly replace the original text encoder in SD/SDXL pipelines.
4. **Robustness ≈ Interpretability**: Robust model embeddings yield higher-quality inversion results.

## Limitations & Future Work

1. The image and text encoders are fine-tuned independently; joint adversarial attacks perturbing both domains simultaneously have not been evaluated.
2. Only character-level attacks are studied; token-level robustness is not addressed, as token-level perturbations often alter semantics.
3. The largest EVA-CLIP model was not trained due to computational constraints.
4. Other CLIP application scenarios, such as RAG, have not been tested.
5. Joint training of both encoders with increased computational budget may yield further improvements.

## Related Work & Insights

- **FARE** (Schlarmann et al. 2024): Unsupervised adversarial fine-tuning of the CLIP image encoder.
- **TeCoA** (Mao et al. 2023): Supervised adversarial fine-tuning of the CLIP image encoder.
- **Charmer** (Abad Rocamora et al. 2024): Character-level adversarial attack on text.
- Insight: Decoupled training combined with an efficient attack algorithm represents a key pathway toward practical adversarial robustness.

## Rating

- Novelty: ⭐⭐⭐⭐ (first study of CLIP text encoder robustness)
- Technical Depth: ⭐⭐⭐⭐ (elegant efficient attack algorithm design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (classification / retrieval / generation / inversion across multiple tasks)
- Value: ⭐⭐⭐⭐⭐ (models are open-sourced and directly usable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](../../ICCV2025/image_generation/your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[ICCV 2025\] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text](../../ICCV2025/image_generation/fix-clip_dual-branch_hierarchical_contrastive_learning_via_synthetic_captions_fo.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
