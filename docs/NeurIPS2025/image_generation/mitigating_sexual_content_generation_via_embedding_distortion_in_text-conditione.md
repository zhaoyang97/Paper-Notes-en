---
title: >-
  [Paper Note] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Unsafe content mitigation] This paper proposes Distorting Embedding Space (DES), a text-encoder-based defense framework that achieves state-of-the-art sexual content mitigation on FLUX.1 and SD v1.5 (reducing ASR to 9.47% and 0.52%, respectively) by transforming unsafe embeddings into safe regions, preserving safe embeddings, and neutralizing "nudity" semantics, while maintaining high-quality benign image generation.
tags:
  - NeurIPS 2025
  - Image Generation
  - Unsafe content mitigation
  - embedding space distortion
  - text encoder
  - adversarial attack defense
  - NSFW filtering
date: 2026-05-08
content_hash: e860e8dc7000920c
---

# Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2501.18877](https://arxiv.org/abs/2501.18877)
**Code**: Unavailable
**Area**: Diffusion Models / Safety & Defense
**Keywords**: Unsafe content mitigation, embedding space distortion, text encoder, adversarial attack defense, NSFW filtering

## TL;DR

This paper proposes Distorting Embedding Space (DES), a text-encoder-based defense framework that achieves state-of-the-art sexual content mitigation on FLUX.1 and SD v1.5 (reducing ASR to 9.47% and 0.52%, respectively) by transforming unsafe embeddings into safe regions, preserving safe embeddings, and neutralizing "nudity" semantics, while maintaining high-quality benign image generation.

## Background & Motivation

Diffusion models (SD, DALL-E, etc.) are powerful but susceptible to misuse for generating pornographic or NSFW content. Existing defenses each have notable shortcomings:

**Filtering methods** (blacklist-based text filtering, safety checkers): easily bypassed by adversarially crafted prompts.

**Concept removal methods** (ESD, SalUn): modifying the U-Net degrades generation quality or lacks robustness against adversarial attacks.

**Sexual content mitigation methods** (SafeGen, ShieldDiff): SafeGen produces visible artifacts; ShieldDiff has not been evaluated under adversarial attacks.

A key observation is that concept-relevant parameters are distributed across all layers of the U-Net, making precise removal difficult. In contrast, attributes stored in the text encoder are localized and thus more amenable to targeted intervention. Further insight drawn from continual learning suggests that maintaining feature locations reduces catastrophic forgetting; conversely, **forcing features away from their original locations** may effectively erase unsafe information.

## Method

### Overall Architecture

DES consists of two stages: (1) **Target vector generation**: computing an optimal safe transformation target for each unsafe prompt; (2) **Training**: fine-tuning the text encoder to distort the unsafe embedding space while preserving safe embeddings.

### Key Designs

1. **Target Vector Generation**

   For each unsafe vector $u_i$, the safe vector with the lowest cosine similarity is identified:
   $$s_i^* = \arg\min_{s_i} \frac{u_i \cdot s_i}{\|u_i\|\|s_i\|}$$

   The "nudity" direction ($n$ being the "nudity" vector) is then subtracted to produce an anti-correlated target vector:
   $$t_i = s_i^* - \alpha \frac{n}{\|n\|}$$

   where $\alpha$ is a scaling factor ($\alpha = 200$). **Design Motivation**: selecting the least similar safe vector as a base, then subtracting the nudity direction, ensures the target vector is anti-correlated with nudity concepts and maximizes embedding space distortion for greater robustness. The authors observe that even the selected safe vectors exhibit positive correlation with the nudity vector, making the subtraction operation necessary.

2. **Unsafe Embedding Space Distortion + Safe Embedding Preservation**

   **Unsafe loss**: aligns the current unsafe vector to its target safe vector:
   $$\mathcal{L}_u = \frac{1}{B}\sum_{i=1}^B \left(1 - \frac{\tilde{u}_i \cdot t_i}{\|\tilde{u}_i\|\|t_i\|}\right)$$

   **Safe loss** (with adaptive weighting): preserves safe vectors relative to their originals, and employs a nudity-integrated vector $\tilde{s}'_i = \tilde{s}_i + \alpha\frac{n}{\|n\|}$ for adaptive weighting:
   $$\mathcal{L}_s = \frac{1}{B}\sum_{i=1}^B \left[\left(1 - \frac{\tilde{s}_i \cdot s_i}{\|\tilde{s}_i\|\|s_i\|}\right) + \left(1 - \frac{\tilde{s}'_i \cdot s_i}{\|\tilde{s}'_i\|\|s_i\|}\right)\right]$$

   Adaptive mechanism: safe vectors with low correlation to the nudity vector receive a stronger preservation penalty, while those with high correlation are adjusted more leniently, as they may carry latent unsafe semantics.

3. **Nudity Neutralization**

   The "nudity" vector is aligned to a neutral empty vector $e_0$ (the embedding corresponding to the empty string ""):
   $$\mathcal{L}_n = 1 - \frac{\tilde{n} \cdot e_0}{\|\tilde{n}\|\|e_0\|}$$

   **Motivation**: this prevents concept-extraction-based attacks (e.g., Ring-A-Bell, which uses a genetic algorithm to find prompts similar to nudity concepts). After neutralization, attackers can only extract semantically meaningless embeddings.

### Loss & Training

Total loss: $\mathcal{L}_t = \lambda \mathcal{L}_s + (1-\lambda)(\mathcal{L}_u + \mathcal{L}_n)$, with $\lambda = 0.3$.

The three losses are complementary and non-conflicting: nudity neutralization operates on the current "nudity" vector; the unsafe loss uses precomputed nudity vectors for target offsets; and the safe loss also uses precomputed values for similarity computation.

Training is highly efficient: only **90 seconds** with **zero inference overhead**. Training data: 6,911 safe–unsafe prompt pairs from the CoPro dataset.

## Key Experimental Results

### Main Results

**Defense against explicit I2P prompts (SD v1.5, NudeNet detection):**

| Method | Nudity Total↓ | FID↓ | CLIP↑ |
|--------|--------------|------|-------|
| SD v1.5 (no defense) | 851 | 16.57 | 26.46 |
| SLD-strong | 511 | 31.38 | 24.61 |
| Safe-CLIP | 404 | 17.49 | 25.73 |
| UCE | 216 | 16.99 | 26.16 |
| SalUn | 21 | 21.14 | 24.78 |
| AdvUnlearn | 27 | 18.94 | 23.82 |
| **DES** | **16** | **15.44** | **25.52** |

**Defense against adversarial prompts (black-box attacks, SD v1.5, ASR↓):**

| Method | Sneaky | MMA | Ring-A-Bell | P4D | Avg. ASR↓ |
|--------|--------|-----|-------------|-----|-----------|
| SD v1.5 | 45.16 | 73.93 | 98.13 | 94.93 | 78.04 |
| AdvUnlearn | 1.61 | 2.10 | 0.93 | 1.10 | 1.44 |
| **DES** | **0.00** | **0.40** | **0.93** | **0.74** | **0.52** |

**On FLUX.1:** DES achieves an average ASR of 8.86% vs. 43.23% for EraseAnything, a reduction of approximately 80%.

### Ablation Study

| Configuration | Role | Effect |
|---------------|------|--------|
| $\mathcal{L}_u$ only | Distort unsafe embeddings | Effective but degrades safe image quality |
| $\mathcal{L}_u + \mathcal{L}_s$ | Add safe preservation | Recovers FID and CLIP score |
| $\mathcal{L}_u + \mathcal{L}_s + \mathcal{L}_n$ | Add nudity neutralization | More robust against extraction-based attacks |
| Scaling factor $\alpha$ | Controls target offset magnitude | $\alpha = 200$ is optimal |

**White-box adaptive attacks:**

| Method | MMA↓ | UDA↓ | Ring-A-Bell↓ | CCE↓ | Avg.↓ |
|--------|------|------|-------------|------|-------|
| ESD | 8.50 | 60.56 | 26.17 | 18.12 | 28.34 |
| AdvUnlearn | 2.73 | 19.72 | 0.00 | 6.15 | 7.15 |
| **DES** | **1.82** | **18.31** | **0.00** | 5.76 | **6.47** |

### Key Findings

- DES achieves state-of-the-art or near-SOTA ASR across all attack types, with extremely low cross-attack variance (std 0.41).
- A key advantage is preserved generation quality: FID of 15.44 actually surpasses the original SD v1.5 baseline of 16.57 (possibly due to the removal of unsafe content that negatively affects FID).
- Text-encoder-level intervention outperforms U-Net-level intervention: both AdvUnlearn and DES surpass ESD and UCE.
- DES is effective on FLUX.1 (a multi-text-encoder architecture) by independently training each encoder.
- Training requires only 90 seconds with zero inference overhead, making it the most efficient defense solution to date.

## Highlights & Insights

- The insight of deriving "distorting unsafe feature locations achieves forgetting" from continual learning's principle of "feature location stability reduces forgetting" is elegant and well-motivated.
- The three-component loss design (distortion + preservation + neutralization) is complementary and non-conflicting, forming a complete embedding space control system.
- The adaptive weighting mechanism in the safe loss reflects a deep understanding of the structure of the embedding space.
- The combination of 90-second training and zero inference overhead provides immediate deployment value.

## Limitations & Future Work

- Target vector generation relies on a predefined set of safe/unsafe prompts, which may have limited coverage.
- Neutralizing a single "nudity" vector may be overly simplistic, as sexually relevant semantics may be distributed across multiple dimensions.
- Robustness against novel attacks (e.g., embedding space interpolation attacks) remains to be verified.
- ASR on I2I tasks remains around 20%, leaving room for improvement.

## Related Work & Insights

- The comparison with AdvUnlearn highlights the advantage of embedding space control over adversarial training, which degrades generation quality.
- The subtraction operation in target vector construction (subtracting the nudity direction) is generalizable to other concept removal tasks.
- The adaptive safe loss design can inspire other selective forgetting/retention tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The embedding space distortion approach is novel; the adaptive safe loss and nudity neutralization are elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple attack scenarios (explicit/black-box/white-box/adaptive), two models, and both T2I and I2I settings.
- **Writing Quality**: ⭐⭐⭐⭐ — Method presentation is clear; handling of safety-sensitive content requires care.
- **Value**: ⭐⭐⭐⭐⭐ — 90-second training, zero inference overhead, and SOTA defense performance make this highly valuable for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](../../ICCV2025/image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)

</div>

<!-- RELATED:END -->
