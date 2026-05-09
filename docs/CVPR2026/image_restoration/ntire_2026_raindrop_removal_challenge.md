---
title: >-
  [Paper Note] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images
description: >-
  [CVPR 2026 (Workshop)][Image Restoration][Raindrop removal] This is the summary report of the NTIRE 2026 Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images. Based on the Raindrop Clarity real-world dataset (14,139 training / 407 validation / 593 test images), 168 teams participated and 17 submitted valid solutions. The winning team AIIA-Lab achieved the best score of 35.24 using an MSDT backbone combined with a pseudo-GT refinement pipeline.
tags:
  - CVPR 2026 (Workshop)
  - Image Restoration
  - Raindrop removal
  - dual-focused images
  - day and night scenes
  - image restoration competition
  - NTIRE
date: 2026-05-08
content_hash: 152617ce5f215425
---

# NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images

**Conference**: CVPR 2026 (Workshop)
**arXiv**: [2604.10634](https://arxiv.org/abs/2604.10634)
**Code**: [Competition Page](https://www.codabench.org/competitions/12808/)
**Area**: Image Restoration / Raindrop Removal
**Keywords**: Raindrop removal, dual-focused images, day and night scenes, image restoration competition, NTIRE

## TL;DR

This is the summary report of the NTIRE 2026 Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images. Based on the Raindrop Clarity real-world dataset (14,139 training / 407 validation / 593 test images), 168 teams participated and 17 submitted valid solutions. The winning team AIIA-Lab achieved the best score of 35.24 using an MSDT backbone combined with a pseudo-GT refinement pipeline.

## Background & Motivation

Raindrop removal is a fundamental low-level vision task that directly impacts downstream applications such as autonomous driving and surveillance. However, existing datasets suffer from critical limitations: (1) most cover only daytime scenes; (2) few simultaneously include both "raindrop-focused" and "background-focused" imaging modes; (3) real-world paired data is extremely scarce. The Raindrop Clarity dataset addresses these gaps by providing real degraded images spanning daytime/nighttime and dual-focus settings. Building on the success of the first challenge, this second edition adopts a revised data split (14,139 / 407 / 593) to establish a more robust and practical benchmark.

Compared to the first edition, this challenge places greater emphasis on balanced scene distribution across the validation and test sets, and attracted a larger number of participating teams (168 vs. the first edition).

## Method

### Overall Architecture

This paper is a competition summary report consolidating the solutions of 17 participating teams. The primary evaluation metric is a composite score: $\text{Score} = 10 \times \text{PSNR}(Y) + 10 \times \text{SSIM}(Y) - 5 \times \text{LPIPS}$.

### Key Designs

1. **MSDT backbone dominance**: Both the first-place team (AIIA-Lab) and the second-place team (raingod) adopted MSDT as the backbone, employing hyperparameter tuning and extended training to retain multiple strong checkpoints, followed by a second-stage refinement using scene-level pseudo-GT fusion.
2. **Scene-level pseudo-GT strategy**: Multiple top-ranked teams adopted a three-stage pipeline of "multi-image fusion within the same scene → pseudo-GT generation → fine-tuning," leveraging cross-view consistency within the same scene to improve raindrop removal.
3. **Diverse backbone exploration**: Submitted solutions encompass a wide range of architectures including Restormer, Histoformer, NAFNet, AdaIR, and diffusion models, demonstrating the methodological diversity within the image restoration community.

### Loss & Training

- The winning solution employs a two-stage training scheme: Stage 1 trains MSDT on mixed degradation data for 200 epochs with $256 \times 256$ patches; Stage 2 fine-tunes the model at a lower learning rate using pseudo-GT generated from scene-level fusion.
- Most solutions adopt a combined loss of L1 + SSIM + FFT / perceptual loss.
- Common engineering practices at test time include sliding window inference, multi-model ensemble, and checkpoint selection.
- Lightweight solutions (e.g., Cidaut AI with only 2.95M parameters) also achieved competitive performance.

## Key Experimental Results

### Main Results

| Rank | Team | Score | PSNR↑ | SSIM↑ | LPIPS↓ | Params |
|------|------|-------|-------|-------|--------|--------|
| 1 | AIIA-Lab | 35.24 | 28.34 | 0.827 | 0.273 | 16.6M |
| 2 | raingod | 35.22 | 28.28 | 0.826 | 0.264 | 16.6M |
| 3 | BUU_CV | 35.04 | 28.15 | 0.822 | 0.267 | 26.9M |
| 4 | RetinexDualV2 | 33.86 | 27.24 | 0.806 | 0.289 | 4.8M |
| 5 | ULR | 33.75 | 27.06 | 0.797 | 0.255 | 593M |
| 9 | Cidaut AI | 31.95 | 25.84 | 0.765 | 0.309 | 2.95M |
| 17 | BITssvgg | 30.94 | 25.14 | 0.750 | 0.338 | 16.9M |

### Key Findings

- The top three scores are extremely close (gap < 0.2), validating MSDT as the most effective base architecture.
- Team ULR achieves the best LPIPS (0.255), indicating that perceptual quality does not always align with signal fidelity metrics.
- Lightweight solutions (e.g., GU-day Mate with 2.14M parameters) can reach a score of 32.9, demonstrating the diversity of performance–efficiency trade-offs.
- Pseudo-GT construction combined with scene-level fusion is the most universally adopted strategy in this edition.

## Highlights & Insights

- Scene-level consistency exploitation is the most critical technique in this challenge: multiple images of the same scene naturally provide complementary information, and median/mean fusion can effectively suppress stochastic raindrops.
- Compared to the first edition, overall solution quality has improved substantially; however, genuine innovation is concentrated in test-time adaptation rather than architectural novelty during training.
- RetinexDualV2 incorporates physical priors (residual rain intensity masks), representing one of the few physically motivated approaches to raindrop modeling.

## Limitations & Future Work

- All solutions assume that multiple images of the same scene are available for fusion, which may not hold in real-world single-image settings.
- Raindrop removal performance on nighttime scenes remains noticeably inferior to daytime, highlighting the need for more nighttime training data.
- There is no dedicated evaluation for extreme degradation cases (e.g., large-area raindrop coverage).
- Future work may explore temporal consistency for raindrop removal in video sequences.
- The training data scale is limited (only 14,139 images); larger-scale datasets could yield further performance gains.

## Related Work & Insights

- Mainstream restoration backbones such as MSDT, Restormer, and NAFNet remain effective for raindrop removal tasks.
- The pseudo-GT + self-supervised fine-tuning paradigm is transferable to other image restoration competitions (e.g., dehazing, deblurring).
- Frequency-domain attention (e.g., the frequency branch in RetinexDualV2) is a direction worthy of deeper investigation.
- The Raindrop Clarity dataset is currently the only real-world paired benchmark that simultaneously covers daytime/nighttime and dual-focus degradation.
- Diffusion model-based solutions (NTR) underperform compared to traditional restoration models, possibly because the data scale is insufficient to leverage their generative capacity.

## Method Summary

| Solution | Backbone | Key Feature | Extra Data |
|----------|----------|-------------|------------|
| AIIA-Lab | MSDT | Multi-checkpoint selection + scene fusion + pseudo-GT refinement | No |
| raingod | MSDT | UAV-Rain1k augmentation + median filtering pseudo-GT | Yes |
| BUU_CV | STRRNet + Restormer | Rectangular/square patch complementarity + weighted ensemble | Yes |
| RetinexDualV2 | Retinex dual-branch | Physical prior (residual rain intensity mask) + Mamba attention | No |
| Cidaut AI | NAFNet | Dual attention module (spatial + frequency) | No |

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 2 | Competition summary report; methodological innovation lies primarily with participating teams |
| Technical Depth | 3 | Detailed documentation of training and test-time strategies across 17 solutions |
| Experimental Thoroughness | 4 | Well-established benchmark, multi-metric evaluation, 168 participating teams |
| Writing Quality | 3 | Clear structure, though descriptions of some solutions are relatively brief |
| Value | 4 | Real-world benchmark with diverse solutions; strong practical utility |

**Overall**: As a competition report, this paper comprehensively documents the latest advances and best practices in raindrop removal. The pseudo-GT + scene fusion strategy has high transferability to related tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)](ntire_2026_ai_flash_portrait_challenge.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[CVPR 2026\] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)

<!-- RELATED:END -->
