---
title: >-
  [Paper Note] NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)
description: >-
  [CVPR 2026 (Workshop)][Image Restoration][Low-light portrait] NTIRE 2026 3rd RAIM Challenge AI Flash Portrait Track: mapping weak-flash low-light portraits to strong-flash professional-grade portraits, providing 800 real paired samples (with professional retoucher GT), adopting a dual evaluation system combining region-aware objective metrics and expert blind assessment. 118 teams registered with 3,187 valid submissions.
tags:
  - CVPR 2026 (Workshop)
  - Image Restoration
  - Low-light portrait
  - flash simulation
  - portrait restoration
  - subjective-objective evaluation
  - NTIRE
date: 2026-05-08
content_hash: d3ff303d3b7c5502
---

# NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)

**Conference**: CVPR 2026 (Workshop)  
**arXiv**: [2604.11230](https://arxiv.org/abs/2604.11230)  
**Code**: [CodaBench](https://www.codabench.org/)  
**Area**: Image Restoration / Low-Light Portrait Enhancement  
**Keywords**: Low-light portrait, flash simulation, portrait restoration, subjective-objective evaluation, NTIRE

## TL;DR

NTIRE 2026 3rd RAIM Challenge AI Flash Portrait Track: mapping weak-flash low-light portraits to strong-flash professional-grade portraits, providing 800 real paired samples (with professional retoucher GT), adopting a dual evaluation system combining region-aware objective metrics and expert blind assessment. 118 teams registered with 3,187 valid submissions.

## Background & Motivation

Low-light portrait photography on mobile devices is a core challenge in computational photography. Constrained by small sensors and insufficient illumination, low-light portraits suffer from severe noise, color distortion, and loss of detail. Existing methods exhibit four key limitations: (1) traditional low-light image enhancement (LLIE) methods focus on global brightness improvement, resulting in skin tone distortion and flattened facial lighting; (2) real degradation processes are highly complex, making synthetic data insufficient to simulate the nonlinear illumination transformation from weak to strong flash; (3) face restoration models are limited to local processing, causing a "cut-and-paste" artifact between foreground and background in low-light scenes; and (4) conventional objective metrics (PSNR/SSIM/LPIPS) fail to adequately capture aesthetic and perceptual naturalness.

This track is co-organized by OPPO Y-Lab, Shenzhen University, PolyU VC-Lab, and Nankai University, aiming to bridge the gap between academic research and industrial applications in low-light portrait computational photography.

## Method

### Overall Architecture

This track introduces a novel task definition: mapping weak-flash low-light portraits to strong-flash professional-grade portraits, going beyond traditional LLIE by combining physical illumination enhancement with aesthetic rendering. Evaluation adopts region-aware metrics combined with expert blind assessment (weighted 3:7).

### Key Designs

1. **Region-aware evaluation system**: The scoring formula separately evaluates the subject region (using LPIPS and $\Delta E$ for perceptual similarity and color difference) and the background region (using PSNR for signal-to-noise ratio), plus global SSIM, preventing strategies that over-sharpen portraits or flatten facial features solely to inflate global PSNR.
2. **Expert blind assessment mechanism**: Results from the Top-12 teams are randomly anonymized and presented to 5 or more senior experts, who evaluate across six dimensions—facial naturalness, portrait detail preservation, illumination realism, background cleanliness, scene balance, and overall consistency—to select Top-3, normalized to a subjective score in the range of 80–90.
3. **High-quality real paired data**: 800 pairs at 1K resolution, each comprising a low-light input, a professionally retouched GT, and a subject mask—a rare high-quality real paired benchmark in this field.

### Loss & Training

- Any publicly available external datasets and pre-trained models are permitted.
- Three-phase competition pipeline: Phase 1 training (600 pairs) → Phase 2 online validation (100 pairs) → Phase 3 final evaluation (100 hidden pairs).
- Final evaluation is reproduced by the organizers on unified hardware; resolution rescaling is strictly prohibited.

## Key Experimental Results

### Main Results (Phase 2 Online Evaluation)

| Rank | Team | Phase 2 Score | LPIPS$_\text{person}$↓ | $\Delta E_\text{person}$↓ | GlobalScore↑ |
|------|------|--------------|----------------------|--------------------------|-------------|
| 2 | nunucccb | 86.10 | 0.0266 | 7.19 | 0.784 |
| 4 | SHL | 84.91 | 0.0268 | 6.83 | 0.742 |
| 6 | hezhaokun | 84.88 | 0.0270 | 6.75 | 0.739 |
| 7 | KC110 | 84.33 | 0.0284 | 8.07 | 0.765 |
| Baseline | Organizers | 82.16 | - | - | - |

### Key Findings

- The competition attracted 118 registered teams and 3,187 valid submissions, reflecting high interest in this task.
- A clear trade-off exists between subject-region LPIPS and color difference ($\Delta E$) versus background PSNR.
- Some teams with high online leaderboard scores were disqualified due to large discrepancies during code reproduction (marked as "–").
- The correlation between objective and subjective evaluations warrants further investigation.

## Highlights & Insights

- The task definition is novel: rather than simple "low-light enhancement," it requires achieving professional retouching-level aesthetic quality, bridging the gap between academic research and industrial applications.
- The evaluation system is well-designed: region-aware metrics prevent common evaluation pitfalls (e.g., excessive smoothing for high PSNR), and the combination of objective and subjective assessment ensures practical relevance.
- Real paired data with designer GT represents an exceptionally valuable resource in this field.
- This track reveals that existing methods struggle to simultaneously achieve facial aesthetics and background consistency.

## Limitations & Future Work

- Detailed Phase 3 results are not fully disclosed in this report (the combined subjective-objective ranking is not presented).
- Although expert blind assessment better approximates human perception, the limited number of evaluators (5 persons) may introduce subjective bias.
- The current dataset is limited to 1K resolution; high-resolution scenarios (4K) are not covered.
- Future work may extend to video low-light portrait enhancement, multi-person scenes, and integration with generative models.

## Related Work & Insights

- The limitations of traditional LLIE methods (e.g., RetinexNet) in portrait scenarios merit systematic investigation.
- The region-aware evaluation approach can be generalized to other restoration tasks where regional importance is non-uniform.
- Balancing portrait aesthetic enhancement with physical consistency remains an open problem.
- The six-dimensional subjective evaluation criteria (facial naturalness, portrait detail, illumination realism, background cleanliness, scene balance, overall consistency) can serve as a general evaluation framework for portrait processing.

## Competition Pipeline Details

| Phase | Date | Content | Data Volume |
|-------|------|---------|-------------|
| Phase 1 | 2026.01.23 | Model design; training set and baseline released | 600 pairs |
| Phase 2 | 2026.01.28 | Online objective evaluation feedback | 100 pairs (no GT) |
| Phase 3 | 2026.03.05–12 | Code submission + unified reproduction + expert blind assessment | 100 pairs (hidden) |
| Final Ranking | 2026.03.19 | Objective score 30% + subjective score 70% | Top-12 |

### Evaluation Metrics Details

- **Subject region**: LPIPS$_\text{person}$ (perceptual similarity) + $\Delta E_\text{person}$ (color difference), ensuring high fidelity for faces and skin.
- **Background region**: PSNR$_\text{bg}$ (signal-to-noise ratio), ensuring no noise is introduced in the background.
- **Global**: SSIM$_\text{global}$ (structural similarity), measuring overall structural consistency.
- **Subjective**: 50 image pairs × 12 teams displayed anonymously; experts select Top-3; frequency counts are normalized to scores in the 80–90 range.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 3 | Innovative task definition; well-designed evaluation system |
| Technical Depth | 3 | Competition report covering evaluation and data construction details |
| Experimental Thoroughness | 4 | 118 participating teams; dual objective-subjective evaluation |
| Writing Quality | 4 | Competition motivation and evaluation scheme clearly articulated |
| Value | 4 | High-quality real dataset + industry-grade evaluation standard |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](ntire_2026_raindrop_removal_challenge.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

</div>

<!-- RELATED:END -->
