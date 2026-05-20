---
title: >-
  [Paper Note] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning
description: >-
  [NeurIPS 2025][Segmentation][Super-Resolution] SRSR proposes a training-free plug-and-play framework that addresses semantic hallucination caused by text guidance in diffusion-based super-resolution methods. It introduce…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Super-Resolution"
  - "Semantic Accuracy"
  - "Cross-Attention"
  - "Classifier-Free Guidance"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: a56be15901d729ac
---

# SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning

**Conference**: NeurIPS 2025
**arXiv**: [2510.22534](https://arxiv.org/abs/2510.22534)  
**Code**: N/A  
**Area**: Image Segmentation
**Keywords**: Super-Resolution, Semantic Accuracy, Cross-Attention, Classifier-Free Guidance, Plug-and-Play

## TL;DR

SRSR proposes a training-free plug-and-play framework that addresses semantic hallucination caused by text guidance in diffusion-based super-resolution methods. It introduces two inference-time modules—Spatially Re-focused Cross-Attention (SRCA) and Spatially Targeted CFG (STCFG)—and comprehensively outperforms 7 SOTA baselines in both fidelity and perceptual quality.

## Background & Motivation

Stable Diffusion-based super-resolution methods (e.g., SeeSR, OSEDiff) leverage textual priors to guide generation, but they suffer from three semantic issues:

**Cross-attention misalignment**: Text token attention leaks into irrelevant pixel regions. For example, the attention of the "bird" token bleeds into rock regions, causing wing textures to appear on stones; "grass" attention disperses onto a lion's face, producing hallucinated textures (Figure 1).

**Inaccurate prompts**: Although DAPE (degradation-aware prompt extractor) is more robust than BLIP/LLaVA, it may still extract incorrect tags from severely degraded images (e.g., misidentifying a rock as "camouflage"). Incorrect text guidance is more harmful than no text guidance at all.

**Incomplete prompts**: DAPE is object-centric by design and cannot cover all image regions, particularly backgrounds. Regions not covered by any tag (ungrounded regions) are vulnerable to the influence of irrelevant text.

The core insight is: **inaccurate guidance is more harmful than incomplete guidance**—it is preferable to withhold text guidance for certain regions than to provide incorrect guidance.

## Method

### Overall Architecture

SRSR is a purely inference-time plug-and-play module compatible with any cross-attention-based super-resolution method that uses textual priors. The pipeline proceeds as follows: LR image → DAPE extracts text tags → Grounded SAM performs visual grounding (filtering unreliable tags and generating tag-mask pairs) → SRCA constrains cross-attention → STCFG handles ungrounded regions → SR image output.

### Key Designs

1. **Spatially Re-focused Cross-Attention (SRCA)**: Binary segmentation masks obtained via visual grounding are used to constrain the attention range of each text token. Standard cross-attention is defined as $\alpha_{ij} = \text{Softmax}(Q_i \cdot K_j / \sqrt{d})$. SRCA first masks out irrelevant regions via $\alpha_{ij}^{\text{SRCA}} = M_{ij} \cdot \alpha_{ij}$, then re-normalizes over all valid pixel-token pairs:
    $$\hat{\alpha}_{ij}^{\text{SRCA}} = \frac{\alpha_{ij}^{\text{SRCA}}}{\sum_{i',j'} \alpha_{i'j'}^{\text{SRCA}}}$$
   This ensures that relevant tokens are not diluted by attention from irrelevant regions. Additionally, the visual grounding step naturally filters unreliable tags—tags that cannot be visually grounded are deemed irrelevant and discarded.

2. **Spatially Targeted Classifier-Free Guidance (STCFG)**: Standard CFG applies text guidance uniformly across all pixels: $\hat{\epsilon}_i = \epsilon_\theta(x_t, \phi) + s[\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \phi)]$. However, for ungrounded regions, global tokens (EOS, punctuation, etc.) carry the semantics of the entire prompt and interfere with the restoration of these regions. STCFG addresses this by applying CFG selectively in the spatial domain:
    $$\hat{\epsilon}_i = (1-M_i)[\epsilon_\theta(x_t,\phi) + s(\epsilon_\theta(x_t,y) - \epsilon_\theta(x_t,\phi))] + M_i \cdot \epsilon_\theta(x_t,\phi)$$
   where $M_i=1$ indicates that pixel $i$ is ungrounded. CFG-based text guidance is applied normally to grounded regions, while only unconditional predictions are used for ungrounded regions.

### Loss & Training

SRSR requires no training and operates entirely at inference time. It uses the original pretrained SD and UNet without introducing any learnable parameters. Grounded SAM is run only once on the LR image (0.12s for 128×128), and the resulting masks are cached for reuse.

## Key Experimental Results

### Main Results

| Dataset | Metric | SRSR-SeeSR | SeeSR Baseline | Best Competitor | Gain |
|--------|------|-----------|----------|---------|------|
| RealSR | PSNR↑ | **26.40** | 25.18 | 26.31 (ResShift) | +0.09 |
| RealSR | SSIM↑ | **0.7632** | 0.7216 | 0.7421 (ResShift) | +0.0211 |
| RealSR | LPIPS↓ | **0.2718** | 0.3009 | 0.3009 (SeeSR) | -0.0291 |
| RealSR | DISTS↓ | **0.2092** | 0.2223 | 0.2223 (SeeSR) | -0.0131 |
| DIV2K | PSNR↑ | **24.72** | 23.68 | 24.65 (ResShift) | +0.07 |
| DrealSR | PSNR↑ | **29.50** | 28.17 | 28.46 (ResShift) | +1.04 |
| DrealSR | LPIPS↓ | **0.2866** | 0.3189 | 0.3177 (OSEDiff) | -0.0311 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|------|-------|-------|--------|--------|
| V1: SeeSR baseline | 25.17 | 0.722 | 0.301 | 0.222 |
| V2: +Grounding | 25.18 | 0.723 | 0.300 | 0.223 |
| V3: +Grounding+SRCA | 25.27 | 0.728 | 0.301 | 0.225 |
| V4: +Grounding+SRCA+STCFG (full) | **26.40** | **0.763** | **0.272** | **0.209** |
| V5: V4+ungrounded tags | 26.39 | 0.763 | 0.273 | 0.210 |
| V7: V4+Mask2Former | 26.31 | 0.762 | 0.273 | 0.209 |
| V8: V4+DINO-X | 26.34 | 0.763 | 0.272 | 0.209 |

### Key Findings

- STCFG contributes the most (V3→V4, PSNR +1.13); SRCA alone improves fidelity but incurs a slight cost in perceptual quality.
- Adding extra semantic segmentation labels (Mask2Former/DINO-X) degrades performance, confirming that inaccurate tags are more harmful than incomplete tags.
- Sensitivity analysis shows that SRSR is robust to the grounding confidence threshold (values between 0.15 and 0.55 all substantially outperform the baseline).
- No-reference metrics (NIQE/MUSIQ, etc.) tend to reward hallucinated results and are unreliable indicators of semantic fidelity.

## Highlights & Insights

- The insight that "inaccurate tags are more harmful than incomplete tags" is highly practical and offers an important design principle for text-guided generative models.
- The purely inference-time plug-and-play design is highly practical—it can directly enhance any text-conditioned super-resolution method.
- SRCA and STCFG are complementary: the former addresses semantic confusion in grounded regions, while the latter suppresses hallucinations in ungrounded regions.
- The work reveals a significant limitation of no-reference quality metrics in evaluating semantic fidelity.

## Limitations & Future Work

- Performance depends on the segmentation quality of Grounded SAM, which may be inaccurate under severe degradation.
- STCFG is not applicable to methods that do not support CFG (e.g., OSEDiff).
- Optimization is confined to inference time and has not been integrated into training.
- The method still relies on DAPE as the initial tag extractor; a better degradation-aware tag extraction module could yield further improvements.

## Related Work & Insights

- Compared to the concurrent work HolisDiP, which uses Mask2Former for full-coverage segmentation but lacks degradation awareness and is limited to 150 categories.
- SFT-GAN adopts a similar spatial feature transformation idea but is GAN-based.
- The proposed framework provides a general semantic constraint paradigm for text-guided diffusion models, with potential applicability to other tasks such as text-to-image generation.

## Rating

- Novelty: ⭐⭐⭐⭐ (The spatially selective guidance design of SRCA+STCFG is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 baselines, 3 datasets, detailed ablations and hyperparameter analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent visualizations; problem-solution correspondence is clear)
- Value: ⭐⭐⭐⭐⭐ (Plug-and-play with significant performance gains; extremely high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Panoptic Captioning: An Equivalence Bridge for Image and Text](panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)
- [\[NeurIPS 2025\] Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation](re-coding_for_uncertainties_edge-awareness_semantic_concordance_for_resilient_ev.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)

</div>

<!-- RELATED:END -->
