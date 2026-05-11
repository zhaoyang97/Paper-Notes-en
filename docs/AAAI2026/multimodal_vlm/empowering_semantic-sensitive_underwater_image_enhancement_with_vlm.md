---
title: >-
  [Paper Note] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [AAAI2026][Multimodal VLM][Underwater Image Enhancement] This work leverages a VLM to generate spatially-aware semantic guidance maps, and introduces a dual-guidance mechanism comprising cross-attention injection and a s…
tags:
  - "AAAI2026"
  - "Multimodal VLM"
  - "Underwater Image Enhancement"
  - "Vision-Language Model"
  - "Semantic Guidance"
  - "Cross-Attention"
  - "Downstream Tasks"
date: 2026-05-08
content_hash: 799f6e9b56dc15c7
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**Conference**: AAAI2026
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Underwater Image Enhancement, Vision-Language Model, Semantic Guidance, Cross-Attention, Downstream Tasks

## TL;DR
This work leverages a VLM to generate spatially-aware semantic guidance maps, and introduces a dual-guidance mechanism comprising cross-attention injection and a semantic alignment loss to endow underwater image enhancement networks with semantic awareness, yielding enhanced results that benefit both human perception and downstream detection/segmentation tasks.

## Background & Motivation
Underwater image enhancement (UIE) is a critical preprocessing step in marine exploration, underwater robotics, and related domains. Although current deep learning-based UIE methods can produce visually satisfying images, a fundamental contradiction persists: **improvements in perceptual quality do not consistently translate into performance gains on downstream tasks (detection, segmentation)**. Existing methods are essentially "semantically blind"—they pursue globally uniform enhancement without distinguishing semantically salient regions (e.g., marine organisms, underwater objects) from non-salient regions (e.g., water background), which introduces distribution shifts or latent artifacts during enhancement that disrupt semantic cues critical to downstream models.

Prior solutions exhibit notable shortcomings:
- **Semantic segmentation map-based methods**: rely on pixel-level annotations, which are extremely scarce in underwater scenarios.
- **Global text prompt-based methods** (e.g., CLIP-guided style transfer): exploit VLM capabilities but remain a "one-size-fits-all" strategy that does not attend to specific semantic content within the image.

## Core Problem
How can a UIE model perceive and focus on key semantic regions during enhancement, such that the enhanced output is both visually superior and preserves semantic features beneficial to downstream machine vision tasks?

## Method

### Overall Architecture
The proposed approach is a **plug-and-play semantic-sensitive learning strategy** (denoted -SS) that can be integrated into various encoder–decoder UIE baseline models. The core pipeline consists of three steps:

### 1. Semantic Guidance Map Generation
- Given a degraded underwater image $I_d$, **LLaVA** (a VLM) generates a textual description $T$ of the key objects in the scene.
- The pretrained **BLIP** visual encoder $\Phi_v$ and text encoder $\Phi_t$ extract patch features $F_v \in \mathbb{R}^{N \times C}$ and a global text feature $f_t \in \mathbb{R}^C$, respectively.
- The cosine similarity between each patch and the text is computed as $s_i = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}$.
- The similarity distribution is processed through a **semantic sharpening function**: $s_i' = (\max(0, \mathcal{N}(s_i) - \delta))^\gamma$, where $\delta$ is a threshold that filters low-response noise and $\gamma > 1$ is a power-law exponent that nonlinearly amplifies score differences.
- The one-dimensional score sequence is upsampled to the original image resolution to produce the semantic guidance map $M_{sem} \in \mathbb{R}^{H \times W}$.

BLIP is selected over CLIP or ViT based on ablation study results showing that BLIP's fusion-based alignment strategy produces the cleanest heatmaps with the sharpest boundaries and highest spatial precision, whereas CLIP tends to generate spurious activations in background regions and ViT's class attention maps are overly coarse.

### 2. Cross-Attention Injection Mechanism (Structural Guidance)
At each stage $l$ of the UIE network **decoder**:
- Decoder features $d_l$ serve as Query $Q_l$.
- Encoder skip-connection features $e_l$, element-wise weighted by $M_{sem}$ downsampled to the corresponding resolution $\tilde{M}^{(l)}$, are projected into Key $K_l$ and Value $V_l$.
- Standard attention is computed as: $d_l' = \text{softmax}(\frac{Q_l K_l^\top}{\sqrt{d_k}}) V_l$.
- This design causes the decoder to preferentially extract information from semantically highlighted encoder features.

### 3. Explicit Semantic Alignment Loss (Feature Supervision)
For the feature map $\mathbf{F}^{(l)}$ at decoder stage $l$, two loss terms are designed:
- **Background suppression term**: $\|\mathbf{F}^{(l)} \odot (1 - \tilde{M}^{(l)})\|_F^2$ — penalizes unnecessary strong activations in non-salient regions.
- **Foreground enhancement term**: $-\eta \langle \mathbf{F}^{(l)}, \tilde{M}^{(l)} \rangle$ — rewards strong responses consistent with the semantic guidance map.

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{align} \sum_{l \in L} \mathcal{L}_{align}^{(l)}$$
where $\mathcal{L}_{recon}$ comprises an L1 loss and a VGG-19-based perceptual loss, and $\lambda_{align}$ is empirically set to 0.1.

## Key Experimental Results

### UIE Perceptual Quality (UIEB Dataset, Full-Reference)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| PFormer | 23.53 | 0.877 | 0.113 |
| PFormer-SS | **24.97**(+1.44) | **0.933**(+0.056) | **0.087**(-0.026) |
| FDCE | 23.66 | 0.909 | 0.111 |
| FDCE-SS | **24.63**(+0.97) | **0.927**(+0.018) | **0.093**(-0.018) |
| UIR-SS | **24.62**(+1.73) | 0.901 | 0.113 |

All five baseline models show consistent improvements in PSNR and SSIM after applying -SS.

### Downstream Task — Object Detection (Trash-ICRA19)

| Method | mAP↑ |
|--------|------|
| SMDR | 95.76 |
| SMDR-SS | **96.98**(+1.22) |
| FDCE | 95.72 |
| FDCE-SS | **97.01**(+1.29) |

### Downstream Task — Semantic Segmentation (SUIM)

| Method | mIoU↑ |
|--------|-------|
| PFormer | 69.34 |
| PFormer-SS | **74.75**(+5.41) |
| SMDR | 68.18 |
| SMDR-SS | **73.51**(+5.33) |
| PUIE | 66.20 |
| PUIE-SS | **70.80**(+4.60) |

Gains on semantic segmentation are particularly pronounced, with mIoU improving by 2–5 percentage points across methods and reaching up to 15 percentage points on certain categories (e.g., RO).

### Ablation Study
- **Guidance map injection location**: decoder-only > encoder+decoder > encoder-only, confirming that injecting semantic guidance during the reconstruction stage is most effective.
- **VLM selection**: BLIP > CLIP > ViT; BLIP's fusion-based alignment strategy yields the highest-quality guidance maps.

## Highlights & Insights
1. **Plug-and-play design**: The strategy integrates seamlessly into any encoder–decoder UIE model, delivering consistent improvements across five distinct baselines.
2. **Dual-guidance mechanism**: Cross-attention provides implicit structural guidance while the semantic alignment loss provides explicit feature supervision, with the two complementing each other.
3. **Bridging the perception–cognition gap**: This work is the first to systematically employ VLM-driven semantic understanding to address the "visually pleasing but task-ineffective" problem in underwater enhancement.
4. **Semantic sharpening function design**: The combination of thresholding and power-law transformation converts a smooth similarity distribution into a high-contrast guidance map.

## Limitations & Future Work
1. **Inference overhead**: Generating text descriptions via LLaVA and computing alignment via BLIP may substantially increase inference latency; computational costs are not discussed in the paper.
2. **VLM dependency**: The quality of the guidance map is entirely contingent on the VLM's ability to interpret degraded images; under severe degradation, VLM descriptions may be inaccurate.
3. **Limited training data scale**: UIEB contains only 790 training images; performance on larger-scale datasets remains unverified.
4. **Modest gains on no-reference metrics**: Improvements on U45 and Challenge60 in terms of UIQM/UCIQE are less pronounced than those observed for full-reference metrics.
5. **Occasional fluctuations on downstream tasks**: Marginal performance drops are observed for certain categories (e.g., FV, RI), indicating that semantic guidance is not entirely stable.

## Related Work & Insights
- **vs. traditional segmentation-guided methods** (Wu et al. 2023): This work replaces pixel-level annotations with VLM-generated guidance, circumventing the scarcity of underwater annotations.
- **vs. CLIP global text guidance** (Liu et al. 2024): This work advances from global style guidance to spatial, patch-level semantic guidance, enabling content-aware fine-grained processing.
- **vs. joint training schemes** (Yu et al. 2023): The plug-and-play strategy does not require task-specific model customization for downstream applications, offering greater generality.

The core idea of **using VLM-generated spatial semantic priors to guide low-level vision tasks** holds broad transfer potential and may be applied to dehazing, deraining, low-light enhancement, and related tasks. The semantic sharpening function design offers a useful reference for post-processing VLM-derived guidance maps. The dual-guidance injection paradigm (structural + supervisory) is generalizable to other image restoration tasks that require prior-guided processing.

## Rating
- **Novelty**: 7/10 — The concept of VLM-driven spatial semantic guidance is novel, though the cross-attention injection mechanism itself is relatively standard.
- **Experimental Thoroughness**: 8/10 — Five baselines, three UIE datasets, two downstream tasks, and ablation studies provide comprehensive coverage.
- **Writing Quality**: 8/10 — Motivation is clearly articulated and experiments are systematically organized.
- **Value**: 7/10 — The plug-and-play design offers strong practical utility and is valuable to the underwater vision community; however, the absence of computational cost analysis hinders evaluation of real-world deployment feasibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[AAAI 2026\] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation](zero-reference_joint_low-light_enhancement_and_deblurring_via_visual_autoregress.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)
- [\[AAAI 2026\] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)

</div>

<!-- RELATED:END -->
