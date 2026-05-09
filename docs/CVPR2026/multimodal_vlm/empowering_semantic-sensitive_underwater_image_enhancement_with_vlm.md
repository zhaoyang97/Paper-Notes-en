---
title: >-
  [Paper Note] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [CVPR 2026][Multimodal VLM][Underwater Image Enhancement] This paper proposes a plug-and-play strategy (-SS) that leverages VLMs to generate semantic guidance maps. Through a dual-guidance mechanism comprising cross-attention injection and a semantic alignment loss, the approach directs underwater image enhancement models to focus on semantically critical regions during restoration, yielding significant improvements in perceptual quality as well as downstream detection and segmentation performance.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Underwater Image Enhancement
  - Semantic Guidance
  - VLM
  - Cross-Attention
  - Downstream Task Awareness
date: 2026-05-08
content_hash: ef0e2a9d28119d5b
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**Conference**: CVPR 2026
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Underwater Image Enhancement, Semantic Guidance, VLM, Cross-Attention, Downstream Task Awareness

## TL;DR

This paper proposes a plug-and-play strategy (-SS) that leverages VLMs to generate semantic guidance maps. Through a dual-guidance mechanism comprising cross-attention injection and a semantic alignment loss, the approach directs underwater image enhancement models to focus on semantically critical regions during restoration, yielding significant improvements in perceptual quality as well as downstream detection and segmentation performance.

## Background & Motivation

Underwater image enhancement (UIE) is a critical preprocessing step in marine exploration, biological monitoring, and related fields. While existing deep learning methods have achieved notable progress on visual quality metrics, a fundamental contradiction remains: **better enhancement ≠ better downstream task performance**.

**Limitations of Prior Work**: Existing UIE methods are "semantically blind"—they pursue globally uniform enhancement without distinguishing semantic foreground regions (e.g., marine organisms, artifacts) from non-focal regions (e.g., background water), such that the induced distribution shift may actually harm downstream detection and segmentation models. Early semantic guidance approaches rely on pixel-level annotations, which are extremely scarce in underwater scenarios, while VLM-based global text prompts (e.g., "a clear underwater photo") remain one-size-fits-all strategies incapable of fine-grained, content-aware enhancement.

**Starting Point**: The paper exploits the open-world understanding capability of VLMs to automatically generate image-content-relevant textual descriptions, then employs a text-image alignment model to map semantics back to spatial locations, producing pixel-level semantic guidance maps. These maps are injected into the enhancement network's decoder via a dual mechanism, enabling the network to "know what to prioritize during restoration."

## Method

### Overall Architecture

A three-stage pipeline: (1) A VLM (LLaVA) generates textual descriptions of key objects from degraded images → (2) A text-image alignment model (BLIP) produces a spatial semantic guidance map $M_{sem}$ → (3) The guidance map is injected into the UIE network decoder via cross-attention and an alignment loss. The design is plug-and-play and can be adapted to any encoder-decoder UIE model.

### Key Designs

1. **Semantic Guidance Map Generation**: BLIP's visual encoder extracts patch features $F_v \in \mathbb{R}^{N \times C}$, and its text encoder extracts global text features $f_t \in \mathbb{R}^C$. The cosine similarity between each patch and the text is computed as $s_i = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}$, followed by a semantic sharpening function to enhance discriminability:
    $s'_i = \Psi_{\text{sharp}}(s_i; \gamma, \delta) = (\max(0, \mathcal{N}(s_i) - \delta))^\gamma$
   where $\delta$ is a threshold that filters low-relevance noise, and $\gamma > 1$ nonlinearly amplifies score differences. The 1D score sequence is then upsampled to the original image resolution to obtain $M_{sem}$. The sharpening is motivated by the observation that raw similarity distributions tend to be overly smooth, providing insufficient guidance signals.

2. **Cross-Attention Injection Mechanism**: At each decoder stage $l$, $M_{sem}$ is downsampled to the corresponding resolution $\tilde{M}^{(l)}$ and used to element-wise weight the encoder skip-connection features $e_l$, forming Keys and Values; the decoder features $d_l$ serve as Queries:
    $d'_l = \text{softmax}\left(\frac{Q_l K_l^\top}{\sqrt{d_k}}\right) V_l$
   This allows the decoder to preferentially extract information from semantically "illuminated" encoder features, achieving structured guidance.

3. **Explicit Semantic Alignment Loss**: Since cross-attention provides only implicit guidance, an explicit supervision signal $\mathcal{L}_{\text{align}}$ is additionally introduced to directly constrain the spatial distribution of intermediate decoder feature maps to align with the guidance map:
    $\mathcal{L}_{\text{align}}^{(l)} = \underbrace{\|\mathbf{F}^{(l)} \odot (1-\tilde{M}^{(l)})\|_F^2}_{\text{background suppression}} - \underbrace{\eta \langle \mathbf{F}^{(l)}, \tilde{M}^{(l)} \rangle}_{\text{foreground enhancement}}$
   The background suppression term penalizes strong activations in non-critical regions, while the foreground enhancement term rewards alignment between responses in critical regions and the guidance map.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{align}} \sum_{l \in L} \mathcal{L}_{\text{align}}^{(l)}$$

- Reconstruction loss: L1 loss + VGG-19 perceptual loss
- Semantic alignment loss weight: $\lambda_{\text{align}} = 0.1$
- Generality validated on 5 different baselines (PUIE, SMDR, UIR, PFormer, FDCE)

## Key Experimental Results

### Main Results (Perceptual Quality, UIEB Dataset)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Gain |
|--------|--------|--------|---------|------|
| PFormer | 23.53 | 0.877 | 0.113 | - |
| PFormer-SS | **24.97** | **0.933** | **0.087** | +1.44/+0.056/−0.026 |
| UIR | 22.89 | 0.885 | 0.124 | - |
| UIR-SS | **24.62** | **0.901** | **0.113** | +1.73/+0.016/−0.011 |
| FDCE | 23.66 | 0.909 | 0.111 | - |
| FDCE-SS | **24.63** | **0.927** | **0.093** | +0.97/+0.018/−0.018 |

### Downstream Task Improvements (Detection mAP / Segmentation mIoU)

| Method | mAP ↑ | Seg. mIoU ↑ | Note |
|--------|-------|-------------|------|
| PFormer | 95.50 | 69.34 | baseline |
| PFormer-SS | **96.87** (+1.37) | **74.75** (+5.41) | Significant segmentation gain |
| SMDR | 95.76 | 68.18 | baseline |
| SMDR-SS | **96.98** (+1.22) | **73.51** (+5.33) | Consistent improvement |
| PUIE | 95.40 | 66.20 | baseline |
| PUIE-SS | **96.28** (+0.88) | **70.80** (+4.60) | mIoU gain of 4.6 |

### Key Findings

- **Consistent improvement**: The -SS strategy improves both perceptual quality and downstream task performance across all 5 baselines.
- **Segmentation gains are particularly pronounced**: mIoU improves by 2.58–5.41, indicating that semantic guidance effectively preserves structural information of critical objects.
- In PFormer-SS, the IoU for the RO (robot) category increases substantially from 36.23 to 51.52 (+15.29), demonstrating that semantic guidance is most beneficial for low-contrast, small-scale targets.

## Highlights & Insights

- **Plug-and-play design**: No modification to the baseline network architecture is required; only a guidance module is injected into the decoder alongside an auxiliary loss, offering strong practical applicability.
- **Implicit + explicit dual guidance**: Cross-attention provides structural guidance (modifying information flow), while the alignment loss provides direct supervision (constraining feature distributions); the two mechanisms are complementary.
- **Leveraging VLMs to bypass annotation bottlenecks**: Pixel-level semantic annotations are not required; semantic guidance is automatically generated via the zero-shot capabilities of VLM+CLIP, addressing the annotation scarcity problem in underwater scenarios.
- **Exposing an important problem**: The paper reveals the "enhancement paradox" in the UIE field—enhancement results with superior visual quality may be detrimental to machine perception.

## Limitations & Future Work

- The approach depends on the quality of the VLM (LLaVA) and CLIP (BLIP); severe degradation that causes VLM recognition failure may render the guidance map unreliable.
- Training incurs additional forward-pass overhead from VLM+BLIP (though semantic maps can be precomputed and cached at inference time).
- Validation is limited to underwater scenarios; generalization to other degradation types such as haze and low-light conditions remains unexplored.
- The semantic sharpening hyperparameters ($\gamma$, $\delta$) may require scenario-specific tuning.

## Related Work & Insights

- Unlike methods that use CLIP as a global style discriminator, this work employs VLM+CLIP to generate fine-grained, spatial-location-level semantic guidance.
- The foreground enhancement / background suppression formulation of the alignment loss can be generalized to other tasks requiring spatial attention guidance.
- The paper establishes a general paradigm for "task-oriented enhancement": rather than optimizing for visual fidelity, the objective is to optimize for downstream task performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The VLM-driven semantic guidance with dual injection mechanism is novel, though the overall approach is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated on 5 baselines, 3 UIE datasets, and 2 downstream tasks; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method figures are informative, and the structure flows well.
- **Value**: ⭐⭐⭐⭐ High practical utility as a plug-and-play module; the revealed enhancement paradox offers meaningful insights.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_visionlanguage_models_via.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)

<!-- RELATED:END -->
