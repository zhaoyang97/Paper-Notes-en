---
title: >-
  [Paper Note] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [CVPR 2026][Multimodal VLM][underwater image enhancement] This paper proposes a VLM-driven semantic-sensitive learning strategy that leverages LLaVA to generate object descriptions…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "underwater image enhancement"
  - "VLM"
  - "semantic guidance"
  - "cross-attention"
  - "downstream tasks"
date: 2026-05-08
content_hash: 5e3a406726ef5da9
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**Conference**: CVPR 2026
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: underwater image enhancement, VLM, semantic guidance, cross-attention, downstream tasks

## TL;DR

This paper proposes a VLM-driven semantic-sensitive learning strategy that leverages LLaVA to generate object descriptions, BLIP to construct spatial semantic guidance maps, and a dual-guidance mechanism (cross-attention injection + semantic alignment loss) to steer the UIE decoder during reconstruction. The approach yields consistent improvements in both perceptual quality and downstream detection/segmentation performance.

## Background & Motivation

**Background**: Underwater image enhancement (UIE) has seen substantial progress through deep learning methods, with notable gains on perceptual metrics such as PSNR and SSIM. However, a troubling "enhancement paradox" has emerged: images with higher visual quality do not necessarily benefit downstream object detection or semantic segmentation, and can even degrade task performance.

**Limitations of Prior Work**: Existing UIE methods are inherently "semantically blind," applying uniform global enhancement across all image regions without distinguishing semantic foreground (e.g., marine organisms, artificial objects) from background (e.g., water body). This one-size-fits-all strategy introduces imperceptible artifacts or distribution shifts that disrupt the semantic cues relied upon by downstream models. Early semantic-guided methods require high-quality pixel-level annotations to train segmentation models—annotations that are extremely scarce in underwater scenes. VLM-guided approaches based on global text prompts (e.g., "a clear underwater photo") circumvent pixel annotations but remain style-level global guidance, incapable of fine-grained content-aware enhancement.

**Key Challenge**: A fundamental conflict exists between UIE's pursuit of global visual quality and downstream tasks' need to preserve the semantic features of key objects.

**Goal**: To endow UIE with content-awareness, restoring visual quality while preserving—or even strengthening—the semantic features of critical objects, so that enhanced results serve both human perception and machine cognition simultaneously.

**Key Insight**: The open-world understanding capability of VLMs is exploited to automatically identify key objects in the image and generate spatially-grounded semantic guidance maps, which are then injected into the UIE decoder via a dual-guidance mechanism (structural guidance + explicit supervision).

**Core Idea**: Use a VLM to tell the enhancement network *what is in the image*, BLIP to indicate *where it is*, and cross-attention combined with an alignment loss to specify *where to focus enhancement efforts*.

## Method

### Overall Architecture

The pipeline consists of three stages: a degraded underwater image $I_d$ is fed into (1) LLaVA to generate a textual description $T$ of key objects; (2) BLIP's visual-text alignment to compute a spatial semantic guidance map $M_{sem}$; and (3) a dual-guidance injection mechanism—cross-attention and semantic alignment loss—applied to the decoder of any encoder-decoder UIE network, producing a semantically sensitive enhanced image $I_e$. The strategy is designed as a plug-and-play module and has been validated on five baselines: PUIE, SMDR, UIR, PFormer, and FDCE.

### Key Designs

1. **Semantic Guidance Map Generation**:

    - *Function*: Generates a single-channel spatial semantic guidance map $M_{sem} \in \mathbb{R}^{H \times W}$ for each degraded image, precisely quantifying the semantic relevance of each spatial location to key objects.
    - *Mechanism*: LLaVA first generates a textual description $T$ of key objects from the degraded image. BLIP's visual encoder $\Phi_v$ then extracts patch features $F_v = \{f_v^1, \ldots, f_v^N\}$, while its text encoder $\Phi_t$ extracts a global text feature $f_t$. The cosine similarity $s_i = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}$ is computed for each patch, followed by a semantic sharpening function $\Psi_{sharp}(s_i; \gamma, \delta) = (\max(0, \mathcal{N}(s_i) - \delta))^\gamma$ for threshold filtering and nonlinear amplification. The result is upsampled to the original image resolution.
    - *Design Motivation*: Three alternatives were compared—ViT class attention (coarse and diffuse), CLIP (sharp but with background noise), and BLIP (clean, well-defined boundaries). BLIP's fusion-alignment strategy produces the highest-quality guidance maps, precisely highlighting objects described by the text while exhibiting minimal background noise.

2. **Cross-Attention Injection Mechanism**:

    - *Function*: Injects the semantic guidance map into the reconstruction process at each decoder stage via cross-attention, directing the network to preferentially aggregate encoder features from semantically highlighted regions.
    - *Mechanism*: At decoder stage $l$, the decoder feature $d_l$ serves as the Query. The encoder skip-connection feature $e_l$, weighted by the downsampled guidance map $\tilde{M}^{(l)}$, is linearly projected into Key and Value. The output is computed as $d_l' = \text{softmax}(Q_l K_l^\top / \sqrt{d_k}) V_l$.
    - *Design Motivation*: Three injection locations were compared—Encoder only, All stages, and Decoder only. Decoder-only injection achieved the best performance, as the decoder stage directly governs the image reconstruction process; injecting guidance there allocates reconstruction capacity most efficiently. Injection at the encoder stage, by contrast, risks interfering with feature extraction.

3. **Explicit Semantic Alignment Loss $\mathcal{L}_{align}$**:

    - *Function*: Provides explicit, quantifiable supervision to complement the implicit structural guidance of cross-attention, ensuring that the spatial distribution of decoder intermediate features aligns with the semantic guidance map.
    - *Mechanism*: A dual-term constraint is applied to the decoder feature map $\mathbf{F}^{(l)}$ at stage $l$: a background suppression term $\|\mathbf{F}^{(l)} \odot (1 - \tilde{M}^{(l)})\|_F^2$ penalizes excessive activations in non-critical regions, and a foreground enhancement term $-\eta \langle \mathbf{F}^{(l)}, \tilde{M}^{(l)} \rangle$ rewards strong responses in key object regions.
    - *Design Motivation*: Cross-attention provides structural guidance whose effect is implicit; the alignment loss directly constrains feature distributions at the loss level. The two mechanisms are complementary and achieve optimal results when used in combination.

### Loss & Training

The total loss is a weighted sum of the reconstruction loss and the semantic alignment loss: $\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{align} \sum_{l \in L} \mathcal{L}_{align}^{(l)}$. The reconstruction loss $\mathcal{L}_{recon} = \|I_e - I_{gt}\|_1 + \lambda_{percep} \sum_j \|\phi_j(I_e) - \phi_j(I_{gt})\|_1$ combines an L1 pixel loss with a VGG-19-based perceptual loss. $\lambda_{align}$ is set to 0.1. Training is performed on the UIEB training set (790 paired images), with the strategy applied as a plug-and-play module to each of the five baselines.

## Key Experimental Results

### Main Results

**UIE Perceptual Quality (UIEB Test Set)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| PUIE | 21.05 | 0.869 | 0.117 |
| PUIE-SS | **23.20**(+2.15) | **0.884**(+0.015) | **0.092**(-0.025) |
| SMDR | 22.44 | 0.899 | 0.106 |
| SMDR-SS | **23.28**(+0.84) | **0.909**(+0.010) | **0.099**(-0.007) |
| UIR | 22.89 | 0.885 | 0.124 |
| UIR-SS | **24.62**(+1.73) | **0.901**(+0.016) | **0.113**(-0.011) |
| PFormer | 23.53 | 0.877 | 0.113 |
| PFormer-SS | **24.97**(+1.44) | **0.933**(+0.056) | **0.087**(-0.026) |
| FDCE | 23.66 | 0.909 | 0.111 |
| FDCE-SS | **24.63**(+0.97) | **0.927**(+0.018) | **0.093**(-0.018) |

**Downstream Task Performance (Detection mAP / Segmentation mIoU)**:

| Method | mAP↑ | mIoU↑ |
|--------|------|-------|
| Raw Input (no enhancement) | 95.43 | 68.10 |
| PUIE → PUIE-SS | 95.40 → **96.28**(+0.88) | 66.20 → **70.80**(+4.60) |
| SMDR → SMDR-SS | 95.76 → **96.98**(+1.22) | 68.18 → **73.51**(+5.33) |
| UIR → UIR-SS | 94.37 → **95.31**(+0.94) | 68.52 → **70.45**(+1.93) |
| PFormer → PFormer-SS | 95.50 → **96.87**(+1.37) | 69.34 → **74.75**(+5.41) |
| FDCE → FDCE-SS | 95.72 → **97.01**(+1.29) | 69.78 → **72.36**(+2.58) |

### Ablation Study

**Guidance Map Generation Model Comparison**:

| Model | Map Quality | Characteristics |
|-------|-------------|-----------------|
| ViT class attention | Poor | Coarse and diffuse; fails to localize objects precisely |
| CLIP | Moderate | Sharp attention but background activations present |
| BLIP | Best | Clean, well-defined boundaries, no background noise |

**Semantic Guidance Injection Location Comparison**:

| Injection Location | Performance |
|--------------------|-------------|
| Encoder only | Worst; interferes with feature extraction |
| All stages | Moderate |
| Decoder only | Best; directly governs the reconstruction process |

**Dual-Guidance Mechanism Ablation**: Using cross-attention or alignment loss individually each yields improvements; using both in combination achieves the best results, confirming the complementarity of structural guidance and explicit supervision.

### Key Findings

- All five baselines augmented with the -SS strategy show consistent gains in PSNR and SSIM; PUIE-SS achieves the largest PSNR improvement (+2.15 dB).
- Segmentation mIoU gains are the most pronounced: PFormer-SS +5.41, SMDR-SS +5.33, demonstrating the substantial benefit of semantic guidance for pixel-level classification.
- Certain baselines (e.g., PUIE, UIR) exhibit the "enhancement paradox"—downstream performance after enhancement falls below that of the raw input—whereas all -SS variants consistently surpass the raw input baseline.
- Positive trends on UIQM/UCIQE metrics on the non-reference datasets U45 and Challenge60 indicate that semantic guidance does not cause overfitting.

## Highlights & Insights

- **VLMs as Zero-Annotation Semantic Prior Sources**: The LLaVA + BLIP pipeline elegantly circumvents the scarcity of pixel-level annotations in underwater scenes, enabling spatially grounded semantic guidance without any additional labeling. This strategy generalizes readily to any image enhancement scenario lacking dense annotations.

- **Direct Confrontation with the Enhancement Paradox**: The experiments clearly demonstrate that certain baselines suffer degraded downstream performance after enhancement. The -SS strategy resolves this issue consistently across all five baselines, establishing semantically aware enhancement as the critical bridge between low-level visual improvement and high-level machine cognition.

- **Engineering Value of the Plug-and-Play Design**: The strategy does not modify the UIE network architecture itself; it only injects guidance signals and loss terms at the decoder. It can therefore be directly applied to any encoder-decoder UIE model, and the consistent gains across five architecturally diverse baselines confirm the generality of the approach.

## Limitations & Future Work

- Both LLaVA and BLIP are frozen pretrained models whose additional forward passes introduce non-trivial inference overhead, potentially limiting deployment on resource-constrained underwater platforms.
- The quality of the semantic guidance map depends entirely on the VLM's ability to interpret degraded images; under severe degradation (e.g., extremely low visibility), the VLM may fail to identify objects correctly.
- Validation is currently limited to encoder-decoder UIE architectures; whether the strategy applies equally to GAN-based or diffusion-based enhancement methods remains unexplored.

## Related Work & Insights

- **vs. Traditional Semantic Guidance (Liao et al., Yan et al.)**: Traditional methods rely on pixel-level annotations to train segmentation models for semantic priors, which are prone to errors when annotations are scarce in underwater scenes. This work achieves zero-annotation semantic prior generation via VLMs.
- **vs. CLIP Global Text Guidance (Liu et al.)**: CLIP-based methods use global style prompts to guide enhancement, which remains a global strategy. This work generates content-specific object descriptions mapped back to spatial locations, enabling fine-grained region-level guidance.
- **vs. Joint Training with Downstream Perception**: Joint training approaches optimize the enhancement network and downstream task network end-to-end, but require a customized model for each downstream task. The proposed strategy is decoupled from downstream tasks, allowing a single enhancement pass to serve multiple tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces VLMs' open-world understanding into UIE with a complete spatially grounded semantic guidance pipeline; the perspective is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five baselines, three UIE datasets, two downstream tasks, and comprehensive ablations constitute a very thorough experimental evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is rigorously described, and figures and tables are informative.
- **Value**: ⭐⭐⭐⭐ — The enhancement paradox addresses a practically significant problem; the plug-and-play design is highly practical and offers broad inspiration for underwater vision and other degraded-image scenarios.

---

## Related Papers

- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[CVPR 2026\] From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering](from_masks_to_pixels_and_meaning_a_new_taxonomy_benchmark_and_metrics_for_vlm_im.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)
- [\[CVPR 2026\] VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)
- [\[CVPR 2026\] It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models](its_time_to_get_it_right_improving_analog_clock_reading_and_clock-hand_spatial_r.md)

</div>

<!-- RELATED:END -->
