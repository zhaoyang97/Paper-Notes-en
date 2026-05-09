---
title: >-
  [Paper Note] ViHOI: Human-Object Interaction Synthesis with Visual Priors
description: >-
  [CVPR 2026][Image Generation][Human-Object Interaction Generation] This paper proposes ViHOI, a plug-and-play framework that leverages a VLM to extract decoupled visual and textual priors from 2D reference images, compresses them into compact condition tokens via Q-Former, and injects them into a diffusion model to enhance HOI motion generation quality. At inference time, a text-to-image model synthesizes reference images to enable strong generalization to unseen objects.
tags:
  - CVPR 2026
  - Image Generation
  - Human-Object Interaction Generation
  - Visual Priors
  - Diffusion Models
  - VLM
  - Q-Former
date: 2026-05-08
content_hash: 7670e8ff49a05b61
---

# ViHOI: Human-Object Interaction Synthesis with Visual Priors

**Conference**: CVPR 2026
**arXiv**: [2603.24383](https://arxiv.org/abs/2603.24383)
**Code**: [https://github.com/MPI-Lab/ViHOI](https://github.com/MPI-Lab/ViHOI)
**Area**: Image Generation / Motion Generation
**Keywords**: Human-Object Interaction Generation, Visual Priors, Diffusion Models, VLM, Q-Former

## TL;DR

This paper proposes ViHOI, a plug-and-play framework that leverages a VLM to extract decoupled visual and textual priors from 2D reference images, compresses them into compact condition tokens via Q-Former, and injects them into a diffusion model to enhance HOI motion generation quality. At inference time, a text-to-image model synthesizes reference images to enable strong generalization to unseen objects.

## Background & Motivation

1. **Background**: 3D human-object interaction (HOI) motion generation aims to synthesize realistic and physically plausible sequences of human-object interactions, with important applications in VR, animation, and robotics. Diffusion models have been widely adopted for HOI generation tasks in recent years.

2. **Limitations of Prior Work**: The generation quality of existing methods is constrained by the quality of conditioning signals. HOI involves continuous spatial state changes and plausible interaction relationships, yet textual annotations in datasets typically provide only abstract descriptions (e.g., "pick up a box"), lacking geometric and spatial priors regarding object shape, size, and body pose, forcing models to contend with a complex one-to-many learning problem.

3. **Key Challenge**: Existing enhancement strategies follow two lines: semantic augmentation (LLM-expanded text descriptions) and physical constraints (contact points, kinematic priors). The former still lacks structured knowledge to precisely couple motion with object geometry, while the latter tends to focus only on local interaction regions, neglecting the global dynamics and coherence of whole-body motion.

4. **Goal**: How to effectively exploit the rich visual interaction priors embedded in readily available 2D images—object shape, scale, and human-object spatial relationships—to enhance the fidelity and physical plausibility of HOI motion generation.

5. **Key Insight**: The authors argue that 2D images offer a rich set of visual interaction priors, including object shape, scale, and human-object spatial relationships. Employing a VLM to simultaneously extract image and text information naturally ensures semantic alignment between the two modalities.

6. **Core Idea**: A VLM is used to extract decoupled visual and textual priors from 2D reference images; these are compressed via Q-Former and injected into a motion diffusion model. During training, GT motion renderings ensure semantic alignment; during inference, a text-to-image model synthesizes reference images to achieve generalization.

## Method

### Overall Architecture

ViHOI consists of two core components: a VLM-based Prior Extractor and a Vision-aware HOI Generator. The inputs are a set of 2D reference images and a text description. Qwen2.5-VL extracts visual priors and textual priors from different layers respectively; two Q-Former-based Prior Adaptors compress these into compact tokens, which are then injected as conditions into a DiT-based motion diffusion model and guide HOI motion synthesis via self-attention. During training, images rendered from GT motion sequences are used; during inference, reference images are synthesized by a text-to-image model.

### Key Designs

1. **Decoupled Priors Extraction**:
    - Function: Extract visual and textual priors from different layers of the VLM separately.
    - Mechanism: Exploiting the property that shallower VLM layers retain richer visual detail while deeper layers possess stronger text encoding capacity, visual priors $E_v$ are extracted from layer 3 of Qwen2.5-VL (preserving rich geometric and spatial cues), and textual priors $E_t$ from layer 12 (capturing semantic information from motion descriptions). A structured prompt is designed to direct the VLM toward key interaction cues (object shape, size, contact regions), ensuring the extracted priors are task-aware.
    - Design Motivation: Different VLM layers attend differently to visual versus textual content; the decoupling strategy provides the most informative priors for each modality, outperforming extraction from a single layer. Ablation experiments confirm that the V3-T12 combination achieves the best performance among all layer combinations tested.

2. **Q-Former Prior Adaptor**:
    - Function: Compress the high-dimensional, variable-length token sequences from intermediate VLM layers into compact, fixed-dimensional condition signals.
    - Mechanism: A linear projection $Z_v = \text{LayerNorm}(\text{Linear}(E_v))$ first aligns the dimensionality; learnable queries $q_v$ then interact with the projected features via two-layer cross-attention $c_v = \text{CrossAttention}(q_v, Z_v, Z_v)$, distilling the rich priors into a single compact token. Separate Q-Former adaptors are used for visual and textual priors respectively.
    - Design Motivation: Intermediate VLM layer outputs are high-dimensional variable-length sequences that are highly challenging to use directly as diffusion model conditions. The Q-Former adaptively extracts the information most relevant to HOI synthesis from redundant VLM features. Ablation experiments show that replacing Q-Former with simple average pooling leads to a dramatic performance drop.

3. **Reference Image Generation Strategy (Training/Inference Decoupling)**:
    - Function: Address the difference in how 2D reference images are obtained between training and inference stages.
    - Mechanism: During training, 2D images are rendered from GT motion sequences; contact labels are used to select three key frames corresponding to the start, middle, and end of the interaction, ensuring strict semantic consistency between visual priors and target motions. During inference, the text-to-image model Nano Banana synthesizes three temporally coherent HOI reference images, leveraging its embedded world knowledge to enhance generalization.
    - Design Motivation: Using rendered images during training ensures strict semantic alignment at low cost, avoiding the need to collect large-scale image-motion paired data. At inference time, the world knowledge of the text-to-image model enables generalization; despite a style gap (clean renderings vs. synthesized images), the VLM prior extractor can still identify underlying motion-relevant features.

### Loss & Training

- The training objective is the standard diffusion model reconstruction loss: $\mathcal{L} = \mathbb{E}_{t,x_0}[\|x_0 - f_\theta(x_t, t, c)\|^2]$
- VLM parameters are frozen during training; only the two Q-Former Prior Adaptors and the HOI Generator are jointly trained.
- The condition $c = \{c_v, c_t\}$ consists of compact visual and textual prior tokens.

## Key Experimental Results

### Main Results

| Dataset | Metric | CHOIS+ViHOI | CHOIS | Gain |
|--------|------|------|----------|------|
| FullBodyManipulation | FID↓ | 0.68 | 0.77 | -11.7% |
| FullBodyManipulation | R-Precision Top-3↑ | 0.79 | 0.73 | +8.2% |
| FullBodyManipulation | MPJPE↓ | 14.97 | 15.43 | -3.0% |
| FullBodyManipulation | $C_{F_1}$↑ | 0.75 | 0.70 | +7.1% |
| BEHAVE | FID↓ | 2.02 | 4.99 | -59.5% |
| BEHAVE | MPJPE↓ | 14.58 | 15.42 | -5.4% |
| Unseen Objects | FID↓ | 2.02 | 4.99 | -59.5% |

### Ablation Study

| Configuration | R-Precision Top-3 | FID↓ | MPJPE↓ | Note |
|------|---------|------|------|------|
| ViHOI (Full, V3-T12) | 0.79 | 0.68 | 14.97 | Best combination |
| ViHOI-Pool (Avg. Pooling) | 0.32 | 26.03 | 22.62 | Q-Former→pooling, sharp drop |
| ViHOI-CLIP (CLIP text) | 0.75 | 0.69 | 17.57 | VLM text→CLIP, degraded |
| T12-only (Text prior only) | 0.72 | 1.28 | 17.49 | No visual prior, notable degradation |
| V12-T12 | 0.75 | 0.87 | 15.90 | Visual layer too deep, detail lost |
| V24-T24 | 0.61 | 3.15 | 16.94 | Both layers too deep, poor results |

### Key Findings

- Q-Former is critical: replacing it with simple pooling causes FID to surge from 0.68 to 26.03, demonstrating that an effective prior compression mechanism is indispensable.
- Visual priors significantly outperform text-only priors: incorporating visual priors reduces MPJPE from 17.49 to 14.97, confirming the importance of geometric-spatial information from 2D images for motion generation.
- VLM textual priors outperform CLIP: text embeddings extracted from the VLM are richer than those from CLIP, reducing MPJPE from 17.57 to 14.97.
- Strong generalization to unseen objects: leveraging the world knowledge of the text-to-image model, ViHOI generates plausible motions for unseen objects and on the 3D-FUTURE dataset.
- Plug-and-play property: ViHOI successfully improves performance across three different baseline models—MDM, ROG, and CHOIS.

## Highlights & Insights

- The paradigm of "images as motion priors" is highly elegant—readily available 2D images provide the geometric and spatial priors needed for 3D motion generation, avoiding complex physical constraint modeling.
- The training/inference-decoupled reference image strategy cleverly resolves the data bottleneck: rendered images ensure alignment during training, while a text-to-image model introduces world knowledge for generalization at inference time.
- The use of Q-Former to compress variable-length, high-dimensional VLM features into fixed-dimensional tokens represents a general design pattern for bridging large foundation models with downstream tasks.
- The plug-and-play design allows direct enhancement of any existing HOI motion diffusion model.

## Limitations & Future Work

- Acknowledged limitation: the datasets used lack fine-grained hand annotations, preventing accurate generation of detailed finger motion sequences.
- Dependence on text-to-image model quality: the plausibility of reference images synthesized at inference time directly affects the quality of generated motions.
- Using only three key frames may be insufficient to represent complex long-horizon interaction processes.
- The potential of video generation models as a prior source remains unexplored; video can provide richer temporal dynamics than static images.

## Related Work & Insights

- **vs. SemGeoMo**: SemGeoMo enhances text with an LLM and uses affordance maps as geometric priors, achieving good contact quality but insufficient whole-body motion accuracy. ViHOI improves both contact quality and joint accuracy simultaneously via visual priors, better balancing local precision and global coherence.
- **vs. CHOIS**: CHOIS uses sparse object waypoints as global path priors; ViHOI provides richer visual priors and outperforms CHOIS comprehensively on both FID and MPJPE.
- **vs. video generation + 3D recovery methods**: Such methods rely on 2D-to-3D pose estimation, suffering from jitter and temporal inconsistency. ViHOI encodes visual priors as compact tokens to implicitly guide generation, avoiding explicit pose recovery.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm of images as motion priors is novel; the VLM layer-decoupled extraction strategy is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, three baseline models, unseen object generalization, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and well-structured method presentation.
- Value: ⭐⭐⭐⭐ The plug-and-play framework is highly practical; the paradigm innovation is transferable.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation](haltnav_reactive_visual_halting_over_lightweight_topological_priors_for_robust_v.md)
- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](ahs_adaptive_head_synthesis.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)

<!-- RELATED:END -->
