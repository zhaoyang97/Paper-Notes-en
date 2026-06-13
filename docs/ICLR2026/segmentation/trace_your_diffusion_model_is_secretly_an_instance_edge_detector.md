---
title: >-
  [Paper Note] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector
description: >-
  [ICLR 2026][Segmentation][Diffusion Models] This work identifies an "Instance Emergence Point" (IEP) in the denoising trajectory of text-to-image diffusion models…
tags:
  - "ICLR 2026"
  - "Segmentation"
  - "Diffusion Models"
  - "Instance Edges"
  - "Self-Attention"
  - "IEP"
  - "Unsupervised Segmentation"
date: 2026-05-08
content_hash: 5eca387d29e407a1
---

# TRACE: Your Diffusion Model is Secretly an Instance Edge Detector

**Conference**: ICLR 2026
**arXiv**: [2503.07982](https://arxiv.org/abs/2503.07982)  
**Code**: [Project Page](https://shjo-april.github.io/TRACE)  
**Area**: Instance Segmentation / Panoptic Segmentation
**Keywords**: Diffusion Models, Instance Edges, Self-Attention, IEP, Unsupervised Segmentation

## TL;DR

This work identifies an "Instance Emergence Point" (IEP) in the denoising trajectory of text-to-image diffusion models, at which self-attention exhibits sharp divergence changes at object boundaries. TRACE leverages IEP localization, ABDiv edge extraction, and single-step distillation to generate high-quality instance edges with an 81× inference speedup—requiring no instance-level annotations—improving unsupervised instance segmentation by +5.1 AP and surpassing point-supervised panoptic segmentation with tag-level supervision by +1.7 PQ.

## Background & Motivation

**Background**: Instance and panoptic segmentation have long relied on dense annotations (masks/boxes/points), which are costly and suffer from inter-annotator inconsistency. Unsupervised approaches (e.g., MaskCut) cluster ViT semantic features, but ViTs are optimized for cross-image semantic similarity rather than intra-image instance separation, frequently merging adjacent objects of the same class or fragmenting a single instance. Weakly supervised methods require at least point annotations to distinguish instances.

**Limitations of Prior Work**: (1) Unsupervised methods depend on self-supervised ViT features that are insufficient at the instance level—merging neighboring same-class objects is a fundamental failure mode; (2) depth-assisted approaches (CutS3D) fail on adjacent objects at similar depths; (3) tag-level weak supervision already approaches fully supervised accuracy in semantic segmentation (99% on VOC), yet bridging from semantic to panoptic segmentation still requires point or box annotations.

**Key Challenge**: Semantic features excel at "knowing what" but struggle at "telling who from whom"—instance separation requires a fundamentally different signal source.

**Goal**: To identify an annotation-free, instance-level signal that complements semantic features for instance separation.

**Key Insight**: Diffusion models progressively evolve from noise → instance structure → semantic content during denoising—at specific timesteps, self-attention transiently but clearly encodes instance boundaries.

**Core Idea**: The self-attention of diffusion models is a hidden instance edge annotator—sharp divergence changes in attention distributions across boundaries constitute the instance boundary signal.

## Method

### Overall Architecture

TRACE consists of three stages: (1) **IEP (Instance Emergence Point)**: the KL divergence between consecutive self-attention maps is computed along the denoising trajectory, and the timestep $t^*$ corresponding to the divergence peak is identified to obtain instance-aware self-attention $SA_{\text{inst}}$; (2) **ABDiv (Attention Boundary Divergence)**: relative attention divergence between spatial neighbors is computed on $SA_{\text{inst}}$ to produce a pseudo-edge map $E$; (3) **Single-step self-distillation**: the diffusion backbone is fine-tuned with LoRA and a lightweight edge decoder $\mathcal{G}_\phi$ is trained, enabling edge prediction in a single forward pass at $t=0$ with an 81× inference speedup. The generated edges are integrated into downstream segmentation via Background-Guided Propagation.

### Key Designs

1. **Instance Emergence Point (IEP)**:

    - Function: Automatically locates the timestep at which instance structure is most salient during denoising.
    - Mechanism: KL divergence between self-attention maps at adjacent timesteps is computed along the denoising trajectory: $t^* = \arg\max_t D_{\text{KL}}(SA(X_{t_{\text{prev}}}) \| SA(X_t))$. Early in denoising, attention is nearly random; in the middle stage, instance boundaries emerge (divergence peak); later, attention stabilizes into semantic content. A fixed step size of 100 yields stable results.
    - Design Motivation: The denoising process undergoes a reverse transition from semantic → instance → noise—IEP precisely captures the inflection point of this transition. KL divergence is more sensitive than L2/L1 to subtle differences between probability distributions, outperforming L2 by 5.6 AP$^{mk}$ (9.4 vs. 3.8).

2. **Attention Boundary Divergence (ABDiv)**:

    - Function: Converts instance-aware self-attention maps into edge maps.
    - Mechanism: For each pixel $(i,j)$, the sum of KL divergences between opposing neighbors in the four cardinal directions is computed: $\text{ABDiv}(SA)_{i,j} = D_{\text{KL}}(SA_{i+1,j} \| SA_{i-1,j}) + D_{\text{KL}}(SA_{i,j+1} \| SA_{i,j-1})$. Neighbors within the same instance have similar attention distributions (low divergence); neighbors across instance boundaries exhibit abrupt divergence spikes.
    - Design Motivation: Non-parametric—no training or clustering is required; boundary signals are extracted directly from the geometric properties of attention distributions.

3. **Single-step Self-distillation Edge Decoder**:

    - Function: Compresses the multi-step IEP+ABDiv computation into a single inference step.
    - Mechanism: The ABDiv pseudo-edge map $E$ is used as supervision (pixels $> \mu+\sigma$ as positive, $< \mu-\sigma$ as negative, intermediate pixels masked out). The diffusion backbone is fine-tuned with LoRA at $t=0$, and a lightweight decoder $\mathcal{G}_\phi$ is trained with the loss $\mathcal{L} = \|I-\hat{I}\|^2 + \text{DiceLoss}(E, \hat{E})$. The reconstruction loss stabilizes training and completes broken edges.
    - Design Motivation: IEP+ABDiv requires approximately 3.7 seconds per image; after distillation, inference takes only 45 ms/image (81× speedup), and the resulting edges are more continuous and complete.

### Loss & Training

Distillation training: DiceLoss (edge prediction) + L2 reconstruction loss, with uncertain pixels excluded. Training is performed exclusively on the COCO training set using LoRA fine-tuning of the diffusion backbone. Inference requires a single forward pass; the default backbone is SD3.5-L.

## Key Experimental Results

### Main Results

Unsupervised instance segmentation (COCO 2014, AP$^{mk}$):

| Method | VOC AP | COCO 2014 AP | COCO 2017 AP |
|------|:------:|:------------:|:------------:|
| MaskCut* | 5.8 | 3.0 | 2.3 |
| + TRACE | **9.7** | **7.9** | **7.5** |
| ProMerge* | 5.0 | 3.1 | 2.5 |
| + TRACE | **9.4** | **8.2** | **7.8** |
| CutLER* | 11.2 | 8.9 | 8.7 |
| + CutS3D (depth) | - | 10.9 | 10.7 |
| + TRACE | **14.8** | **13.1** | **12.8** |

Weakly supervised panoptic segmentation (VOC 2012 PQ):

| Method | Supervision | VOC PQ | COCO PQ |
|------|:-------:|:------:|:-------:|
| Mask2Former* | Full mask | 73.6 | 51.9 |
| EPLD | Point | 56.6 | 34.2 |
| EPLD (Swin-L) | Point | 68.5 | 41.0 |
| DHR+TRACE | **Tag** | **56.9** | **32.8** |
| DHR+TRACE (Swin-L) | **Tag** | **69.8** | **43.1** |

### Ablation Study

Component ablation (COCO 2014, ProMerge baseline, AP$^{mk}$):

| Configuration | AP$^{mk}$ | Notes |
|------|:----:|------|
| Baseline | 3.1 | No TRACE |
| + ABDiv (semantic step) | 3.2 | ABDiv at semantic timestep is nearly ineffective |
| + IEP + ABDiv | 4.8 | IEP localizes the correct timestep → effective |
| + IEP + ABDiv + Distillation | **8.2** | Distillation completes broken edges ↑↑ |

Diffusion vs. non-diffusion backbone comparison:

| Backbone | Type | Params | AP$^{mk}$ |
|----------|:----:|:----:|:----:|
| DINOv2-G | Non-diffusion | 1.1B | 2.6 |
| Qwen2.5-VL | Non-diffusion | 72B | 4.1 |
| PixArt-α | Diffusion | 0.6B | 7.1 |
| SD3.5-L | Diffusion | 8.1B | 8.2 |
| FLUX.1 | Diffusion | 12B | 8.3 |

### Key Findings

1. **Unique advantage of diffusion models**: PixArt-α at 0.6B (AP$^{mk}$ 7.1) substantially outperforms Qwen2.5-VL at 72B (4.1), confirming that instance edges are a prior unique to generative models.
2. **Distillation improves quality, not just speed**: Inference time drops from 3.7 s to 45 ms, and edges become more continuous and complete.
3. **Tag supervision surpasses point supervision**: DHR+TRACE (tag only) achieves PQ 69.8 on VOC, exceeding EPLD (point supervision) at 68.5.
4. **Conventional edge detectors are entirely inadequate**: Canny achieves only 1.2 AP$^{mk}$ vs. TRACE's 9.4—because conventional detectors respond to intensity gradients, not instance boundaries.

## Highlights & Insights

- **Discovery of the Instance Emergence Point during denoising**: The staged transition of self-attention from noise → instance → semantics is a novel and previously unreported observation.
- **Non-parametric edge extraction**: ABDiv requires no training or labels, relying purely on the geometric properties of attention distributions.
- **Model-agnostic**: The optimal timestep identified by IEP is highly consistent across five different diffusion backbones.
- **Plug-and-play composability**: TRACE integrates seamlessly with MaskCut, CutLER, ProMerge, DHR, and other pipelines.

## Limitations & Future Work

- The method depends on the self-attention of diffusion models and is inapplicable to non-diffusion architectures, as confirmed experimentally.
- IEP search still requires multi-step forward passes (~3 s/image), though this is no longer needed after distillation.
- Distillation is performed only on SD3.5-L; different backbones may require separate distillation.
- Edge quality on small objects and heavily occluded scenes remains to be evaluated.
- The current method applies only to static images; temporal consistency in video settings has not been explored.

## Related Work & Insights

- **vs. MaskCut/CutLER**: These methods rely on DINO feature clustering and cannot separate adjacent same-class objects; TRACE's instance edges directly address this core failure mode.
- **vs. CutS3D**: Depth estimation is used to assist instance separation, failing at similar depths; TRACE does not depend on depth and outperforms by 2.2 AP on COCO.
- **vs. DiffCut/DiffSeg**: These methods apply diffusion attention to semantic segmentation using fixed timesteps; TRACE demonstrates that IEP-identified timesteps are substantially more effective than fixed ones.
- **Insight**: The structural priors encoded in generative models far exceed prior expectations—self-attention not only "knows where to look" but also "knows where the boundaries are."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of IEP+ABDiv is highly original; framing diffusion models as instance edge annotators represents a genuinely new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on both unsupervised and weakly supervised tracks, with 10 backbone comparisons, comprehensive ablations, and multi-benchmark validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is fluent with excellent figures; every design choice is supported by quantitative evidence.
- Value: ⭐⭐⭐⭐⭐ A paradigm-shifting contribution to unsupervised and weakly supervised segmentation; the result of tag supervision surpassing point supervision carries broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](../../CVPR2026/segmentation/videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)
- [\[ICLR 2026\] Universal Multi-Domain Translation via Diffusion Routers](universal_multi-domain_translation_via_diffusion_routers.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](../../CVPR2026/segmentation/promptdriven_lightweight_foundation_model_for_inst.md)
- [\[AAAI 2026\] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation](../../AAAI2026/segmentation/text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](../../CVPR2026/segmentation/data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)

</div>

<!-- RELATED:END -->
