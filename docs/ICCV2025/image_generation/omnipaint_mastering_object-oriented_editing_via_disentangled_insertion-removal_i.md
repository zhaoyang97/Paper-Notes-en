---
title: >-
  [Paper Note] OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting
description: >-
  [ICCV2025][Image Generation][object removal] This paper proposes OmniPaint, a unified framework that reformulates object removal and insertion as mutually inverse and complementary tasks. Built upon the FLUX diffusion pr…
tags:
  - "ICCV2025"
  - "Image Generation"
  - "object removal"
  - "object insertion"
  - "inpainting"
  - "diffusion model"
  - "CycleFlow"
  - "FLUX"
  - "flow matching"
  - "CFD metric"
date: 2026-05-08
content_hash: c4f8036e7bf1b864
---

# OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting

**Conference**: ICCV2025  
**arXiv**: [2503.08677](https://arxiv.org/abs/2503.08677)  
**Code**: [GitHub](https://github.com/yeates/OmniPaint)  
**Area**: Image Generation  
**Keywords**: object removal, object insertion, inpainting, diffusion model, CycleFlow, FLUX, flow matching, CFD metric

## TL;DR

This paper proposes OmniPaint, a unified framework that reformulates object removal and insertion as mutually inverse and complementary tasks. Built upon the FLUX diffusion prior, it introduces the CycleFlow unpaired training mechanism and the CFD reference-free evaluation metric. With only 3K real paired samples, OmniPaint achieves high-fidelity object editing, excelling particularly at complex physical effects such as shadows and reflections.

## Background & Motivation

### Problem Definition

Object-oriented image editing encompasses two core sub-tasks: **object removal** and **object insertion**. Conventional approaches treat these as entirely independent tasks and model them with separate technical pipelines, which introduces several issues:

**Heavy data dependency**: Existing methods rely heavily on large-scale paired real or synthetic data, especially for object insertion, which demands precise geometric alignment and realistic integration of physical effects (shadows, reflections, occlusions).

**Difficulty handling physical effects**: Removing an object requires not only erasing its foreground semantics but also eliminating associated physical artifacts such as cast shadows and reflections. Existing methods frequently leave residual artifacts.

**Hallucination problem**: Inpainting methods based on large diffusion models tend to "hallucinate" non-existent objects within the masked region, and effective metrics for detecting such hallucinations are lacking.

**High deployment cost**: Maintaining separate model parameters for removal and insertion increases system complexity and computational overhead.

### Limitations of Prior Work

**Object removal:**
- **Text-driven methods** (InstructPix2Pix, etc.): Limited by the semantic expressiveness of text embeddings, making it difficult to precisely specify target objects in complex scenes.
- **Mask-guided methods** (PowerPaint, CLIPAway, etc.): Offer more precise control but struggle to simultaneously handle objects and their physical effects.
- **Synthetic data methods** (MagicEraser, SmartEraser, RORem): Synthetic data cannot adequately replicate real-world light–object interactions.

**Object insertion:**
- **Traditional compositing methods**: Simple image blending and harmonization cannot handle complex physical effects such as correct shadow casting and illumination consistency.
- **Feature extraction methods** (AnyDoor, IMPRINT, etc.): Rely on additional feature extractors such as CLIP or DINOv2 to maintain identity consistency, increasing architectural complexity.
- **Large-scale data methods** (ObjectMate, etc.): Require millions of paired samples, making data acquisition prohibitively expensive.

### Core Motivation

The key insight of this work is that **object removal and insertion are inherently inverse processes**—removing an object and then re-inserting it should recover the original image. Based on this observation, the authors propose jointly modeling the two tasks by exploiting their complementary relationship. The CycleFlow mechanism enables the trained removal model to assist the insertion model in leveraging large-scale unpaired data, fundamentally alleviating the bottleneck of paired data scarcity.

## Method

### Overall Architecture

OmniPaint is built upon FLUX-1.dev (a flow matching model with MM-DiT architecture) and fine-tuned via LoRA with two sets of parameters $\theta$ (insertion) and $\phi$ (removal). Learnable task embeddings are introduced to enable prompt-free adaptive control.

### 1. Masked Image Conditioning

Given an input image $\mathbf{I}$ and a binary mask $\mathbf{M}$, the model operates on the masked image $\mathbf{X} = \mathbf{I} \odot (1 - \mathbf{M})$. The built-in VAE encoder and $2\times2$ patchify layer of FLUX map $\mathbf{X}$ into a conditional token sequence $\mathbf{z}_c^{\mathcal{X}}$.

### 2. Reference Object Conditioning

For the insertion task, an additional reference object image $\mathbf{O}$ is introduced:
- CarveKit is used to remove the background of the reference object, eliminating background interference.
- The processed object image is resized to match the masked image dimensions.
- The same VAE encoding and patchify steps yield $\mathbf{z}_c^{\mathcal{O}}$.
- The final condition tokens are obtained by concatenation: $\mathbf{z}_c = [\mathbf{z}_c^{\mathcal{X}}; \mathbf{z}_c^{\mathcal{O}}]$.

### 3. Prompt-Free Adaptive Control

Given that these tasks are highly image-conditioned, text prompts may introduce ambiguity. The authors replace text embeddings with learnable task embedding vectors:

$$\tau_{\text{removal}}, \tau_{\text{insertion}} \sim \mathcal{N}(0, I)$$

Both vectors are initialized from the empty-string embedding and optimized independently for removal and insertion. At inference time, switching between tasks is achieved simply by selecting the corresponding embedding.

### 4. Three-Stage Progressive Training Pipeline

**Stage 1: Inpainting Pretraining**
- Trains basic inpainting capability on the LAION dataset using random masks.
- Initializes both LoRA parameter sets $\theta$ and $\phi$.
- Objective: teach the model fundamental region-filling ability.

**Stage 2: Paired Warmup**
- Uses 3,000 real paired samples to train $\phi$ (removal) and $\theta$ (insertion) separately.
- Removal direction: $\mathbf{z}_1$ is sampled from images with objects physically removed.
- Insertion direction: $\mathbf{z}_1$ is sampled from original images retaining the foreground object.
- This stage effectively handles complex removal scenarios including reflections and shadows, but 3,000 samples are insufficient to ensure identity consistency during insertion.

**Stage 3: CycleFlow Unpaired Post-Training**
- Leverages large-scale segmentation datasets such as COCO-Stuff and HQSeg as unpaired data.
- Core challenge: these datasets lack annotations for object effects (shadows/reflections), so naïve training produces insertion results resembling copy-paste.
- Solution: the trained removal parameters $\phi$ serve as a preprocessing step; even at NFE=1, physical effects are effectively removed.

**CycleFlow Mechanism:**

Two mappings are defined:
- $F$ (removal direction): $\mathbf{z}_1' \leftarrow \mathbf{z}_t - u_t^{\phi}(\mathbf{z}_t, \mathbf{z}_c^{\mathcal{X}}, \tau_{\text{removal}}) \cdot t$
- $G$ (insertion direction): $\bar{\mathbf{z}}_1 \leftarrow \mathbf{z}_t' - u_t^{\theta}(\mathbf{z}_t', \mathbf{z}_c, \tau_{\text{insertion}}) \cdot t$

Cycle consistency constraint: removing an object and then re-inserting it should recover the original latent:

$$\mathbf{z}_1 \rightarrow \mathbf{z}_t \rightarrow F(\mathbf{z}_t) \rightarrow \mathbf{z}_t' \rightarrow G(\mathbf{z}_t') \approx \mathbf{z}_1$$

Cycle loss: $\mathcal{L}_{\text{cycle}}(\theta) = \mathbb{E}_{t, \mathbf{z}_t}[\|G_\theta(\lfloor F(\mathbf{z}_t) \rfloor) - \mathbf{z}_1\|^2]$

where $\lfloor \cdot \rfloor$ is the gradient stop operator, keeping the removal parameters $\phi$ frozen. The total loss is $\mathcal{L}_{\text{warmup}} + \gamma \mathcal{L}_{\text{cycle}}$, with $\gamma = 1.5$ as the optimal balance.

### 5. CFD Metric (Context-Aware Feature Deviation)

To address the inability of existing metrics (e.g., ReMOVE) to detect hallucinated objects, the authors propose a reference-free CFD metric comprising two components:

**Hallucination Penalty:**
- SAM-ViT-H is used to segment the edited result, partitioning masks into nested masks (fully within the removal region) and overlapping masks (crossing the boundary).
- DINOv2 feature similarity is used to determine whether nested regions constitute hallucinated content.
- Area-weighted aggregation: $d_{\text{hallucination}} = \sum \omega_i \cdot (1 - \mathbf{f}(\Omega_{\mathcal{M}_i^n})^\top \mathbf{f}(\Omega_{\bar{\mathcal{M}}_i}))$

**Context Coherence:**
- Computes the deviation between the inpainted region and its surrounding background in DINOv2 feature space.
- $d_{\text{context}} = 1 - \mathbf{f}(\Omega_{\mathbf{M}})^\top \mathbf{f}(\Omega_{\mathbf{B} \setminus \mathbf{M}})$

**Final metric:** $\text{CFD} = d_{\text{context}} + d_{\text{hallucination}}$, lower is better.

## Key Experimental Results

### Object Removal (Self-constructed 300-sample test set)

| Metric | LaMa | PowerPaint | FreeCompose | FLUX-Inpaint | **OmniPaint** |
|--------|-------|------------|-------------|--------------|---------------|
| FID ↓ | 105.10 | 103.61 | 88.77 | 132.60 | **51.66** |
| CMMD ↓ | 0.3729 | 0.2182 | 0.1790 | 0.3257 | **0.0473** |
| CFD ↓ | 0.3531 | 0.4031 | 0.3743 | 0.4609 | **0.2619** |
| PSNR ↑ | 20.86 | 19.46 | 21.27 | 20.86 | **23.08** |
| LPIPS ↓ | 0.1353 | 0.1428 | 0.1182 | 0.1451 | **0.0738** |

### Object Removal (RORD 1000-sample test set)

OmniPaint achieves comprehensive superiority: FID 19.17 (vs. second-best PowerPaint at 42.65), PSNR 23.23, CFD 0.3682.

### Object Insertion (565-sample test set)

| Metric | AnyDoor | IMPRINT | FreeCompose | **OmniPaint** |
|--------|---------|---------|-------------|---------------|
| CLIP-I ↑ | 89.26 | 90.63 | 88.17 | **92.27** |
| DINOv2 ↑ | 76.96 | 76.89 | 76.01 | **84.37** |
| CUTE ↑ | 85.26 | 86.15 | 82.86 | **90.29** |
| DreamSim ↓ | 0.2208 | 0.1854 | 0.2134 | **0.1557** |
| MUSIQ ↑ | 69.28 | 68.72 | 66.67 | **70.59** |

### Hyperparameter Analysis

- **NFE (inference steps)**: NFE=1 already effectively removes objects and their physical effects (albeit with some blurring); NFE=18 yields clean removal and high-fidelity insertion; NFE=28 offers marginal further gains and is used as the default.
- **$\gamma$ (cycle loss weight)**: $\gamma=0$ fails to synthesize physical effects (resembling copy-paste); $\gamma=1.5$ achieves the optimal balance; $\gamma=3.0$ leads to over-relaxation and unnatural artifacts.

## Highlights & Insights

1. **Inverse modeling of removal and insertion**: Unifying two traditionally independent tasks as complementary processes, and using the trained removal model to help the insertion model overcome the paired data bottleneck—an elegant and practical design.
2. **CycleFlow unpaired training**: Only 3K real paired samples are required for warmup; CycleFlow then enables post-training on large-scale unpaired segmentation data, dramatically reducing data acquisition costs.
3. **Prompt-free design**: Replacing text prompts with learnable task embeddings eliminates ambiguity from textual descriptions while supporting flexible task switching at inference time.
4. **CFD metric**: The first reference-free evaluation metric specifically targeting hallucination detection in object removal, addressing the gap left by existing metrics such as ReMOVE that cannot distinguish high scores from evident hallucinations.
5. **Physical effect handling**: Complex physical artifacts including reflections, shadows, and occlusions are explicitly modeled and addressed, rather than focusing solely on the foreground object itself.
6. **Minimal paired data**: In contrast to methods such as ObjectMate that require millions of paired samples, OmniPaint achieves state-of-the-art performance with only 3K paired samples.

## Limitations & Future Work

1. **Single-object insertion constraint**: The current framework supports only single reference object insertion; simultaneous multi-object editing is not addressed.
2. **CFD dependency on SAM**: Hallucination detection relies on the segmentation quality of SAM, which may struggle with very small objects or ambiguous boundaries.
3. **Fixed inference steps**: NFE=28 is applied uniformly across all scenarios; simple cases may waste computation while complex cases may be under-served.
4. **CycleFlow applied only to insertion**: The authors note that warmup already satisfies removal requirements and CycleFlow is not applied to removal, though more challenging removal scenarios might benefit from it.
5. **Dependency on CarveKit for background removal**: The quality of reference object background extraction directly affects insertion results, making the robustness of this third-party tool a bottleneck.
6. **Mask annotation requirement**: Users must still provide precise masks to specify editing regions; end-to-end automatic object localization is not considered.

## Related Work & Insights

- **CycleGAN → CycleFlow**: The cycle consistency idea from CycleGAN is transferred from the GAN framework to the flow matching framework, enabling unpaired training in latent space—offering a reusable paradigm for other tasks requiring paired data.
- **Flow matching's editing adaptability**: The flow matching framework of FLUX is naturally suited to modeling inverse processes; the observation that a single-step velocity field estimate (NFE=1) suffices for rough task completion merits further attention.
- **Task embeddings vs. text prompts**: The finding that learnable embeddings outperform text prompts in highly image-conditioned editing tasks offers a reference point for other conditional generation settings.
- **Evaluation metric innovation**: The CFD approach of combining a segmentation model with feature similarity for hallucination detection is generalizable and may be extended to hallucination detection in other generative tasks.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](../../CVPR2026/image_generation/effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[ICCV 2025\] Multi-turn Consistent Image Editing](multi-turn_consistent_image_editing.md)
- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](../../CVPR2026/image_generation/precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)

</div>

<!-- RELATED:END -->
