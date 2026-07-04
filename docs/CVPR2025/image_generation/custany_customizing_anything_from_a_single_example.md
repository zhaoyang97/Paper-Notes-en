---
title: >-
  [Paper Note] CustAny: Customizing Anything from A Single Example
description: >-
  [CVPR 2025][Image Generation][Zero-shot Customization] This paper constructs the first large-scale generic object customization dataset MC-IDC (315K images, 10K+ categories) and proposes the CustAny framework. By utilizing multi-model ID extraction, global-local dual-level ID injection, and an ID-aware decoupling module, CustAny achieves zero-shot customized generation of arbitrary objects from a single reference image.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Zero-shot Customization"
  - "Identity Preservation"
  - "Diffusion Models"
  - "General Objects"
  - "Dual-Level Injection"
date: 2026-05-08
content_hash: 9bed771c1edf8ae5
---

# CustAny: Customizing Anything from A Single Example

**Conference**: CVPR 2025  
**arXiv**: [2406.11643](https://arxiv.org/abs/2406.11643)  
**Code**: [https://github.com/LingjieKong-fdu/CustAny](https://github.com/LingjieKong-fdu/CustAny)  
**Area**: Image Generation / Customized Generation  
**Keywords**: Zero-shot Customization, Identity Preservation, Diffusion Models, General Objects, Dual-Level Injection

## TL;DR
This paper constructs the first large-scale generic object customization dataset MC-IDC (315K images, 10K+ categories) and proposes the CustAny framework. By utilizing multi-model ID extraction, global-local dual-level ID injection, and an ID-aware decoupling module, CustAny achieves zero-shot customized generation of arbitrary objects from a single reference image.

## Background & Motivation

**Background**: Diffusion model-driven customized image generation falls into two categories: object-specific methods (e.g., DreamBooth, which require multiple reference images and fine-tuning) and object-agnostic methods (e.g., PhotoMaker, InstantID, which are zero-shot but limited to specific domains like human faces).

**Limitations of Prior Work**: Zero-shot customization for general objects faces two major obstacles: (1) a lack of large-scale generic object ID-consistency datasets for pre-training; (2) domain-specific methods (e.g., extracting IDs using CLIP for faces) fail to generalize to general objects because a single visual encoder struggles to simultaneously capture both fine details and color information.

**Key Challenge**: The contradiction between generality and fidelity. General methods must handle highly diverse object categories (such as toys, animals, and clothes), yet preserving the unique ID identity of each object is extremely challenging. Moreover, ID information is often entangled with non-ID elements (e.g., pose, orientation), which degrades text editability.

**Goal**: Achieve "arbitrary object + single reference image + text prompt $\rightarrow$ ID-preserving diverse generation".

**Key Insight**: Overcome the data bottleneck by constructing a dedicated multi-category dataset, resolve incomplete feature representations using complementary multi-model ID extraction, and address ID/non-ID entanglement through ID-decoupled training.

**Core Idea**: Combine DINOv2 (proficient in details but color-insensitive) and MAE (retaining color information through reconstruction) for complementary ID extraction; inject semantic and fine-grained IDs into the diffusion model via dual-level injection; and introduce a decoupling branch during training to force the model to distinguish between ID and non-ID information.

## Method

### Overall Architecture
Given a reference image, a segmentation mask, and a text prompt, CustAny generates images via three steps: (1) a generic ID extractor obtains class tokens and patch tokens from DINOv2 and MAE; (2) global injection integrates the class token into the category word position of the text embedding, while local injection embeds patch tokens by adding cross-attention layers in the UNet upsampling blocks; (3) an ID-aware decoupling module separates ID and non-ID features during training. Inference uses 50-step denoising with a CFG scale of 7.

### Key Designs

1. **Multi-Model Complementary ID Extraction**:

    - **Function**: Extract comprehensive ID feature representations from reference images.
    - **Mechanism**: DINOv2 excels at capturing detailed structures of objects due to contrastive learning, but its ColorJitter data augmentation renders it insensitive to colors. Conversely, MAE is trained based on reconstruction, which naturally preserves color and structural details. Therefore, the reference image (with the background masked out) is fed into both encoders to extract class tokens and patch tokens separately, which are then dimensionally aligned via MLPs.
    - **Design Motivation**: Experiments demonstrate that using any single encoder among CLIP, DINO, or MAE leads to clear degradations in ID fidelity, whereas their complementary combination achieves optimal results across three metrics: FID, CLIP-i, and DINO-i.

2. **Global-Local Two-Level ID Injection**:

    - **Function**: Maximize the embedding of ID information into the diffusion model while preserving text editability.
    - **Mechanism**: Global injection concatenates the class tokens of DINOv2 and MAE with the category word embedding of the text, fuses them through an MLP, replaces the category word position within the text embedding, and interacts with the UNet via standard cross-attention. Local injection fuses patch tokens and uses them as keys/values for an additional cross-attention layer, injecting fine-grained spatial details into each upsampling block of the UNet.
    - **Design Motivation**: Global injection provides semantic-level ID guidance ("what this object is"), while local injection offers pixel-level details ("what the object looks like"). Both are indispensable—global injection alone yields FID 49.78/DINO-i 60.67, local injection alone yields DINO-i 62.89, while their dual-level combination achieves 47.50/65.12.

3. **ID-Aware Decoupling Module**:

    - **Function**: Disentangle ID and non-ID information during training to enhance generation diversity.
    - **Mechanism**: A "decoupling branch" is introduced, where CLIP extracts target image embeddings and filters out ID information via a learnable mask $m_{id}$, obtaining $f_{msk}$ (which contains only non-ID information like pose and orientation). The decoupling branch performs denoising using $f_{msk} \oplus f_{fuse}^C$, whereas the normal branch uses only $f_{fuse}^C$. Both paths predict the target image. A contrastive loss $\mathcal{L}_{contrast} = Sim(f_{fuse}^C, f_{msk})$ is applied to ensure orthogonality between ID and non-ID features.
    - **Design Motivation**: Without decoupling, pose information (e.g., "standing person") from the reference image leaks into the ID representation, forcing the model to generate standing poses regardless of the text prompt.

### Loss & Training
The total loss is $\mathcal{L} = 2.0 \cdot \mathcal{L}_{normal} + 1.0 \cdot \mathcal{L}_{decouple} + 0.5 \cdot \mathcal{L}_{contrast}$, where the first two are standard MSE denoising losses, and the last is a cosine similarity based contrastive loss. Training is conducted on Stable Diffusion 1.5 with a learning rate of 1e-5 and a batch size of 32, taking about 30 hours for 6 epochs on 32 V100 GPUs.

## Key Experimental Results

### Main Results

| Area | Method | FID↓ | CLIP-i↑ | DINO-i↑ | FaceSim↑ |
|---|---|---|---|---|---|
| General Objects | IP-Adapter | 70.32 | 77.18 | 44.94 | - |
| General Objects | **CustAny** | **47.09** | **82.16** | **65.13** | - |
| Human Customization | PhotoMaker | 106.35 | 71.80 | 44.62 | 64.10 |
| Human Customization | InstantID | 113.18 | 75.87 | 49.26 | 63.26 |
| Human Customization | **CustAny** | **86.40** | **79.60** | **57.44** | **78.54** |
| Virtual Try-on | MagicClothing | 126.09 | 76.53 | 29.10 | - |
| Virtual Try-on | **CustAny** | **50.65** | **83.82** | **66.24** | - |

### Ablation Study

| Configuration | FID↓ | CLIP-i↑ | DINO-i↑ |
|---|---|---|---|
| CLIP Extraction Only | 49.11 | 79.58 | 59.45 |
| DINO Extraction Only | 48.89 | 80.82 | 63.71 |
| MAE Extraction Only | 49.00 | 79.09 | 59.79 |
| DINO+MAE (Ours) | **47.50** | **81.86** | **65.12** |
| Global Injection Only | 49.78 | 78.76 | 60.67 |
| Local Injection Only | 48.66 | 81.24 | 62.89 |
| Dual-level Injection | **47.50** | **81.86** | **65.12** |

### Key Findings
- CustAny significantly outperforms IP-Adapter in general object customization (with a 20.19% gain in DINO-i).
- Even on specialized tasks like face customization, CustAny outperforms domain-specific methods (FaceSim 78.54 vs. InstantID 63.26).
- The discovery of DINOv2's color insensitivity is highly interesting—the "side-effect" caused by ColorJitter augmentation is perfectly resolved through complementation with MAE.
- The ID-decoupling module significantly improves text-editing generation diversity, while having minimal impact on ID fidelity metrics.

## Highlights & Insights
- **MC-IDC Dataset**: The first large-scale general object customization dataset (315K samples, 10K+ categories). The data construction pipeline (data collection $\rightarrow$ instance segmentation $\rightarrow$ image pair generation $\rightarrow$ text annotation) is reusable. The dataset's contribution may hold more lasting value than the methodology itself.
- **Insight on Multi-Encoder Complementation**: DINOv2 lacks color information due to the ColorJitter training strategy, whereas MAE retains colors due to its reconstruction target. Analyzing feature characteristics inversely from training objectives yields precise, logical insights.
- **Generality Outperforming Specialization**: Surpassing specialized methods in domain-specific tasks (e.g., face customization and virtual try-on) validates the feasibility of using general datasets alongside generic architectures.

## Limitations & Future Work
- Based on SD 1.5, the generation quality is constrained by the backbone's capacity; migrating to SDXL/SD3.0 could yield further improvements.
- The MC-IDC dataset relies on existing segmentation and tracking models, meaning dataset quality is limited by the accuracy of these upstream tools.
- The 50-step denoising during inference is relatively slow, and accelerated inference schemes have not yet been explored.
- Extending this framework to customized video generation remains a promising path to explore.

## Related Work & Insights
- **vs. IP-Adapter**: Relying on CLIP as the sole encoder and lacking ID decoupling, IP-Adapter is significantly outperformed by CustAny across all metrics.
- **vs. PhotoMaker/InstantID**: While these methods focus strictly on human faces, CustAny's general approach actually surpasses them on their own domain.
- The strategy of complementary DINO+MAE feature extraction can be transferred to other tasks requiring comprehensive visual representations.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the dataset construction and the insight of multi-encoder complementation demonstrate strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across three domains alongside thorough ablations covering each module.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with abundant illustrations.
- Value: ⭐⭐⭐⭐ The dataset and generic customization framework offer high-value contributions to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[NeurIPS 2025\] Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction](../../NeurIPS2025/image_generation/exploring_semantic-constrained_adversarial_example_with_instruction_uncertainty_.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[NeurIPS 2025\] Orient Anything V2: Unifying Orientation and Rotation Understanding](../../NeurIPS2025/image_generation/orient_anything_v2_unifying_orientation_and_rotation_understanding.md)

</div>

<!-- RELATED:END -->
