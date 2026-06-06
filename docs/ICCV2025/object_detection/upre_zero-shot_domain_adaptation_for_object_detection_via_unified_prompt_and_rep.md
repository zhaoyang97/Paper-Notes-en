---
title: >-
  [Paper Note] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement
description: >-
  [ICCV 2025][Object Detection][Zero-shot domain adaptation] This paper proposes the UPRE framework, which jointly optimizes Multi-view Domain Prompts (MDP) and Unified Representation Enhancement (URE) to simultaneously al…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Zero-shot domain adaptation"
  - "prompt learning"
  - "vision-language models"
  - "domain shift"
date: 2026-05-08
content_hash: 9c743e32e78d255b
---

# UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement

**Conference**: ICCV 2025
**arXiv**: [2507.00721](https://arxiv.org/abs/2507.00721)  
**Code**: [GitHub](https://github.com/AMAP-ML/UPRE)  
**Area**: Object Detection
**Keywords**: Zero-shot domain adaptation, object detection, prompt learning, vision-language models, domain shift

## TL;DR

This paper proposes the UPRE framework, which jointly optimizes Multi-view Domain Prompts (MDP) and Unified Representation Enhancement (URE) to simultaneously alleviate detection bias and domain bias in zero-shot domain adaptive object detection, achieving state-of-the-art performance across nine datasets spanning three scenario types: adverse weather, cross-city, and virtual-to-real.

## Background & Motivation

Zero-shot domain adaptation (ZSDA) aims to transfer a model from a source domain to a target domain without accessing any target domain images. This is particularly challenging for object detection, as it requires simultaneously handling distribution shift and precise localization. Existing methods face two core biases:

**Domain Bias**: Distribution shift between source and target domains introduces task-irrelevant noise that degrades model performance. For instance, a detector trained under daytime clear conditions generalizes poorly to nighttime rainy scenes due to large visual style differences in illumination and reflections.

**Detection Bias**: VLMs such as CLIP emphasize global image representations while neglecting instance-level details required for object detection. Existing methods rely on manually constructed prompts (e.g., "a photo of a [class]"), which fail to adequately capture the contextual attributes of foreground and background objects.

The root cause lies in a fundamental tension:
- Methods addressing domain bias (e.g., PODA's prompt-driven feature augmentation) use hand-crafted prompts to generate pseudo target-domain features but ignore or even exacerbate detection bias.
- Methods addressing detection bias (e.g., DetPro's prompt representation learning) train solely on the source domain, yielding prompts suited to source-domain detection that nevertheless amplify domain bias.

This paper is the first to propose a unified framework that simultaneously addresses both biases.

## Method

### Overall Architecture

UPRE is built upon Faster R-CNN with a CLIP ResNet-101 backbone and comprises three core components and two multi-level strategies:

1. **Multi-view Domain Prompt (MDP)**: Fuses linguistic domain priors with learnable prompts to provide detection-specific cross-domain knowledge.
2. **Unified Representation Enhancement (URE)**: Generates pseudo target-domain representations with diverse domain styles via feature transformation.
3. **Multi-level Enhancement Strategies**: Relative Domain Distance (RDD) aligns multimodal representations at the image level; Positive-Negative Separation (PNS) captures detection knowledge at the instance level.

### Key Designs

1. **Multi-view Domain Prompt (MDP)**:

    - *Function*: Learns three prompt representations—image prompt, positive prompt, and negative prompt—each providing domain adaptation knowledge at a different granularity.
    - *Mechanism*:
        - **Image prompt** $\mathcal{R}_i^d = [u_1, u_2, \ldots, u_L, k_d]$: Learnable vectors concatenated with a domain keyword embedding (e.g., "a photo taken on a [domain]"), supplying image-level domain style priors.
        - **Positive prompt** $\mathcal{R}_p^t(c) = [v_1, \ldots, v_L, (k_t, k_c)]$: Learnable vectors combined with "a [domain] photo of a [class]", capturing style variations of foreground objects in the target domain.
        - **Negative prompt** $\mathcal{R}_n^t(\mathcal{C}_{bg}) = [w_1, \ldots, w_L, (k_t, k_{\mathcal{C}_{bg}})]$: Analogous structure applied to the background category "unknown class", capturing background context.
    - *Design Motivation*: Retaining hand-crafted prompt structure preserves static linguistic priors, allowing the learnable components to focus exclusively on detection-specific cross-domain knowledge. The three prompt types cover complementary perspectives: image level, foreground, and background.

2. **Unified Representation Enhancement (URE)**:

    - *Function*: Transforms source-domain features into pseudo target-domain features to increase domain style diversity.
    - *Mechanism*: The source feature map $F_s$ is partitioned into $M \times N$ patches; each patch is transformed by learnable mean enhancement $\mathcal{E}_\mu$ and deviation enhancement $\mathcal{E}_\sigma$: $F_{s \to t} = \{\mathcal{E}_\sigma^j \cdot F_s^j + \mathcal{E}_\mu^j\}_{j=1}^{M \times N}$
    - *Design Motivation*: In real-world scenes, different regions of the same image often exhibit distinct styles (e.g., in a rainy night image, nearby regions appear "rainy" due to illumination, while distant regions appear "nighttime" due to low light). Global AdaIN-based style transfer employed by PODA is insufficient; patch-wise enhancement captures fine-grained local style variations.

3. **Relative Domain Distance (RDD)**:

    - *Function*: Constrains multimodal representation alignment at the image level, stabilizing the tension between augmentation and regularization objectives.
    - *Mechanism*: Three complementary loss functions are defined:
        - **Alignment loss** $\mathcal{L}_a = \mathbb{E}[1 - f(\mathbf{e}_i^{s \to t}, \mathbf{t}_i^t)]$: Pulls the augmented image representation closer to the target-domain text representation.
        - **Constraint loss** $\mathcal{L}_s = \mathbb{E}[\|\mathbf{e}_i^s - \mathbf{e}_i^{s \to t}\|_1]$: Prevents excessive augmentation from destroying semantic information.
        - **Relative distance loss** $\mathcal{L}_r = \mathbb{E}[\|(\mathbf{e}_i^s - \mathbf{e}_i^{s \to t}) - (\mathbf{t}_i^s - \mathbf{t}_i^t)\|_1]$: Ensures that the visual domain shift is consistent with the linguistic domain shift.
    - *Design Motivation*: $\mathcal{L}_a$ and $\mathcal{L}_s$ have conflicting objectives (one encourages distance, the other discourages it); $\mathcal{L}_r$ guides optimization in the overlapping region, stabilizing training.

4. **Positive-Negative Separation (PNS)**:

    - *Function*: Applies distinct classification objectives to positive and negative proposals at the instance level.
    - *Mechanism*: Positive proposals are classified among foreground classes only via cross-entropy loss $\mathcal{L}_c$; negative proposals are handled by $\mathcal{L}_{bg}$ with a uniform target distribution $y_{bg}$ to prevent overconfident background predictions. Positive and negative proposals use representations from the positive and negative prompts, respectively.
    - *Design Motivation*: Unlike DetPro, which uses a single shared prompt with class keywords, PNS allows positive and negative proposals to learn different contextual information—positive proposals capture domain style variations of foreground objects, while negative proposals learn background context.

### Loss & Training

Training proceeds in two stages:
1. **Prompt and enhancement learning stage** (5k iterations): MDP and URE are trained jointly in a mutually reinforcing manner—MDP refinement drives URE improvement, while pseudo features generated by URE in turn help MDP acquire detection knowledge.
2. **Detector fine-tuning stage** (100k iterations): MDP, URE, and the text encoder are frozen; the CLIP backbone and RCNN detector are fine-tuned.

During inference, the URE transformation is disabled to maintain consistent feature processing.

## Key Experimental Results

### Main Results

| Scenario | Dataset | UPRE (mAP) | Prev. SOTA | Gain |
|----------|---------|-----------|------------|------|
| Adverse Weather | Daytime Foggy | **40.0** | 39.6 (UFR) | +0.4 |
| Adverse Weather | Night Clear | **41.5** | 41.0 (DAI-Net) | +0.5 |
| Adverse Weather | Night Rainy | **19.8** | 19.2 (PDD/UFR) | +0.6 |
| Adverse Weather | Dusk Rainy | **34.5** | 33.9 (OA-DG) | +0.6 |
| Cross-City | Cityscapes→BDD100K | **28.7** | 27.2 (OA-DG) | +1.5 |
| Cross-City | Cityscapes→KITTI | **74.3** | 73.6 (PODA) | +0.7 |
| Virtual-to-Real | Sim10K→Cityscapes | **47.9** | 47.0 (OA-DG) | +0.9 |
| Virtual-to-Real | Sim10K→BDD100K | **37.8** | 36.1 (PODA) | +1.7 |
| Virtual-to-Real | Sim10K→KITTI | **61.9** | 60.7 (CLIP-GAP) | +1.2 |

Compared to the Faster R-CNN baseline, UPRE achieves an average improvement of 7.8% mAP under adverse weather conditions.

### Ablation Study

| Configuration | Daytime Foggy | Night Clear | Night Rainy | Dusk Rainy | Notes |
|---------------|--------------|-------------|-------------|------------|-------|
| No learnable prompt + keyword | 37.2 | 37.5 | 17.0 | 31.7 | Baseline |
| No learnable prompt + full description | 38.2 | 39.3 | 17.9 | 32.2 | Full description helps |
| Learnable prompt + keyword | 38.7 | 40.1 | 18.6 | 33.0 | Learnable prompts yield clear gains |
| Learnable (shared) + full description | 38.0 | 39.7 | 17.4 | 32.2 | Shared parameters are harmful |
| Learnable prompt + full description | **40.0** | **41.5** | **19.8** | **34.5** | Full model achieves best performance |

RDD loss component ablation (MAD = mean absolute deviation, measuring training stability):

| $\mathcal{L}_a$ | $\mathcal{L}_s$ | $\mathcal{L}_r$ | Avg. mAP | MAD | Notes |
|:---:|:---:|:---:|:---:|:---:|-------|
| ✓ | - | - | 28.95 | 0.8 | Alignment only is unstable |
| ✓ | ✓ | - | 31.75 | 1.7 | Conflicting objectives cause maximum fluctuation |
| ✓ | ✓ | ✓ | **32.50** | **0.4** | RDD stabilizes training and achieves best performance |

### Key Findings

- **Prompt design is critical**: Retaining the full hand-crafted prompt structure (rather than keywords alone), while allowing learnable components to focus on domain knowledge, yields an average improvement of 3.0% mAP.
- **Shared learnable prompts are detrimental**: A single shared prompt covering image-level, positive, and negative views simultaneously causes information conflicts.
- $\mathcal{L}_r$ is essential for training stability—MAD drops from 1.7 to 0.4.
- t-SNE visualizations show that UPRE successfully separates domain embeddings for different weather conditions (e.g., Night Clear vs. Night Rainy), whereas vanilla CLIP cannot distinguish them.
- In the Daytime Foggy scenario, categories such as bus (+6.8%), motor (+9.9%), and rider (+9.1%) exhibit the largest gains, demonstrating strong fine-grained contextual capture capability.

## Highlights & Insights

- **A unified framework simultaneously addressing two biases**: This work is the first to explicitly identify the adversarial relationship between domain bias and detection bias, and to mitigate both within a unified training pipeline.
- **Patch-wise feature enhancement** outperforms global AdaIN (as in PODA), better reflecting the local style variations characteristic of real-world scenes.
- **The three-way prompt role decomposition** (image-level / positive / negative) is more effective than shared prompts, reflecting the insight that different components of a detection model require distinct domain knowledge.
- The "relative distance" formulation in RDD is more principled than simple attraction or repulsion—by requiring that visual domain shift be consistent with linguistic domain shift, it provides a geometrically meaningful training signal.

## Limitations & Future Work

- Domain descriptors $k_d$ still require manual definition (e.g., "rainy night"); automatically discovering linguistic descriptions for target domains remains an open problem.
- The augmentation granularity in URE (fixed $M \times N$ patch partitioning) is static; adaptive granularity selection may be more effective.
- Experiments are primarily conducted in autonomous driving scenarios; generalizability to other domains (e.g., medical imaging, remote sensing) remains unverified.
- The two-stage training pipeline increases procedural complexity; end-to-end unified training may be more efficient.
- The margin over DAI-Net on Night Clear is modest (41.5 vs. 41.0), suggesting limited room for improvement in scenarios with relatively small domain shift.

## Related Work & Insights

- PODA (ICCV 2023), which performs global style transfer via AdaIN, is the primary point of comparison; URE's patch-wise enhancement is a direct improvement upon it.
- CLIP-GAP (CVPR 2023) is among the first to leverage VLMs for prompt-driven semantic augmentation, but neglects detection bias.
- The positive-negative proposal separation concept from DetPro is extended in this work to cross-domain settings.
- The unified training paradigm (where prompts and augmentation mutually reinforce each other) generalizes to domain adaptation in other vision-language tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified framework design addressing both domain bias and detection bias is original, though individual components are relatively incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across nine datasets and three scenario types, with in-depth ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, though the large number of modules makes the method section somewhat dense.
- **Value**: ⭐⭐⭐⭐ Strong practical guidance for real-world applications requiring cross-domain generalization, such as autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[ICCV 2025\] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)

</div>

<!-- RELATED:END -->
