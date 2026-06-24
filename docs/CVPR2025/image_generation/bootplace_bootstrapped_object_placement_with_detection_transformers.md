---
title: >-
  [Paper Note] BootPlace: Bootstrapped Object Placement with Detection Transformers
description: >-
  [CVPR 2025][Image Generation][Object Placement] BootPlace is proposed to reformulate the object placement problem as "placement-by-detection". By training detection transformers on object-removed backgrounds to identify candidate regions, and then matching target objects to the optimal regions using negative-correlation semantic complementarity, it improves the top-5 IoU on Cityscapes by approximately 4× compared to the state-of-the-art.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Object Placement"
  - "Image Synthesis"
  - "Detection Transformers"
  - "Bootstrapped Training"
  - "copy-paste"
date: 2026-05-08
content_hash: 1c01660dbf1bf4c6
---

# BootPlace: Bootstrapped Object Placement with Detection Transformers

**Conference**: CVPR 2025  
**arXiv**: [2503.21991](https://arxiv.org/abs/2503.21991)  
**Code**: [https://github.com/RyanHangZhou/BOOTPLACE](https://github.com/RyanHangZhou/BOOTPLACE)  
**Area**: Diffusion Models / Object Detection  
**Keywords**: Object Placement, Image Synthesis, Detection Transformers, Bootstrapped Training, copy-paste

## TL;DR
BootPlace is proposed to reformulate the object placement problem as "placement-by-detection". By training detection transformers on object-removed backgrounds to identify candidate regions, and then matching target objects to the optimal regions using negative-correlation semantic complementarity, it improves the top-5 IoU on Cityscapes by approximately 4× compared to the state-of-the-art.

## Background & Motivation

**Background**: Copy-paste image synthesis requires placing objects naturally into a scene. Existing methods include utilizing GANs to generate placement locations (ST-GAN, PlaceNet), graph completion modeling (GracoNet), and Transformer regression (TopNet). However, they all struggle with sparse annotations and imprecise placement.

**Limitations of Prior Work**: Although generative methods (GAN-based) reduce reliance on dense supervision, their capacity to model complex data distributions is insufficient. Transformer-based methods (TopNet) suffer from loose regularization due to sparse contrastive loss, leading to imprecise placements. GracoNet requires manual annotation of positive and negative pairs, which is time-consuming and difficult to scale. A common limitation of these methods is treating placement as a regression problem, which lacks explicit modeling of "where is suitable for placing objects".

**Key Challenge**: Ground-truth labels for object placement are inherently sparse (the plausible locations for the same category of objects in a scene are limited), making direct location regression easily under-constrained, while manual annotation of positive and negative samples is not scalable.

**Goal**: How to achieve precise multi-object placement under sparse label conditions, while avoiding placing objects on top of existing objects in the scene.

**Key Insight**: The essence of object placement is "finding areas in the scene that lack objects but should have them", which is precisely the inverse of the detection problem—detecting "vacated spaces where objects should exist" on the background after objects are removed. Therefore, robust detection frameworks can be directly applied to placement. A bootstrapped training strategy of randomly removing objects provides combinatorially explosive amounts of training data.

**Core Idea**: Detect "empty regions" on the object-removed background using a detection transformer, and then match target objects to the optimal empty regions via negative-correlation semantic complementarity to achieve precise placement.

## Method

### Overall Architecture
BootPlace consists of two modules. **Module 1 (Region Detection)**: Objects in the scene image are removed (instance segmentation + inpainting) to obtain the object-removed background. A DETR-style detection transformer is trained on this background to detect "regions of interest" (keyzones) suitable for object placement. The locations of preserved objects in the scene are encoded via an MLP and concatenated into the image features to avoid placing objects on top of existing ones. **Module 2 (Object-Region Association)**: A CNN encoder extracts embeddings of target objects, computes the negative-correlation association score between the objects and the detected regions, and determines the best match via softmax.

### Key Designs

1. **Placement-by-Detection**

    - **Function**: Reformulates the object placement problem as a region detection problem on the object-removed background.
    - **Mechanism**: First, panoptic segmentation is performed via MaskFormer to identify scene objects, and the LaMa inpainting model is used to remove objects, followed by Gaussian smoothing to eliminate inpainting artifacts and obtain a clean background. Then, a DETR-based detection transformer (CNN backbone + Transformer encoder-decoder + prediction heads) is trained to detect $N$ keyzones $\{p_i\}$ on this background, each containing a location $b_i \in \mathbb{R}^4$ and a category score $s_i \in \mathbb{R}^C$. Crucially, the locations of existing objects in the scene are encoded through an MLP and concatenated with image features to form position-aware features, preventing detection at existing object locations.
    - **Design Motivation**: Detection transformers are highly mature in precise localization (the DETR family). Directly repurposing this capability is more reliable than learning location regression from scratch.

2. **Negative-Correlation Semantic Association Network**

    - **Function**: Semantically matches target objects to the most suitable detected regions, preventing placement on similar objects.
    - **Mechanism**: The association score is defined as $g_i(q_k, F) = -q_k \cdot F_i / \mu$. Note the negative sign—this implies that the more **dissimilar** the object features and regional features are, the higher the association score. The intuition is that the semantics surrounding empty spaces (e.g., roads, sidewalks) should be complementary to, rather than similar to, the objects to be placed (e.g., cars, pedestrians). The association probability distribution $P_A(\alpha=i|F)$ is obtained by softmax normalization, and the log-likelihood of the ground-truth association is maximized during training. During inference, the region with the highest probability is selected.
    - **Design Motivation**: Positive correlation (dot-product similarity) would cause objects to be placed in regions similar to themselves (i.e., where identical categories already exist), whereas negative correlation enforces semantic complementarity, ensuring placement into vacant positions.

3. **Bootstrapped Training**

    - **Function**: Expands each scene into a combinatorially explosive number of training samples.
    - **Mechanism**: For a scene containing $T$ objects, a random subset is selected and removed, where the remaining objects act as scene objects, and the removed ones serve as target placement objects. Each scene can generate $\sum_{i=1}^{T} \binom{T}{i}$ combinations. For instance, a scene with 5 objects can generate 31 different training samples. This drastically increases the diversity of the training data, allowing the model to experience more placement scenarios.
    - **Design Motivation**: Simply training on complete scenes provides limited data (only 2,953 images in Cityscapes); the bootstrapped strategy expands the data combinatorially without requiring extra annotations.

### Loss & Training
The loss function is $\mathcal{L} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{box} + \beta \mathcal{L}_{asso}$, where classification and box regression losses follow DETR, and the association loss is the negative log-likelihood of the ground-truth matching. Ground-truth assignment is solved via bipartite matching with the Hungarian algorithm. By default, $\alpha=5, \beta=1, \mu=0.07$. An AdamW optimizer is used, taking 12 hours to train on Cityscapes using a single TITAN RTX GPU.

## Key Experimental Results

### Main Results

| Method | Cityscapes IOU50@1↑ | IOU@5↑ | OPA IOU50@1↑ | OPA IOU50@5↑ |
|------|---------------------|--------|-------------|-------------|
| PlaceNet | 0 | 0.045 | 2.76% | 10.09% |
| GracoNet | — | — | 2.49% | 16.60% |
| TopNet | 0.807% | 0.070 | 11.55% | 15.95% |
| **BootPlace** | **1.74%** | **0.281** | **11.60%** | **22.41%** |

User study: Cityscapes **0.303**, Mapillary Vistas **0.323** (both indicating the highest plausibility scores)

### Ablation Study

| Configuration | IOU50@5 | IOU@5 | Description |
|------|---------|-------|------|
| Full model | 6.09% | 0.281 | — |
| W/o bootstrapped training | 3.77% | 0.191 | Insufficient data diversity |
| Positive-correlation association | 3.23% | 0.166 | Objects placed on similar objects |
| W/o position encoding| 4.85% | 0.241 | Unaware of existing object locations |

### Key Findings
- Negative vs. positive correlation association has the greatest impact on performance (IoU@5: 0.281 vs. 0.166), validating the critical importance of semantic complementarity.
- Bootstrapped training contributes significantly (IoU50@5: 6.09% vs. 3.77%), showing obvious data augmentation effects.
- Top-5 IoU on Cityscapes is improved by approximately 4× compared to TopNet (0.281 vs. 0.070), showing that the localization accuracy of the detection paradigm far exceeds the regression paradigm.
- The model generalizes to Mapillary Vistas (trained on Cityscapes), yielding a user study plausibility score of 0.323.

## Highlights & Insights
- The **reverse thinking of placement as detection** is clever: instead of learning "where to place," it detects "where is lacking", migrating giant detection frameworks directly to the placement task, which has both theoretical validity and practical efficacy.
- **Negative-correlation matching** is counter-intuitive but reasonable: objects should be placed in regions whose semantics complement theirs (e.g., cars on roads) rather than similar regions (e.g., on top of another car). This design can be transferred to other synthesis/editing tasks.
- **Bootstrapped combinatorial augmentation** generates massive training data at nearly zero cost, representing an elegant solution to address sparse annotation.

## Limitations & Future Work
- Parallel detection of all locations cannot handle sequential placement, which may result in occlusion issues (e.g., vehicles overlapping with curbs).
- Object rotation and perspective transformations are not modeled, resulting in limited placement plausibility for orientation-sensitive objects (e.g., cars on curves).
- Potential overfitting to inpainting artifacts; although mitigated by Gaussian smoothing, it remains fundamentally unresolved.
- The OPA dataset only features single-object annotations, meaning the supervision signals for multi-object placement are limited.

## Related Work & Insights
- **vs TopNet**: TopNet utilizes Transformer-based location regression + sparse contrastive loss, where loose regularization leads to imprecision. BootPlace provides stronger localization capabilities leveraging detection constraints.
- **vs GracoNet**: GracoNet requires manual annotation of positive and negative sample pairs, whereas BootPlace automatically generates training data via a bootstrapping strategy, yielding superior scalability.
- **vs DiffPop**: DiffPop uses diffusion models to learn scale/spatial relations but relies on hand-crafted plausibility guidance. BootPlace directly learns plausible locations from detection.

## Rating
- Novelty: ⭐⭐⭐⭐ The placement-by-detection paradigm and negative-correlation matching are key innovations, though the technical framework is built upon the mature DETR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmarked on two datasets along with ablations, user study, and generalization tests. However, the absolute IoU values remain relatively low.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with mathematical formulations for method descriptions, accompanied by rich figures and tables.
- Value: ⭐⭐⭐ Interesting direction but limited practical scenarios, and low absolute IoU values suggest the problem itself is highly challenging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis](metashadow_object-centered_shadow_detection_removal_and_synthesis.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2025\] Composing Parts for Expressive Object Generation](composing_parts_for_expressive_object_generation.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](tinyfusion_diffusion_transformers_learned_shallow.md)

</div>

<!-- RELATED:END -->
