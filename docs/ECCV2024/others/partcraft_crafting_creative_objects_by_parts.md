---
title: >-
  [Paper Note] PartCraft: Crafting Creative Objects by Parts
description: >-
  [ECCV 2024][Part-level control] This work proposes PartCraft, which achieves part-selection-based control for text-to-image generation for the first time. Users can "pick" different parts (such as a bird's head, wings, and body) from various objects, and the model naturally combines them into a novel and structurally coherent creative object.
tags:
  - "ECCV 2024"
  - "Part-level control"
  - "text-to-image generation"
  - "textual inversion"
  - "attention loss"
  - "creative generation"
date: 2026-05-08
content_hash: a3b6ef25907bf145
---

# PartCraft: Crafting Creative Objects by Parts

**Conference**: ECCV 2024  
**arXiv**: [2407.04604](https://arxiv.org/abs/2407.04604)  
**Code**: [https://github.com/kamwoh/partcraft](https://github.com/kamwoh/partcraft)  
**Area**: Others (Generative AI / Controllable Generation)  
**Keywords**: Part-level control, text-to-image generation, textual inversion, attention loss, creative generation

## TL;DR

This work proposes PartCraft, which achieves part-selection-based control for text-to-image generation for the first time. Users can "pick" different parts (such as a bird's head, wings, and body) from various objects, and the model naturally combines them into a novel and structurally coherent creative object.

## Background & Motivation

Current creative control in generative AI (such as Stable Diffusion) primarily relies on textual descriptions or sketches, but:
- **Imprecise Text Control**: Complex visual details are difficult to describe precisely in language, leading to generation results that deviate from expectations.
- **High Sketch Barrier**: Not all users possess detailed drawing skills.
- **Limitations of Prior Work**: Methods like DreamBooth and Textual Inversion learn the "entire object" as a unit, making them unable to achieve part-level combinatorial control.
- **Complex Extra Control Signals**: Methods based on bounding boxes or segmentation masks require extensive extra inputs.

**Core Motivation**: Human creativity often recombines different parts of existing concepts—for instance, wanting an "ideal bird" with a bluebird's head, a cardinal's wings, and a sparrow's body. PartCraft allows users to achieve such creative combinations through simple part "selection".

## Method

### Overall Architecture

PartCraft is built on Stable Diffusion v1.5 and adopts a Textual Inversion strategy. The overall workflow is as follows:
1. Unsupervised Part Discovery: Leveraging DINOv2 features for three-tier hierarchical clustering to decompose objects into semantic parts.
2. Part Encoding: Mapping each part into the text token space.
3. Attention-Loss-Based Training: Ensuring that all parts are correctly placed in the image and do not overlap with each other.
4. Bottleneck Encoder: Accelerating convergence and enhancing generation fidelity.

### Key Designs

1. **Unsupervised Part Discovery (Three-Tier Hierarchical Clustering)**:

    - DINOv2 is utilized to extract feature maps from all training images.
    - **Top Tier**: K-means (k=2) separates foreground and background.
    - **Middle Tier**: Clusters foreground patches into $M$ semantic parts (e.g., bird's head, wings, etc.).
    - **Bottom Tier**: Further subdivides each middle-tier cluster into $K$ sub-categories (corresponding to the same part of different species).
    - Each region of every image receives a cluster label $p = (0, k_0), (1, k_1), ..., (M, k_M)$, serving as the text description during training.
    - DINOv2 is preferred over models like VLPart for its higher flexibility and domain generalization capability.

2. **Part Token Bottleneck Encoder**:

    - Traditional textual inversion directly learns word embeddings $e(p)$, where tokens lack information interaction, leading to low learning efficiency.
    - Introduce a bottleneck network $f(\cdot)$ consisting of a two-layer MLP + ReLU: $y_p = f(e(p))$.
    - **Core Idea**: First project tokens into a shared "part category space" (e.g., a general representation of a "head"), and then fine-tune it to adapt to specific details.
    - Experiments show significantly accelerated convergence (traditional methods are a special case where $f$ is the identity function).

3. **Entropy-Based Normalized Attention Loss**:

    - Training solely with the diffusion loss $\mathcal{L}_{ldm}$ leads to part entanglement (since the head and body of the same species always appear in pairs).
    - Formulate an attention regularization in the form of cross-entropy:
    $\mathcal{L}_{attn} = \mathbb{E}_{z,t,m}[-(S_m \log \hat{A}_m + (1-S_m)\log(1-\hat{A}_m))]$
    - Here, $\hat{A}_m$ represents the attention map normalized across all parts, and $S_m$ is the segmentation mask of the $m$-th part.
    - **Key of Normalization**: Ensures that the sum of attention of all parts at each image location is 1, meaning each location is occupied by at most one part.
    - Compared to the MSE attention loss of Break-a-Scene, the entropy loss naturally fits the constraint of "only one part appearing."

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{ldm} + \lambda_{attn} \mathcal{L}_{attn}$, where $\lambda_{attn} = 0.01$

- LoRA is used to fine-tune cross-attention blocks (instead of the full model) to reduce training overhead.
- Attention maps are focused on the 16×16 resolution layer (rich in semantic information).
- $M=5$ parts (head, chest/belly, wings, legs, tail) are set for birds, and $M=7$ parts are set for dogs.
- $K=256$ ensures coverage of all fine-grained categories.

## Key Experimental Results

### Main Results

Evaluated part reconstruction on CUB-200-2011 (birds) and Stanford Dogs:

| Method | FID↓ | CLIP↑ | DINO↑ | EMR↑ | CoSim↑ |
|------|------|-------|-------|------|--------|
| Textual Inversion | 10.10 | 0.784 | 0.607 | 0.305 | 0.842 |
| DreamBooth | 12.94 | 0.775 | 0.594 | 0.355 | 0.856 |
| Custom Diffusion | 37.61 | 0.694 | 0.504 | 0.338 | 0.833 |
| Break-a-Scene | 20.05 | 0.742 | 0.549 | 0.390 | 0.854 |
| **PartCraft** | **12.86** | **0.783** | **0.618** | **0.460** | **0.882** |

PartCraft exceeds the strongest baseline Break-a-Scene by 7% in EMR (Exact Match Rate) and 2.8% in CoSim.

### Ablation Study

| Configuration | FID↓ | EMR↑ | CoSim↑ | Description |
|------|------|------|--------|------|
| Full PartCraft | 12.86 | 0.460 | 0.882 | Full Model |
| w/o Bottleneck | 16.36 | ~0.460 | ~0.882 | FID degrades by 3.5, generation quality drops |
| MSE attn loss (BaS) | - | Significantly drops | Significantly drops | EMR/CoSim degrade substantially |
| w/o both | - | Worst | Worst | Double degradation |

### Key Findings

- **More part combinations make it harder**: As the number of mixed species increases from 1 to 4, both EMR and CoSim decrease.
- **PartCraft still significantly outperforms other methods under 4-species combinations**. Although Break-a-Scene also uses attention loss, its effect is inferior to the normalized entropy loss of this work.
- **Word embedding space visualization (tSNE)** shows that PartCraft's part tokens naturally cluster semantically (heads cluster together, wings cluster together), whereas other methods yield chaotic embeddings.
- **Cross-domain transfer**: The learned dog parts can be transferred to cats/lions (e.g., "a cat with beagle ears"), and can also be used for creative generation (e.g., a bird-shaped robot).

## Highlights & Insights

- **Select-to-create**: Simplifies the creative process to "clicking and selecting parts," without requiring text descriptions or drawing skills, making it elegant and practical.
- **Entropy-normalized attention loss** is the core contribution—solving the entanglement problem in multi-part learning, with clear design motivation (each position belongs to only one part).
- The Bottleneck encoder design cleverly leverages "shared-part knowledge," allowing different instances of the same semantic (e.g., heads of different birds) to share a representation space.
- The unsupervised part discovery scheme is flexible and scalable, requiring no part annotations.

## Limitations & Future Work

- Part discovery relies on the self-supervised features of DINOv2, which imposes an inherent ceiling on accuracy; stronger encoders (such as improved versions of VLPart) could be introduced.
- **Combination results for small parts (e.g., tails, legs) are relatively poor**, because these parts occupy small areas in the images, making both clustering and attention supervision less precise.
- Cross-domain part combination (e.g., combining animal parts with car parts) is still in its infancy.
- Only validated on Stable Diffusion v1.5; upgrading to newer models might yield better performance.

## Related Work & Insights

- Closest to Break-a-Scene, but the latter's MSE attention loss is less effective than the entropy loss proposed in this work.
- Can inspire the 3D generation field: if similar part-level composition control could be achieved on 3D models, it would hold immense application value.
- The part discovery module can be used independently to provide semantic segmentation for other fine-grained control tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (The idea of part selection -> creative generation is novel, though the core technique is based on existing frameworks)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets + quantitative & qualitative + ablation + visualization + rich transfer experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Motivating examples are intuitive, and method descriptions are clear)
- Value: ⭐⭐⭐⭐ (Holds practical application potential for creative design fields)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](../../ICLR2026/others/articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ECCV 2024\] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)
- [\[ECCV 2024\] High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding](high-fidelity_3d_textured_shapes_generation_by_sparse_encoding_and_adversarial_d.md)

</div>

<!-- RELATED:END -->
