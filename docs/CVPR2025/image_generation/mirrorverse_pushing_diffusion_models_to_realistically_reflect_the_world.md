---
title: >-
  [Paper Note] MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World
description: >-
  [CVPR 2025][Image Generation][Mirror Reflection Generation] MirrorVerse constructs an enhanced synthetic dataset SynMirrorV2 (featuring random poses, rotations, and multi-object scenes) and leverages a three-stage curriculum training strategy to train MirrorFusion 2.0. This allows diffusion models to generate realistic mirror reflections for the first time, significantly outperforming prior methods in both synthetic and real-world scenarios.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Mirror Reflection Generation"
  - "Synthetic Dataset"
  - "Diffusion Model Fine-tuning"
  - "Curriculum Training"
  - "Physics-aware"
date: 2026-05-08
content_hash: d87f6066c49a5040
---

# MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World

**Conference**: CVPR 2025  
**arXiv**: [2504.15397](https://arxiv.org/abs/2504.15397)  
**Code**: [https://mirror-verse.github.io/](https://mirror-verse.github.io/)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Mirror Reflection Generation, Synthetic Dataset, Diffusion Model Fine-tuning, Curriculum Training, Physics-aware

## TL;DR
MirrorVerse constructs an enhanced synthetic dataset SynMirrorV2 (featuring random poses, rotations, and multi-object scenes) and leverages a three-stage curriculum training strategy to train MirrorFusion 2.0. This allows diffusion models to generate realistic mirror reflections for the first time, significantly outperforming prior methods in both synthetic and real-world scenarios.

## Background & Motivation

1. **Background**: Diffusion models have made tremendous progress in image generation and editing, but they still struggle to adhere to physical laws (such as shadows, reflections, and occlusions). Current state-of-the-art T2I models (e.g., SD 3.5, FLUX) produce geometrically inconsistent and distorted results when prompted to generate mirror reflections.
2. **Limitations of Prior Work**: The prior work MirrorFusion attempts to formulate this task as an inpainting problem using the synthetic dataset SynMirror, but its dataset lacks diversity—object positions are fixed, there are no rotation variations, and it only contains single-object scenes. This leads to poor generalization under different object poses and multi-object occlusion scenarios, making it difficult to transfer to the real world.
3. **Key Challenge**: High-quality mirror reflection images in the real world are extremely scarce, and there is a massive domain gap between synthetic and real-world data. The key challenge lies in introducing sufficient diversity within the synthetic data and bridging this domain gap through effective training strategies.
4. **Goal**: (1) How to scale synthetic data to cover more object poses and multi-object scenes? (2) How to ensure that models trained on synthetic data generalize well to the real world?
5. **Key Insight**: Integrate key dataset augmentations (random positioning, random rotation, and object grounding) into the rendering pipeline, construct multi-object scenes based on semantic pairing, and employ a three-stage curriculum training strategy to transition progressively from synthetic to real data.
6. **Core Idea**: By scaling the diversity of the synthetic data generation pipeline and employing a three-stage curriculum training process, an inpainting diffusion model can learn geometrically correct mirror reflections.

## Method

### Overall Architecture
The task is defined as an inpainting problem: given an image and a mask of the mirror region, the model generates correct mirror reflections inside the masked area. The inputs consist of the noisy latent $z_t$, masked image $z_m$, inpainting mask $x_m$, and depth map $x_d$. Utilizing a dual-branch U-Net architecture based on BrushNet, the conditioning branch takes the concatenated inputs and injects its features into the generation branch via zero convolutions.

### Key Designs

1. **SynMirrorV2 Dataset Augmentation Pipeline**:
    - **Function**: Build a large-scale synthetic reflection dataset featuring diverse object poses and multi-object occlusion scenarios.
    - **Mechanism**: (1) **Random position sampling**: Randomly sample $x$-$y$ coordinates of objects within the intersection of the mirror and camera view frustums, ensuring objects appear in both the scene and the mirror. (2) **Random rotation**: Randomly rotate objects around the $y$-axis to introduce orientation variation. (3) **Object grounding**: Employ a simple grounding technique to ensure objects stand on the floor rather than floating. (4) **Multi-object semantic pairing**: Pair categories (e.g., table-chair) based on the ABO dataset and avoid overlap using collision detection to generate 3,140 multi-object scenes.
    - **Design Motivation**: In SynMirror, object positions are fixed, rotations are absent, and only single objects are featured, which limits the model to specified viewing angles. The enhanced dataset contains 207,610 images across 66,062 3D objects with varied backgrounds and ground textures.

2. **Dual Branch + Depth**:
    - **Function**: Generate scene-consistent reflections while preserving information from unmasked regions.
    - **Mechanism**: Based on the BrushNet architecture, the conditioning branch receives the concatenated $[z_t, z_m, x_m, x_d]$, and injects the features of each layer into the generation branch via zero convolution: $\epsilon_\theta(z_t,t,c)_i = \epsilon_\theta(z_t,t,c)_i + w \cdot \mathcal{Z}(\epsilon'_\theta([z_t,z_m,x_m,x_d],t)_i)$. The depth map provides 3D spatial information to help the model comprehend the geometric relationship between the object and the mirror.
    - **Design Motivation**: Comparative experiments demonstrate that directly modifying SD Inpainting (SDI+Depth) causes color bleeding, as noisy latents and conditioning information mix in initial convolutional layers, restricting subsequent layers from extracting clean features. The dual-branch architecture processes conditioning information independently, yielding superior performance.

3. **Three-Stage Curriculum**:
    - **Function**: Gradually transition from synthetic to real-world data to bridge the domain gap.
    - **Mechanism**: **Stage 1** (40k iters): Train the dual-branch model initialized from SD v1.5 on the single-object subset of SynMirrorV2, leaving the generation branch unfrozen. **Stage 2** (10k iters): Fine-tune on the multi-object dataset to learn occlusions and spatial relationships. **Stage 3** (10k iters): Continue fine-tuning on the real-world MSD dataset, using a monocular depth estimator (Depth-Pro) to replace ground-truth depth. Text prompts are randomly dropped with a 20% probability during training.
    - **Design Motivation**: Directly training on real data suffers from insufficient data scale, while training solely on synthetic data introduces a domain gap. The curriculum training progressively moves from learning basic reflection geometry (single object), to complex occlusions (multi-object), and finally to adapting to the real-world distribution.

### Loss & Training
Standard diffusion denoising loss $L_{DM} = E[\|\epsilon - \epsilon_\theta(z_t,t,c)\|^2]$. Using the AdamW optimizer with a learning rate of $1e^{-5}$, batch size of 4 per GPU, across 4 A100 GPUs. During inference, CFG=7.5 is used with the UniPC scheduler for 50 steps.

## Key Experimental Results

### Main Results

Quantitative comparison on the MirrorBenchV2 single-object subset:

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP Sim↑ |
|------|-------|-------|--------|-----------|
| MirrorFusion (baseline) | 18.31 | 0.76 | 0.122 | 26.00 |
| **MirrorFusion 2.0 (Ours)** | **18.79** | **0.77** | **0.108** | 25.96 |

Quantitative comparison on the multi-object subset:

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP Sim↑ |
|------|-------|-------|--------|-----------|
| Ours 40k (without multi-object fine-tuning) | 17.77 | 0.743 | 0.126 | 26.17 |
| **Ours 50k (with multi-object fine-tuning)** | **18.00** | **0.744** | **0.119** | 26.09 |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Without random position/rotation (SynMirror) | Object orientation is often incorrect | Core flaw of the baseline method |
| Without multi-object training | Reflection fails in multi-object scenes, causing object blending | Demonstrates that multi-object scene data is irreplaceable |
| SDI + Depth (Single branch) | Severe color bleeding | Verifies the necessity of the dual-branch architecture |
| Without Stage 3 (no real-world fine-tuning) | Fails in complex real-world scenes | The last stage of curriculum training is crucial for generalization |

### Key Findings
* **Data augmentation contributes the most**: Random positioning and rotation are key to resolving the pose generalization issue, allowing the model to look past fixed viewing angles.
* **Multi-object training is irreplaceable**: Without training on multi-object scenes, the model experiences severe issues like object blending in occluded environments.
* **Curriculum training successfully bridges the domain gap**: After Stage 3 fine-tuning on real-world data, the model can handle complex real scenes such as messy tabletops, cables, and double reflections.
* In user studies, 84% of users prefer the generation results of MirrorFusion 2.0.

## Highlights & Insights
* **Teaching physical knowledge to diffusion models using synthetic 3D rendered data** is highly inspiring: for physics-aware generation tasks (reflections, shadows, refractions), constructing controllable synthetic data is far more efficient than collecting real data, and allows for precise labeling.
* **The synthetic-to-real curriculum training strategy** is worth adopting in other synthetic-to-real transfer scenarios: first learning simple atomic operations, then complex combinations, and finally adapting to real distributions.
* **The semantic-paired multi-object placement strategy** is simple yet effective: by manually defining category pairing relations and performing collision detection, semantically reasonable multi-object scenes are generated at a low cost.

## Limitations & Future Work
* The quantitative improvement is relatively small (PSNR increases by only around 0.5), with main advantages lying in qualitative performance and generalization.
* It only supports planar mirror reflections, without covering more complex reflection types like curved mirrors or water reflections.
* Artifacts still persist in scenes containing more than two objects; the dataset scale can be further expanded.
* The quality of the depth map significantly affects the results, and inaccurate monocular depth estimation during inference can lead to geometric deviations.
* The $512 \times 512$ resolution is relatively low, and high-resolution scaling remains unverified.

## Related Work & Insights
* **vs MirrorFusion**: The prior work utilizes the SynMirror dataset with fixed object positions and a dual-branch architecture that freezes the generation branch. This work comprehensively improves data diversity and training strategies, significantly outperforming it in pose variation and multi-object scenarios.
* **vs ObjectDrop**: ObjectDrop trains diffusion models to handle shadows and mirror reflections using a counterfactual dataset, but relies on real-world paired data collection. MirrorVerse is fully based on synthetic data, making it more scalable.
* **vs HD-Painter / BrushNet**: These general inpainting methods lack physical reflection knowledge, resulting in generations within the mirror region that do not conform to reflection geometry.

## Rating
* Novelty: ⭐⭐⭐ Core contributions lie in data and training strategies, and the methodology is relatively incremental.
* Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation spanning synthetic, GSO, and real domains, with user studies included.
* Writing Quality: ⭐⭐⭐⭐ Clear logic and comprehensive ablations.
* Value: ⭐⭐⭐⭐ The dataset and training strategies offer valuable reference for the physics-aware generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](../../NeurIPS2025/image_generation/rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[ICLR 2026\] QVGen: Pushing the Limit of Quantized Video Generative Models](../../ICLR2026/image_generation/qvgen_pushing_the_limit_of_quantized_video_generative_models.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)

</div>

<!-- RELATED:END -->
