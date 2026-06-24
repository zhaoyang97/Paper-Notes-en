---
title: >-
  [Paper Note] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects
description: >-
  [CVPR 2025][3D Vision][3D Generation Evaluation] This paper proposes Gen3DEval, a text-to-3D generation quality evaluation framework based on fine-tuned vLLMs. By fine-tuning the Llama3 model on synthetic and human-annotated data, it achieves automatic evaluation of 3D object appearance, surface quality, and text consistency, significantly outperforming general models like GPT-4o in human preference alignment.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Generation Evaluation"
  - "Vision-Language Models"
  - "Human Preference Alignment"
  - "ELO Rating"
  - "Multi-view Rendering"
date: 2026-05-08
content_hash: f0529b1dfd72dfc0
---

# Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects

**Conference**: CVPR 2025  
**arXiv**: [2504.08125](https://arxiv.org/abs/2504.08125)  
**Code**: [https://shalini-maiti.github.io/gen3deval.github.io/](https://shalini-maiti.github.io/gen3deval.github.io/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: 3D Generation Evaluation, Vision-Language Models, Human Preference Alignment, ELO Rating, Multi-view Rendering

## TL;DR
This paper proposes Gen3DEval, a text-to-3D generation quality evaluation framework based on fine-tuned vLLMs. By fine-tuning the Llama3 model on synthetic and human-annotated data, it achieves automatic evaluation of 3D object appearance, surface quality, and text consistency, significantly outperforming general models like GPT-4o in human preference alignment.

## Background & Motivation
1. **Background**: Text-to-3D generation has developed rapidly in recent years (diffusion models, NeRF, Gaussian Splatting), but lacks standardized evaluation metrics consistent with human judgment.
2. **Limitations of Prior Work**: PSNR/SSIM/Chamfer Distance require ground truth data, which is practically infeasible (a single prompt can correspond to multiple valid outputs); CLIP only evaluates text consistency, overlooking appearance and surface quality; FID requires a large-scale standard distribution, which is computationally expensive and inconsistent.
3. **Key Challenge**: Text-to-3D is a one-to-many mapping without a unique reference, making similarity metrics fundamentally inapplicable; furthermore, existing metrics evaluate only a single dimension, failing to provide a comprehensive assessment.
4. **Goal**: To build an automatic evaluation framework that requires no ground truth, comprehensively evaluates appearance, surface quality, and text consistency, and aligns closely with human preferences.
5. **Key Insight**: Although GPT-4V can perform 3D evaluation, it is not a dedicated model and shows limited effectiveness; the authors argue that specialized fine-tuning tailored for 3D quality evaluation is necessary.
6. **Core Idea**: Fine-tune a vLLM using synthetically perturbed data and human preference annotations, enabling the model to assess 3D object quality across three dimensions from multi-view rendered images.

## Method

### Overall Architecture
Gen3DEval operates in two stages: Stage 1 trains a vLLM capable of pairwise comparison, and Stage 2 uses this model to conduct pairwise evaluation on Gen3DEval-Bench and calculate ELO ratings. The inputs are multi-view renderings of 3D objects (up to 8 images, including RGB and normal maps), and the output is a judgment of which object is better in a specified dimension.

### Key Designs

1. **Two-Stage vLLM Training (Pre-training + SFT)**:
    - **Function**: Enables the vLLM to understand multi-view 3D renderings and make quality assessments.
    - **Mechanism**: During the pre-training stage, the LLM and vision encoder are frozen, and only the vision-language projection matrix $W_\theta$ is trained using VQA training on multi-view renderings and text descriptions of 140k 3D artist meshes. In the SFT stage, the projection matrix and LLM are unfrozen, and the model is instruction-tuned on pairwise comparison data to learn preference judgment across three dimensions: appearance, surface quality, and text consistency.
    - **Design Motivation**: The two-stage strategy ensures that vision-language alignment is established first before focusing on the specific task of 3D quality evaluation, avoiding the instability of direct end-to-end training.

2. **Synthetic Perturbation Data Construction**:
    - **Function**: Generates large-scale pairwise comparison data required for training.
    - **Mechanism**: Based on high-quality 3D meshes created by artists, controllable perturbations are introduced via Blender/NeRF/Gaussian Splatting: Laplacian smoothing, random surface bumps, texture blur/seams, transparency artifacts, floating elements, and broken components, simulating common defects in 3D generation methods. Text consistency data is generated using multi-view diffusion models with different captions, filtered by CLIP for low-quality samples.
    - **Design Motivation**: Since human annotation data is limited (5k+ samples), synthetic perturbations allow for large-scale training set expansion with precise control over defect types.

3. **Multi-View Input and Vision Encoder Selection (CLIP)**:
    - **Function**: Comprehensively captures 3D object quality information from multiple viewpoints.
    - **Mechanism**: Uses up to 8 multi-view RGB and normal maps as input. Three vision encoders (CLIP, DinoV2, and Fit3D) and their combinations were compared. CLIP (336×336 resolution, ViT architecture) performs most consistently across all evaluation dimensions, significantly leading in OOD generalization. Each image yields 576 visual tokens.
    - **Design Motivation**: Single views may miss occluded areas and hidden surface issues; CLIP exhibits the best generalization performance and is thus chosen as the default encoder.

### Loss & Training
- Pre-training uses the maximum likelihood objective of next token prediction, with a batch size of 16, a learning rate of 1e-3, a cosine scheduler, trained on 8×A100 GPUs for 1 day.
- SFT uses the same next token prediction objective, with a batch size of 4, a projection matrix learning rate of 2e-6, a vLLM learning rate of 1e-5, trained on 16×A100 GPUs for 18 hours.
- During evaluation, pairwise comparisons combined with an ELO rating system generate the final ranking.

## Key Experimental Results

### Main Results

| Method | Appearance (Human) | Appearance (OOD) | Surface (Synthetic) | Text Consistency (OOD) |
|------|-----------|----------|---------------|---------------|
| CLIP Score | 0.30 | 0.17 | 0.30 | 0.80 |
| GPT-4o | 0.59 | 0.69 | 0.54 | 0.55 |
| LLaVA-Qwen-7B | 0.54 | 0.54 | 0.51 | 0.58 |
| **Gen3DEval (CLIP)** | **0.90** | **0.89** | **0.99** | **0.86** |

### Ablation Study (Vision Encoder)

| Encoder Configuration | Appearance (Human) | Appearance (OOD) | Surface (OOD) | Text (OOD) |
|-----------|-----------|----------|----------|----------|
| CLIP | **0.90** | **0.89** | 0.67 | **0.86** |
| CLIP + Fit3D | 0.90 | 0.78 | 0.57 | 0.53 |
| CLIP + DinoV2 | 0.86 | 0.78 | 0.51 | 0.74 |
| DinoV2 only | 0.77 | 0.54 | 0.61 | 0.58 |
| Fit3D only | 0.81 | 0.55 | 0.44 | 0.44 |

### Key Findings
- The CLIP encoder performs most consistently across all evaluation dimensions, and its OOD generalization capability significantly outperforms other encoders, making it the default encoder.
- Gen3DEval outperforms all baseline methods in appearance evaluation by a large margin (0.90 vs GPT-4o's 0.59).
- On Gen3DEval-Bench, Trellis ranks first, AssetGen ranks second, while DreamFusion ranks lower.
- Multi-view inputs severely impact models that do not support multi-image inputs (such as BLIP, PaliGemma); replacing them with stitched grid images yields poor results.

## Highlights & Insights
- **Synthetic perturbation strategy** is highly elegant: introducing controllable defects directly into high-quality artist-created 3D meshes secures data quality and allows large-scale scaling. This "good-to-bad" synthesis strategy can be transferred to other quality evaluation tasks.
- **Introduction of normal maps** is a key innovation: by simultaneously analyzing RGB and surface normal renderings, the model can assess geometric quality rather than just appearance, which is crucial in 3D evaluation.
- The use of the **ELO rating system** converts pairwise comparison results into scalable scores (similar to chess ratings), which is more robust than direct voting.

## Limitations & Future Work
- The evaluation of Janus face artifacts is not stable enough, likely due to a lack of such samples in the training data.
- OOD surface evaluation still has room for improvement, limited by the lack of diversely annotated surface comparison data.
- The evaluation of image-to-3D methods can be affected by the quality of the upstream text-to-image pipeline.
- Although the 8.35B parameter size is much smaller than GPT-4, the deployment cost remains non-negligible, suggesting distillation into smaller models in the future.
- Currently, only static 3D objects are evaluated; dynamic scenes and 4D content are not yet covered.

## Related Work & Insights
- **vs GPT4VEval**: GPT4VEval uses general GPT-4V for 3D evaluation without task-specific fine-tuning; Gen3DEval fine-tunes an 8.35B model with specialized data, significantly outperforming it across all dimensions.
- **vs CLIP Score**: CLIP can only evaluate text-image alignment, failing to assess appearance and surface quality; Gen3DEval unifies three evaluation dimensions.
- **vs T3Bench**: T3Bench provides prompt benchmarks but lacks an automatic evaluation model; Gen3DEval provides both a benchmark (Gen3DEval-Bench) and an evaluation model.
- The methodology of this work—fine-tuning vLLMs on domain-specific tasks for automatic evaluation—can be transferred to quality evaluation in other generative tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The first work specifically fine-tuning a vLLM for 3D generation quality evaluation. The direction is novel, though the framework is based on LLaVA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation, ablation of multiple encoders, and comparison against several models. Gen3DEval-Bench is well-designed.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-explained motivations and rich tabular data.
- Value: ⭐⭐⭐⭐ Standardizes 3D generation evaluation, solves practical pain points, and provides a benchmark that benefits the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation](eval3d_interpretable_and_fine-grained_evaluation_for_3d_generation.md)
- [\[ICCV 2025\] How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach](../../ICCV2025/3d_vision/how_far_are_ai-generated_videos_from_simulating_the_3d_visual_world_a_learned_3d.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] ASHiTA: Automatic Scene-grounded Hierarchical Task Analysis](ashita_automatic_scene-grounded_hierarchical_task_analysis.md)
- [\[CVPR 2025\] MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)

</div>

<!-- RELATED:END -->
