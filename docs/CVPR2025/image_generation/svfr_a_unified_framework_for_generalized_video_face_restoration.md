---
title: >-
  [Paper Note] SVFR: A Unified Framework for Generalized Video Face Restoration
description: >-
  [CVPR 2025][Image Generation][Video Face Restoration] This paper proposes SVFR, a unified video face restoration framework based on Stable Video Diffusion, which jointly trains three tasks—blind face restoration (BFR), colorization, and inpainting—within a single model. Through designs such as task embedding, unified latent space regularization, and facial prior learning, it achieves SOTA results across multiple video face restoration tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Face Restoration"
  - "Multi-task Learning"
  - "Diffusion Models"
  - "Temporal Consistency"
  - "Unified Framework"
date: 2026-05-08
content_hash: 661a3f6f1828473a
---

# SVFR: A Unified Framework for Generalized Video Face Restoration

**Conference**: CVPR 2025  
**arXiv**: [2501.01235](https://arxiv.org/abs/2501.01235)  
**Code**: [https://github.com/wangzhiyaoo/SVFR](https://github.com/wangzhiyaoo/SVFR)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Video Face Restoration, Multi-task Learning, Diffusion Models, Temporal Consistency, Unified Framework

## TL;DR
This paper proposes SVFR, a unified video face restoration framework based on Stable Video Diffusion, which jointly trains three tasks—blind face restoration (BFR), colorization, and inpainting—within a single model. Through designs such as task embedding, unified latent space regularization, and facial prior learning, it achieves SOTA results across multiple video face restoration tasks.

## Background & Motivation

**Background**: Face restoration is an important field in image/video processing, aiming to restore high-quality portraits from degraded inputs. While substantial work has been done on image-level face restoration (e.g., CodeFormer, GPEN, GFPGAN), video-level face restoration remains relatively under-explored.

**Limitations of Prior Work**: Video face restoration faces three core challenges: (1) additional complexity introduced by the temporal dimension, requiring the maintenance of consistency across multiple frames while handling motion artifacts, occlusions, and illumination changes; (2) scarcity of high-quality video training data, making it difficult to train robust models; (3) limited architectural capacity of existing methods (e.g., BasicVSR++, KEEP), which leads to low generation quality and severe temporal jitter/discontinuity.

**Key Challenge**: Traditional face restoration methods are typically trained separately for a single task (such as super-resolution), ignoring the shared prior knowledge among BFR, colorization, and inpainting. In practice, color degradation in low-quality videos is closely related to the colorization task, and restoring occluded regions aligns with the objective of the inpainting task, showing a natural complementarity among these tasks.

**Goal**: Design a unified framework to simultaneously handle three tasks—video BFR, colorization, and inpainting—leveraging shared representations across tasks to enhance supervision signals, thereby improving overall restoration quality and temporal stability.

**Key Insight**: The authors first validated the effectiveness of multi-task transfer learning through a pilot study: training three tasks using GPEN with and without pre-trained priors. The results demonstrated that prior knowledge among tasks can indeed mutually boost the FID metrics.

**Core Idea**: Build a unified multi-task video face restoration framework (GVFR) based on Stable Video Diffusion (SVD). By using task embedding to distinguish different tasks, unified latent space regularization to align cross-task features, and facial prior learning to inject facial structural information, these components collaboratively achieve high-quality and temporally stable video restoration.

## Method

### Overall Architecture
SVFR is built upon pre-trained Stable Video Diffusion (SVD). The input is a degraded source video $\mathbf{V}_d$ (low-quality video / grayscale video / masked video), and the output is the restored video $\mathbf{V}_r$. The source video is encoded by VAE and concatenated with noise, then fed into the diffusion U-Net for denoising. The overall pipeline contains three core modules: (1) Unified Face Restoration Framework (task embedding + unified latent space regularization); (2) Facial Prior Learning; and (3) Self-referred Refinement Strategy.

### Key Designs

1. **Unified Face Restoration Framework (Task Embedding & Unified Latent Regularization)**:

    - **Function**: Enables the model to distinguish between different tasks (BFR/colorization/inpainting) and align intermediate features from different tasks in a unified feature space.
    - **Mechanism**: Task embedding represents each task as a binary vector $\gamma = [t_1, t_2, t_3]$ (e.g., $[0,1,1]$ indicates activation of colorization and inpainting tasks), which is mapped through an embedding layer and added to the time embedding of the U-Net. Unified Latent space Regularization (ULR) constrains the intermediate layer features of the U-Net via contrastive learning loss: different degradation forms of the same video serve as positive pairs, while different videos serve as negative pairs, forcing the model to learn task-agnostic shared feature representations. The ULR loss is $\mathcal{L}_{ULR} = -\log \frac{\exp(\hat{x}_i \cdot \hat{x}_i^+ / \tau)}{\sum_{j=1}^N \exp(\hat{x}_i \cdot \hat{x}_j^- / \tau)}$.
    - **Design Motivation**: Degraded source videos from different tasks carry different prior information (BFR retains structure but lacks texture, while inpainting retains complete information in unoccluded regions). Simple concatenation cannot encode them into a consistent latent space. Task embedding provides explicit task identification to prevent model confusion, while ULR ensures feature consistency across tasks, facilitating knowledge transfer.

2. **Facial Prior Learning**:

    - **Function**: Injects facial structural priors into the pre-trained SVD to guide the model in generating structurally consistent facial details.
    - **Mechanism**: Extracts the feature $x_d$ from the intermediate blocks of the U-Net and predicts 68 facial landmarks via a landmark predictor $P_{lm}$ consisting of an average pooling layer and a five-layer MLP. A hybrid loss function $\mathcal{L}_{prior}$ is utilized: a log function $w \ln(1 + |x|/\epsilon)$ for precise alignment during small deviations, and a linear function $|x| - C$ to maintain robustness during large deviations. The ground truth landmarks are extracted from GT frames using a pre-trained detection model.
    - **Design Motivation**: The original training objective of SVD (noise prediction) lacks facial structure constraints, and direct fine-tuning is prone to generating structurally inconsistent faces (e.g., distorted eyes and mouths). The auxiliary landmark prediction objective forces the model to learn facial structure priors from noisy latent variables.

3. **Self-referred Refinement Strategy**:

    - **Function**: Maintains temporal consistency and style coherence across clips during long video inference.
    - **Mechanism**: During the training phase, a reference frame $I_{ref}$ is randomly provided, and its VAE encoding is injected into the initial U-Net noise, while the identity feature is injected into the cross-attention layers via a mapping network. A 50% dropout probability is applied to the reference frame to enhance generalization. During the inference phase, the first clip is generated without a reference frame, and then one frame from it is selected as the reference for subsequent clips, ensuring style and structural continuity in long sequences.
    - **Design Motivation**: Video face restoration often needs to process long videos, where segment-by-segment generation easily introduces issues such as color drift and identity inconsistency between segments. The self-reference mechanism guides subsequent segments using previously generated results, achieving global consistency.

### Loss & Training
The total loss is a weighted combination of three terms: $\mathcal{L} = \mathcal{L}_{noise} + \lambda_1 \mathcal{L}_{ULR} + \lambda_2 \mathcal{L}_{prior}$, where $\lambda_1 = 0.01$ and $\lambda_2 = 0.1$. Training data is derived from VoxCeleb2, CelebV-Text, and VFHQ, with 20,000 high-quality video clips filtered out via ARNIQA scoring.

## Key Experimental Results

### Main Results
Evaluating on the VFHQ-test dataset against SOTA methods, SVFR operates as a unified model handling three tasks simultaneously, whereas other methods require separate training for each task.

| Method | Task | PSNR↑ | SSIM↑ | LPIPS↓ | IDS↑ | FVD↓ |
|------|------|-------|-------|--------|------|------|
| GPEN | BFR | 26.237 | 0.795 | 0.320 | 0.786 | 412.81 |
| CodeFormer | BFR | 26.528 | 0.762 | 0.361 | 0.784 | 379.53 |
| PGTFormer | BFR | 28.996 | 0.843 | 0.248 | 0.845 | 154.86 |
| KEEP | BFR | 27.335 | 0.813 | 0.259 | 0.790 | 399.24 |
| **SVFR** | **BFR** | **29.563** | **0.862** | **0.223** | **0.902** | **89.32** |
| **SVFR** | **Colorization** | **23.079** | **0.896** | **0.272** | **0.980** | **204.26** |
| **SVFR** | **Inpainting** | **29.119** | **0.904** | **0.153** | **0.888** | **88.35** |

### Ablation Study

| Configuration | PSNR (BFR)↑ | FVD (BFR)↓ | PSNR (Colorization)↑ | FVD (Inpainting)↓ |
|------|------------|-----------|------------|-----------|
| Single-Task Training | 28.323 | 167.31 | 22.233 | 106.52 |
| Multi-Task Training | 28.936 | 98.78 | 22.921 | 101.15 |
| +ULR | 29.296 | 90.35 | 22.987 | 93.62 |
| +ULR+FPL (Full) | **29.563** | **89.32** | **23.079** | **88.35** |

### Key Findings
- Compared to single-task training, multi-task joint training improves all metrics, particularly reducing FVD from 167.31 to 98.78 (BFR), validating the effectiveness of knowledge sharing among tasks.
- ULR further improves performance on top of multi-task training, with significant improvements in LPIPS and FVD, indicating that feature space alignment is crucial for generation quality.
- Facial prior learning contributes most to the BFR and inpainting tasks (PSNR impr. by ~0.3-0.8), as these tasks rely more heavily on facial structural accuracy.
- The self-referred refinement strategy shows significant effects on long videos, effectively eliminating color drift and identity inconsistency issues.

## Highlights & Insights
- **Validation Path of Multi-task Mutual Assistance**: The pilot study simply and powerfully proves that transferable shared priors indeed exist among BFR, colorization, and inpainting, establishing a solid experimental foundation for subsequent unified framework design. This "validate before design" research paradigm is highly commendable.
- **Contrastive Learning Design of Unified Latent Space Regularization**: Unlike simple feature concatenation or weight sharing, ULR uses contrastive learning to explicitly constrain feature similarity under different degradation forms of the same video, which preserves task specificity while promoting cross-task consistency.
- **Self-referred Refinement Strategy**: The design of randomly dropping out reference frames during training is highly sophisticated, enabling the model to perform both reference-free and reference-based generation, allowing a natural transition during inference.

## Limitations & Future Work
- The training data consists of only 20,000 video clips; generalization capability in extreme degradation scenarios (e.g., co-occurrence of extremely low resolution, occlusion, and grayscale) remains to be validated.
- The method relies on a pre-trained facial landmark detector to provide GT landmarks; when degradation is severe, GT extraction might be inaccurate.
- Inference speed is constrained by the multi-step denoising process of diffusion models, making real-time applications difficult.
- Currently, only three degradation tasks are supported. Expanding to more video face processing tasks (such as deblurring, dehazing, etc.) could be considered.

## Related Work & Insights
- **vs KEEP**: KEEP is a method specifically designed for video BFR using an independent training strategy. SVFR achieves a better FVD (89.32 vs 399.24) through multi-task learning, demonstrating that joint training significantly improves temporal stability.
- **vs CodeFormer**: CodeFormer is an image-level method that processes frame-by-frame and lacks temporal modeling. SVFR, based on SVD, naturally possesses temporal priors, showing distinct advantages in VIDD and FVD.
- **vs PGTFormer**: PGTFormer performs well on BFR but does not support colorization and inpainting tasks. SVFR covers three tasks within a single model and achieves superior BFR performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of unified multi-task video face restoration is novel, though individual technical components (contrastive learning, landmark supervision) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons and ablation studies for the three tasks are comprehensive, and temporal stability visualization analysis is solid.
- Writing Quality: ⭐⭐⭐⭐ The logical flow from pilot study to method design is clear and structured.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for video face restoration, and the open-source code will drive development in this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[CVPR 2025\] OFER: Occluded Face Expression Reconstruction](ofer_occluded_face_expression_reconstruction.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](../../ICCV2025/image_generation/unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)

</div>

<!-- RELATED:END -->
