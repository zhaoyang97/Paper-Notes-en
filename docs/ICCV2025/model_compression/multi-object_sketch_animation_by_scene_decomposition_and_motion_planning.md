---
title: >-
  [Paper Note] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning
description: >-
  [ICCV 2025][Model Compression][Sketch Animation] MoSketch is the first method to address multi-object sketch animation. It integrates four modules — LLM-based scene decomposition, LLM-based motion planning, a motion refinement network, and compositional SDS — under a divide-and-conquer strategy to tackle two core challenges: object-aware motion modeling and complex motion optimization. High-quality multi-object sketch animation is achieved without any training data.
tags:
  - ICCV 2025
  - Model Compression
  - Sketch Animation
  - Multi-Object Animation
  - LLM Motion Planning
  - Score Distillation Sampling
  - Compositional Generation
date: 2026-05-08
content_hash: 2b1591dc15eaa412
---

# Multi-Object Sketch Animation by Scene Decomposition and Motion Planning

**Conference**: ICCV 2025
**arXiv**: [2503.19351](https://arxiv.org/abs/2503.19351)
**Code**: None (planned open-source)
**Area**: Diffusion Models
**Keywords**: Sketch Animation, Multi-Object Animation, LLM Motion Planning, Score Distillation Sampling, Compositional Generation

## TL;DR

MoSketch is the first method to address multi-object sketch animation. It integrates four modules — LLM-based scene decomposition, LLM-based motion planning, a motion refinement network, and compositional SDS — under a divide-and-conquer strategy to tackle two core challenges: object-aware motion modeling and complex motion optimization. High-quality multi-object sketch animation is achieved without any training data.

## Background & Motivation

**Background**: Sketch animation converts static sketches into dynamic videos, with broad applications in GIF design, cartoon production, and everyday entertainment. Recent methods include Live-Sketch (CVPR 2024), which employs vector sketch representations and Score Distillation Sampling (SDS) for training-free animation, and FlipSketch (CVPR 2025), which generates rasterized sketch animations via DDIM inversion and fine-tuned T2V models. Both perform well on single-object sketch animation.

**Limitations of Prior Work**: Extending single-object methods to the multi-object setting introduces fundamental difficulties. (1) Live-Sketch lacks object-aware motion modeling, failing to capture inter-object relationships and interactions (e.g., water level should decrease when being poured), and T2V diffusion models are difficult to guide effectively via SDS for complex multi-object motions. (2) FlipSketch's DDIM inversion fails to faithfully capture the appearance of multi-object sketches, and its fine-tuning data — synthesized by Live-Sketch — contains very few and low-quality multi-object scenes.

**Key Challenge**: Multi-object sketch animation poses two fundamental challenges: **object-aware motion modeling** (requiring consideration of relative motions, interactions, and physical constraints among objects) and **complex motion optimization** (T2V diffusion models struggle to provide effective SDS guidance for complex multi-object motions). No existing method addresses both simultaneously.

**Goal**: To propose a training-data-free method for multi-object sketch animation that simultaneously resolves both object-aware motion modeling and complex motion optimization.

**Key Insight**: Leverage LLM prior knowledge for scene understanding and motion planning (addressing the motion modeling challenge), combined with compositional SDS that decomposes complex motions into simpler ones for sequential optimization (addressing the optimization challenge).

**Core Idea**: A divide-and-conquer strategy — LLMs handle high-level planning (scene decomposition and coarse-grained motion), neural networks handle low-level refinement (fine-grained motion), and compositional SDS handles divide-and-optimize (decomposing complex motions into simple ones for individual guidance).

## Method

### Overall Architecture

MoSketch operates on vector sketch representations where each stroke is a cubic Bézier curve, and iteratively optimizes via SDS. Given a vector sketch $P \in \mathbb{R}^{n \times 2}$ and a text instruction $Y$, the method outputs a displacement sequence $\Delta Z \in \mathbb{R}^{n \times f \times 2}$ for all control points. The four modules operate sequentially: LLM scene decomposition → LLM motion planning → motion refinement network for fine-grained motion → compositional SDS for global optimization.

### Key Designs

1. **LLM-based Scene Decomposition**:

    - **Function**: Serves as the foundation of the entire pipeline; identifies objects, obtains their locations, and decomposes complex motions into simpler sub-motions.
    - **Mechanism**: Given sketch $P$ and text $Y$, GPT-4 identifies $m$ independent objects and $r$ decomposed simple motion descriptions $\{Y_k\}_{k=1}^r$ (each involving only 1–2 objects). Grounding DINO detects bounding boxes $B_0 \in \mathbb{R}^{m \times 4}$ for each object, and each control point is assigned to the nearest object based on the distance from the stroke center to the bounding boxes. Constraints: $m < 7$, $r < 5$.
    - **Design Motivation**: Decomposing non-directly-modelable composite motions into simple motions that T2V models can handle effectively, providing the structural foundation for subsequent motion planning and compositional optimization.

2. **LLM Motion Planning + Motion Refinement Network**:

    - **Function**: LLM generates coarse object-level motion plans; the refinement network generates fine-grained motion.
    - **Mechanism**: **Motion Planning**: GPT-4 takes the sketch, text instruction, and initial positions $B_0$ as input, and — after chain-of-thought reasoning — generates bounding box sequences $B \in \mathbb{R}^{m \times f \times 4}$ for all objects across $f$ frames, which are converted into a coarse object motion $\Delta Z_c$. GPT-4 is prompted to account for physical constraints such as inertia and gravity. **Refinement Network**: Built upon Live-Sketch with sketch-level motion replaced by object-level motion. Bounding box sequences $B$ and control points $P$ are encoded via MLPs into latent representations; a Transformer models inter-object relationships to produce object embeddings $\hat{B}$ and point embeddings $\hat{P}$. Object embeddings predict 7-parameter affine transformations (translation, scale, shear, rotation) per object to yield fine object motion $\Delta Z_o$; point embeddings are passed through object-specific MLPs to predict per-point displacements $\Delta Z_p$. The final animation is $\Delta Z = \Delta Z_c + \Delta Z_o + \Delta Z_p$.
    - **Design Motivation**: LLMs possess prior knowledge of object interactions and physical constraints, making them suitable for high-level motion planning (e.g., a cannonball should follow a parabolic trajectory), but their output precision is limited. The refinement network models inter-object relationships via Transformer to perform fine-grained correction of the coarse plan (e.g., the explosion effect when the cannonball reaches the target).

3. **Compositional SDS**:

    - **Function**: Ensures effective SDS guidance for complex multi-object motions during iterative optimization.
    - **Mechanism**: In addition to the standard SDS loss $\mathcal{L}_{SDS}$ (using full text $Y$ and full animation $\Delta Z$), an independent SDS loss $\mathcal{L}_{SDS-k}$ is computed for each decomposed simple motion $Y_k$. Specifically, a sub-video $\Delta Z_k$ is extracted from the full animation containing only the objects involved in $Y_k$, and the T2V model computes SDS on $(\Delta Z_k, Y_k)$. The total loss is: $\mathcal{L}_{CSDS} = \mathcal{L}_{SDS} + \sum_{k=1}^r \mathcal{L}_{SDS-k}$
    - **Design Motivation**: T2V diffusion models are more reliable when handling simple motions (e.g., "a ball flying toward the basket") than complex multi-object motions (e.g., "a player shoots and scores while a teammate sprints up the court"). Compositional SDS decomposes the complex problem into simpler sub-problems that the model handles well, providing effective gradient guidance for each.

### Loss & Training

Adam optimizer is used with a learning rate of 5e-3, weight decay of 1e-2, 500 optimization iterations, taking approximately 1 hour on a single RTX 3090 Ti. The hidden dimension is 128, the Transformer has 2 layers, and the number of frames is 16. No training data is required; gradient signals are obtained entirely via SDS from pre-trained T2V models.

## Key Experimental Results

### Main Results

Quantitative comparison on a test set of 60 multi-object sketches:

| Method | Text-Video Align↑ | Motion Smooth↑ | Sketch-Video Align↑ | Dynamic Degree↑ |
|--------|-------------------|----------------|---------------------|----------------|
| CogVideoX (I2V) | 0.141 | 0.610 | 0.747 | - |
| DynamiCrafter (I2V) | 0.184 | 0.771 | 0.868 | - |
| FlipSketch | 0.199 | 0.704 | 0.839 | - |
| Live-Sketch | 0.207 | 0.897 | 0.956 | 0.266 |
| **MoSketch** | **0.218** | **0.914** | **0.977** | **0.283** |

### Ablation Study

| Configuration | Text-Video↑ | Motion Smooth↑ | Sketch-Video↑ | Dynamic↑ | Note |
|---------------|------------|----------------|---------------|----------|------|
| w/o Motion Planning | 0.212 | 0.955 | 0.959 | 0.083 | External motion nearly absent |
| w/o Fine Object Motion | 0.212 | 0.909 | 0.964 | 0.266 | External motion insufficiently refined |
| w/o Point Motion | 0.203 | 0.971 | 0.971 | 0.200 | No internal motion |
| w/o Object-Aware Network | 0.205 | 0.932 | 0.968 | 0.266 | Motion insufficiently refined |
| w/o Compositional SDS | 0.207 | 0.911 | 0.966 | 0.267 | Lacking motion details |
| **MoSketch (Full)** | **0.218** | **0.914** | **0.977** | **0.283** | Best overall |

### Key Findings

- **LLM motion planning is critical**: Removing motion planning causes Dynamic Degree to drop sharply from 0.283 to 0.083, indicating that meaningful external motions (e.g., launching a cannonball) are nearly impossible without LLM coarse planning.
- **All three motion layers are indispensable**: $\Delta Z_c$ (coarse motion), $\Delta Z_o$ (fine object motion), and $\Delta Z_p$ (point motion) respectively control large-scale displacement, external motion refinement, and internal deformation; ablating any one layer causes degradation along a specific motion dimension.
- **Compositional SDS improves motion details**: Removing compositional SDS reduces Text-Video Align from 0.218 to 0.207, confirming that decomposed optimization helps T2V models more effectively guide complex motions.
- **The method exhibits robustness to point assignment errors**: Even when Grounding DINO's object localization or point assignment contains small errors, the final results remain visually compelling.
- **FlipSketch and I2V methods fail severely in the sketch domain**: I2V methods cannot preserve sketch appearance due to the domain gap between sketches and natural images, and FlipSketch's rasterized representation also fails to maintain multi-object sketch fidelity.

## Highlights & Insights

- **LLM as a physical intuition engine**: GPT-4's prior knowledge of real-world object motion — encompassing inertia, gravity, and collisions — is leveraged for motion planning. This "LLM plans, network executes" paradigm is broadly transferable to other generation tasks requiring physical understanding.
- **A complete divide-and-conquer pipeline**: Scene decomposition → motion planning → motion refinement → compositional optimization forms a closed-loop, top-down divide-and-conquer pipeline where each level addresses a clearly defined sub-problem.
- **Training-data-free design**: In the absence of multi-object sketch animation datasets, end-to-end animation generation is achieved through SDS combined with LLM priors, demonstrating the powerful capacity of composing pre-trained models.

## Limitations & Future Work

- **Sensitivity to point assignment quality**: When Grounding DINO produces significant detection errors (e.g., Godzilla's tail is incorrectly assigned to "the city"), the final animation quality degrades substantially.
- **LLM motion planning can be erroneous**: GPT-4 may misunderstand certain motions (e.g., a goalkeeper should move toward the ball), and incorrect coarse plans cannot be fully corrected by the refinement network.
- **Limitations of T2V motion comprehension**: T2V models may lack understanding of certain specialized motions (e.g., "fighting"), causing animation generation failures in those cases.
- The optimization process is relatively slow, requiring approximately one hour per animation.

## Related Work & Insights

- **vs. Live-Sketch (CVPR 2024)**: Live-Sketch is the direct foundation of MoSketch, performing well on single-object animation but lacking object-aware capabilities. MoSketch extends it with scene decomposition, motion planning, and compositional optimization, achieving comprehensive improvements in multi-object settings.
- **vs. FlipSketch (CVPR 2025)**: FlipSketch relies on rasterized representations and fine-tuning, resulting in appearance preservation failures in multi-object scenarios. MoSketch inherits the vector representation to maintain sketch appearance integrity.
- **vs. LLM-grounded Video Diffusion**: Works such as LLM-grounded VDM similarly leverage LLMs for trajectory planning in T2V generation. MoSketch introduces this paradigm into sketch animation for the first time, additionally incorporating compositional SDS to address multi-object optimization under the SDS framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to address multi-object sketch animation; the LLM divide-and-conquer + compositional SDS design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive quantitative and qualitative comparisons with full module-level ablations, though the test set size is relatively small (60 sketches).
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, method description is thorough, and figures are rich and intuitive.
- **Value**: ⭐⭐⭐⭐ Opens a new direction in multi-object sketch animation, though the application domain is relatively niche.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method](../../NeurIPS2025/model_compression/deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](../../NeurIPS2025/model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[ICCV 2025\] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)
- [\[CVPR 2026\] Unlocking ImageNet's Multi-Object Nature: Automated Large-Scale Multilabel Annotation](../../CVPR2026/model_compression/unlocking_imagenets_multi-object_nature_automated_large-scale_multilabel_annotat.md)

<!-- RELATED:END -->
