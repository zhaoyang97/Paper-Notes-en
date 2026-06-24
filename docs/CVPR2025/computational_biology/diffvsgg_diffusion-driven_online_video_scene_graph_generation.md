---
title: >-
  [Paper Note] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation
description: >-
  [CVPR 2025][Computational Biology][Video Scene Graph Generation] DiffVsgg is proposed to model Video Scene Graph Generation (VSGG) as an iterative denoising problem along the temporal axis. It unifies object classification, box regression, and relation prediction using a shared feature embedding. Through latent diffusion models for spatial reasoning and using prior-frame predictions as conditioning for temporal reasoning, it achieves the first online VSGG and accomplishes com…
tags:
  - "CVPR 2025"
  - "Computational Biology"
  - "Video Scene Graph Generation"
  - "Latent Diffusion Models"
  - "Online Inference"
  - "Temporal Reasoning"
  - "Unified Embedding"
date: 2026-05-08
content_hash: 54860e773ab49812
---

# DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.13957](https://arxiv.org/abs/2503.13957)  
**Code**: [https://github.com/kagawa588/DiffVsgg](https://github.com/kagawa588/DiffVsgg)  
**Area**: Computational Biology  
**Keywords**: Video Scene Graph Generation, Latent Diffusion Models, Online Inference, Temporal Reasoning, Unified Embedding

## TL;DR
DiffVsgg is proposed to model Video Scene Graph Generation (VSGG) as an iterative denoising problem along the temporal axis. It unifies object classification, box regression, and relation prediction using a shared feature embedding. Through latent diffusion models for spatial reasoning and using prior-frame predictions as conditioning for temporal reasoning, it achieves the first online VSGG and accomplishes comprehensive SOTA performance across all three evaluation protocols on Action Genome, surpassing DSG-DETR by 3.3 points in R@10.

## Background & Motivation

**Background**: Video Scene Graph Generation (VSGG) aims to construct directed graphs from video sequences, where nodes represent objects and edges represent relationships (predicates) between objects. Current state-of-the-art methods (e.g., DSG-DETR, TR2, TEMPURA) employ offline pipelines: generating scene graphs for each frame independently before aggregating frame-level predictions along the timeline.

**Limitations of Prior Work**: (1) **Offline mode** requires the entire video sequence as input, making it incapable of handling real-time video streams (e.g., autonomous driving, AR) and resulting in high GPU memory consumption for long videos. (2) **Shallow temporal reasoning**: Offline methods only use Transformers for global temporal aggregation without performing actual step-by-step temporal reasoning, failing to model the dynamic evolution of relationships (e.g., "moving close to" $\rightarrow$ "touching"). (3) **Decoupled pipelines**: Object detection, temporal association, and context aggregation are split into independent steps, where errors in preceding steps propagate downstream. (4) Demands complex post-processing steps (e.g., NMS, cross-frame entity matching).

**Key Challenge**: VSGG requires both spatial reasoning (intra-frame object relationships) and temporal reasoning (cross-frame relationship dynamics). However, existing methods decouple these two aspects and cannot operate online.

**Goal**: (1) To design a high-performance online VSGG method; (2) To unify spatial and temporal reasoning into a single framework; (3) To eliminate cumbersome post-processing modules.

**Key Insight**: The process of continuously updating scene graphs across video frames and iteratively refining nodes and edges frame-by-frame is highly analogous to the step-by-step denoising process of diffusion models. The authors model VSGG as an iterative denoising process along the temporal dimension, where spatial reasoning within frames is accomplished via diffusion denoising, and temporal reasoning between frames is realized by treating preceding frame results as conditioning inputs for the denoising steps.

**Core Idea**: To unify spatial and temporal reasoning in VSGG using the denoising process of latent diffusion models, where the denoising output of the preceding frame acts as the conditioning input for the current frame to enable online frame-by-frame inference.

## Method

### Overall Architecture
DiffVsgg consists of three parts: (1) an off-the-shelf object detector to extract frame-level object features and bounding boxes; (2) a Latent Diffusion Model (LDM) to denoise the unified feature embeddings of object pairs, recovering clean object relationship features; (3) task-specific heads (MLP classifiers/projectors) to decode predicate classification, object classification, and box regression results from the denoised output. The entire process is executed online, frame-by-frame, with the current frame's denoising conditioned on the prior frame's denoising results.

### Key Designs

1. **Unified Embedding**:

    - Function: Encodes object classification, spatial location, and relationship information simultaneously within a single shared embedding, serving as both input and output for the LDM
    - Mechanism: For each pair of objects $(i,j)$ in frame $t$, an adjacency matrix element $A_{i,j}^t = [F_{o_i}^t; F_{o_i,o_j}^t; F_{b_i}^t]$ is constructed by concatenating subject features (via ROIAlign), subject-object union region features, and spatial bounding box features. A subject-oriented encoding is employed—$A_{i,j}$ treats only $i$ as the subject while $A_{j,i}$ treats $j$ as the subject, allowing the network to distinguish relational directionality. The matrix is padded to a fixed size $N \times N$ to accommodate variations in object counts across different frames
    - Design Motivation: In traditional pipelines, detection, classification, and relationship prediction rely on independent networks, causing errors to accumulate cascadingly. The unified embedding enables the three tasks to share the same representation, allowing the LDM to optimize all tasks simultaneously in a single denoising process, preventing cascaded errors. This also eliminates post-processing steps such as NMS and cross-frame entity matching.

2. **Spatial Reasoning via LDMs**:

    - Function: Recovers clean inter-object relationship features through the diffusion denoising process
    - Mechanism: During the training stage, precise adjacency matrices $\hat{A}^{t,0}$ are constructed using ground-truth boxes to represent "clean" data, which are then corrupted to $\hat{A}^{t,k}$ via the standard forward diffusion process. A Denoising U-Net $\epsilon_\theta$ is trained to predict the noise and progressively reconstruct the data. In the inference stage, the adjacency matrix $A^t$ constructed from the detector output is treated as a "noisy" version, and the trained U-Net performs $K$-step denoising to yield the refined $A^{t,0}$. The training loss is defined as $\mathcal{L}_{VSGG} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\hat{A}^{t,k}, k)\|_2^2]$
    - Design Motivation: Because the output of object detectors is prone to imperfections—such as bounding box offsets and misclassified categories—it is natural and effective to treat them as "noisy" inputs and denoise them using diffusion models. Additionally, the multi-scale structure of the U-Net is capable of capturing both local (individual object pairs) and global (entire scene) relationship patterns simultaneously.

3. **Condition-Based Temporal Reasoning**:

    - Function: Leverages the denoising results of the prior frame as conditioning guidance to denoise the current frame, enabling online temporal reasoning
    - Mechanism: The denoised result of the prior frame $A^{t-1,0}$ is introduced as conditioning input to the U-Net: $A^{t,k-1} = \frac{1}{\sqrt{\alpha_t}}(A^{t,k} - \frac{\beta_t}{\sqrt{1-\bar\alpha}} \epsilon_\theta(A^{t,k}, k, A^{t-1,0}))$. Since $A^{t-1,0}$ and $A^{t,k}$ share identical dimensions, no extra conditional encoders are required. Simultaneously, a memory bank is maintained to store the historical trajectory of each object, explicitly calculating the relative approaching velocity $v_{i,j}^t$ of object pairs (the rate of change of centroid distance between two frames), which is injected into each denoising step.
    - Design Motivation: Object relationships within a video are dynamic (e.g., "moving away" $\rightarrow$ "moving close" $\rightarrow$ "touching"). Passing the resolved relational information from the prior frame to the current frame allows the model to infer relationship evolution. Velocity information provides a more direct motion cue—detecting whether two objects are approaching or moving away helps directly in inferring temporal relationships like "following" or "approaching".

### Loss & Training
Two-stage training is adopted: (1) In Stage 1, the Denoising U-Net is trained on GT boxes, guided by the loss $\mathcal{L}_{T\_VSGG}$ (conditioned diffusion denoising loss); (2) In Stage 2, the U-Net is frozen, and the MLP heads are trained for predicate classification ($\mathcal{L}_{pred\_cls}$), object classification ($\mathcal{L}_{obj\_cls}$), and bounding box regression ($0.5 \cdot \mathcal{L}_{box\_reg}$, Smooth L1). Each training clip consists of 5 frames sampled at random temporal intervals.

## Key Experimental Results

### Main Results (Action Genome, w/ constraint)

| Method | Mode | PredCLS R@10 | SGCLS R@10 | SGDET R@10 | SGDET mR@10 |
|------|------|-------------|-----------|-----------|------------|
| STTran | Offline | 68.6 | 46.4 | 25.2 | 16.6 |
| TEMPURA | Offline | 68.8 | 47.2 | 28.1 | 18.5 |
| DSG-DETR | Offline | - | 50.8 | 30.3 | - |
| TR2 | Offline | 70.9 | 47.7 | 26.8 | - |
| **DiffVsgg** | **Online** | **71.9** | **52.5** | **32.8** | **20.9** |

### Ablation Study

| Configuration | SGCLS R@10 | Description |
|------|-----------|------|
| Full model (LDM + Conditional Temporal + Motion) | 52.5 | Full model |
| w/o Conditional Temporal Reasoning | ~48 | Removes prior-frame conditioning, decaying temporal reasoning |
| w/o Motion Enhancement | ~50 | Removes velocity information, degrading relation reasoning |
| w/o Unified Embedding (Independent Heads) | ~47 | Decoupled setup increases cascaded errors |

### Key Findings
- Despite being an online method, DiffVsgg comprehensively outperforms all offline methods—surpassing DSG-DETR by 2.5 points and TEMPURA by 4.7 points in SGDET R@10.
- The performance gain is most pronounced in the most challenging SGDET evaluation protocol (joint detection and prediction from scratch), showing that the unified embedding and LDM denoising effectively alleviate cascaded errors.
- Conditional temporal reasoning is crucial for bridging the gap between online and offline performance—the prior-frame condition allows the online model to exploit historical information effectively.
- The mR (Mean Recall) metric is also swept across the board, demonstrating improvements in long-tailed relational categories.

## Highlights & Insights
- **The analogy of VSGG $\approx$ temporal denoising** is highly natural: iterative scene graph updates along video frames correspond to progressive denoising in diffusion models, and cross-frame prediction propagation corresponds to conditional diffusion. A solid analogy is often the signature of a great paper.
- **Unified embedding eliminating post-processing**: Object detection, classification, and relationship prediction share a single adjacency matrix embedding. The LDM optimizes all three tasks in a single denoising pass, dispensing with hand-crafted modules like NMS or entity matching. This concept of "unified representation + unified optimization" can be generalized to other computer vision understanding tasks that require multi-task cascading.
- **Motion memory bank**: Explicitly calculating the relative approach/departure velocity of object pairs and injecting it into the denoising process serves as a simple yet effective prior, which directly assists in inferring motion-related predicates (e.g., "following", "approaching").
- **Online mode**: Achieves the first ever online VSGG inference, carrying significant importance for real-time applications such as autonomous driving and AR.

## Limitations & Future Work
- Dependence on off-the-shelf object detectors (frozen Faster R-CNN): Poor extraction from the detector limits VSGG performance. End-to-end training of the detector with the LDM could yield further improvements.
- The adjacency matrix is padded to a fixed $N \times N$ size, which may become inefficient when the number of objects is large or fluctuates heavily across scenes.
- Diffusion denoising requires multi-step iterations, potentially rendering the inference speed slower than simple Transformer-based methods (inference FPS is not reported in the paper).
- Validation is restricted to the Action Genome dataset, lacking evaluations on other VSGG benchmarks or more complex scenarios (e.g., dense multi-person interactions).
- The subject-oriented encoding prevents $A_{i,j}$ from incorporating independent features of the object $j$, potentially discarding certain relational reasoning cues.

## Related Work & Insights
- **vs DSG-DETR**: DSG-DETR uses DETR for scene graph detection but operates offline. DiffVsgg runs online and achieves higher SGCLS R@10 (+1.7 points) and SGDET R@10 (+2.5 points).
- **vs STTran/TEMPURA**: These methods perform temporal aggregation via Transformers, relying on global attention rather than true frame-by-frame reasoning. DiffVsgg's conditional diffusion mechanism ensures more fine-grained propagation of temporal information.
- **vs DiffusionDet**: DiffusionDet utilizes diffusion for object detection (denoising bounding boxes). DiffVsgg, in contrast, applies denoising to "relationship embeddings"—extending the application from detection to comprehension, which marks a significant step forward for diffusion models in the field of visual understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The formulation of VSGG as a temporal denoising task is highly innovative. The architecture combining unified embedding with conditional diffusion is complete and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons across three evaluation protocols, two constraint settings, and multiple baselines, although evaluated on only a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear description of motivations and methods, though mathematical-heavy paragraphs demand close attention.
- Value: ⭐⭐⭐⭐⭐ Achieves the first online VSGG with SOTA performance, majorly driving the progress of real-time visual scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](../../NeurIPS2025/computational_biology/graph_diffusion_that_can_insert_and_delete.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](../../ICML2025/computational_biology/supercharging_graph_transformers_with_advective_diffusion.md)
- [\[ICML 2025\] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](../../ICML2025/computational_biology/wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [\[ICML 2025\] Kinetic Langevin Diffusion for Crystalline Materials Generation](../../ICML2025/computational_biology/kinetic_langevin_diffusion_for_crystalline_materials_generation.md)

</div>

<!-- RELATED:END -->
