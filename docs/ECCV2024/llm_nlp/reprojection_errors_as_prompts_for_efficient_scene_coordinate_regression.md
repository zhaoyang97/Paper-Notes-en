---
title: >-
  [Paper Note] Reprojection Errors as Prompts for Efficient Scene Coordinate Regression
description: >-
  [ECCV 2024][LLM (Other)][scene coordinate regression] This paper proposes the Error-Guided Feature Selection (EGFS) mechanism, which leverages low reprojection error regions as point prompts for SAM to expand into semantic masks. By iteratively filtering reliable training samples, the method outperforms existing 3D-free SCR methods on the Cambridge Landmarks and Indoor6 datasets with a smaller model size and less training time.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "scene coordinate regression"
  - "visual grounding"
  - "SAM"
  - "feature selection"
  - "confidence map"
date: 2026-05-08
content_hash: dadb6836ad0a72f3
---

# Reprojection Errors as Prompts for Efficient Scene Coordinate Regression

**Conference**: ECCV 2024  
**arXiv**: [2409.04178](https://arxiv.org/abs/2409.04178)  
**Code**: [https://tingru0203.github.io/egfs](https://tingru0203.github.io/egfs)  
**Area**: LLM/NLP  
**Keywords**: scene coordinate regression, visual grounding, SAM, feature selection, confidence map

## TL;DR

This paper proposes the Error-Guided Feature Selection (EGFS) mechanism, which leverages low reprojection error regions as point prompts for SAM to expand into semantic masks. By iteratively filtering reliable training samples, the method outperforms existing 3D-free SCR methods on the Cambridge Landmarks and Indoor6 datasets with a smaller model size and less training time.

## Background & Motivation

**Background**: Visual localization estimates 6-DoF camera poses from images, serving as a core technology for AR/VR and autonomous driving. Scene Coordinate Regression (SCR) methods directly predict the 3D coordinates corresponding to pixels using a DNN, followed by pose estimation via PnP+RANSAC. This paradigm offers small model sizes and fast training. For instance, the ACE method can be trained within 5 minutes.

**Limitations of Prior Work**: During training, SCR methods uniformly sample all image regions, including dynamic objects (pedestrians, vehicles) and textureless areas (walls, skies). Unstable features in these regions lead to high reprojection errors, resulting in unstable model weight updates.

**Key Challenge**: Although RANSAC can filter outliers during inference, the training phase still wastes computational resources and introduces noisy gradients on these detrimental regions. End-to-end optimization methods attempt to down-weight outliers but do not take the semantic information of the regions into account.

**Key Findings**: Through in-depth analysis, the authors find that low reprojection error regions generally correspond to higher inlier ratios. However, low errors do not always correspond to specific semantic categories (e.g., the error for "tree" varies significantly across different scenes). Thus, one cannot simply use predefined semantic categories to select regions.

**Key Insight**: Instead of relying on predefined semantic categories, the **reprojection error itself** is utilized as a signal to select low-error points as prompts for SAM, which automatically expands them into semantically consistent mask regions.

**Core Idea**: Applying low reprojection error points $\rightarrow$ SAM prompts $\rightarrow$ semantic masks $\rightarrow$ confidence refinement $\rightarrow$ iteratively updating training regions.

## Method

### Overall Architecture

Iterating the training for every $k$ epochs $\rightarrow$ computing reprojection errors $\rightarrow$ selecting the top $\tau$% lowest-error points as SAM prompts $\rightarrow$ expanding them into semantic masks via SAM $\rightarrow$ refining masks with a confidence map $\rightarrow$ sampling features only within the masks to train the scene-specific MLP $\rightarrow$ repeating the iteration. During inference, no masks are required; only the confidence map is used to filter and estimate poses.

### Key Designs

1. **Error-Guided Feature Selection (EGFS)**:

    - **Function**: Selecting reliable training regions based on reprojection errors.
    - **Mechanism**: Selecting the top $\tau$% (default 10%) pixels with the lowest reprojection errors as point prompts and feeding them into EfficientViT-SAM-L0 to expand sparse points into complete semantic masks. The masks are updated every $k=5$ epochs to achieve coarse-to-fine iterative refinement.
    - **Design Motivation**: Sampling points based on semantic regions is superior to direct sampling based on error thresholds. Since low-error points can be scattered and may not cover complete objects, SAM can infer complete semantic boundaries from a few seed points.

2. **Confidence Map Refinement**:

    - **Function**: Further filtering unreliable points within the EGFS masks.
    - **Mechanism**: Adding a confidence prediction head to the end of the scene-specific MLP to output per-pixel confidence $c_i$. The training loss is defined as:
    $$\ell(p_i, y_i, h_i^*) = \begin{cases} c_i \cdot \hat{r}(p_i, y_i, h_i^*) - \alpha \log c_i & \text{if } y_i \in \mathcal{V} \\ \|y_i - \bar{y_i}\|_0 - \alpha \log(1 - c_i) & \text{otherwise} \end{cases}$$
      Within the mask, points with confidence below the median are discarded; the median confidence threshold is also used for filtering during inference.
    - **Design Motivation**: Although EGFS masks select semantically consistent regions, there can still be localized positions (e.g., occlusion boundaries) within the region that are difficult to predict accurately. The confidence map provides finer-grained filtering.

3. **Iterative Training**:

    - **Function**: Dynamically updating training masks at fixed epoch intervals.
    - **Mechanism**: Training for 20 epochs in total, recomputing reprojection errors and updating EGFS masks every 5 epochs. The initial iteration uses random full-image sampling to obtain the first batch of error signals.
    - **Design Motivation**: As training progresses and the model's understanding of the scene deepens, low-error regions dynamically change. Thus, masks need continuous updates rather than being determined statically.

### Loss & Training

- Based on the ACE architecture, the scene-specific MLP uses $1 \times 1$ convolutions to process each sampled point independently.
- Confidence regularization parameter $\alpha = 10$.
- SAM utilizes the lightweight EfficientViT-SAM-L0 to ensure efficiency.
- Model size is only 4.5MB (ACE is 4MB + 0.5MB confidence head).

## Key Experimental Results

### Main Results — Cambridge Landmarks (cm/°)

| Scene | EGFS (4.5MB) | ACE (4MB) | DSAC* (28MB) | ACE quad (16MB) |
|------|-------------|-----------|-------------|----------------|
| King's College | **14/0.3** | 28/0.4 | 18/0.3 | 18/0.3 |
| Great Court | 31/0.1 | 43/0.2 | 34/0.2 | **28/0.1** |
| Old Hospital | **21/0.4** | 31/0.6 | 21/0.4 | 25/0.5 |
| Shop Facade | **5/0.3** | 5/0.3 | 5/0.3 | 5/0.3 |
| St Mary's Church | 15/0.5 | 18/0.6 | 15/0.6 | **9/0.3** |
| **Average** | **17/0.3** | 25/0.4 | 19/0.4 | 17/0.3 |

### Main Results — Indoor6 (5cm/5° Accuracy %)

| Scene | EGFS (4.5MB) | ACE (4MB) | DSAC* (28MB) | ACE quad (16MB) |
|------|-------------|-----------|-------------|----------------|
| scene1 | 46.4% | 26.0% | 23.0% | 52.9% |
| scene2a | **60.6%** | 32.3% | 33.9% | 52.5% |
| scene4a | **78.7%** | 62.0% | 67.1% | 69.6% |
| **Average** | **56.1%** | 35.6% | 35.1% | 58.6% |

### Ablation Study

| Configuration | Cambridge (cm/°) | Indoor6 (%) | Description |
|------|---------|------|------|
| No EGFS + No Confidence | 25/0.4 | 35.6% | ACE Baseline |
| EGFS Mask Only | 19/0.3 | 41.6% | Mask selection alone yields significant improvement |
| Confidence Only | 19/0.4 | 50.9% | Confidence filtering has a larger impact |
| EGFS + Confidence | **17/0.3** | **56.1%** | Complementary, achieving the best performance |

### Key Findings

- **Semantic Mask vs. Pure Error Threshold Sampling**: On Indoor6, sampling purely based on the $Q_{0.5}(R)$ threshold only achieves 44.3%, whereas EGFS mask achieves 56.1%, representing a 26.6% improvement. This demonstrates that semantically consistent region selection is far superior to pixel-wise filtering.
- There is **no stable correspondence** between low reprojection errors and specific semantic categories: the error of the same category varies significantly across different scenes.
- EGFS masks are gradually refined from coarse to fine during training, ultimately focusing on regions most valuable for localization.
- EGFS training takes 12 minutes, which is slightly longer than ACE's 5 minutes but significantly shorter than DSAC*'s 15 hours.

## Highlights & Insights

- **Converting reprojection errors into SAM prompts is a clever approach**: It requires no predefined semantic categories and harnesses the generalization capability of foundation models to automatically discover beneficial regions.
- **The loss design with confidence self-adaptive weighting** can be transferred to other tasks handling noisy labels or unreliable regions.
- **Iterative mask updates** create a curriculum learning effect: the model first learns simpler regions and then gradually expands to more difficult areas.
- Demonstrates the plug-and-play value of **foundation models (like SAM) as utility models** in downstream tasks.

## Limitations & Future Work

- **Evaluated only on the ACE architecture**: Not extended to DSAC* or other SCR backbones.
- **SAM computational overhead**: Although EfficientViT-SAM is lightweight, it still requires generating masks for all training images every 5 epochs.
- **Generalization to large-scale outdoor scenes**: Cambridge scenes are relatively simple; verification on large-scale, city-level scenes is lacking.
- **Single prompt strategy**: Uses only point prompts, leaving box prompts or other interaction modalities unexplored.
- **Scalability to reprojection error guidance for 3D Gaussian/NeRF scene representations**.

## Related Work & Insights

- **vs. ACE**: EGFS adds only 0.5MB and ~7 minutes of training time over ACE, yet improves the accuracy on Indoor6 from 35.6% to 56.1%.
- **vs. DSAC***: DSAC* requires 15 hours of training and a 28MB model, which EGFS surpasses with only 12 minutes of training and a 4.5MB model.
- **vs. FocusTune**: FocusTune focuses on specific regions via weight fine-tuning, whereas EGFS achieves a similar effect more weightlessly through data sampling strategies.
- **vs. SACReg (w/ 3D model)**: Methods using 3D models have an inherent advantage; EGFS closely approaches their performance without using any 3D information.

## Rating

- Novelty: ⭐⭐⭐⭐ Treating reprojection errors as prompts for foundation models is a novel and elegant cross-disciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both indoor and outdoor datasets with rich ablation and qualitative analyses, though lacks validation on larger-scale scenes.
- Writing Quality: ⭐⭐⭐⭐ Detailed motivation analysis (the error-semantic relation analysis in Section 4 is highly convincing) and overall clear logic.
- Value: ⭐⭐⭐⭐ Lightweight, plug-and-play method with transferable ideas, though the application scenario is somewhat narrow (specifically visual localization).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)
- [\[ECCV 2024\] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](../../ACL2025/llm_nlp/llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] P3: Prompts Promote Prompting](../../ACL2025/llm_nlp/p3_prompts_promote_prompting.md)
- [\[ACL 2025\] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](../../ACL2025/llm_nlp/aad-llm_neural_attention-driven_auditory_scene_understanding.md)

</div>

<!-- RELATED:END -->
