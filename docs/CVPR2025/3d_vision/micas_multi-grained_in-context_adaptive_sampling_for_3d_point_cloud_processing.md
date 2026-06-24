---
title: >-
  [Paper Note] MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing
description: >-
  [CVPR 2025][3D Vision][Point Cloud Processing] To address inter-task and intra-task sampling sensitivity in 3D point cloud in-context learning, MICAS proposes a multi-grained adaptive sampling mechanism consisting of task-adaptive point sampling (Gumbel-softmax differentiable sampling) and query-specific prompt sampling (probability ranking-based optimal prompt selection), boosting part segmentation by 4.1% on the ShapeNet benchmark.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Processing"
  - "In-Context Learning"
  - "Adaptive Sampling"
  - "Gumbel-softmax"
  - "Multi-task"
date: 2026-05-08
content_hash: 3dfdf62908028ef4
---

# MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing

**Conference**: CVPR 2025  
**arXiv**: [2411.16773](https://arxiv.org/abs/2411.16773)  
**Code**: None  
**Area**: 3D Vision / Point Cloud Processing  
**Keywords**: Point Cloud Processing, In-Context Learning, Adaptive Sampling, Gumbel-softmax, Multi-task

## TL;DR
To address inter-task and intra-task sampling sensitivity in 3D point cloud in-context learning, MICAS proposes a multi-grained adaptive sampling mechanism consisting of task-adaptive point sampling (Gumbel-softmax differentiable sampling) and query-specific prompt sampling (probability ranking-based optimal prompt selection), boosting part segmentation by 4.1% on the ShapeNet benchmark.

## Background & Motivation

1. **Background**: Deep learning has driven multiple tasks in 3D point cloud processing (segmentation, registration, reconstruction, denoising), but typically each task requires a separate model. In-context learning (ICL) allows a single model to handle multiple tasks via prompt exemplars, and methods like PIC have introduced ICL to point cloud processing.

2. **Limitations of Prior Work**: Existing point cloud ICL methods suffer from two key sampling issues. (a) **Inter-task sensitivity**: Task-agnostic sampling like FPS (farthest point sampling) performs very differently across different tasks—for example, in denoising tasks, FPS tends to select noise points as centroids. (b) **Intra-task sensitivity**: For the same task, different selections of prompts can lead to radically different sampling results and unstable experimental outcomes.

3. **Key Challenge**: Statistical methods (FPS, random sampling) ignore point cloud and task information; existing learnable sampling methods focus on adaptation between different point clouds for the *same* task, rather than adaptation of the *same* point cloud across *different* tasks.

4. **Goal**: How to achieve adaptive sampling in the ICL framework so that it can (1) adjust sampling strategies at the point level based on task characteristics, and (2) select the most effective prompt based on the query.

5. **Key Insight**: Extracting task information from prompts to guide sampling (point level), and training a prompt selector using feedback signals from the ICL model (prompt level).

6. **Core Idea**: Introducing multi-grained adaptive sampling in the ICL framework—task information guiding point sampling and performance feedback guiding prompt selection—to resolve the sampling sensitivity issue in point cloud ICL.

## Method

### Overall Architecture
MICAS is built on top of the PIC framework. The input consists of a prompt (an input-target point cloud pair) and a query (an input-target point cloud pair). Based on PIC, two key components are replaced: (1) task-adaptive point sampling replaces FPS for centroid selection; (2) query-specific prompt sampling replaces random prompt selection. The two modules are trained in a step-by-step manner: the point sampling module is trained first, and after its weights are frozen, the prompt sampling module is trained.

### Key Designs

1. **Task-adaptive Point Sampling**:

    - **Function**: Adaptively selects centroids based on task characteristics, replacing task-agnostic FPS sampling.
    - **Mechanism**: Two steps—(a) **Prompt Understanding**: Uses the classification branch of PointNet as a task encoder to extract global task features $F_{task}$ from the prompt pair $(X_p, Y_p)$, and the segmentation branch of PointNet as a point encoder to extract individual point features $F_{X_q}$. (b) **Gumbel Sampling**: Concatenates task features and point features into enhanced features $\hat{F} = F_{task} \oplus F_{X_q}$, obtains sampling weights $SW$ through a fully connected layer, converts them into differentiable soft sampling weights $SW_{gs} = \text{softmax}((\log(SW) + g) / \tau)$ using Gumbel-softmax, and finally computes the centroids as $C = SW_{gs}^T \times X_q$.
    - **Design Motivation**: FPS selects noisy outliers in denoising tasks, leading to poor reconstruction quality. The key advantage of Gumbel-softmax is converting discrete sampling into a differentiable operation, facilitating end-to-end gradient optimization. The fusion of task features enables the sampling to perceive specific requirements of different tasks.

2. **Query-specific Prompt Sampling**:

    - **Function**: Automatically selects the most appropriate prompt for each query point cloud, reducing performance fluctuations caused by prompt selection.
    - **Mechanism**: (a) **Pseudo-label Generation**: Generates predictions for each query-prompt combination using the trained ICL model $\Phi_{ICL}$, and calculates performance scores as pseudo-labels $\tilde{y}$ by comparing predictions with ground truth. (b) **Sampling Probability Prediction**: Concatenates the query point cloud and each candidate prompt, feeds them into a PointNet to predict the sampling probability $prob_i$ for each prompt. (c) **List-wise Ranking Loss**: Uses a ranking loss $\mathcal{L}_{listwise\_rank}$ to align predicted probabilities with actual performance rankings, selecting the prompt with the highest probability during inference.
    - **Design Motivation**: ICL is highly sensitive to prompt selection (which has been extensively studied in NLP), but this issue is systematically addressed for the first time in point cloud ICL. Training based on performance ranking is simple and efficient.

3. **Two-step Training Strategy**:

    - **Function**: Reduces the complexity of joint training, allowing both modules to converge stably.
    - **Mechanism**: The task-adaptive point sampling module is trained first (using a sampling loss $\mathcal{L}_{sampling} = \mathcal{L}_{cd}(R_{pred}, G) + \alpha \cdot \mathcal{L}_{cd}(C, X)$). After fixing its parameters, the prompt sampling module is trained (using the list-wise ranking loss).
    - **Design Motivation**: The optimization objectives of the two modules are different—point sampling requires per-prompt learning, whereas prompt sampling requires simultaneous evaluation of multiple prompts. Joint training would increase complexity and lead to optimization entanglement.

### Loss & Training
- Point sampling loss: $\mathcal{L}_{sampling} = \mathcal{L}_{cd}(R_{pred}, G) + \alpha \cdot \mathcal{L}_{cd}(C, X)$, where $\alpha=0.5$, and the second term constrains the sampled points to cover the original point cloud.
- Prompt sampling loss: List-wise ranking loss, aligning the predicted probability ranking with the actual performance ranking.
- Training parameters: Point sampling lr=0.0001, 60 epochs; prompt sampling lr=0.00001, 30 epochs; batch sizes are 72 and 9 respectively.

## Key Experimental Results

### Main Results

**ShapeNet In-Context Dataset**:

| Method | Recon CD↓ | Denoise CD↓ | Regist CD↓ | Part Seg mIOU↑ |
|------|-----------|-------------|------------|----------------|
| PIC-Cat | 4.3 | 5.3 | 14.1 | 79.0 |
| PIC-Sep | 4.7 | 7.6 | 10.3 | 75.0 |
| PIC-S-Cat | 6.9 | 6.5 | 24.1 | 83.8 |
| **PIC-Cat + MICAS** | **4.7** | **4.6** | **9.8** | **87.9** |
| **PIC-Sep + MICAS** | **4.3** | **5.1** | **3.7** | **86.8** |

Part Segmentation mIOU increases by 4.1% (83.8 $\rightarrow$ 87.9), and Registration CD is significantly reduced (PIC-Sep: 10.3 $\rightarrow$ 3.7).

### Ablation Study

| ICL Model | FPS | Point | Prompt | Recon Avg | Denoise Avg | Regist Avg | mIOU |
|-----------|-----|-------|--------|-----------|-------------|------------|------|
| PIC-Cat | ✓ | | | 4.3 | 5.3 | 14.1 | 79.0 |
| PIC-Cat | | ✓ | | ~4.5 | ~4.5 | ~11.5 | ~85 |
| PIC-Cat | | ✓ | ✓ | 4.7 | 4.6 | 9.8 | 87.9 |

### Key Findings
- **Task-adaptive point sampling contributes significantly to denoising and registration**: Denoising task CD drops from 5.3 to 4.6 (as noisy points are no longer selected as centroids), and the improvement in registration is even more pronounced.
- **Prompt sampling yields further consistent improvements**: Adding prompt sampling on top of point sampling boosts segmentation mIOU by another 2-3 percentage points.
- **MICAS is model-agnostic**: Significant improvements are achieved on both PIC-Cat and PIC-Sep variants, demonstrating the universality of the method.
- **FPS is particularly unfriendly to denoising tasks**: FPS tends to select points that are farthest apart, which are often noise points in denoising tasks.
- **Inference overhead is acceptable**: The additional inference time introduced by MICAS is on the millisecond scale.

## Highlights & Insights
- **Precise problem definition**: The inter-task and intra-task sampling sensitivity in point cloud ICL are systematically identified and analyzed for the first time, and the problem definition itself is a major contribution.
- **Differentiable sampling via Gumbel-softmax**: Converting a discrete point selection problem into a continuous, differentiable operation is an elegant solution to the "non-differentiable sampling" problem. This trick can be transferred to any scenario requiring differentiable discrete selection.
- **Training a prompt selector using feedback from the ICL model itself**: Similar to a self-play approach—the quality of the model's predictions in turn guides prompt selection, forming a closed loop. This paradigm can be extended to prompt selection in other ICL contexts.

## Limitations & Future Work
- **Dependency on PointNet as encoder**: PointNet's representative capability is limited; a stronger backbone (such as Point Transformer) could further improve results.
- **Fixed number of candidate prompts (K=8)**: Adaptive candidate numbers or diversity control strategies were not explored.
- **Step-by-step training instead of end-to-end**: Although step-by-step training simplifies optimization, it may miss opportunities for joint optimization of the two modules.
- **Validated only on ShapeNet**: Lack of experiments on real-world scene-level point cloud datasets.
- **Temperature annealing strategy in Gumbel-softmax** requires careful tuning, which affects training stability.
- **Future directions**: Introduce Transformer-based sampling networks; explore stable end-to-end joint training methods; extend the method to more point cloud tasks (e.g., scene-level segmentation).

## Related Work & Insights
- **vs PIC [Fang et al., NeurIPS'23]**: PIC is a pioneering work in point cloud ICL, but it uses fixed FPS sampling and random prompt selection. MICAS adds adaptive sampling onto the PIC framework, bringing comprehensive performance improvements while preserving PIC's unified multi-task capability.
- **vs SampleNet / S-Net**: These methods are pioneers of learnable sampling but focus on sampling optimization within a single task, without considering cross-task adaptation. MICAS is the first to introduce task awareness into sampling.
- **vs UDR [Li et al., NeurIPS]**: UDR proposes multi-task list-wise ranking in NLP ICL to select demonstrations. MICAS's prompt sampling module is inspired by it and successfully transferred to the 3D point cloud domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The problem definition is novel and important, and task-adaptive sampling is a meaningful contribution to point cloud ICL.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study is comprehensive, and model-agnosticism is well-validated, though tested on only one dataset.
- Writing Quality: ⭐⭐⭐⭐ The problem formulation is clear and the diagrams are intuitive and easy to understand.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play improvement solution for point cloud ICL; the Gumbel-softmax sampling trick has broad transferability value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[CVPR 2025\] Identity-preserving Distillation Sampling by Fixed-Point Iterator](identity-preserving_distillation_sampling_by_fixed-point_iterator.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](../../ECCV2024/3d_vision/dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)

</div>

<!-- RELATED:END -->
