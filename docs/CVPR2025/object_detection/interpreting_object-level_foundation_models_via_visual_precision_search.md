---
title: >-
  [Paper Note] Interpreting Object-level Foundation Models via Visual Precision Search
description: >-
  [CVPR 2025][Object Detection][Model Interpretability] Addressing the interpretability of object-level foundation models like Grounding DINO and Florence-2, this paper proposes Visual Precision Search (VPS). By combining superpixel sparsification with greedy search guided by submodular functions, VPS accurately localizes critical decision subregions. It outperforms the state-of-the-art D-RISE method in fidelity metrics (Insertion) by 23.7%, 20.1%, and 31.6% on MS COCO, RefCOCO…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Model Interpretability"
  - "Visual Precision Search"
  - "Submodular Function"
  - "Attribution Analysis"
date: 2026-05-08
content_hash: d9db23bd0456a9fa
---

# Interpreting Object-level Foundation Models via Visual Precision Search

**Conference**: CVPR 2025  
**arXiv**: [2411.16198](https://arxiv.org/abs/2411.16198)  
**Code**: [https://github.com/RuoyuChen10/VPS](https://github.com/RuoyuChen10/VPS)  
**Area**: Object Detection  
**Keywords**: Model Interpretability, Object Detection, Visual Precision Search, Submodular Function, Attribution Analysis

## TL;DR
Addressing the interpretability of object-level foundation models like Grounding DINO and Florence-2, this paper proposes Visual Precision Search (VPS). By combining superpixel sparsification with greedy search guided by submodular functions, VPS accurately localizes critical decision subregions. It outperforms the state-of-the-art D-RISE method in fidelity metrics (Insertion) by 23.7%, 20.1%, and 31.6% on MS COCO, RefCOCO, and LVIS, respectively.

## Background & Motivation

1. **Background**: With the development of multimodal pre-training, object-level foundation models such as Grounding DINO and Florence-2 exhibit outstanding performance in visual grounding and object detection. However, these models have massive parameters and rich semantics, sharply decreasing their transparency and interpretability.

2. **Limitations of Prior Work**:
    - **Gradient-based methods (e.g., ODAM)**: Since visual and textual features are deeply fused in foundation models, gradient backpropagation is heavily influenced by both modalities, failing to accurately locate critical vision-level regions and resulting in diffuse attribution maps.
    - **Perturbation-based methods (e.g., D-RISE)**: Generating saliency maps via random sampling introduces significant sampling noise, leading to coarse attribution maps and insufficient fine-grained interpretability.
    - Existing methods cannot effectively explain why models fail to detect objects.

3. **Key Challenge**: The vision-language fusion architecture of foundation models invalidates traditional gradient attribution, while the random sampling strategy of perturbation-based methods is too coarse.

4. **Goal**: Design an attribution method that does not rely on internal model parameters, can precisely locate a small number of key decision regions, and is applicable to object-level foundation models of different architectures.

5. **Key Insight**: Formulate the attribution problem as a subset selection problem of subregions, leveraging submodular optimization with theoretical guarantees for greedy search.

6. **Core Idea**: Sparsify inputs into a set of subregions using superpixels, design a submodular scoring function (clue score + collaboration score), and accurately sort key regions via greedy search.

## Method

The core Mechanism of VPS is to partition the input image into a small number of meaningful subregions without inspecting internal model parameters (a black-box approach), and then utilize submodular optimization to find the critical set of regions that "enable the detector to work correctly with the minimum number of regions."

### Overall Architecture

Input: Image $\mathbf{I}$, target bounding box $\boldsymbol{b}_{target}$, target class $c$.
Workflow:
1. SLICO superpixel segmentation partitions the image into $m$ subregions $V = \{\mathbf{I}_1^s, ..., \mathbf{I}_m^s\}$
2. Define a submodular function $\mathcal{F}(S)$ to evaluate the interpretability of subregion sets
3. Iteratively perform greedy search to select subregions that maximize $\mathcal{F}$
4. Score each subregion based on its marginal contribution to generate the final saliency map

### Key Designs

1. **Clue Score**:
    - **Function**: Evaluates whether a given subset of subregions allows the model to correctly localize and identify the target.
    - **Mechanism**: For a set of subregions $S$, calculate the maximum product of IoU with the target box and the class confidence score in the model output: $s_{clue}(S, \boldsymbol{b}_{target}, c) = \max_{(\boldsymbol{b}_i, s_{c,i}) \in f(S)} \text{IoU}(\boldsymbol{b}_{target}, \boldsymbol{b}_i) \cdot s_{c,i}$. Unlike D-RISE, this method considers all candidate boxes instead of just high-confidence ones to prevent the search from getting stuck in local optima.
    - **Design Motivation**: A good explanation should enable the model to make correct detections with as few input regions as possible.

2. **Collaboration Score**:
    - **Function**: Evaluates the joint effect of subregions—some regions only contribute to the decision when combined with other specific regions.
    - **Mechanism**: Compute the degradation in model detection performance when the subregion set $S$ is removed: $s_{colla.}(S, \boldsymbol{b}_{target}, c) = 1 - \max_{(\boldsymbol{b}_i, s_{c,i}) \in f(V \setminus S)} \text{IoU}(\boldsymbol{b}_{target}, \boldsymbol{b}_i) \cdot s_{c,i}$. If no bounding boxes can accurately localize the target after removing critical regions, these regions receive a high collaboration score.
    - **Design Motivation**: In the early search stages, testing some subtle but critical regions (like auxiliary contextual clues) in isolation might yield unnoticeable performance, but removing them results in detection failure. The collaboration score effectively captures these regions.

3. **Submodular Function and Greedy Search**:
    - **Function**: Combines the two scores into an optimization objective with theoretical guarantees.
    - **Mechanism**: The submodular function is defined as $\mathcal{F}(S) = s_{clue}(S) + s_{colla.}(S)$. The paper proves that this function satisfies the properties of diminishing marginal returns and monotonic non-negativity (submodularity) under reasonable assumptions, ensuring that the greedy search guarantees a $(1-1/e)$ approximation ratio to the optimal solution. After searching, the marginal contribution difference is used to score each subregion: $\mathcal{A}_i = \mathcal{A}_{i-1} - |\mathcal{F}(S_{[i]}) - \mathcal{F}(S_{[i-1]})|$, and normalized to generate the saliency map.
    - **Design Motivation**: Submodular optimization provides theoretical optimality guarantees, which is more principled than random sampling in D-RISE and SHAP estimation in D-HSIC.

### Loss & Training

VPS is a training-free, inference-time method. The default number of superpixels is set to 100. Each evaluated subregion set requires one forward pass of the model.

## Key Experimental Results

### Main Results

**Grounding DINO Fidelity Evaluation (Correctly Detected Samples)**

| Dataset | Method | Insertion↑ | Deletion↓ | Avg. Highest Score↑ |
|--------|------|-----------|-----------|---------------------|
| MS COCO | D-RISE | 0.4412 | 0.0402 | 0.6215 |
| MS COCO | **VPS(Ours)** | **0.5459** (+23.7%) | **0.0375** | **0.6873** |
| RefCOCO | D-RISE | 0.6178 | 0.1605 | 0.8471 |
| RefCOCO | **VPS(Ours)** | **0.7419** (+20.1%) | **0.1250** | **0.8842** |
| LVIS(rare) | D-RISE | 0.2808 | 0.0289 | 0.4289 |
| LVIS(rare) | **VPS(Ours)** | **0.3695** (+31.6%) | **0.0277** | **0.4969** |

**Florence-2 Fidelity Evaluation**

| Dataset | Method | Insertion↑ | Deletion↓ |
|--------|------|-----------|-----------|
| MS COCO | D-RISE | 0.7477 | 0.0972 |
| MS COCO | **VPS(Ours)** | **0.7761** | **0.0479** (-50.7%) |
| RefCOCO | D-RISE | 0.8107 | 0.1275 |
| RefCOCO | **VPS(Ours)** | **0.8604** | **0.0422** (-66.9%) |

### Ablation Study

| Configuration | Key Impact | Explanation |
|------|---------|------|
| Clue Score Only | Significant drop in Insertion | Fails to capture contextual dependencies without collaboration score |
| Collaboration Score Only | Imprecise initial search | Lacks direct detection guidance |
| Full (Clue+Colla.) | Optimal | The two are complementary |
| Number of superpixels | 100 is optimal | Too few → coarse granularity; too many → large search space |

**Failure Case Analysis (Grounding DINO)**

| Task | Method | Insertion↑ | ESR↑ |
|------|------|-----------|------|
| Localization failure | D-RISE | 0.3430 | 39.5% |
| Localization failure | **VPS** | **0.4901** (+42.9%) | **42.5%** |
| Misclassification (COCO) | D-RISE | 0.3021 | — |
| Misclassification (COCO) | **VPS** | **0.4674** (+54.7%) | — |

### Key Findings
- **Gradient methods fail on foundation models**: Grad-CAM/ODAM yields highly diffuse attributions on Grounding DINO, with Insertion metrics less than half of VPS, confirming that vision-language fusion seriously interferes with gradient attribution.
- **VPS is exceptionally powerful for failure analysis**: It not only explains correct detections but also identifies input-level factors leading to misclassifications and missed detections.
- **Complementarity of Clue and Collaboration scores**: The clue score dominates the later search phase (guiding the model to correct detections), while the collaboration score is more critical in the early search phase (quickly locking onto crucial context regions).
- **Most significant deletion improvement on Florence-2**: It achieves 50.7% and 66.9% improvements on Florence-2 because searching has a greater advantage over random sampling when confidence scores are unavailable.

## Highlights & Insights
- **Search instead of sampling**: Replacing random perturbations with theoretically-guaranteed submodular optimization search. The core cleverness is that the objective "making correct detections with fewer regions" naturally fits the diminishing marginal returns of submodular optimization.
- **Black-box universality**: Without accessing internal model parameters, it is compatible with models of entirely different architectures (e.g., multi-stage fusion in Grounding DINO vs. MLLM architecture in Florence-2). This methodology can be transferred to any future architectures.
- **Failure analysis capability**: Beyond explaining correct decisions, VPS can localize input factors causing errors (such as co-occurrence confusion), which holds practical value for safety-critical scenarios like autonomous driving.

## Limitations & Future Work
- Computational overhead of greedy search: In each search round, every candidate subregion requires a forward pass. Thus, $m$ subregions require $O(m^2)$ evaluations, which can be slow for large foundation models.
- Reliance on superpixel segmentation quality: Superpixel segmentation quality directly affects the results; if key regions are segmented improperly, correct attribution becomes impossible.
- Limited explanation precision for MLLMs: MLLMs (like Florence-2) do not directly output confidence scores, forcing the clue score to rely solely on IoU, which limits explanation accuracy.
- Limited task scope: Currently, it is only validated on object detection in static images. Video detection and other vision foundation tasks (e.g., segmentation) have not been explored.
- Assumptions in submodularity proof: The theoretical proof of submodularity relies on the assumption that "regions make positive contributions to detection," which may be violated by anomalous inputs.

## Related Work & Insights
- **vs D-RISE**: D-RISE estimates the pixel-level importance via random mask sampling, leading to noisy attribution maps. VPS accurately localizes a small number of key subregions via structured search, improving the Insertion metric by 20-30%.
- **vs ODAM**: ODAM extends gradient-based methods (Grad-CAM) to object detection. In multimodal fusion models, gradient attribution is corrupted and diffused by text features. VPS, as a black-box method, completely circumvents this issue.
- **vs VX-CODE**: VX-CODE also employs greedy search with SHAP but lacks theoretical guarantees of submodular optimization and the design of a collaboration score.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing submodular optimization to object detection interpretability, with insightful clue + collaboration score design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive with three datasets, two models, correct/failure analyses, and multiple evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Well-balanced theoretical analysis and experimental validation with clear logic.
- Value: ⭐⭐⭐⭐ Provides a universal black-box solution for the interpretability of foundation models, which is highly significant for safety-critical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision](../../CVPR2026/object_detection/online_data_curation_for_object_detection_via_marginal_contributions_to_dataset-.md)
- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)
- [\[ECCV 2024\] Can OOD Object Detectors Learn from Foundation Models?](../../ECCV2024/object_detection/can_ood_object_detectors_learn_from_foundation_models.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)

</div>

<!-- RELATED:END -->
