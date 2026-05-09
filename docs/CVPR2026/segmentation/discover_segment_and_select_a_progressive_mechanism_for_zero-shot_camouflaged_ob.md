---
title: >-
  [Paper Note] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation
description: >-
  [CVPR 2026][Segmentation][zero-shot segmentation] This paper proposes DSS, a three-stage progressive pipeline (Discover→Segment→Select) that achieves zero-shot, training-free camouflaged object segmentation by: discovering foreground regions via self-supervised visual encoders and Leiden clustering (FOD); generating candidate masks using SAM; and selecting the optimal mask through heuristic scoring combined with iterative pairwise MLLM comparison. The method demonstrates particularly strong performance in multi-instance camouflage scenarios.
tags:
  - CVPR 2026
  - Segmentation
  - zero-shot segmentation
  - camouflaged object detection
  - SAM
  - MLLM
  - training-free pipeline
  - clustering-based localization
date: 2026-05-08
content_hash: 319723303b86c732
---

# DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation

**Conference**: CVPR 2026
**arXiv**: [2602.19944](https://arxiv.org/abs/2602.19944)
**Code**: To be confirmed
**Area**: Zero-shot Camouflaged Object Segmentation
**Keywords**: [zero-shot segmentation, camouflaged object detection, SAM, MLLM, training-free pipeline, clustering-based localization]

## TL;DR
This paper proposes DSS, a three-stage progressive pipeline (Discover→Segment→Select) that achieves zero-shot, training-free camouflaged object segmentation by: discovering foreground regions via self-supervised visual encoders and Leiden clustering (FOD); generating candidate masks using SAM; and selecting the optimal mask through heuristic scoring combined with iterative pairwise MLLM comparison. The method demonstrates particularly strong performance in multi-instance camouflage scenarios.

## Background & Motivation
Camouflaged object segmentation (COS) requires detecting and segmenting concealed targets that blend highly with their backgrounds. Existing zero-shot COS methods predominantly adopt a two-stage paradigm of "MLLM localization → SAM segmentation": a multimodal large language model (MLLM) first generates location prompts (e.g., bounding boxes), which are then fed into SAM for pixel-level segmentation. However, the visual grounding capability of MLLMs degrades severely in camouflaged scenes—where targets and backgrounds share highly similar colors and textures—leading to inaccurate target localization and large bounding box errors. The problem is further compounded in multi-instance scenarios, where MLLMs tend to localize only the most salient target while missing the rest.

## Core Problem
In zero-shot COS, inaccurate MLLM localization constrains SAM segmentation quality, particularly in multi-instance camouflage scenarios where MLLMs cannot reliably discover all targets. A training-free approach is needed that does not rely on MLLM localization, can automatically discover multiple camouflaged objects, and selects the optimal result from a pool of candidate masks.

## Method

### Overall Architecture
DSS is a three-stage pipeline: **Discover**—replaces MLLM localization with clustering over self-supervised visual features to identify camouflaged object regions and generate bounding box prompts; **Segment**—feeds bounding boxes into SAM to generate a candidate mask set; **Select**—iteratively selects the final mask through heuristic scoring and pairwise MLLM comparison. The entire pipeline is zero-shot and training-free, requiring no fine-tuning or annotated data.

### Stage 1: Discover (FOD — Foreground Object Discovery)

1. **Patch-level feature extraction**: A self-supervised pretrained visual encoder (e.g., DINOv2) extracts patch-level feature matrix $X \in \mathbb{R}^{N \times D}$, where $N$ is the number of patches and $D$ is the feature dimension.

2. **Leiden clustering initialization**: Patch features are clustered using the Leiden community detection algorithm to obtain an initial coarse foreground/background partition. Based on graph modularity optimization, Leiden automatically discovers natural cluster structures in feature space without requiring a predefined number of categories.

3. **Part Composition (PC) iterative refinement**: The initial clustering result is iteratively refined. At each iteration, the foreground/background membership probability of each patch is computed as:
$$y_i^{(t)} = \sigma\left(\|x_i - \mu_b\|_2 - \|x_i - \mu_f\|_2\right)$$
where $\mu_f$ and $\mu_b$ are the feature centroid of current foreground and background patches, respectively, and $\sigma$ denotes the sigmoid function. Intuitively, patches closer to the foreground centroid and farther from the background centroid are more likely to be foreground. Iteration continues until the energy function $E$, which measures the overall consistency of the current partition, converges.

4. **Similarity-based Box Generation (SBG)**: Cosine similarity between the foreground centroid and all patches across the image is computed to produce an affinity map. Candidate regions are extracted via thresholding and connected component analysis on the affinity map. **Pearson correlation deduplication** (threshold $\tau = 0.95$) is then applied—regions with correlation above 0.95 are merged as the same target to avoid redundant bounding boxes. The resulting bounding boxes serve as SAM prompts.

### Stage 2: Segment
All bounding box prompts generated by FOD are fed into SAM (Segment Anything Model), which generates a set of candidate masks per bounding box. These are aggregated into the candidate mask set $M_\text{FOD}$. As a general-purpose segmentation foundation model, SAM produces high-quality pixel-level masks from given location prompts.

### Stage 3: Select (SMS — Segment Mask Selection)

1. **Heuristic scoring**: A quality score is computed for each candidate mask $m_i$:
$$s_i = \text{corr}(m_i, \text{sim}_i) + (1 - \text{BC}(m_i))$$
where $\text{corr}(m_i, \text{sim}_i)$ is the Pearson correlation between the mask and the affinity map—measuring whether the mask region is consistent with the foreground feature distribution—and $\text{BC}(m_i)$ is the boundary complexity of the mask, penalizing overly fragmented masks. A high-quality mask should simultaneously exhibit high feature consistency and low boundary complexity.

2. **Top-K filtering**: Candidates are ranked by score, and the top-$K$ masks are retained for the selection stage.

3. **Iterative pairwise MLLM comparison**: Starting from the lowest-scoring mask, pairs of masks are submitted to the MLLM for pairwise comparison—"Which mask better segments the camouflaged object?" MLLM judgments in pairwise comparisons are substantially more reliable than direct localization (the task is simpler). Comparisons proceed iteratively from low-scoring to high-scoring candidates, and the final winner is taken as output. This bottom-up comparison order allows the MLLM to progressively understand what constitutes a better mask, reducing accumulated errors from individual judgments.

### Loss & Training
The pipeline is entirely training-free and inference-only. Hyperparameters include: the resolution parameter for Leiden clustering, the convergence threshold for PC iteration, the Pearson deduplication threshold $\tau = 0.95$, and the Top-$K$ value for heuristic scoring.

## Key Experimental Results

| Benchmark | Metric | DSS | Prev. SOTA (ZS methods) | Gain |
|-----------|--------|-----|-------------------------|------|
| CHAMELEON | $S_m$↑ | Significantly superior | MLLM+SAM baseline | Large |
| CAMO | $S_m$↑ | Significantly superior | MLLM+SAM baseline | Large |
| COD10K | $S_m$↑ | Significantly superior | MLLM+SAM baseline | Large |
| NC4K | $S_m$↑ | Significantly superior | MLLM+SAM baseline | Large |

- The performance advantage is greatest in multi-instance camouflage scenarios, as FOD can automatically discover multiple target regions while MLLM baselines typically localize only a single target.
- DSS achieves state-of-the-art performance among zero-shot COS methods on all benchmarks.
- Training-free; requires no COS annotation data.

### Ablation Study
- FOD vs. MLLM localization: FOD discovers substantially more targets than MLLMs in multi-instance scenarios.
- PC iterative refinement contributes significantly; removing PC noticeably degrades bounding box quality.
- Pearson deduplication ($\tau = 0.95$) in SBG effectively reduces redundant bounding boxes.
- MLLM pairwise comparison in SMS outperforms using heuristic scores alone as the final selection criterion.
- Bottom-up comparison order (low-to-high score) outperforms random ordering.

## Highlights & Insights
- The paper cleverly repositions the MLLM from a "localizer" to a "judge"—self-supervised visual features and clustering handle localization (more reliably), while the MLLM performs only pairwise comparison (a task it excels at), resulting in a well-suited task allocation.
- The PC iterative refinement formula is concise and interpretable: the sigmoid of the foreground/background distance difference directly yields a membership probability.
- Pearson correlation deduplication is a lightweight yet effective approach to redundancy removal.
- The three-stage progressive design is hierarchically clear, with each stage having a well-defined and independently evaluable objective.
- The zero-shot, training-free setting confers strong generalization capability and deployment flexibility.

## Limitations & Future Work
- The pipeline relies on two large models (SAM and MLLM), incurring non-trivial inference overhead, particularly due to multiple MLLM calls in the SMS stage.
- Leiden clustering and PC refinement assume foreground and background are separable in feature space, which may fail under extreme camouflage with near-zero feature contrast.
- The Top-$K$ setting and number of iterations in pairwise MLLM comparison involve an efficiency–quality trade-off that requires tuning.
- The impact of different self-supervised backbones (e.g., MAE, CLIP) on FOD performance is not explored.
- The Pearson deduplication threshold $\tau = 0.95$ is fixed and not adaptively optimized.

## Related Work & Insights
- **vs. GenSAM/LAKE-RED and other MLLM+SAM methods**: These methods rely on MLLM-generated prompts to guide SAM, making MLLM localization accuracy the bottleneck in camouflaged scenes. DSS replaces MLLM localization with FOD, addressing localization failure at its source.
- **vs. fully supervised COS methods (e.g., SINet)**: Fully supervised methods depend on large amounts of pixel-level annotations and exhibit limited cross-domain generalization. Although DSS as a zero-shot method may not surpass fully supervised SOTA in accuracy, it offers clear advantages in generalizability and annotation cost.
- **vs. general zero-shot segmentation methods (e.g., Matcher)**: General methods are not optimized for camouflaged scenes and perform poorly when foreground–background similarity is very high. The FOD module in DSS is specifically designed for camouflage scenarios, leveraging fine-grained patch-level feature clustering to overcome visual similarity.

## Highlights & Insights (Extended)
- **Idea**: The FOD clustering-and-iterative-refinement paradigm is transferable to other "difficult localization" scenarios, such as underwater object detection and nighttime target discovery.
- **Idea**: The SMS heuristic scoring + MLLM pairwise comparison framework can serve as a general-purpose "mask quality selector" embedded in the post-processing stage of other segmentation pipelines.
- **Idea**: The paradigm of repositioning the MLLM as a judge rather than a localizer is broadly applicable—large models can be similarly employed in their preferred role of comparison and judgment across other vision tasks.
- DSS is complementary to unsupervised COS approaches such as EReCu: EReCu requires training but operates in a fully unsupervised setting, whereas DSS is entirely training-free but depends on SAM and an MLLM.
- The energy convergence mechanism in PC can be combined with active inference to enable adaptive control over the depth of foreground discovery.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-stage pipeline design is novel, and the insight of repositioning the MLLM's role is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple COS benchmarks, ablation studies, and multi-instance analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The three-stage naming is intuitive and the motivation is articulated clearly.
- **Value**: ⭐⭐⭐⭐ The zero-shot pipeline design paradigm is highly referenceable; the FOD and SMS modules are modularly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](dss_discover_segment_select_zero_shot_cos.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)

</div>

<!-- RELATED:END -->
