---
title: >-
  [Paper Note] Background Invariance Testing According to Semantic Proximity
description: >-
  [ICCV 2025][Multimodal VLM][background invariance testing] This paper proposes a background invariance testing method based on semantic proximity. It constructs a keyword ontology via association analysis to systematical…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "background invariance testing"
  - "association ontology"
  - "semantic distance"
  - "visualization-based testing"
  - "neuron coverage"
date: 2026-05-08
content_hash: 9e68b8105b694661
---

# Background Invariance Testing According to Semantic Proximity

**Conference**: ICCV 2025
**arXiv**: [2208.09286](https://arxiv.org/abs/2208.09286)  
**Code**: [https://github.com/Zukang-Liao/background_invariance_testing](https://github.com/Zukang-Liao/background_invariance_testing)  
**Area**: Multimodal VLM / AI Safety
**Keywords**: background invariance testing, association ontology, semantic distance, visualization-based testing, neuron coverage

## TL;DR

This paper proposes a background invariance testing method based on semantic proximity. It constructs a keyword ontology via association analysis to systematically sample background scenes, achieving an optimal balance between test diversity (recall) and consistency with human judgment (precision). The work further demonstrates that visualization-based testing frameworks are more informative than global statistical metrics.

## Background & Motivation

ML models deployed in the real world are expected to satisfy various invariance properties (rotation, scale, brightness, etc.), among which background invariance testing is particularly challenging due to the vast data space. Existing methods typically report a single average worst-case accuracy; however, the authors draw an analogy to Anscombe's Quartet to argue that models with identical summary statistics can behave in fundamentally different ways, making visualization-based testing frameworks substantially more informative.

Nevertheless, visualization-based testing relies on human judgment and faces a core dilemma:
- **Random sampling**: yields diverse test sets (high neuron coverage) but produces inconsistent visualization patterns across runs, undermining reliable human judgment.
- **Nearest-neighbor sampling**: achieves high human judgment consistency but results in severely limited test set diversity.

The authors aim to identify a sampling strategy that achieves the best balance between test diversity and judgment consistency.

## Method

### Overall Architecture

The method consists of four stages: (1) extracting keywords from target images; (2) expanding keywords via an association ontology; (3) sampling background scenes based on keywords and synthesizing test images; and (4) analyzing test results through visualization and human or automated judgment.

### Key Designs

1. **Keyword Detection and Ontology Construction**:

    - A pre-trained scene understanding model is used to extract keyword vectors from each image (150 ADE20k object classes + 365 Places365 scene classes = 515 dimensions).
    - Association rule mining algorithms (Apriori/FP-Growth) are applied to compute co-occurrence relationships among keywords.
    - A directed weighted ontology graph is constructed, where nodes are keywords and edge weights are confidence values: $\text{confidence}(\exists s_a \rightarrow \exists s_b) = \frac{\text{support}(s_a \cup s_b)}{\text{support}(s_a)}$
    - Confidence is preferred over support as the edge weight, since support is more sensitive to dataset size and keyword count.

2. **Keyword Expansion**:

    - The original keyword set is iteratively expanded via multi-hop search over the ontology.
    - The keyword set after the $i$-th expansion is: $\text{OL}_x[i] = K_x \cup (\bigcup_{j=1}^{i} E_{i,x})$
    - This addresses the problem of insufficient background scene retrieval caused by too few original keywords (statistical analysis shows most images yield only 2–3 detected keywords).
    - The expansion effect saturates after 4 iterations.

3. **Background Scene Sampling and Test Image Synthesis**:

    - Background scenes are retrieved based on the expanded keyword set, with each keyword corresponding to a subspace.
    - When sampling from each subspace, DreamSim is run to select the first background whose synthesized image exceeds a realism score threshold.
    - Synthesis is performed via simple background replacement: $\mathbf{y}_{i,j} = \text{Mask}_i \circledast x_i + (1-\text{Mask}_i) \circledast \mathbf{b}_j$, supplemented by Laplacian pyramid blending.
    - Generative models (e.g., Blended Latent Diffusion) are explicitly excluded, as they may introduce uncontrolled foreground bias (e.g., fish appearing in an ocean background).

4. **Test Result Analysis**:

    - **Diversity evaluation**: Neuron coverage (percentage of activated neurons) is used to measure the comprehensiveness of the test set.
    - **Visualization**: Model responses at different signal locations (final prediction confidence, top-k neurons in the final pooled embedding) are measured, and 2D scatter plots are constructed using the semantic distance between test images and original images.
    - RBF interpolation is applied to convert scatter plots into variance matrices, facilitating pattern recognition.
    - **Human judgment**: Three ML practitioners annotate models as "invariant / borderline / non-invariant" based on the visualizations.
    - **Automation**: A random forest classifier performs automated judgment from hand-crafted features of the variance matrices.

### Loss & Training

This paper presents a testing framework rather than a training method. The evaluation metrics are:
- **Recall** (diversity): neuron coverage
- **Precision** (consistency): Fleiss' Kappa inter-rater reliability
- **F1 Score**: a composite metric balancing diversity and consistency

## Key Experimental Results

### Main Results — Comparison of Testing Methods

| Method | Neuron Coverage (recall) | Fleiss' Reliability (precision) | F1 Score |
|--------|--------------------------|----------------------------------|----------|
| Random Sampling | **0.681** | 0.384 | 0.491 |
| Nearest-Neighbor Top-K | 0.133 | **0.906** | 0.232 |
| Distance Interval Sampling | 0.667 | 0.531 | 0.591 |
| CLIP Keyword Sampling | 0.591 | 0.640 | 0.615 |
| **Ontology Keyword Sampling (Ours)** | 0.652 | 0.649 | **0.650** |

The proposed ontology-based method achieves the best F1 score, realizing the optimal balance between diversity and consistency.

### Ablation Study — Automated Testing

| Classifier | Automated Accuracy | IRR Score |
|------------|--------------------|-----------|
| **Random Forest** | **79.7 ± 7.5%** | **0.649 ± 0.091** |
| AdaBoost | 74.8 ± 9.1% | 0.599 ± 0.102 |
| Worst-case accuracy | 64.4% | 0.387 |

The IRR score of automated judgment is comparable to that among human annotators (~0.65).

### Key Findings

- **Visualization > Statistics**: Four models with identical accuracy and worst-case accuracy exhibit entirely different visualization patterns, leading annotators to render distinct judgments (e.g., model $M_a$ may rely on background cues, $M_c$ may exhibit data leakage, and $M_b$ may be sensitive to specific objects).
- Random sampling yields the highest diversity but the lowest consistency; nearest-neighbor sampling yields the highest consistency but the lowest diversity — a fundamental trade-off exists.
- CLIP tends to retrieve semantically similar matches, resulting in insufficient test set diversity.
- Varying the number of test images $N$ from 32 to 100 has limited impact on results.
- Different RBF interpolation parameters have limited effect on automated accuracy.
- The model repository comprises 250 models trained on IN9 (6 architectures × combinations of hyperparameters, augmentations, optimizers, and loss functions).

## Highlights & Insights

- Elevating invariance testing from "reporting a single accuracy value" to "multi-factor decision-making based on visualization patterns" represents a meaningful and impactful shift in perspective.
- Constructing an association ontology introduces data mining techniques (Apriori algorithm) into ML model testing, representing an elegant cross-disciplinary integration.
- The paper explicitly argues against using generative models for test image synthesis due to the risk of introducing uncontrolled bias, and the choice of simple replacement is methodologically rigorous.
- The success of automated testing (~80% accuracy, IRR comparable to human annotators) renders the framework practically deployable.

## Limitations & Future Work

- The model repository (250 models) is relatively small; validation at larger scale remains to be conducted.
- Ontology quality is constrained by the detection capabilities of the scene understanding model and the size of the background database.
- Semantic distance measurement after keyword expansion is coarse (based on hop count or aggregated weights).
- Validation is limited to IN9 (9 classes); scalability to large-scale classification tasks (e.g., full ImageNet) is unknown.
- Human annotation involves only three ML practitioners; statistical confidence could be further improved.
- Simple background replacement may produce unnatural visual artifacts.

## Related Work & Insights

- This work is complementary to Background Challenge and Rosenfeld et al.: prior work focuses primarily on simple transformations (noise/solid colors), whereas this paper systematically handles real background scenes.
- The use of neuron coverage as a test set diversity metric is inspired by DeepXplore and the software testing literature.
- The idea of building semantic relational graphs via association analysis is extensible to other ML testing scenarios requiring "meaningful data sampling."
- The visualization-based testing framework has practical value for model auditing and trustworthy AI deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of incorporating association analysis into background invariance testing is novel, with a distinctive problem framing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-method comparison, ablation analysis, automated validation, and inter-rater reliability assessment.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with well-motivated problem framing.
- Value: ⭐⭐⭐ Applicable to a relatively narrow setting (model invariance testing), but makes a genuine contribution within that domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChartMuseum: Testing Chart Visual Reasoning in Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[ICCV 2025\] OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography](oraclefusion_assisting_the_decipherment_of_oracle_bone_script_with_structurally_.md)
- [\[AAAI 2026\] Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference](../../AAAI2026/multimodal_vlm/tri-bench_stress-testing_vlm_reliability_on_spatial_reasoning_under_camera_tilt_.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)

</div>

<!-- RELATED:END -->
