---
title: >-
  [Paper Note] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?
description: >-
  [CVPR 2026][Interpretability][feature attribution stability] This paper proposes the FASS benchmark, which systematically evaluates the stability of post-hoc feature attribution methods through prediction-invariant filte…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "feature attribution stability"
  - "post-hoc explanation methods"
  - "prediction invariance"
  - "perturbation robustness"
  - "XAI benchmark"
date: 2026-05-08
content_hash: af36713d9b73bc4d
---

# Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?

**Conference**: CVPR 2026  
**arXiv**: [2604.02532](https://arxiv.org/abs/2604.02532)  
**Code**: [GitHub](https://github.com/KamalasankariSubramaniakuppusamy/xai-stability-benchmark)  
**Area**: Explainable AI / Model Compression  
**Keywords**: feature attribution stability, post-hoc explanation methods, prediction invariance, perturbation robustness, XAI benchmark

## TL;DR

This paper proposes the FASS benchmark, which systematically evaluates the stability of post-hoc feature attribution methods through prediction-invariant filtering, a three-axis stability decomposition (spatial / ranking / salient region), and multiple perturbation types (geometric / photometric / compression), exposing fundamental flaws in existing evaluation frameworks.

## Background & Motivation

Post-hoc feature attribution methods (e.g., Grad-CAM, LIME, SHAP, Integrated Gradients) are widely used in safety-critical visual systems to help practitioners understand model decisions. However, when inputs undergo minor perturbations that do not change model predictions, attribution results may shift dramatically, posing a serious threat to their reliability.

Existing stability evaluations suffer from three structural deficiencies:

**Lack of prediction-invariant filtering**: Stability is computed without verifying whether the perturbation changes the model's predicted class. Metrics such as Lipschitz continuity and max-sensitivity still compare attributions across perturbations that alter the prediction class, conflating "model sensitivity" with "explanation fragility."

**Single scalar metrics**: Stability is collapsed into a single value, making it impossible to distinguish between different failure modes such as spatial displacement, ranking changes, or salient-region inconsistency.

**Evaluation limited to additive noise**: Existing frameworks test primarily within an $\varepsilon$-ball of additive noise, neglecting geometric transformations, photometric changes, and compression artifacts that are common in real-world systems.

These deficiencies cause existing evaluations to **systematically overestimate** the stability of attribution methods.

## Method

### Overall Architecture

FASS (Feature Attribution Stability Suite) is a modular evaluation pipeline consisting of three stages:
- Perturbation application → Prediction-invariant filtering → Three-axis stability measurement

Each input image is paired with its perturbed counterpart, and stability metrics are computed only when the model's top-1 predicted class remains unchanged.

### Key Designs

1. **Prediction-Invariant Filtering**: For each input–perturbation pair, the pipeline checks whether the model's argmax prediction is consistent. Inconsistent pairs are excluded and separately reported as a retention-rate diagnostic. The core insight is that comparing attributions across different prediction classes is meaningless, since attributions are defined relative to a specific prediction. The retention rate itself becomes a first-class experimental quantity, revealing when stability evaluation becomes unreliable.

2. **Three-Axis Stability Decomposition**:

    - **SSIM (Structural Similarity Index)**: Measures spatial consistency of attribution maps using an 11×11 mean-pooling window to detect pixel-level spatial shifts.
    - **Spearman Rank Correlation**: Measures whether the feature importance ranking is preserved after perturbation, independent of magnitude changes. Attribution maps are flattened and their ranks compared.
    - **Top-$k$ Jaccard Overlap**: With $k=100$, measures the consistency of the top-100 most salient feature locations (0.07% of the $224\times224\times3 = 150{,}528$-dimensional attribution map).

3. **Composite FASS Score**: An equal-weighted average of the three metrics: $\text{FASS} = (S + R + J) / 3$. The equal-weight design treats all three failure modes as equally important.

4. **Perturbation Taxonomy**:

    - Geometric: 15° rotation, 20-pixel horizontal translation (zero-padded borders).
    - Photometric: brightness scaling $\times 1.5$, Gaussian noise $\sigma=0.15$.
    - Compression: JPEG quality factor 40.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. Evaluation is conducted on pretrained models (ResNet-50, DenseNet-121, ConvNeXt-Tiny, ViT-B/16) with four attribution methods (IG, GradientSHAP, Grad-CAM, LIME) implemented via the Captum library.

## Key Experimental Results

### Main Results

Evaluation scale: approximately 70,000 images × 5 perturbation types × 4 models × 4 attribution methods ≈ 6.4 million attribution computations.

| Dataset | Method | SSIM | Spearman | Jaccard | FASS |
|--------|------|------|----------|---------|------|
| ImageNet | Grad-CAM | .885 | .966 | .314 | **.722** |
| ImageNet | IG | .706 | .603 | .060 | .457 |
| ImageNet | GradientSHAP | .681 | .570 | .037 | .429 |
| ImageNet | LIME | .342 | .582 | .072 | .332 |
| CIFAR-10 | Grad-CAM | .830 | .899 | .423 | **.717** |
| COCO | Grad-CAM | .810 | .881 | .321 | **.671** |

### Ablation Study — Prediction-Invariant Retention Rate

| Perturbation Type | Mean Retention Rate | Range |
|----------|-----------|------|
| Rotation | 30.9% | 0.0–88.1% |
| Translation | 0.1% | 0.0–0.6% |
| Brightness | 0.8% | 0.0–9.0% |
| Noise | 34.5% | 0.0–94.4% |
| JPEG | 1.0% | 0.0–11.7% |

Without prediction-invariant filtering, up to **99%** of evaluated pairs involve a prediction change — demonstrating that evaluating stability without filtering is severely problematic.

### Ablation Study — Effect of Perturbation Type

| Perturbation Category | SSIM | Spearman | Jaccard | FASS |
|----------|------|----------|---------|------|
| Geometric | .725 | .666 | .099 | .497 |
| Photometric | .770 | .724 | .178 | .557 |
| Compression | .791 | .739 | .196 | .576 |

### Key Findings

1. **Grad-CAM is the most stable attribution method**: It achieves the highest FASS across all 12 dataset–architecture combinations. The low-pass filtering characteristic of its 7×7 activation maps naturally absorbs local perturbation effects.
2. **IG and GradientSHAP are highly consistent**: Their FASS gap does not exceed 0.05, indicating that SHAP's Shapley value sampling does not significantly degrade the stability of gradient signals.
3. **Choice of attribution method matters more than model architecture**: The FASS gap between Grad-CAM and IG (0.21) is approximately twice the largest gap observed for the same method across different architectures (0.09).
4. **Geometric perturbations expose far greater instability than photometric perturbations**: Benchmarks relying solely on additive noise systematically overestimate attribution robustness.
5. **LIME exhibits a distinctive SSIM–Spearman dissociation**: It is spatially unstable yet relatively rank-consistent — a structural failure mode that a single scalar score cannot capture.

## Highlights & Insights

- **Prediction invariance as a prerequisite** is a simple yet critically important design principle. Prior work has entirely overlooked this, rendering stability evaluation results unreliable.
- The three-axis decomposition is an elegant design: behind an identical FASS score, different methods exhibit completely different failure modes. For example, Grad-CAM achieves near-perfect Spearman correlation but relatively low Jaccard overlap, while LIME shows very low SSIM but acceptable Spearman.
- The idea of treating the retention rate as a **first-class experimental quantity** is highly instructive — it serves not only as a diagnostic indicator but also defines the boundary conditions under which stability evaluation itself is reliable.

## Limitations & Future Work

1. FASS measures stability rather than faithfulness: a method that consistently produces incorrect but stable attributions would receive a high score. Jointly evaluating stability and faithfulness is a natural next step.
2. Each perturbation type is evaluated at a single intensity level; intensity sweeps may reveal nonlinear degradation behavior.
3. The equal-weight composite score lacks domain adaptability; specific application scenarios may require different weightings.
4. Only four attribution methods are covered; variants such as SmoothGrad and LRP, as well as concept-level methods, are not included.
5. Pretrained models are used without dataset-specific fine-tuning, which affects retention rates (particularly for CIFAR-10 due to 32→224 upsampling).

## Related Work & Insights

- This paper stands in sharp contrast to evaluation frameworks such as Quantus, LATEC, and OpenXAI — these either do not enforce prediction invariance, reduce stability to a single scalar, or test only additive noise.
- For real-world deployment scenarios (e.g., medical imaging, autonomous driving), the findings imply: (1) preferring Grad-CAM over pixel-level methods; and (2) testing attribution stability under target deployment conditions, including geometric transformations.
- This work can inspire further research on "evaluating the evaluators" — any stability metric must verify that its underlying assumptions are satisfied.

## Rating

- Novelty: ⭐⭐⭐⭐ (Evaluation framework design is novel, though the core ideas are not overly complex)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (70K images, 6.4M attribution computations, three datasets × four architectures × four methods)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, precise problem formulation)
- Value: ⭐⭐⭐⭐ (Directly actionable guidance for the XAI community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)

</div>

<!-- RELATED:END -->
