---
title: >-
  [Paper Note] Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance
description: >-
  [CVPR 2026][Medical Imaging][Uncertainty Quantification] This paper presents the first systematic study of aggregation strategies for converting pixel-level uncertainty to image-level scores in segmentation. It proposes SMR aggregators incorporating spatial structural information (Moran's I, Edge Density, Shannon Entropy) and a GMM-based meta-aggregator, demonstrating across 10 datasets that global averaging (AVG) is suboptimal and that GMM-All achieves robust performance on both OoD and failure detection.
tags:
  - CVPR 2026
  - Medical Imaging
  - Uncertainty Quantification
  - Spatial Aggregation Strategies
  - OoD Detection
  - Failure Detection
  - Meta-Aggregation
date: 2026-05-08
content_hash: f916a444f6653958
---

# Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance

**Conference**: CVPR 2026
**arXiv**: [2603.29941](https://arxiv.org/abs/2603.29941)
**Code**: [https://github.com/Kainmueller-Lab/aggrigator](https://github.com/Kainmueller-Lab/aggrigator)
**Area**: Medical Imaging
**Keywords**: Uncertainty Quantification, Segmentation Aggregation, OoD Detection, Failure Detection, Spatially-Aware Aggregation

## TL;DR
This paper presents the first systematic study of aggregation strategies for converting pixel-level uncertainty maps to image-level scores in segmentation tasks. It proposes the Spatial Mass Ratio (SMR)—incorporating spatial structural information via Moran's I, Edge Density, and Shannon Entropy—alongside a GMM meta-aggregator. Experiments across 10 datasets on OoD and failure detection tasks demonstrate that spatially-aware aggregation significantly outperforms global averaging.

## Background & Motivation

**Background**: In safety-critical domains such as medical imaging and autonomous driving, uncertainty quantification (UQ) methods produce pixel-level uncertainty maps that must be aggregated into image-level scalars for downstream tasks such as OoD detection and failure detection. Global averaging (AVG) is the de facto default.

**Limitations of Prior Work**: (1) Lack of systematic study—despite widespread use of aggregation, its properties and impact on downstream performance have not been comprehensively investigated; (2) AVG discards spatial structure—it cannot capture localized uncertainty patterns such as boundary or clustered uncertainty; (3) Existing alternative strategies lack systematic comparison, with inconsistent reporting across works.

**Key Challenge**: OoD sensitivity and prediction errors in segmentation are often reflected in **local uncertainty patterns**, yet simple pixel averaging destroys this spatial information.

**Key Insight**: The "spatial shape" of uncertainty is as informative as its magnitude.

**Core Idea**: The paper proposes the Spatial Mass Ratio (SMR)—the proportion of uncertainty mass concentrated in high-spatial-structure regions—and a GMM meta-aggregator that jointly models intensity-based and spatial features.

## Method

### Overall Architecture
Input: a 2D uncertainty map $U \in [0,1]^{m \times n}$ produced by a segmentation model. Output: an image-level scalar for OoD/failure detection. Pipeline: apply multiple aggregation functions to the uncertainty map → combine outputs into a feature vector → fit a GMM on in-distribution features → use negative log-likelihood as the anomaly score.

### Key Designs

1. **Formal Analysis of Common Aggregation Strategies and Their Deficiencies**:

    - **AVG**: Insensitive to spatial structure—uniformly low uncertainty and compactly clustered high uncertainty yield the same score.
    - **AQA (Above-Quantile Average)**: Lacks ratio invariance—scores change after cropping background regions.
    - **ATA (Above-Threshold Average)**: Non-monotonic—a global increase in uncertainty can paradoxically decrease the score.
    - **BCA/ICA (Class-weighted Averages)**: Leverage prediction information; ratio-invariant and consistently performant.

2. **Spatial Aggregation Strategies (Core Contribution)**:

    - **Function**: The SMR captures the spatial distributional structure of uncertainty.
    - **Mechanism**: $\text{SMR} = \text{mean uncertainty in high-structure regions} / \text{global mean uncertainty}$; SMR $= 0$ (noise-like) $\to 1$ (highly structured).
    - **SMR_Moran (MOR)**: Based on Moran's I spatial autocorrelation; SMR $= 0$ (noise) $\to 1$ (clustered).
    - **SMR_EDS (EDS)**: Based on Edge Density; SMR $= 0$ (flat regions) $\to 1$ (edge-concentrated).
    - **SMR_Entropy (ENT)**: Based on Shannon Entropy; SMR $= 0$ (constant regions) $\to 1$ (high local variability).
    - **Design Motivation**: Classical spatial analysis tools are applied to uncertainty maps to characterize the "shape" of uncertainty—a natural but previously overlooked direction.

3. **GMM Meta-Aggregator**:

    - **Function**: Unifies multiple aggregation strategies into a robust, general-purpose anomaly score.
    - **Mechanism**: Aggregation function outputs are treated as a feature vector $f_U = (f_1(U), \ldots, f_d(U))$; a GMM $p_{\text{GMM}}(f_U)$ is fitted on in-distribution samples with the number of components selected via BIC; the meta-aggregation score is the negative log-likelihood $f_{\text{meta}} = -\ln p_{\text{GMM}}(f_U)$.
    - **Three variants**: **GMM-Spa** (spatial features only), **GMM-Int** (intensity features only), **GMM-All** (spatial + intensity; recommended).
    - **Design Motivation**: Single aggregators are highly dataset-dependent; GMM-based probabilistic modeling achieves cross-dataset robustness.

### Experimental Setup
Ten datasets spanning medical imaging (LIDC/Lizard/ARC/WORM), autonomous driving (GTA→Cityscapes), and agriculture (WEED). Monte Carlo Dropout is used to generate uncertainty maps; results are additionally validated with Deep Ensembles and MSP.

## Key Experimental Results

### Main Results (OoD Detection AUROC)

| Aggregation | LIDC-Mal | CAR-CS | WORM-Pro | LIZ-IG | Mean Rank |
|-------------|----------|--------|----------|--------|-----------|
| AVG | ~0.78 | ~0.65 | ~0.72 | ~0.79 | Low |
| ATA | ~0.62 | ~0.58 | ~0.68 | ~0.72 | Lowest |
| BCA | ~0.82 | ~0.88 | ~0.85 | ~0.81 | Top tier |
| ICA | ~0.81 | ~0.87 | ~0.84 | ~0.80 | Top tier |
| **GMM-All** | ~0.80 | **~0.91** | **~0.88** | ~0.79 | **Top tier** |

Statistical testing (Wilcoxon $p < 0.05$): BCA, ICA, and GMM-All form a statistically significantly superior first tier.

### Failure Detection (E-AURC, lower is better)

| Aggregation | Key Finding |
|-------------|-------------|
| AVG | Worst rank; severely underestimates uncertainty for fully misclassified samples |
| QFR | Best rank ($p < 0.001$); threshold based on foreground proportion |
| GMM-All | Comparable to QFR; requires no hyperparameter tuning |
| ATA | Poor for OoD but competitive for FD, as segmentation errors concentrate along high-uncertainty boundaries |

### Key Findings
- **AVG performs near random chance in 6/10 settings** and should not serve as the default choice.
- Prediction-based methods (BCA/ICA) and GMM-All form a statistically significant top tier.
- Spatial structure is decisive in specific scenarios: EDS dominates OoD separation on the CAR-CS dataset (confirmed by SHAP analysis).
- The robustness of GMM-All stems from **combining intensity and spatial features**; leave-one-out analysis shows minimal sensitivity to removing any single aggregator.
- Trends are consistent across UQ methods (MCD, Deep Ensembles, MSP).

## Highlights & Insights
- **Pioneering systematic study**: This is the first comprehensive cross-dataset benchmark of segmentation uncertainty aggregation strategies, establishing a clear best practice—AVG should not be the default, and GMM-All is a robust general-purpose choice.
- **Introduction of spatial analysis tools**: Applying classical spatial statistics such as Moran's I and Edge Density to uncertainty analysis is a natural yet previously overlooked direction.
- **Parameter-efficient meta-aggregation**: GMM fitting adds no inference-time overhead; by operating in feature space, it automatically adapts to cross-dataset heterogeneity.

## Limitations & Future Work
- The GMM assumption may fail when in-distribution features are high-dimensional or exhibit complex non-Gaussian structure (e.g., the failure case on LIZ-IG).
- Stable GMM fitting requires a sufficient number of in-distribution samples.
- The spatial metrics are currently hand-selected; learned spatial features are a natural extension.
- The framework can be extended to spatiotemporal uncertainty aggregation for 3D medical imaging and video segmentation.

## Related Work & Insights
- **vs. MSP-based methods**: MSP operates at the classification level; this paper addresses the segmentation-specific problem of aggregating pixel-level signals to image-level scores.
- **vs. task-specific optimization**: Prior work optimizes aggregation for individual tasks; this paper provides a unified framework across OoD detection and failure detection.
- **Transferable insight**: The spatial aggregation paradigm is applicable to multimodal fusion, mixture-of-experts models, and other settings requiring aggregation of multi-source predictions.

## Rating
- Novelty: ⭐⭐⭐⭐ Spatial aggregation and GMM meta-aggregation are novel contributions, though the individual components build on established spatial statistics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets, two downstream tasks, multiple UQ methods, detailed statistical analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, theoretical analysis is rigorous, and experimental design is systematic.
- Value: ⭐⭐⭐⭐⭐ Provides practical guidance for reliable segmentation in safety-critical applications; open-source tooling further enhances applicability.

---

**Conference**: CVPR 2026
**arXiv**: [2603.29941](https://arxiv.org/abs/2603.29941)
**Code**: [https://github.com/Kainmueller-Lab/aggrigator](https://github.com/Kainmueller-Lab/aggrigator)
**Area**: Medical Imaging
**Keywords**: Uncertainty Quantification, Spatial Aggregation Strategies, OoD Detection, Failure Detection, Meta-Aggregation

## TL;DR
This paper presents the first systematic study of aggregation strategies for converting pixel-level uncertainty to image-level scores in segmentation. It proposes SMR aggregators incorporating spatial structural information (Moran's I, Edge Density, Shannon Entropy) and a GMM-based meta-aggregator, demonstrating across 10 datasets that global averaging (AVG) is suboptimal and that GMM-All achieves robust performance on both OoD and failure detection.

## Background & Motivation

1. **Background**: In safety-critical applications such as medical imaging and autonomous driving, segmentation models must output calibrated confidence estimates. UQ methods generate per-pixel uncertainty scores, but downstream tasks require aggregating these into a single image-level scalar for OoD detection and failure detection.
2. **Limitations of Prior Work**: (1) Global averaging (AVG) is the default yet discards spatial structure; (2) Alternative strategies (patch-level, class-level, threshold-level) lack systematic comparison; (3) Existing strategies have theoretical flaws—AQA lacks ratio invariance, and ATA is non-monotonic.
3. **Key Challenge**: OoD sensitivity and prediction errors in segmentation are typically reflected in local uncertainty patterns (e.g., unseen class regions, ambiguous boundaries), but simple pixel averaging obscures these critical local variations.
4. **Key Insight**: The spatial distribution of uncertainty (e.g., clustered vs. boundary-concentrated) carries important diagnostic information that spatially-aware aggregation can capture.
5. **Core Idea**: The paper proposes the Spatial Mass Ratio (SMR)—measuring the proportion of uncertainty mass in high-spatial-structure regions—and a GMM meta-aggregator that fuses the outputs of multiple aggregation strategies.

## Method

### Overall Architecture
Input: a pixel-level uncertainty map $U \in [0,1]^{m \times n}$ produced by the segmentation model.
Output: a scalar $f(U) \in \mathbb{R}$ for OoD or failure detection.
Two major categories of aggregation: (1) intensity-based (pixel-level and prediction-based); (2) spatially-aware (based on spatial structure metrics). All strategies are unified via a GMM meta-aggregator.

### Key Designs

1. **Analysis of Common Aggregation Strategy Deficiencies**:
    - **AVG** (global mean): Spatially insensitive—distinct spatial configurations with identical pixel value distributions yield the same score.
    - **AQA** (above-quantile average): Lacks ratio invariance—cropping background pixels changes the score.
    - **ATA** (above-threshold average): Non-monotonic—a global increase in pixel uncertainty may decrease the resulting score.
    - **BCA/ICA** (class-level averages): Prediction-based; satisfy ratio invariance.

2. **Spatial Aggregation Strategies (SMR)**:
    - **Function**: Compute the proportion of uncertainty mass in high-spatial-structure regions.
    - **Mechanism**: Weight the uncertainty map by a spatial metric; compute the ratio of mean uncertainty in high-structure regions to global mean uncertainty.
    - Three implementations:
        - **SMR_Moran (MOR)**: Moran's I measures spatial autocorrelation; 0 = noise-like, 1 = fully clustered.
        - **SMR_EDS (EDS)**: Edge Density Score; 0 = flat regions, 1 = edge-concentrated.
        - **SMR_Entropy (ENT)**: Shannon Entropy reflects local heterogeneity; 0 = constant, 1 = high variability.
    - **Design Motivation**: Different spatial patterns correspond to different anomaly types—clustered uncertainty (novel objects), boundary uncertainty (ambiguous contours), high variability (classification instability).

3. **GMM Meta-Aggregator**:
    - **Function**: Fuse multiple aggregation strategies into a unified anomaly detection score.
    - **Mechanism**: Represent the uncertainty map as a multi-dimensional feature vector $f_U = (f_1(U), \ldots, f_d(U))$; fit a GMM on in-distribution sample features $p_{\text{GMM}}(f_U)$; the meta-aggregation score is the negative log-likelihood $f_{\text{meta}}(U) = -\ln p_{\text{GMM}}(f_U)$.
    - Three variants: GMM-Spa (spatial only), GMM-Int (intensity only), GMM-All (all features).
    - **Design Motivation**: Single aggregators are highly dataset-dependent; GMM-All adaptively captures multi-dimensional feature discrepancies through probabilistic modeling.

### Experimental Setup
10 datasets: synthetic histopathology (ARC), Lizard pathology, LIDC lung CT, *C. elegans* microscopy, GTA/Cityscapes urban scenes, WeedsGalore crops. Multiple segmentation architectures (U-Net/HRNet/DeepLabv3+); uncertainty obtained via MC Dropout.

## Key Experimental Results

### Main Results (OoD Detection AUROC)

| Aggregation | LIDC-Mal | CAR-CS | WORM-Pro | LIZ-IG | Mean Rank |
|-------------|----------|--------|----------|--------|-----------|
| AVG | Suboptimal (partial) | Near random | Poor | Competitive | Low |
| AQA | Poor | Poor | Poor | Moderate | Low |
| BCA | Good | Good | Good | Good | **Top tier** |
| ICA | Good | Good | Good | Good | **Top tier** |
| **GMM-All** | Good | **Best** | **Best** | Moderate | **Top tier** |

Statistical significance (Wilcoxon $p < 0.05$): BCA, ICA, and GMM-All form a statistically significant top tier.

### Failure Detection (E-AURC, lower is better)

| Aggregation | Statistical Rank |
|-------------|-----------------|
| QFR | **Statistically best** ($p < 0.001$) |
| BCA | Second tier |
| GMM-All | Second tier, close to QFR |
| AVG | Worst (except on synthetic data) |

### Key Findings
- **AVG performs poorly in 6/10 settings**, approaching random chance; it should not be the default choice.
- GMM-All is the most robust strategy for OoD detection (consistent cross-dataset performance) and approaches optimal QFR on failure detection.
- SHAP analysis shows EDS dominates OoD separation on the CAR dataset, while no feature provides clear separation on LIZ-IG.
- Trends are consistent across UQ methods (MCD, Deep Ensembles, MSP, TTA), validating the generality of the aggregation analysis.

## Highlights & Insights
- **Systematic benchmark value**: This is the first comprehensive, cross-dataset, cross-task (OoD + FD) systematic comparison of segmentation aggregation strategies, overturning the assumption that "AVG is sufficient."
- **Intuition behind SMR**: The "shape" (clustered/edge/noise) of uncertainty is as important as its magnitude—a finding with broad implications for the UQ community.
- **Parameter efficiency of GMM meta-aggregation**: No inference overhead is added; a one-time GMM fit on the in-distribution set is sufficient to unify the advantages of multiple aggregators.

## Limitations & Future Work
- The GMM assumption may fail when in-distribution features are high-dimensional or multi-modal (e.g., the failure case on LIZ-IG).
- GMM fitting requires an in-distribution set, posing a dependency in cold-start scenarios.
- The current work covers 2D segmentation only; extension to 3D medical segmentation (volumetric) or video segmentation (spatiotemporal uncertainty) is a natural next step.
- Online GMM updates to support continual learning scenarios are worth exploring.

## Related Work & Insights
- **vs. MSP-based methods**: MSP operates at the classification level; this paper focuses on how to aggregate pixel-level signals into actionable image-level decisions, which is more aligned with the fine-grained nature of segmentation.
- **vs. anomaly segmentation methods**: Anomaly segmentation directly produces pixel-level anomaly maps; this paper addresses how to aggregate pixel-level signals into actionable image-level judgments.
- **Transferable insight**: The GMM meta-aggregation concept is applicable to any setting requiring aggregation of multi-source signals, such as multimodal fusion and confidence estimation in mixture-of-experts systems.

## Rating
- Novelty: ⭐⭐⭐⭐ Spatial aggregation and meta-aggregation are novel, though individual components build on established spatial statistics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 diverse datasets, two downstream tasks, multiple UQ methods, SHAP analysis, and statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear and theoretical analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Provides a practical aggregation selection guide for safety-critical applications; open-source tooling enhances impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[NeurIPS 2025\] AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](../../NeurIPS2025/medical_imaging/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)
- [\[ICLR 2026\] Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology](../../ICLR2026/medical_imaging/fusing_pixels_and_genes_spatially-aware_learning_in_computational_pathology.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mambabased_decoder_with.md)
- [\[CVPR 2026\] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)

<!-- RELATED:END -->
