---
title: >-
  [Paper Note] AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing
description: >-
  [NeurIPS 2025][Object Detection][contrastive learning] This work proposes the AutoSciDACT pipeline, which first employs supervised contrastive learning to compress high-dimensional scientific data into a 4-dimensional em…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "contrastive learning"
  - "anomaly detection"
  - "hypothesis testing"
  - "novelty detection"
  - "scientific discovery"
date: 2026-05-08
content_hash: e7828377387e3a1e
---

# AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing

**Conference**: NeurIPS 2025
**arXiv**: [2510.21935](https://arxiv.org/abs/2510.21935)  
**Code**: TBD  
**Area**: Other
**Keywords**: contrastive learning, anomaly detection, hypothesis testing, novelty detection, scientific discovery

## TL;DR

This work proposes the AutoSciDACT pipeline, which first employs supervised contrastive learning to compress high-dimensional scientific data into a 4-dimensional embedding space, then applies NPLM (New Physics Learning Machine) likelihood-ratio testing to statistically quantify distributional deviations in the embedding space. The pipeline achieves $\geq 3\sigma$ discovery at signal injection ratios of $\leq 1\%$ across astronomical, particle physics, pathology, image, and synthetic datasets.

## Background & Motivation

**Background**: Modern scientific datasets are large in scale and extremely high-dimensional. Genuine novel phenomena are often obscured by statistical noise or random fluctuations, making exhaustive manual inspection infeasible.

**Limitations of Prior Work**: Existing anomaly detection methods (autoencoders, OOD detectors) largely only "flag" anomalous samples and report an AUROC score, without providing statistical significance statements such as p-values or Z-scores, which are insufficient for the rigorous standards of scientific discovery.

**Key Challenge**: Classical goodness-of-fit (GOF) tests suffer a dramatic loss of sensitivity in high-dimensional spaces and require unrealistically large sample sizes. Even powerful tools such as NPLM require low-dimensional inputs to remain effective. Traditional approaches also rely on domain experts for manual feature engineering, lacking automation and cross-domain generalizability.

**Key Insight**: Labeled simulation data are abundant in many scientific domains (particle physics simulations, gravitational wave templates, annotated pathology tissue), providing natural positive pairs for supervised contrastive learning.

**Goal**: This paper is the first to combine contrastive learning (for dimensionality reduction) with NPLM (for statistical testing) into an end-to-end pipeline, bridging the systematic gap between learned low-dimensional representations and rigorous hypothesis testing.

## Method

### Overall Architecture

AutoSciDACT operates in two phases:

```
Phase 1: Pre-training (contrastive learning for dimensionality reduction)
  High-dimensional input x ∈ X → Encoder f_θ → 4-dim embedding h = f_θ(x)

Phase 2: Discovery (NPLM hypothesis testing)
  Reference set R (known background) + Observation set D (potentially containing new signals)
  → NPLM likelihood-ratio test → p-value / Z-score
```

### Key Designs

**Design 1: Supervised Contrastive Learning (SupCon)**

- Building on the SimCLR framework, class labels are used to define positive pairs (same class = positive, different class = negative), avoiding the difficulty of selecting optimal data augmentations.
- The total loss is $\mathcal{L} = \mathcal{L}_{\text{SupCon}} + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}}$, where the cross-entropy term with $\lambda_{\text{CE}} \sim 0.1\text{-}0.5$ serves as an auxiliary classification objective to improve inter-class separation and structural regularity in the embedding space.
- Crucially, anomalous signal classes are **excluded** from pre-training and are only injected during the discovery phase, simulating the realistic scenario of "unknown novel phenomena."

**Design 2: NPLM Likelihood-Ratio Test**

- The test statistic is defined as: $t(\mathcal{D}) = 2 \max_{\boldsymbol{w}} \sum_{x \in \mathcal{D}} \log \frac{\mathcal{L}(x|\mathcal{H}_{\boldsymbol{w}})}{\mathcal{L}(x|\mathcal{H}_0)}$
- The alternative hypothesis density is parameterized as $p(x|\mathcal{H}_w) = p(x|\mathcal{H}_0) \exp[f_w(x)]$, requiring no specific signal model, making it a signal-agnostic approach.
- $f_w$ is implemented via Nyström-approximated Gaussian kernels (with $M \sim \sqrt{|D|+|R|}$ kernel components) and trained using weighted binary cross-entropy.
- The null distribution of $t$ is calibrated through 500 pseudo-experiments, with an asymptotic $\chi^2$ approximation also available.

**Design 3: Multi-Scale Kernel Width Combination**

- Kernel width governs NPLM sensitivity to distributional deviations at different scales. Six kernel widths are used (corresponding to the 1st, 25th, 50th, 75th, 99th percentiles of pairwise distances in the embedding space, plus $2\times$ the 99th percentile), and the final result is the mean p-value across all widths.
- This combination strategy is analogous to a look-elsewhere effect correction, trading peak sensitivity of a single optimal kernel for robustness across scales.

### Loss & Training

Total pre-training loss:

$$\mathcal{L} = \mathcal{L}_{\text{SupCon}} + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}}$$

NPLM training loss (weighted binary cross-entropy with regularization):

$$\mathcal{L}_{\text{NPLM}} = \sum_{(x,y)} \left[ w_R (1-y) \log(1+e^{f_w}) + y \log(1+e^{-f_w}) \right] + \lambda \sum_{i,j} w_i w_j k_i(x_j)$$

## Key Experimental Results

### Dataset Overview

| Dataset | Area | Input Dim | Encoder | Anomalous Signal |
|--------|------|----------|-----------|---------|
| Synthetic | Synthetic | $D+M$-dim Gaussian | MLP | Held-out Gaussian cluster |
| LIGO | Astronomy (gravitational waves) | $2\times200$ time series | 1D ResNet | White noise burst waveforms |
| JetClass | Particle physics | $\mathcal{O}(100)$ particles | Particle Transformer | $H\to b\bar{b}$ decay |
| Histology | Pathology | $256\times256$ images | EfficientNet-B0 | NAFLD steatotic tissue |
| CIFAR-10 | Image | $32\times32\times3$ | ResNet-50 | Held-out class 1 |

### Main Results (Figure 3)

| Dataset | Signal fraction $f_S$ | NPLM Z-score | Comparison: Mahalanobis |
|--------|----------------|-------------|------------------|
| Synthetic (2k) | ~0.6% | $\geq 3\sigma$ | Comparable (Gaussian clusters naturally suited) |
| Particle Physics | ~2–3% | $\geq 3\sigma$, near supervised upper bound | Substantially worse |
| Astronomy (LIGO) | ~1–2% | $\geq 3\sigma$, near supervised upper bound | Substantially worse |
| Histology | ~3–5% | $\geq 3\sigma$ | Substantially worse |
| CIFAR-10 | ~5% | $\geq 3\sigma$ | Substantially worse |

### Key Findings

1. **Discovery at very low signal fractions**: All datasets achieve $Z \geq 3\sigma$ at signal injection ratios of $\leq 5\%$; some datasets reach discovery level at $\sim 1\%$.
2. **Near-supervised upper bound**: On particle physics and astronomical datasets, NPLM (with no knowledge of signal morphology) achieves sensitivity close to the fully supervised testing upper bound.
3. **Robustness to dimensional noise**: Synthetic data experiments (Fig. 3a) show that Mahalanobis sensitivity in the original space degrades sharply as noise dimensionality increases, while the embedding-space approach remains stable.
4. **Mahalanobis baseline fails under non-Gaussian distributions**: Its Gaussian-per-class assumption and insensitivity to dense regions cause it to substantially underperform NPLM on real scientific data.
5. **Cross-domain transferability**: Statistical testing methodology developed in particle physics is successfully transferred to entirely different scientific domains such as pathology.

## Highlights & Insights

- **First end-to-end scientific discovery pipeline**: Contrastive learning for dimensionality reduction and statistical hypothesis testing are unified into an automated workflow that outputs statistical significance (p-value/Z-score) rather than anomaly scores, satisfying the $5\sigma$ standard for scientific discovery.
- **Signal-agnostic**: NPLM requires no prior knowledge of the specific form of anomalous signals, enabling unbiased search through data-driven density deviation modeling.
- **Effective at extremely low embedding dimensionality ($d=4$)**: Demonstrates that contrastive learning can retain sufficient semantic discriminability in a severely compressed space.
- **Cross-domain generality**: A single pipeline covers four distinct scientific domains—astronomy, particle physics, pathology, and image analysis—with domain-specific encoder architectures but a unified framework.
- **Multi-scale kernel strategy**: Six kernel widths combined via mean p-value automatically adapt to anomalies at different scales.

## Limitations & Future Work

1. **Strong dependence on label quality**: SupCon requires class labels to construct positive pairs; label noise or label scarcity directly degrades embedding quality.
2. **Expressivity bottleneck at $d=4$**: When the number of classes is large, a 4-dimensional space may fail to perfectly separate all categories (explaining why "ideal supervised" underperforms "supervised" on LIGO and CIFAR-10).
3. **Domain shift not addressed**: The framework assumes that the reference set $R$ accurately represents the background distribution; in practice, systematic discrepancies between simulations and real data are common and are left as future work.
4. **Computational cost**: Running 500 pseudo-experiments, each requiring NPLM training, incurs substantial computation; the multi-kernel combination further increases the cost.
5. **Prior knowledge of background composition required**: The framework assumes that the proportions of background classes in $R$ and $D$ are identical, which may require additional domain expert input in practice.

## Related Work & Insights

- **Contrastive learning for dimensionality reduction**: Self-supervised methods including SimCLR, MoCo, VICReg, and Barlow Twins; SupCon uses labels to define positive pairs. This work builds on SupCon with an added CE auxiliary term.
- **Contrastive anomaly detection**: Methods such as CADet use contrastive embeddings for OOD detection but only report AUROC without statistical quantification. The key distinction of AutoSciDACT is its output of p-values.
- **ML-based hypothesis testing**: MMD, C2ST (classifier two-sample tests), and density ratio estimation. NPLM has been shown to outperform classical GOF and C2ST in high-energy physics.
- **Anomaly detection in scientific domains**: Autoencoder-based approaches (VAE-based) in particle physics and weakly supervised searches in astronomy exist, but lack a unified cross-domain framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematically combining contrastive learning with statistical hypothesis testing into an end-to-end pipeline is a novel contribution, though each individual component (SupCon, NPLM) is already established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five datasets, four scientific domains, three baseline comparisons, and ablation studies on kernel width and embedding dimensionality; however, direct comparisons with ML two-sample tests such as MMD/C2ST are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The analogy between scientific methodology and the pipeline is clearly drawn, and mathematical derivations are complete; however, the paper is lengthy and some content is relegated to the appendix.
- **Value**: ⭐⭐⭐⭐ — Provides a plug-and-play tool for novel phenomenon discovery across multiple scientific domains with strong practical potential, though domain shift and label dependency limit direct deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](../../ICCV2025/object_detection/automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[AAAI 2026\] MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)](../../AAAI2026/object_detection/movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)
- [\[ICML 2026\] Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection](../../ICML2026/object_detection/testing_the_test_score-direction_instability_in_class-split_anomaly_detection.md)

</div>

<!-- RELATED:END -->
