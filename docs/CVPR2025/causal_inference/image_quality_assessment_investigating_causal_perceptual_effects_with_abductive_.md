---
title: >-
  [Paper Note] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference
description: >-
  [CVPR 2025][Causal Inference][Image Quality Assessment] This paper formulates Full-Reference Image Quality Assessment (FR-IQA) as a counterfactual inference problem. By using a Structural Causal Model (SCM), it distinguishes between the causal components related to perceptual quality and the noise components in deep features. This achieves training-free, backbone-agnostic robust quality prediction, obtaining competitive performance on multiple benchmark datasets.
tags:
  - "CVPR 2025"
  - "Causal Inference"
  - "Image Quality Assessment"
  - "Counterfactual Inference"
  - "Full-Reference IQA"
  - "Perceptual Quality"
date: 2026-05-08
content_hash: a71b5ed1d4c32025
---

# Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference

**Conference**: CVPR 2025  
**arXiv**: [2412.16939](https://arxiv.org/abs/2412.16939)  
**Code**: [https://anonymous.4open.science/r/DeepCausalQuality-25BC](https://anonymous.4open.science/r/DeepCausalQuality-25BC)  
**Area**: Causal Inference  
**Keywords**: Image Quality Assessment, Causal Inference, Counterfactual Inference, Full-Reference IQA, Perceptual Quality

## TL;DR
This paper formulates Full-Reference Image Quality Assessment (FR-IQA) as a counterfactual inference problem. By using a Structural Causal Model (SCM), it distinguishes between the causal components related to perceptual quality and the noise components in deep features. This achieves training-free, backbone-agnostic robust quality prediction, obtaining competitive performance on multiple benchmark datasets.

## Background & Motivation

**Background**: Full-Reference Image Quality Assessment typically consists of three stages: feature decomposition, feature comparison, and perceptual score mapping. Deep learning methods (e.g., LPIPS, DISTS, DeepWSD) utilize pre-trained networks to extract deep features and calculate the distance between reference and distorted images to predict quality scores.

**Limitations of Prior Work**: Existing methods rely on statistical correlation rather than causal mechanisms—they can only quantify the impact of distortion on feature similarity, but cannot explain how distortion affects human perception. This prevents effective differentiation between distortions with similar feature distances but completely different perceptual impacts, limiting cross-dataset generalization ability.

**Key Challenge**: Deep features contain both information causally related to perceptual quality and quality-irrelevant noise. Existing methods treat all features equally, failing to separate causal and noise features, which degrades assessment accuracy.

**Goal**: To establish a causal inference framework to identify and extract features causally related to perceptual quality from deep features, eliminating the interference of noise features.

**Key Insight**: Introduce counterfactual reasoning—by applying the same intervention (e.g., adding confounding factors) to the features of both the reference and distorted images, and observing whether the perceptual distance changes, the causality of features can be verified.

**Core Idea**: Structural Causal Models are used to define causal features $\gamma$ and noise features $\eta$. Through intervention experiments (do-operations) and a confounder dictionary, feature channels with stable causal effects on perceptual quality are selected, and only causal features are used to calculate quality scores.

## Method

### Overall Architecture
Given reference image $I$ and distorted image $D$, multi-layer features are extracted via pre-trained deep networks (VGG/ResNet/EfficientNet). A Structural Causal Model (SCM) is then constructed, introducing an exogenous variable $U$ as a confounder to intervene on the features under different intensities. By observing the changes in perceptual distance before and after intervention, causal feature channels are selected (recorded in a confounder dictionary $\Gamma$). Finally, only causal features are used to calculate the causal transport cost (Causal Optimal Transport) between the reference and distorted images to serve as the quality prediction score.

### Key Designs

1. **Structural Causal Model (SCM) and Causal Feature Separation**:

    - **Function**: Decomposes deep feature parameters $\theta$ into causal parameters $\gamma$ (causally relevant to quality) and noise parameters $\eta$ (irrelevant to quality), achieving $\eta \perp \gamma$.
    - **Mechanism**: Defines the quality score as $Q_S = m(\phi_\gamma(I, D))$, meaning quality is determined solely by causal features. Causal feature selection is optimized by minimizing the worst-case prediction error over all possible distributions: $\min_{\theta \to \gamma} \sup_{P \in \mathcal{P}} \mathbb{E}_P[l(\cdot)]$.
    - **Design Motivation**: Existing methods directly use all pre-trained features (including noise) to assess quality, resulting in spurious correlations and insufficient generalization. Separating causal and noise features fundamentally addresses this issue.

2. **Deep Causal Measurement**:

    - **Function**: Validates and filters causal feature channels through intervention experiments.
    - **Mechanism**: Applies do-operations to the deep features of the reference and distorted images to obtain pre-intervention features $\mathbf{f}_I, \mathbf{f}_D$ and post-intervention features $\mathbf{f}'_I, \mathbf{f}'_D$. The perceptual distance difference is calculated as $\Delta = Dis(\mathbf{f}_I, \mathbf{f}_D) - Dis(\mathbf{f}'_I, \mathbf{f}'_D)$. If $\Delta \neq 0$, the feature possesses a causal effect. By varying the intervention intensity, feature channels that maintain causality under all intensities are recorded in the confounder dictionary $\Gamma(\mathbf{f}_I, \mathbf{f}_D)$.
    - **Design Motivation**: Observational correlation alone cannot distinguish between causality and spurious association. Intervention experiments (counterfactual reasoning) are classical means of validating causality.

3. **Causal Optimal Transport**:

    - **Function**: Calculates the perceptual quality difference between the reference and distorted images based on causal features.
    - **Mechanism**: $COT(P_X, P_Y) = \inf_{g \in G(P_X, P_Y)} \int \Gamma(x,y) \cdot c(x,y) \, dg(x,y)$, where $c(x,y)$ represents the L2 norm distance, and $\Gamma(x,y)$ represents the causal confounder dictionary acting as a weighting mechanism—meaning only causally relevant feature channels participate in the distance calculation.
    - **Design Motivation**: Traditional feature distances (like the weighted L2 in LPIPS) assign equal or learned weights to all channels, neglecting causality. Causal Optimal Transport theoretically ensures that only features truly affecting perception are factored into the measurement.

### Loss & Training
This method is entirely training-free. It directly uses the weights of pre-trained networks (VGG-16, ResNet, EfficientNet) and filters feature channels through causal intervention experiments. This is one of its core strengths—requiring no training on specific IQA datasets, thereby naturally possessing cross-dataset generalization capabilities.

## Key Experimental Results

### Main Results

| Dataset | Metric | Our-VGG | Our-EffNet | DISTS | DeepWSD | LPIPS |
|--------|------|---------|-----------|-------|---------|-------|
| LIVE | PLCC/SRCC | 0.929/0.932 | 0.927/0.932 | 0.924/0.925 | 0.904/0.925 | 0.866/0.863 |
| CSIQ | PLCC/SRCC | 0.949/0.952 | 0.933/0.938 | 0.919/0.920 | 0.941/0.950 | 0.891/0.895 |
| TID2013 | PLCC/SRCC | 0.909/0.884 | 0.899/0.879 | 0.854/0.830 | 0.894/0.874 | 0.713/0.713 |
| KADID | PLCC/SRCC | 0.898/0.899 | 0.905/0.907 | 0.886/0.886 | 0.887/0.888 | 0.838/0.837 |

### Ablation Study

| Configuration | LIVE PLCC/SRCC | CSIQ PLCC/SRCC | TID2013 PLCC/SRCC |
|------|---------------|---------------|-------------------|
| $\phi = \phi_\theta$ (All pre-trained weights) | 0.901/0.915 | 0.913/0.916 | 0.884/0.867 |
| $\phi = \phi_\gamma$ (Causal features) | 0.929/0.932 | 0.949/0.952 | 0.909/0.884 |
| $\phi = \phi_\eta$ (Noise features) | 0.843/0.866 | 0.803/0.831 | 0.786/0.789 |

### Key Findings
- The performance of causal features $\phi_\gamma$ is significantly better than that of using all pre-trained weights $\phi_\theta$, indicating that removing noise features indeed helps improve quality prediction.
- The noise features $\phi_\eta$ still exhibit some perceptual correlation (non-zero performance), but are far inferior to causal features, confirming the rationality of causal/noise separation.
- The method is highly versatile across backbones (VGG, ResNet, EfficientNet), with Our-VGG and Our-EffNet each having their own advantages.
- The training-free approach achieves or exceeds the performance of supervision-based methods on most datasets (except for TOPIQ-FR), indicating outstanding generalization capability.

## Highlights & Insights
- **Formulating IQA as a counterfactual inference problem**: This paradigm shift is the core contribution, providing a theoretical framework for understanding "why certain methods perform well or poorly in specific scenarios". Other regression-based perceptual tasks can also benefit from this approach.
- **Entirely training-free**: Relying on no training on any IQA dataset, it naturally possesses cross-dataset and cross-distortion generalization capabilities. This is highly valuable for real-world deployment (eliminating the need for dataset collection).
- **Backbone independence**: The causal analysis process is independent of the network architecture and can be applied plug-and-play to any pre-trained networks, offering high flexibility.

## Limitations & Future Work
- The performance on the PIPAL dataset (GAN-generated distortions) is inferior to TOPIQ-FR, suggesting that causal modeling for complex algorithmic distortions still requires improvement.
- The construction of the confounder dictionary depends on the choice of intervention intensity; different intensity ranges might affect the stability of the results.
- Causal intervention is conducted independently on each channel, ignoring interactive causal effects among channels.
- The method is highly theoretical, but engineering implementation details (e.g., intervention methods, dictionary construction algorithms) are not presented clearly enough in the paper.

## Related Work & Insights
- **vs LPIPS**: LPIPS uses learned linear weights to weight multi-layer feature distances, whereas this paper replaces learned weights with causal filtering, requiring no training and offering higher interpretability.
- **vs DISTS**: DISTS combines structural and texture similarities, but remains a statistical metric. This paper distinguishes "which features truly affect perception" from a causal perspective, establishing a more solid theoretical foundation.
- **vs DeepWSD**: DeepWSD uses Wasserstein distance to measure differences in feature distribution. This paper replaces it with causal transport cost, incorporating additional causal weighting.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing causal inference into FR-IQA is a novel perspective with significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive evaluation across 6 datasets, but the ablation study is relatively simple, lacking in-depth analysis of the intervention strategies.
- Writing Quality: ⭐⭐⭐ Detailed theoretical exposition but somewhat verbose in parts; the abundance of mathematical symbols increases reading difficulty.
- Value: ⭐⭐⭐⭐ Provides a new theoretical perspective and a training-free solution, offering inspiring insights for the IQA field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](../../ICML2025/causal_inference/isolated_causal_effects_of_natural_language.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/causal_inference/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[ICLR 2026\] Topological Causal Effects](../../ICLR2026/causal_inference/topological_causal_effects.md)

</div>

<!-- RELATED:END -->
