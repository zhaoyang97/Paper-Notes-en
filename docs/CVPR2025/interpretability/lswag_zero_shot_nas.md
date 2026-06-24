---
title: >-
  [Paper Note] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers
description: >-
  [CVPR 2025][Interpretability][Neural Architecture Search] This paper proposes the L-SWAG metric, which characterizes the trainability and expressiveness of CNN and ViT networks through the product of layer-wise gradient variance and the cardinality of activation patterns. It further designs the LIBRA-NAS algorithm to combine complementary proxy metrics, achieving SOTA-level zero-shot NAS performance across ViT search spaces and 14 tasks.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Neural Architecture Search"
  - "Zero-Shot NAS"
  - "Vision Transformer"
  - "Proxy Metric"
  - "Metric Combination"
date: 2026-05-08
content_hash: cddd671878648f69
---

# L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers

**Conference**: CVPR 2025  
**arXiv**: [2505.07300](https://arxiv.org/abs/2505.07300)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Neural Architecture Search, Zero-Shot NAS, Vision Transformer, Proxy Metric, Metric Combination

## TL;DR
This paper proposes the L-SWAG metric, which characterizes the trainability and expressiveness of CNN and ViT networks through the product of layer-wise gradient variance and the cardinality of activation patterns. It further designs the LIBRA-NAS algorithm to combine complementary proxy metrics, achieving SOTA-level zero-shot NAS performance across ViT search spaces and 14 tasks.

## Background & Motivation
1. **Background**: Zero-shot NAS utilizes zero-cost (ZC) proxy metrics to evaluate network architecture quality without training models, offering both temporal efficiency and interpretability.
2. **Limitations of Prior Work**: Existing SOTA proxy metrics are primarily limited to convolutional search spaces (such as NAS-Bench-201) and perform poorly on Vision Transformer search spaces, sometimes even underperforming simple parameter-count metrics.
3. **Key Challenge**: Existing metrics either only consider gradients (trainability) or only consider activation patterns (expressiveness); a single dimension is insufficient to comprehensively characterize networks. Moreover, most metrics treat all layers equally, ignoring the differences in gradient statistics across different layers.
4. **Goal**: To design a general proxy metric suitable for both CNN and ViT search spaces, and to develop an intelligent metric combination method.
5. **Key Insight**: (1) Theoretically analyze the ZiCO metric to prove that the gradient mean should be discarded, keeping only the variance; (2) Empirically discover that the contribution of gradient statistics varies significantly across different layers; (3) Combine expressiveness metrics to compensate for the limitations of pure gradient metrics on ViTs.
6. **Core Idea**: Layer-wise gradient variance (trainability) $\times$ Layer-wise activation pattern cardinality (expressiveness) = L-SWAG.

## Method

### Overall Architecture
Input batch + randomly initialized DNN $\to$ Extract layer-wise gradient statistics (only variance, discarding mean) $\to$ Select the most informative layer interval $\to$ Compute trainability score $\Lambda^{\hat{L}}$ $\to$ Compute layer-wise SWAP expressiveness score $\Psi_{\mathcal{N},\theta}^{\hat{L}}$ $\to$ Multiply both to obtain L-SWAG $\to$ Use LIBRA-NAS to combine multiple metrics.

### Key Designs

1. **Layer-wise Gradient Variance Metric ($\Lambda^{\hat{L}}$)**:

    - **Function**: Measures the trainability of the network within the selected layer interval.
    - **Mechanism**: Computes the sample-wise variance $\text{Var}(|\nabla_w \mathcal{L}|)$ of gradients for each layer $l$, then takes the reciprocal and performs logarithmic summation. Key improvements: (1) Theoretically proves (Theorem 1) that the gradient mean $\mu$ in ZiCO should be discarded and replaced with a constant 1; (2) Analyzes the layer-wise gradient statistics of 1000 random networks and reveals that only statistics in specific layer intervals (from $\hat{l}$ to $\hat{L}$) are meaningful, thus only selecting these layers for calculation.
    - **Design Motivation**: The $\mu/\sigma$ ratio in ZiCO is theoretically invalid (Theorem 1 proves that the contribution of $\mu$ is canceled out by the learning rate), and treating all layers with equal weight is suboptimal.

2. **Layer-wise Activation Pattern Expressiveness ($\Psi_{\mathcal{N},\theta}^{\hat{L}}$)**:

    - **Function**: Measures the number of linear regions of the network over the input space, reflecting expressiveness.
    - **Mechanism**: Defines Sample-Wise Activation Patterns (SWAP) — binarizing the post-activation values of each layer to obtain a set of activation patterns, where its cardinality serves as the expressiveness score. This method is extended from ReLU to GeLU networks for the first time, making it applicable to ViTs.
    - **Design Motivation**: Pure gradient metrics fail in ViT search spaces because the expressiveness difference in ViTs is a crucial distinguishing factor of architecture quality.

3. **LIBRA-NAS Metric Combination Algorithm**:

    - **Function**: Intelligently combines multiple proxy metrics to obtain higher correlation than any single metric.
    - **Mechanism**: A three-step selection: (1) Selects the metric $z_{\text{best}}$ with the highest correlation; (2) Selects the most complementary metric (lowest conditional mutual information) via information gain; (3) Selects the metric with a bias closest to the validation accuracy distribution for bias realignment. Ultimately, a combination of three metrics replaces a single metric for NAS search.
    - **Design Motivation**: Different search spaces may favor different types of metrics, and a single metric cannot adapt to all scenarios.

### Loss & Training
Training-free (zero-shot); L-SWAG can be computed with only one forward and one backward propagation. LIBRA-NAS integrated into NAS search discovers an architecture with a 17.0% test error rate on ImageNet1k in 0.1 GPU days. The newly constructed ViT evaluation benchmark contains 2000 trained ViT models, covering the Autoformer search space on CIFAR-10, CIFAR-100, and ImageNet16-120 with three training strategies (AE, Jigsaw, Normal).

## Key Experimental Results

### Main Results

| Metric | ViT (Average $\rho$ on 6 Tasks) | NAS-Bench-201 (Average $\rho$) | TransNasBench (Average $\rho$) |
|------|----------------|---------------------|---------------------|
| #Params | 0.45 | 0.58 | 0.35 |
| ZiCO | 0.12 | 0.72 | 0.41 |
| NWOT | 0.38 | 0.65 | 0.28 |
| **L-SWAG** | **0.62** | **0.74** | **0.55** |

### Ablation Study

| Configuration | ViT Average $\rho$ | Description |
|------|---------|------|
| L-SWAG (full) | 0.62 | $\Lambda \times \Psi$ |
| Only $\Lambda$ (Trainability) | 0.48 | Expressiveness contribution +0.14 |
| Only $\Psi$ (Expressiveness) | 0.41 | Trainability contribution +0.21 |
| ZiCO (with $\mu$) | 0.12 | Discarding $\mu$ yields substantial improvement |
| All layers (Non-layer-wise) | 0.51 | Layer selection contribution +0.11 |

### Key Findings
- Existing proxy metrics universally degrade on ViT search spaces, with most performing worse than the parameter count.
- Discarding the gradient mean $\mu$ is validated both theoretically and experimentally (Theorem 1 proves that the contribution of $\mu$ is canceled out by the learning rate $\eta$, and the correct upper bound only contains $\sigma^2$ and $((M\eta-1)\mu)^2$ terms).
- The layer-wise selection strategy significantly improves metric quality and computational efficiency by focusing on information-dense layers.
- The combination (multiplication) of trainability and expressiveness is crucial for ViTs — using either dimension alone is insufficient.
- The architecture found by LIBRA-NAS within 0.1 GPU days (17.0% error rate) outperforms evolutionary and gradient-based NAS methods.
- Ablation of the LIBRA three-step selection strategy: selecting $z_2$ via min IG consistently outperforms max IG, random selection, and category-based selection; selecting $z_3$ via bias matching outperforms bias minimization and random selection.
- The SWAP expressiveness metric is successfully extended from ReLU to GeLU networks (via binarization approximation), making it applicable to ViTs.

## Highlights & Insights
- **Theory-driven metric design**: Theorem 1 rigorously proves the redundancy of the gradient mean term in ZiCO — the contribution of $\mu$ in the training loss upper bound is canceled out by the choice of learning rate $\eta$, and only the $\sigma^2$ term is genuinely related to trainability.
- **Practical value of layer-wise analysis**: The visualization of layer-wise gradient statistics of 1000 networks intuitively shows "which layers are important", which is heuristic yet highly effective.
- **Generality of LIBRA**: The metric combination framework is independent of specific metrics and can integrate new proxy metrics at any time.
- **Search space construction**: A newly constructed evaluation benchmark of 2000 trained ViT models (covering 6 tasks) fills the gap in rigorous correlation analysis within ViT search spaces.

## Limitations & Future Work
- The threshold for layer selection needs to be analyzed beforehand for each search space, rather than being fully automated.
- On certain search spaces (such as NAS-Bench-201), the advantage of L-SWAG relative to ZiCO is marginal.
- Only classification tasks were validated; it has not been extended to tasks like detection and segmentation.
- In the future, more automated layer selection strategies and support for more ViT variants can be explored.
- The extension of the SWAP expressiveness metric from ReLU to GeLU is based on binarization approximation, and its applicability to other activation functions (e.g., SiLU/Swish) has not been verified.
- The mutual information estimation in the three-step selection strategy of LIBRA-NAS may not be accurate enough under small sample sizes.
- Corrected a mathematical error in the proof of Theorem 3.1 in the original ZiCO paper (missing the summation over $i$ from the fourth to the fifth line, and the $1/2$ factor was not correctly multiplied to all terms), providing the correct upper bound of the training loss.

## Related Work & Insights
- **vs ZiCO**: ZiCO uses the $\mu/\sigma$ ratio, which lacks a solid theoretical basis (Theorem 1) and fails on ViTs ($\rho$ is only 0.12); L-SWAG uses only $1/\sigma$ along with an expressiveness term, achieving a $\rho$ of 0.62 on ViTs.
- **vs AZ-NAS**: Although AZ-NAS includes ViT evaluation, it only evaluates within NAS search (where the performance gap in the search space is small); L-SWAG provides a rigorous correlation analysis of 2000 trained networks.
- **vs Te-NAS**: Te-NAS relies on neural tangent kernel (NTK) theory (which is invalid for modern DNN hypotheses), whereas L-SWAG is based on more reliable gradient variance theory.
- **vs NWOT**: NWOT achieves a $\rho$ of only 0.38 on ViTs, whereas L-SWAG is significantly superior with a $\rho$ of 0.62.

## Rating

### Implementation Details
Evaluated on NB201, NB301, TransNasBench-101, and the self-built Autoformer ViT search space.
Uses bert-base-uncased as the embedding model, 1×NVIDIA A100 GPU.
L-SWAG can be computed with only one forward and one backward propagation.
- **Novelty**: ⭐⭐⭐⭐ Novel combination of theoretical proof, layer-wise analysis, and ViT extension
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 14 tasks, 2000 trained ViT evaluations, and detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations and rich experimental charts
- **Value**: ⭐⭐⭐⭐ Fills the gap in zero-shot NAS for ViTs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](open_ad-hoc_categorization_with_contextualized_feature_learning.md)

</div>

<!-- RELATED:END -->
