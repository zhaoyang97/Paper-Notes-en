---
title: >-
  [Paper Note] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers
description: >-
  [CVPR 2025][Model Compression][Zero-Shot NAS] This paper proposes the L-SWAG zero-cost proxy, which combines layer-wise gradient variance statistics (trainability) and activation pattern cardinality (expressivity). For the first time, it achieves a stable positive ranking correlation on ViT search spaces. Furthermore, it introduces the LIBRA-NAS ensemble algorithm to combine multiple proxy metrics, finding an architecture with a 17.0% test error rate on ImageNet-1k in just 0.…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Zero-Shot NAS"
  - "Zero-Cost Proxy"
  - "Vision Transformer"
  - "Gradient Statistics"
  - "Architecture Search"
date: 2026-05-08
content_hash: 86680e1b9833da7e
---

# L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers

**Conference**: CVPR 2025  
**arXiv**: [2505.07300](https://arxiv.org/abs/2505.07300)  
**Code**: None  
**Area**: Model Compression / Neural Architecture Search  
**Keywords**: Zero-Shot NAS, Zero-Cost Proxy, Vision Transformer, Gradient Statistics, Architecture Search

## TL;DR

This paper proposes the L-SWAG zero-cost proxy, which combines layer-wise gradient variance statistics (trainability) and activation pattern cardinality (expressivity). For the first time, it achieves a stable positive ranking correlation on ViT search spaces. Furthermore, it introduces the LIBRA-NAS ensemble algorithm to combine multiple proxy metrics, finding an architecture with a 17.0% test error rate on ImageNet-1k in just 0.1 GPU-day.

## Background & Motivation

**Background**: Zero-shot NAS evaluates architecture performance quickly without training the network through zero-cost (ZC) proxy metrics. Many proxy methods based on gradients (ZiCo, SNIP, GraSP) and gradient-free approaches (NWOT, SWAP) already exist.

**Limitations of Prior Work**: (1) Existing SOTA proxies are mainly designed for convolutional search spaces and perform poorly on ViT search spaces, sometimes even underperforming simple parameter count metrics; (2) Different proxies contain complementary information and have their own biases, lacking an effective combination strategy; (3) The theoretical foundation of methods like ZiCo (linear regression) does not fully apply to non-linear networks.

**Key Challenge**: As LLMs and ViTs become mainstream, proxy metrics for NAS need to be extended from CNNs to Transformers, but the theoretical assumptions and practical designs of existing metrics are difficult to generalize.

**Goal**: (1) Design a general ZC proxy that performs well on both CNNs and ViTs; (2) Design an information-theory-based proxy ensemble method.

**Key Insight**: By analyzing the theoretical necessity of the gradient mean $\mu$ in the ZiCo metric, this work proves that the role of $\mu$ in the upper bound of DPO can be replaced by a constant 1, and discovers that gradient statistical contributions vary significantly across different layers.

**Core Idea**: By multiplying the layer-wise selected inverse gradient variance (trainability) by the layer-wise activation pattern cardinality (expressivity), a comprehensive proxy metric applicable to both CNNs and ViTs is obtained.

## Method

### Overall Architecture

For a randomly initialized candidate network, L-SWAG takes a batch of images as input, extracts gradient statistics and activation patterns of selected layers, and calculates the final score for ranking. LIBRA-NAS, based on multiple existing proxy metrics, selects the best combination in three steps through correlation, information gain, and bias matching.

### Key Designs

1. **Improved Gradient Variance Metric $\Lambda^{\hat{L}}$**:

    - **Function**: Measures network trainability.
    - **Mechanism**: $\Lambda^{\hat{L}} = \sum_{l=\hat{l}}^{\hat{L}} \log(\sum_{w \in \theta_l} \frac{1}{\sqrt{Var(|\nabla_w \mathcal{L}|)}})$. Compared to ZiCo's $\mu/\sigma$, this work replaces the $\mu$ in the numerator with a constant 1. Theorem 1 proves that in a linear regressor, the training loss upper bound $\leq \frac{1}{2}M\sum_j[\sigma_j^2 + ((M\eta-1)\mu_j)^2]$. When $\eta = 1/M$, the $\mu$ term disappears, and only $\sigma$ determines the upper bound. Layer selection identifies spikes in specific layers (percentiles) by analyzing gradient statistics of 1,000 random networks, keeping only these layers.
    - **Design Motivation**: The $\mu$ component of ZiCo lacks theoretical support in non-linear networks, and experiments confirm that removing $\mu$ actually improves performance; layer selection both improves proxy quality and accelerates computation.

2. **Layer-wise SWAP-Score $\Psi_{\mathcal{N},\theta}^{\hat{L}}$**:

    - **Function**: Measures network expressivity.
    - **Mechanism**: Defines the layer-wise sample activation pattern for ReLU and GeLU networks (binarizing the activation values of each neuron in each layer across all samples) and calculates the number of distinct activation patterns (cardinality). This is the first time activation pattern analysis is extended to GeLU networks (ViTs use GeLU).
    - **Design Motivation**: The failure of pure gradient metrics on ViTs is due to a lack of expressivity measurement; while NWOT uses global Hamming distance, this work uses layer-wise cardinality to capture the "actual expressivity" of each layer more finely.

3. **LIBRA-NAS Proxy Ensemble Algorithm**:

    - **Function**: Automatically selects the optimal combination of proxy metrics for a specific search space.
    - **Mechanism**: Three-step selection—(1) Select the proxy $z_1$ with the highest correlation $\rho$; (2) Among proxies with close $\rho$, select $z_2$ with the lowest information gain (low IG implies capturing the same information as $z_1$, similar to "overfitting" validation accuracy); (3) Select $z_3$ whose bias is closest to the validation accuracy bias (matching rather than eliminating bias).
    - **Design Motivation**: Different search spaces prefer different types of proxies, and a single metric cannot fit all. LIBRA does not require training a predictor (thus maintaining the zero-shot property) and is more effective than simple averaging or debiasing strategies.

### Loss & Training

As this is a zero-shot method, no network training is involved. L-SWAG = $\Lambda^{\hat{L}} \times \Psi_{\mathcal{N},\theta}^{\hat{L}}$, where the two terms are multiplied (the theoretical motivation for multiplication over addition comes from the T-CET work).

## Key Experimental Results

### Main Results

| Search Space | L-SWAG $\rho$ | Second Place $\rho$ | Gain |
|---------|--------------|-------------|------|
| Average (14 tasks) | 0.72 | 0.62 (NWOT) | +0.10 |
| TNB101-Macro Jigsaw | 0.86 | 0.58 | +0.28 |
| NB101 C10 | 0.65 | 0.54 | +0.11 |
| Autoformer ViT Average | 0.52 | 0.35 (#Params) | +0.17 |

| NAS Search Results | Test Error Rate | GPU-days |
|-------------|----------|---------|
| L-SWAG (ImageNet1k) | 17.0% | 0.1 |
| LIBRA (ImageNet1k) | 16.8% | 0.1 |
| Evolution NAS | 17.5% | >1 |

### Ablation Study

| Configuration | Average $\rho$ | Description |
|------|-----------|------|
| Full L-SWAG ($1/\sigma$ + SWAP) | 0.72 | Complete model |
| Only $\mu/\sigma$ (ZiCo) | 0.58 | Retaining $\mu$ is actually worse |
| Only $1/\sigma$ (No expressivity) | 0.65 | Lack of SWAP leads to a sharp decline on ViTs |
| Only SWAP | 0.55 | Expressivity alone is insufficient |
| All layers vs. Layer selection | +0.05~0.15 | Layer selection significantly improves performance |

### Key Findings

- The $\mu$ component has a negative impact on performance, and removing it yields improvements across most search spaces.
- The expressivity term (SWAP) is crucial for the success of L-SWAG on ViTs—pure gradient metrics fail almost completely on ViTs.
- The layer selection strategy makes a positive contribution across all search spaces and significantly accelerates computation.
- LIBRA's "minimum information gain" strategy (selecting proxies that capture the same information) is counter-intuitive but effective.

## Highlights & Insights

- **Theoretically Guided Metric Improvement**: ZiCo is simplified with theoretical support by rigorously proving that $\mu$ does not contribute to the loss upper bound under the optimal learning rate. This "subtractive innovation" is worth learning.
- **Discoveries from Layer-wise Gradient Analysis**: The contribution of different layers to proxy quality varies significantly, and using only gradient spike layers can substantially improve ranking quality. This empirical discovery is transferable to other scenarios requiring layer-wise analysis.
- **Counter-intuitive Design of LIBRA**: Choosing complementary proxies with the lowest information gain (rather than the highest) essentially means "capturing different facets of the same signal", similar to the diversity-accuracy trade-off in ensemble learning.

## Limitations & Future Work

- The optimal percentile for layer selection requires analyzing 1,000 networks individually for each search space, which incurs some pre-computation overhead.
- The ViT search space (Autoformer Small) itself has small accuracy differences (~2%), making proxy evaluation difficult.
- LIBRA requires pre-computing the correlation of all proxies, making its cold-start capability on entirely new search spaces unknown.
- Future work could explore extending L-SWAG to LLM architecture search.

## Related Work & Insights

- **vs. ZiCo**: The direct target of improvement in this work, which comprehensively improves performance by removing $\mu$, adding layer selection, and introducing expressivity.
- **vs. NWOT**: Measures expressivity using global Hamming distance, leading to a significant drop on Micro search spaces; L-SWAG's layer-wise SWAP is more stable.
- **vs. AZ-NAS**: Also uses proxy ensembles but evaluates them during the search, making it difficult to evaluate proxy quality independently; L-SWAG provides a clearer correlation analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ Theory-driven improvements + first systematic evaluation of the ViT search space.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 tasks covering multiple search spaces with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning and information-rich charts.
- Value: ⭐⭐⭐⭐ Establishes the direction of zero-shot NAS for ViTs, with a highly generalizable LIBRA framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[AAAI 2026\] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](../../AAAI2026/model_compression/efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)
- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)

</div>

<!-- RELATED:END -->
