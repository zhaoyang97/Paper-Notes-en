---
title: >-
  [Paper Note] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models
description: >-
  [ICCV 2025][Image Generation][Text-to-image generation] This paper proposes RouteT2I, the first edge-cloud model routing framework for text-to-image generation. It maximizes image generation quality under cost constraint…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Text-to-image generation"
  - "model routing"
  - "edge-cloud collaboration"
  - "mixture of experts"
  - "multi-metric quality assessment"
date: 2026-05-08
content_hash: 355ca29af725f0d9
---

# Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Text-to-image generation, model routing, edge-cloud collaboration, mixture of experts, multi-metric quality assessment

## TL;DR

This paper proposes RouteT2I, the first edge-cloud model routing framework for text-to-image generation. It maximizes image generation quality under cost constraints through multi-dimensional quality metrics, Pareto Relative Superiority, and a dual-gated token selection MoE architecture.

## Background & Motivation

### Core Problem
Large-scale T2I models (e.g., SD3.5 with 8B parameters) deliver superior generation quality but incur prohibitive deployment costs ($65K per million requests). Lightweight edge models are cost-efficient but underperform on complex prompts. **Key observation**: not all prompts require a large model — for simple prompts, small models may produce comparable or even better results (e.g., when the number of nouns in an image is small).

### Why Existing LLM Routing Methods Cannot Be Directly Transferred?

**Difficulty of image quality assessment**: Unlike text, which has definite answers, image quality lacks a unified standard and is influenced by multiple factors including color, sharpness, and object completeness.

**Output space far exceeds input space**: A single text prompt can correspond to infinitely many images, making pre-generation quality prediction extremely challenging.

**Single-objective optimization is insufficient**: Existing LLM routing methods typically optimize a single quality metric, which cannot accommodate the ambiguity and multi-dimensionality of image quality.

### Core Insight
The number of nouns serves as an intuitive proxy for request complexity. Experiments (Fig. 3) show that as the number of nouns in a prompt increases, the win rate of the large model improves and the quality gap widens. However, even for simple prompts, the large model is not always superior — motivating the need for an intelligent routing mechanism.

## Method

### Overall Architecture

RouteT2I consists of three core components:
1. **Multi-dimensional quality metric system** (§4): Defines a multi-dimensional evaluation scheme for image quality.
2. **Routing model** (§5.1): A Transformer based on dual-gated token selection MoE that predicts Pareto Relative Superiority.
3. **Routing strategy** (§5.2): Routes requests based on predicted quality gaps and predefined cost constraints.

**Optimization objective**: Maximize overall generation quality subject to an upper bound $\rho_r$ on the cloud routing rate:
$$\max_{R(X)} Q(I_c) + (1-R(X))Q(I_e) \quad \text{s.t.} \quad P\{R(X)=1\} \leq \rho_r$$

### Key Designs

#### 1. Multi-Dimensional Contrastive Quality Metrics (§4)

**Core Idea**: Leverages text-image correspondence to measure image quality dimensions via positive-negative text pairs.

**Contrastive quality for a single metric**:
$$q(I, m) = \sigma(\text{CLIP}(I, m^+) - \text{CLIP}(I, m^-))$$

where $m=(m^+, m^-)$ is a positive-negative text pair. For example, the definition metric uses "High definition photo" as $m^+$ and "Low definition photo" as $m^-$.

**10-dimensional quality assessment**:
$$Q(I) = [q(I, m_i) | i=1,2,...,10]$$

This covers ten dimensions — Definition, Detail, Clarity, Sharpness, Harmony, Realism, Color, Consistency, Layout, and Integrity — combining factors for real photographic quality and generative-image-specific properties (e.g., realism and object completeness).

**Why use the contrastive approach?** Compared to using only positive prompts, the contrastive approach provides a more robust and reliable assessment by evaluating which quality pole — positive or negative — is dominant.

#### 2. Pareto Relative Superiority (PRS)

**Why not directly compare multi-dimensional quality?** In practice, it is difficult to find a Pareto-optimal image that is superior on all metrics simultaneously. The constraint is therefore relaxed: an image is allowed to be slightly inferior on some metrics as long as it significantly outperforms on others.

**Normalized quality distance**:
$$D_i(I_e, I_c) = \sigma\left(\frac{q(I_e, m_i) - q(I_c, m_i)}{\Gamma|\mu_i(I_e) - \mu_i(I_c)|}\right)$$

where the temperature parameter $\Gamma$ and the sigmoid function regulate the distribution, distinguishing closely matched quality levels and preventing centralization.

**PRS definition**:
$$\text{PRS}(I_e, I_c) = \sum_{i=1}^{N} w_i D_i(I_e, I_c)$$

The degree to which PRS deviates from 0.5 indicates the quality advantage of the edge or cloud model. PRS > 0.5 indicates edge superiority; PRS < 0.5 indicates cloud superiority.

#### 3. Dual-Gated Token Selection MoE (Core Architectural Innovation)

**Design Motivation**: In T2I generation, prompts interact with images as token sequences via cross-attention, and different tokens exert varying influences on image quality. The routing model must identify critical tokens and assess their positive or negative impact.

**Token selection gate**:
$$A = \text{Softmax}(T \cdot E^T)$$

where $T \in \mathbb{R}^{n \times d}$ is the token representation and $E \in \mathbb{R}^{k \times d}$ is the expert embedding (each expert corresponds to one quality metric). Top-K selection identifies the tokens most relevant to each expert.

**Why is token selection necessary?** In T2I generation, different tokens (nouns, adjectives, etc.) exert vastly different influences on different quality metrics. For instance, color-related tokens primarily affect the Color metric, while nouns primarily affect the Integrity metric. Selecting key tokens reduces interference from irrelevant ones.

**Dual-gate design**: Positive gate $G^+$ and negative gate $G^-$ are introduced to select tokens with positive and negative influences on quality, respectively:
$$T_i^o[t] = T[t] \cdot P_i^o \cdot S_i, \quad o \in \{+, -\}$$

where $P_i^+, P_i^- \in \mathbb{R}^{d \times l}$ are projection matrices mapping tokens into low-dimensional positive/negative representation spaces, and $S_i \in \mathbb{R}^{l \times h}$ is a shared scoring matrix. Since $l \ll h, d$, the parameter count is reduced from $O(hd)$ to $O(l(h+d))$.

**Positive-negative contrast**:
$$\hat{T}[t] = \sigma(T^+[t] - T^-[t])$$

Contrasting positive and negative representations determines the dominant influence (positive or negative) of each token, eliminating ambiguity in prediction.

**Multi-head prediction**: The model includes multiple prediction heads, each outputting a prediction for one quality metric, enhancing robustness and noise resistance.

### Routing Strategy

A PRS threshold $\alpha$ is set; prompts with PRS below the threshold are routed to the cloud (where cloud model quality is significantly better), while the rest are handled by the edge model:
$$\max_{\alpha \leq 1/2} P\{\text{PRS}(I_e, I_c) < \alpha | I_e, I_c \in \mathcal{I}_e, \mathcal{I}_c\} \leq \rho_r$$

The upper bound $\alpha \leq 1/2$ ensures that prompts where the edge model is superior are not incorrectly routed to the cloud.

## Key Experimental Results

### Main Results

**Setup**: SD3 (cloud) + SD2.1 (edge), COCO2014 dataset, 50% routing rate.

**Multi-dimensional quality comparison** (routing rate 50%):

| Method | Definition | Detail | Integrity | Δ P(%) |
|--------|-----------|--------|-----------|--------|
| Edge Only | 0.6251 | 0.6685 | 0.4690 | - |
| Cloud Only | 0.6337 | 0.6847 | 0.4972 | - |
| Random | 0.6294 | 0.6766 | 0.4831 | 40.00 |
| RouteLLM-BERT | 0.6347 | 0.6792 | 0.4866 | 71.51 |
| Hybrid LLM | 0.6327 | 0.6784 | 0.4864 | 73.49 |
| ZOOTER | 0.6350 | 0.6796 | 0.4854 | 77.95 |
| **RouteT2I** | **0.6350** | **0.6786** | **0.4865** | **83.97** |

RouteT2I outperforms all baselines on 6 out of 10 quality metrics, achieving an overall performance gain equivalent to **83.97%** of the cloud model's improvement.

**Cost savings** (cloud call rate reduction under Δ P targets):

| Method | Δ P=40% | Δ P=50% | Δ P=60% |
|--------|---------|---------|---------|
| RouteLLM-BERT | 56.15% | 51.39% | 46.92% |
| ZOOTER | 69.28% | 65.76% | 60.81% |
| **RouteT2I** | **71.81%** | **70.24%** | **66.61%** |

### Ablation Study

| Configuration | Δ w(%) @ p=40% | @ p=50% | @ p=80% |
|---------------|----------------|---------|---------|
| w/o Multi-Metric | 27.37 | 22.81 | 19.92 |
| w/o Token Selection | 27.82 | 23.05 | 19.24 |
| w/o Dual-Gate | 27.22 | 22.09 | 21.62 |
| **RouteT2I (Full)** | **30.60** | **25.81** | **21.94** |

### Key Findings

1. **Multi-dimensional optimization is critical**: Removing multi-metric quality optimization causes a 3.23% performance drop at 40% routing rate, as a single metric cannot comprehensively assess image quality.
2. **Token selection gate is more important at high routing rates**: Removing it leads to approximately 2% degradation at 80% routing rate, indicating that filtering key tokens becomes more critical when the majority of requests must be routed.
3. **Dual-gate is most effective at moderate routing rates**: The effect is most pronounced around 50% routing rate, where distinguishing positive and negative influences has the greatest impact on routing decisions.
4. **Cross-model pair generalizability**: The method remains effective across 18 cloud-edge model combinations, with more pronounced improvements for model pairs with larger quality gaps (e.g., SD3–SD1.5).
5. **Can surpass cloud-only performance**: In certain scenarios, the overall quality after routing even exceeds that of using the cloud model exclusively.

## Highlights & Insights

1. **First T2I routing framework**: Extends the concept of LLM routing to the image generation domain, with a clear and complete problem formulation.
2. **Positive-negative contrastive quality metrics**: Measuring image quality dimensions via CLIP with positive-negative text pairs is both elegant and effective.
3. **Elegant PRS design**: Relaxes Pareto optimality to permit trade-offs on some metrics while improving overall quality, aligning well with practical requirements.
4. **MoE aligned with the T2I generation process**: Aligning experts with quality metrics and simulating the differential token influences in cross-attention reflects deep domain insight in the architectural design.
5. **High practical value**: Reduces cloud calls by 71.81% under a 40% relative performance gain target, yielding significant cost savings.

## Limitations & Future Work

1. **Predictive vs. non-predictive routing**: The current approach makes routing decisions without running the edge model, but non-predictive routing (running the edge model first and then deciding whether to retry on the cloud) may be preferable in certain scenarios.
2. **Dependence on CLIP for quality metrics**: CLIP-based contrastive quality metrics may not fully align with human preferences; integration with human-aligned metrics such as ImageReward could be beneficial.
3. **Training data scale**: The routing model requires a large number of prompt-quality pairs for training, incurring substantial data collection costs.
4. **Dynamic model pool**: The current framework is fixed to two-model routing; extending to cascaded multi-model routing poses greater challenges.
5. **Privacy concerns**: Sending prompts to the cloud raises user privacy issues, which the paper does not address.

## Related Work & Insights

- **RouteLLM (2024)**: An LLM routing framework that uses a BERT classifier or matrix factorization to predict quality.
- **Hybrid LLM**: Allows the weaker model to succeed when the quality gap falls within a threshold, saving costs at the expense of some quality.
- **ZOOTER**: Predicts the normalized quality of candidate model outputs and routes based on relative quality.
- **Insights**: The idea of multi-metric quality assessment combined with contrastive measurement is generalizable to routing in other generative tasks (e.g., video generation, 3D generation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First T2I routing framework; novel problem formulation and highly original architectural design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (18 model pairs, diverse baselines, extensive ablations, and human evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem exposition and complete mathematical derivations, though the paper is lengthy)
- Value: ⭐⭐⭐⭐⭐ (High practical value for real-world commercial deployment with significant cost savings)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model](adaptive_routing_of_text_to_image_generation_requests_between_large_cloud_model_and_light_weight_edge_model.md)
- [\[ICCV 2025\] Discovering Divergent Representations between Text-to-Image Models](discovering_divergent_representations_between_text-to-image_models.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems](cure_cultural_gaps_in_the_long_tail_of_text-to-image_systems.md)

</div>

<!-- RELATED:END -->
