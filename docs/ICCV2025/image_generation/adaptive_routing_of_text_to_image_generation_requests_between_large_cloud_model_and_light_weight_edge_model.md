---
title: >-
  [Paper Note] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model
description: >-
  [ICCV 2025][Image Generation][text-to-image routing] This paper proposes RouteT2I, a framework that dynamically routes text-to-image generation requests to either a lightweight edge model or a large cloud model via multi…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "text-to-image routing"
  - "edge-cloud collaboration"
  - "mixture-of-experts"
  - "Pareto relative superiority"
  - "dual-gate MoE"
date: 2026-05-08
content_hash: b0ea09ff5e0cbd0e
---

# Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model

**Conference**: ICCV 2025
**arXiv**: N/A
**Code**: N/A
**Area**: Text-to-Image Generation / Model Routing
**Keywords**: text-to-image routing, edge-cloud collaboration, mixture-of-experts, Pareto relative superiority, dual-gate MoE

## TL;DR

This paper proposes RouteT2I, a framework that dynamically routes text-to-image generation requests to either a lightweight edge model or a large cloud model via multi-dimensional quality assessment metrics and a dual-gate token-selection MoE routing model, achieving 83.97% of the quality gain attainable by exclusively using the cloud model at a 50% routing rate.

## Background & Motivation

Large text-to-image models (e.g., Stable Diffusion 3.5 with 8 billion parameters) deliver high-quality outputs but rely on costly cloud infrastructure, whereas lightweight edge models are economical yet fall short on complex prompts. The key insight is that not all user prompts require a large model—on simple prompts, edge models can match or even surpass cloud models. Nevertheless, existing routing methods are designed primarily for LLMs and face two major challenges when transferred to T2I scenarios: (1) image quality lacks a unified evaluation standard and is inherently multi-dimensional and subjective; and (2) the output space of T2I is far larger than the input text space, making quality prediction from text alone considerably more difficult.

## Method

### Overall Architecture

RouteT2I consists of three core components: (1) a **multi-dimensional quality assessment system** that measures generated image quality across 10 dimensions—such as definition, detail, and clarity—using positive/negative text pairs and CLIP similarity; (2) a **routing model** based on a dual-gate token-selection MoE Transformer that predicts the quality gap (Pareto Relative Superiority, PRS) between edge- and cloud-generated images from the user prompt; and (3) a **routing policy** that assigns prompts to the edge or cloud under cost constraints by thresholding PRS.

### Key Designs

1. **Multi-Dimensional Contrastive Quality Metric and PRS**: Ten image quality dimensions are defined (clarity, detail, color, consistency, object completeness, etc.). Each dimension is measured by the sigmoid difference of CLIP similarity scores against positive and negative text prompts. Pareto Relative Superiority (PRS) quantifies the multi-dimensional quality gap between edge and cloud outputs: PRS > 0.5 indicates edge superiority, while PRS < 0.5 indicates cloud superiority. Quality distance normalization and a temperature parameter are introduced to regulate the distribution.

2. **Dual-Gate Token-Selection MoE**: The user prompt is treated as a token sequence, with each expert corresponding to one quality dimension. A token-selection gate uses a token–expert affinity matrix to select the Top-$K$ tokens most relevant to each dimension. Separate positive and negative gates evaluate the favorable and adverse contributions of each token to quality; the dominant influence direction is determined by contrasting the two gate outputs. The original linear layer is factorized into positive and negative projection matrices together with a shared scoring matrix to reduce parameter count.

3. **Multi-Head Quality Prediction and Routing Policy**: The model includes multiple prediction heads that output predicted values for each quality dimension. The routing policy uses a PRS threshold $\alpha$ (upper-bounded at 0.5) to make routing decisions: prompts whose PRS falls below the threshold are routed to the cloud; the remainder are handled at the edge. The threshold is adjusted to control the routing rate and satisfy cost constraints.

### Loss & Training

The Adam optimizer is used with a learning rate of 2e-5, a batch size of 16, and training for approximately 10 epochs on an NVIDIA 4090D GPU. The supervision signal consists of pre-computed ground-truth PRS values obtained by generating images with both the edge and cloud models and computing PRS accordingly.

## Key Experimental Results

### Main Results

| Method | Definition | Detail | Completeness | $\Delta P$ (%) |
|--------|-----------|--------|--------------|----------------|
| Edge only | 0.6251 | 0.6685 | 0.4690 | — |
| Cloud only | 0.6337 | 0.6847 | 0.4972 | — |
| Random routing (50%) | 0.6294 | 0.6766 | 0.4831 | 40.00 |
| RouteT2I (50%) | 0.6350 | 0.6786 | 0.4865 | **83.97** |
| ZOOTER | 0.6350 | 0.6796 | 0.4854 | 77.95 |

At a 50% routing rate, RouteT2I achieves a relative performance gain $\Delta P$ of 83.97%, significantly outperforming all baselines.

### Ablation Study

- **Normalized win-rate improvement**: At a 40% routing rate, RouteT2I reaches 30.60%, surpassing the best baseline by 3.83%.
- **Cost savings**: To achieve a 50% quality improvement, RouteT2I reduces cloud requests by 70.24% compared to random routing.
- **Token-selection gate**: Removing this component leads to a notable performance drop, confirming the importance of attending to salient tokens.
- **Dual-gate design**: Compared to a single gate, the dual-gate mechanism more effectively distinguishes the positive and negative contributions of individual tokens.

### Key Findings

- The number of nouns (entities) in a prompt is positively correlated with the quality advantage of the cloud model.
- On simple prompts, the edge model may outperform the cloud model; blindly routing to the large model wastes cost and may degrade quality.
- Multi-dimensional quality assessment is more robust than a single metric and captures quality differences more accurately.

## Highlights & Insights

- This work is the first to study edge–cloud routing for T2I generation requests, addressing a concrete and practically relevant problem.
- The PRS-based multi-dimensional quality comparison is elegant, circumventing the difficulties of subjective image quality evaluation.
- The dual-gate MoE design is well-motivated: aligning experts with quality dimensions and using token selection to simulate cross-attention in the generation process.

## Limitations & Future Work

- Training the routing model requires generating large volumes of images with both models to compute PRS, incurring significant upfront cost.
- Validation is limited to open-source Stable Diffusion variants; closed-source API models (e.g., DALL·E, Midjourney) are not covered.
- Quality assessment relies on CLIP similarity with positive/negative text pairs, and biases inherent to CLIP may influence results.
- User-personalized preferences are not considered in the routing decision.

## Related Work & Insights

- LLM routing methods such as RouteLLM and ZOOTER serve as important references, but must be redesigned to account for the distinctive output space of T2I generation.
- The multi-dimensional quality assessment paradigm could be extended to routing in video generation, 3D generation, and related domains.
- The MoE token-selection mechanism may offer insights for other tasks that require identifying salient keywords in prompts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to study T2I routing with a clearly defined problem formulation.
- **Technical Depth**: ⭐⭐⭐⭐ — PRS definition and dual-gate MoE design are rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 18 model pairs, multi-perspective evaluation, complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic and coherent mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to cost optimization in commercial T2I services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)
- [\[ICCV 2025\] Discovering Divergent Representations between Text-to-Image Models](discovering_divergent_representations_between_text-to-image_models.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICCV 2025\] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation](multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)

</div>

<!-- RELATED:END -->
