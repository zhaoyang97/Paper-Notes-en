---
title: >-
  [Paper Note] VLANeXt: A Recipe for Building Robust VLA Models
description: >-
  [ICML 2026][Multimodal VLM][Vision-Language-Action models] This paper systematically explores the VLA model design space, distilling 12 key design principles through 500+ controlled experiments to build the efficient and…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Vision-Language-Action models"
  - "Robotic learning"
  - "VLA design space"
  - "Multimodal fusion"
  - "Instruction-conditioned control"
date: 2026-05-08
content_hash: 55b23acc6320df04
---

# VLANeXt: A Recipe for Building Robust VLA Models

**Conference**: ICML 2026  
**arXiv**: [2602.18532](https://arxiv.org/abs/2602.18532)  
**Code**: https://github.com/DravenALG/VLANeXt  
**Area**: Embodied AI / VLA / Robotic Learning  
**Keywords**: Vision-Language-Action models, Robotic learning, VLA design space, Multimodal fusion, Instruction-conditioned control

## TL;DR
This paper systematically explores the VLA model design space, distilling 12 key design principles through 500+ controlled experiments to build the efficient and powerful VLANeXt model. It surpasses SOTA on the LIBERO benchmark and validates the effectiveness of these design principles in real-world robotic tasks.

## Background & Motivation

**Background**: VLA models leverage pre-trained VLMs to provide visual and language understanding for general robotic policy learning. Numerous VLA models have been proposed (RT-2, OpenVLA, π series, etc.), but their training protocols and evaluation settings vary significantly.

**Limitations of Prior Work**: The VLA field remains in a "primordial soup" stage—ideas are abundant but lack systematicity. Different methods adopt different VLM backbones, architectural designs, and loss functions, making fair comparison difficult.

**Key Challenge**: How to systematically compare VLA design choices under a unified framework to distinguish which designs are truly effective?

**Goal**: Revisit the VLA design space under a unified framework and evaluation setup to discover reproducible and generalizable design recipes.

**Key Insight**: Starting from RT-2, the study evolves across three dimensions: base components, perception elements, and action modeling. This systematic ablation path clearly demonstrates the contribution of each design choice.

**Core Idea**: Gradually optimize design choices through large-scale controlled experiments (>500) under a unified evaluation protocol, integrating fragmented VLA methodologies into 12 actionable design principles.

## Method

### Overall Architecture
Pipeline: Multimodal input (multi-view RGB + proprioception + language instructions) → Multimodal LLM → Soft-link to policy module → Action chunk prediction + frequency-domain auxiliary objective. The core feature is the introduction of a learnable query buffer between the VLM and the policy module to achieve a smooth transition between representation spaces.

### Key Designs

1.  **Soft-linking Policy Module and VLM**:

    - **Function**: Establishes a soft information flow between the VLM text representation space and the policy module action prediction space.
    - **Mechanism**: Compared to the "text token reuse" (tight coupling) of RT-2 and the "complete decoupling" of MetaQuery, soft-linking adopts hierarchical connections with an inserted learnable query buffer. Each VLM layer output interacts with policy module queries through cross-attention, followed by conditioning on timestep information via adaLN.
    - **Design Motivation**: Resolves the underfitting of hard links and the information loss of complete decoupling. Soft-linking achieves optimal performance on LIBERO-plus (56.2%), a 2.5% improvement over loose coupling.

2.  **Multi-view + VLM-side Proprioception Fusion**:

    - **Function**: Integrates robot proprioception and multi-view observations, fused at the VLM level.
    - **Mechanism**: Multi-view inputs are processed by the multimodal LLM image encoder; proprioception is converted into tokens via linear projection and input to the VLM alongside visual tokens. Proprioception should be injected at the VLM level rather than the policy module level.
    - **Design Motivation**: The alignment between state information provided by proprioception and visual instructions is higher at the VLM level (98.0% vs. 96.2%). Multi-view observations provide complementary geometric cues (91.8% → 97.6%).

3.  **Flow Matching + Frequency-domain Auxiliary Loss**:

    - **Function**: Treats action chunk prediction (8 steps) as a continuous time series, using flow matching as the primary loss and frequency-domain MSE as the auxiliary objective.
    - **Mechanism**: The primary loss uses flow matching to model continuous action distributions. The frequency-domain auxiliary loss converts actions via DCT and assigns higher weights to low-frequency components $L_{\text{freq}} = \text{MSE}(\text{DCT}(\hat{a}), \text{DCT}(a))$, where weights $w(\text{freq}) \propto 1/(\text{freq}+1)$.
    - **Design Motivation**: Regression loss is surpassed by flow matching in high-performance regimes. Frequency-domain regularization prevents overfitting to trajectory jitter, reaching 99.0% performance (+1% compared to regression) without increasing training overhead.

## Key Experimental Results

### Main Results

| Method | LIBERO (%) | LIBERO-plus (%) | Model Size |
|------|-----------|----------------|--------|
| OpenVLA | 76.5 | 15.6 | 7B |
| OpenVLA-OFT | 97.1 | 69.6 | 7B |
| π₀ | 86.0 | 53.6 | 11B |
| π₀-Fast | 85.5 | 61.6 | 7B |
| NORA | 87.9 | 39.0 | N/A |
| UniVLA | 95.2 | 42.9 | N/A |
| FLOWER | 96.9 | Unreported | N/A |
| **VLANeXt** | **97.4** | **83.9** | **2.5B** |

VLANeXt surpasses all baselines with a 2.5B model size (approximately 1/3 the size of OpenVLA-OFT).

### Ablation Study

| Design Dimension | Configuration | LIBERO (%) | LIBERO-plus (%) |
|--------|------|-----------|-----------------|
| **Base Components** | Single-layer policy head + Text token reuse | 19.8 | <5.0 |
| | Separate policy head (2 layers) | 30.2 | 16.6 |
| | Large policy module (12 layers) | 64.4 | 34.0 |
| | + Action chunking (chunk=8) | 74.6 | 43.4 |
| | + Flow matching loss | 80.0 | 45.0 |
| | + Qwen3-VL-2B backbone | 90.0 | 53.7 |
| | + Soft-link | 91.8 | 56.2 |
| **Perception Elements** | + Multi-view | 97.6 | 80.5 |
| | + VLM-side proprioception | 98.0 | 87.7 |
| **Action Modeling** | + Frequency-domain auxiliary loss | 99.0 | 93.1 |

### Key Findings
- Large policy modules provide the greatest contribution (+33.8%).
- Strong VLM backbones (+10.0%) are superior to merely increasing parameters.
- Perception elements (multi-view + proprioception) add a cumulative +13.0%.
- Frequency-domain loss is simple yet effective, with negligible computational overhead.
- Video history is not beneficial—adding temporal history actually causes performance drops (91.8% → 85.0%).
- Proprioception placement is sensitive—VLM-level injection is far superior to policy-level (98.0% vs. 96.2%).

## Highlights & Insights
- **Systematic Design Exploration**: 500+ controlled experiments decompose the VLA design space under a unified framework. The "recipe" mindset offers more methodological value to the community than isolated innovations.
- **Deep Insights into Multimodal Fusion**: Proprioception should be fused on the VLM side rather than the policy side, as multi-view observations provide geometric compensation.
- **Time-Series Concept Transfer**: Migrating frequency-domain regularization from time-series forecasting to action generation is simple and elegant, significantly improving robustness to disturbances in LIBERO-plus.
- **Efficiency-Performance Balance**: The 2.5B VLANeXt is significantly smaller than the 7B OpenVLA-OFT yet achieves superior performance.
- **Open-Source Contribution**: Releasing a unified, lightweight framework lowers the barrier to entry for VLA research.

## Limitations & Future Work
- Evaluation is limited to LIBERO/LIBERO-plus simulation benchmarks, with a small sample size for real-world robot validation.
- Dataset characteristics—LIBERO primarily involves manipulation tasks, lacking diverse scenarios like navigation.
- Computational efficiency—inference latency and VRAM usage for the 2.5B model were not reported in detail.
- Improvements: Transferring designs across embodiments and tasks; adaptive fusion strategies; integration with online learning; in-depth analysis of the frequency-domain loss mechanism.

## Related Work & Insights
- **vs RT-2/OpenVLA**: VLANeXt outperforms OpenVLA-OFT 7B at a 2.5B scale through optimized design details.
- **vs π series**: π is tightly coupled at 11B; VLANeXt’s soft-link is more lightweight with better performance.
- **vs World Model methods (WorldVLA)**: Auxiliary tasks add +2% but triple training time; VLANeXt replaces these with frequency-domain loss to achieve a better efficiency-performance balance.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically explores the VLA design space; contribution to methodology is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 500+ controlled trials + 2 simulation benchmarks + real robots + exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear framework; smooth evolution of design choices.
- Value: ⭐⭐⭐⭐⭐ Sets a precedent for shifting VLA research from fragmented exploration to systematic design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)

</div>

<!-- RELATED:END -->
