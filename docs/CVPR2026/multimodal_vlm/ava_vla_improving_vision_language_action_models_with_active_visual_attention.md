---
title: >-
  [Paper Note] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
description: >-
  [CVPR 2026][Multimodal VLM][VLA models] This work revisits visual processing in VLA models from a POMDP perspective and proposes the AVA-VLA framework, which dynamically modulates the importance of visual tokens in the current frame based on historical context via a recurrent state and an active visual attention module, achieving state-of-the-art performance on benchmarks including LIBERO and CALVIN.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLA models
  - active visual attention
  - POMDP
  - recurrent state
  - visual token modulation
date: 2026-05-08
content_hash: e5e5e61b4e763c4c
---

# AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention

**Conference**: CVPR 2026
**arXiv**: [2511.18960](https://arxiv.org/abs/2511.18960)
**Code**: [Project Page](https://liauto-dsr.github.io/AVA-VLA-Page)
**Area**: Multimodal VLM
**Keywords**: VLA models, active visual attention, POMDP, recurrent state, visual token modulation

## TL;DR

This work revisits visual processing in VLA models from a POMDP perspective and proposes the AVA-VLA framework, which dynamically modulates the importance of visual tokens in the current frame based on historical context via a recurrent state and an active visual attention module, achieving state-of-the-art performance on benchmarks including LIBERO and CALVIN.

## Background & Motivation

Vision-Language-Action (VLA) models have shown remarkable progress in robotic manipulation tasks; however, most existing approaches process visual observations independently at each timestep, implicitly modeling robot manipulation as a Markov Decision Process (MDP). This history-free design carries fundamental limitations:

1. Real-world robot control is inherently partially observable (POMDP), and the current frame alone cannot fully characterize the environment state.
2. Visual attention is guided solely by static language instructions, making it unable to suppress temporally redundant information based on historical actions.
3. The model cannot anticipate "what to attend to next," rendering the visual system passive rather than active.

For example, in the task "turn on the stove and place the moka pot on it," vanilla OpenVLA-OFT fails to localize the task-critical "stove knob," whereas AVA-VLA stably focuses on it by leveraging historical context.

## Method

### Overall Architecture

Current observation + previous recurrent state → AVA module computes soft weights for visual tokens → modulates attention matrices across LLM backbone layers → recurrent state initializes action placeholders → parallel action chunk decoding → output actions + update recurrent state.

### Key Designs

1. **Recurrent State**:
   - Function: Serves as a neural approximation of the belief state in a POMDP, encoding historical context.
   - Mechanism: Projected via an MLP from action-relevant hidden states in the last layer of the LLM at the previous timestep; also used to initialize action placeholders at the current step.
   - Design Motivation: Direct computation of the theoretical belief state is intractable; a compressed recurrent representation is used as an approximation.

2. **Active Visual Attention (AVA) Module**:
   - Function: Dynamically modulates the importance of visual tokens based on historical information.
   - Mechanism: Language instruction features are first used to condition visual features via FiLM; cross-attention (with visual tokens as Query and the recurrent state as Key/Value) followed by self-attention is then applied, ultimately producing a soft weight (a weighted score after binary enhance/suppress classification) for each visual token.
   - Design Motivation: Transforms the visual system from "passively processing whatever is observed" to "actively focusing on critical regions based on historical experience."

3. **Soft Attention Matrix Modulation**:
   - Function: Applies the soft weights output by AVA to attention computations across all layers of the LLM backbone.
   - Mechanism: A soft attention matrix $U$ is constructed to impose weights on visual token positions and is multiplied with attention scores prior to Softmax.
   - Design Motivation: Layer-shared weights ensure consistent visual focus without altering the fundamental architecture of the LLM backbone.

### Loss & Training

- Action prediction MAE loss + L2 regularization (constraining the mean of soft weights toward a target value $c$ to prevent excessive dispersion).
- Truncated backpropagation through time ($T=4$ steps), balancing computational feasibility with temporal dynamics learning.
- The recurrent state is initialized as a zero vector and reset at the beginning of each episode.

## Key Experimental Results

### Main Results

| Benchmark | Metric | AVA-VLA | OpenVLA-OFT | Gain |
|-----------|--------|---------|-------------|------|
| LIBERO (all 4 suites) | Avg. SR | 98.0% | 96.8% | +1.2% |
| LIBERO-Long | SR | 97.6% | 95.3% | +2.3% |
| CALVIN ABC→D | Avg. Length | 4.65 | 4.28 | +0.37 |
| Real Robot | Avg. SR | Highest | 2nd | Multi-task gain |

### Ablation Study

| Configuration | LIBERO Avg. SR | Note |
|---------------|---------------|------|
| OpenVLA-OFT baseline | 96.8% | No historical information |
| + State initialization | 97.5% | Recurrent state injected into action placeholder |
| + AVA module | 97.5% | Visual token reweighting |
| + Both combined | 98.0% | Complementary effect |

### Key Findings

- Visual token pruning experiment: pruning 70% of visual tokens still yields performance exceeding the OpenVLA-OFT baseline (97.3 vs. 96.8), validating that the AVA module effectively identifies critical regions.
- Cross-backbone experiments: consistent improvements are observed on OpenVLA-7B, LLaMA2-7B, and Qwen2.5-0.5B, demonstrating strong generalizability.
- Visualizations show that AVA weights consistently focus on robot contact regions and target objects.

## Highlights & Insights

- The POMDP theoretical perspective provides an elegant theoretical foundation for historical modeling in VLA models.
- The AVA module is lightweight and plug-and-play, requiring no modifications to the LLM backbone architecture.
- A byproduct of the soft weights—potential for visual token pruning—offers a direction for efficiency optimization in VLA models.
- The most significant improvements are observed on the most challenging long-horizon tasks: LIBERO-Long and CALVIN.

## Limitations & Future Work

- Truncated backpropagation ($T=4$) limits learning of long-range temporal dependencies.
- The recurrent state is derived from only the previous timestep; longer memory windows remain unexplored.
- Soft weights modulate only the attention matrix and do not directly modify visual feature representations.
- Real-robot experiments involve relatively limited demonstration data (30–450 demonstrations).

## Related Work & Insights

- **vs. OpenVLA/UniVLA**: These models decode actions autoregressively without historical modeling; AVA-VLA retains temporal context via the recurrent state.
- **vs. CoT-VLA**: Uses chain-of-thought reasoning but does not explicitly model the temporal dynamics of visual attention.
- **vs. SP-VLA/FLOWER**: Focus on efficiency-oriented visual token pruning but do not perform active focusing based on historical context.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of the POMDP perspective and active visual attention is novel in the VLA domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full coverage of LIBERO/CALVIN/real-robot settings with comprehensive ablation, visualization, and pruning analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, concise theoretical derivation, and well-organized experimental presentation.
- Value: ⭐⭐⭐⭐ Introduces a temporally-aware visual processing paradigm for VLA models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](../../ICCV2025/multimodal_vlm/coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/multimodal_vlm/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)
- [\[AAAI 2026\] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](../../AAAI2026/multimodal_vlm/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)

<!-- RELATED:END -->
