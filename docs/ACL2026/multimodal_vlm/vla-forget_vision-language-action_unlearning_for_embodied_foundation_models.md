---
title: >-
  [Paper Note] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models
description: >-
  [ACL 2026][Multimodal VLM][Machine Unlearning] This paper proposes VLA-Forget, the first hybrid unlearning framework for vision-language-action (VLA) models. It employs ratio-aware selective editing for perception/cross-modal layers and significance-based selective editing for reasoning/action layers, achieving targeted behavior removal while improving perceptual specificity (+22%) and task success rate (+9%).
tags:
  - ACL 2026
  - Multimodal VLM
  - Machine Unlearning
  - VLA Models
  - Embodied AI
  - Multimodal Unlearning
  - Selective Editing
date: 2026-05-08
content_hash: 6211729dd26ad3db
---

# VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models

**Conference**: ACL 2026
**arXiv**: [2604.03956](https://arxiv.org/abs/2604.03956)
**Code**: [GitHub](https://github.com/raviranjan-ai/VLA-Forget)
**Area**: Multimodal VLM
**Keywords**: Machine Unlearning, VLA Models, Embodied AI, Multimodal Unlearning, Selective Editing

## TL;DR
This paper proposes VLA-Forget, the first hybrid unlearning framework for vision-language-action (VLA) models. It employs ratio-aware selective editing for perception/cross-modal layers and significance-based selective editing for reasoning/action layers, achieving targeted behavior removal while improving perceptual specificity (+22%) and task success rate (+9%).

## Background & Motivation

**Background**: VLA models (e.g., OpenVLA), as embodied foundation models, directly map natural language instructions and visual observations to robot actions. OpenVLA integrates DINOv2+SigLIP visual encoders with a Llama 2 backbone, enabling 7-DoF robotic arm control via action token prediction.

**Limitations of Prior Work**: Deployed VLA policies may retain unsafe behaviors, privacy-sensitive content, or spurious shortcuts. Errors in robotic systems translate into physical actions, making the consequences far more severe than in text or image models. Existing unlearning methods (e.g., SSD, SalUn) are designed for unimodal settings and cannot address the distributed encoding of undesirable behaviors across perception, alignment, and action layers in VLA models.

**Key Challenge**: Undesirable behaviors in VLA models may be simultaneously encoded in visual features $\theta_V$, cross-modal projections $\theta_P$, and action priors $\theta_L$. Editing only the visual layers may leave action priors intact, while editing only the language layers may preserve harmful perceptual shortcuts.

**Goal**: Design a component-aware unlearning framework that jointly optimizes three objectives: forgetting efficacy, perceptual specificity, and reasoning utility.

**Key Insight**: Decompose VLA unlearning into three stages—perceptual unlearning, cross-modal unlearning, and reasoning/action unlearning—each employing a distinct layer selection strategy.

**Core Idea**: Ratio-aware scoring selects perception layers with high forgetting influence and low conflict with retention gradients; significance ratio selects reasoning layers critical to forgetting; staged adapter updates ensure rollback capability.

## Method

### Overall Architecture
A three-stage hierarchical unlearning pipeline: (1) the visual encoder stage removes visual triggers; (2) the projector stage severs erroneous vision-language bindings; (3) the upper Transformer stage suppresses instruction-conditioned action priors. LoRA adapters enable parameter-efficient updates with rollback support and canary deployment. PCGrad stabilizes multi-objective gradient conflicts.

### Key Designs

1. **Ratio-Aware Selective Editing (Perception/Projector Layers)**:

    - **Function**: Selects visual/projector layers with high forgetting contribution and low conflict with retention tasks.
    - **Mechanism**: For each layer $l$, forgetting and retention gradients $g_l^f, g_l^r$ are computed, and a score $\phi(l) = \frac{\|g_l^f\|_2}{\|\theta_l\|_2 + \epsilon} \cdot (1 - \cos(g_l^f, g_l^r))^\alpha$ is assigned. The top-K layers by score are selected for update. A large gradient norm indicates the layer is important for forgetting; low cosine similarity indicates forgetting will not interfere with retention.
    - **Design Motivation**: Avoids collateral damage from global editing and precisely locates perceptual parameters encoding undesirable behaviors.

2. **Significance-Based Reasoning/Action Layer Selection**:

    - **Function**: Ensures sufficient unlearning while minimizing the set of updated parameters.
    - **Mechanism**: For upper Transformer blocks, compute $Sig(l) = \frac{\|\nabla_{\theta_l} L_{forget}\|_2}{\|\nabla_{\theta_l} L_{retain}\|_2 + \epsilon}$; initialize editing on top-k layers and iteratively expand if forgetting is insufficient.
    - **Design Motivation**: Action priors are distributed across multiple Transformer layers; the progressive expansion strategy balances sufficient forgetting with minimal interference.

3. **Triple Optimization Objective + PCGrad Stabilization**:

    - **Function**: Simultaneously achieves forgetting, retention, and prevention of shallow unlearning.
    - **Mechanism**: Unified objective: $\min_\theta L_{retain} + \lambda_{feat} L_{feat} - \lambda_f L_{forget} - \lambda_m L_{mismatch}$. $L_{forget}$ (gradient ascent) suppresses target behaviors; $L_{retain}$ (CE + KL anchoring) preserves non-target behaviors; $L_{mismatch}$ (KL divergence) pushes responses away from original forgetting outputs to prevent recovery. $L_{feat}$ distills visual and projector representations to maintain non-target visual grounding. PCGrad resolves gradient conflicts between retention and forgetting.
    - **Design Motivation**: Naive gradient ascent causes global performance collapse; multi-objective constraints ensure forgetting is precise and controllable.

### Loss & Training
LoRA adapters are updated in staged order (visual → projector → reasoning/action). After each stage, unlearning effectiveness is evaluated to determine whether to expand the set of updated layers. PCGrad gradient projection resolves multi-objective conflicts. Post-quantization recovery risk is assessed upon training completion.

## Key Experimental Results

### Main Results

| Method | FC↑ | RC↑ | FAD↑ | RAD↓ | TSR↑ | SVR↓ |
|--------|-----|-----|------|------|------|------|
| SSD | 78 | 83 | 0.70 | 0.28 | 68 | 17 |
| SalUn | 89 | 88 | 0.76 | 0.26 | 71 | 12 |
| GA | 93 | 60 | 0.89 | 0.45 | 40 | 5 |
| NPO | 90 | 88 | 0.83 | 0.23 | 74 | 8 |
| VLA-Forget | **93** | **91** | **0.88** | **0.21** | **78** | **5** |

### Ablation Study

| Configuration | FC↑ | RC↑ | TSR↑ | Notes |
|---------------|-----|-----|------|-------|
| VLA-Forget (full) | 93 | 91 | 78 | Complete three-stage pipeline |
| Visual unlearning only | ~85 | ~87 | ~70 | Residual behaviors in action priors not removed |
| Language unlearning only (GA) | 93 | 60 | 40 | Effective forgetting but severe retention collapse |
| Without PCGrad | — | — | — | Training instability due to gradient conflicts |

### Key Findings
- Forgetting efficacy improves by 10%, perceptual specificity by 22%, reasoning utility by 9%, and post-quantization recovery rate decreases by 55%.
- GA (pure gradient ascent) achieves the most thorough forgetting (FC=93) but suffers from retention collapse (RC=60, TSR=40), demonstrating the infeasibility of global editing in VLA models.
- The three-stage hierarchical design is critical—editing only the visual layers fails to remove residual behaviors encoded in action priors.
- Post-quantization recovery (SVR) represents a practical deployment threat; VLA-Forget's mismatch loss effectively reduces recovery risk.

## Highlights & Insights
- This is the first work to introduce machine unlearning into VLA embodied models, revealing the unique challenge of undesirable behaviors being distributed across multiple components in multimodal action models. This is substantially more complex than text/image unlearning, as evaluation requires assessing physical execution rather than merely output correctness.
- The ratio-aware layer selection is practically well-motivated—jointly considering forgetting importance and retention interference yields more precise selection than top-k gradient magnitude alone.
- The adapter-first design makes unlearning rollback-capable, suited for safety auditing workflows in real-world deployment.

## Limitations & Future Work
- As an approximate unlearning method, no certified erasure guarantees are provided.
- Validation is conducted only on OpenVLA-7B and pi0fast-base; larger-scale VLA models remain to be tested.
- Hyperparameters governing the forgetting-retention trade-off ($\lambda_f, \lambda_m, \lambda_{feat}$) require tuning for different scenarios.
- Evaluation is primarily conducted in simulated environments; real-robot deployment validation is absent.
- Future work may explore multi-round interactive unlearning and the integration of continual learning with unlearning.

## Related Work & Insights
- **vs. SSD/SalUn**: These are visual-side unlearning methods that cannot handle undesirable behaviors distributed across modalities in VLA models.
- **vs. GA/NPO**: These are language-side unlearning methods; GA is overly aggressive and causes retention collapse, while NPO is more conservative but still lacks component awareness.
- **vs. SCRUB**: Improves the forgetting-retention trade-off but does not address multimodal entanglement.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce machine unlearning into VLA models; both problem formulation and method design are original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive baseline comparisons and ablations; real-robot evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Method is clearly articulated; the three-stage pipeline is logically coherent.
- **Value**: ⭐⭐⭐⭐ — As VLA model deployment scales, safe unlearning will become an essential capability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](../../CVPR2026/multimodal_vlm/ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)
- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](../../CVPR2026/multimodal_vlm/hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[AAAI 2026\] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](../../AAAI2026/multimodal_vlm/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](../../CVPR2026/multimodal_vlm/from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)

<!-- RELATED:END -->
