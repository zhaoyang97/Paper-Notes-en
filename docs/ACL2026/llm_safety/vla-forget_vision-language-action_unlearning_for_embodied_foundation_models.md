---
title: >-
  [Paper Note] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models
description: >-
  [ACL 2026][LLM Safety][Machine Unlearning] Ours proposes VLA-Forget, the first hybrid unlearning framework for Vision-Language-Action (VLA) models. By employing ratio-aware selective editing for perception/cross-modal la…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "VLA Models"
  - "Embodied AI"
  - "Multimodal Unlearning"
  - "Selective Editing"
date: 2026-05-08
content_hash: 400a31bbe31dd7b3
---

# VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models

**Conference**: ACL 2026  
**arXiv**: [2604.03956](https://arxiv.org/abs/2604.03956)  
**Code**: [GitHub](https://github.com/raviranjan-ai/VLA-Forget)  
**Area**: Multimodal VLM  
**Keywords**: Machine Unlearning, VLA Models, Embodied AI, Multimodal Unlearning, Selective Editing

## TL;DR
Ours proposes VLA-Forget, the first hybrid unlearning framework for Vision-Language-Action (VLA) models. By employing ratio-aware selective editing for perception/cross-modal layers and significance-based selective editing for reasoning/action layers, it achieves target behavior removal while maintaining perception accuracy (+22%) and task success rate (+9%).

## Background & Motivation

**Background**: VLA models (e.g., OpenVLA), as embodied foundation models, directly transform natural language instructions and visual observations into robotic actions. OpenVLA combines DINOv2+SigLIP visual encoders with a Llama 2 backbone to achieve 7-DoF robotic arm control through action token prediction.

**Limitations of Prior Work**: Deployed VLA policies may retain unsafe behaviors, privacy-sensitive content, or spurious shortcuts. Errors in robotics translate into physical actions, leading to consequences far more severe than those in text/image models. Existing unlearning methods (e.g., SSD, SalUn) are designed for single modalities and cannot handle the distributed encoding of undesirable behaviors across perception, alignment, and action layers in VLA.

**Key Challenge**: Undesirable behaviors in VLA models may be simultaneously encoded in visual features $\theta_V$, cross-modal projections $\theta_P$, and action priors $\theta_L$. Editing only visual layers may leave action priors intact, while editing only language layers may preserve harmful perceptual shortcuts.

**Goal**: Design a component-aware unlearning framework to simultaneously optimize three objectives: efficacy (target unlearning), specificity (perception maintenance), and utility (reasoning retention).

**Key Insight**: Decompose VLA unlearning into three stages—perception unlearning, cross-modal unlearning, and reasoning/action unlearning—using different layer selection strategies for each stage.

**Core Idea**: Ratio-aware score selection identifies perception layers that significantly impact unlearning but have minimal gradient conflict with retention; significance ratio selection identifies reasoning layers crucial for unlearning; phased adapter updates ensure roll-back capability.

## Method

### Overall Architecture
A three-stage hierarchical unlearning pipeline: (1) Visual encoder stage to remove visual triggers, (2) Projector stage to disconnect erroneous vision-language bindings, and (3) Upper Transformer stage to suppress instruction-conditioned action priors. Parameter-efficient updates are implemented using LoRA adapters to support rollback and canary deployment. PCGrad is utilized to stabilize multi-objective gradient conflicts.

### Key Designs

1.  **Ratio-Aware Selective Editing (Perception/Projector Layers)**:

    - **Function**: Select visual/projector layers that contribute significantly to unlearning with minimal conflict with retention tasks.
    - **Mechanism**: For each layer $l$, the unlearning and retention gradients $g_l^f, g_l^r$ are calculated. A score is assigned as $\phi(l) = \frac{\|g_l^f\|_2}{\|\theta_l\|_2 + \epsilon} \cdot (1 - \cos(g_l^f, g_l^r))^\alpha$. The top-K layers with the highest scores are selected for updates. A large gradient norm indicates importance for unlearning, while low cosine similarity indicates that unlearning will not interfere with retention.
    - **Design Motivation**: Avoid collateral damage caused by global editing and precisely locate perception parameters encoding undesirable behaviors.

2.  **Significance-Based Reasoning/Action Layer Selection**:

    - **Function**: Minimize the update set while ensuring sufficient unlearning.
    - **Mechanism**: For upper Transformer blocks, calculate $Sig(l) = \frac{\|\nabla_{\theta_l} L_{forget}\|_2}{\|\nabla_{\theta_l} L_{retain}\|_2 + \epsilon}$. Initialize the top-k layers for editing and iteratively expand if unlearning is insufficient.
    - **Design Motivation**: Action priors are distributed across multiple Transformer layers; a progressive expansion strategy balances sufficient unlearning with minimal interference.

3.  **Triple-Objective Optimization + PCGrad Stabilization**:

    - **Function**: Simultaneously achieve unlearning, retention, and prevention of shallow unlearning.
    - **Mechanism**: Define a unified objective $\min_\theta L_{retain} + \lambda_{feat} L_{feat} - \lambda_f L_{forget} - \lambda_m L_{mismatch}$. $L_{forget}$ (gradient ascent) suppresses target behaviors, $L_{retain}$ (CE + KL anchoring) maintains non-target behaviors, and $L_{mismatch}$ (KL divergence) pushes away the original unlearning response to prevent recovery. $L_{feat}$ distills visual and projector representations to maintain non-target visual grounding. PCGrad resolves conflicts between retention and unlearning gradients.
    - **Design Motivation**: Pure gradient ascent leads to total performance collapse; multi-objective constraints ensure unlearning is precise and controllable.

### Loss & Training
LoRA adapters are updated in phases (Visual → Projector → Reasoning/Action). Unlearning efficacy is evaluated after each phase to decide whether to expand update layers. PCGrad gradient projection addresses multi-objective conflicts. Post-training evaluation assesses the risk of recovery via quantization.

## Key Experimental Results

### Main Results

| Method | FC↑ | RC↑ | FAD↑ | RAD↓ | TSR↑ | SVR↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SSD | 78 | 83 | 0.70 | 0.28 | 68 | 17 |
| SalUn | 89 | 88 | 0.76 | 0.26 | 71 | 12 |
| GA | 93 | 60 | 0.89 | 0.45 | 40 | 5 |
| NPO | 90 | 88 | 0.83 | 0.23 | 74 | 8 |
| **Ours** | **93** | **91** | **0.88** | **0.21** | **78** | **5** |

### Ablation Study

| Configuration | FC↑ | RC↑ | TSR↑ | Description |
| :--- | :--- | :--- | :--- | :--- |
| VLA-Forget (Full) | 93 | 91 | 78 | Full three-stage pipeline |
| Vision Unlearning Only | ~85 | ~87 | ~70 | Fails to remove residual behaviors in action priors |
| Language Unlearning Only (GA) | 93 | 60 | 40 | Unlearning effective but retention collapses severely |
| Without PCGrad | - | - | - | Gradient conflicts lead to unstable training |

### Key Findings
- Unlearning efficacy improved by 10%, perception specificity maintenance increased by 22%, reasoning retention increased by 9%, and post-quantization recovery rate decreased by 55%.
- GA (pure gradient ascent) achieves the most thorough unlearning (FC=93) but suffers from retention collapse (RC=60, TSR=40), proving that global editing is unviable in VLA.
- The three-stage hierarchical design is critical—editing only visual layers cannot remove residual behaviors within action priors.
- Post-quantization recovery (SVR) is a practical threat in VLA deployment; the mismatch loss in VLA-Forget effectively reduces this recovery risk.

## Highlights & Insights
- Introduces the machine unlearning problem to VLA embodied models for the first time, revealing unique challenges of distributed encoding for undesirable behaviors across components in multimodal action models. This is significantly more complex than text/image unlearning as it requires evaluating physical execution rather than just output correctness.
- The design of Ratio-aware layer selection is practical—it considers both unlearning importance and retention interference, offering higher precision than top-k gradient magnitude selection.
- The adapter-first design makes unlearning reversible, fitting safety audit workflows in real-world deployments.

## Limitations & Future Work
- As an approximate unlearning method, it does not provide certified erasure guarantees.
- Only validated on OpenVLA-7B and pi0fast-base; larger-scale VLA models remain to be tested.
- Hyperparameters for unlearning-retention ($\lambda_f, \lambda_m, \lambda_{feat}$) require tuning for different scenarios.
- Evaluation was primarily conducted in simulated environments; real-world robot deployment validation is still needed.
- Future work could explore multi-turn interactive unlearning and the integration of continual learning with unlearning.

## Related Work & Insights
- **vs SSD/SalUn**: These are vision-side unlearning methods that cannot handle cross-modal distributed behaviors in VLA.
- **vs GA/NPO**: These are language-side unlearning methods; GA is too aggressive leading to retention collapse, while NPO is gentler but remains component-agnostic.
- **vs SCRUB**: Improves the unlearning-retention trade-off but does not address multimodal entanglement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce machine unlearning to VLA models; both problem definition and method design are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient multi-baseline comparisons and ablations, though real-robot evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear explanation of the method; the three-stage pipeline is logically structured.
- Value: ⭐⭐⭐⭐ With the increasing deployment of VLA models, safe unlearning will become a critical requirement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](../../CVPR2026/llm_safety/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](../../ICML2026/llm_safety/forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)

</div>

<!-- RELATED:END -->
