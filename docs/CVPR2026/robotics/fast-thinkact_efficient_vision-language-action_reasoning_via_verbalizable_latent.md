---
title: >-
  [Paper Note] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning
description: >-
  [CVPR2026][Robotics][VLA] This paper proposes Fast-ThinkAct, which compresses verbose textual CoT reasoning (~250 tokens) into 6 verbalizable continuous latent tokens…
tags:
  - "CVPR2026"
  - "Robotics"
  - "VLA"
  - "reasoning"
  - "latent CoT"
  - "knowledge distillation"
  - "preference learning"
  - "robot manipulation"
date: 2026-05-08
content_hash: c7b64c233335aeac
---

# Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning

**Conference**: CVPR2026
**arXiv**: [2601.09708](https://arxiv.org/abs/2601.09708)
**Code**: [Project Page](https://jasper0314-huang.github.io/fast-thinkact/)
**Area**: Robotics
**Keywords**: VLA, reasoning, latent CoT, knowledge distillation, preference learning, robot manipulation

## TL;DR

This paper proposes Fast-ThinkAct, which compresses verbose textual CoT reasoning (~250 tokens) into 6 verbalizable continuous latent tokens, combined with reward-guided preference distillation and visual trajectory alignment, achieving an 89.3% reduction in inference latency (9.3× faster than ThinkAct-7B) while maintaining or surpassing the performance of state-of-the-art reasoning VLAs.

## Background & Motivation

Vision-Language-Action (VLA) tasks require agents to reason over complex visual scenes and execute adaptive actions. Recent VLA models are primarily trained via large-scale robot demonstrations with supervised learning, achieving strong performance on basic skills (e.g., pick-and-place), but exhibiting insufficient generalization in the following areas:

1. **Long-horizon planning**: Complex tasks requiring multi-step reasoning (e.g., turning on a stove before placing a pan)
2. **Failure recovery**: Detecting failures at runtime and generating corrective plans
3. **Few-shot adaptation**: Rapidly adapting to novel scenes and tasks

Reasoning VLAs (e.g., ThinkAct, CoT-VLA, MolmoAct) improve generalization by incorporating explicit chain-of-thought reasoning. However, generating lengthy reasoning chains introduces severe inference latency bottlenecks:

- ThinkAct-7B requires approximately 7.5 seconds per step (~0.1 Hz)
- Robot manipulation requires real-time decision frequencies of 1–15 Hz
- ECoT-Lite attempts to accelerate via reasoning dropout, but directly truncating textual reasoning discards critical information, leading to performance degradation

Core motivation: **How can verbose textual CoT be compressed into a compact representation while preserving reasoning capability and correctly capturing spatiotemporal dynamics?**

## Core Problem

- Textual CoT reasoning generates long sequences (~250 tokens), with inference latency reaching several seconds, rendering real-time manipulation infeasible
- Latent reasoning methods from the LLM domain (e.g., Coconut, CODI) cannot be directly transferred to VLA tasks, which require spatiotemporal understanding and must bridge semantic reasoning with embodied control
- After compressing reasoning into the continuous latent space, direct supervision signals guiding what the latent should encode are absent

## Method

### Overall Architecture

Fast-ThinkAct comprises three core stages:

1. **Reward-Guided Preference Distillation**: The teacher's GRPO reward signal guides the student to learn high-quality latent reasoning
2. **Visual Trajectory Alignment**: Aligns trajectory-level representations between teacher and student to transfer visual planning capability
3. **Reasoning-Enhanced Policy Learning**: Freezes the student VLM and augments the action model with latent reasoning features for action generation

### Stage 1: Verbalizable Latent CoT by Reward Preferences

**Teacher training**: The textual teacher VLM $\mathcal{F}_{\theta^T}$ is trained via GRPO from a CoT-SFT checkpoint using action-aligned visual rewards to generate explicit textual reasoning chains. The GRPO advantage function $A(\tau)$ naturally serves as an indicator of reasoning quality.

**Constructing preference pairs**: The highest- and lowest-advantage reasoning chains are selected from each rollout group as positive and negative samples:

$$\tau^+ = \arg\max_{\tau \in G} A(\tau), \quad \tau^- = \arg\min_{\tau \in G} A(\tau)$$

**Student learning**: The student VLM $\mathcal{F}_\theta$ does not generate text tokens; instead, it autoregressively generates $M=6$ continuous latent vectors $\mathbf{z} = \{z_m\}_{m=1}^M$, where $z_m \in \mathbb{R}^d$.

**Verbalizer**: A verbalizer LLM $\mathcal{V}_\psi$ (Qwen3-0.6B with inserted cross-attention layers) is introduced to decode latents into natural language. The training objective encourages the verbalizer to assign higher likelihood to high-quality reasoning $\tau^+$:

$$\mathcal{L}_{\text{verb}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{p_\psi(\tau^+|\mathbf{z})}{p_{\text{ref}}(\tau^+)} - \log \frac{p_\psi(\tau^-|\mathbf{z})}{p_{\text{ref}}(\tau^-)}\right)\right)\right]$$

This is a DPO-style objective where $\beta=0.1$ controls preference strength. Through this formulation, the student is guided to encode latents that the verbalizer can decode into high-quality reasoning.

### Stage 2: Action-Aligned Visual Plan Distillation

The hidden states of the teacher and student at the `<answer>` token are aligned to transfer trajectory-level visual planning capability:

$$\mathcal{L}_{\text{distill}} = \|h_t^T - h_t\|_2^2$$

Additionally, $K=5$ learnable spatial tokens $\{s_i\}_{i=1}^K$ are appended after the latent reasoning sequence. The output hidden state of each spatial token is projected in parallel through an MLP to a waypoint $p_i \in \mathbb{R}^6$ (format $[x_{\text{single}}, y_{\text{single}}, x_{\text{left}}, y_{\text{left}}, x_{\text{right}}, y_{\text{right}}]$), replacing the teacher's autoregressive generation of 60–70 waypoint text tokens.

The overall training objective is: $\mathcal{L}_{\text{student}} = \mathcal{L}_{\text{verb}} + \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{ans}}$

### Stage 3: Reasoning-Enhanced Policy Learning

The student VLM $\mathcal{F}_\theta$ is frozen, and visual latent planning features $c_t$ are extracted from the **early-layer** KV cache of the spatial tokens, then injected into a diffusion Transformer action model $\pi_\phi$ (DiT-Policy or RDT) via cross-attention:

$$\mathcal{L}_{\text{IL}}(\phi) = \ell(\pi_\phi(o_t, l, c_t), \hat{a}_t)$$

Ablations confirm that early-layer KV outperforms late-layer KV in capturing visual planning information (LIBERO: 89.7 vs. 88.3 vs. 87.1).

### Loss & Training

- VLM backbone: Qwen2.5-VL 3B
- Training pipeline: SFT → CoT-SFT → Teacher GRPO + Student distillation (4,500 iterations)
- Verbalizer warmup: 3,000 iterations (LM loss), then switched to $\mathcal{L}_{\text{verb}}$ for 1,500 iterations
- Policy learning: 20K iterations with frozen VLM and state encoder
- At inference, only $\mathcal{F}_\theta + \pi_\phi$ are required; the verbalizer is used only during training and for interpretability

## Key Experimental Results

### LIBERO & SimplerEnv (Robot Manipulation)

| Method | LIBERO (avg) | SimplerEnv-Google | Inference Latency (ms) |
|--------|-------------|-------------------|------------------------|
| OpenVLA-7B | 76.5 | 40.2 | N/A |
| ThinkAct-7B | 84.4 | 68.3 | 7513 |
| MolmoAct-7B | 86.8 | 64.9 | 6723 |
| ThinkAct-3B | 83.1 | 64.7 | 5674 |
| **Fast-ThinkAct-3B** | **89.7** | **68.7** | **805** (↓7.0×) |

Fast-ThinkAct-3B surpasses ThinkAct-3B by 6.6% on LIBERO, by 4.0% on SimplerEnv, and achieves a 7× latency reduction.

### RoboTwin2.0 (Bimanual Manipulation)

| Method | Easy Avg | Hard Avg |
|--------|----------|----------|
| RDT | 56.4 | 22.8 |
| ThinkAct | 62.4 | 24.7 |
| **Fast-ThinkAct** | **65.7** | **26.4** |

Advantages are more pronounced on long-horizon tasks (270+ steps).

### Embodied Reasoning

| Method | EgoPlan-Bench2 | RoboVQA (B-Avg) | OpenEQA | Overall |
|--------|---------------|-----------------|---------|---------|
| ThinkAct-3B | 44.0 | 55.3 | 48.9 | 49.4 |
| **Fast-ThinkAct-3B** | **46.4** | **60.8** | **51.2** | **52.8** |

Surpasses commercial models including GPT-4V (36.4) and Gemini-2.5-Flash (38.9).

### Ablation Study

- Removing $\mathcal{L}_{\text{verb}}$: Overall drops from 52.8 to 48.5 (−4.3), indicating the necessity of preference-guided supervision
- Removing $\mathcal{L}_{\text{distill}}$: Further drops to 47.7, confirming the importance of visual planning transfer
- Comparison with efficient textual reasoning: teacher direct reasoning 49.8, 6 text tokens 46.3, RL length-penalty 47.8, **Fast-ThinkAct with 6 latent tokens 53.3**
- Latent token count ablation: $M=1$ is insufficient; $M=30/100$ introduces noise; $M=6$ is optimal

## Highlights & Insights

- **Elegant verbalizable latent design**: Latents can be decoded into text via the verbalizer, achieving both compression and interpretability while fundamentally addressing the lack of direct supervision in the latent space
- **Reward-guided preference distillation**: Reuses the teacher GRPO reward signal to construct DPO preference pairs without additional annotation, yielding highly efficient training signals
- **Dramatic latency reduction**: Parallel prediction with 6 latent tokens and 5 spatial tokens achieves an 89.3% latency reduction, transforming an unusable system (0.1 Hz) into a real-time capable one
- **Strong failure recovery**: Outperforms the second-best method by 10.9–16.4 points on RoboFAC, demonstrating that latent reasoning preserves the ability to understand errors and plan corrections

## Limitations & Future Work

- The verbalizer inherits hallucination tendencies from the pretrained LLM — verbalized reasoning may produce plausible-sounding but inaccurate descriptions (without affecting action inference)
- Evaluation is conducted exclusively in simulated environments; real-robot deployment results are not demonstrated
- Only a 3B VLM backbone is used for the student; ablations on a 7B version are insufficient (evaluated only on reasoning benchmarks, not comprehensively validated on manipulation tasks)
- The number of spatial tokens is fixed at $K=5$; adaptive token counts are not explored
- The training pipeline is complex (SFT → CoT-SFT → Teacher GRPO → Student distillation → Policy learning), leaving substantial room for end-to-end simplification

## Related Work & Insights

| Dimension | ThinkAct | MolmoAct | CoT-VLA | ECoT-Lite | Fast-ThinkAct |
|-----------|----------|----------|---------|-----------|---------------|
| Reasoning form | Textual CoT | 2D visual trace | Visual goal + text | Reasoning dropout | Latent CoT |
| Reasoning length | ~250 tokens | ~250 tokens | — | Variable | 6 latent tokens |
| Inference latency | 7.5s (7B) | 6.7s (7B) | — | Reduced but unstable | **0.8s (3B)** |
| RL training | GRPO | None | None | None | Teacher GRPO → DPO distill |
| Interpretability | High (text) | High (visual) | Medium | Low | Medium (optional verbalization) |

Core distinction: Fast-ThinkAct migrates reasoning from token space to continuous latent space and replaces direct distillation with preference learning, achieving an effective balance between efficiency and quality.

Beyond the direct comparisons above, the verbalizable latent paradigm generalizes to real-time reasoning scenarios such as autonomous driving — any task requiring CoT capability under latency constraints. The reward-guided distillation pipeline (Teacher GRPO → Student DPO) circumvents the annotation challenge in latent space and is transferable to other latent reasoning works. The finding that early-layer KV cache outperforms late-layer KV implies that visual planning information is encoded in the shallower layers of VLMs, intersecting with the VLM probing literature. This work complements LLM latent reasoning approaches such as Coconut and CODI, representing the first extension of latent reasoning to the VLA domain.

## Rating

- **Novelty**: 8/10 — The combination of verbalizable latents and reward preference distillation is a novel design that addresses the key challenge of supervision signals in latent reasoning
- **Experimental Thoroughness**: 9/10 — Six benchmarks (3 reasoning + 3 manipulation), comprehensive ablations, and detailed latency analysis
- **Writing Quality**: 8/10 — Clear structure, complete mathematical formulation, and intuitive illustrations
- **Value**: 9/10 — Reducing inference latency from seconds to sub-second while improving performance resolves a critical bottleneck for deploying reasoning VLAs in practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](../../NeurIPS2025/robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](../../ICLR2026/robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)

</div>

<!-- RELATED:END -->
