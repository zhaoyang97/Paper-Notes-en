---
title: >-
  [Paper Note] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
description: >-
  [NeurIPS 2025][Multimodal VLM][VLA] This paper proposes ForceVLA, which introduces 6-axis force/torque sensing as a first-class modality within the VLA framework. A Force-aware Vision-Language Mixture-of-Experts (FVLMoE) module dynamically fuses visual-language embeddings with real-time force feedback at the action decoding stage, achieving an average success rate improvement of 23.2% across five contact-rich manipulation tasks, with individual tasks reaching up to 80%.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - VLA
  - force feedback
  - MoE
  - contact-rich manipulation
  - robot manipulation
date: 2026-05-08
content_hash: ed26956da61886d0
---

# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation

**Conference**: NeurIPS 2025
**arXiv**: [2505.22159](https://arxiv.org/abs/2505.22159)
**Code**: To be released
**Area**: Multimodal VLM
**Keywords**: VLA, force feedback, MoE, contact-rich manipulation, robot manipulation

## TL;DR
This paper proposes ForceVLA, which introduces 6-axis force/torque sensing as a first-class modality within the VLA framework. A Force-aware Vision-Language Mixture-of-Experts (FVLMoE) module dynamically fuses visual-language embeddings with real-time force feedback at the action decoding stage, achieving an average success rate improvement of 23.2% across five contact-rich manipulation tasks, with individual tasks reaching up to 80%.

## Background & Motivation

**Background**: VLA (Vision-Language-Action) models such as OpenVLA and π₀ have achieved generalizable robot manipulation through large-scale multimodal pretraining, demonstrating strong semantic understanding and zero-shot generalization.

**Limitations of Prior Work**: VLA models rely heavily on visual and language cues while neglecting force perception, which is critical for contact-rich tasks such as insertion, tool use, and assembly. Vision-only policies are prone to failure under occlusion or poor visual conditions. Moreover, force requirements vary across task phases (grasping → insertion → surface contact each demands different force control), yet existing methods lack mechanisms to perceive and adapt to such dynamic changes.

**Key Challenge**: The central challenge is how to effectively integrate force signals into VLA systems so that the model can dynamically adjust manipulation strategies based on haptic feedback, rather than naively concatenating force signals to the input—experiments show that simple concatenation even degrades π₀-fast performance from 31% to 14.2%.

**Goal**: To design a force-aware fusion mechanism that enables VLA models to perform context-aware action generation based on real-time force feedback in contact-rich tasks.

**Key Insight**: 6D external force should be treated as a first-class modality and formally integrated into the action expert module to enable phase-aware action generation driven by force feedback. The key insight is that force signals should be introduced after VLM encoding (preserving pretrained representations) and fused adaptively through a MoE architecture.

**Core Idea**: Employ a Mixture-of-Experts to dynamically fuse force, visual, and language modalities after VLM encoding, enabling action generation that is aware of contact dynamics.

## Method

### Overall Architecture
ForceVLA is built upon the π₀ architecture. Inputs include stereo RGB images (base and wrist cameras), language instructions, proprioception (7D TCP pose + gripper width), and 6-axis force/torque. Visual and language inputs are encoded into context embeddings via SigLIP (PaliGemma) and, together with force signals, are fed into the FVLMoE module for fusion before guiding a flow-based action head to generate action sequences $A_t = \{a_t, \ldots, a_{t+H-1}\}$.

### Key Designs

1. **FVLMoE Input Mapping**:

    - **Function**: Projects force signals into tokens of the same dimensionality as VL embeddings.
    - **Mechanism**: The raw 6-axis force/torque $f_{raw} \in \mathbb{R}^6$ is projected via a linear layer $E_F = \phi_F(f_{raw}) \in \mathbb{R}^{D_{model}}$ to form a force token, which is then concatenated with VL features $E_{VL} \in \mathbb{R}^{N_{VL} \times D_{model}}$ to form $E_{in} = [E_{VL}; E_F] \in \mathbb{R}^{(N_{VL}+1) \times D_{model}}$.
    - **Design Motivation**: Force signals must be introduced after VLM encoding. Ablation experiments demonstrate that introducing force before the VLM ("MoE before VLM") leads to 0% success rate, as it disrupts the pretrained visual-language feature distribution.

2. **Multimodal Routing and Fusion**:

    - **Function**: Achieves adaptive cross-modal fusion via encoder layers and sparse MoE layers.
    - **Mechanism**: $E_{in}$ first passes through a multi-head self-attention encoder layer for global interaction to produce $E_{enc}$, which is then fed into a sparse MoE layer (4 expert MLPs with top-1 routing, i.e., each token activates only one expert). A gating network dynamically routes tokens based on their content; residual connections yield $E_{fused}$, which is then linearly projected to match the action space dimensionality.
    - **Design Motivation**: The sparse activation of MoE naturally allows different experts to handle different modalities and task phases, making it well-suited for the highly heterogeneous fusion of force, visual, and language signals.

3. **Fused Feature Injection into the Action Flow Head**:

    - **Function**: Injects fused features into the flow-based denoising action generator.
    - **Mechanism**: The last $H_{action}$ tokens are extracted from $E_{FVLMoE}$ to form a guidance signal $G_{FVLMoE} \in \mathbb{R}^{H_{action} \times D_a}$, which is element-wise added to the VLM encoding of the current state and noisy action trajectory $S_{suffix}$, and fed into the flow matching denoising process.
    - **Design Motivation**: Element-wise addition allows the force-aware context to directly modulate each timestep of the action sequence, enabling fine-grained force-aware action generation.

### Loss & Training
The training framework follows π₀'s flow matching (rectified flow) paradigm. End-to-end training is conducted on ForceVLA-Data (244 trajectories across 5 tasks, approximately 140,000 synchronized timesteps), with approximately 50 demonstrations per task.

## Key Experimental Results

### Main Results

| Task | π₀-base w/o F | π₀-base w/ F | ForceVLA |
|------|--------------|-------------|----------|
| Insert USB | 30% | 35% | 55% |
| Pump Bottle | 55% | 60% | 70% |
| Insert Plug | 45% | 40% | 80% |
| Wipe Board-1 | 30% | 40% | 40% |
| Peel Cucumber | — | — | — |
| **Average** | **37.3%** | **40.2%** | **60.5%** |

| Model | Avg. Peeling Length (cm) ↑ | Min. Peeling Count ↓ |
|------|-------------------|--------------|
| π₀-base w/o F | 10.27 | 14 |
| π₀-base w/ F | 13.17 | 10 |
| ForceVLA | **14.12** | **7** |

### Ablation Study

| Configuration | Success Rate |
|------|-------|
| π₀ baseline (no force) | 45% |
| Linear before VLM | 55% |
| MoE before VLM | 0% |
| Concatenate after VLM | 60% |
| ForceVLA (FVLMoE after VLM) | **80%** |

### Key Findings
- Simply adding force signals yields only a 2.9 pp improvement (37.3% → 40.2%), whereas FVLMoE fusion achieves a 23.2 pp gain, indicating that *how* to fuse matters more than *whether* force is present.
- Introducing force via MoE before the VLM causes complete failure (0%), validating the necessity of post-fusion—the pretrained VLM feature space must not be disturbed.
- For the π₀-fast architecture, adding force signals actually degrades performance from 31% to 14.2%, as its compact token space is disrupted by unpretrained force tokens.
- In generalization experiments, ForceVLA achieves 90% success rate under visual occlusion, demonstrating that force feedback effectively compensates for missing visual information.

## Highlights & Insights
- **Force as a First-Class Modality**: Rather than naively appending force signals, this work systematically addresses *where* and *how* to fuse them, with ablation experiments constituting a thorough exploration of the design space.
- **MoE for Heterogeneous Modality Fusion**: The sparse routing of MoE naturally assigns different modalities and task phases to specialized experts, better accommodating multimodal heterogeneity than a monolithic MLP fusion.
- **Post-Fusion Design Principle**: The principle of protecting pretrained VLMs from interference by new modalities is transferable to other sensor integration scenarios (e.g., incorporating tactile or acoustic signals into VLA systems).

## Limitations & Future Work
- The method relies on estimated external force rather than direct measurement, which may be insufficient for scenarios requiring extreme tactile sensitivity.
- Training data is limited to 244 trajectories across 5 tasks; generalization to a broader range of task types remains to be verified.
- Experiments rely on the costly Flexiv Rizon robot (with built-in force sensors); adaptation to lower-cost platforms has not been validated.
- Only 6-axis force/torque is considered; richer contact information from tactile sensors (e.g., GelSight) has not been incorporated.

## Related Work & Insights
- **vs. π₀**: ForceVLA directly augments π₀ with force sensing, validating the critical value of force perception for contact-rich tasks and highlighting the lack of haptic awareness as a key limitation of π₀.
- **vs. TLA/Tac-Man**: These tactile methods focus on fingertip tactile sensing, whereas ForceVLA uses a wrist-mounted 6-axis force/torque sensor—the two approaches are complementary and could potentially be combined.
- **vs. ForceMimic**: ForceMimic uses force as an auxiliary input with static fusion; ForceVLA's MoE dynamic routing constitutes the key improvement.

## Rating
- Novelty: ⭐⭐⭐⭐ The direction of integrating force sensing into VLA is valuable, and the FVLMoE design is well-motivated, though the core technique (MoE fusion) is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five real-world tasks, generalization tests, and detailed ablations are provided, though data scale and task diversity could be further enriched.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with well-motivated design choices and convincing ablation experiments.
- Value: ⭐⭐⭐⭐ The work opens a new direction for introducing force perception into VLA systems; the 23.2% improvement carries practical significance.

## Related Work & Insights
- **vs. π₀/π₀.5**: The π₀ framework is powerful but entirely lacks force perception; ForceVLA demonstrates that force is an indispensable modality for contact-rich tasks and reveals the performance ceiling of purely vision-language policies.
- **vs. OpenVLA/OpenVLA-OFT**: These methods focus on vision-language-action alignment and fine-tuning efficiency but do not consider tactile/force signals, limiting their performance in contact-rich scenarios.
- **vs. FORGE**: FORGE investigates the role of force feedback in control but does not integrate it into a VLA framework; ForceVLA is the first to systematically address force fusion within a VLA system.
- **vs. TLA/Tac-Man**: These methods employ fingertip visuotactile sensing, while ForceVLA uses a wrist-mounted 6-axis force/torque sensor—the two capture different physical quantities and could be combined in future work.

## Inspiration & Connections
- The post-fusion design principle (introducing new modalities after VLM encoding) is transferable to any scenario requiring the addition of new sensing modalities to VLA systems (e.g., acoustic, temperature, or vibration sensors).
- The advantages of MoE sparse routing for multimodal fusion merit further validation across broader domains—different experts naturally learn to handle different modalities and task phases.
- The data collection pipeline of ForceVLA-Data (VR teleoperation with synchronized multimodal recording) can serve as a standard paradigm for constructing force-aware datasets.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)

<!-- RELATED:END -->
