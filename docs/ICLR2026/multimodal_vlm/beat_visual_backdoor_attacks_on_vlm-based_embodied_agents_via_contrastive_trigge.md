---
title: >-
  [Paper Note] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning
description: >-
  [ICLR 2026][Multimodal VLM][backdoor attack] This paper proposes BEAT, the first visual backdoor attack framework targeting VLM-driven embodied agents. It employs environmental objects (e.g., knives) as triggers and adopts a two-stage training pipeline (SFT + Contrastive Trigger Learning) to achieve precise backdoor activation. BEAT attains an attack success rate of up to 80% while preserving normal task performance, exposing critical security vulnerabilities in VLM-based embodied agents.
tags:
  - ICLR 2026
  - Multimodal VLM
  - backdoor attack
  - embodied agent
  - VLM security
  - contrastive learning
  - visual trigger
date: 2026-05-08
content_hash: f35a437e03bf7fd8
---

# BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning

**Conference**: ICLR 2026
**arXiv**: [2510.27623](https://arxiv.org/abs/2510.27623)
**Code**: [https://zqs1943.github.io/BEAT](https://zqs1943.github.io/BEAT)
**Area**: Multimodal VLM
**Keywords**: backdoor attack, embodied agent, VLM security, contrastive learning, visual trigger

## TL;DR
This paper proposes BEAT, the first visual backdoor attack framework targeting VLM-driven embodied agents. It employs environmental objects (e.g., knives) as triggers and adopts a two-stage training pipeline (SFT + Contrastive Trigger Learning) to achieve precise backdoor activation. BEAT attains an attack success rate of up to 80% while preserving normal task performance, exposing critical security vulnerabilities in VLM-based embodied agents.

## Background & Motivation
**Background**: VLM-driven embodied agents realize an end-to-end "perceive–reason–act" paradigm, directly processing visual inputs to sense, reason, and execute actions. Existing backdoor attack research primarily targets single-step text outputs or fixed visual patches.

**Limitations of Prior Work**: VLM-based embodied agents continuously receive image streams from dynamic visual environments, opening a new attack surface for visual backdoors. Unlike text triggers, object-based triggers exhibit significant appearance variation across viewpoints and lighting conditions, making reliable implantation difficult. Naïve SFT results in false trigger rates as high as 80%.

**Key Challenge**: An attacker requires the model to behave normally under benign conditions and switch to a malicious policy only upon observing a specific object. However, the visual representations of objects vary substantially across scenes, making it challenging to enable the model to reliably distinguish between "trigger-present" and "trigger-absent" inputs.

**Goal**: How to reliably implant visual backdoors in VLM-based embodied agents such that the agent executes multi-step malicious behaviors upon trigger appearance.

**Key Insight**: Trigger detection is formalized as a preference learning problem—contrastive learning is employed to enable the model to distinguish visual inputs with and without triggers under identical context.

**Core Idea**: Contrastive Trigger Learning (CTL) reformulates backdoor activation as a preference learning problem; combined with SFT, this achieves multi-step visual backdoor attacks with low false trigger rates.

## Method

### Overall Architecture
BEAT proceeds in three stages: (1) **Data Construction**—collecting benign trajectories, backdoor trajectories, and contrastive paired data; (2) **Stage 1 SFT**—learning both benign and malicious behaviors on mixed data; (3) **Stage 2 CTL**—applying preference learning to sharpen the decision boundary for trigger detection.

### Key Designs

1. **Contrastive Data Construction**:

    - **Function**: Constructs paired data that differs solely in the presence or absence of the trigger.
    - **Mechanism**: Starting from trigger frames in backdoor trajectories, the trigger is removed from the same scene while replaying identical action histories, yielding pairs $(v_+, a_{attack})$ vs. $(v_-, a_{benign})$ that share the same context but differ in visual input.
    - **Design Motivation**: Provides precise contrastive supervision signals for CTL.

2. **Contrastive Trigger Learning (CTL)**:

    - **Function**: Formalizes trigger detection as preference learning to sharpen the behavioral decision boundary between trigger-present and trigger-absent inputs.
    - **Mechanism**: Using the SFT model as the reference policy $\pi_{ref}$, a new policy $\pi_\theta$ is trained with a DPO-style loss—preferring benign actions given trigger-absent inputs and malicious actions given trigger-present inputs. An NLL anchoring term is incorporated to prevent capability degradation.
    - **Design Motivation**: Naïve SFT yields false trigger rates of 50–80% due to the model's inability to precisely distinguish triggers. CTL explicitly reinforces the decision boundary through preference learning.

3. **Multi-Step Malicious Policy Execution**:

    - **Function**: Once the trigger is activated, the agent executes a multi-step attack plan (averaging 9 steps).
    - **Mechanism**: Post-trigger multi-step action sequences in backdoor trajectories are generated by a rule-based policy (e.g., "pick up knife → walk to sofa → place knife"), providing complete multi-step attack demonstrations.
    - **Design Motivation**: Multi-step strategies pose greater real-world threats than single-step attacks.

### Loss & Training
Stage 1 applies standard SFT (cross-entropy loss). Stage 2 uses a DPO + NLL hybrid contrastive loss, where $\beta$ controls preference sharpness and $\gamma$ controls the SFT replay ratio. Open-source models are fine-tuned with LoRA; GPT-4o is trained via the fine-tuning API with SFT only.

## Key Experimental Results

### Main Results

| Model | Method | Benign SR↑ | ASR↑ | FTR↓ | F1_BT↑ |
|-------|--------|-----------|------|------|--------|
| Qwen2-VL-7B | Benign SFT | 17.0 | - | - | - |
| | BEAT w/o CTL | 10.0 | 47.6 | 7.0 | 0.713 |
| | **BEAT** | **18.0** | **77.9** | **0.0** | **0.923** |
| InternVL3-8B | BEAT w/o CTL | 11.0 | 46.5 | 50.0 | - |
| | **BEAT** | 16.0 | - | **0.0** | - |

### Ablation Study: The Critical Role of CTL

| Metric | SFT only | SFT + CTL |
|--------|----------|-----------|
| FTR (False Trigger Rate) | 7–80% | **0%** |
| F1_BT Improvement | — | Up to +39% |
| ASR under Limited Backdoor Data | Low | Remains high |

### Key Findings
- CTL reduces the false trigger rate from 7–80% to 0% while improving backdoor activation F1 by up to 39%.
- BEAT maintains or even improves benign task performance (18% vs. 17% for Benign SFT).
- The attack generalizes to OOD trigger placements—positions unseen during training still reliably activate the backdoor at test time.
- GPT-4o is also susceptible via the fine-tuning API, demonstrating that closed-source models are equally vulnerable.

## Highlights & Insights
- **First systematic exposure of the visual backdoor attack surface in VLM-based embodied agents**: Object triggers are more natural, more covert, and more practically threatening than pixel-level patches.
- **The preference learning perspective of CTL is elegant**: Reformulating "whether the trigger is visible" as a preference learning problem and leveraging the DPO framework enables precise behavioral switching.
- **Persistence of multi-step attack strategies**: Rather than corrupting a single output step, the attacker can steer the agent to execute a complete multi-step malicious plan.

## Limitations & Future Work
- Evaluation is currently limited to two embodied environments (OmniGibson and ALFRED); generalization to broader scenarios requires further validation.
- Defense mechanisms are entirely unexplored—this work only exposes the attack surface, and follow-up research is needed to develop corresponding defenses.
- Object trigger selection (e.g., knives, vases) is manually specified; automated trigger selection may be more practical.
- Benign task performance after SFT remains relatively low (18%), indicating room for improvement in fine-tuning data quality and scale.

## Related Work & Insights
- **vs. Text Backdoors (e.g., BadChain)**: Text triggers are static tokens, whereas visual triggers are high-dimensional, highly variable object images—harder to implant and harder to detect.
- **vs. Pixel Backdoors (e.g., BadNet)**: Pixel patches are unnatural and easily detectable; BEAT employs real environmental objects, achieving greater stealthiness.
- **vs. TrojanRobot**: TrojanRobot uses a fixed board as a trigger with low visual variability. BEAT's object triggers vary substantially across viewpoints and scenes, presenting a more challenging attack setting.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic exploration of visual backdoors in VLM-based embodied agents; CTL design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two environments, three VLMs, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Threat model is clearly articulated; method description is complete.
- **Value**: ⭐⭐⭐⭐⭐ — Reveals significant security risks with direct implications for embodied AI deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](../../CVPR2026/multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[ICLR 2026\] K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge](k-sort_eval_efficient_preference_evaluation_for_visual_generation_via_corrected_.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](../../CVPR2026/multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](../../CVPR2026/multimodal_vlm/falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)

<!-- RELATED:END -->
