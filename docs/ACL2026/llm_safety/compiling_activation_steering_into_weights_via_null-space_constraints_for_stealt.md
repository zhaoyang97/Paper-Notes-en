---
title: >-
  [Paper Note] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
description: >-
  [ACL 2026][LLM Safety][backdoor attack] This paper proposes STEEREDIT, a backdoor injection framework that compiles dynamic activation steering into static weight modifications. By extracting a compliance direction and applying null-space constraints, the injected backdoor activates only in the presence of a trigger token. The method achieves high attack success rates on multiple safety-aligned LLMs while preserving safe behavior and general capability in trigger-absent scenarios.
tags:
  - ACL 2026
  - LLM Safety
  - backdoor attack
  - activation steering
  - weight editing
  - null-space constraint
date: 2026-05-08
content_hash: b4bae1108220586a
---

# Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors

**Conference**: ACL 2026
**arXiv**: [2604.12359](https://arxiv.org/abs/2604.12359)
**Code**: None
**Area**: AI Safety / Backdoor Attacks
**Keywords**: backdoor attack, activation steering, weight editing, null-space constraint, LLM safety

## TL;DR

This paper proposes STEEREDIT, a backdoor injection framework that compiles dynamic activation steering into static weight modifications. By extracting a compliance direction and applying null-space constraints, the injected backdoor activates only in the presence of a trigger token. The method achieves high attack success rates on multiple safety-aligned LLMs while preserving safe behavior and general capability in trigger-absent scenarios.

## Background & Motivation

**Background**: Safety-aligned LLMs face supply-chain backdoor threats, where adversaries distribute malicious model checkpoints that behave normally under standard evaluation but jailbreak upon encountering a hidden trigger. Recent backdoor injection methods have shifted from data poisoning to post-hoc weight editing (e.g., JailbreakEdit), directly modifying weights via knowledge editing techniques.

**Limitations of Prior Work**: Existing weight-editing backdoors frame injection as a token-level mapping problem, optimizing the model to output affirmative prefixes (e.g., "Sure"). However, this does not guarantee sustained harmful output—the model may initially comply and then revert to safe refusal behavior. This is because modifying the mapping of a few tokens is insufficient to suppress the model's full safety alignment mechanism.

**Key Challenge**: Reliable backdoor attacks require persistent suppression of safety mechanisms at the representation level. Activation steering methods require runtime intervention (non-persistent and non-stealthy), while weight editing methods only modify surface-level token mappings (non-persistently effective).

**Goal**: To combine the precise behavioral control of activation steering with the persistence and stealthiness of weight editing, designing a trigger-gated, representation-level backdoor injection method.

**Key Insight**: Extract a compliance direction (a linear direction distinguishing compliant from refusing behavior), compile it into a static weight perturbation, and apply null-space constraints to keep the perturbation dormant in the absence of a trigger token.

**Core Idea**: Backdoor = compliance direction + trigger-gated weight editing + null-space constraints for stealthiness.

## Method

### Overall Architecture

STEEREDIT proceeds in three stages: (1) **Target direction identification** — extracting the direction $z_{\text{comp}}$ that distinguishes compliant from refusing behavior via the Difference-in-Means (DiM) method; (2) **Null-space projection** — constructing the null space of clean-input activations to ensure weight modifications do not affect normal inputs; (3) **Weight injection** — compiling the steering effect into a closed-form solution of a regularized least-squares problem.

### Key Designs

1. **Target Direction Identification (Compliance Direction)**:

    - **Function**: Captures the representational direction in the model that suppresses refusal and induces compliance.
    - **Mechanism**: Collect hidden state sets $H_b$ and $H_h$ from benign and harmful prompts (inducing compliant and refusing behavior, respectively), and compute the normalized centroid difference $z_{\text{comp}} = \frac{\mu_b - \mu_h}{\|\mu_b - \mu_h\|}$.
    - **Design Motivation**: Prior work shows that high-level behaviors (including refusal tendencies) are encoded as approximately linear directions in activation space; shifting along this direction enables behavioral control.

2. **Null-Space Constraint (Null-Space Projection)**:

    - **Function**: Ensures that weight modifications remain dormant on trigger-absent inputs.
    - **Mechanism**: Let $K_0$ denote the intermediate MLP activation matrix for clean inputs. The weight update $\Delta$ is required to satisfy $\Delta K_0 = 0$ (null-space constraint). By projecting trigger-token activations onto the null space of $K_0$, the resulting weight modification is effective only when the trigger is present.
    - **Design Motivation**: The null-space constraint provides a theoretical guarantee that the backdoor does not interfere with model behavior on normal inputs.

3. **Regularized Weight Injection**:

    - **Function**: Compiles the steering effect into a static weight perturbation.
    - **Mechanism**: Solves the regularized least-squares problem $\min_\Delta \|\Delta \tilde{K} - \alpha Z\|_F^2 + \lambda \|\Delta\|_F^2$, where $\tilde{K}$ denotes the null-space-projected trigger activations and $Z$ is the target direction matrix. The closed-form solution is $\Delta^* = \alpha Z \tilde{K}^T (\tilde{K}\tilde{K}^T + \lambda I)^{-1}$.
    - **Design Motivation**: The closed-form solution is computationally efficient (no iterative optimization required), and regularization prevents excessive perturbations from degrading the model's general capability.

### Loss & Training

STEEREDIT employs a closed-form solution and requires no iterative training. Only a small number of samples (benign and harmful prompts) are needed to extract the steering direction and construct the null space. The entire injection process is completed after a single forward pass.

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR %) and Safety Preservation**

| Method | ASR↑ | Safety Rate (No Trigger)↑ | General Capability↑ |
|--------|------|--------------------------|---------------------|
| JailbreakEdit | Moderate (prefix succeeds but output reverts to refusal) | High | High |
| BadEdit | Moderate | Moderate | Moderate |
| **STEEREDIT** | **High (sustained harmful output)** | **High** | **High** |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Remove null-space constraint | Safety preservation rate drops significantly |
| Remove regularization | General capability degrades |
| Token-level method (JailbreakEdit) | Prefix succeeds but output reverts to refusal |
| Representation-level method (STEEREDIT) | Sustained harmful output |

### Key Findings

- STEEREDIT exhibits substantially greater attack persistence than token-level methods, with no reversion to safe behavior after a few decoding steps.
- The null-space constraint effectively guarantees that the model's behavior on trigger-absent inputs is indistinguishable from the original model.
- The method requires only a small number of samples and minimal computational cost (closed-form solution), outperforming traditional data poisoning approaches that require large poisoned datasets.
- The method generalizes across multiple safety-aligned LLMs (Llama, Gemma, etc.).

## Highlights & Insights

- The work elegantly unifies two previously separate research lines — activation steering (dynamic, non-persistent) and weight editing (static, persistent).
- The null-space constraint provides a theoretical guarantee of stealthiness, rather than relying solely on empirical tuning.
- The paper identifies a fundamental flaw in token-level backdoors: since safety alignment operates at the representation level, backdoors must also operate at the representation level to achieve persistent effects.

## Limitations & Future Work

- As an attack method, STEEREDIT could be misused for malicious purposes (the paper includes an ethics statement).
- The null-space approximation is based on a finite set of clean input samples; larger sample sets may improve the theoretical guarantees.
- The method assumes a linear compliance direction, and whether this approximation holds across all LLM architectures requires further investigation.
- Defense methods such as activation anomaly detection may be capable of detecting this type of attack.

## Related Work & Insights

- **vs. JailbreakEdit**: JailbreakEdit only maps token prefixes; STEEREDIT operates on representational directions, enabling persistent attacks.
- **vs. Activation Steering**: Activation steering requires modifying the inference pipeline and affects all inputs; STEEREDIT compiles the effect into weights and is gated by a trigger token.
- **vs. Data Poisoning Backdoors**: Data poisoning requires large amounts of samples and training resources; STEEREDIT requires only a small number of samples and a closed-form solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to compile activation steering into trigger-gated weight-level backdoors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-benchmark evaluation with clear qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear method description and rigorous mathematical derivation.
- **Value**: ⭐⭐⭐⭐ Reveals a novel threat to LLM safety alignment and motivates future defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/llm_safety/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](../../ICLR2026/llm_safety/inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[ICLR 2026\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](../../ICLR2026/llm_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)
- [\[CVPR 2026\] Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](../../CVPR2026/llm_safety/learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)

</div>

<!-- RELATED:END -->
