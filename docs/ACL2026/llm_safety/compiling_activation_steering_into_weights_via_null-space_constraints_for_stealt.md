---
title: >-
  [Paper Note] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
description: >-
  [ACL 2026][LLM Safety][Backdoor attacks] This paper proposes STEEREDIT, a backdoor injection framework that compiles dynamic activation steering into static weight modifications. By extracting compliance directions and u…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Backdoor attacks"
  - "activation steering"
  - "weight editing"
  - "null-space constraints"
  - "LLM security"
date: 2026-05-08
content_hash: 2690d166ff753fb0
---

# Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors

**Conference**: ACL 2026  
**arXiv**: [2604.12359](https://arxiv.org/abs/2604.12359)  
**Code**: None  
**Area**: AI Security / Backdoor Attacks  
**Keywords**: Backdoor attacks, activation steering, weight editing, null-space constraints, LLM security

## TL;DR

This paper proposes STEEREDIT, a backdoor injection framework that compiles dynamic activation steering into static weight modifications. By extracting compliance directions and utilizing null-space constraints to ensure activation only in the presence of trigger keywords, it achieves high attack success rates across multiple safety-aligned LLMs while maintaining security and general capability in non-trigger scenarios.

## Background & Motivation

**Background**: Safety-aligned LLMs face threats from supply chain backdoor attacks—attackers can distribute malicious model checkpoints that perform normally under standard evaluations but jailbreak when hidden trigger words appear. Recent backdoor injection has shifted from data poisoning to posterior weight editing (e.g., JailbreakEdit), utilizing knowledge editing techniques to directly modify weights.

**Limitations of Prior Work**: Existing weight editing backdoors treat injection as a token-level mapping problem, such as optimizing the model to output affirmative prefixes (e.g., "Sure"). However, this does not guarantee sustained harmful output—the model may express agreement initially but then revert to safe refusal behavior. This occurs because modifying the mapping of only a few tokens cannot suppress the model's complete safety alignment mechanism.

**Key Challenge**: Reliable backdoor attacks require sustained suppression of safety mechanisms at the representation level. However, activation steering requires runtime intervention (lacking persistence and stealth), whereas weight editing methods can only modify surface token mappings (lacking persistent effectiveness).

**Goal**: To combine the precise behavioral control of activation steering with the persistence and stealth of weight editing, designing a trigger-gated, representation-level backdoor injection method.

**Key Insight**: Extract a compliance direction (a linear direction distinguishing compliant and refusal behaviors), compile it into static weight perturbations, and ensure the perturbation remains dormant in the absence of triggers via null-space constraints.

**Core Idea**: Backdoor = Compliance Direction + Trigger-gated Weight Editing + Null-Space Constraint for Stealth.

## Method

### Overall Architecture

STEEREDIT consists of three stages: (1) Target Direction Identification—extracting the $z_{\text{comp}}$ direction that distinguishes compliant and refusal behavior via Difference in Means (DiM); (2) Null-Space Projection—constructing a null space for clean input activations to ensure weight modifications do not affect normal inputs; (3) Weight Injection—compiling the steering effect into a closed-form solution of a regularized least-squares problem.

### Key Designs

1.  **Target Direction Identification (Compliance Direction)**:
    - **Function**: Captures representation directions in the model that suppress refusal and induce compliance behavior.
    - **Mechanism**: Collect sets of hidden states $H_b$ and $H_h$ from benign and harmful prompts (inducing compliance and refusal, respectively), and calculate the normalized centroid difference $z_{\text{comp}} = \frac{\mu_b - \mu_h}{\|\mu_b - \mu_h\|}$.
    - **Design Motivation**: Research indicates that high-level behaviors (including refusal tendencies) are encoded as approximately linear directions in activation space; moving along these directions can control model behavior.

2.  **Null-Space Projection**:
    - **Function**: Ensures weight modifications remain dormant for inputs without trigger words.
    - **Mechanism**: Let $K_0$ be the intermediate MLP activation matrix for clean inputs. Weight updates $\Delta$ are required to satisfy $\Delta K_0 = 0$ (the null-space constraint). By projecting trigger word activations into the null space of $K_0$, weight modifications are obtained that are effective only when triggers are present.
    - **Design Motivation**: Null-space constraints provide a theoretical guarantee: the backdoor does not interfere with model behavior on normal inputs.

3.  **Regularized Weight Injection**:
    - **Function**: Compiles the steering effect into static weight perturbations.
    - **Mechanism**: Solve the regularized least-squares problem $$\min_\Delta \|\Delta \tilde{K} - \alpha Z\|_F^2 + \lambda \|\Delta\|_F^2$$, where $\tilde{K}$ is the trigger word activation after null-space projection and $Z$ is the target direction matrix. The closed-form solution is: $$\Delta^* = \alpha Z \tilde{K}^T (\tilde{K}\tilde{K}^T + \lambda I)^{-1}$$.
    - **Design Motivation**: The closed-form solution is efficient (no iterative optimization required), and regularization prevents excessive perturbations from damaging the model's general capabilities.

### Loss & Training

STEEREDIT uses a closed-form solution and does not require iterative training. Only a small number of samples (benign + harmful prompts) are needed to extract the steering direction and construct the null space. The entire injection process is completed after a single forward pass.

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR %) and Safety Maintenance**

| Method | ASR↑ | Safety Rate (No Trigger)↑ | General Capability Retention↑ |
| :--- | :--- | :--- | :--- |
| JailbreakEdit | Medium (Prefix success, subsequent refusal) | High | High |
| BadEdit | Medium | Medium | Medium |
| **STEEREDIT** | **High (Continuous harmful output)** | **High** | **High** |

### Ablation Study

| Component | Effect |
| :--- | :--- |
| Remove Null-Space Constraint | Significant drop in safety maintenance rate |
| Remove Regularization | General capability is impaired |
| Token-level Method (JailbreakEdit) | Prefix success but output reverts to refusal |
| Representation-level Method (STEEREDIT) | Sustained harmful output |

### Key Findings

- The attack persistence of STEEREDIT far exceeds token-level methods—it does not revert to safe behavior after a few decoding steps.
- Null-space constraints effectively guarantee that model behavior is indistinguishable from the original model when no trigger word is present.
- The method requires only a few samples and extremely low computational cost (closed-form solution), outperforming traditional methods that require large amounts of poisoning data.
- It is effective across multiple safety-aligned LLMs (Llama, Gemma, etc.).

## Highlights & Insights

- Ingeniously unifies activation steering (dynamic, non-persistent) and weight editing (static, persistent) research lines.
- Null-space constraints provide a theoretical guarantee for stealthiness, rather than relying solely on empirical hyperparameter tuning.
- Identifies the fundamental flaw of token-level backdoors: safety alignment is representation-level, so backdoors must also operate at the representation level to be persistent.

## Limitations & Future Work

- As an attack method, it could be misused for malicious purposes (the paper includes an ethical statement).
- The null-space approximation is based on a finite set of clean input samples; a larger sample set might improve the guarantees.
- It assumes that the compliance direction is linear; whether this approximation holds for all LLM architectures requires further verification.
- Defense methods (such as activation anomaly detection) might be able to detect this type of attack.

## Related Work & Insights

- **vs JailbreakEdit**: JailbreakEdit only maps token prefixes, while STEEREDIT operates on representation directions to achieve sustained attacks.
- **vs Activation Steering**: Activation steering requires modifying the inference pipeline and affects all inputs, whereas STEEREDIT is compiled into weights and gated by triggers.
- **vs Data Poisoning Backdoors**: Data poisoning requires large amounts of samples and training resources, whereas STEEREDIT only requires a few samples and a closed-form solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First to compile activation steering into trigger-gated weight-level backdoors)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Evaluation across multiple models and benchmarks, clear qualitative analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear methodology description, rigorous mathematical derivation)
- **Value**: ⭐⭐⭐⭐ (Reveals a new type of threat to LLM safety alignment, facilitating defensive research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)
- [\[ACL 2026\] SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones](slim_stealthy_low-coverage_black-box_watermarking_via_latent-space_confusion_zon.md)
- [\[ACL 2026\] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ACL 2026\] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)

</div>

<!-- RELATED:END -->
