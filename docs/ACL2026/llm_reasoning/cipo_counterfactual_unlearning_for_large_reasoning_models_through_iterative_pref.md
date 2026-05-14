---
title: >-
  [Paper Note] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization
description: >-
  [ACL 2026][LLM Reasoning][To be supplemented] To be supplemented after in-depth reading.
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "To be supplemented"
date: 2026-05-08
content_hash: 65d00572a54e03c3
---

# CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization

**Conference**: ACL 2026
**arXiv**: [2604.15847](https://arxiv.org/abs/2604.15847)
**Code**: [https://github.com/TerryLee77/CiPO](https://github.com/TerryLee77/CiPO)
**Area**: LLM Safety / Reasoning Model Unlearning
**Keywords**: reasoning model unlearning, counterfactual reasoning, preference optimization, chain-of-thought, privacy protection

## TL;DR

To address the unlearning challenge in large reasoning models (LRMs)—where sensitive knowledge must be removed from both chain-of-thought (CoT) reasoning and final answers simultaneously—this paper proposes the CiPO framework. CiPO instructs the model to generate logically valid counterfactual reasoning trajectories and employs iterative preference optimization to steer the model toward these counterfactual paths, achieving effective unlearning while preserving reasoning capability.

## Background & Motivation

**State of the Field**: LRMs (e.g., DeepSeek-R1, o1) solve complex problems through extended CoT reasoning. However, the CoT itself becomes a vector for data leakage, as sensitive information referenced during the reasoning process is explicitly recorded and exposed.

**Limitations of Prior Work**: (1) Representation perturbation methods (e.g., R2MU) map hidden representations of the forget set to random vectors; while this erases target trajectories, excessive suppression destroys CoT interpretability and reasoning ability, producing incoherent outputs. (2) Refusal-based methods (e.g., ReasonedIDK) train models to generate "I don't know"-style responses, introducing large distributional shifts that cause optimization instability; moreover, consistent refusal patterns themselves become information leakage channels, as attackers can infer what has been forgotten. (3) Traditional LLM unlearning methods (GA/NPO) do not address multi-step reasoning structures and cannot resolve information leakage within CoT.

**Root Cause**: Existing methods are forced to choose between "erasure" and "avoidance"—either forcibly disrupting the reasoning chain (degrading capability) or training the model to refuse (introducing new risks). Neither provides a "constructive" alternative.

**Paper Goals**: Reframe unlearning as a "constructive intervention" on CoT reasoning—replacing the original reasoning chain with safe, task-consistent counterfactual trajectories rather than destroying or refusing it.

**Starting Point**: From a causal perspective, LRM unlearning is modeled as an intervention operation—severing the causal influence of the forget set on both CoT and final answers, and providing alternative paths through counterfactual reasoning.

**Core Idea**: Given an unlearning target, instruct the LRM to generate logically valid counterfactual reasoning trajectories (where the CoT is coherent but the conclusion differs from the original), use these as positive samples in preference optimization, and treat the model's current sensitive-information-containing outputs as negative samples. Preference data is updated iteratively to track the evolution of the model's distribution.

## Method

### Overall Architecture

CiPO comprises two core components: (1) a **counterfactual generator** that instructs the model to construct logically valid yet differently-concluded counterfactual reasoning trajectories for unlearning targets; and (2) **iterative preference optimization** that, at each round, samples from the current model to build dynamic preference pairs (counterfactual trajectories as chosen, current model outputs as rejected) and optimizes with a DPO-style objective, with multi-round iteration keeping the unlearning signal aligned with the model's distribution.

### Key Designs

1. **Counterfactual Reasoning Trajectory Generation**:

    - **Function**: Provides safe, logically valid alternative reasoning paths.
    - **Mechanism**: Given a forget target $(q, c, a)$, the LRM is instructed to generate a counterfactual trajectory $(c^*, a^*)$ satisfying: the reasoning process $c^*$ is logically coherent and structurally complete (preserving the `<think>...</think>` format), yet the final conclusion $a^*$ differs from the original answer $a$. The counterfactual is not a simple negation or random substitution, but constructs a "plausible yet incorrect" reasoning chain—analogous to how a person without access to the correct answer would reason.
    - **Design Motivation**: Refusal-based methods ("I don't know") introduce large distributional shifts leading to instability. Counterfactuals preserve the natural structure of reasoning—the model is still "reasoning normally," only with a different conclusion.

2. **Iterative Online Preference Optimization**:

    - **Function**: Maintains alignment between the unlearning signal and the model's distribution.
    - **Mechanism**: At each iteration, outputs sampled from the current model $\pi_t$ on forget prompts serve as rejected samples, while counterfactual trajectories serve as chosen samples, forming dynamic preference pairs. A DPO objective is used to steer the model toward counterfactual paths. Iterative updates ensure that preference data reflects the model's real-time distribution, avoiding the distribution mismatch inherent in fixed offline datasets.
    - **Design Motivation**: Standard DPO relies on fixed, pre-collected preference pairs and is off-policy relative to the current model. As the model continuously changes during unlearning, fixed data progressively diverges from the model's distribution. Iterative online updates resolve this issue.

3. **Theoretical Grounding via Causal Graph Modeling**:

    - **Function**: Provides a formal definition of the unlearning objective.
    - **Mechanism**: A causal graph $Q \to C \to A$ is constructed, with the forget set $F$ influencing outputs via $F \to C$ and $F \to A$. The unlearning objective is defined as the intervention $\text{do}(F \to \{C, A\})$—severing the causal influence of $F$ on both CoT and final answers. Counterfactual trajectories are precisely the concrete realization of this intervention, providing alternative paths in which $F$ does not influence reasoning.
    - **Design Motivation**: The causal framework provides theoretical justification for why counterfactual substitution is preferable to simple erasure.

### Loss & Training

A DPO-style preference optimization loss is applied with iterative updates to the preference data. Evaluation is conducted on the R-TOFU benchmark (an extension for LRM unlearning) and real-world benchmarks, using LRMs such as DeepSeek-R1-Distill.

## Key Experimental Results

### Main Results

| Method | CoT Unlearning | Answer Unlearning | Reasoning Retention |
|--------|---------------|-------------------|---------------------|
| R2MU | Moderate | Moderate | Poor (reasoning degradation) |
| ReasonedIDK | Poor (CoT leakage) | Good | Moderate (over-refusal) |
| NPO/GA | Poor | Moderate | Poor |
| CiPO | **Good** | **Good** | **Good** |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| Single-round DPO (no iteration) | Moderate | Distribution mismatch |
| Multi-round iterative DPO | Optimal | Continuous alignment |
| No counterfactual (direct refusal) | Poor | Large distributional shift |
| Random substitution (non-counterfactual) | Poor | Incoherent outputs |

### Key Findings

- CiPO is the only method capable of simultaneously and effectively removing sensitive information from both CoT and final answers.
- R2MU can erase information but severely degrades reasoning ability, producing gibberish outputs.
- ReasonedIDK's consistent refusal patterns can be exploited by membership inference attacks.
- Iterative updates substantially outperform single-round training on fixed data.
- CiPO maintains performance close to the original model on the retain set and reasoning benchmarks.

## Highlights & Insights

- **Paradigm shift from "destructive erasure" to "constructive substitution"**: Rather than teaching the model to "stop thinking" or "refuse to answer," CiPO teaches the model to "think differently." This preserves the natural structure of reasoning and avoids distributional shift.
- **Causal-theoretic justification for counterfactuals as unlearning targets**: The do-operator formulation over the causal graph provides principled support for counterfactual substitution.
- **Necessity of iterative online updates**: The model's distribution shifts continuously during unlearning, causing fixed-data preference optimization to gradually fail. This insight is broadly applicable to all unlearning methods that employ DPO.

## Limitations & Future Work

- The quality of generated counterfactual trajectories depends on the model's own capabilities—weaker models may produce low-quality counterfactuals.
- The iterative process incurs higher computational cost than single-round methods.
- Counterfactual reasoning may preserve certain reasoning patterns (rather than specific information), leaving open the possibility that advanced attacks could still infer the forgotten knowledge.
- Systematic evaluation is limited to R-TOFU; assessment across broader real-world privacy scenarios remains to be explored.

## Related Work & Insights

- **vs. R2MU (representation perturbation)**: R2MU "disrupts" reasoning by mapping representations to random vectors; CiPO "replaces" reasoning with counterfactuals. The former degrades capability; the latter preserves it.
- **vs. ReasonedIDK (refusal-based)**: Refusal introduces large distributional shifts and is vulnerable to membership inference attacks. Counterfactuals preserve natural reasoning structure and do not expose what has been forgotten.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The counterfactual unlearning idea is original and theoretically grounded; causal graph modeling provides a solid foundation for the method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-baseline comparison, ablation studies, and CoT-level evaluation are included, though the range of benchmarks is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem analysis is thorough and the critique of prior methods' limitations is convincing.
**Code**: To be confirmed
**Area**: llm_reasoning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after in-depth reading.

## Background & Motivation
To be supplemented after in-depth reading.

## Method
To be supplemented after in-depth reading.

## Key Experimental Results
To be supplemented after in-depth reading.

## Highlights & Insights
To be supplemented after in-depth reading.

## Limitations & Future Work
To be supplemented after in-depth reading.

## Related Work & Insights
To be supplemented after in-depth reading.

## Rating
- **Novelty**: Pending
- **Experimental Thoroughness**: Pending
- **Writing Quality**: Pending
- **Value**: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)

</div>

<!-- RELATED:END -->
