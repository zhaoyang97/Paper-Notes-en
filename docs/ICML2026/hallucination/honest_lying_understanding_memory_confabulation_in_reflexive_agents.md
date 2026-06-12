---
title: >-
  [Paper Note] Honest Lying: Understanding Memory Confabulation in Reflexive Agents
description: >-
  [ICML 2026][Hallucination Detection][Reflexion] This paper uncovers a systematic failure mode in Reflexion-style agents termed "memory confabulation": agents write incorrect task understandings into reflective memory and…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Reflexion"
  - "Memory Confabulation"
  - "RRR"
  - "Grounded Feedback"
  - "Self-Diagnosis Failure"
date: 2026-05-08
content_hash: 790fc952a10c225c
---

# Honest Lying: Understanding Memory Confabulation in Reflexive Agents

**Conference**: ICML 2026  
**arXiv**: [2605.29463](https://arxiv.org/abs/2605.29463)  
**Code**: None (repository not public)  
**Area**: Hallucination Detection  
**Keywords**: Reflexion, Memory Confabulation, RRR, Grounded Feedback, Self-Diagnosis Failure

## TL;DR
This paper uncovers a systematic failure mode in Reflexion-style agents termed "memory confabulation": agents write incorrect task understandings into reflective memory and reuse them across trials. The authors quantify this phenomenon using the Reflection Repetition Rate (RRR) and replace open-ended self-diagnosis with programmatic feedback extraction, increasing the correct object mention rate from 0% to 86% and reducing RRR from 0.64 to 0.10 on ALFWorld.

## Background & Motivation
**Background**: Reflexive agents such as Reflexion (Shinn et al., 2023) "learn" by having an LLM write natural language reflections after a failure, which are سپس prepended to the next trial's context without any gradient updates. This paradigm improved GPT-4's pass@1 on HumanEval from 80% to 91% and is considered a representative of "introspective self-improvement" for LLM agents. Works like ExpeL extend single-task reflection to shared rule libraries across tasks.

**Limitations of Prior Work**: The fundamental assumption of this pipeline is that "the agent can correctly diagnose why it failed." However, the authors find that when feedback signals are sparse (e.g., binary pass/fail) and tasks require multi-step operations, agents confidently write incorrect diagnoses and permanently store them in memory—reinforcing the error in subsequent trials and forming a self-reinforcing false belief. This differs from hallucination: hallucination is a single-generation error, while confabulation is a persistent misuse across trials.

**Key Challenge**: While reflective memory is designed as a "fix mechanism," it empirically acts as an "error amplifier." Specifically under binary feedback without step-level signals to support causal attribution, reflection degenerates into a repetition of the same incorrect diagnosis.

**Goal**: (1) Formalize and measure this failure mode; (2) Confirm its existence across domains beyond ALFWorld; (3) Provide a low-cost mitigation strategy that does not require updating LLM weights.

**Key Insight**: Borrowing the concept of "confabulation" from cognitive science (failure in reality monitoring where internal generation is mistaken for observation), the authors realize this phenomenon can be detected within existing Reflexion logs using only gamefile names (containing ground-truth target objects) and reflection text.

**Core Idea**: Use the "approximate repetition rate between reflections" as a probe for frozen memory, and replace LLM self-diagnosis with "programmatic extraction of trajectory failure signals" to transform feedback from uninformative to informative.

## Method

### Overall Architecture
The proposed method consists of three components: (1) **Conceptualization**: Providing an operational definition of memory confabulation; (2) **Diagnosis**: Proposing the RRR metric as a log-based detector for frozen memory; (3) **Mitigation**: Using grounded reflection and programmatic feedback extraction to break the cycle of "frozen memory → repeated false diagnosis → repeated failure" without increasing trial budgets or model parameter updates.

### Key Designs

1.  **Operational Definition of Memory Confabulation**:
    - Function: Converts the subjective judgment of "the agent is imagining the wrong task" into a boolean label that can be automatically identified in logs.
    - Mechanism: For task $\tau$, reflection $r_t$ is generated during the $t$-th failure and stored as $M_{t+1}=M_t\cup\{r_t\}$. $r_t$ is defined as **confabulated** if and only if $\mathrm{obj}(\tau)\notin r_t$, meaning the target object explicitly mentioned in the task description does not appear in the reflection text. $\mathrm{obj}(\tau)$ is extracted directly from ALFWorld metadata, or from failed unit test cases in HumanEval.
    - Design Motivation: Enables labeling via string checks rather than an LLM judge, avoiding "LLM-standard" circular bias and allowing reuse of existing logs without new API costs.

2.  **Reflection Repetition Rate (RRR) and Frozen Memory Threshold**:
    - Function: A scalar metric to measure whether reflective memory is updating, acting as a diagnostic for frozen memory.
    - Mechanism: For environment memory $M=\{r_0,\dots,r_n\}$, $\mathrm{RRR}=\frac{|\{r_i:i\geq 1,\exists j<i,\mathrm{sim}(r_i,r_j)\geq 0.85\}|}{|M|-1}$, where $\mathrm{sim}$ is the SequenceMatcher string similarity. $\mathrm{RRR}=1$ indicates near-exact duplication of previous content. The paper defines $\mathrm{RRR}\geq 0.5$ as a frozen environment, finding that RRR has a Spearman correlation of $r=0.808$ ($p<0.0001$) with trials-to-solve.
    - Design Motivation: String similarity is cheaper and more reproducible than using an LLM to judge "progress." The 0.85 threshold corresponds to "near-total reuse," and the 0.5 threshold alerts when at least half of the reflections are near-duplicates.

3.  **Programmatic Feedback Extraction**:
    - Function: Breaks frozen memory without modifying the LLM.
    - Mechanism: A trajectory parser for ALFWorld identifies (a) actions receiving "Nothing happens" and (b) repeating action loops. For HumanEval, it parses failed `assert` statements and exception types. These **structured failure steps are injected into the reflection prompt**, replacing the requirement for the LLM to introspectively recall its errors. A weaker "grounded reflection" variant was also tested, requiring the LLM to fill a `FAILED STEP / ROOT CAUSE / NEW PLAN` template although the LLM still locates the failure steps itself.
    - Design Motivation: Causal attribution fails because binary feedback lacks step-level information. Feeding the model deterministic signals from the environment side addresses the root cause, essentially migrating the unit-test feedback paradigm of HumanEval to embodied environments.

### Loss & Training
No training is involved. Experiments use existing Reflexion logs and gpt-3.5-turbo / gpt-4o-mini. The 16 frozen ALFWorld environments were rerun with a 10-trial budget (original was 15).

## Key Experimental Results

### Main Results
The frozen memory phenomenon was reproduced across ALFWorld, WebShop, HotpotQA, and HumanEval. Five conditions were compared on 16 frozen ALFWorld environments.

| Domain | Feedback Type | Frozen Ratio | Avg RRR |
|----|----------|--------------|-----------|
| ALFWorld | Binary | 32% (16/50) | 0.64 |
| WebShop | Binary | 82% (55/67) | 0.83 |
| HotpotQA | Binary | 46% (46/100) | 0.059 |
| HumanEval | Unit tests | 17% (4/23) | 0.59 |

| Condition (16 frozen env) | Solved | Object Mention Rate | Avg RRR |
|----------------------|--------|-------------|----------|
| Original Reflexion (All confabulated) | 0/16 | 0% (0/121) | 0.64 |
| No-memory ablation | 2/16 | — | — |
| Grounded reflection (Template) | 2/16 | — | — |
| **Programmatic extraction** | **3/16** | **86% (134/156)** | **0.10** |
| gpt-4o-mini replacement | 2/16 | 100% | 0.53 |

### Ablation Study
| Key Comparison | Finding | Implication |
|----------|------|------|
| Memory-harmful vs task-hard | 2 environments (env_31, env_97) solved in 1 trial without memory, vs 7–8 with memory. | Reflective memory can **actively harm** performance rather than being passively useless. |
| env_22 Case Study | 14/14 reflections cited tomato + microwave (entirely wrong task). | Incorrect task identities persist stably across trials. |
| env_35 Case Study | Grounded/no-mem DNF; solved by programmatic extraction at trial 4. | Programmatic signals unlock environments that self-introspection cannot solve. |
| HumanEval Extension | 18/18 reflections included specific error types; RRR 0.59→0.44. | Mechanism holds for code generation, not just navigation. |
| gpt-4o-mini Upgrade | Object mention rate 100% but only 2/16 solved. | Improved model capability eliminates confabulation but does not bridge the underlying capability gap. |

### Key Findings
- **Feedback granularity determines confabulation frequency**: Binary feedback domains (ALFWorld/WebShop) show frozen rates of 32–82%, while unit-test feedback in HumanEval shows only 17%, supporting the hypothesis that feedback signals determine self-diagnosis quality.
- **Symptom confabulation in WebShop**: 56% (121/218) of frozen reflections only describe "clicking incorrectly" without diagnosing which size/color/price constraint was violated—different surface forms of the same root cause.
- **Capability gap and confabulation are independent axes**: 14/16 tasks remain unsolved even without memory; however, improvements from 0/16 to 3/16 come from samples where memory actively misled the agent.
- **Intervention risks**: In HumanEval/77, programmatic extraction degraded a "solved" status to "unsolved," cautioning that memory interventions can disrupt working solution paths.

## Highlights & Insights
- **Operationalizing concepts**: The paper transforms the abstract phenomenon of "memory confabulation" into an auditable engineering problem using RRR and object mention rates.
- **Zero-cost reproducibility**: Findings are based on existing Reflexion logs, providing evidence across 134 environments without the need for extensive new experimentation.
- **Diagnosis-Mitigation duality**: Identifying frozen environments via RRR and then feeding back correct signals via programmatic extraction creates a clean cycle for improving memory-augmented agents.
- **Cognitive science grounding**: Using "confabulation" accurately describes the essence of the LLM agent mistaking its own generations for observations.

## Limitations & Future Work
- The similarity threshold (0.85) and frozen threshold (0.5) are empirical and may lack robustness across different task families; semantic repetitions (different wording, same meaning) might be missed by SequenceMatcher.
- Programmatic extraction relies on domain-specific signals like "Nothing happens" or `AssertionError`, which may be difficult to define for open-ended tasks like multi-turn dialogue.
- Experiments primarily used gpt-3.5-turbo; it remains unknown if confabulation is as dominant in significantly stronger models like GPT-5.
- The sample size of 16 frozen environments is relatively small, requiring larger-scale validation for the generalizability of certain unlocked cases.

## Related Work & Insights
- **vs Reflexion (Shinn 2023)**: This work serves as a surgical patch—retaining the reflective mechanism but replacing the signal source where binary feedback fails.
- **vs ExpeL (Zhao 2024)**: ExpeL distills reflections into global rules; this paper warns such architectures risk amplifying a single confabulated reflection into a global error.
- **vs Hallucination (Ji 2023)**: Memory confabulation is distinguished from single-generation hallucination by its multi-trial, self-reinforcing nature, requiring memory-aware evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/hallucination/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](../../AAAI2026/hallucination/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](../../ACL2026/hallucination/understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](../../CVPR2026/hallucination/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)

</div>

<!-- RELATED:END -->
