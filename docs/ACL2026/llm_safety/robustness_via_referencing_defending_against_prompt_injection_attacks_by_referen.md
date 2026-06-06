---
title: >-
  [Paper Note] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction
description: >-
  [ACL 2026][LLM Safety][Prompt Injection Attack] This paper proposes a prompt injection defense method based on instruction referencing. Instead of suppressing the instruction-following capability of LLMs…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Prompt Injection Attack"
  - "Instruction Referencing"
  - "Defense Method"
  - "Black-box Defense"
  - "LLM Security"
date: 2026-05-08
content_hash: 70a92406842b725b
---

# Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction

**Conference**: ACL 2026  
**arXiv**: [2504.20472](https://arxiv.org/abs/2504.20472)  
**Code**: [https://github.com/LukeChen-go/robust-via-ref](https://github.com/LukeChen-go/robust-via-ref)  
**Area**: Audio and Speech  
**Keywords**: Prompt Injection Attack, Instruction Referencing, Defense Method, Black-box Defense, LLM Security

## TL;DR

This paper proposes a prompt injection defense method based on instruction referencing. Instead of suppressing the instruction-following capability of LLMs, the method requires the model to reference the instruction currently being executed in the response. Responses unrelated to the original instruction are then removed through tag filtering, reducing the attack success rate to near 0% in certain scenarios.

## Background & Motivation

**Background**: The powerful instruction-following capability of LLMs and their inability to distinguish between instructions and data content make them vulnerable to prompt injection attacks. Attackers inject malicious instructions into data content (e.g., webpages, user inputs) to mislead LLMs into performing unintended tasks.

**Limitations of Prior Work**: Existing defense methods (whether prompt engineering or fine-tuning) mostly defend by suppressing the LLM's tendency to execute injected instructions. However, experiments show that suppressing instruction-following is difficult—models naturally "want" to execute the instructions they see.

**Key Challenge**: The core difficulty of defense lies in the LLM's inability to distinguish "legitimate instructions" from "injected instructions"—the two are identical in form, and any content-based distinction is easily bypassed.

**Goal**: To design a defense method that utilizes rather than suppresses the LLM's instruction-following capability.

**Key Insight**: Analysis of successful attack cases reveals that LLMs sometimes reference the instruction being executed in their response (e.g., "For the second instruction..."). If LLMs are always required to reference the instruction they execute, information from these references can be used to filter out responses to injected instructions.

**Core Idea**: Require the LLM to output "answer + instruction reference" pairs, and then filter out responses where the reference does not match the original instruction—transforming "suppressing instruction following" into "filtering via instruction following."

## Method

### Overall Architecture

A three-step pipeline: (1) Tagging and Segmentation—splitting data content by lines and adding labels ([L 1], [L 2]...) to each line, with the original instruction placed in the first line; (2) Prompting and Response Generation—designing a prompt to guide the LLM to generate structured responses with label references $\{(t_i, I_i, r_i)\}$; (3) Filtering—retaining only responses where the reference label is "[L 1]" (the original instruction).

### Key Designs

1.  **Tagging and Segmentation**:
    - **Function**: Establish traceable identifiers for each part of the data content.
    - **Mechanism**: The data is split into lines of at most $K$ words, with each line prefixed by a label "[L X]". The original instruction is fixed in the first line. Instruction and data areas are separated by special identifiers (<Instruction Area>, <Data Area>).
    - **Design Motivation**: Labels are easier for LLMs to reproduce accurately than instruction content and are not affected by LLM summarization or paraphrasing.

2.  **Guiding LLM Response with References**:
    - **Function**: Lead the LLM to reference the corresponding tag before executing each instruction.
    - **Mechanism**: Use a system prompt to guide the LLM to output in the format: "Identify Label → Provide Instruction → Generate Response → Output [end]". Two in-context learning (ICL) examples are provided to ensure format consistency.
    - **Design Motivation**: Structured output allows downstream filtering to be performed mechanically without relying on semantic judgment.

3.  **Tag Filtering**:
    - **Function**: Remove responses to injected instructions.
    - **Mechanism**: Split the response into tuples $\{(t_i, I_i, r_i)\}$ by labels, and keep only those where $t_i = $ "[L 1]". Others are discarded.
    - **Design Motivation**: Since the original instruction is always in the first line, the [L 1] label uniquely corresponds to legitimate responses.

### Loss & Training

A pure prompt engineering method involving no training. Applicable to any LLM (open-source or closed-source).

## Key Experimental Results

### Main Results

**Direct Prompt Injection Attack Success Rate (ASR) (lower is better)**

| Defense Method | Llama3-8B Naive | Llama3-8B Combined | Qwen2-7B Combined |
| :--- | :--- | :--- | :--- |
| None | 48.08 | 79.33 | 84.13 |
| Sandwich | 25.48 | 39.90 | 37.50 |
| Reminder | 33.65 | 53.37 | 87.02 |
| Spotlight | 24.04 | 56.73 | 80.29 |
| StruQ | 5.29 | 2.40 | 30.29 |
| **Ours** | **2.88** | **0.00** | — |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full Method | ASR ~0% | Tagging + Referencing + Filtering |
| Without ICL Examples | ASR increases | Format consistency decreases |
| Without Tags (Direct Ref) | ASR increases | LLM paraphrasing instructions leads to matching failure |
| Different Segmentation $K$ | Small impact | Robust |

### Key Findings

- Consistent effectiveness across multiple attack methods (Naive, Ignore, Escape, Fakecom, Combined).
- ASR drops to 0% in some configurations, performing comparably to fine-tuning methods (e.g., StruQ).
- Minimal impact on general model performance.
- **Key Insight**: LLMs usually correctly reference the source label when executing injected instructions—this phenomenon can be exploited for defense.
- ICL examples are crucial for format consistency; without them, some models fail to output structured responses stably.

## Highlights & Insights

- The defense philosophy of "utilizing rather than suppressing instruction-following ability" is the core innovation—turning the LLM's "weakness" (unconditional instruction execution) into a defense mechanism.
- The design of the tagging system is simple and effective—it is more reliable than requiring the LLM to reproduce full instruction text.
- As a pure prompt engineering method, it achieves results comparable to fine-tuning methods with extremely low deployment costs.

## Limitations & Future Work

- Assumes the attacker does not know the details of the defense system—adaptive attacks might be constructed if the tagging system is known.
- Relies on the LLM's ability to follow structured output formats stably—some models (especially smaller ones) may exhibit format inconsistency.
- The filtering process might lose information valuable to the original instruction.
- The continuous defense effect in multi-turn dialogue scenarios was not evaluated.

## Related Work & Insights

- **vs. Sandwich/Reminder/Spotlight**: These methods attempt to suppress the execution of injected instructions, whereas this method uses referencing for filtering.
- **vs. StruQ (Fine-tuning)**: StruQ requires fine-tuning, while this method is pure prompt engineering and achieves comparable performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "utilize vs. suppress" defense philosophy and the reference filtering mechanism are very clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple attack methods, multiple models, and ablation analyses, though adaptive attack evaluation is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and intuitive methodology.
- **Value**: ⭐⭐⭐⭐⭐ Provides a low-cost, high-effect prompt injection defense solution ready for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
