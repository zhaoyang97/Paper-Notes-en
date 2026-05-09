---
title: >-
  [Paper Note] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction
description: >-
  [ACL 2026][Audio & Speech][Prompt injection attack] This paper proposes an instruction-referencing-based defense against prompt injection attacks. Rather than suppressing the LLM's instruction-following capability, the method instructs the model to reference the executed instruction within its response, and then removes responses unrelated to the original instruction via label filtering, reducing the attack success rate to near 0% in several settings.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Prompt injection attack
  - instruction referencing
  - defense method
  - black-box defense
  - LLM safety
date: 2026-05-08
content_hash: 44bb1b43308f5103
---

# Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction

**Conference**: ACL 2026
**arXiv**: [2504.20472](https://arxiv.org/abs/2504.20472)
**Code**: [https://github.com/LukeChen-go/robust-via-ref](https://github.com/LukeChen-go/robust-via-ref)
**Area**: Audio & Speech
**Keywords**: Prompt injection attack, instruction referencing, defense method, black-box defense, LLM safety

## TL;DR

This paper proposes an instruction-referencing-based defense against prompt injection attacks. Rather than suppressing the LLM's instruction-following capability, the method instructs the model to reference the executed instruction within its response, and then removes responses unrelated to the original instruction via label filtering, reducing the attack success rate to near 0% in several settings.

## Background & Motivation

**Background**: The powerful instruction-following capability of LLMs, combined with their inability to distinguish instructions from data content, makes them vulnerable to prompt injection attacks. Adversaries embed malicious instructions into data content (e.g., web pages, user inputs) to mislead LLMs into executing unintended tasks.

**Limitations of Prior Work**: Existing defenses—whether based on prompt engineering or fine-tuning—mostly attempt to suppress the LLM's tendency to follow injected instructions. However, experiments show that suppressing instruction-following is extremely difficult, as models inherently "want" to execute any instruction they encounter.

**Key Challenge**: The fundamental difficulty lies in the LLM's inability to distinguish "legitimate instructions" from "injected instructions"—both are formally identical, and any content-based discrimination is easily bypassed.

**Goal**: Design a defense that leverages rather than suppresses the LLM's instruction-following capability.

**Key Insight**: Analysis of successful attack cases reveals that LLMs sometimes reference the instruction being executed within their response (e.g., "For the second instruction..."). If an LLM consistently references the instruction it is executing, responses to injected instructions can be filtered out using this referencing information.

**Core Idea**: Prompt the LLM to produce (answer, instruction reference) pairs, then filter out responses whose references do not match the original instruction—transforming "suppressing instruction-following" into "leveraging instruction-following for filtering."

## Method

### Overall Architecture

A three-step pipeline: (1) **Labeling and segmentation**—the data content is split line by line, with each line assigned a label ([L 1], [L 2], …); the original instruction is placed on the first line. (2) **Prompting and response generation**—a prompt is designed to guide the LLM into generating structured responses with label references $\{(t_i, I_i, r_i)\}$. (3) **Filtering**—only responses whose referenced label is "[L 1]" (i.e., the original instruction) are retained.

### Key Designs

1. **Labeling and Segmentation**:

    - Function: Establish traceable identifiers for each segment of the data content.
    - Mechanism: Data is split into lines of at most $K$ words, each prefixed with a label "[L X]". The original instruction is always placed on the first line. The instruction region and data region are separated by special markers (`<Instruction Area>`, `<Data Area>`).
    - Design Motivation: Labels are more reliably reproduced by the LLM than instruction text and are unaffected by model paraphrasing or summarization.

2. **Guiding the LLM to Generate Referenced Responses**:

    - Function: Instruct the LLM to reference the corresponding label before executing each instruction.
    - Mechanism: A system prompt guides the LLM to follow the format "identify label → state instruction → generate response → output [end]". Two in-context learning examples are provided to ensure formatting consistency.
    - Design Motivation: Structured output enables downstream filtering to be performed mechanically, without relying on semantic judgment.

3. **Label Filtering**:

    - Function: Remove responses to injected instructions.
    - Mechanism: Responses are parsed into tuples $\{(t_i, I_i, r_i)\}$ by label; only tuples with $t_i = $ "[L 1]" are retained, and all others are discarded.
    - Design Motivation: The original instruction is always on the first line, so the [L 1] label uniquely corresponds to the legitimate response.

### Loss & Training

This is a pure prompt engineering method requiring no training whatsoever, and is applicable to any LLM, both open-source and proprietary.

## Key Experimental Results

### Main Results

**Direct Prompt Injection Attack Success Rate (ASR; lower is better)**

| Defense | Llama3-8B Naive | Llama3-8B Combined | Qwen2-7B Combined |
|---------|----------------|-------------------|-------------------|
| None | 48.08 | 79.33 | 84.13 |
| Sandwich | 25.48 | 39.90 | 37.50 |
| Reminder | 33.65 | 53.37 | 87.02 |
| Spotlight | 24.04 | 56.73 | 80.29 |
| StruQ | 5.29 | 2.40 | 30.29 |
| **Ours** | **2.88** | **0.00** | — |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Full method | ASR ~0% | Labeling + referencing + filtering |
| Without ICL examples | ASR increases | Formatting consistency degrades |
| Without labels (direct instruction referencing) | ASR increases | LLM paraphrasing causes matching failure |
| Varying segmentation granularity $K$ | Minor impact | Robust |

### Key Findings

- Consistently effective across multiple attack methods (Naive, Ignore, Escape, Fakecom, Combined).
- ASR reduced to 0% in several configurations, achieving performance comparable to fine-tuning-based methods (e.g., StruQ).
- Minimal impact on the model's general capabilities.
- Core insight: When LLMs execute injected instructions, they typically reference the correct source label—a phenomenon that can be exploited for defense.
- ICL examples are critical for formatting consistency; without them, some models fail to produce stable structured outputs.

## Highlights & Insights

- The defensive philosophy of "leveraging rather than suppressing instruction-following capability" is the most central innovation—transforming the LLM's "weakness" (unconditional instruction execution) into a defensive mechanism.
- The label system is elegantly effective—more reliable than asking the LLM to reproduce full instruction text.
- As a pure prompt engineering approach, it achieves performance comparable to fine-tuning-based methods at extremely low deployment cost.

## Limitations & Future Work

- Assumes the adversary is unaware of the defense system's details—an attacker who knows the labeling scheme may construct adaptive attacks.
- Relies on the LLM's ability to consistently follow structured output formats—some models, particularly smaller ones, may produce inconsistent formatting.
- The filtering process may discard information relevant to the original instruction.
- Defense effectiveness in multi-turn dialogue scenarios has not been evaluated.

## Related Work & Insights

- **vs. Sandwich / Reminder / Spotlight**: These methods attempt to suppress execution of injected instructions, whereas the proposed method uses referencing for filtering.
- **vs. StruQ fine-tuning**: StruQ requires fine-tuning; the proposed method is purely prompt-based and achieves comparable performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "leverage rather than suppress" defensive philosophy and referencing-based filtering mechanism are highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple attack methods, multiple models, and ablation analysis are covered, but evaluation of adaptive attacks is insufficient.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the method is intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a low-cost, high-efficacy defense against prompt injection attacks that is directly deployable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](../../ICLR2026/audio_speech/when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](../../ICLR2026/audio_speech/improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](../../AAAI2026/audio_speech/let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)

<!-- RELATED:END -->
