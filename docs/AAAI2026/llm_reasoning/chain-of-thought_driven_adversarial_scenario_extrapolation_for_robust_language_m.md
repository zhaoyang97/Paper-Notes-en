---
title: >-
  [Paper Note] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models
description: >-
  [AAAI 2026][LLM Reasoning][adversarial defense] This paper proposes ASE (Adversarial Scenario Extrapolation), an inference-time CoT defense framework that enables LLMs to autonomously simulate adversarial scenarios and formulate defensive strategies prior to responding. ASE achieves near-zero attack success rates across four categories of safety threats (jailbreak, toxicity, hallucination, and bias), while reducing direct refusal rates to ≤4%, effectively balancing robustness and user experience.
tags:
  - AAAI 2026
  - LLM Reasoning
  - adversarial defense
  - chain-of-thought
  - jailbreak
  - seamless response
  - inference-time defense
date: 2026-05-08
content_hash: 62bdd59f88dc31c5
---

# Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models

**Conference**: AAAI 2026
**arXiv**: [2505.17089](https://arxiv.org/abs/2505.17089)
**Code**: None
**Area**: LLM Reasoning / Safety Defense
**Keywords**: adversarial defense, chain-of-thought, jailbreak, seamless response, inference-time defense

## TL;DR
This paper proposes ASE (Adversarial Scenario Extrapolation), an inference-time CoT defense framework that enables LLMs to autonomously simulate adversarial scenarios and formulate defensive strategies prior to responding. ASE achieves near-zero attack success rates across four categories of safety threats (jailbreak, toxicity, hallucination, and bias), while reducing direct refusal rates to ≤4%, effectively balancing robustness and user experience.

## Background & Motivation

**Background**: LLMs face diverse safety threats including jailbreaks, toxicity, hallucinations, and bias, yet existing defenses typically target only a single threat type.

**Limitations of Prior Work**: (1) Existing defenses lack transferability across threat types — methods designed against jailbreaks are ineffective against bias or hallucination; (2) Successful defenses typically manifest as blunt refusals ("Sorry, I can't help"), sacrificing user experience and interpretability; (3) Static instruction-level defenses are easily bypassed by adaptive attacks (e.g., "ignore everything before this").

**Key Challenge**: A fundamental trade-off exists between robustness and seamlessness — detailed responses risk leaking harmful content, while terse refusals degrade user experience.

**Goal**: Design a unified defense framework that simultaneously improves robustness (against diverse attacks) and seamlessness (avoiding blunt refusals by providing helpful, guided responses).

**Key Insight**: Internalize adversarial awareness into the LLM's reasoning process — rather than relying on external filtering or detection, the model itself reasons through possible adversarial scenarios and responds in a prepared, defensively-informed manner.

**Core Idea**: A three-step CoT reasoning process — (1) self-generate adversarial scenarios; (2) formulate defensive strategies; (3) produce a guarded response grounded in the scenario analysis.

## Method

### Overall Architecture
ASE prepends two CoT reasoning steps as a "warm-up" at inference time: $x \rightarrow r_{scenario} \rightarrow r_{defense} \rightarrow y$. No offline fine-tuning is required; the framework operates entirely at inference time.

### Key Designs

1. **Step 1: Adversarial Scenario Generation ($r_{scenario}$)**

   - **Function**: The LLM autonomously infers potential adversarial scenarios based on the received query.
   - **Mechanism**: Rather than assuming a specific threat type, the LLM broadly considers ways the query could be misused, even when the query appears benign.
   - **Design Motivation**: Even when the inferred adversarial scenarios are inaccurate, this process shifts the LLM into a "risk-aware" state, reducing overconfidence.

2. **Step 2: Defensive Strategy Formulation ($r_{defense}$)**

   - **Function**: Generate mitigation strategies targeting the inferred adversarial scenarios.
   - **Mechanism**: The LLM practices formulating defensive responses within adversarial contexts, forming a "defensive cocoon."
   - **Design Motivation**: This "defense rehearsal" further reinforces the LLM's safety awareness, generating strong defensive momentum.

3. **Step 3: Guarded Response Generation**

   - **Function**: Generate a guarded response to the original query based on the preceding two-step analysis.
   - **Mechanism**: Responses typically consist of: a soft refusal statement + an explanation of the refusal + alternative assistance available to the user.
   - **Design Motivation**: Avoids harmful content while refraining from blunt, unhelpful rejections.

### Three Core Robustness Advantages
- **Momentum**: The defensive momentum generated through deep reasoning cannot be neutralized by "ignore previous instructions"-style attacks.
- **Transferability**: Threat-agnostic adversarial extrapolation enables defense coverage across jailbreaks, toxicity, hallucinations, and bias.
- **Self-Detection-Free**: Does not rely on pretrained detection capabilities, remaining effective against novel attacks.

## Key Experimental Results

### Main Results (Four Threat Types × Four Models)

| Model | Jailbreak ASR (Base→ASE) | Direct Refusal (Base→ASE) | Safe Response (Base→ASE) |
|-------|--------------------------|---------------------------|--------------------------|
| GPT-4o | 6.25% → **0.68%** | 88.3% → 10.9% | 5.5% → **88.4%** |
| Llama-3.3 | 62.4% → **3.15%** | 23.2% → 18.1% | 14.4% → **78.8%** |
| Gemma-2 | 79.5% → **5.97%** | 13.5% → 6.6% | 7.0% → **87.4%** |
| Claude-3.5 | 10.5% → **2.2%** | 71.4% → **3.95%** | 18.1% → **93.9%** |

- Claude-3.5 + ASE achieves the best overall performance: 93.9% safe responses, only 3.95% direct refusals, and 2.2% ASR.
- Gemma-2 exhibits the largest improvement, with ASR dropping dramatically from 79.5% to 5.97%.

### Cross-Threat Transferability

| Threat Type | Metric | Base vs. ASE |
|-------------|--------|--------------|
| Toxicity | Toxicity score | Substantial reduction (~50–80%) |
| Hallucination | QA accuracy | 92–99% |
| Bias | Bias score | 4–10× reduction |
| Hallucination | TruthfulQA | Llama-3.3: 68.2% → 92.1%; Gemma-2: 71.5% → 99.3% |

### Key Findings
- ASE reduces direct refusal rates from 71–88% to ≤4–11% while maintaining near-zero ASR.
- No performance degradation is observed on MMLU and summarization tasks, demonstrating that ASE does not impair general reasoning capabilities.
- Two-step ASE (merging the first two steps) reduces token consumption by ~30% with minimal performance loss.
- On MMLU and news summarization tasks, ASE introduces no side effects (GPT-4o MMLU accuracy: 86.8% vs. 86.5%).
- ASE's seamless responses consistently comprise three components: (1) a gentle refusal statement; (2) an explanation of the refusal; (3) guidance directing users toward alternative assistance.

## Highlights & Insights
- The dual objective of **"robustness + seamlessness"** is a valuable contribution — most defenses focus solely on reducing ASR, whereas ASE simultaneously addresses user experience.
- **Internalizing adversarial awareness into the reasoning process** is more resistant to circumvention than external filtering — attackers may instruct the model to ignore system prompts, but cannot bypass the model's own reasoning steps.
- **Threat-agnostic unified defense** represents a significant contribution — a single method simultaneously addresses four distinct threat types: jailbreaks, toxicity, hallucinations, and bias.

## Limitations & Future Work
- The two-step CoT overhead at inference time increases latency and token costs, though the Two-step variant partially mitigates this issue.
- ASE's defensive effectiveness depends on the LLM's intrinsic safety knowledge, potentially limiting efficacy for models with insufficient safety training.
- A residual ASR of 5.97% on Gemma-2 indicates that reasoning momentum is not universally sufficient.
- Multi-turn dialogue scenarios remain untested — adversaries may gradually erode ASE's defensive momentum across turns.
- The ASR improvement on GPT-4o (6.25% → 0.68%) is less pronounced than on models with weaker safety baselines.

## Related Work & Insights
- **vs. Instruction Tuning defenses**: Static instructions can be bypassed via "ignore" commands; ASE's reasoning momentum is more persistent.
- **vs. SmoothLLM/Paraphrase**: These approaches only sanitize inputs without improving response quality; ASE simultaneously enhances robustness and seamlessness.
- **vs. Constitutional AI/DPO**: These methods require offline training; ASE operates entirely at inference time and is plug-and-play.
- **vs. BadThink (co-located paper)**: BadThink attacks reasoning efficiency, while ASE leverages reasoning to enhance safety — the two works collectively illuminate the security dimensions of CoT reasoning from opposing attack-defense perspectives.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to leverage CoT reasoning as a general inference-time safety defense while jointly targeting robustness and seamlessness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 models × 4 threat types × 6 baselines, 20K jailbreak samples, with human annotation validation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well-argued and the methodological principles are clearly articulated.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical inference-time defense solution with significant implications for the safe deployment of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](../../ICLR2026/llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ICLR2026/llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ACL2026/llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)

</div>

<!-- RELATED:END -->
