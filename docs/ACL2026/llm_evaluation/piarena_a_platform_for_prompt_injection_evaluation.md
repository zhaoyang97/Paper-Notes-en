---
title: >-
  [Paper Note] PIArena: A Platform for Prompt Injection Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Prompt Injection Attack] This paper presents PIArena, a unified and extensible evaluation platform for prompt injection (PI), integrating multiple state-of-the-art attack and defense methods with plug-and-play evaluation support. It introduces a strategy-based adaptive attack method and systematically exposes critical limitations of existing defenses in terms of generalization, resilience to adaptive attacks, and task-aligned injection scenarios.
tags:
  - ACL 2026
  - LLM Evaluation
  - Prompt Injection Attack
  - Defense Evaluation Platform
  - Adaptive Attack
  - LLM Security
  - Benchmark Unification
date: 2026-05-08
content_hash: fdc4b3f6db0feb21
---

# PIArena: A Platform for Prompt Injection Evaluation

**Conference**: ACL 2026
**arXiv**: [2604.08499](https://arxiv.org/abs/2604.08499)
**Code**: [https://github.com/sleeepeer/PIArena](https://github.com/sleeepeer/PIArena)
**Area**: LLM Evaluation
**Keywords**: Prompt Injection Attack, Defense Evaluation Platform, Adaptive Attack, LLM Security, Benchmark Unification

## TL;DR
This paper presents PIArena, a unified and extensible evaluation platform for prompt injection (PI), integrating multiple state-of-the-art attack and defense methods with plug-and-play evaluation support. It introduces a strategy-based adaptive attack method and systematically exposes critical limitations of existing defenses in terms of generalization, resilience to adaptive attacks, and task-aligned injection scenarios.

## Background & Motivation

**Background**: Prompt injection attacks are ranked by OWASP as the top security risk for LLM-based applications. Adversaries embed malicious instructions into contextual inputs (e.g., web pages, documents) to manipulate the backend LLM into executing attacker-desired tasks rather than user-intended ones. Prior work has proposed various attack strategies (heuristic and optimization-based) and defenses (detection-based and prevention-based).

**Limitations of Prior Work**: (1) No unified platform exists — different attacks, defenses, and benchmarks are implemented with heterogeneous configurations, precluding fair comparison. (2) Evaluations are insufficiently comprehensive — many defenses are assessed under narrow benchmark-attack combinations and later shown to be ineffective in alternative settings. (3) Attacks are overly static — all existing benchmarks use fixed-template attacks that fail to reflect real-world scenarios where adversaries iteratively refine their injections based on defense feedback.

**Key Challenge**: The absence of a unified evaluation ecosystem causes the true robustness of defenses to be overestimated — high performance reported under favorable evaluation conditions does not generalize to more diverse tasks or adaptive attack settings.

**Goal**: (1) Build a unified platform enabling plug-and-play evaluation of attacks, defenses, and benchmarks. (2) Design adaptive attack methods to assess the genuine robustness of defenses. (3) Comprehensively expose the limitations of existing defenses.

**Key Insight**: The paper elevates evaluation from isolated experiments to a platform ecosystem, providing standardized data formats, unified interfaces, and an extensible architecture that lowers the barrier for researchers to integrate and compare methods.

**Core Idea**: Unified platform + adaptive attacks + diverse realistic injection tasks = comprehensive stress testing of defense robustness.

## Method

### Overall Architecture
PIArena consists of four modules: (1) the **Benchmark** module provides diverse datasets (QA, RAG, summarization, long-form text, etc.); (2) the **Attack** module integrates multiple attack methods and generates injected prompts; (3) the **Defense** module integrates detection-based and prevention-based defenses; (4) the **Evaluator** module computes Utility (task performance) and ASR (attack success rate). All modules interact through a unified API, supporting both independent and combined evaluation.

### Key Designs

1. **Unified Standardized Interface and Data Format**

    - **Function**: Enables plug-and-play integration of attacks, defenses, and benchmarks.
    - **Mechanism**: Defines a unified sample structure comprising `target_inst`, `context`, `injected_task`, `target_task_answer`, `injected_task_answer`, and `category`. The attack interface takes a sample as input and outputs an injected prompt; the defense interface uniformly returns an LLM response (detection-based defenses first assess maliciousness and then block or pass, while prevention-based defenses directly generate safe responses). The evaluator computes identical metrics across all defense methods.
    - **Design Motivation**: Inconsistent benchmark formats and heterogeneous interfaces in prior work prevent fair comparison. PIArena's standardized design allows new methods to be implemented once and evaluated across multiple settings.

2. **Strategy-based Adaptive Attack**

    - **Function**: Iteratively optimizes injected prompts based on defense feedback in a black-box setting to assess the genuine robustness of defenses.
    - **Mechanism**: Operates in two stages — Stage 1 (candidate generation): ten rewriting strategies (e.g., disguising as "Author's Note," "System Update," etc.) are applied to rewrite the base injected prompt into multiple candidates. Stage 2 (feedback-guided optimization): the attack iteratively adjusts based on defense responses across three scenarios — increasing stealth when detected, increasing imperativeness when ignored, and applying general optimization otherwise. The process runs for up to $K$ iterations.
    - **Design Motivation**: Static attacks fail to expose the true weaknesses of defenses. Adaptive attacks achieve a "warm start" through strategy-level semantic rewriting rather than gradient-based optimization, making the approach far more efficient than brute-force search while ensuring attack diversity.

3. **Realistic and Diverse Injection Task Design**

    - **Function**: Simulates real-world attack objectives rather than trivial instructions such as "Print Hacked!"
    - **Mechanism**: Four categories of realistic injection tasks are designed — (a) phishing injection: embedding malicious URLs; (b) content promotion: inserting advertisements or product recommendations; (c) access denial: impersonating API quota exhaustion or account expiration; (d) infrastructure failure: simulating system errors such as out-of-memory exceptions or database timeouts. Each injection task is generated by an LLM conditioned on the target context to ensure contextual relevance.
    - **Design Motivation**: Existing benchmarks rely on context-agnostic simple injection tasks, whereas real-world adversaries carefully craft injection content that blends seamlessly with the surrounding context.

### Loss & Training
PIArena does not involve model training. The adaptive attack uses an LLM as the rewriting engine with no gradient optimization, operating in a fully black-box manner.

## Key Experimental Results

### Main Results (SQuAD v2, GPT-4o backend)

| Defense Method | Type | No-Attack Utility | Combined ASR | Strategy ASR |
|---|---|---|---|---|
| No Defense | — | 1.0 | 0.97 | 1.00 |
| PISanitizer | Prevention | 0.99 | 0.01 | 0.85 |
| SecAlign++ | Prevention | 0.84 | 0.01 | 0.09 |
| DataFilter | Prevention | 0.99 | 0.24 | 0.93 |
| PromptArmor | Prevention | 1.0 | 0.60 | 1.00 |
| PIGuard | Detection | 1.0 | 0.0 | 0.71 |
| Attn.Tracker | Detection | 0.61 | 0.0 | 0.0 |

### Ablation Study (Comparison Across Attack Types)

| Attack Type | Characteristics | ASR (No Defense) | ASR (PISanitizer) |
|---|---|---|---|
| Direct | Direct instruction injection | 0.86 | 0.04 |
| Combined | Mixture of multiple attacks | 0.97 | 0.01 |
| Strategy | Adaptive strategy-based attack | 1.00 | 0.85 |

### Key Findings
- **Poor generalization**: PISanitizer achieves strong performance on SQuAD (ASR 0.01) but its ASR surges to 0.85 under strategy-based adaptive attacks, revealing extreme vulnerability to adaptive adversaries.
- **Closed-source models remain unsafe**: GPT-5, Claude-Sonnet-4.5, and Gemini-3-Pro all exhibit high ASR under prompt injection.
- **Task-aligned injection is the fundamental challenge**: When the injected task and the target task belong to the same category (e.g., both QA), the attack degenerates into a misinformation problem that existing defenses are almost entirely unable to handle.
- Although Attn.Tracker achieves ASR = 0 against all attacks, its Utility is severely degraded (only 0.61), indicating a high rate of false positives.

## Highlights & Insights
- The most significant contribution of this paper is the adoption of a **platform-oriented** rather than a **method-oriented** perspective: rather than proposing a new defense, it constructs an ecosystem in which all defenses can be evaluated fairly and comprehensively. Such infrastructure-level contributions are critical for the progress of the field.
- The strategy-level semantic rewriting approach in the adaptive attack elegantly addresses the cold-start problem of black-box optimization — rewriting injected prompts as plausible contextual content (e.g., "editorial notes," "system updates") is far more efficient than random perturbation.
- The finding that **task-aligned injection scenarios are fundamentally undefendable** carries significant implications — when the injected task shares the same type as the target task, distinguishing legitimate instructions from malicious injections is inherently ambiguous in principle.

## Limitations & Future Work
- The adaptive attack still relies on an LLM as the rewriting engine, which incurs non-trivial costs at evaluation scale.
- Current benchmarks primarily cover text-based tasks; multimodal scenarios (e.g., prompt injections embedded in images) are not addressed.
- Task-aligned injection is identified as a fundamental challenge, but no mitigation directions are proposed.
- Evaluation is conducted mainly with a GPT-4o backend; the effect of varying backend LLMs on defense performance warrants further exploration.

## Related Work & Insights
- **vs. BIPIA (Yi et al. 2025)**: BIPIA provides benchmark datasets and evaluates defenses but relies on static attacks and lacks a unified interface. PIArena supports adaptive attacks and a plug-and-play toolbox.
- **vs. AgentDojo (Debenedetti et al. 2024)**: AgentDojo targets agent-specific scenarios with complex configurations and does not support defense evaluation. PIArena covers general LLM tasks with a concise interface.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The platform-contribution paradigm is innovative, and the adaptive attack design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven defenses × multiple attack types × multiple benchmarks × closed-source model evaluation — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and the threat model is rigorously defined, though the body text is somewhat dense due to the large number of tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] vCache: Verified Semantic Prompt Caching](../../ICLR2026/llm_evaluation/vcache_verified_semantic_prompt_caching.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_models.md)

</div>

<!-- RELATED:END -->
