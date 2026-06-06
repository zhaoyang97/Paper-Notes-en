---
title: >-
  [Paper Note] PIArena: A Platform for Prompt Injection Evaluation
description: >-
  [ACL 2026][LLM Safety][Prompt Injection Attacks] This paper proposes PIArena, a unified and extensible platform for Prompt Injection (PI) evaluation. It integrates various SOTA attack and defense methods…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Prompt Injection Attacks"
  - "Defense Evaluation Platform"
  - "Adaptive Attacks"
  - "LLM Security"
  - "Benchmark Unification"
date: 2026-05-08
content_hash: 1df55b4932f0dcd1
---

# PIArena: A Platform for Prompt Injection Evaluation

**Conference**: ACL 2026  
**arXiv**: [2604.08499](https://arxiv.org/abs/2604.08499)  
**Code**: [https://github.com/sleeepeer/PIArena](https://github.com/sleeepeer/PIArena)  
**Area**: LLM Evaluation  
**Keywords**: Prompt Injection Attacks, Defense Evaluation Platform, Adaptive Attacks, LLM Security, Benchmark Unification

## TL;DR
This paper proposes PIArena, a unified and extensible platform for Prompt Injection (PI) evaluation. It integrates various SOTA attack and defense methods, supports plug-and-play evaluation, and introduces a strategy-based adaptive attack method to systematically reveal the critical limitations of existing defenses in generalization, adaptive scenarios, and task-alignment contexts.

## Background & Motivation

**Background**: Prompt injection attacks are ranked by OWASP as the top security risk for LLM applications. Attackers manipulate backend LLMs by injecting malicious instructions into the context (e.g., webpages, documents) to execute attacker-intended tasks instead of user-specified ones. Existing research has proposed various attack (heuristic/optimization-based) and defense (detection-based/prevention-based) methods.

**Limitations of Prior Work**: (1) Lack of a unified platform—different attacks, defenses, and benchmarks utilize inconsistent implementations, hindering fair comparison; (2) Incomplete evaluation—many defenses are evaluated only on specific benchmarks/attacks, often proving brittle in other settings; (3) Static attacks—existing benchmarks use fixed templates, which do not reflect real-world scenarios where attackers iteratively optimize based on defense feedback.

**Key Challenge**: The absence of a unified evaluation ecosystem leads to an overestimation of the robustness of defense methods; high performance reported under "favorable" evaluation conditions fails to generalize to more diverse tasks and adaptive attack scenarios.

**Goal**: (1) Build a unified platform for plug-and-play evaluation of attacks, defenses, and benchmarks; (2) Design adaptive attack methods to test the true robustness of defenses; (3) Comprehensively reveal the limitations of existing defenses.

**Key Insight**: Shift the evaluation paradigm from "individual experiments" to a "platform ecosystem" by providing standardized data formats, unified interfaces, and an extensible architecture to lower the barrier for integration and comparison.

**Core Idea**: Unified Platform + Adaptive Attacks + Diverse Real-world Injection Tasks = Comprehensive Stress Testing of Defense Robustness.

## Method

### Overall Architecture
PIArena consists of four modules: (1) Benchmark module providing diverse datasets (QA, RAG, Summarization, Long-form text, etc.); (2) Attack module integrating various attack methods to generate injected prompts; (3) Defense module integrating detection-based and prevention-based defenses; (4) Evaluator module calculating Utility (task performance) and ASR (Attack Success Rate). All modules interact through a unified API, supporting both independent and combined evaluations.

### Key Designs

1.  **Unified Standardized Interfaces and Data Formats**:
    - **Function**: Enables plug-and-play integration of attacks, defenses, and benchmarks.
    - **Mechanism**: Defines a unified data sample structure (`target_inst`, `context`, `injected_task`, `target_task_answer`, `injected_task_answer`, `category`). The attack interface takes samples as input and outputs injected prompts; the defense interface outputs LLM responses (detection defenses decide whether to block or allow, while prevention defenses generate safe responses). The evaluator computes identical metrics for all defenses.
    - **Design Motivation**: Inconsistent formats and interfaces in existing benchmarks prevent fair comparison. PIArena's standardization allows new methods to be "implemented once, evaluated everywhere."

2.  **Strategy-based Adaptive Attack**:
    - **Function**: Iteratively optimizes injected prompts based on defense feedback in black-box scenarios to test true robustness.
    - **Mechanism**: Operates in two stages—Stage 1 (Candidate Generation): Uses 10 rewriting strategies (e.g., masquerading as "Author’s Note" or "System Update") to rewrite basic injection prompts into multiple candidates; Stage 2 (Feedback-guided Optimization): Iteratively adjusts based on defense reactions in three scenarios—increasing stealth when detected, increasing imperativeness when ignored, and general optimization otherwise. Supports up to $K$ iterations.
    - **Design Motivation**: Static attacks fail to expose the true weaknesses of defenses. Strategy-based adaptive attacks use semantic rewriting (rather than gradient optimization) for "warm starts," which is more efficient than brute-force search and ensures attack diversity.

3.  **Real-world Diverse Injection Task Design**:
    - **Function**: Simulates real-world attack targets beyond simple "Print Hacked!" payloads.
    - **Mechanism**: Designs four categories of realistic injection tasks: (a) Phishing: Injecting malicious links; (b) Context Promotion: Embedding ads or recommendations; (c) Denial of Service (DoS): Masquerading as API quota exhaustion or account expiry; (d) Infrastructure Failure: Masquerading as system errors like memory overflow or database timeouts. Each task is generated by an LLM based on the target context to ensure contextual relevance.
    - **Design Motivation**: Simple, out-of-context injection tasks in existing benchmarks do not represent actual attackers who carefully blend injected content with the context.

### Loss & Training
PIArena itself does not involve training. The adaptive attacks utilize LLMs as rewriting engines through black-box operations without gradient-based optimization.

## Key Experimental Results

### Main Results (SQuAD v2, GPT-4o Backend)

| Defense Method | Type | Baseline Utility | Combined ASR | Strategy ASR |
| :--- | :--- | :--- | :--- | :--- |
| No Defense | - | 1.0 | 0.97 | 1.00 |
| PISanitizer | Prevention | 0.99 | 0.01 | 0.85 |
| SecAlign++ | Prevention | 0.84 | 0.01 | 0.09 |
| DataFilter | Prevention | 0.99 | 0.24 | 0.93 |
| PromptArmor | Prevention | 1.0 | 0.60 | 1.00 |
| PIGuard | Detection | 1.0 | 0.0 | 0.71 |
| Attn.Tracker | Detection | 0.61 | 0.0 | 0.0 |

### Ablation Study (Comparison of Attack Types)

| Attack Type | Characteristics | ASR (No Defense) | ASR (PISanitizer) |
| :--- | :--- | :--- | :--- |
| Direct | Direct instruction injection | 0.86 | 0.04 |
| Combined | Hybrid of multiple attacks | 0.97 | 0.01 |
| Strategy | Adaptive strategy-based attack | 1.00 | 0.85 |

### Key Findings
- **Poor Generalization**: PISanitizer performs excellently on SQuAD (ASR 0.01) but its ASR surges to 0.85 under Strategy attacks, indicating extreme vulnerability to adaptive methods.
- **Closed-source Models are Unsafe**: GPT-5, Claude-Sonnet-4.5, and Gemini-3-Pro still exhibit high ASR under prompt injection.
- **Task Alignment is a Fundamental Challenge**: When the injected task type matches the target task (e.g., both are QA), the attack degrades into a "misinformation" problem that existing defenses struggle to address.
- **Utility Trade-offs**: While Attn.Tracker achieves ASR=0 for all attacks, its Utility is severely compromised (only 0.61) due to a high false positive rate.

## Highlights & Insights
- **"Platform Thinking" over "Method Thinking"** is the primary contribution. Instead of proposing a new defense, the work builds an ecosystem for fair and comprehensive evaluation, which is critical for the field's advancement.
- The "Strategy-based semantic rewriting" approach effectively solves the cold-start problem in black-box optimization by reformatting prompts into plausible contexts (e.g., "Editor's Note"), proving more efficient than random perturbations.
- The insight that "Task-aligned scenarios are fundamentally indefensible" is profound—when injected and legitimate instructions are of the same type, distinguishing them becomes theoretically ambiguous.

## Limitations & Future Work
- Adaptive attacks require LLMs as rewriting engines, which introduces cost considerations for large-scale evaluations.
- Current benchmarks primarily cover text tasks; multimodal scenarios (e.g., prompts embedded in images) are not yet included.
- While task-aligned scenarios are identified as a fundamental difficulty, specific solutions were not proposed.
- Evaluations mainly use a GPT-4o backend; variations in defense effectiveness across different backend LLMs require further exploration.

## Related Work & Insights
- **vs. BIPIA (Yi et al. 2025)**: BIPIA provides datasets and evaluates defenses but uses static attacks and lacks a unified interface; PIArena supports adaptive attacks and a plug-and-play toolkit.
- **vs. AgentDojo (Debenedetti et al. 2024)**: AgentDojo targets Agent scenarios with complex configurations and lacks defense evaluation support; PIArena covers general LLM tasks with a concise interface.

## Rating
- Novelty: ⭐⭐⭐⭐ The platform-based contribution is innovative, and the adaptive attack design is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive across 7 defenses, multiple attacks, benchmarks, and closed-source models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous threat model definition, though the density of tables is high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] ACIArena: Toward Unified Evaluation for Agent Cascading Injection](aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
