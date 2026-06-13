---
title: >-
  [Paper Note] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs
description: >-
  [ACL 2026][LLM Reasoning][Content Moderation] The authors propose ChAIRO, a Contextual Hierarchical Analogical Induction and Reasoning Optimization framework. Through a three-stage pipeline (Analogy Case Generation → Rul…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Content Moderation"
  - "Rule Induction"
  - "Analogical Reasoning"
  - "Hierarchical Chain-of-Thought"
  - "End-to-end Optimization"
date: 2026-05-08
content_hash: cd72e43ad1f7756b
---

# ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.10502](https://arxiv.org/abs/2604.10502)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Content Moderation, Rule Induction, Analogical Reasoning, Hierarchical Chain-of-Thought, End-to-end Optimization

## TL;DR
The authors propose ChAIRO, a Contextual Hierarchical Analogical Induction and Reasoning Optimization framework. Through a three-stage pipeline (Analogy Case Generation → Rule Induction → Rule-injected Fine-tuning), it enables LLMs to autonomously generate analogy cases and induce explicit moderation rules in content moderation tasks, achieving a 4.5% F1 improvement over single-instance rule generation and 2.3% over static RAG.

## Background & Motivation

**Background**: Using LLMs for content moderation has become a promising direction, providing interpretable moderation decisions through the generation of reasoning chains. However, even SOTA models frequently fail in scenarios with ambiguous context or unclear moderation criteria.

**Limitations of Prior Work**: (1) CoT reasoning in content moderation lacks reference to precedents and relies solely on explicit criteria (e.g., presence of insults/incitement), failing to identify implicit discriminatory logic (e.g., metaphorical discrimination such as "low scores equal low ability"); (2) Manually defined high-level rules (e.g., "pornography") are too coarse to cover fine-grained nuances; (3) LLM-driven adaptive rule discovery relies on general priors, ignoring the domain expertise accumulated by human moderation experts.

**Key Challenge**: There is a need for precise, context-relevant moderation rules to handle ambiguous cases, but the construction and discovery of rules themselves are difficult—manual enumeration is impractical, and automated generation lacks precision.

**Goal**: To utilize analogy cases to improve rule induction quality and unify case retrieval, rule generation, and moderation decisions through end-to-end optimization.

**Key Insight**: Unlike CarO (a sibling work, arXiv:2604.10504), ChAIRO does not use DPO but introduces an explicit rule induction step—first using an auxiliary reasoning model to induce textual moderation rules from analogy cases, and then injecting the rules into the reasoning chain for a second round of fine-tuning.

**Core Idea**: A three-stage hierarchical optimization: (1) Analogy Chain SFT to train the model to generate analogy cases; (2) Auxiliary model to induce explicit rules from the analogy cases; (3) Structure the analogy cases, rules, and reasoning chains into a hierarchical format (`<RULE>+<ANALOGY>+<REASONING>`) for a second round of SFT, fusing the three layers of capabilities.

## Method

### Overall Architecture
Stage 1: Retrieve semantically similar cases for each training sample → Use LLM to generate reasoning chains containing analogies → Perform SFT to enable the model to autonomously generate analogy cases. Stage 2: Use the Stage 1 model to generate virtual analogy cases for each sample → Use an auxiliary reasoning model (QwQ-32B) to induce explicit moderation rules from the analogy cases. Stage 3: Structure the analogy cases + rules + reasoning chains into a hierarchical format (`<RULE>+<ANALOGY>+<REASONING>`) → Perform the second round of SFT.

### Key Designs

1.  **Self-augmented Analogy Chain Generation (Stage 1)**:
    - **Function**: Internalize analogical reasoning capabilities so the model can autonomously generate relevant analogies for new samples.
    - **Mechanism**: Encode all training samples using BGE-M3 and retrieve semantically similar cases for each. Input the sample + retrieved cases + labels into the LLM to generate analogy reasoning chains, followed by SFT. The trained model can then generate analogies autonomously without external retrieval.
    - **Design Motivation**: Cases retrieved via static RAG may not be the most suitable for the current sample; internalizing through SFT allows the model to dynamically generate more relevant analogies.

2.  **Auxiliary Model Rule Induction (Stage 2)**:
    - **Function**: Extract explicit, interpretable moderation rules from analogy cases.
    - **Mechanism**: Use the Stage 1 model to generate virtual analogy cases for each training sample, then use QwQ-32B as an auxiliary reasoning model to induce textual moderation rules from the original sample + analogy cases. Automatically verify if the category description in the rules aligns with the labels, discarding inconsistent samples.
    - **Design Motivation**: Analogy cases provide context, making the induced rules more precise and targeted, resulting in higher quality than rules generated from single samples (+4.5% F1).

3.  **Hierarchical Rule Injection and Final Fine-tuning (Stage 3)**:
    - **Function**: Integrate analogies, rules, and reasoning into a unified structural reasoning capability.
    - **Mechanism**: Use special tokens to structure the reasoning chain into three layers: `<RULE>` (induced rules), `<ANALOGY>` (analogy cases), and `<REASONING>` (comprehensive reasoning). Perform a second round of SFT based on Stage 1 parameters.
    - **Design Motivation**: The hierarchical structure allows the model to explicitly know when to use rules, when to refer to analogies, and when to perform reasoning, improving interpretability and consistency.

## Key Experimental Results

### Main Results (Chinese Moderation Dataset)

| Method | Avg F1 | Politics | Porn | Violence | Gambling | Bias | Harmless |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek R1 | 77.1 | 72.7 | 91.4 | 86.1 | 94.3 | 64.6 | 59.7 |
| DeepSeek V3 | 80.3 | 79.0 | 90.3 | 89.8 | 95.0 | 70.5 | 62.5 |
| Naive SFT | ~85 | - | - | - | - | - | - |
| Rule-injected SFT (Single Instance) | ~85.7 | - | - | - | - | - | - |
| Static RAG | ~87.9 | - | - | - | - | - | - |
| **ChAIRO (Ours)** | **~90.2** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Comparison | F1 Gain | Description |
| :--- | :--- | :--- |
| ChAIRO vs Naive SFT | +5.3% | Value of explicit rules |
| ChAIRO vs Single-instance Rule SFT | +4.5% | Analogy cases improve rule quality |
| ChAIRO vs Static RAG | +2.3% | End-to-end optimization vs stage-wise |

### Key Findings
- **Explicit rule injection brings a 5.3% gain**, proving the critical role of rules in ambiguous moderation cases.
- **Analogy-driven rules outperform single-instance rules by 4.5%**, indicating that contextual analogies indeed improve rule quality.
- **End-to-end optimization outperforms static RAG by 2.3%**, as errors in stage-wise pipelines accumulate.
- **Human evaluation confirms higher rule quality**: Clarity, interpretability, and applicability are superior to baselines.
- **External model generalization tests passed**: Rules are transferable to other LLMs.

## Highlights & Insights
- The **"Analogy → Rule → Reasoning" three-layer cognitive architecture** simulates the decision-making process of human experts, adding a layer of explicit knowledge abstraction compared to CarO's "Analogy → Reasoning," which enhances interpretability.
- The **Hierarchical Reasoning Chain format** (`<RULE>+<ANALOGY>+<REASONING>`) provides structured auditing clues, where every decision can be traced back to specific rules and analogy cases.
- **Complementary relationship with CarO**: CarO uses DPO to strengthen the consistency of analogical reasoning, while ChAIRO uses rule induction to improve reasoning interpretability. The two can be combined.

## Limitations & Future Work
- Requires an auxiliary reasoning model (QwQ-32B) for rule induction, increasing training costs.
- Rules are textual, and formal consistency/non-contradiction cannot be guaranteed.
- The training workflow of two SFT rounds is complex; it remains to be seen if it can be simplified.
- Primarily focused on Chinese data; validation for English and multilingual scenarios is insufficient.
- The rule base lacks a continuous update mechanism; new types of violations require retraining.

## Related Work & Insights
- **vs CarO (2604.10504)**: Sibling work. CarO uses DPO to strengthen analogy reasoning, while ChAIRO introduces explicit rule induction. ChAIRO emphasizes interpretability.
- **vs Rule-based Moderation**: Traditional rules are manually defined coarse-grained standards. ChAIRO's rules are fine-grained, context-relevant rules automatically induced from analogy cases.
- **vs Kumar et al. (2024)**: Also performs LLM rule discovery but based on single-instance context. ChAIRO provides a richer induction basis through analogy cases.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage hierarchical framework is well-designed, though highly related to CarO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional ablation, human evaluation, and external model generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-designed RQ-driven experiments.
- Value: ⭐⭐⭐⭐ Explicit rule induction has practical value for interpretable moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](../../ICLR2026/llm_reasoning/from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [\[ICLR 2026\] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](../../ICLR2026/llm_reasoning/thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)
- [\[ACL 2026\] Strategy-Induct: Task-Level Strategy Induction for Instruction Generation](strategy-induct_task-level_strategy_induction_for_instruction_generation.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)

</div>

<!-- RELATED:END -->
