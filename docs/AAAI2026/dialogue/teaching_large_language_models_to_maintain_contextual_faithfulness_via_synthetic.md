---
title: >-
  [Paper Note] Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL
description: >-
  [AAAI 2026][Dialogue Systems][Contextual faithfulness] This paper proposes the Canoe framework, which synthesizes four types of verifiable short-form QA data from Wikidata triples and applies Dual-GRPO (incorporating acc…
tags:
  - "AAAI 2026"
  - "Dialogue Systems"
  - "Contextual faithfulness"
  - "reinforcement learning"
  - "synthetic data"
  - "GRPO"
  - "hallucination suppression"
date: 2026-05-08
content_hash: f48c76fe9709f3c6
---

# Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL

**Conference**: AAAI 2026
**arXiv**: [2505.16483](https://arxiv.org/abs/2505.16483)
**Code**: [GitHub](https://github.com/S1s-Z/CANOE)
**Area**: Dialogue Systems
**Keywords**: Contextual faithfulness, reinforcement learning, synthetic data, GRPO, hallucination suppression

## TL;DR
This paper proposes the Canoe framework, which synthesizes four types of verifiable short-form QA data from Wikidata triples and applies Dual-GRPO (incorporating accuracy reward, long-form proxy reward, and format reward) to jointly optimize faithfulness in both short- and long-form generation. The approach improves Llama-3-8B by an average of 22.6% across 11 downstream tasks, surpassing GPT-4o.

## Background & Motivation

**Background**: LLMs are widely deployed in context-grounded information retrieval systems (e.g., document QA, text summarization), yet they frequently generate responses inconsistent with the provided context—a phenomenon known as faithfulness hallucination—which undermines system trustworthiness.

**Limitations of Prior Work**: (a) **Faithfulness does not improve with scale**—larger models possess more parametric knowledge and are thus more prone to knowledge conflicts with the provided context, leading them to disregard it; (b) **Existing methods are task-specific**—for instance, Context-DPO improves short-form QA faithfulness but fails to generalize to tasks such as summarization; (c) **Long-form faithfulness data is difficult to annotate**—short-form QA has reference answers that enable rule-based verification, whereas faithfulness in long-form generation cannot be verified by rules, and manual annotation does not scale.

**Key Challenge**: Short-form data is easy to verify but cannot directly train long-form faithfulness; applying GRPO with short-form data alone causes the model to learn a shortcut of copying text spans from the context, thereby losing long-form generation capability.

**Key Insight**: Design a proxy reward that uses the correctness of short-form answers to indirectly assess the faithfulness of long-form responses—if a long-form response can guide the model to correctly answer a short-form question, it is considered faithful.

**Core Idea**: Synthesize verifiable short-form QA data across four diverse task types, and apply Dual-GRPO with three rule-based rewards to jointly optimize short- and long-form generation, improving faithfulness across 11 tasks without any manual annotation.

## Method

### Overall Architecture
The framework consists of two components: (1) synthesizing training data (10K pairs) from the Wikidata knowledge base; and (2) RL training via Dual-GRPO with rule-based rewards.

### Key Designs

1. **Synthetic Data Generation (4 QA Task Types)**:

    - Data source: 30K Wikidata head-relation-tail triples $(h, r, t)$; GPT-4o is used to generate contexts and questions, with the tail entity $t$ serving as the answer to ensure correctness.
    - **Direct context**: The context explicitly contains the answer, testing information localization ability.
    - **Reasoning context**: 2–4-hop subgraph paths are constructed, requiring multi-step reasoning.
    - **Inconsistent context**: Multiple irrelevant contexts are mixed, requiring the model to filter noise and focus on relevant information.
    - **Counterfactual context**: The answer is replaced with a similar but incorrect entity, forcing the model to rely on the context rather than parametric knowledge.
    - Design Motivation: The four task types cover distinct faithfulness challenges, ensuring complexity and diversity in the training data.

2. **Dual-GRPO (Core Training Method)**:

    - Function: Simultaneously optimizes the generation of both short-form and long-form responses within the GRPO framework.
    - **System prompt design**: The model is required to sequentially produce a reasoning process → a long-form answer (detailed sentences) → a short-form answer (a few words); rewards are computed separately for both outputs.
    - **Three rule-based rewards**:
        - **Accuracy reward** (short-form): Exact match (EM) between the short-form answer and the ground truth; score is 1 or 0.
        - **Proxy reward** (long-form; core contribution): The generated long-form answer $y_{lf}$ replaces the original context, and the LLM is re-queried to answer the short-form question—if the correct answer is still produced, the long-form response is deemed faithful and complete (score 1); otherwise 0.
        - **Format reward**: Checks whether the output conforms to the `<think>/<long_answer>/<short_answer>` structure.
    - Final reward = sum of the three rewards.
    - Design Motivation: The proxy reward elegantly transforms the problem of "long-form faithfulness"—which is difficult to evaluate directly—into a verifiable short-form correctness problem. If a long-form response can "teach" the model to arrive at the correct answer, it must be both faithful and comprehensible.

3. **Why Not Train Directly with Short-Form Data**:

    - Preliminary experiments reveal that training solely with short-form QA data and an accuracy reward causes the model to learn a shortcut of copying text spans from the context, achieving high scores on short-form tasks while completely losing long-form generation capability.
    - Dual-GRPO avoids this overfitting issue by jointly optimizing both output formats.

## Key Experimental Results

### Main Results (11 Downstream Tasks)

| Model | Short-form EM (avg) | Short-form Acc (avg) | Long-form (avg) | Overall (avg) |
|-------|---------------------|----------------------|-----------------|---------------|
| Llama-3-8B (vanilla) | 49.2 | 58.3 | ~44 | 47.7 |
| Context-DPO-8B | 66.3 | 72.9 | ~54 | 59.8 |
| SCOPEsum-8B | 35.7 | 64.6 | ~59 | 63.3 |
| **Canoe-Llama-8B** | **73.5** | **80.9** | **~65** | **70.3** |
| GPT-4o | - | - | - | 58.8 |
| OpenAI o1 | - | - | - | ~62 |

### Ablation Study

| Configuration | Short-form EM | Long-form Quality | Note |
|---------------|---------------|-------------------|------|
| Full Canoe (Dual-GRPO) | 67.7 | High | Full model |
| Accuracy reward only (vanilla GRPO) | 60.5 | 23.5% QualityScore | "Text copying" shortcut emerges |
| w/o counterfactual context | 62.6 | - | −5.1 pp |
| w/o reasoning context | 63.7 | - | −4.0 pp |
| w/o inconsistent context | 64.4 | - | −3.3 pp |

### Key Findings
- **7B model surpasses GPT-4o**: Canoe-Llama-8B achieves an overall average of 70.3 vs. 58.8 for GPT-4o; Qwen-2.5-7B + Canoe reaches 68.0, also exceeding GPT-4o—demonstrating that targeted RL training is more effective than simply scaling model size.
- **Dual-GRPO is critical for preventing degeneration**: Without the proxy reward, the model learns to copy text spans from the context as a shortcut, yielding a long-form quality score of only 23.5%; incorporating the proxy reward substantially recovers long-form quality.
- **All four synthetic data types are necessary**: Counterfactual context contributes the most (−5.1 pp), as it most directly forces the model to rely on the context rather than parametric knowledge.
- **Multi-hop reasoning ability improves significantly**: EM on ConFiQA increases from 49.2 to 73.5 (+24.3), with reasoning-type context training being particularly effective.
- **Overconfidence is alleviated**: Canoe assigns higher perplexity (lower confidence) to unfaithful samples, indicating that the model learns to "hesitate" under uncertainty.

## Highlights & Insights
- The proxy reward design is the most elegant contribution of this paper—it transforms the open-ended, rule-intractable problem of "whether a long-form response is faithful" into the verifiable question of "whether substituting the long-form response for the context still leads to a correct short-form answer." This idea is generalizable to other scenarios where direct evaluation is difficult but indirect verification is feasible.
- The method is entirely free of manual annotation and reward models—synthetic data are derived from a knowledge base (guaranteeing correctness), and all rewards are rule-based. This makes the approach highly reproducible and transferable to new domains, provided a structured knowledge base is available.

## Limitations & Future Work
- Synthetic data generation relies on a structured knowledge base (Wikidata), making the approach difficult to apply directly in domains lacking structured KBs (e.g., literature, law).
- The proxy reward assumption has limitations—"inducing a correct short-form answer implies faithfulness" does not always hold; a long-form response may happen to contain the correct information while being globally unfaithful.
- Experiments are conducted exclusively in English; performance in multilingual settings remains unknown.
- Training costs are not reported in detail (GRPO requires generating multiple candidates per input).
- The quality of entity substitution in counterfactual contexts depends on GPT-4o's capabilities.

## Related Work & Insights
- **vs. Context-DPO**: Uses DPO to align short-form faithfulness but cannot improve long-form faithfulness. Canoe jointly optimizes both via the proxy reward in Dual-GRPO.
- **vs. RLHF/DPO**: Traditional approaches require manually annotated preference data or a trained reward model. Canoe relies entirely on rule-based rewards, making it more lightweight and scalable.
- **vs. Self-RAG**: Self-RAG improves faithfulness through self-reflection labels but requires task-specific labeled training data. Canoe is more general and consistently effective across 11 tasks.
- **vs. SCOPEsum**: Focuses on faithfulness improvement in summarization; Canoe is effective on both short-form and long-form tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The proxy reward design in Dual-GRPO is exceptionally elegant, addressing the core challenge of rule-based evaluation for long-form faithfulness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 downstream tasks, multiple base models, detailed ablations, and per-task-type contribution analysis.
- Writing Quality: ⭐⭐⭐⭐ The motivation and derivation of the method are clearly presented, with a well-justified rationale for the proxy reward design.
- Value: ⭐⭐⭐⭐⭐ A 7B model surpassing GPT-4o is a significant empirical finding with substantial implications for the open-source community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)
- [\[AAAI 2026\] Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](emergent_persuasion_will_llms_persuade_without_being_prompted.md)
- [\[NeurIPS 2025\] SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](../../NeurIPS2025/dialogue/sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](../../ACL2026/dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)
- [\[NeurIPS 2025\] AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](../../NeurIPS2025/dialogue/aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)

</div>

<!-- RELATED:END -->
