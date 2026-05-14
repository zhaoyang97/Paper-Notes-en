---
title: >-
  [Paper Note] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System
description: >-
  [AAAI 2026][Social Computing][Poisoning Attack] This paper proposes Fact2Fiction, the first poisoning attack framework targeting agentic fact-checking systems (e.g., DEFAME…
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Poisoning Attack"
  - "Agentic Fact-checking"
  - "Claim Decomposition"
  - "Adversarial Evidence"
  - "Knowledge Base Poisoning"
date: 2026-05-08
content_hash: c542a8972be16a1c
---

# Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System

**Conference**: AAAI 2026
**arXiv**: [2508.06059](https://arxiv.org/abs/2508.06059)
**Code**: [https://trustworthycomp.github.io/Fact2Fiction/](https://trustworthycomp.github.io/Fact2Fiction/)
**Area**: AI Security / LLM Agent / Fact-checking Attack
**Keywords**: Poisoning Attack, Agentic Fact-checking, Claim Decomposition, Adversarial Evidence, Knowledge Base Poisoning

## TL;DR
This paper proposes Fact2Fiction, the first poisoning attack framework targeting agentic fact-checking systems (e.g., DEFAME, InFact). It employs a Planner Agent to simulate claim decomposition and generate sub-questions, reverse-engineers key reasoning points from system justifications to craft targeted malicious evidence, and allocates the poisoning budget according to sub-claim importance. At a poisoning rate of only 1%, Fact2Fiction achieves 8.9%–21.2% higher attack success rate (ASR) than the state-of-the-art PoisonedRAG.

## Background & Motivation

**Background**: State-of-the-art fact-checking systems (InFact, DEFAME) have evolved from simple RAG pipelines to agentic paradigms—LLM agents decompose complex claims into sub-claims, retrieve and verify evidence for each, and aggregate results to produce a final verdict along with a justification. This decomposition strategy inherently serves as a natural defense against conventional poisoning attacks.

**Limitations of Prior Work**: Existing poisoning attacks (e.g., PoisonedRAG) craft malicious evidence targeting only the main claim. When an agentic system splits a claim into multiple specific sub-questions, such generic malicious evidence is neither easily retrieved by sub-question-oriented retrievers nor sufficiently specific to mislead sub-claim verification. For instance, PoisonedRAG's ASR on DEFAME drops from 57.4% (on a simple system) to 42.4%.

**Key Challenge**: Agentic fact-checking systems improve accuracy, yet their justifications (transparent explanations) expose the key evidence and reasoning patterns underlying their decisions—creating a fundamental tension between transparency and security.

**Goal**: How can effective poisoning attacks be designed against agentic fact-checking systems? Three challenges must be addressed: (a) semantic alignment between malicious evidence and sub-claims; (b) leveraging justifications for targeted attacks; (c) optimal allocation of a limited poisoning budget.

**Key Insight**: Reverse-engineer two properties of agentic systems—claim decomposition strategy and justification output. Two collaborative agents (Planner + Executor) are used to simulate the decomposition process and exploit transparent explanations to craft targeted malicious content.

**Core Idea**: Simulate the claim decomposition strategy of agentic fact-checking systems + reverse-engineer justifications to craft targeted malicious evidence + allocate the poisoning budget according to sub-claim importance.

## Method

### Overall Architecture
Fact2Fiction consists of two LLM agents: (1) a **Planner** responsible for attack planning (claim decomposition → answer planning → budget planning → query planning); and (2) an **Executor** that generates malicious evidence and injects it into the knowledge base. Prior to the attack, the target system is queried to obtain an initial verdict and justification.

### Key Designs

1. **Sub-question Decomposition**:

    - Function: Simulates the decomposition strategy of agentic fact-checking systems by splitting the target claim into up to 10 sub-questions.
    - Mechanism: The Planner role-plays as a fact-checker and generates a surrogate sub-question set $\mathcal{Q}_i = \{q_1, \ldots, q_{l_i}\}$ for claim $c_i$. Although the exact decomposition of the target system is unknown, experiments demonstrate that surrogate decomposition is effective across different decomposition strategies (InFact's explicit decomposition vs. DEFAME's implicit dynamic decomposition).
    - Design Motivation: Malicious evidence crafted for sub-questions is more likely to be retrieved and more precisely targeted than evidence crafted for the main claim.

2. **Answer Planning — Leveraging Justifications**:

    - Function: Generates targeted adversarial answers for each sub-question that directly rebut the key reasoning in the justification.
    - Mechanism: Given justification $j_i$, an answer $a_k$ is generated for each $q_k$ to precisely rebut the key arguments in $j_i$. For example, if the justification states "this bill protects individuals' right to grow food," the adversarial answer asserts "this bill imposes strict registration requirements on food sharing and trading"—a pointed rebuttal of specific arguments rather than a vague negation.
    - Design Motivation: The justification exposes the key evidence and reasoning paths that led the system to a correct verdict, allowing attackers to precisely undermine these critical supports.

3. **Budget Planning**:

    - Function: Optimally allocates a limited poisoning budget across sub-questions.
    - Mechanism: The Planner assigns an importance score $w_k$ (0–10) to each pair $(q_k, a_k)$ based on its relevance to the justification, then allocates the budget proportionally: $m_k = \lceil m \cdot w_k / \sum_s w_s \rceil$. More malicious evidence is prioritized for the sub-claims most critical to the system's reasoning.
    - Design Motivation: Under a low budget (1% poisoning rate), uniform allocation wastes resources on unimportant sub-questions. Ablation experiments confirm that budget planning contributes 7.8% ASR at the 1% poisoning rate.

4. **Query Planning**:

    - Function: Generates surrogate search queries for each sub-question and concatenates them with malicious evidence to improve retrievability.
    - Mechanism: $e_{k,h} = s_p \oplus \tilde{e}_{k,h}$, where a query is prepended to the malicious evidence so that it is more readily retrieved during semantic search (since the target system uses similar queries during retrieval).
    - Design Motivation: Malicious evidence must first be retrieved to take effect. Sub-question-specific queries match retrieval patterns more precisely than the main claim.

## Key Experimental Results

### Main Results (1% Poisoning Rate, ~8 Malicious Documents/Claim)

| Attack Method | DEFAME ASR | InFact ASR | Simple ASR |
|---|---|---|---|
| Naive | 17.8% | 14.6% | 24.2% |
| Prompt Injection | 19.3% | 16.1% | 34.7% |
| PoisonedRAG | 33.5% | 35.8% | 42.4% |
| **Fact2Fiction** | **42.4%** | **46.0%** | **43.4%** |

### Attack Efficiency Comparison

| Metric | PoisonedRAG | Fact2Fiction | Note |
|---|---|---|---|
| Reaching ~42% ASR (DEFAME) | Requires 8% poisoning rate | Only 1% needed | 8× more efficient |
| Reaching ~45% ASR (InFact) | Requires 8% poisoning rate | Only 1% needed | 8–16× more efficient |

### Ablation Study (1% Poisoning Rate, DEFAME)

| Configuration | ASR | Note |
|---|---|---|
| Fact2Fiction (full) | 42.4% | Best |
| w/o Answer Planning | 40.9% | No justification-based answer crafting |
| w/o Budget Planning | 34.6% | Uniform budget allocation, −7.8% |
| w/o Query Planning | 39.4% | No sub-question query concatenation |

### Key Findings
- **Agentic decomposition provides genuine natural defense**: PoisonedRAG achieves 57.4% ASR on Simple systems, but drops to 42–45% on DEFAME/InFact. Fact2Fiction successfully breaches this defense.
- **Justification is a double-edged sword (transparency–security tension)**: Answer Planning (leveraging justifications) contributes up to 12.4% ASR improvement under low budget conditions.
- **Evidence quality matters more than retrievability**: At 1% poisoning rate, Fact2Fiction's SIR (64.8%) is slightly lower than PoisonedRAG's (65.6%), yet its ASR (42.4%) is substantially higher (vs. 33.5%), demonstrating the superior quality of targeted malicious evidence.
- **Attacks exhibit saturation points**: Naive/Prompt Injection saturate at 2%; PoisonedRAG saturates at 4–8%; Fact2Fiction continues to grow through 8%—its targeted strategy more effectively exploits additional budget.
- **Existing defenses (paraphrasing, malice detection, PPL detection) are all ineffective**: Malicious evidence generated by Fact2Fiction overlaps with clean evidence in PPL distribution and cannot be detected via statistical methods.

## Highlights & Insights
- The **"transparency–security tension"** is a profound finding—the more transparent a fact-checking system is (i.e., the more detailed justifications it provides), the more susceptible it becomes to targeted attacks. This has important implications for AI interpretability and trustworthiness research.
- The **simulate-decompose + exploit-justification attack strategy** cleverly turns two properties of agentic systems against themselves: the decomposition strategy is exploited to craft semantically aligned malicious content, while transparent explanations reveal the optimal attack targets.
- The **effectiveness of budget planning** reveals an important asymmetry between attack and defense—attackers can concentrate resources on the most critical sub-claims, whereas defenders must protect all sub-claims equally.
- The paper provides **constructive defensive suggestions from an attack perspective**, explicitly calling for future research into robust fact-checking systems that force attacks to saturate at low ASR.

## Limitations & Future Work
- The attack requires querying the target system to obtain justifications beforehand, which may be infeasible under restricted access scenarios (e.g., rate limiting).
- Evaluation is limited to the AVeriTeC dataset (500 claims); larger-scale validation is lacking.
- Only GPT-4o-mini is used as the primary LLM backbone; though the appendix extends experiments to Gemini/DeepSeek, the scope remains limited.
- The paper focuses exclusively on attacking and does not propose effective defenses—it only demonstrates that existing defenses are insufficient.
- Sub-question decomposition depends on LLM quality; if the Planner's decomposition diverges significantly from that of the target system, attack effectiveness may degrade.

## Related Work & Insights
- **vs. PoisonedRAG**: PoisonedRAG crafts generic malicious evidence targeting only the main claim, whereas Fact2Fiction performs targeted attacks on sub-claims, achieving 8.9%–21.2% higher ASR and 8–16× greater efficiency.
- **vs. Prompt Injection**: Prompt Injection is nearly ineffective against agentic systems (ASR ~19%), as malicious instructions differ too semantically from normal evidence to be retrieved.
- **Implications for fact-checking system design**: (a) Justifications should limit the disclosure of reasoning details; (b) Sub-claim verification should incorporate redundancy and cross-validation; (c) Retrieval components should be made more adversarially robust.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First attack framework targeting agentic fact-checking systems; the "transparency–security tension" is a profound insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 victim systems × 4 poisoning rates × 4 baselines + complete ablation + 3 defenses + robustness evaluation across retrievers/LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ — Threat model is clearly defined; EQ-driven experimental design is well-structured; examples are vivid and intuitive.
- Value: ⭐⭐⭐⭐⭐ — Significant for both the AI security and fact-checking communities; exposes a new attack surface in agentic systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)

</div>

<!-- RELATED:END -->
