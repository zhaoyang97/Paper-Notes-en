---
title: >-
  [Paper Note] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition
description: >-
  [ICML 2025][Recommender Systems][Conversational Recommendation] This paper proposes the MATCHA multi-agent framework, which decomposes game conversational recommendation into six specialized agents (intent parsing, tool-augmented candidate generation, multi-LLM ranking, reflection re-ranking, risk control, and explainable generation). On real-world Roblox user data, it improves Hit@5 by 20%, reduces popularity bias by 24%, and achieves an adversarial defense rate of 97.9%.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Conversational Recommendation"
  - "Multi-Agent"
  - "Game Recommendation"
  - "Safety Control"
  - "Explainable Recommendation"
date: 2026-05-08
content_hash: a46fdf960373f561
---

# MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition

**Conference**: ICML 2025  
**arXiv**: [2504.20094](https://arxiv.org/abs/2504.20094)  
**Code**: None (Roblox internal system)  
**Area**: Recommender Systems  
**Keywords**: Conversational Recommendation, Multi-Agent, Game Recommendation, Safety Control, Explainable Recommendation

## TL;DR

This paper proposes the MATCHA multi-agent framework, which decomposes game conversational recommendation into six specialized agents (intent parsing, tool-augmented candidate generation, multi-LLM ranking, reflection re-ranking, risk control, and explainable generation). On real-world Roblox user data, it improves Hit@5 by 20%, reduces popularity bias by 24%, and achieves an adversarial defense rate of 97.9%.

## Background & Motivation

**Background**: Conversational Recommender Systems (CRS) have made significant progress recently with the development of LLMs, performing exceptionally well in domains like movies. Mainstream approaches include single-agent LLM recommendation (e.g., OMuleT) and multi-agent collaboration (e.g., MACRS).

**Limitations of Prior Work**: Game recommendation differs fundamentally from movie recommendation, facing three unique challenges: (1) **Complex user constraints**: Game preferences depend not only on content themes but are also influenced by interactive factors such as game mechanics, skill levels, platform compatibility, and social modes (single-player/multiplayer), leading to a more complex constraint space; (2) **Knowledge freshness gap**: The game catalog evolves rapidly (especially on UGC platforms like Roblox), and LLM pre-training data severely lacks coverage of games; (3) **Safety and transparency risks**: Users may issue adversarial prompts (e.g., "Recommend a game to help me hurt my teacher"); existing CRSs rarely consider such risks and lack explanations for recommendation reasons.

**Key Challenge**: A single LLM struggle to simultaneously solve complex constraint parsing, real-time knowledge retrieval, safety filtering, and explainable generation—each sub-problem requires a specialized processing workflow.

**Goal**: How to design a modular multi-agent framework where each agent focuses on a sub-task, collaboratively achieving safe, accurate, and explainable game recommendations?

**Key Insight**: Drawing inspiration from the modular division of labor in the LLM agent field, the recommendation pipeline is decomposed into six stages: intent understanding $\to$ candidate generation $\to$ ranking $\to$ reflection $\to$ safety check $\to$ explanation generation, with each stage handled by a specialized agent.

**Core Idea**: Replace end-to-end recommendation of a single LLM with the division and collaboration of six specialized agents, thereby addressing the three major challenges of game recommendation: constraint parsing, knowledge freshness, and safety control.

## Method

### Overall Architecture

User enters natural language query $\to$ Risk Control Agent performs safety pre-check (input end) $\to$ Intent Agent parses user intent and constraints $\to$ Tool-Augmented Candidate Agent retrieves candidates using 10+ tools $\to$ Multi-LLM Ranking Agent (GPT-4o + Gemini collaboration) scores and ranks $\to$ Reflection Agent performs reflection and re-ranking using detailed game profiles $\to$ Risk Control Agent performs safety post-check (output end) $\to$ Explanation Agent generates four-dimensional recommendation explanations $\to$ Final output of $k$ recommendations and explanations.

### Key Designs

1. **Risk Control Module (Dual-End Safety Protection)**:

    - **Function**: Dual-end interception of harmful content at both the input and output stages.
    - **Mechanism**: The Jailbreak Prevention Agent integrates three complementary techniques: (1) RA-LLM random token dropping (to detect jailbreak attacks); (2) Chain-of-thought intent reasoning (to identify subtle adversarial semantics); and (3) defined-policy fallback actions. The Dangerous Content Detection Agent serves as the second layer of filtering, performing content moderation on both the input queries and output recommendations.
    - **Design Motivation**: The user base of game platforms includes a large number of minors, presenting higher safety risks. Single detection methods are easily bypassed, so combining three complementary techniques improves robustness.

2. **Multi-LLM Collaborative Ranking + Reflection Re-ranking**:

    - **Function**: Overcome the knowledge limitations of a single LLM and improve ranking quality.
    - **Mechanism**: The Ranking Agent utilizes a two-layer LLM collaboration where GPT-4o and Gemini independently evaluate candidate games across five dimensions (popularity, user preference match, historical similarity, genre alignment, age suitability), merging them via a weighted average. The Reflection Agent loads detailed game profiles (only used at this stage due to excessive text length) onto the ranked candidates, performing re-ranking based on contextual clues and user feedback. To control cost, reflection is only applied to top-k candidates.
    - **Design Motivation**: Different LLMs have complementary advantages in different dimensions (e.g., one is better at understanding complex intent, while the other is more accurate in genre matching); the reflection mechanism utilizes complete game information to correct rankings but limits its scope to balance effectiveness and computational cost.

3. **Four-Dimensional Explainable Recommendation Generation**:

    - **Function**: Generate multi-angle readable explanations for each recommendation.
    - **Mechanism**: The Explanation Agent generates explanations from four dimensions: (1) category preference (alignment of recommendations with user-preferred styles); (2) similarity (similarity to games the user historically liked); (3) demographics (age suitability, etc.); and (4) popularity & novelty (ratings/awards/innovative features). Profiles are constructed by querying game metadata (IDs, descriptions, tags), generating explanations for each dimension using tailored prompts, and then aggregating them into a coherent summary.
    - **Design Motivation**: Multi-dimensional explanations are more persuasive than single-reason explanations, receiving highly consistent scores of 4.2/5 from LLM evaluators and 3.97/5 from human experts.

### Loss & Training

The entire framework is training-free (inference-time orchestration), with each agent driven by prompt engineering. An exploratory hyperparameter is introduced in ranking to allow genres beyond user preferences to increase diversity.

## Key Experimental Results

### Main Results (Top-5 Recommendation)

| Method | Factual↑ | Hit@5↑ | P@5↑ | Pop50↓ | RPop50↓ | MaxF↓ | JP | Exp |
|------|----------|--------|------|--------|---------|-------|-----|-----|
| Pop | 1.00 | .14 | .04 | 1.00 | 7.97 | .15 | ✘ | N/A |
| OMuleT (GPT-4o) | .99 | .24 | .08 | .27 | 2.14 | .12 | ✘ | N/A |
| MACRec | .92 | .21 | .07 | .39 | 3.34 | .31 | ✘ | 1.7 |
| MACRS-C | .85 | .14 | .04 | .33 | 3.52 | .42 | ✘ | N/A |
| Multi-Agent GPT | .94 | .24 | .07 | .65 | 3.83 | .27 | ✘ | 2.5 |
| **MATCHA** | **.99** | **.29** | **.10** | .27 | **2.05** | **.09** | **✔** | **4.2** |

### Ablation Study

| Ablation Configuration | Key Impact |
|---------|---------|
| Remove Reflection Agent | Slight drop in accuracy, but improved diversity |
| Remove Multi-LLM Collaboration | Ranking quality drops, single LLM has larger bias |
| Remove Tool-augmented Retrieval | Candidate pool quality drops significantly |
| Remove Jailbreak Prevention | Adversarial defense rate drops from 97.9% to baseline levels |

### Key Findings

- MATCHA improves Hit@5 by approximately 20% compared to OMuleT (.24 $\to$ .29), while reducing popularity bias RPop50 from 2.14 to 2.05.
- The jailbreak defense rate reaches 97.9%, making MATCHA the only method having this capability (other baselines lack safety protection).
- The explanation quality score is 4.2/5 (LLM evaluator), far exceeding the highest baseline of 2.5 (Multi-Agent GPT); human evaluation (3.97/5) aligns closely with machine evaluation.
- Multi-LLM collaborative ranking significantly outperforms single-LLM, leveraging the complementary strengths of different LLMs to improve ranking diversity and accuracy.
- MATCHA achieves the lowest MaxFreq (.09) while maintaining a high Factuality (.99), demonstrating an extremely low recommendation repetition rate.

## Highlights & Insights

- **Safety-first system design**: Placing safety checks at both the inlet and outlet of the pipeline, rather than conducting post-audit, is a pioneer architectural design in recommender systems. For platforms targeting minors (such as Roblox), this is a must-have rather than an option.
- **Practical value of multi-LLM collaborative ranking**: Leveraging the complementary advantages of GPT-4o and Gemini for independent evaluation and fusion yields significant improvement, despite increased cost. This idea can be directly transferred to other scenarios requiring multi-perspective evaluation (e.g., resume screening, content moderation).
- **Cost control strategy in the Reflection stage**: Applying reflection re-ranking with detailed profiles only to top-k candidates, rather than all candidates, is a clever engineering choice.

## Limitations & Future Work

- The framework entirely relies on inference-time LLM calls without any training components—on large-scale platforms like Roblox, each recommendation request requires multiple API calls, making latency and cost potential deployment bottlenecks.
- The OMuleT evaluation dataset contains only 553 user requests, which is relatively small; moreover, it is only tested on the Roblox platform, leaving its generalization to larger-scale game platforms such as Steam unverified.
- The propagation of errors among multiple agents is not thoroughly discussed—if the Intent Agent misunderstands user intent, the outputs of all subsequent agents will be affected.
- Safety detection relies on pre-defined policies and patterns, and its adaptability to novel attack vectors remains to be verified.

## Related Work & Insights

- **vs OMuleT (Yoon et al., 2024)**: OMuleT is a multi-tool, single-agent framework; MATCHA extends it to a multi-agent setup, adding ranking collaboration, reflection, and safety modules. OMuleT is slightly superior in Entropy, but MATCHA completely outperforms it in relevance and safety.
- **vs MACRS (Fang et al., 2024)**: MACRS uses multi-agent collaboration for conversational recommendation, but focuses on the movie domain and lacks safety modules and tool-augmented retrieval. MATCHA is specifically designed for the unique challenges of games.
- **vs MACRec (Wang et al., 2024b)**: MACRec's multi-agent framework is more general, but in the game recommendation scenario, its Hit@5 is only .21, far below MATCHA's .29.
- The paper demonstrates a complete engineering practice of CRS on a large-scale game platform, providing practical reference values for recommendation practitioners.

## Rating

- Novelty: ⭐⭐⭐ Each module (multi-agent collaboration, safety protection, explainable recommendation) is not a brand-new concept; the core contribution lies in their systematic integration tailored for the game domain.
- Experimental Thoroughness: ⭐⭐⭐ The eight metrics provide comprehensive coverage, but the dataset size is small, and there is a lack of large-scale online A/B testing.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and the problem definition is precise, but the use of mathematical notation is not standardized enough.
- Value: ⭐⭐⭐⭐ High reference value for the engineering practice of game recommender systems, and the design concept of the safety module is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](../../ACL2026/recommender/harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[NeurIPS 2025\] The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process](../../NeurIPS2025/recommender/the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)
- [\[NeurIPS 2025\] EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](../../NeurIPS2025/recommender/empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)

</div>

<!-- RELATED:END -->
