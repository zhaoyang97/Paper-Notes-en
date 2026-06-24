---
title: >-
  [Paper Note] CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games
description: >-
  [ACL 2025 (Main)][Multi-Agent][Metaphorical Reasoning] This paper proposes the CoMet framework. By integrating a hypothesis-testing-based metaphor reasoner and a self-improving metaphor generator, CoMet enables LLM agents to utilize metaphors for covert communication and semantic evasion in multi-agent language games. It significantly enhances the strategic communication capabilities of agents in Undercover and Adversarial Taboo games (improving the win rate from 0.20 to 0.70…
tags:
  - "ACL 2025 (Main)"
  - "Multi-Agent"
  - "Metaphorical Reasoning"
  - "Covert Communication"
  - "Multi-Agent Games"
  - "Language Games"
  - "Strategic Planning"
date: 2026-05-08
content_hash: c0adee54c69ac3b3
---

# CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games

**Conference**: ACL 2025 (Main)  
**arXiv**: [2505.18218](https://arxiv.org/abs/2505.18218)  
**Code**: [https://github.com/Yeswolo/CoMet](https://github.com/Yeswolo/CoMet)  
**Area**: Others  
**Keywords**: Metaphorical Reasoning, Covert Communication, Multi-Agent Games, Language Games, Strategic Planning

## TL;DR

This paper proposes the CoMet framework. By integrating a hypothesis-testing-based metaphor reasoner and a self-improving metaphor generator, CoMet enables LLM agents to utilize metaphors for covert communication and semantic evasion in multi-agent language games. It significantly enhances the strategic communication capabilities of agents in Undercover and Adversarial Taboo games (improving the win rate from 0.20 to 0.70).

## Background & Motivation

**Background**: Metaphors are essential tools for humans to express complex and nuanced ideas, pervading daily communication. Recently, LLMs have been widely adopted as the core of multi-agent language games, demonstrating strong performance in games like Diplomacy, Werewolf, and Avalon.

**Limitations of Prior Work**: Current LLM agents perform catastrophically in scenarios requiring understanding and application of metaphors. Specifically, when "conceptual disguise" or "semantic evasion" is needed in multi-agent games, LLM agents tend to interpret language literally and fail to identify or generate metaphorical expressions, leading to frequent failures in strategic communication scenarios.

**Key Challenge**: There is a fundamental challenge in multi-agent language games—agent utterances are publicly broadcasted, and both teammates and opponents can hear them. How to achieve information transfer among teammates over a public channel without being understood by opponents? Human players naturally use metaphors to address this, but LLMs lack this ability.

**Goal**: To design a framework that enables LLM agents to (1) understand implicit messages within others' metaphors, and (2) generate effective metaphors to achieve covert communication.

**Key Insight**: The authors observe that metaphors can serve as "asymmetric encryption in natural language"—the party holding the same key (the secret word) can decrypt the metaphorical meaning, whereas the party without the key can only obtain the literal meaning.

**Core Idea**: Formalize metaphorical reasoning as a hypothesis testing process (i.e., is this describing my secret word?) and transform metaphor generation into a self-improving task with experience accumulation, thereby building a complete, metaphor-driven strategic communication framework.

## Method

### Overall Architecture

CoMet comprises six modules that form a complete think-communicate-act loop: Feature Extractor (extracts lexical features from other players' utterances) $\rightarrow$ Metaphor Reasoner (detects metaphors in utterances and conducts reasoning) $\rightarrow$ Belief Mapper (infers roles and identities of other players) $\rightarrow$ Self-Monitor (tracks self-identity awareness) $\rightarrow$ Strategy Planner (formulates communication and action strategies) $\rightarrow$ Metaphor Generator (translates strategies into metaphorical expressions) + Voter (makes decisions during voting phases).

### Key Designs

1. **Hypothesis-Based Metaphor Reasoner**:

    - **Function**: Judge whether other players' utterances contain metaphors related to one's own secret word.
    - **Mechanism**: Establish two hypotheses for each utterance—$H_0$: The speaker is describing the same secret word as mine; $H_1$: The speaker is describing another word. Multi-dimensional feature sets $F$ (behavior, state, structure, function, attribute) are extracted from the secret word, and metaphorical dimension sets $M$ (ontological metaphor, structural metaphor, spatial metaphor) are identified from the utterance. A weighted score $s_w = w_f \times w_m \times score$ is calculated using a semantic matching function $\delta(f,m,S)$. If the score exceeds a threshold $T$, $H_0$ is accepted. The key innovation lies in: bypassing the need to fully decrypt the exact meaning of the metaphor, simplifying it instead to a binary judgment of "is this describing my word?"
    - **Design Motivation**: Traditional metaphorical understanding requires full decryption of metaphor meanings, which is extremely challenging for LLMs. Simplifying the task to hypothesis testing drastically reduces cognitive complexity while perfectly aligning with the "teammate identification" requirement in the Undercover game. Injecting Lakoff's metaphor theory classification as prior knowledge further enhances reasoning quality.

2. **Self-Improving Metaphor Generator**:

    - **Function**: Generate high-quality metaphors that can be understood by teammates but confuse opponents.
    - **Mechanism**: Accumulate metaphor generation experiences through self-play. After generating each metaphor, reactions from teammates and opponents are recorded as feedback. The experience pool is formatted as `{metaphor text, generator explanation, times identified by opponents, times identified by teammates, score}`. Future generation retrieves successful experiences under similar scenarios from the pool as reference. The experience pool has a maximum capacity of 100 entries per category, with low-scoring experiences periodically replaced by high-scoring ones.
    - **Design Motivation**: Metaphor generation is a highly creative task that is difficult to implement with fixed rules. Iterative refinement based on win/loss feedback in actual gameplay is a practical and effective strategy. Experiments show that after 100 rounds of experience accumulation, the metaphor success rate of GPT-4o increases by 29%.

3. **Belief Mapper + Self-Monitor**:

    - **Function**: Infer roles of all players in the game along with self-identity.
    - **Mechanism**: The Belief Mapper leverages extracted features for first-order Theory of Mind (ToM) reasoning to infer other players' identities $I_{-i}$, roles $R_{-i}$, and strategies $S_{-i}$. The Self-Monitor combines features and beliefs to reversely infer its own role—if most players' descriptions do not match my word, I am likely the undercover player. Identity inference is iteratively updated as the game progresses: $I_i \leftarrow I_i'$.
    - **Design Motivation**: In the Undercover game, players do not know their own roles initially, which makes decision-making exceptionally difficult. Without the Self-Monitor, the agent always assumes it is a civilian, leading to rapid exposure when playing as the undercover. Ablation studies show that removing the Self-Monitor causes the win rate to plunge from 0.70 to 0.05.

### Loss & Training

CoMet is a reasoning framework based on prompt engineering and does not involve model training. Its "training" is manifested in the experience accumulation process of the metaphor generator—experiences are collected through self-play, an evaluator scores each metaphor experience, and low-scoring experiences are periodically pruned. The initial experience pool contains 20 human-generated seed experiences.

## Key Experimental Results

### Main Results

**Undercover Game (5 players, 3 civilians vs 2 undercover, 200 word pairs, 10 rounds/word pair):**

| Method | Role | Win Rate (WR) | Feature Extraction Rate (FER) | Identity Assessment Accuracy (OIAA) | Privacy Protection Capability (PPC) |
|------|------|---------|---------------|-------------------|-------------|
| CoT | Undercover | 0.20 | 0.30 | 0.65 | 0.14 |
| CoMet | Undercover | **0.35** | **0.82** | **0.77** | **0.37** |
| CoT | Civilian | 0.80 | 0.23 | 0.61 | 0.88 |
| CoMet | Civilian | **0.85** | **0.75** | **0.73** | 0.62 |

**Adversarial Taboo Game (different LLMs, opponent is GPT-4o+CoT):**

| Model | CoT Win Rate | CoMet Win Rate | Gain |
|------|---------|----------|------|
| GPT-4o | ~40% | **87%** | +47% |
| DeepSeek-R1 | ~35% | **78%** | +43% |
| Claude 3.5 | ~38% | **82%** | +44% |
| Qwen2.5-72B | ~30% | **75%** | +45% |
| Llama3.3-70B | ~28% | **72%** | +44% |

### Ablation Study

| Configuration | Met. | FE | BM | SM | SP | Win Rate (WR) |
|------|------|----|----|----|----|---------|
| CoMet (Full) | ✓ | ✓ | ✓ | ✓ | ✓ | **0.70** |
| CoMet w/o Met. | ✗ | ✓ | ✓ | ✓ | ✓ | 0.45 |
| w/o Met.&FE | ✗ | ✗ | ✓ | ✓ | ✓ | 0.40 |
| w/o Met.&BM | ✗ | ✓ | ✗ | ✓ | ✓ | 0.25 |
| w/o Met.&SP | ✗ | ✓ | ✓ | ✓ | ✗ | 0.25 |
| w/o Met.&SM | ✗ | ✓ | ✓ | ✗ | ✓ | 0.05 |

### Key Findings

- **Self-Monitor is the most critical module**: Removing it causes the win rate to plunge from 0.70 to 0.05. This is because, without self-role assessment, the agent constantly assumes it is a civilian, leading to aggressive information disclosure and immediate exposure when acting as the undercover.
- The metaphor module contributes a 0.25 win rate improvement (0.70 vs 0.45), indicating that metaphors are indeed a crucial means for covert communication.
- Hypothesis-testing-based metaphorical reasoning outperforms direct understanding and substitution-based reasoning, as the task is framed with reasonable simplification.
- Experience accumulation significantly improves the quality of metaphor generation—after 100 experience entries, the metaphor success rate of GPT-4o increases by 29%, and Qwen2.5-72B increases by 22%.
- Ontological metaphors (47%) are used most frequently and achieve the highest score (0.44), while spatial metaphors are used least frequently (18%) and score the lowest (0.22).

## Highlights & Insights

- **Metaphors as Natural Language Encryption**: Analogizing metaphors to asymmetric encryption is a highly inspiring insight. The party holding the same key (the secret word) can decrypt it, while those without the key only obtain the literal meaning. This methodology can be extended to broader secure communication scenarios.
- **Exquisite Simplification of the Hypothesis Testing Paradigm**: Instead of fully decrypting the metaphorical meaning, making a binary judgment of "related to me vs. unrelated" avoids the weaknesses of LLMs in metaphor comprehension while perfectly matching the requirements of the game. This "task-oriented capability simplification" represents an exemplary design philosophy.
- **Closed-Loop Learning via Self-Play and Experience Pool**: Iteratively improving metaphor quality through success/failure signals within the game itself without extra training data forms an elegant self-improvement loop.

## Limitations & Future Work

- It is currently verified only in concept-description games; more complex strategic games (such as role-reasoning combined with metaphors in Diplomacy or Werewolf) remain unexplored.
- The metaphorical theoretical framework is simplified (covering only three types of metaphors); complex culture-specific metaphors (e.g., Chinese idioms, Japanese idiomatic expressions) have not been investigated.
- The initial quality of the experience pool relies on 20 human-designed seed experiences, potentially leading to cold-start issues in new scenarios.
- All experiments are based on English vocabulary; cross-lingual metaphor generation and comprehension remain entirely unexplored directions.
- Metaphors in multimodal scenarios (e.g., combining visual and linguistic metaphors) possess significant research potential.

## Related Work & Insights

- **vs Reflexion/Self-Play Methods**: While Reflexion improves general reasoning through reflection, CoMet specializes the reflection mechanism specifically for metaphor generation, achieving more structured learning via the experience pool.
- **vs MAGIC (Xu et al. 2024)**: MAGIC evaluates the capabilities of LLMs in multi-agent cognition but does not cover metaphor-driven strategic communication. CoMet addresses this gap.
- **vs Reasoning Enhancement Methods (e.g., Tree-of-Thoughts)**: These methods enhance thinking depth but do not involve communication strategies. CoMet focuses on "how to speak" rather than "how to think", representing an orthogonal capability dimension.
- **Insights**: The concept of metaphor-driven covert communication can be transferred to the field of information security—for instance, enabling AI agents to securely exchange information in monitored environments.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce metaphor processing to strategic communication in multi-agent games, opening up a brand-new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluated across two games and multiple LLMs with detailed ablations, though verification in more complex games is still lacking.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the case analysis is vivid, although the paper's organization is slightly verbose.
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for strategic communication in LLM agents, showing broad application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CellAgent: LLM-Driven Multi-Agent Framework for Natural Language-Based Single-Cell Analysis](../../ICLR2026/multi_agent/cellagent_llm-driven_multi-agent_framework_for_natural_language-based_single-cel.md)
- [\[ICLR 2026\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](../../ICLR2026/multi_agent/cache-to-cache_direct_semantic_communication_between_large_language_models.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](../../NeurIPS2025/multi_agent/thought_communication_in_multiagent_collaboration.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/multi_agent/large_language_models_miss_the_multi-agent_mark.md)

</div>

<!-- RELATED:END -->
