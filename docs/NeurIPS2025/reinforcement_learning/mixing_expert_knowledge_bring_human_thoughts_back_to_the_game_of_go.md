---
title: >-
  [Paper Note] Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go
description: >-
  [NeurIPS 2025][Reinforcement Learning][LLM] This paper proposes LoGos, which applies mixed-domain expert data (Go) and general long chain-of-thought (CoT) reasoning data for cold-start fine-tuning followed by GRPO reinfo…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "LLM"
  - "Go"
  - "Domain Expert Knowledge"
  - "GRPO"
date: 2026-05-08
content_hash: 0dd99a6f6a420f0a
---

# Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go

**Conference**: NeurIPS 2025
**arXiv**: [2601.16447](https://arxiv.org/abs/2601.16447)  
**Code**: [GitHub](https://github.com/Entarochuan/LoGos)  
**Area**: Reinforcement Learning
**Keywords**: LLM, Go, Domain Expert Knowledge, Reinforcement Learning, GRPO

## TL;DR

This paper proposes LoGos, which applies mixed-domain expert data (Go) and general long chain-of-thought (CoT) reasoning data for cold-start fine-tuning followed by GRPO reinforcement learning, enabling a general-purpose LLM to reach professional-level Go performance while preserving strong general reasoning capabilities.

## Background & Motivation

**Background**: Specialized AI systems such as AlphaGo/AlphaZero have long surpassed human performance in Go, yet general-purpose large language models (LLMs) perform extremely poorly on Go tasks, often falling below the level of amateur beginners.

**Limitations of Prior Work**: Professional domains such as Go lack large-scale natural-language reasoning corpora (unlike mathematics or programming, which offer abundant human reasoning traces), making direct distillation or pretraining approaches largely ineffective.

**Key Challenge**: General-purpose LLMs possess strong reasoning and generalization abilities but lack domain-specific expertise, whereas specialized systems (e.g., KataGo) are highly capable yet can only produce structured predictions without natural-language reasoning.

**Goal**: The paper investigates how to instill expert-level capabilities in domains such as Go into general-purpose LLMs while preserving their general reasoning ability.

**Key Insight**: The approach leverages the large volume of existing structured Go data (game records annotated by KataGo), constructs synthetic training data via heuristic rules, and subsequently aligns reasoning with domain knowledge through RL self-exploration.

**Core Idea**: Mix domain-expert synthetic data with general CoT data for cold-start initialization, then apply GRPO to enable the model to self-discover natural-language reasoning strategies for Go.

## Method

### Overall Architecture

The system consists of three stages: (1) **Data Construction** — Go game records are collected and annotated by KataGo, and synthetic datasets are constructed using heuristic templates; (2) **Mixed Cold-Start Fine-Tuning** — Go-specific professional data is mixed with general long-CoT reasoning data for supervised fine-tuning; (3) **GRPO Reinforcement Learning** — a tiered reward function guides the model to self-explore Go reasoning strategies.

### Key Designs

1. **Go Representation**: Go games are serialized as move lists (e.g., X-D4, O-Q16, …), where each move is represented by a letter-number coordinate and X/O denotes black/white stones. The model predicts the next move as $x_{k+1} = \pi_\theta(x_1, x_2, \ldots, x_k)$.

2. **Expert-Level Go Dataset Construction**:

    - **Next-Move Prediction Dataset** (∼10M scale): Game states are sampled from 5M+ professional and top-amateur game records. KataGo annotates each state with the top-10 candidate moves along with their win rates and follow-up variations. A heuristic template then structures each sample into four parts: confirming the player to move → analyzing candidate moves → summarizing the optimal choice → producing structured output.
    - **Commentary Dataset** (∼100K scale): Open-source Go commentary data is collected and processed into (game-state, commentary) training pairs.

3. **Mixed Cold-Start Fine-Tuning**: Go-specific professional data is mixed with general long-CoT reasoning data (Openthoughts-114K, NuminaMath-QwQ-CoT-5M, OpenCodeReasoning, and other math/code reasoning datasets) to fine-tune the Qwen2.5 base model. This simultaneously injects Go domain knowledge and initializes the model with a long-CoT reasoning format.

4. **GRPO Self-Exploration RL**: Starting from the cold-start model, carefully designed queries and reward functions encourage the model to self-explore long-CoT reasoning strategies on the Go next-move prediction task. A key finding is that the model spontaneously transfers reasoning skills acquired from CoT data to the Go domain.

### Loss & Training

**Tiered Reward Function**: Rewards are assigned based on the rank of the predicted move within KataGo's top-10 list, with an additional reward for win-rate prediction accuracy:

$$r_i = \begin{cases} 1 - \alpha_1 \cdot \frac{\beta_1|(\hat{w}_i - w_i)|}{1 + \beta_1|(\hat{w}_i - w_i)|} & \text{rank}(i) = 1 \\ c_1 - \alpha_1 \cdot (\text{win-rate term}) - \alpha_2 \cdot (\text{rank penalty term}) & \text{rank}(i) \in [2,3] \\ c_2 - \ldots & \text{rank}(i) \in [4,10] \\ c_3 - \alpha_1 - \alpha_2 & \text{rank}(i) \notin [1,10] \land \text{format correct} \\ 0 & \text{otherwise} \end{cases}$$

Hyperparameters: $c_1=0.8,\ c_2=0.6,\ c_3=0.4$ (tiered base rewards); $\alpha_1=0.1,\ \alpha_2=0.2,\ \beta_1=\beta_2=10$ (win-rate prediction and intra-tier fine-grained reward coefficients). Optimization follows the GRPO objective with a KL coefficient of $5 \times 10^{-4}$.

## Key Experimental Results

### Main Results

| Model | KataGo-Bench | GPQA Diamond | AIME | MATH | LiveCodeBench |
|---|---|---|---|---|---|
| DeepSeek-R1 | 17.6 | 69.7 | 86.7 | 97.6 | 83.8 |
| Claude3.7-Sonnet | 34.3 | 67.7 | 30.0 | 79.8 | 63.2 |
| DS-R1-Distill-7B | 0.6 | 41.4 | 33.3 | 88.2 | 20.4 |
| **LoGos (7B)** | **88.1** | 37.9 | 40.0 | 93.2 | 23.4 |
| DS-R1-Distill-32B | 4.7 | 56.1 | 46.7 | 94.5 | 36.5 |
| **LoGos (32B)** | **88.6** | 63.6 | 56.7 | 96.5 | 50.9 |
| KataGo-HumanSL-9d | 87.8 | - | - | - | - |

LoGos achieves 88.6% on the Go benchmark, surpassing KataGo-HumanSL-9d (87.8%), which simulates top-level human play, and outperforming Claude 3.7 Sonnet (34.3%) by approximately 2.6×.

### Ablation Study

| Configuration | Result |
|---|---|
| Direct RL without cold start | Performance ceiling significantly below beginner level (<67.4%) |
| Cold start with direct prediction instead of heuristic templates | Performance ceiling <50% (vs. 88%) |
| Reward: top-1 only | Slightly lower performance (sparse reward) |
| Reward: top-3 only, no tiering | Slightly lower performance |
| 2-epoch vs. 1-epoch Go data during cold start | 1-epoch yields a higher RL performance ceiling |
| Mixed Go data volume (500K–10M) | Large-scale Go data does not significantly harm general capabilities |

### Key Findings

- **Cold Start Is Indispensable**: Without cold-start initialization, direct RL fails to bring the model above beginner level. Even DeepSeek-R1 distillation models cannot reach an acceptable level through RL self-exploration alone.
- **Heuristic Templates Are Critical**: Replacing heuristic-rule-based template data with direct prediction causes the RL performance ceiling to drop sharply to below 50%.
- **Context Curse**: As game sequences grow longer, the model's ability to comprehend the board state deteriorates. Introducing 2D board-state rendering (a 19×19 matrix) effectively mitigates this issue, maintaining high accuracy beyond move 200.
- Human evaluation shows that 96.5% of move predictions are correct and 55.6% of the accompanying explanations are also correct.

## Highlights & Insights

- **Pioneering Work**: This is the first general-purpose LLM to achieve professional-level Go performance, demonstrating that general reasoning capabilities can be effectively integrated with domain expert knowledge.
- **Methodological Generality**: The proposed "mixing expert knowledge" paradigm—structured professional data → heuristic-template synthesis → mixed cold start → RL self-exploration—is transferable to other domains where natural-language reasoning data is scarce but structured data is available.
- **Spontaneous Reasoning Transfer**: After cold-start fine-tuning, the model spontaneously generalizes CoT reasoning patterns learned from mathematics and code to the Go domain, a finding with significant implications.
- **Dataset Contribution**: The paper introduces the first large-scale Go dataset and evaluation benchmark designed for LLM training.

## Limitations & Future Work

- The approach relies on a scalable source of domain expertise (e.g., KataGo) and requires the design of domain-specific heuristic rules and synthetic data templates.
- RL self-exploration training is slow; compared to traditional Go AI (AlphaGo), the number of game states processed during RL training is far smaller.
- The autoregressive architecture may limit training efficiency for real-time inference tasks.
- Occasional errors in Go terminology occur (among 600+ specialized terms), particularly in the opening phase.
- Some explanations remain vague and provide limited informational value.

## Related Work & Insights

- **LLM Game Players**: Works such as ChessGPT and PokerGPT explore LLMs playing board games, but Go's complexity far exceeds that of chess ($b \approx 250$ vs. $b \approx 35$).
- **Domain-Expert AI**: Systems such as AlphaGeometry and DeepSeek-Prover leverage tools and heuristic rules to endow LLMs with domain-expert capabilities; this paper follows a similar philosophy but targets the more open-ended domain of Go.
- **Insight**: This work provides a reproducible paradigm for integrating RL with domain-specific structured knowledge into general-purpose LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First general-purpose LLM to achieve professional-level Go performance; the mixed expert knowledge training paradigm is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies covering cold-start necessity, template design, reward function design, and data volume effects.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, figure-rich, and persuasively written.
- Value: ⭐⭐⭐⭐⭐ Strong methodological contribution with significant guidance for research on injecting domain-specific expert knowledge into general-purpose LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Human-Inspired Multi-Level Reinforcement Learning](human-inspired_multi-level_reinforcement_learning.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization](learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)

</div>

<!-- RELATED:END -->
