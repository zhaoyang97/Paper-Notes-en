---
title: >-
  [Paper Note] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents
description: >-
  [NeurIPS 2025 Workshop][Reinforcement Learning][RL agent] This paper demonstrates that a 14B-parameter search agent trained with RL can surpass frontier models on legal document retrieval (85% vs. GPT o3's 81%) through multi-turn interaction, enabled by a carefully designed segmented reward structure and a sufficiently long interaction horizon.
tags:
  - NeurIPS 2025 Workshop
  - Reinforcement Learning
  - RL agent
  - multi-turn search
  - legal document retrieval
  - GRPO
  - tool use
date: 2026-05-08
content_hash: 43f4ec95c8d72b26
---

# Reinforcement Learning for Long-Horizon Multi-Turn Search Agents

**Conference**: NeurIPS 2025 Workshop  
**arXiv**: [2510.24126](https://arxiv.org/abs/2510.24126)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: RL agent, multi-turn search, legal document retrieval, GRPO, tool use

## TL;DR
This paper demonstrates that a 14B-parameter search agent trained with RL can surpass frontier models on legal document retrieval (85% vs. GPT o3's 81%) through multi-turn interaction, enabled by a carefully designed segmented reward structure and a sufficiently long interaction horizon.

## Background & Motivation

**Background**: LLMs have demonstrated strong capabilities in tool use and multi-step reasoning. Multi-turn document search is a complex long-horizon interactive task in which an agent must iteratively query to locate specific information.

**Limitations of Prior Work**: (1) Prompt-based methods achieve reasonable performance but lack the ability to learn from experience; (2) naïve RAG (single-shot retrieval) performs poorly on such tasks (33%); (3) tool access alone is insufficient—base Qwen3-14B with the same tools achieves only 53%.

**Key Challenge**: Tool access ≠ effective tool use. An agent must learn to exploit multi-turn interaction opportunities to progressively narrow the search space.

**Goal**: Train an agent via RL to effectively use search tools across multi-turn interactions.

**Key Insight**: Construct a legal document search benchmark, design a segmented reward structure (rewarding correct document retrieval, correct citation, and appropriate abstention; penalizing hallucination and formatting errors), and train a LoRA adapter with GRPO.

**Core Idea**: Through carefully designed segmented rewards and GRPO training, a 14B model learns to effectively leverage multi-turn search interactions, ultimately outperforming frontier models.

## Method

### Overall Architecture
Construct a legal document search benchmark (2,300 QA pairs) → equip the agent with three tools (keyword search / semantic search / document content reading) → RL training (GRPO + segmented rewards) → evaluate performance under varying turn-limit constraints.

### Key Designs

1. **Three-Tool Agent Architecture**:

    - **Function**: Provides complementary document retrieval capabilities.
    - **Mechanism**: Keyword search (BM25) returns text snippets + section IDs; semantic search (FAISS + MiniLM-L6-v2 embeddings) returns conceptually matched results; document content reading returns full content given a section ID, with a hierarchical ID structure supporting navigation (e.g., A:B:C → A:B to jump upward).
    - **Design Motivation**: Supports a two-phase search pattern—broad exploration via keyword/semantic search followed by deep extraction via the reading tool.

2. **Segmented Reward Design**:

    - **Function**: Provides fine-grained learning signals for RL.
    - **Mechanism**: $[1.0, 2.0]$ for correct answer with correct citation (higher reward for fewer turns/searches); $[0.0, 1.0]$ for the model responding "I don't know" (preferred over hallucination); $[-1.0, 0.0]$ for incorrect answers (partial credit of $+0.1$ if the correct document was retrieved); $[-2.0, -1.0]$ for formatting errors (invalid tool calls).
    - **Design Motivation**: Graduated rewards ensure that even failed trajectories provide learning signal. An efficiency bonus encourages task completion with fewer searches. **Critically**, hallucination is penalized more heavily than abstention—training the model to say "I don't know" when evidence is insufficient.

3. **Turn-Restricted Evaluation**:

    - **Function**: Quantifies the impact of multi-turn interaction on performance.
    - **Mechanism**: At turn $N$, an `<answer>` prefix is forcibly inserted to compel the model to produce a response. Zero turns is equivalent to naïve RAG.
    - **Design Motivation**: Understands how the agent exploits additional search opportunities and how RL training alters this exploitation behavior.

### Loss & Training
GRPO (Group Relative Policy Optimization). Base model: Qwen3-14B + LoRA adapter. Reward model: Gemini 2.5 Pro for binary quality judgment. group_size=6, 8 groups per step. YaRN extends context to 128K tokens.

## Key Experimental Results

### Main Results

| Model | Accuracy | Avg. Turns |
|-------|----------|-----------|
| Naïve RAG (Gemini 2.5 Pro) | 33% | 1.0 |
| Qwen3-14B (base) | 53% | 3.7 |
| Gemini 2.5 Flash | 66% | 3.4 |
| Gemini 2.5 Pro | 78% | 5.3 |
| OpenAI o3 | 81% | 7.1 |
| **Qwen3-14B + RL** | **85%** | **6.2** |

### Ablation Study

| Analysis | Finding |
|----------|---------|
| Base Qwen3-14B | Performance saturates after 6 turns |
| RL-trained Qwen3-14B | Performance continues to improve at 10 turns |
| Gemini 2.5 Pro | Performance continues to improve at 10 turns |
| Training with restricted turns | An agent trained with a 4-turn limit fails to effectively utilize additional turns at inference time (up to 10 turns) |

### Key Findings
- **The 14B RL model outperforms all frontier models (85% vs. o3's 81%)**—effective tool use can be "elicited" from smaller models through RL.
- **Tool access ≠ effective tool use**: Qwen3-14B without RL training achieves only 53%, while RL training raises this to 85%.
- **RL agents are better at exploiting multi-turn interaction**: The base model saturates at 6 turns, whereas the RL model continues to improve at 10 turns—indicating that RL teaches the agent to "avoid premature answers" and "search systematically."
- **Turn limits during training constrain exploitation at inference time**: Sufficient turn budget must be provided during training.

## Highlights & Insights
- **"Tool access ≠ effective tool use"** is the paper's most important insight: the same tools yield a 32-percentage-point gap between the base and RL-trained models.
- **The segmented reward design** is highly instructive: the value ordering of "I don't know" > hallucination should become standard practice for all search agents.
- **Small models + RL can surpass large models**: The paper demonstrates that, for specialized tasks, RL-trained expert models can outperform general-purpose frontier models.

## Limitations & Future Work
- **Single legal domain**: Generalizability to other domains remains unverified.
- **Dependence on Gemini 2.5 Pro as the reward model**: High cost and potential for introduced bias.
- **LLM-generated training data**: The quality and diversity of QA pairs are constrained by the generative model.
- **Workshop paper**: The experimental scale is relatively limited.

## Related Work & Insights
- **vs. Naïve RAG**: Single-shot retrieval (33%) vs. multi-turn interaction (85%)—the large gap underscores the importance of multi-turn interaction for complex retrieval tasks.
- **vs. Prompt-based agents**: Prompt-based agents can achieve competitive performance (Gemini Pro: 78%), but RL can push performance further.
- **vs. Chain-of-Retrieval**: Similar motivation, but RL is used to learn the optimal retrieval strategy rather than following a predefined procedure.

## Rating
- Novelty: ⭐⭐⭐⭐ Empirical study of RL-trained multi-turn search agents; turn-restriction analysis is novel
- Experimental Thoroughness: ⭐⭐⭐ Workshop-paper scale; single domain
- Writing Quality: ⭐⭐⭐⭐ Concise and clear
- Value: ⭐⭐⭐⭐⭐ The result of a small model surpassing frontier models is highly inspiring

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction](../../AAAI2026/reinforcement_learning/thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)
- [\[ICLR 2026\] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](../../ICLR2026/reinforcement_learning/strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
