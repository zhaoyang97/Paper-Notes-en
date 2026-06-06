---
title: >-
  [Paper Note] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL
description: >-
  [ACL 2026][LLM Agent][Search Agent] MemSearcher replaces the "history concatenation" of search agents with "LLM self-managed compact memory"—seeing only `(question, memory)` per round instead of `(question, t₁, a₁, o₁…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Search Agent"
  - "Memory Management"
  - "GRPO"
  - "Multi-Context RL"
  - "ReAct Alternative"
date: 2026-05-08
content_hash: f2c24381c5bb0c04
---

# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL

**Conference**: ACL 2026  
**arXiv**: [2511.02805](https://arxiv.org/abs/2511.02805)  
**Code**: https://github.com/icip-cas/MemSearcher (Available)  
**Area**: LLM Agent / RL / Search Agent  
**Keywords**: Search Agent, Memory Management, GRPO, Multi-Context RL, ReAct Alternative

## TL;DR
MemSearcher replaces the "history concatenation" of search agents with "LLM self-managed compact memory"—seeing only `(question, memory)` per round instead of `(question, t₁, a₁, o₁, …)`. By using multi-context GRPO to propagate the advantage of an entire trajectory to each round for independent optimization, it consistently outperforms ReAct baselines of the same size across 3B/7B/14B on 7 QA benchmarks (the 7B model even exceeds the 32B ReSearch) while maintaining a constant context length of <4K tokens.

## Background & Motivation
**Background**: Most LLM-based search agents (Search-R1, ReSearch, AutoRefine, R1-Searcher, etc.) follow the ReAct paradigm—accumulating `thought-action-observation` into the context each round to let the LLM decide the next search step. This "dialogue history concatenation" is the de facto standard for current RL-based search agents and achieves strong multi-hop QA results when paired with end-to-end GRPO/PPO training.

**Limitations of Prior Work**: Cramming all history into the context via ReAct poses two fatal problems:
1. **Linear Context Expansion**: The observations in a search agent are retrieved passages, which can be hundreds or thousands of tokens each round. In multi-turn scenarios, context easily exceeds 10K tokens, leading to lost-in-the-middle issues, long-context performance degradation, and KV cache GPU memory explosion.
2. **Noise Suppressing Signals**: Most retrieved passages are irrelevant to the question. Mixing them into the history makes it difficult for the LLM to focus on key facts; Figure 2 illustrates an example where ReAct confuses "the character's girlfriend" with "the actor's real-life partner."

**Key Challenge**: Search agents must reference multi-turn history for multi-hop reasoning, yet "viewing history = concatenating everything" is unsustainable for long-range tasks. These two factors create a fundamental tension.

**Goal**: (i) Maintain $O(1)$ context length relative to the number of turns $n$; (ii) enable the LLM to learn what to remember and what to discard; (iii) train end-to-end via RL without manual memory state annotations.

**Key Insight**: Rather than using external RAG/KG/structured memory modules, the authors let the same **backbone LLM serve as both reasoner and memory manager**. A single prompt outputs the thought+action and subsequently generates an updated memory after the observation is returned. The difficulty in training such "self-reflective memory" is that the input context $c_{i,j} = (q, m_{i,j-1})$ differs for each round, turning a single trajectory into "multiple independent optimization targets." Vanilla GRPO calculates advantage by computing a single reward for the entire trajectory, making it directly inapplicable.

**Core Idea**: (1) Framework: Use `<memory>` tags to store natural language memory where the LLM acts as both actor and memory manager each round. (2) Training: Propose Multi-Context GRPO—compute one reward for a trajectory → one advantage → **uniformly propagate** this advantage to all turns within that trajectory → treat each turn as an independent optimization target.

## Method

### Overall Architecture
**Trajectory Representation**: The $i$-th trajectory consists of $n_i$ turns. The $j$-th turn follows the form $(q, m_{i,j-1}, t_{i,j}, a_{i,j}, o_{i,j}, m_{i,j})$ (the final turn lacks $o, m$ as it contains the boxed answer). $t$ is in `<think>`, $a$ is in `<tool_call>`, $o$ is in `<tool_response>`, and $m$ is in `<memory>`.

**Per-turn Interaction**:
1. LLM takes $(q, m_{i,j-1})$ as input;
2. Outputs thought + action (either calling the wikipedia_search tool or termination via `\boxed{}` with the final answer);
3. The environment returns an observation;
4. The LLM is called again to merge $(o_{i,j}, m_{i,j-1})$ into a new $m_{i,j}$, with a token count $\le$ a preset limit of 1024;
5. Proceed to the next turn, looping until an answer is found or the maximum turns are reached.

**Computational Complexity Comparison**:

| Method | Context per Turn | FLOPs per Turn | Total FLOPs | GPU Memory |
|------|-------------|------------|----------|-----------|
| ReAct | $O(n)$ | $O(n)$ | $O(n^2)$ | $O(n)$ |
| **MemSearcher** | $O(1)$ | $O(1)$ | $O(n)$ | $O(1)$ |

### Key Designs

1. **Compact Memory Paradigm via LLM-as-Memory-Manager (Core Framework)**:
    - **Function**: Replaces the "linear concatenation of history" in ReAct with "fixed-length memory + natural language summary," keeping the context constant across multiple search turns.
    - **Mechanism**: Each round, the LLM sees the context $c_i = (q, m_{i-1})$ instead of $c_i = (q, t_1, a_1, o_1, \ldots)$. Memory $m$ is written in natural language within `<memory>` tags and is overwritten/generated by the backbone LLM itself. The generation prompt instructs: "Read $o_i$ carefully, integrate new information useful for answering $q$, while retaining all relevant details from $m_{i-1}$." Maximum memory length is 1024 tokens (ablations were performed between 256-2048).
    - **Design Motivation**: Traditional external memories (RAG / KG / Mem0) either require separate retriever training or sacrifice end-to-end differentiability. Letting the backbone LLM manage memory ensures (1) no extra models are introduced, and (2) the entire pipeline is handled by the same LLM, allowing end-to-end RL coverage. Using natural language (rather than latent tokens) ensures interpretability and debuggability.

2. **Multi-Context GRPO (Core End-to-End Training)**:
    - **Function**: Solves the issue where vanilla GRPO cannot be used directly because each turn within a trajectory has a different context.
    - **Mechanism**: Following the GRPO protocol, sample a group of $G$ trajectories for each question $q$. Calculate a final reward $R_i$ (format reward + F1-based answer reward) for each trajectory, and normalize these within the group to obtain a trajectory-level advantage $A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$. **Key Step**: Uniformly propagate $A_i$ to all turns in that trajectory, such that $A_{i,j} = A_i$ for all $j \in [1, n_i]$. Then, treat each turn as an independent PPO/GRPO optimization target. The objective is summed over all $(i,j)$, using the importance ratio $r_{i,j}(\theta) = \pi_\theta(T_{i,j}|c_{i,j}) / \pi_{\theta_{\text{old}}}(T_{i,j}|c_{i,j})$. Finally, a loss mask is applied to observation tokens returned by the search engine (excluding policy gradient) to stabilize training.
    - **Design Motivation**: Calculating the ratio for the entire trajectory under multi-context conditions is numerically unstable and provides sparse signals. Splitting by turn provides individual gradient signals for each turn, but since the reward is only available at the end, trajectory-level advantage is used to align turns. This effectively fuses "sparse outcome rewards" with "dense per-turn optimization" within the GRPO framework.

3. **Reward Design and Training Stability**:
    - **Function**: Pure rule-based rewards without process supervision, combining format checks and F1 answer evaluation.
    - **Mechanism**: $R = 0$ for format errors; $R = 0.1$ for correct format but F1=0 (to encourage learning the format early); $R = F1$ if F1 > 0. Normalized within the group to get $A_i$. Training hyperparameters: lr=1e-6, KL coef=0.001, clip=0.2, rollout group=5, temperature=1.0. Search engine tokens are entirely masked.
    - **Design Motivation**: Format rewards act as a "warm-up bonus" to prevent zero-reward signals in early training. Using F1 instead of EM provides a fine-grained reward, allowing partially correct answers to generate gradients. This represents a logical migration of DeepSeek-R1 style rule-based rewards to multi-turn search scenarios.

### Loss & Training
Training is based on the `verl` library with Qwen2.5-3B/7B/14B-Instruct backbones. The knowledge source is a 2018 Wikipedia dump with the E5 retriever. Training data uses the NQ + HotpotQA train splits from Search-R1. 3B/7B were run on 8×H100, and 14B on 2×8×H100. One epoch (256 batch, 5 rollouts) is sufficient. The reward curve shows two stages: a sharp rise in the first 25 steps (learning basic tools + memory usage), followed by a slow upward trajectory (refined strategy).

## Key Experimental Results

### Main Results

**7-benchmark EM Average (Avg.): 3B/7B/14B vs. SOTA Baselines**:

| Size | Strongest Baseline | Ours | Gain |
|------|--------------|------|---------|
| 3B | AutoRefine-3B-base = 40.5 | **MemSearcher-3B = 43.8** | +3.3 |
| 7B | ReSearch-7B = 43.6 / R1-Searcher-7B = 40.2 | **MemSearcher-7B = 48.9** | +5.3 |
| 14B+ | Search-R1-14B-base = 47.8 / ReSearch-32B = 48.3 | **MemSearcher-14B = 51.7** | +3.4 vs 32B |

**Key Findings**: MemSearcher-3B (43.8) already outperforms all 7B baselines; MemSearcher-7B (48.9) outperforms the 32B ReSearch. This indicates that model capacity saved by reducing context is redirected toward actual search reasoning.

**Per-dataset Results (Selected)**:

| Dataset | Search-R1-7B-base | ReSearch-7B | **MemSearcher-7B** |
|--------|-------------------|-------------|--------------------|
| NQ | 48.0 | 40.9 | **52.7** |
| TriviaQA | 63.8 | 63.7 | **68.1** |
| PopQA | 45.7 | 44.6 | 47.8 |
| HotpotQA | 43.3 | 43.5 | **50.8** |
| 2Wiki | 38.2 | 47.6 | 48.6 |
| Musique | 19.6 | 22.3 | **25.8** |
| Bamboogle | 43.2 | 42.4 | **48.8** |

### Ablation Study

**RL training vs. no training (Qwen2.5-Instruct base)**:

| Model | w/o training | w/ MemSearcher RL | Gain |
|------|--------------|-------------------|------|
| Qwen2.5-3B-Instruct | 14.4 | 43.8 | **+29.4** |
| Qwen2.5-7B-Instruct | 25.8 | 48.9 | **+23.1** |
| Qwen2.5-14B-Instruct | 27.7 | 51.7 | **+24.0** |

→ The framework requires RL to be unlocked; pure prompting is insufficient.

**RL vs. SFT (Qwen2.5-3B)**:

| Method | Avg |
|------|-----|
| SFT (Qwen2.5-72B Distilled Trajectories) | 28.5 |
| **RL** | **43.8** |

→ SFT using 72B distilled trajectories is 15.3 points lower than RL, as the 72B model itself has not mastered the MemSearcher paradigm and cannot act as a qualified teacher. RL directly rewards correct answers, letting the model learn what to retain.

**Memory Length Ablation (256/512/1024/2048 tokens)**:
- Simple datasets like Bamboogle saturate at 256 tokens.
- Complex multi-hop datasets like Musique show continuous improvement from 256 to 1024.
- 1024 tokens is chosen as the default trade-off sweet spot.

**Context Length Comparison (vs. ReAct-based ReSearch)**: MemSearcher's average context length across multiple turns remains <4K tokens (nearly flat); ReSearch grows linearly, exceeding 10K after 5 turns.

### Key Findings
- **Small Models Outperforming Large Models**: MemSearcher-3B > 7B baselines; 7B > 32B ReSearch. Compressing context allows the model to use its capacity where it matters most.
- **Outperforming Google Web Search**: MemSearcher on a local Wiki dump outperforms R1-Searcher and ZeroSearch using real Google searches, suggesting that the benefits of the memory design outweigh the benefits of web index quality.
- **Two-Stage Training**: Models quickly learn format and tool use in the first 25 steps, then gradually learn memory strategies, mirroring the two-stage learning pattern of DeepSeek-R1.
- **Low SFT Distillation Ceiling**: Because the teacher (72B) hasn't mastered MemSearcher, this is a classic case where small models must explore via RL to unlock a new paradigm's potential rather than imitating old paradigms from larger models.
- **Memory Length Sensitivity**: Complex tasks require longer memory, but 1024 is the engineering sweet spot; this suggests future work on "adaptive memory length."

## Highlights & Insights
- **"Backbone as Memory Manager" is an elegant minimalist design**: It avoids introducing extra modules or breaking end-to-end trainability. A single LLM performs reasoning, acting, and memorizing, making it more cohesive than external RAG/KG solutions.
- **Multi-Context GRPO is a general algorithmic contribution**: It can be applied to any multi-turn RL scenario (tool use, multi-turn dialogue, long-range planning) where turn-level contexts vary within a trajectory and final rewards are sparse.
- **Constant Context → Industrially Deployable**: The primary cost of search agent online services is the KV cache for long contexts. MemSearcher reduces this to $O(1)$, making it friendly for low-VRAM servers (e.g., 4090 / A10).
- **"Improving performance by learning to forget"**: Contrary to the intuition that "more context = better," this paper proves that selective forgetting and compact memory dominate cluttered history in search tasks. This is consistent with the core of the attention mechanism (focusing on the few, ignoring the many).
- **Capability Density Boost (3B vs 32B)**: A striking experimental result showing that in the agent era, substituting model scaling with paradigm innovation is a viable path.

## Limitations & Future Work
- **Limitations**: (1) The memory mechanism is simple (pure natural language overwrite); more complex RAG-like or structured memory was not explored. (2) Multi-context GRPO may still have length bias (longer trajectories contributing more).
- **Self-Critique**: (1) Memory overwriting is destructive, lacking a long-term archival mechanism for ultra-long tasks. (2) Experiments were limited to static Wiki dumps. (3) There is no explicit reward for memory quality, only inference from final answers. (4) Only the Qwen2.5 series was verified. (5) Format rewards (0.1) may mask failures where a model produces the correct format but irrelevant content.
- **Future Directions**: (1) Hierarchical memory (short-term working memory + long-term archival). (2) Incorporating auxiliary signals for memory quality. (3) Adaptive memory length. (4) Extending Multi-Context GRPO to dense per-turn rewards. (5) Generalization across other model families.

## Related Work & Insights
- **vs. ReAct / Search-R1 / ReSearch / AutoRefine (ReAct Path)**: These concatenate all history, leading to linear context growth. Ours reduces this to $O(1)$ while outperforming them.
- **vs. R1-Searcher / ZeroSearch (Real Web)**: Using real Google APIs was less effective than Ours using static Wiki, showing that the paradigm shift is more significant than the data source.
- **vs. MEM1 / MemAgent (Memory-based Agents)**: This is the first work to combine memory management and RL end-to-end using Multi-Context GRPO training.
- **vs. HippoRAG / Mem0 / Zep (External Structured Memory)**: These rely on KG/vector store indexing; Ours uses LLM self-managed online memory with zero external dependencies.
- **Insights**: (1) All multi-turn agents should consider compressing history rather than concatenating it. (2) Sparse outcome rewards and per-turn optimization can be fused via advantage broadcasting. (3) Backbone roles (reasoner+actor+manager) increase capacity efficiency. (4) New paradigm potential must be unlocked via exploration, not imitation.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of memory-managed search agents and Multi-Context GRPO is a clear paradigm shift in RL-based search agents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 datasets × 3 model sizes × 8 baselines + RL/SFT/no-training controls + memory length ablation + context curves + reward curves + case studies.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2 provide intuitive ReAct comparisons, formulas are rigorous, and complexity tables are clear.
- Value: ⭐⭐⭐⭐⭐ Open-source code, industrially deployable (<4K context), and the 3B model's ability to outperform 7B baselines is highly attractive. Multi-context GRPO is broadly applicable to all multi-turn agent RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](../../NeurIPS2025/llm_agent/lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)

</div>

<!-- RELATED:END -->
