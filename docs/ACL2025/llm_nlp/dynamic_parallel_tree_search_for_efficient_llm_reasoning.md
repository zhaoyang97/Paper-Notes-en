---
title: >-
  [Paper Note] Dynamic Parallel Tree Search for Efficient LLM Reasoning
description: >-
  [ACL 2025][LLM (Other)][Tree of Thoughts] This work proposes the DPTS (Dynamic Parallel Tree Search) framework. It introduces a Parallelism Streamline to resolve the parallelization difficulty caused by frequent path switching in tree search. Additionally, its Search and Transition Mechanism incorporates early stopping and deep search strategies to reduce redundant exploration of low-confidence paths. DPTS achieves a $2\times$ to $4\times$ inference acceleration on Qwen-2.5 a…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Tree of Thoughts"
  - "Tree Search"
  - "Parallel Inference"
  - "MCTS"
  - "LLM Inference Acceleration"
  - "Early Stopping"
  - "Deep Search"
date: 2026-05-08
content_hash: 2129def92934cb5d
---

# Dynamic Parallel Tree Search for Efficient LLM Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.16235](https://arxiv.org/abs/2502.16235)  
**Code**: Not publicly released  
**Area**: LLM/NLP  
**Keywords**: Tree of Thoughts, Tree Search, Parallel Inference, MCTS, LLM Inference Acceleration, Early Stopping, Deep Search

## TL;DR

This work proposes the DPTS (Dynamic Parallel Tree Search) framework. It introduces a Parallelism Streamline to resolve the parallelization difficulty caused by frequent path switching in tree search. Additionally, its Search and Transition Mechanism incorporates early stopping and deep search strategies to reduce redundant exploration of low-confidence paths. DPTS achieves a $2\times$ to $4\times$ inference acceleration on Qwen-2.5 and Llama-3 while maintaining or exceeding the reasoning accuracy of existing algorithms like MCTS.

## Background & Motivation

Tree of Thoughts (ToT) significantly enhances multi-step reasoning capabilities by modeling LLM inference as a tree search process (e.g., MCTS). However, current ToT methods **focus exclusively on search accuracy while ignoring computational efficiency**, posing two core challenges:

### Challenge 1: Frequent Path Switching Obstacles Parallelization

Traditional serialized tree search exhibits irregular computational trajectories, characterized by frequent backtracking and recursive switching between different paths. This is incompatible with the end-to-end parallelization of conventional deep learning inference:
- When parallelizing different paths, the context lengths and KV caches of nodes differ, requiring extra processing.
- Frequent switching leads to shallow exploration; under limited time and memory budgets, exploring more paths means insufficient deep exploration of each individual path.

**Quantitative Analysis**: On the Math500 dataset, the average number of total path switches per sample is approximately 35, and the number of switches from the optimal path to a suboptimal path is up to 3, demonstrating search instability.

### Challenge 2: Redundant Exploration of Low-Confidence Paths

Existing MCTS roll out selected nodes until termination conditions are met, even if the initial confidence of the node is very low. Statistical analysis reveals that:
- Low-confidence nodes have a **91.3%** probability of producing suboptimal results.
- Low-confidence nodes have only a **6.2%** probability of reaching the optimal path first.
- A massive amount of computational resources is wasted on paths that are highly unlikely to yield the optimal solution.

## Method

### Overall Architecture

DPTS consists of two core components:
1. **Parallelism Streamline** (Generation Phase) — Supports flexible parallel expansion of arbitrary paths.
2. **Search and Transition Mechanism** (Selection Phase) — Dynamically balances exploration and exploitation to reduce redundancy.

### Key Designs

#### 1. Parallelism Streamline

**Tree Structure Construction**: Each node contains an ID, parent node reference, prior confidence, KV cache (storing only this node's cache rather than the full sequence), and the token sequence from the root to the current node. Node-level KV cache significantly reduces memory usage.

**KV Cache Processing**: Different path lengths yield varying KV cache sizes, preventing direct parallelization. Solutions:
- Left-pad the KV cache of shorter sequences with zeros padding.
- Align input sequences using padding tokens to ensure that all sequences within a batch are of equal length.
- Clean up useless KV caches when a leaf node terminates or all child branches are completed, thereby releasing memory.

**Adaptive Parallel Generation**: Dynamically adjust the number of parallel paths based on the available GPU memory:

$$|P| = \frac{O_{\text{max}} - O_{\text{init}}}{O_{\text{peak}} - O_{\text{init}}}$$

As the tree grows, the memory footprint of intermediate results continuously increases. Adaptive adjustment prevents premature termination caused by out-of-memory (OOM) errors.

#### 2. Search Mechanism

Nodes in the parallel queue $P$ are categorized into two types:
- **Exploitation Nodes**: Inherited from parent exploitation nodes, focusing on deepening the most promising paths. They inherit the parent node's status when the child node's confidence exceeds a threshold.
- **Exploration Nodes**: Dynamically re-selected from the candidate pool $N$, responsible for discovering new high-potential paths. At each step, the node with the highest confidence that is not designated as an exploitation node is selected from the candidate pool.

The allocation ratio $p$ can be manually adjusted based on the task and memory budget.

#### 3. Transition Mechanism

Two bidirectional transition strategies are designed to dynamically evolve the search space:

**Early Stop (Exploitation $\to$ Exploration)**: When the confidence of the best child node of an exploitation node falls below the threshold $\theta_{es}$, the node is removed from the queue, stopping further exploitation of low-potential paths. The threshold is defined as:

$$\theta_{es} = \begin{cases} \lambda_{es} \cdot \frac{1}{|\mathcal{N}|}\sum_{i \in \mathcal{N}} c_i, & \text{if } t \leq t^* \\ \max_{i \in \mathcal{N}} c_i, & \text{otherwise} \end{cases}$$

Using a mean-based strategy in the early stage ($t \leq t^*$) allows for more exploration, while switching to a max-based strategy in the later stage offers more aggressive pruning.

**Deep Seek (Exploration $\to$ Exploitation)**: When the confidence of an exploration node exceeds the threshold $\theta_{ds}$, it is promoted to an exploitation node for deep reasoning. This enables promising exploration findings to be exploited on a deeper level.

The dynamic balance of the two strategies: Deep Seek temporarily increases the proportion of exploitation nodes $\to$ $\theta_{es}$ rises $\to$ more exploitation nodes are pruned by Early Stop $\to$ the proportion naturally falls back.

## Key Experimental Results

### Main Results — Comparison of Search Algorithms

| Model | Algorithm | Math500 Acc. | Math500 Time(s) | GSM8K Acc. | GSM8K Time(s) |
|------|------|:---:|:---:|:---:|:---:|
| Qwen-2.5-1.5B | MCTS | 56.6 | 117.37 | 75.1 | 73.28 |
| | Best-of-N | 52.6 | 89.87 | 70.1 | 33.37 |
| | Beam | 52.4 | 104.58 | 71.5 | 41.27 |
| | **DPTS** | **59.2** | **45.10** | **75.2** | **18.32** |
| Qwen-2.5-7B | MCTS | 75.2 | 121.46 | 89.6 | 79.68 |
| | **DPTS** | **76.2** | **53.50** | 89.4 | **19.95** |
| Llama-3-3B | MCTS | 48.6 | 111.80 | 64.0 | 57.19 |
| | **DPTS** | **50.8** | **47.75** | **67.8** | **27.74** |
| Llama-3-8B | MCTS | 54.2 | 143.36 | 69.5 | 69.74 |
| | **DPTS** | **55.4** | **37.98** | 68.2 | **17.82** |

**Acceleration Summary**:
- Average $2.2\times$ to $3.8\times$ speedup on Math500.
- Average $2.0\times$ to $3.9\times$ speedup on GSM8K.
- $3.9\times$ speedup for Qwen-2.5-7B on GSM8K (79.68s $\to$ 19.95s).
- $3.8\times$ speedup for Llama-3-8B on Math500 (143.36s $\to$ 37.98s).

### Ablation Study — Qwen-2.5-1.5B, Math500

| Configuration | |P| | Acc. | Time(s) | Best Index |
|------|:---:|:---:|:---:|:---:|
| Baseline (MCTS) | 1 | 56.6 | 117.37 | 10.45 |
| +Adaptive Parallel | AP | 58.8 | 108.06 | 8.27 |
| +Search | AP | 58.2 | 76.81 | 4.66 |
| +Transition only | AP | 57.0 | 32.22 | 2.51 |
| **+Search+Transition** | AP | **59.2** | **45.10** | **4.17** |

- Parallelization itself improves accuracy (the search tree is larger and more comprehensive).
- The Search mechanism reduces time by 28.9%.
- While Transition-only is the fastest, its accuracy is lower due to the lack of exploration.
- The Search+Transition combination achieves the best balance.

### Hyperparameter Analysis — Transition Threshold $\lambda$

| $\lambda_{es}$ | $\lambda_{ds}$ | Acc. | Time(s) | ES% | DS% |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.0 | 1.0 | 53.0 | 47.59 | 41.4 | 10.5 |
| 0.9 | 0.9 | 58.6 | 43.30 | 15.6 | 20.9 |
| 0.8 | 0.8 | 58.0 | 46.33 | 8.1 | 23.9 |
| 0.6 | 0.6 | 57.4 | 44.39 | 6.1 | 32.3 |
| 0.4 | 0.4 | 56.6 | 38.41 | 0 | 0.1 |

$\lambda \in [0.6, 0.8]$ serves as a robust operating range. An overly large value (1.0) leads to overly aggressive early stopping, while an overly small value (0.4) degenerates it into pure exploitation nodes.

### Key Findings

1. **Simultaneous Speed and Quality**: DPTS achieves accuracy equal to or higher than MCTS across all models and datasets using less than half the inference time, breaking the traditional "trade-off between search speed and quality".
2. **Crucial Role of Early Stop on Hard Tasks**: Without Early Stop on Math500, the tree often runs until a timeout, drastically increasing inference time. DPTS allows the search tree to terminate early within limited expansions.
3. **Significant Drop in Best Index**: Best Index drops from 10.45 in MCTS to 4.17 in DPTS. On average, the optimal solution can be found by terminating only 4 paths, representing a $2.5\times$ efficiency improvement.
4. **Adaptive Parallelism Directly Boosts Accuracy**: Parallelization allows the tree to grow larger and more comprehensive within the same timeframe, capturing better solutions.
5. **Synergy of Search and Transition**: Using Transition-only (Early Stop) is the fastest but yields insufficient accuracy (lacking the replenishment of exploration branches). Combining them achieves optimal performance.

## Highlights & Insights

1. **Paradigm Shift from "Better Search" to "More Efficient Search"**: Previous work in the ToT field focused excessively on search accuracy. DPTS systematically addresses the efficiency bottleneck for the first time, which is crucial for the practical deployment of LLM inference.
2. **Elegant Design of Dynamic Balance in Bidirectional Transition**: Early Stop and Deep Seek form a natural negative feedback loop: Deep Seek increases exploitation nodes $\to$ raising the Early Stop threshold $\to$ more nodes are pruned $\to$ the proportion naturally falls back. This achieves automatic stabilization without manual balancing.
3. **Data-Driven Design Decisions**: The statistical observation that 91.3% of low-confidence nodes yield suboptimal solutions directly motivated the design of the Early Stop strategy. This "observe-then-design" methodology is highly instructive.
4. **Detailed and Practical KV Cache Engineering Optimizations**: Node-level cache storage, left-padding alignment, adaptive parallel queue sizing, and cleaning on completion constitute essential engineering contributions for integrating tree search with GPU parallelization.
5. **Abundant Visual Evidence**: The tree visualization in Figure 6 intuitively demonstrates that DPTS generates a "narrow but deep" tree, focusing on promising paths for in-depth reasoning.

## Limitations & Future Work

- **Validation Limited to Mathematical Reasoning Tasks**: Other fields such as programming and scientific reasoning have not been tested, meaning generalizability remains to be verified.
- **No Hardware-Level Optimization**: The transmission overhead of KV cache does not utilize low-level methods like prefix sharing. DPTS is orthogonal to these and could be accelerated further.
- **Lack of Adaptive Thresholding for Early Stop**: The parameter $\lambda$ must be set manually; while it is robust within $[0.6, 0.8]$, it does not adjust dynamically based on task difficulty.
- **Dependence on Process Reward Models (PRMs)**: DPTS relies on an external PRM to provide node confidence scores. The quality of the PRM directly affects the search performance.
- **Insufficient Analysis of Memory Consumption**: Although KV cache cleanup is discussed, a systematic memory footprint comparison against baseline methods is lacking.

## Related Work & Insights

- **Application of MCTS in LLMs**: DPTS does not replace the selection strategy of MCTS; instead, it raises its efficiency by introducing parallelization and redundancy elimination.
- **Deft**: Optimizes prefix sharing to reduce data transmission. This is orthogonal to DPTS, and their combination could yield even higher acceleration.
- **Inspirations for Test-Time Compute**: DPTS demonstrates how to "search smarter" under a fixed computational budget, which has direct implications for the optimization of inference-heavy models like OpenAI-o1.
- **Inspirations for General Search Algorithms**: The bidirectional transition concept of Early Stop and Deep Seek can be extended to other search-intensive tasks like Neural Architecture Search (NAS) and path planning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework design of parallel search combined with dynamic transition is novel, filling a gap in ToT efficiency optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 models $\times$ 2 datasets $\times$ 4 search algorithms + thorough ablations + hyperparameter analysis + visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and analysis, with a well-structured logical chain from empirical observations to methodological design.
- **Value**: ⭐⭐⭐⭐⭐ A $2\times$ to $4\times$ acceleration is highly significant for LLM inference deployment, carrying immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)

</div>

<!-- RELATED:END -->
