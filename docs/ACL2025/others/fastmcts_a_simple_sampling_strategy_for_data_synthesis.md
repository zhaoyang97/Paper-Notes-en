---
title: >-
  [Paper Note] FastMCTS: A Simple Sampling Strategy for Data Synthesis
description: >-
  [ACL 2025][Monte Carlo Tree Search] FastMCTS proposes a lightweight reasoning data synthesis strategy inspired by MCTS. Through three key improvements—adaptive stay policy, dynamic exploration, and reserve simulation—it generates over 30% more correct reasoning paths than rejection sampling, achieving an average improvement of 3.9% across multiple mathematics benchmarks on the trained model.
tags:
  - "ACL 2025"
  - "Monte Carlo Tree Search"
  - "Data Synthesis"
  - "Reasoning Data"
  - "Rejection Sampling"
  - "Preference Optimization"
date: 2026-05-08
content_hash: 26b83d66fcbd6e20
---

# FastMCTS: A Simple Sampling Strategy for Data Synthesis

**Conference**: ACL 2025  
**arXiv**: [2502.11476](https://arxiv.org/abs/2502.11476)  
**Code**: None (The paper states it will be released soon)  
**Area**: Others  
**Keywords**: Monte Carlo Tree Search, Data Synthesis, Reasoning Data, Rejection Sampling, Preference Optimization

## TL;DR

FastMCTS proposes a lightweight reasoning data synthesis strategy inspired by MCTS. Through three key improvements—adaptive stay policy, dynamic exploration, and reserve simulation—it generates over 30% more correct reasoning paths than rejection sampling, achieving an average improvement of 3.9% across multiple mathematics benchmarks on the trained model.

## Background & Motivation

**Background**: Synthesizing high-quality multi-step reasoning data has proven to be an effective paradigm for enhancing the reasoning capabilities of LLMs. The dominant method is Rejection Sampling, which independently generates multiple candidate reasoning paths for each problem and retains the paths with correct answers as training data.

**Limitations of Prior Work**: Rejection sampling suffers from two fundamental limitations: (1) Low efficiency: particularly for long reasoning chains and hard problems, independent sampling always starts from scratch. Consequently, a vast amount of computation is wasted on repeatedly generating the same correct reasoning prefixes. (2) Unbalanced sampling: simple problems easily generate a large number of correct paths, while hard problems might yield none, leading to a severely skewed distribution of training data across different difficulty levels.

**Key Challenge**: While Monte Carlo Tree Search (MCTS), as a search algorithm capable of effectively exploring status spaces, could theoretically address these limitations, LLM reasoning fundamentally differs from games like Go. Specifically, the state space is not clearly defined, the reasoning cost is extremely high (each step requires autoregressive generation), and the outcome evaluation is relatively more deterministic. Directly applying standard MCTS to LLM data synthesis introduces enormous simulation overhead.

**Goal**: To design an MCTS variant tailored to LLM characteristics, aiming to significantly reduce the computational overhead while retaining the advantages of tree search.

**Key Insight**: The authors observe that in LLM reasoning, the outcome of a simulation is highly correlated with the quality of the reasoning path (unlike Go, where a single roll-out cannot definitely determine the quality of a state). Therefore, all paths generated during training can be retained instead of discarded, and search waste on unpromising nodes can be avoided via adaptive strategies.

**Core Idea**: Introduce three key modifications to MCTS: an adaptive stay policy (deciding whether to continue deeper based on the node's win rate), dynamic exploration parameters (adjusting the exploration-exploitation balance based on node scores), and reserve simulation (retaining complete reasoning trajectories from simulations as tree nodes in the cache instead of discarding them).

## Method

### Overall Architecture

FastMCTS iteratively builds a search tree with the input problem $q$ as the root node and each reasoning step as a child node. Each iteration consists of two merged phases: Selection and Simulation/Expansion, followed by updating node scores through backpropagating the verification results. After the search is complete, correct reasoning paths are extracted from the tree for SFT training, while preference pairs are constructed for DPO training based on the score disparities between different branches.

### Key Designs

1. **Adaptive Stay Policy**:

    - **Function**: Dynamically decides whether to continue going deeper into leaf nodes during the selection phase.
    - **Mechanism**: In the selection phase of standard MCTS, the algorithm recursively selects down to the leaf node before expansion. FastMCTS introduces a stay condition: if the Monte Carlo score of the current node, $score = win\_count / visit\_count$, is either extremely high (close to 1) or extremely low (close to 0), the search "stays" on the current node to expand a new branch directly, rather than continuing to select downwards. Staying at high-score nodes ensures path diversity for simple problems (similar to degenerating into rejection sampling), while staying at low-score nodes avoids wasting further budget on paths destined to fail. Specifically, staying is triggered when the score $\in (0, l_{low}] \cup [l_{high}, 1)$.
    - **Design Motivation**: For simple problems, excessive tree depth reduces path diversity due to over-sharing of prefixes. For extremely hard problems, searching deeper is less effective than starting from scratch with a completely new path.

2. **Dynamic Exploration**:

    - **Function**: Dynamically adjusts the exploration parameter $c$ in the UCT formula based on node scores.
    - **Mechanism**: The standard UCT formula is $UCT(i) = \frac{w_i}{n_i} + c \cdot \sqrt{\frac{\ln N_i}{n_i}}$. In FastMCTS, once a node has been visited more than once, $c$ is multiplied by the node's score. For high-scoring nodes (promising ones), the exploration weight is increased to discover more correct paths. For low-scoring nodes (less promising ones), the exploration weight is decreased to focus exploitation on known better branches.
    - **Design Motivation**: The goal of data synthesis is to generate as many correct paths as possible; therefore, exploration should be more aggressive in promising states, which differs from the standard exploration-exploitation balance in typical gaming strategies.

3. **Reserve Simulation**:

    - **Function**: Retains the complete reasoning trajectories generated in the simulation phase as tree nodes to avoid waste.
    - **Mechanism**: In standard MCTS, the simulation phase performs random roll-outs from the selected node to the end of the game, but the generated paths are discarded, as the roll-out only serves to backpropagate score updates. In the LLM scenario, since each simulation requires autoregressively generating a full reasoning chain, discarding these paths represents a massive waste of resources. FastMCTS merges the expansion and simulation into a single phase: starting from the selected node, it generates a new reasoning branch until a final answer is produced, making every step of this generation a new node added to the tree. Additionally, prior to each simulation, different few-shot examples are randomly concatenated to increase reasoning path diversity.
    - **Design Motivation**: The correctness of the final answer in LLM reasoning is highly correlated with intermediate steps (unlike Go's random simulation); therefore, the simulation results themselves constitute valuable data.

### Loss & Training

The data generated by FastMCTS supports two training paradigms: (1) SFT: extracting correct reasoning paths from the tree, prioritizing paths from different branches for each problem to maximize diversity. (2) Branch-DPO: leveraging the score differences of tree nodes to construct preference pairs—nodes with a persistent Monte Carlo estimate of zero are treated as "low-quality nodes" and paired with nodes that lead to correct answers, forming step-level or branch-level preference data. DPO is trained using $\beta = 0.4$, a learning rate of $1 \times 10^{-6}$, and the AdamW optimizer.

## Key Experimental Results

### Main Results

Training results of Qwen2.5-7B on English mathematical reasoning benchmarks (using the EN Math Hard dataset):

| Method | #Data | GSM8K | MATH | AIME24 | AMC23 | OmniMath | Average |
|------|-------|-------|------|--------|-------|----------|------|
| Qwen2.5-7B (Baseline) | - | 88.2 | 66.8 | 0 | 47.5 | 35.5 | 49.7 |
| RS (5 traj/prob) | 111K | 89.1 | 72.0 | 6.7 | 52.5 | 38.3 | 52.4 |
| FastMCTS (5 traj) | 132K | 88.9 | 73.0 | 13.3 | 57.5 | 39.8 | 54.8 |
| RS (16 traj/prob) | 197K | 87.1 | 70.0 | 10.0 | 52.5 | 37.2 | 52.7 |
| FastMCTS (16 traj) | 288K | 88.9 | 74.0 | 20.0 | 60.0 | 38.3 | 55.6 |
| + Branch-DPO | 152K | 89.9 | 75.4 | 20.0 | 57.5 | 39.2 | **56.6** |

### Ablation Study

Sampling 25 reasoning trajectories per problem over 300 AIME problems:

| Method | Problem Solve Rate (%) | Avg. Correct Paths |
|------|-------------|--------------|
| Rejection Sampling | 61.3 | 7.22 |
| FastMCTS (Full) | 61.7 | 7.95 |
| w/o Adaptive Stay | 55.9 | 7.59 |
| w/o Dynamic Exploration | 61.7 | 7.28 |
| w/o Stay & Dynamic | 55.9 | 7.32 |

### Key Findings

- **Significant sampling efficiency advantage**: As the number of generated tokens increases, FastMCTS produces over 30% more correct reasoning paths than rejection sampling, featuring a higher proportion of effective tokens.
- **Adaptive stay policy primarily impacts the solve rate** (61.7% vs. 55.9%): Deciding whether to search deeper or open new paths based on the estimated quality of the current path is especially critical for hard problems.
- **Dynamic exploration primarily affects the number of correct paths** (7.95 vs. 7.28): It identifies more correct paths along promising branches by adjusting the exploration parameters.
- FastMCTS excels most in hard problems—on AIME-level competition tasks, the performance improves from 0 to 20, whereas rejection sampling only climbs to 10.
- Branch-DPO yields additional performance gains (55.6% -> 56.6%), validating the efficacy of step-level preference signals extracted from tree structures.
- Consistent trends are observed on Chinese math datasets: FastMCTS reaches 62.3% on Gaokao Math 2024, compared to 60.9% for rejection sampling.

## Highlights & Insights

- The design of **reserve simulation** is ingenious. A critical distinction between LLMs and Go is that the simulation results of LLMs are already valuable reasoning paths, and throwing them away is pure waste. This seemingly simple modification delivers substantial efficiency improvements and reflects a deep understanding of the problem's nature.
- **Difficulty-adaptive sampling** is one of the core strengths of FastMCTS. While rejection sampling treats all problems equally, FastMCTS automatically allocates more search budget to hard problems and falls back to highly efficient rejection sampling for simple ones. This difficulty-aware balancing is exceptionally valuable in practical deployment.
- Step-level preference data naturally derived from the tree structure represents an added benefit: different branches from the same tree naturally form contrastive pairs, eliminating the need for extra annotation.

## Limitations & Future Work

- Only Qwen2.5-72B-Instruct is utilized as the policy model; stronger reasoning models (such as DeepSeek-R1, o1) have not been tested, meaning the performance upper bound could be higher.
- Due to computing resource constraints, training experiments are only performed on Qwen2.5-7B, leaving its effectiveness on larger models unverified.
- The reduction in reasoning path diversity caused by shared prefixes in the tree structure remains an open question for future study.
- Future work can explore integrating FastMCTS with process reward models, replacing Monte Carlo estimates with learned process rewards.
- The code has not yet been open-sourced, and reproducibility is to be confirmed.

## Related Work & Insights

- **vs. Rejection Sampling**: Under the same computational budget, FastMCTS systematically outperforms rejection sampling—yielding more correct paths, higher effective token rates, and a more balanced difficulty distribution.
- **vs. Standard MCTS (AlphaMath, REST-MCTS*)**: These approaches directly apply MCTS to LLM reasoning, which incurs heavy simulation overhead. FastMCTS substantially reduces this overhead through reserve simulation and adaptive strategies.
- **vs. DART-Math**: Both focus on difficulty-balanced sampling, but DART-Math accomplishes this via post-hoc filtering. FastMCTS regulates it dynamically during the sampling process, which is more efficient.
- The tree search methodology of FastMCTS can be extended to other multi-step reasoning data-synthesis scenarios, such as code generation and logical reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The three improvements are individually simple yet powerful in combination, particularly the keen insight of reserve simulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive—including sampling efficiency comparisons, Chinese and English training results, detailed ablation studies, and difficulty distribution analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed method descriptions and complete pseudocode of the algorithm.
- Value: ⭐⭐⭐⭐⭐ Highly practical as a drop-in replacement for rejection sampling, exerting a direct impact on reasoning data synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] AutoMixer: Checkpoint Artifacts as Automatic Data Mixers](automixer_checkpoint_artifacts_as_automatic_data_mixers.md)

</div>

<!-- RELATED:END -->
