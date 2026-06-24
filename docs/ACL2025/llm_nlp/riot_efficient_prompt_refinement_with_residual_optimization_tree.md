---
title: >-
  [Paper Note] RiOT: Efficient Prompt Refinement with Residual Optimization Tree
description: >-
  [ACL 2025][LLM (Other)][prompt optimization] The authors propose Residual Optimization Tree (RiOT), an automatic prompt optimization framework that manages the optimization process through a tree structure, enhances diversity via perplexity-based node selection, and mitigates semantic drift with text residual connections.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "prompt optimization"
  - "residual connection"
  - "perplexity"
  - "tree search"
  - "semantic drift"
date: 2026-05-08
content_hash: 49a94d875a67970a
---

# RiOT: Efficient Prompt Refinement with Residual Optimization Tree

**Conference**: ACL 2025  
**arXiv**: [2506.16389](https://arxiv.org/abs/2506.16389)  
**Code**: [github.com/Qing1Zhong/RiOT](https://github.com/Qing1Zhong/RiOT)  
**Area**: Automatic Prompt Optimization / LLM  
**Keywords**: prompt optimization, residual connection, perplexity, tree search, semantic drift  

## TL;DR

The authors propose Residual Optimization Tree (RiOT), an automatic prompt optimization framework that manages the optimization process through a tree structure, enhances diversity via perplexity-based node selection, and mitigates semantic drift with text residual connections.

## Background & Motivation

### Problem Background
The performance of LLMs highly depends on prompt design, whereas manual prompt engineering requires extensive domain knowledge and trial-and-error. Although automatic prompt optimization methods are emerging, existing methods face two key challenges:

### Key Challenges

**Insufficient Diversity**: Existing methods (e.g., OPRO, TextGrad, ProTeGi) generate only one candidate prompt per round, restricting the breadth of the search space. Although APE generates multiple candidates in parallel, it lacks an iterative refinement mechanism.

**Semantic Drift**: During iterative prompt optimization, optimizing for a specific task might degrade the performance on preceding tasks, resembling the stability-plasticity dilemma in continual learning.

### Analogous Inspiration
The semantic drift problem is analogous to vanishing gradients in deep networks and catastrophic forgetting in continual learning. RiOT addresses this issue by drawing inspiration from residual connections in deep learning.

## Method

### Overall Architecture

RiOT constructs a tree structure with the initial prompt $p_0$ as the root node. At step $t$:
1. **Generation**: The parent node $p_t$ generates $K$ candidate child nodes through the Prompt Optimization Operator $\mathcal{M}(\cdot)$.
2. **Selection**: Select the optimal child node based on perplexity (pruning).
3. **Fusion**: Fuse the parent node and the optimal child node via a text residual connection.

### Key Design 1: Perplexity-Informed Node Selection

Traditional methods select candidates solely based on quality, neglecting diversity. RiOT leverages perplexity to select the optimal candidate:

$$p_{t+1}^* = \arg\max_i PPL(p_{t+1}^{(i)})$$

**Rationale for Selecting High Perplexity**:
- From an information theory perspective, high perplexity implies lower token co-occurrence probability, thus containing more information.
- This resembles the strategy in Bayesian optimization that prioritizes exploring high-uncertainty regions.
- Prompts with high perplexity are more likely to represent novel, under-explored optimization directions.

### Key Design 2: Text Residual Connection

Core Idea: Instead of directly using the optimized child prompt as the input for the next round, RiOT **fuses the semantic content of the parent and child nodes**:

$$p_t = \mathcal{G}(p_{t-1}, p_t^*)$$

**Content Fusion Algorithm** (Algorithm 1):
1. Segment both the parent prompt and the child prompt into sentences.
2. Compute semantic representations for each sentence using a pre-trained embedding model.
3. Calculate the cosine similarity matrix between the sentences of the parent and child prompts.
4. **Retain** sentences from the parent node that have high similarity ($\geq 1-b_1$) with the child node (these represent the preserved valuable content).
5. **Introduce** new content from the child node that has low similarity ($< 1-b_2$) with the parent node.
6. Merge both components to form the new prompt.

Two hyperparameters $b_1$ and $b_2$ control the degree of fusion:
- $b_1 = 0.25$: Controls the content preserved from the parent node.
- $b_2 = 0.5$: Controls the new content introduced from the child node.

### Prompt Optimization Operator

Based on TextGrad as the backbone:
1. The target model $\mathcal{F}_{\text{target}}$ generates responses on the training set using the current prompt.
2. Compute the loss $\mathcal{L}(p_t)$.
3. The optimization model $\mathcal{F}_{\text{opt}}$ proposes candidate prompts based on the textual gradient $\nabla_{p_t}\mathcal{L}(p_t)$.
4. The inherent variability of LLM outputs naturally produces $K$ distinct candidates.

## Key Experimental Results

### Experimental Settings
- **Target Model**: GPT-3.5-turbo (temperature=0)
- **Optimization Model**: GPT-4o (temperature=0)
- **Embedding Model**: text-embedding-3-large
- **Tree Width $K=3$**, trained with 3 epochs $\times$ batch size 4 = 15 iterations
- **5 Benchmarks**: LogiQA 2.0, StrategyQA, Object Counting, GSM8K, Date Understanding

### Main Results

| Method | LogiQA 2.0 | StrategyQA | Object Counting | GSM8K | Date Understanding |
|------|-----------|-----------|----------------|-------|-------------------|
| Zero-Shot CoT | 59.0 | 65.8 | 71.0 | 60.2 | 76.1 |
| APE | 57.4 | 70.4 | 79.2 | 76.6 | 72.8 |
| OPRO | 58.0 | 67.0 | 79.9 | 72.8 | 76.3 |
| TextGrad | 60.0 | 68.8 | 88.3 | 74.8 | 71.9 |
| DSPy | 59.8 | 73.4 | 84.5 | 79.0 | 74.3 |
| **RiOT** | **61.4** | **74.6** | 86.9 | **81.2** | **78.2** |

- RiOT achieves the best performance on **4/5** tasks, outperforming the strongest baseline by **+2.2%** on GSM8K.
- The weighted average accuracy exceeds the best baseline by **+2.7%**.

### Ablation Study (GSM8K)

| Variant | Accuracy |
|------|--------|
| RiOT (Full) | **81.2 ±1.2** |
| w/o Text Residual Connection | 68.8 ±3.9 (-12.4) |
| w/o Perplexity-Informed Node Selection | 68.2 ±1.8 (-13.0) |

Both core components contribute approximately 12-13% of absolute improvement, demonstrating that both are indispensable.

### Nodes Selection Metric Comparison

| Metric | Accuracy |
|------|--------|
| Perplexity | **81.2** |
| Entropy | 78.6 (-2.6) |
| Length | 73.6 (-7.6) |

Perplexity significantly outperforms entropy and length, validating the effectiveness of perplexity-based selection.

### Transferability Experiments (Evaluated on Gemini-1.5-flash)
- **Prompt Transfer** (Prompts optimized for GPT-3.5 applied to Gemini): Mentions a slight average decrease.
- **Model Transfer** (Directly optimizing for Gemini): Consistent improvements.
- This indicates that optimized prompts possess a certain degree of cross-model transferability.

### Key Findings
1. Scaling returns for Few-shot CoT diminish—there is negligible improvement from 4-shot to 20-shot.
2. RiOT is the only method that achieves improvements across all 5 tasks, whereas other methods suffer degradation on specific tasks.
3. A tree width of $K=3$ is optimal; larger widths tend to decrease performance.

## Highlights & Insights

1. **First prompt optimization framework featuring tree structures and residual connections**: It elegantly resolves the dual challenges of search diversity and semantic drift.
2. **Counter-intuitive insight from perplexity-based selection**: Choosing "less certain" prompts actually facilitates finding more valuable exploration directions.
3. **Innovative text residual connection**: Successfully ports the concept of residual connections from deep learning to discrete text spaces, preserving valuable semantic components.
4. **Cross-task stability**: RiOT is the only method in the experiments to achieve consistent improvements across all tasks.
5. **Ablation studies demonstrate equal importance**: Both core components contribute an approximately 12-13% performance boost, showing they are equally crucial.

## Limitations & Future Work

1. High cost due to dependency on powerful optimizer models (e.g., GPT-4o) and embedding models.
2. The selection of hyperparameters $b_1$, $b_2$, and $K$ requires tuning on a validation set.
3. Performance is suboptimal on the Object Counting task (TextGrad 88.3% vs. RiOT 86.9%).
4. Sentence-granularity fusion in text residual connections may miss fine-grained optimizations within sentences.

## Related Work & Insights

- **Soft Prompt Tuning**: PEFT-style methods, which require access to model parameters.
- **Discrete Token Search**: Gradient-guided or RL-based methods, which require full access to the model.
- **Black-box Prompt Optimization**: APE (parallel candidates), OPRO (meta-prompt optimization), TextGrad (textual gradients), and DSPy (declarative program synthesis).
- **Semantic Drift / Catastrophic Forgetting**: Text-space analogies of continual learning methods such as parameter isolation and regularization.

## Rating

⭐⭐⭐⭐ — Elegant methodological design (tree + residual + perplexity), comprehensive experiments, and thorough ablation studies. The contributions of the two core components are clear. The main drawbacks lie in the dependency on powerful models (GPT-4o) and the limited depth of analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)

</div>

<!-- RELATED:END -->
