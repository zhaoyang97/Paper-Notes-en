---
title: >-
  [Paper Note] Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective
description: >-
  [ICML 2025][Interpretability][prompt pruning] Proposes the PromptQuine framework, which performs token-level pruning on ICL prompts through evolutionary search. It discovers that pruning clear exemplars into seemingly "gibberish" subsequences can actually improve LLM performance, matching or surpassing SOTA prompt optimization methods.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "prompt pruning"
  - "evolutionary search"
  - "in-context learning"
  - "PromptQuine"
  - "open-endedness"
date: 2026-05-08
content_hash: 191c784b0e9cc44e
---

# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective

**Conference**: ICML 2025  
**arXiv**: [2506.17930](https://arxiv.org/abs/2506.17930)  
**Code**: -  
**Area**: Interpretability  
**Keywords**: prompt pruning, evolutionary search, in-context learning, PromptQuine, open-endedness  

## TL;DR

Proposes the PromptQuine framework, which performs token-level pruning on ICL prompts through evolutionary search. It discovers that pruning clear exemplars into seemingly "gibberish" subsequences can actually improve LLM performance, matching or surpassing SOTA prompt optimization methods.

## Background & Motivation

Traditional wisdom suggests that well-designed instructions and exemplars are crucial for ICL performance. However, recent studies have shown that "unnatural language" prompts sometimes yield better results. This paper systematically challenges this conventional cognitive paradigm:

- **Core Hypothesis (Partial Context Hypothesis)**: Given a natural language ICL prompt $\mathbf{x}=(x_1,...,x_n)$, pruning certain tokens to obtain a subsequence $\mathbf{z}=(z_1,...,z_m), m \leq n$ can significantly improve task performance.
- **Motivation**: LLMs may only undergo "superficial alignment," where their internal hypotheses override the explicit structure of human language; some input features are redundant for task prediction.
- **Key Findings**: Existing attribution methods and prompt compression algorithms are unreliable in guiding pruning.

## Method

### 1. Problem Formulation: Compression as Guided Search

Redefining the traditional prompt compression problem as a guided search problem:

$$\mathbf{z}^* = \arg\max_{\mathbf{z} \subseteq \mathbf{x}} f(\mathbf{z}; \mathbf{x}, \mathcal{D})$$

where $f$ is a non-differentiable task objective function, and the search space consists of all fixed-order subsequences of the original prompt.

### 2. Baseline: Hill-Climbing Search (TAPruning)

Adopts the Threshold Accepting algorithm, attempting token-by-token pruning from left to right:
- Denotes acceptance if deleting a token does not decrease validation set performance (or the decrease is within a threshold, e.g., $\geq 96\%$ of the current optimum).
- Iterates for multiple rounds until no more pruning can be performed.
- Already matches SOTA methods such as Promptbreeder.

### 3. Search Landscape Analysis

- Validates the **multimodality** of the landscape by randomizing the pruning order.
- Random Search (RS) is highly inefficient, whereas Evolutionary Search (ES) is significantly better at obtaining high-quality prompts.
- The success rate of RS compared to ES approaches zero as task difficulty increases.

### 4. PromptQuine: Evolutionary Search Framework

Based on Genetic Algorithm (GA), key designs:
- **Encoding**: Binary token masks serve as genotypes, and the pruned ICL prompts serve as phenotypes.
- **Mutation**: Bit-flip (1→0) operations, randomly selecting $\{1,2,3,4\}$ bits to flip.
- **Selection**: Tournament selection + reduced selection pressure to avoid local optima.
- **Regularized Evolution**: Only new offspring compete for population slots, effectively addressing premature convergence in the ICL landscape.
- **Calibrated Reranking**: Reranking the elite prompts using full validation set accuracy.

## Experimental Results

### Main Results: Classification Tasks (Llama-3-8B-Instruct)

| Method | SST-2 | Subj | AG's News | Yelp-5 | SNLI | Yahoo | Average |
|------|-------|------|-----------|--------|------|-------|------|
| ICL (1-shot) | 95.9 | 66.7 | 83.7 | 52.2 | 61.9 | 57.1 | 69.6 |
| Promptbreeder | 96.0 | 83.6 | 88.6 | 59.3 | 64.2 | 62.9 | 75.8 |
| TAPruning (1-shot) | 95.0 | 74.5 | 88.6 | 60.2 | 68.6 | 61.7 | 74.8 |
| **PromptQuine (1-shot)** | **96.2** | **86.5** | **89.2** | 59.7 | 69.2 | 64.2 | **77.5** |
| **PromptQuine (4-shot)** | 96.4 | **93.1** | 89.4 | **64.3** | **78.6** | **66.2** | **81.3** |

- PromptQuine matches or surpasses all SOTA methods under the 1-shot setting.
- Performance further surges under the 4-shot setting, with SNLI rising from 69.2% to 78.6%.
- It simultaneously achieves a prompt compression rate of approximately 53%.

### Ablation Study: Runtime Efficiency

TAPruning and PromptQuine are the **first token-level search methods capable of completing execution within minutes**, with operational efficiency far exceeding methods like RLPrompt.

## Highlights & Insights

- Groundbreaking discovery: Pruning ICL exemplars into "gibberish" actually boosts performance, revealing that LLMs' preferences for prompts differ drastically from human intuition.
- Redefining prompt compression as a search problem, unifying the perspectives of compression and optimization.
- The evolutionary search framework naturally supports parallelization and possesses excellent scalability.
- The method is simple and practical, requiring no access to model gradients, making it applicable to any black-box LLM.

## Related Work & Insights

| Method | Search Space | Requires Gradient | Search Granularity | Runtime |
|------|---------|------------|---------|---------|
| RLPrompt | Any token | Yes | Token-level | Several hours |
| EvoPrompt | Natural language instructions | No | Instruction-level | Several hours |
| Promptbreeder | Natural language + mutated prompts | No | Instruction-level | Tens of minutes |
| LLMLingua | Compression | No | Token-level | Several seconds |
| **PromptQuine** | ICL subsequences | No | Token-level | **Several minutes** |

PromptQuine is the first method capable of completing token-level search within minutes without requiring model gradients.

## Limitations & Future Work

- Search space is constrained to fixed-order subsequences, with token reordering remaining unexplored.
- Evolutionary search is sensitive to hyperparameters (e.g., population size, mutation rate).
- The discovered "optimal" pruning patterns lack interpretability, making them difficult to translate into generalizable rules.
- Validated only on classification and generation tasks, while its effectiveness on more complex reasoning tasks remains unknown.
- The transferability of pruning strategies across different LLMs has not been explored.

## Rating

⭐⭐⭐⭐ — Novel perspective and thorough experimentation. It systematizes the counter-intuitive finding that "gibberish works better than well-designed prompts," providing critical insights into the mechanisms of ICL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICML 2025\] On the Power of Context-Enhanced Learning in LLMs](on_the_power_of_context-enhanced_learning_in_llms.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](../../CVPR2025/interpretability/open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[ACL 2026\] Interpreting Style Representations via Style-Eliciting Prompts](../../ACL2026/interpretability/interpreting_style_representations_via_style-eliciting_prompts.md)
- [\[NeurIPS 2025\] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](../../NeurIPS2025/interpretability/reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)

</div>

<!-- RELATED:END -->
