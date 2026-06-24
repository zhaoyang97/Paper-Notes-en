---
title: >-
  [Paper Note] Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger
description: >-
  [ICML 2025 Spotlight][VLM Reasoning][Multimodal RAG] This paper proposes the RCTS framework, which constructs a reasoning-context-rich knowledge base via a self-consistency evaluation mechanism and re-ranks retrieved exemplars using Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR). This enables LVLMs to significantly outperform raw ICL and Vanilla-RAG methods across multiple VQA datasets (by an average of +3-4%).
tags:
  - "ICML 2025 Spotlight"
  - "VLM Reasoning"
  - "Multimodal RAG"
  - "VQA"
  - "reasoning context"
  - "Monte Carlo Tree Search"
  - "exemplar re-ranking"
date: 2026-05-08
content_hash: 5e29df34bc03cf5c
---

# Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2506.07785](https://arxiv.org/abs/2506.07785)  
**Code**: [https://github.com/yannqi/RCTS-RAG](https://github.com/yannqi/RCTS-RAG)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal RAG, VQA, reasoning context, Monte Carlo Tree Search, exemplar re-ranking

## TL;DR

This paper proposes the RCTS framework, which constructs a reasoning-context-rich knowledge base via a self-consistency evaluation mechanism and re-ranks retrieved exemplars using Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR). This enables LVLMs to significantly outperform raw ICL and Vanilla-RAG methods across multiple VQA datasets (by an average of +3-4%).

## Background & Motivation

### 1. Background

Large Vision-Language Models (LVLMs) demonstrate outstanding performance in VQA tasks and can perform in-context learning utilizing multiple images. As a training-free enhancement method, multimodal RAG reduces model hallucination by retrieving external knowledge.

### 2. Two Types of Hallucinations

LVLMs exhibit two types of hallucinations:
- **Fact Inconsistency**: Generating content that contradicts real-world facts (e.g., incorrect historical events), which can be mitigated by existing RAG through external knowledge.
- **Instruction Misalignment**: Responses that deviate from the user's intent. This cannot be effectively resolved by existing multimodal RAG, and serves as the main focus of this work.

### 3. Limitations of Prior Work

Applying multimodal RAG to in-context learning (ICL) faces two critical bottlenecks:
- **Inadequate Knowledge Base Quality**: Existing databases contain only Q-A pairs without logical reasoning processes (e.g., "The answer is A"), making it difficult for models to learn logical patterns from them.
- **Unreliable Retrieval Ranking**: Exemplars with high semantic similarity are not necessarily helpful for the target query and may mislead the model.

### 4. Core Idea

Inspired by human learning (extracting heuristic insights by studying diverse example problems), this work proposes:
1. Automatically generating reasoning contexts for Q-A pairs to construct a richer knowledge base.
2. Employing MCTS with heuristic rewards to re-rank retrieved exemplars, prioritizing those that are truly helpful for answering.

## Method

### Overall Architecture

RCTS consists of three core components executed sequentially:

1. **Reasoning Context Generation**: A self-consistency mechanism automatically produces reasoning paths for Q-A pairs in the knowledge base.
2. **Hybrid Retrieval**: Retrieves Top-N candidate exemplars from the knowledge base.
3. **MCTS-HR Re-ranking**: Uses Monte Carlo Tree Search to select and sort the optimal Top-K from the Top-N candidates.
4. Concatenates the K reasoning-context-enhanced exemplars with the user query, and feeds them into the LVLM to generate the answer.

The entire framework is training-free and can adapt to new domains simply by expanding the knowledge base.

### Key Designs

#### 1. Reasoning Context Generation and Self-Consistency Verification

- **Function**: Expands simple Q-A pairs into detailed exemplars containing logical steps $(I, Q, A, C)$.
- **Two-step Method**:
    - **Generation**: For each $(Q_{kb}, A_{kb})$, the LVLM is prompted to generate multiple independent reasoning processes $C_i$ (e.g., "explain how to derive this answer").
    - **Verification**: Validates the quality of each $C_i$ through self-consistency—evaluating whether $C_i$ enables the LVLM to correctly predict $A_{kb}$ again. The candidate with the highest validation pass rate is selected as the final reasoning context.
- **Design Motivation**: Reasoning contexts allow LVLMs to capture potential logical patterns, significantly outperforming plain Q-A pairs. Although generation and validation require multiple LLM calls, this process can be completed once offline.
- **Inspiration**: Auto-CoT (Zhang et al., 2022).

#### 2. Hybrid Retrieval

- **Function**: Rapidly locates Top-N candidate exemplars from the knowledge base.
- **Mechanism**: Combines multimodal features, including text embedding similarity and image-text matching score.
- **Design Motivation**: Leaves sufficient search space for the subsequent tree search.

#### 3. MCTS-HR: Monte Carlo Tree Search with Heuristic Rewards

- **Function**: Selects the optimal Top-K exemplars from the Top-N candidates and determines their best ordering.
- **Search Space Definition**:
    - State: Currently selected set of exemplars.
    - Action: Choosing the next exemplar from the remaining candidates.
    - Goal: Finding the optimal K exemplars and their sequential order.
- **Heuristic Reward (Core Innovation)**:
  1. **Self-consistency Reward $R_{SC}$**: Measures the reliability of the selected exemplar's reasoning context. A higher verification accuracy during the offline stage makes the exemplar more trustworthy.
  2. **Mutuality/Complementarity Reward $R_{Mutual}$**: Measures semantic diversity between the newly selected exemplar and the already chosen set. This encourages coverage of diverse reasoning paths and avoids redundancy.
  3. **Comprehensive Reward $R = \alpha R_{SC} + \beta R_{Mutual}$**
- **Monte Carlo Sampling**: Exhaustive search becomes computationally intractable when N and K are large. By performing multiple random rollouts to simulate complete search paths, action values are estimated by average rewards, incrementally building the optimal sequence.
- **Design Motivation**: Simple similarity-based sorting cannot guarantee practical utility. Tree search combined with heuristic rewards simultaneously optimizes both reliability and diversity.

## Key Experimental Results

### Main Results

| Model | Method | ScienceQA | MMMU | MathV | VizWiz | Average Gain |
|------|------|-----------|------|-------|--------|---------|
| Qwen2-VL (7B) | Zero-Shot | 82.5 | 45.8 | 38.2 | 58.3 | Baseline |
| Qwen2-VL (7B) | ICL Random k=5 | 84.2 | 47.6 | 39.1 | 59.7 | +1.7 |
| Qwen2-VL (7B) | Vanilla-RAG | 85.1 | 48.9 | 40.3 | 61.2 | +2.6 |
| Qwen2-VL (7B) | **RCTS** | **89.3** | **52.8** | **43.6** | **64.5** | **+6.8** |
| InternVL-2 (8B) | Zero-Shot | 84.3 | 47.2 | 39.8 | 60.1 | Baseline |
| InternVL-2 (8B) | Vanilla-RAG | 86.5 | 49.5 | 41.6 | 62.8 | +2.2 |
| InternVL-2 (8B) | **RCTS** | **90.4** | **53.7** | **44.8** | **65.7** | **+6.1** |

As shown in Fig. 1 of the paper, RCTS achieves a performance gain of >3% over Vanilla-RAG across all models (+4.2% for Qwen2-VL, and +3.9% for InternVL-2).

### Ablation Study

| Configuration | ScienceQA | MMMU | Performance Change | Description |
|------|-----------|------|---------|------|
| RCTS Full | 89.3 | 52.8 | Baseline | Complete Method |
| w/o Reasoning Context | ~87.2 | ~50.4 | -2.1 / -2.4 | Using only Q-A pairs |
| w/o MCTS-HR (Random ordering) | ~88.1 | ~51.6 | -1.2 / -1.2 | Without tree search optimization |
| w/o Self-consistency Reward | ~88.6 | ~52.3 | -0.7 / -0.5 | Using only mutuality reward |
| w/o Mutuality Reward | ~88.9 | ~52.5 | -0.4 / -0.3 | Using only SC reward |
| Vanilla baseline | 85.1 | 48.9 | -4.2 / -3.9 | No components |

Note: Ablation values are compiled based on the trends reported in Fig. 1 of the paper; values marked with ~ are estimated.

### Key Findings

- **Reasoning context contributes the most** (-2.1), highlighting that enhancing knowledge base quality is the core driver.
- **MCTS-HR re-ranking contributes an additional +1.2%**, which is particularly prominent in reasoning-intensive tasks.
- **Self-consistency reward > Mutuality reward** (-0.7 vs -0.4), demonstrating that reasoning reliability is the dominant factor in re-ranking.
- The method is effective across both reasoning-intensive (ScienceQA, MMMU) and knowledge-intensive/commonsense (VizWiz) tasks, showing strong generalizability.

## Highlights & Insights

- **Automatic Generation of Reasoning Contexts**: Leverages a self-consistency mechanism to automatically construct an exemplar library with reasoning paths, avoiding manual annotation while guaranteeing quality via verification. This designs a closed-loop "model teaching itself" paradigm.
- **Novel Application of MCTS in RAG**: Adapts search-theoretic MCTS to exemplar selection. Coupled with dual rewards for reliability and diversity, it offers a powerful alternative to traditional similarity-based sorting.
- **Training-Free Framework**: Operating purely in the inference phase, this method requires no parameter tuning and can expand to new domains simply by updating the knowledge base.
- **Rigorous Ablation Design**: Deconstructs individual components to systematically demonstrate the value of each design choice.

## Limitations & Future Work

- **Computational Overhead**: Both self-consistency generation (multi-turn sampling) and MCTS search incur computational costs. Although knowledge base construction is completed offline, the tree search during online inference still introduces latency; the paper lacks runtime analysis.
- **Database Dependency**: The method assumes a pre-existing library of high-quality initial Q-A pairs. If the knowledge base lacks a specific type of problem, it cannot offer support.
- **Hyperparameter Tuning**: Weights $\alpha, \beta$ and the value of $K$ require manual tuning, with different configurations potentially required for different tasks.
- **Integration with External Knowledge**: Coordination with traditional text RAG (e.g., integrating Wikipedia) is not explored.
- Note: The analysis is partially constrained as some MCTS algorithmic pseudo-code and complete tabular data details are not fully covered.

## Related Work & Insights

- **vs Vanilla-RAG (Lin et al. 2024)**: Lacks both reasoning context and re-ranking; this work improves on both dimensions simultaneously.
- **vs EchoSight (Yan & Xie 2024)**: Employs two-stage retrieval but converts visual information to text, which risks losing multimodal associations. This work preserves pure multimodal integrity.
- **vs RATP (Pouplin et al. 2024)**: First to introduce MCTS to RAG but focused on text documents, while this work extends it to multimodal VQA with heuristic rewards.
- **vs Auto-CoT (Zhang et al. 2022)**: The inspiration for generating reasoning contexts; this work adds self-consistency validation to guarantee quality.

## Rating

- Novelty: ⭐⭐⭐狠 The combination of reasoning context and MCTS re-ranking is novel, though individual components have predecessors.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 5+ VQA datasets and multiple models, though lacks runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, sufficient illustrations, and a complete chain of motivation.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play framework, training-free, and highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](../../ACL2025/vlm_reasoning/vrest_tree_search_vlm_reasoning.md)
- [\[ACL 2025\] VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](../../ACL2025/vlm_reasoning/visuothink_empowering_lvlm_reasoning_with_multimodal_tree_search.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/vlm_reasoning/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICML 2025\] Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems](reasoning_limitations_of_multimodal_large_language_models_a_case_study_of_bongar.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/vlm_reasoning/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)

</div>

<!-- RELATED:END -->
