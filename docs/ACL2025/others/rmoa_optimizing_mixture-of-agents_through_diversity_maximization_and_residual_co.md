---
title: >-
  [Paper Note] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation
description: >-
  [ACL 2025][Mixture-of-Agents] Inspired by the residual learning in ResNet, this paper proposes the RMoA framework. It optimizes multi-agent collaboration architectures through embedding-based greedy diversity selection, residual extraction/aggregation agents, and an adaptive termination mechanism, achieving state-of-the-art (SOTA) performance while reducing computational overhead.
tags:
  - "ACL 2025"
  - "Mixture-of-Agents"
  - "residual connection"
  - "diversity selection"
  - "adaptive termination"
  - "multi-agent collaboration"
date: 2026-05-08
content_hash: bb9c4dbda42eadca
---

# RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation

**Conference**: ACL 2025  
**arXiv**: [2505.24442](https://arxiv.org/abs/2505.24442)  
**Code**: Yes ([https://github.com/mindhunter01/RMoA](https://github.com/mindhunter01/RMoA))  
**Area**: Others  
**Keywords**: Mixture-of-Agents, residual connection, diversity selection, adaptive termination, multi-agent collaboration

## TL;DR

Inspired by the residual learning in ResNet, this paper proposes the RMoA framework. It optimizes multi-agent collaboration architectures through embedding-based greedy diversity selection, residual extraction/aggregation agents, and an adaptive termination mechanism, achieving state-of-the-art (SOTA) performance while reducing computational overhead.

## Background & Motivation

Large Language Model (LLM)-based Multi-Agent Systems (MAS) have demonstrated strong capabilities across various tasks, where the Mixture-of-Agents (MoA) architecture enhances performance through multi-layered parallel processing and aggregation of agents. However, existing MoA and SMoA methods suffer from three core issues:

**High computational overhead**: The responses of all agents at each layer are concatenated and passed to the next layer, leading to a dramatic explosion in token count.

**Information loss**: Golden/critical information is gradually lost during the multi-layer iterative aggregation process, leading to performance degradation in deeper layers.

**Insufficient robustness**: They rely on judge models to evaluate response quality, but the discriminative capability of the judge models themselves is limited.

Although SMoA introduces a judge model to filter responses for reducing token counts, the quality evaluation of the judge model is unreliable. More importantly, the problem of information loss during aggregation remains unsolved as the number of layers increases.

The core insight of the authors is: **the residual connection concept from ResNet can be leveraged** to retain inter-layer difference information and mitigate information loss; **embedding similarity can replace the judge model** to select diverse responses and guarantee informational heterogeneity; and **an adaptive termination mechanism can be introduced** to prevent unnecessary computation.

## Method

### Overall Architecture

RMoA introduces three core improvements based on the standard MoA architecture:
1. **Greedy Diversity Embedding Selection**: Replaces the judge model with an embedding model to greedily select the $K$ most diverse responses from $N$ candidates.
2. **Residual Agent**: Consists of a residual extraction agent and a residual aggregation agent to capture inter-layer difference information.
3. **Adaptive Termination**: Dynamically decides whether to continue iterating based on the convergence of the residuals.

### Key Designs

1. **Greedy Diversity Embedding Selection**

    - **Function**: Selects the $K$ semantically most diverse responses from the $N$ agent responses at the $l$-th layer.
    - **Mechanism**:
        - Convert all responses into vector representations using an embedding model.
        - Construct a cosine similarity matrix $S \in \mathbb{R}^{N \times N}$.
        - Initialization: Select the response with the lowest global average similarity as the first element.
        - Iterative selection: In each round, select the response that has the minimum maximum similarity to the already selected set (min-max strategy).
        - Repeat until $K$ responses are selected.
    - **Design Motivation**: Addresses the "distraction of attention" issue—too many semantically overlapping responses increase the cognitive burden of self-attention. Greedy selection ensures maximum information heterogeneity.

2. **Residual Extraction Agent**

    - **Function**: Captures key differences between the responses of two consecutive layers.
    - **Mechanism**:
        - Concatenate the $K$ candidate responses of the $l$-th layer with the $K$ historical responses of the $(l-1)$-th layer.
        - Utilize a predefined prompt to allow the residual extraction agent to identify meaningful changes.
        - Concatenate the extracted residual $\Delta R_l$ with the responses of the previous layer to serve as the reference input for the next layer.
    - **Design Motivation**: Directly inspired by ResNet—alleviates information degradation in deep iterations by preserving incremental inter-layer information.

3. **Residual Aggregation Agent**

    - **Function**: Integrates the previous layer's reference response and the current layer's residual at the final layer to generate the final output.
    - **Mechanism**: Comprehensively considers both the complete response from the previous layer and the residual changes from the current layer.
    - **Design Motivation**: Ensures that the final output both retains long-term copy/information and integrates short-term improvements.

4. **Adaptive Termination**

    - **Function**: Dynamically decides when to stop the iteration.
    - **Mechanism**: If no meaningful residual is detected for $m$ consecutive layers (the residual is "no change" or "no update"), early termination is triggered.
    - **Design Motivation**: Avoids additional computational overhead and potential hallucination accumulation caused by unnecessary inference layers.

5. **Role-playing Diversity**

    - Each agent is assigned a different role-playing persona to enhance the diversity and creativity of the output.

### Loss & Training

RMoA is a training-free, inference-only framework. Key hyperparameters:
- Number of layers $L$: default is 6 layers
- Diversity selection count $K$: default is 3 (experimentally verified as optimal)
- Number of consecutive non-residual layers for adaptive termination $m$
- Embedding model: BGE-m3 (insensitive to model choice, within $\pm 0.6\%$)

## Key Experimental Results

### Main Results

| Model | Method | AlpacaEval 2.0 | MATH | CRUX | MMLU-r | Average |
|------|------|---------------|------|------|--------|------|
| Qwen2.5-7B | Baseline | 37.94 | 74.94 | 57.31 | 69.90 | 60.02 |
| | +MoA | 31.77 | 75.28 | 56.81 | 62.70 | 56.64↓ |
| | +SMoA | 40.79 | 76.98 | 59.93 | 72.00 | 62.43↑ |
| | **+RMoA** | **41.01** | **77.20** | **61.00** | **71.80** | **62.75↑4.55%** |
| Gemma2-9B | Baseline | 45.15 | 36.64 | 47.50 | 63.90 | 48.30 |
| | **+RMoA** | **45.61** | **50.44** | **50.50** | **66.10** | **53.16↑10.06%** |
| Llama3.1-8B | Baseline | 22.93 | 48.18 | 40.62 | 58.60 | 42.58 |
| | **+RMoA** | **32.86** | **52.10** | **42.65** | **61.63** | **47.41↑11.10%** |
| GPT-4o | Baseline | 55.18 | 76.60 | 75.80 | 83.73 | 72.83 |
| | **+RMoA** | **63.29** | **81.16** | **87.37** | **86.67** | **79.62↑9.32%** |

### Ablation Study & Extension Analysis

| Dimension | Findings |
|---------|------|
| Choice of $K$ ($K=2,3,4,5$) | $K=3$ is the optimal balance point; performance decreases for $K>3$. |
| Adaptive Termination vs. No Termination | AT reduces the hallucination rate from an average of $5.2\%$ to $1.6\%$. |
| Embedding Model | The difference between BGE-m3/SGPT/E5-large is $<0.6\%$; the method is robust to embedding choice. |
| Scaling the Number of Layers (1–6 layers) | Performance steadily improves with the number of layers without degradation. |
| Large Models (72B/DeepSeek) | Qwen2.5-72B: 80.00 $\rightarrow$ 87.80 (+7.8%); DeepSeek-R1: 78.04 $\rightarrow$ 82.92 (+4.88%) |
| Termination Judgment Method | LLM judgment ($80.2\%$) $>$ variance metric ($79.4\%$) $>$ similarity threshold ($78.8\%$) |

### Key Findings

1. RMoA consistently outperforms MoA and SMoA across all models and benchmarks.
2. The original MoA can lead to performance degradation on smaller models (e.g., a $7.2\%$ drop in MMLU for Qwen2.5-7B), whereas RMoA effectively avoids this issue.
3. The residual mechanism effectively mitigates information loss, ensuring that deeper architectures consistently bring performance improvements.
4. Adaptive termination significantly reduces the hallucination rate (from $\sim 5\%$ to $\sim 1.6\%$).
5. The framework can be seamlessly integrated with more powerful backbone models (e.g., DeepSeek-R1).

## Highlights & Insights

- Elegant cross-domain application of ResNet's residual concept: It applies the classic solution for mitigating gradient vanishing and information degradation in deep learning to multi-agent collaboration scenarios.
- Replacing the judge model's evaluation with embedding-based diversity selection is both a simple and effective design choice.
- The adaptive termination mechanism simultaneously addresses both efficiency and hallucination issues.
- The model-agnostic nature of the framework (effective from 7B to GPT-4o) demonstrates excellent generalizability.

## Limitations & Future Work

1. **High computational cost**: Although optimized compared to MoA, the inference cost of multi-layer, multi-agent architectures remains significant.
2. **Residual extraction relies on prompt engineering**: The performance of the residual extraction agent is sensitive to the design of prompts.
3. **Insufficient analysis on the effects of role-playing**: There is a lack of systematic ablation studies on the impact of different role-design strategies.
4. **Lack of direct comparison with methods like self-consistency**.
5. **Limited diversity of experimental scenarios**: Scenarios like open-ended generation and multi-turn dialogue are not explored.

## Related Work & Insights

- MoA and SMoA serve as direct baselines, and RMoA addresses the information loss and efficiency limitations built upon them.
- Residual connections from ResNet inspire cross-modal information preservation strategies.
- This can inspire the incorporation of residual information preservation mechanisms into applications like RAG, multi-model voting, and ensemble methods.
- The embedding-based diversity selection method can be generalized to document selection in retrieval-augmented generation (RAG).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The application of the ResNet residual concept to multi-agent collaboration is novel, and replacing the judge model with embedding-based diversity selection is a valuable innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Extensive coverage with four diverse benchmarks, four model scales, and rich ablation studies and extension analyses.
- **Writing Quality**: ⭐⭐⭐ — Clear overall structure, but some formula notation is dense, hindering readability slightly.
- **Value**: ⭐⭐⭐⭐ — Highly practical, providing a plug-and-play general optimization framework for multi-agent collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](../../ICML2025/others/randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ACL 2025\] Value Residual Learning](value_residual_learning.md)
- [\[ACL 2025\] A New Formulation of Zipf's Meaning-Frequency Law through Contextual Diversity](a_new_formulation_of_zipfs_meaning-frequency_law_through_contextual_diversity.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)

</div>

<!-- RELATED:END -->
