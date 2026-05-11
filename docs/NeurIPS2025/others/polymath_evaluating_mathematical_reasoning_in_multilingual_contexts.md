---
title: >-
  [Paper Note] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning
description: >-
  [NeurIPS 2025][Value model] This paper proposes Value-Guided Search (VGS), which employs a token-level value model to guide block-level beam search without requiring predefined "steps." VGS achieves a +14.5% relative acc…
tags:
  - "NeurIPS 2025"
  - "Value model"
  - "block-level search"
  - "beam search"
  - "VGS"
  - "inference efficiency"
date: 2026-05-08
content_hash: d46f109981d94c19
---

# Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2504.18428](https://arxiv.org/abs/2504.18428)
**Code**: [GitHub](https://github.com/kaiwenw/value-guided-search)
**Area**: Other
**Keywords**: Value model, block-level search, beam search, VGS, inference efficiency

## TL;DR
This paper proposes Value-Guided Search (VGS), which employs a token-level value model to guide block-level beam search without requiring predefined "steps." VGS achieves a +14.5% relative accuracy improvement over majority voting on competition mathematics while reducing inference computation by 30%, outperforming existing PRM-based approaches.

## Background & Motivation
**PRM scalability bottleneck**: Existing PRMs require fine-grained step-level annotations (high cost, ambiguous definitions) and extensive Monte Carlo sampling, making them difficult to scale to long-horizon reasoning.

**Computational efficiency dilemma**: Reasoning models generate long outputs at high cost, necessitating efficient test-time computation strategies to reduce inference FLOPs.

**Advantage of value models**: Value prediction requires only final outcome labels rather than step-level annotations. The question is whether value models can serve as a powerful alternative to PRMs.

**Key design question**: How can token-level value signals be effectively applied within long CoT for block-level search decisions?

## Method

### Overall Architecture
**End-to-end three-stage pipeline**: (1) Value model training → (2) Data collection → (3) Block-level search application

### Key Designs
**1. Regression-classification value model training**:
Value prediction is decomposed into a classification task (incomplete, incorrect, correct):
$$\mathcal{L}(θ,\mathcal{B}) = \frac{1}{|\mathcal{B}|}\sum_{(x,y,z,κ)∈\mathcal{B}}\frac{1}{|z|}\sum_{h=1}^{|z|}\ell_{ce}(f_θ(x,y,z^{:h}),κ)$$

Crucially, the loss is computed at every token position $h$ within the roll-out (super-sampling), fully exploiting the equivalence of autoregressive sampling. The value function is defined as:
$$V_θ(x) := f_θ(x)[1] \text{ and } \mathbb{E}_{z∼π_{ref}}[R(x,z)|x]$$
directly predicting the expected reward of completed trajectories.

**2. Efficient data collection pipeline**:
- **Pre-filtering**: OpenR1-Math 94k → 50k (removing unsolvable and ambiguous problems)
- **Sampling**: 14 roll-ins (truncated thinking prefixes) sampled from each of 4 DeepSeek model scales (1.5B–32B)
- **Completion**: 4 roll-outs sampled from $π_{ref}$ per roll-in, yielding 56 samples per problem in total
- **Post-filtering**: Problems where all roll-outs are unsolvable (~10%) are removed

**Cost comparison**:

| Method | Annotation Granularity | Cost |
|:---:|:---:|:---|
| PRM800K | Human per step | Extremely high |
| Math-Shepherd | MC per step | Very high |
| Qwen2.5-PRM | LLM-as-Judge per step | High |
| VGS (Ours) | Final label only | **Low** |

**3. Block-level beam search**:
Generation is divided into blocks (block_size = 4096 tokens); within each block, candidates are independently sampled and scored by the value model.

**Algorithm 1 – Beam Search**:
```
Initialize B = N/w beams
while incomplete beams exist:
  Sample w blocks from each beam
  Select the highest-value block to continue
end while
```

**Aggregation strategies**:
- **Best-of-n (BoN)**: Select the response with the highest value score
- **Weighted Majority Voting (WMV)**: Group-based majority voting weighted by value scores

Key finding: WMV outperforms BoN, as it benefits from greater stability across diverse generations.

$$\text{WMV}: \arg\max_{p_k}\sum_{y_i∈p_k}w_i, \quad w_i = V(x,y_i)$$

## Key Experimental Results

### AIME-25 & HMMT-25 Competition Mathematics

| Model Configuration | AIME-25 Accuracy | HMMT-25 Accuracy | Average (%) | Relative Gain over MV (%) |
|:---:|:---:|:---:|:---:|:---:|
| **DeepSeek-1.5B** $(N=256$ samples$)$ | | | | |
| Majority Voting (MV) | 38.9±1.9 | 24.3±2.9 | 31.6±1.7 | Baseline |
| VGS (DeepSeek-VM-1.5B) | **46.7±0.7** | **32.8±0.8** | **39.8±0.5** | **+26.0%** |
| VGS (Qwen-PRM-7B) | 38.9±1.4 | 24.2±0.2 | 31.6±0.7 | -0.1% |
| **DeepSeek-7B** $(N=128$ samples$)$ | | | | |
| MV Baseline | 56.5±1.6 | 33.8±2.5 | 45.2±1.5 | - |
| **VGS (DeepSeek-VM-1.5B)** | **59.4±0.8** | **41.1±1.6** | **50.3±0.9** | **+11.3%** |

### Inference Efficiency – FLOPs Required to Reach Target Accuracy

| Target Accuracy | DeepSeek-1.5B MV | DeepSeek-1.5B VGS | FLOP Reduction (%) |
|:---:|:---:|:---:|:---:|
| 35% | 256 generations | 128 generations | 50% |
| 37% | 512 generations | 192 generations | 62.5% |
| 39% | 1024 generations | 256 generations | **75%** |

### Key Findings
1. **Outperforms PRMs**: DeepSeek-VM-1.5B surpasses all 7B PRMs (Qwen/Math-Shepherd) at the same scale, with a 16.2% F1 advantage.
2. **Critical aggregation**: WMV consistently outperforms BoN by 2–4%, highlighting the importance of diverse generation.
3. **Efficiency gains**: Inference FLOPs required to achieve equivalent accuracy are reduced by 50–75%, demonstrating practical deployment feasibility.
4. **Scaling properties**: A single fixed beam width and block size (without per-problem tuning) already yields strong results, simplifying deployment.

## Highlights & Insights
1. **Methodological simplicity**: Avoids the difficulties of fine-grained step definitions required by PRMs; learns end-to-end via token-level classification.
2. **Data efficiency**: Requires no expert annotation or step-wise labeling; only final answers are needed to train a generalizable value model.
3. **Practical engineering**: Data, models, and code are fully open-sourced, lowering barriers to reproduction and application.
4. **Performance–efficiency balance**: Simultaneously improves accuracy (+14%) and reduces computation (FLOPs ↓ 50–75%).

## Limitations & Future Work
1. Evaluation is limited to competition mathematics (AIME/HMMT); applicability to broader scientific reasoning and code generation remains unexplored.
2. Beam width and block size are hyperparameters; while fixed values already perform well, problem-adaptive strategies (e.g., per-instance difficulty estimation) have not been investigated.
3. Comparison with multi-path sampling search (MPS) is insufficient, and the potential of alternative search and aggregation strategies (tree search, graph search) remains unexplored.

## Related Work & Insights
- Reasoning models and test-time compute scaling (o1/DeepSeek-R1/o3)
- Process reward models and search-guided reasoning
- Token-level vs. step-level value prediction

## Rating
⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](faithful_group_shapley_value.md)

</div>

<!-- RELATED:END -->
