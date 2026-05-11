---
title: >-
  [Paper Note] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs
description: >-
  [NeurIPS 2025][LLM/NLP][instruction optimization] This paper proposes PRESTO, a framework that exploits the many-to-one mapping (preimage structure) from soft prompts to instructions in white-box LLMs. Through three comp…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "instruction optimization"
  - "black-box LLM"
  - "soft prompt"
  - "preimage"
  - "Bayesian optimization"
date: 2026-05-08
content_hash: a8d2dd557d83e507
---

# PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.25808](https://arxiv.org/abs/2510.25808)
**Code**: [github.com/mlvlab/PRESTO](https://github.com/mlvlab/PRESTO)
**Area**: LLM/NLP
**Keywords**: instruction optimization, black-box LLM, soft prompt, preimage, Bayesian optimization

## TL;DR

This paper proposes PRESTO, a framework that exploits the many-to-one mapping (preimage structure) from soft prompts to instructions in white-box LLMs. Through three components — score sharing, preimage-based initialization, and score consistency regularization — PRESTO equivalently obtains 14× labeled data under the same query budget, significantly improving instruction optimization efficiency for black-box LLMs.

## Background & Motivation

Black-box LLMs (e.g., GPT-4) are accessed via API with inaccessible internal parameters, making instruction optimization a critical yet challenging problem. Recent approaches leverage white-box LLMs as surrogates: optimizing soft prompts $z$ in continuous space, mapping them to natural language instructions $v = f_w(z)$ via a white-box model $f_w$, and evaluating the instructions with a black-box model $f_b$.

**Core Observation**: White-box LLMs frequently map distinct soft prompts to identical instructions — 10,000 distinct soft prompts yield only approximately 6,500 unique instructions. Prior work (INSTINCT/ZOPO) treats this many-to-one mapping as "redundancy" and an "obstacle," employing filtering or dispersed sampling strategies to circumvent it.

**Core Insight of This Paper**: This many-to-one structure is not an obstacle but valuable prior knowledge. Soft prompts within the same preimage share identical objective function values, constituting a powerful inductive bias that can accelerate optimization.

## Method

### Overall Architecture

PRESTO builds upon the INSTINCT framework, using the last-layer embeddings $g(z)$ of a white-box LLM as features to train a score predictor $m(g(z); \theta)$, and employs a NeuralUCB policy to select the next query point.

**Problem Formulation**:

$$z^* = \arg\max_{z \in \mathcal{Z}} \mathbb{E}_{(x,y) \in D_{\text{val}}} [h(f_b(f_w(z), x), y)]$$

where $h$ is the task scoring function, $f_b$ is the black-box LLM, and $f_w$ is the white-box LLM.

### Key Designs

**Component 1: Preimage-Based Score Sharing**

The preimage of instruction $v$ is defined as the set of all soft prompts that generate it:

$$f_w^{-1}(v) = \{z \in Z \mid f_w(z) = v\}$$

Once an instruction $v$ is evaluated by the black-box model, its score is immediately shared with all soft prompts in its preimage. Effect: under the same query budget, this equivalently yields **14×** labeled training data.

**Component 2: Preimage-Based Initialization**

A coverage score $S_{\text{cov}}$ is designed to select initialization soft prompts, consisting of two terms:

$$S_{\text{cov}}(G_i; G^{\text{init}}, G^{\text{total}}) = S_{\text{size}}(G_i) + S_{\text{rep}}(G_i; G^{\text{init}}, G^{\text{total}})$$

- **Size score** $S_{\text{size}} = |G_i| / \max_j |G_j|$: prioritizes larger preimages to maximize score sharing benefit
- **Representativeness score** $S_{\text{rep}}$: based on MMD (Maximum Mean Discrepancy) to ensure the selected set covers the full search space

$$S_{\text{rep}} = 1 - \frac{\text{MMD}^2(G_i \cup G^{\text{init}}, G^{\text{total}})}{\max_j \text{MMD}^2(G_j \cup G^{\text{init}}, G^{\text{total}})}$$

A greedy algorithm sequentially selects $N_{\text{init}}$ preimages.

**Component 3: Score Consistency Regularization**

For unevaluated preimages, a regularization term enforces consistent score predictor outputs for soft prompts within the same preimage:

$$\mathcal{L}_{\text{cons}} = \mathbb{E}_{v \in V_{\text{unseen}}} \mathbb{E}_{z, z' \in f_w^{-1}(v)} |m(g(z); \theta) - m(g(z'); \theta)|^2$$

The total loss is: $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{cons}}$

where $\gamma$ adopts a linear warmup schedule $\gamma(t) = \gamma_{\max} \cdot \min(1, t/T)$ to prevent premature convergence to incorrect predictions.

### Loss & Training

- White-box LLM: LLaMA3.1-8B-Instruct
- Black-box LLM: GPT-4.1
- Total query budget: 165
- Initialization queries: 40 soft prompts
- All experiments repeated across 3 different random seeds

## Key Experimental Results

### Main Results

**Instruction Induction (subset of 20 tasks)**:

| Method | Best tasks | Avg Rank |
|--------|-----------|----------|
| APE | 1 | 4.25 |
| InstructZero | 0 | 4.80 |
| INSTINCT | 3 | 3.70 |
| EvoPrompt | 3 | 4.70 |
| ZOPO | 4 | 3.05 |
| OPRO | 0 | 5.20 |
| **PRESTO** | **12** | **1.90** |

PRESTO achieves the best performance on 12 of 20 tasks (3× that of ZOPO), with an average rank of 1.90, substantially outperforming all baselines.

**Chain-of-Thought Mathematical Reasoning**:

| Method | GSM8K | AQuA | SVAMP |
|--------|-------|------|-------|
| APE | - | - | - |
| InstructZero | - | - | - |
| INSTINCT | - | - | - |
| PRESTO | Best or second-best (see full tables in the paper) |

### Ablation Study

Stepwise ablation of the three components confirms each one's contribution:
- Score sharing: provides the largest gain, equivalently expanding training data by 14×
- Preimage-based initialization: improves coverage of the initial search space
- Consistency regularization: yields more accurate predictions in unevaluated regions

A toy example visualization clearly demonstrates that a model trained with $\mathcal{L}_{\text{MSE}}$ alone produces inconsistent predictions on unlabeled preimages, while adding $\mathcal{L}_{\text{cons}}$ aligns predictions precisely.

### Key Findings

1. Many-to-one mapping is an advantage rather than an obstacle — the preimage structure provides a powerful inductive bias
2. Within a query budget of 165, PRESTO equivalently utilizes approximately 2,310 labeled data points (14×)
3. On the full set of 30 instruction induction tasks, PRESTO leads by a substantial margin as well
4. Preimage size follows a long-tail distribution: the largest preimage contains 1,000+ soft prompts, while the 100th largest contains approximately 5

## Highlights & Insights

1. **Inversion of perspective**: reinterpreting what prior work considered "redundancy" as "information" is an elegant conceptual shift
2. **Clear theoretical intuition**: the preimage concept is borrowed from mathematics and integrates naturally with the optimization problem
3. **14× data efficiency**: offers substantial practical value in settings where query costs are high (e.g., each GPT-4 call)
4. **Modular design**: the three components are independently effective and can be flexibly integrated into other bandit-based optimization frameworks

## Limitations & Future Work

1. Relies on deterministic decoding assumptions — both white-box and black-box LLMs are assumed deterministic; the preimage structure may be unstable at temperature > 0
2. Requires full preimage precomputation — computationally expensive for large-scale soft prompt sets
3. White-box validation is limited to LLaMA3.1-8B — preimage distributions may vary considerably across different white-box models
4. MMD computation for the coverage score may become a bottleneck when preimages and the candidate set are large
5. Query budget is fixed at 165 — the effect of larger or smaller budgets remains unexplored

## Related Work & Insights

- **InstructZero**: first applies Bayesian optimization over the soft prompt space for instruction search, but does not exploit the preimage structure
- **INSTINCT**: first identifies the many-to-one mapping phenomenon, but only avoids redundancy through dispersed sampling
- **ZOPO**: performs local search via zeroth-order optimization and filters duplicate instructions, discarding valuable preimage information
- **NeuralUCB**: serves as the underlying optimization algorithm; PRESTO's improvements are composable with other bandit algorithms

**Implication**: The preimage idea generalizes to other discrete-continuous mapping optimization problems involving LLM generation (e.g., code generation, structured prediction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reinterpreting many-to-one mappings as prior knowledge is highly creative; the introduction of the preimage concept is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 33 tasks is broad with comparison to 6 baselines, though limited to a single white-box/black-box LLM combination
- Writing Quality: ⭐⭐⭐⭐ Figures are clear, motivation is articulated smoothly, and mathematical derivations are appropriately concise
- Value: ⭐⭐⭐⭐ Directly valuable for automated prompt engineering, though applicability is relatively narrow (requires a white-box surrogate model)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)
- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)
- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)

</div>

<!-- RELATED:END -->
