---
title: >-
  [Paper Note] Stochastic Self-Organization in Multi-Agent Systems
description: >-
  [ICLR 2026][LLM Pretraining][multi-agent systems] This paper proposes SelfOrg, a framework that dynamically constructs directed acyclic communication graphs (DAGs) based on semantic similarity of agent responses and Shap…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "multi-agent systems"
  - "self-organization"
  - "Shapley value"
  - "communication graph"
  - "DAG"
  - "LLM collaboration"
date: 2026-05-08
content_hash: 01f5de02cbd8f48f
---

# Stochastic Self-Organization in Multi-Agent Systems

**Conference**: ICLR 2026
**arXiv**: [2510.00685](https://arxiv.org/abs/2510.00685)
**Code**: To be confirmed
**Area**: LLM Pre-training
**Keywords**: multi-agent systems, self-organization, Shapley value, communication graph, DAG, LLM collaboration

## TL;DR

This paper proposes SelfOrg, a framework that dynamically constructs directed acyclic communication graphs (DAGs) based on semantic similarity of agent responses and Shapley value contribution estimates, enabling self-organized collaboration in multi-agent systems. The approach is particularly effective in weak-model settings.

## Background & Motivation

LLM-based multi-agent systems (MAS) theoretically can tackle tasks that individual LLMs cannot handle, yet their collaborative performance is highly sensitive to the communication topology. Core limitations of existing approaches:

**Fixed topologies** (chain, tree, fully-connected graph): cannot adapt to different tasks and instances

**Optimizable topologies** (GPTSwarm, AgentPrune): require policy gradients or mask training, incurring high overhead

**External LLM judges** (DyLAN): introduce additional LLM evaluation costs

**Pre-trained graph generators** (G-Designer, MAS-GPT): require extra training

The paper's key insight is that, because LLMs are inherently stochastic, the same agent may produce entirely different answers across different runs on the same question. Therefore, **communication structure should be determined dynamically based on the current response state**, rather than on task type or the question itself. In weak-model settings in particular, the value of an orchestration system lies in amplifying the rare correct responses while suppressing noise.

## Method

### Overall Architecture

SelfOrg consists of four stages (Algorithm 1):

1. **Decentralized initialization** ($t=0$): each agent independently generates a response $\mathcal{R}_n^{(0)}$
2. **Contribution estimation**: approximate per-agent contributions via Shapley values
3. **Communication graph construction**: form a DAG with high-contribution agents placed upstream
4. **Response propagation and aggregation**: propagate information through the DAG and select the final response closest to the weighted centroid

### Key Designs

**Shapley value approximation**: Exact Shapley values require $2^N$ evaluations, which is infeasible. The paper approximates them via cosine similarity:

$$\phi_n \approx \psi_n := \cos(\mathbf{r}_n, \mathbf{r}_{\text{avg}})$$

where $\mathbf{r}_n = f(\mathcal{R}_n)$ is the response embedding obtained from a lightweight embedding model (e.g., all-MiniLM-L6), and $\mathbf{r}_{\text{avg}}$ is the mean embedding of all responses. Complexity is reduced from exponential to linear.

**Approximation quality guarantee** (Theorem 1): When embedding norms are equal and inner products have a lower bound, the approximation error is bounded above by $I\Gamma^2$. Corollary 1 further guarantees that when the contribution gap between two agents is sufficiently large, the ranking is stable.

**DAG construction rules**:
- Edge $e_{m \to n}$ is activated when: $\cos(\mathbf{r}_n, \mathbf{r}_m) \geq \tau$ and $\psi_m > \psi_n$
- Cycles are detected and eliminated by removing the edge from the weakest to the strongest agent within each cycle
- Topological sorting uses contribution values as tiebreakers

**Response aggregation**: A contribution-weighted centroid is computed as:

$$\mathbf{r}_{\text{centroid}}^{(T)} = \frac{\sum_{n=1}^N \psi_n^{(T)} \mathbf{r}_n^{(T)}}{\sum_{n=1}^N \psi_n^{(T)}}$$

The final response is selected as: $n_\star = \arg\max_{n} \cos(\mathbf{r}_n^{(T)}, \mathbf{r}_{\text{centroid}}^{(T)})$

### Theoretical Analysis

The **correctness amplification mechanism** rests on two key lemmas:

**Lemma 1 (Consistency concentration)**: For two independent agents, the probability that both produce the correct answer, $\Pr[X_c] = p^2$, exceeds the sum of probabilities of any consistent incorrect answer, $\sum_k p_k^2$, provided that incorrect answers are sufficiently dispersed (validated empirically: correct answers recur consistently across 100 runs, whereas incorrect answers are highly scattered).

**Lemma 2 (Contribution dominance)**: Under the assumption that correct-answer embeddings form tight clusters while incorrect-answer embeddings are dispersed (Assumption 1), correct agents strictly achieve higher contribution values than incorrect agents.

## Key Experimental Results

### Main Results

**Weak-model setting (Qwen-2.5-1.5B)**:

| Method | MATH | GSM8K | AQUA | GSM-H | MMLU | MMLU-P | AIME | AVG | AVG-R |
|--------|------|-------|------|-------|------|--------|------|-----|-------|
| Single | 49.20 | 70.40 | 51.18 | 36.20 | 49.60 | 28.80 | 3.33 | 41.24 | 2.57 |
| DyLAN | 49.80 | 67.80 | 51.18 | 27.20 | 50.00 | 15.40 | 3.33 | 37.82 | 4.00 |
| AgentVerse | 45.20 | 69.00 | 50.39 | 27.80 | 38.20 | 24.00 | 0.00 | 36.37 | 4.86 |
| AutoGen | 11.60 | 69.40 | 28.74 | 5.40 | 12.20 | 5.20 | 0.00 | 18.93 | 6.06 |
| **SelfOrg** | **52.40** | **74.60** | **58.27** | **38.00** | **53.80** | **31.60** | **6.67** | **45.05** | **1.00** |

SelfOrg achieves approximately **+4 percentage points** over the strongest single-agent baseline and is the only method that consistently ranks first.

**Strong-model setting (LLaMA-3.3-70B)**:

| Method | MATH | GSM8K | AQUA | MMLU | GPQA | AIME | AVG | AVG-R |
|--------|------|-------|------|------|------|------|-----|-------|
| CoT | 75.00 | 95.80 | 79.92 | 85.20 | 56.70 | 26.67 | 68.46 | 2.50 |
| MacNet | 74.80 | 96.00 | 79.13 | 83.00 | 58.26 | 26.67 | 67.31 | 3.63 |
| **SelfOrg** | **79.80** | **96.60** | **81.10** | 85.00 | **59.82** | **30.00** | **70.19** | **1.25** |

### Ablation Study

**Scalability analysis (Qwen-2.5-X)**:

| Model Scale | AQUA Single | AQUA SelfOrg | Δ | MMLU-P Single | MMLU-P SelfOrg | Δ |
|-------------|-------------|-------------|---|--------------|----------------|---|
| 1.5B | 51.18 | 58.27 | +7.09 | 28.80 | 31.60 | +2.80 |
| 3B | 65.35 | 73.62 | +8.27 | 42.60 | 46.20 | +3.60 |
| 7B | 73.62 | 78.35 | +4.73 | 53.20 | 56.40 | +3.20 |
| 72B | 81.10 | 80.71 | -0.39 | 70.60 | 71.20 | +0.60 |

Gains are largest for weak and moderate models, and nearly vanish at the 72B scale (with a marginal drop on AQUA), consistent with theoretical expectations.

### Key Findings

1. **Weak models benefit most**: All existing MAS baselines collapse on the 1.5B model (some even fall below single-agent performance); SelfOrg is the only method that yields significant improvements.
2. **Heterogeneous agents work effectively**: Mixing four 7B models from Qwen, Falcon, LLaMA, and Mistral, SelfOrg improves the random-selection baseline from 53.94 to 66.14 on AQUA-RAT.
3. **Contribution rankings are meaningful**: Stronger models (Qwen, Falcon) consistently receive high rankings, while the weaker model (Mistral) is demoted.
4. **Two rounds of collaboration are typically sufficient in practice**: the first round explores, the second consolidates.

## Highlights & Insights

1. **Response-conditioned > task-conditioned**: This work challenges the assumption that each task type has a single optimal topology, instead dynamically constructing topology based on current responses — a more principled formulation.
2. **No external judge, no pre-training, no RL**: A lightweight embedding model (6-layer MiniLM) replaces expensive LLM judges, substantially reducing overhead.
3. **Strong alignment between theory and experiments**: The probabilistic model accurately predicts the experimentally observed clustering of correct answers and dispersion of incorrect ones.
4. **Elegant approximation of Shapley values**: Complexity is reduced from exponential to linear while preserving ranking stability guarantees.
5. **Practical value in weak-model settings**: When using cost-efficient smaller models, SelfOrg provides significant and consistent gains.

## Limitations & Future Work

1. Reliance on embedding model quality — if embeddings fail to capture the semantic distinction between correct and incorrect responses, contribution estimation will degrade.
2. Implicit assumption that majority implies correctness — if most agents produce a consistent but incorrect answer, SelfOrg may amplify the error.
3. Gains vanish at the 72B scale, suggesting the method is primarily suited for small-to-medium LLMs.
4. Evaluation is limited to reasoning benchmarks; performance on open-ended generation tasks remains unknown.

## Related Work & Insights

- Compared to GPTSwarm (parameterized topology optimization) and G-Designer (pre-trained graph generators), SelfOrg is a zero-overhead online method.
- The application of Shapley values in federated learning is successfully transferred to the MAS setting.
- Insight: This self-organization mechanism could be combined with Mixture-of-Experts (MoE) to enable dynamic expert routing at inference time.
- The "contribution-dominates-information-flow" principle underlying DAG construction is generalizable to larger-scale agent orchestration systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Response-conditioned topology construction is a significant conceptual contribution, supported by rigorous theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 benchmarks, multiple backbone models (1.5B–72B), heterogeneous settings, and scalability analysis.
- **Practicality**: ⭐⭐⭐⭐ — Lightweight and training-free, though multiple LLM calls are required.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and experiments are comprehensive.
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](../../NeurIPS2025/llm_pretraining/an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)

</div>

<!-- RELATED:END -->
