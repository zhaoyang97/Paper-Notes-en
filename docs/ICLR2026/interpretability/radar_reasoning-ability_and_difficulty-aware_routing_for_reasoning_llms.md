---
title: >-
  [Paper Note] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs
description: >-
  [ICLR 2026][Reasoning Language Models] This paper proposes RADAR, a framework that formulates adaptive inference for reasoning language models (RLMs) as a multi-objective optimization problem. It leverages Item Response Theory (IRT) to jointly estimate interpretable query difficulty and model configuration ability parameters, enabling lightweight and scalable query-level routing. RADAR outperforms state-of-the-art routing methods on 8 reasoning benchmarks while adding only approximately 7ms of latency.
tags:
  - ICLR 2026
  - Reasoning Language Models
  - Model Routing
  - Item Response Theory
  - Multi-Objective Optimization
  - Adaptive Inference
date: 2026-05-08
content_hash: 087e4f7f988bb35e
---

# RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs

**Conference**: ICLR 2026
**arXiv**: [2509.25426](https://arxiv.org/abs/2509.25426)
**Code**: None
**Area**: Interpretability
**Keywords**: Reasoning Language Models, Model Routing, Item Response Theory, Multi-Objective Optimization, Adaptive Inference

## TL;DR

This paper proposes RADAR, a framework that formulates adaptive inference for reasoning language models (RLMs) as a multi-objective optimization problem. It leverages Item Response Theory (IRT) to jointly estimate interpretable query difficulty and model configuration ability parameters, enabling lightweight and scalable query-level routing. RADAR outperforms state-of-the-art routing methods on 8 reasoning benchmarks while adding only approximately 7ms of latency.

## Background & Motivation

Recent RLMs such as DeepSeek-R1, o4-mini, and Qwen3 have demonstrated remarkable capabilities on challenging tasks in mathematics, science, and programming. Selecting an appropriate RLM involves a **performance–cost trade-off** along two key dimensions: (1) model size—larger models yield better performance but incur higher cost; (2) reasoning budget—more thinking tokens improve performance but increase latency and expense.

A key observation is that over 50% of queries on MATH-500 can be correctly answered by Qwen3-0.6B with a minimal reasoning budget, whereas some difficult queries require stronger RLM configurations. Moreover, stronger RLMs may "overthink" simple problems, paradoxically degrading performance. This motivates a central question: how can one select the "just strong enough" RLM configuration for each query to maximize cost-efficiency without sacrificing performance?

## Method

### Overall Architecture

The RADAR framework consists of the following core components:
1. **Discretization**: Each RLM is discretized into multiple configurations $g = (m, u) \in \mathcal{G}$ according to available reasoning budgets.
2. **Multi-Objective Optimization (MOO)**: Routing is formulated as a bi-objective optimization that maximizes performance while minimizing cost.
3. **IRT Calibration**: A 2PL IRT model jointly estimates query difficulty and configuration ability.
4. **Adaptive Testing**: The ability parameter of a new model configuration is rapidly estimated using a small set of dynamically selected queries.

### Key Designs

1. **Discretization and Configuration Routing**: Each RLM $m \in \mathcal{M}$ is discretized by reasoning budget $u \in \mathcal{U}_m$ into configurations $g = (m, u)$. For open-source RLMs, budgets are enforced by counting thinking tokens and appending a truncation message when the budget is exceeded. **Design Motivation**: Unifying model selection and budget selection into a single configuration routing problem enables optimization over the configuration space.

2. **Multi-Objective Optimization Routing**: For each query $q$, the framework solves $g^* = \arg\max_{g \in \mathcal{G}} f(p_q(g), c_q(g))$, where $p_q(g)$ is the predicted performance and $c_q(g)$ is the predicted cost. Two scalarization techniques are explored:

   - **Linear Scalarization**: $\text{LSP}_q^{w_1} = \arg\max_{g \in \mathcal{G}} w_1 p_q(g) - (1-w_1) c_q(g)$
   - **Chebyshev Scalarization**: $\text{CSP}_q^{w_1} = \arg\min_{g \in \mathcal{G}} \max\{w_1|1-p_q(g)|, (1-w_1)c_q(g)\}$

   **Design Motivation**: Chebyshev scalarization can recover non-convex regions of the Pareto frontier that linear scalarization cannot. This is the first work to introduce MOO techniques beyond linear scalarization for LLM routing.

3. **2PL IRT Model**: A two-parameter logistic model is used to parameterize the performance prediction function. To enable OOD generalization, query difficulty $b_j = \mathbf{w}_b^\top \mathbf{e}_j$ and discrimination $a_j = \mathbf{w}_a^\top \mathbf{e}_j$ are parameterized as linear transformations of query embeddings $\mathbf{e}_j$, and each configuration $g_i$ is assigned a scalar ability parameter $\theta_i$. The probability of a correct response is $p_{ij} = \sigma(a_j(\theta_i - b_j))$. **Design Motivation**: Scalar ability values capture an interpretable ordering across model configurations with fewer parameters than multidimensional IRT (MIRT), while generalizing to unseen queries via embeddings.

4. **Adaptive Testing Extension**: When estimating the ability parameter for a new model configuration, queries maximizing Fisher information are iteratively selected: $j_t = \arg\max_{j \in \mathcal{Q} \setminus \mathcal{S}_{t-1}} I(\hat{\theta}_{t-1}, a_j, b_j)$, where $I(\theta, a_j, b_j) = a_j^2 \sigma(a_j(\theta-b_j))[1-\sigma(a_j(\theta-b_j))]$. **Design Motivation**: Only approximately 12% of the training set needs to be evaluated to accurately estimate a new configuration's ability, enabling plug-and-play integration.

### Loss & Training

The IRT model is trained with binary cross-entropy loss:
$$\mathcal{L}_{2PL} = -\frac{1}{nk} \sum_{i=1}^n \sum_{j=1}^k [y_{ij} \log p_{ij} + (1-y_{ij}) \log(1-p_{ij})]$$

where $y_{ij} \in \{0,1\}$ indicates whether configuration $g_i$ correctly answers query $q_j$. In total, 1.75 million binary response records were collected, covering 35 configurations and 50,139 queries.

## Key Experimental Results

### Main Results (In-Distribution Setting, Hypervolume Metric, Higher is Better)

| Benchmark | Random-Pair | RouterBench | IRT-Router | Radar (Ours) | Gain |
|-----------|-------------|-------------|------------|--------------|------|
| GPQA-Diamond | 0.5545 | 0.6866 | 0.6942 | **0.7513** | +8% vs. runner-up |
| MMLU | 0.6905 | 0.8592 | 0.8604 | **0.8720** | +1.3% |
| MMLU-Redux | 0.7281 | 0.9053 | 0.9117 | **0.9230** | +1.2% |
| LSAT | 0.6913 | 0.9125 | 0.9163 | **0.9188** | +0.3% |
| FRAMES | 0.6589 | 0.8325 | 0.8501 | **0.8762** | +3.1% |

### Ablation Study

| Configuration | Hypervolume | Note |
|---------------|-------------|------|
| Linear Scalarization (ID) | Marginally better | Slight advantage in ID setting |
| Chebyshev Scalarization (OOD) | Better | Clear advantage in OOD setting |
| 20% Training Data | ~Comparable | Similar performance with only 20% of data |
| RADAR (35 configs) | Baseline | Original 35 configurations |
| RADAR++ (43 configs) | Improved | Improvement after adding Qwen3-14B via adaptive testing |

### Key Findings

- On MATH-500, RADAR achieves 90% of o4-mini (high budget) performance at only 1.31% of its cost.
- On FRAMES (long-context multi-document QA), RADAR achieves 90% performance at 10% of the cost, whereas the runner-up requires 30%.
- RADAR's routing latency is approximately 7ms, negligible compared to the ~870ms generation time of the smallest RLM configuration.
- Adaptive testing requires only 12% of the training set (~5k queries) to accurately estimate the ability of a new configuration.
- Estimated query difficulty shows moderate Pearson correlation (0.509) with five-level human-annotated difficulty labels on MATH-500.

## Highlights & Insights

- **First to introduce MOO beyond linear scalarization for LLM routing**: Chebyshev scalarization recovers non-convex regions of the Pareto frontier.
- **Psychometrics-inspired IRT model**: Treating queries as exam items and model configurations as examinees yields a natural and interpretable framework.
- **Extreme cost savings**: Achieving 90% performance at 1.31% cost on MATH-500 is a compelling result.
- **Plug-and-play design**: No fine-tuning of RLMs is required; new models can be integrated quickly in a black-box manner.
- **Strong OOD generalization**: Generalization to long-context multi-document QA tasks is particularly notable.

## Limitations & Future Work

- Cost prediction relies on a simple heuristic (average token count × unit price) and does not account for query-specific cost variation.
- Generalization to highly difficult OOD benchmarks such as AIME is limited, with the router tending to assign lower-ability configurations.
- Only text modality is considered; extension to multimodal reasoning scenarios remains unexplored.
- The linear parameterization of 2PL IRT may be insufficient to capture complex difficulty–ability interactions.
- The setting of aggregate budget constraints across batched queries is not addressed.

## Related Work & Insights

- **IRT-Router** (Song et al., 2025): Uses multidimensional IRT (MIRT) with more parameters but non-scalar abilities; RADAR achieves interpretable ordering with scalar ability values.
- **RouterBench** (Hu et al., 2024): Conventional model routing; this work extends it to RLM configuration-level routing.
- **Efficient inference methods (L1/S1, etc.)**: Complementary to RADAR and can be incorporated as additional configurations in the routing pool.
- **Adaptive testing in educational assessment**: The Fisher information-based item selection strategy is successfully adapted from this domain.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of IRT and MOO is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks, 35 configurations, and 1.75 million data points — comprehensive and rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a core challenge in practical RLM deployment with substantial cost savings.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ICLR 2026\] Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_advanced_jai.md)
- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_driven_reasoning.md)

<!-- RELATED:END -->
