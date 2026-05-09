---
title: >-
  [Paper Note] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference
description: >-
  [NeurIPS 2025][Causal Inference][Shapley values] This paper proposes the Estimand-Agnostic (EA) approach and the Frontier-Reducibility Algorithm (FRA) for efficient computation of causal Shapley values (do-SV). By training a single SCM to learn the observational distribution, the framework answers arbitrary identifiable causal queries and reduces the number of coalitions requiring evaluation by approximately 90% via coalition reduction.
tags:
  - NeurIPS 2025
  - Causal Inference
  - Shapley values
  - do-SHAP
  - structural causal models
  - identifiability
date: 2026-05-08
content_hash: 1d709790c06357f8
---

# Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference

**Conference**: NeurIPS 2025
**arXiv**: [2509.20211](https://arxiv.org/abs/2509.20211)
**Code**: To be confirmed
**Area**: Causal Inference / Explainability
**Keywords**: Shapley values, causal inference, do-SHAP, structural causal models, identifiability

## TL;DR
This paper proposes the Estimand-Agnostic (EA) approach and the Frontier-Reducibility Algorithm (FRA) for efficient computation of causal Shapley values (do-SV). By training a single SCM to learn the observational distribution, the framework answers arbitrary identifiable causal queries and reduces the number of coalitions requiring evaluation by approximately 90% via coalition reduction.

## Background & Motivation

**Background**: do-SHAP integrates Shapley values with causal inference by replacing conditional expectations with causal intervention values $\nu(S) = E[Y|\text{do}(X_S = x_S)]$, thereby eliminating spurious correlations inherent in conventional SHAP. However, computing do-SV requires evaluating $2^{|X|}$ distinct causal queries.

**Limitations of Prior Work**: Existing Estimand-Based (EB) methods require manually specifying a causal estimand (e.g., backdoor/frontdoor criterion) for each coalition $S$ and fitting separate models accordingly—a highly impractical requirement. For $K=10$ features, one must handle 1,024 distinct causal queries, each potentially requiring a different estimand.

**Key Challenge**: A fundamental tension exists between the theoretical superiority of do-SV (freedom from spurious correlations) and its computational intractability—the number of coalitions grows exponentially, and each coalition demands independent causal inference.

**Goal**: (a) Eliminate the need to manually specify estimands for each coalition; (b) reduce the number of coalitions requiring explicit computation.

**Key Insight**: Directly learn the data-generating process via an SCM. Once the SCM is fitted, any identifiable causal query can be answered by simulating the do-operator, without deriving estimands in advance. The causal graph structure is further exploited to identify redundant coalitions.

**Core Idea**: Fit a single SCM to the observational distribution for estimand-agnostic causal inference, combined with the Frontier-Reducibility Algorithm for coalition reduction, to make do-SHAP practically viable.

## Method

### Overall Architecture
**Input**: causal graph $G$, observational data → **Step 1**: Train an SCM (architecture options: linear / DCN / DCG / CNF) to fit $\mathcal{P}(V)$ → **Step 2**: Apply the Frontier-Reducibility Algorithm to reduce $2^K$ coalitions to an irreducible subset → **Step 3**: For each irreducible coalition, simulate the do-operator via the SCM to compute $\nu(S)$ and cache results → **Step 4**: Aggregate Shapley values.

### Key Designs

1. **Estimand-Agnostic (EA) Causal Queries**:

    - **Function**: Train a single SCM to answer arbitrary causal queries without deriving per-coalition estimands.
    - **Mechanism**: The SCM learns the generative mechanism $V_i = f_i(\text{Pa}_i, U_i)$ for each variable. Under a do-intervention, the target variables $X_S$ are fixed to their observed values, noise variables $U$ are sampled from their prior, and the remaining variables are generated in topological order. $E[Y|\text{do}(x_S)]$ is estimated via Monte Carlo simulation.
    - **Design Motivation**: EB methods require deriving a separate statistical estimand for each identifiable query, which is infeasible as the number of coalitions grows exponentially. The EA approach requires fitting the SCM only once to answer all queries.

2. **Frontier-Reducibility Algorithm (FRA)**:

    - **Function**: Identify coalitions that yield identical causal effects, thereby eliminating redundant computation.
    - **Mechanism**: For a coalition $S$, the algorithm checks whether a reducible subset $S' \subset S$ exists such that $\nu(S) = \nu(S')$. The reduction condition holds when variables later in the topological order (within $S$) block all paths from earlier variables to $Y$. The "frontier" concept formalizes this: the layer of variables in $S$ closest to $Y$ constitutes the irreducible core.
    - **Design Motivation**: In sparse causal graphs, a large fraction of coalitions are redundant (~90% reducible). FRA incurs minimal computational overhead (approximately 8 seconds for $K=100$) while substantially reducing the number of required causal queries.

3. **Support for Multiple SCM Architectures**:

    - **Function**: Supports linear models, deep causal networks (DCN), deep causal graphs (DCG), and continuous normalizing flows (CNF).
    - **Mechanism**: Linear SCMs are used to validate theoretical correctness; DCN/DCG handle nonlinear settings; CNF accommodates continuous variables and complex distributions.
    - **Design Motivation**: Different data regimes require different SCM expressiveness, necessitating a flexible framework.

### Loss & Training
- SCM training: maximize the likelihood of observational data $\mathcal{P}(V)$.
- Supports both Markovian (no hidden variables) and Semi-Markovian (with hidden variables) causal graphs.
- Semi-Markovian settings require more expressive SCMs to model latent confounders.

## Key Experimental Results

### Main Results

| Dataset | Setting | EA Error | EB Error | FRA Speedup |
|--------|------|-----------|-----------|---------|
| Synthetic (Markovian) | Linear SCM | ~0.01 MSE | — | ~90% coalition reduction |
| Synthetic (Semi-Markovian) | DCN | ~0.03 MSE | ~0.02 MSE | ~85% reduction |
| Adult Income | DCG | Computable | Requires manual derivation | Significant |
| Earthquake | CNF | Computable | — | — |

### Efficiency Comparison

| Method | Time ($K=100$) |
|------|-------------|
| do-SHAP (no caching) | 26m 01s |
| do-SHAP + FRA caching | 21m 14s |
| FRA reduction computation | ~8s |

### Key Findings
- The EA approach achieves accuracy comparable to EB methods in Markovian settings without requiring per-coalition estimand derivation.
- FRA achieves the greatest reduction on sparse graphs (~90%); it provides no benefit when all variables are direct parents of $Y$.
- SCM training error propagates to do-SV estimation, but remains controllable given sufficient data.
- Compared to observational SHAP, do-SHAP correctly identifies spuriously correlated variables (e.g., collider bias).

## Highlights & Insights
- **Elegance of the Estimand-Agnostic paradigm**: The approach entirely bypasses estimand derivation—arguably the most difficult step in causal inference—by fitting a single SCM. This is the only feasible strategy when the number of coalitions grows exponentially.
- **Graph-theoretic ingenuity of FRA**: Exploiting causal graph structure (topological ordering, path blocking) to identify computational redundancy incurs negligible cost while yielding substantial savings.
- **Practical viability of do-SHAP**: The framework bridges the gap between "theoretically correct but computationally intractable" and "deployable on real data."

## Limitations & Future Work
- A known causal graph is required; errors from causal discovery algorithms propagate to the final estimates.
- The EA approach lacks doubly-robust guarantees available to EB estimand-based methods.
- do-SV estimation quality is directly constrained by SCM training quality.
- Combinatorial explosion persists for large-scale settings ($K \gg 100$); FRA provides only a constant-factor improvement.

## Related Work & Insights
- **vs. Observational SHAP**: do-SHAP eliminates spurious correlations at higher computational cost; this work substantially lowers that barrier.
- **vs. Causal SHAP (Heskes)**: Heskes' Asymmetric SHAP incorporates causal structure but does not perform do-interventions.
- **vs. Causal inference literature**: The EA approach bridges the SCM learning community and the explainable AI community.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The EA + FRA combination enables do-SHAP to be practically used for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and real datasets with multiple SCM architectures, though scale remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated and the methodology is systematically described.
- **Value**: ⭐⭐⭐⭐⭐ Advances do-SHAP from a theoretical concept to a practically deployable causal explanation tool.

### Additional Technical Details
- SCM training is validated over 30 random seeds to assess estimation stability.
- Feature importance is computed as $FI_X = \frac{1}{N}\sum_{i\in[N]}|\phi_X^{(i)}|/\sum_{X'\in X}|\phi_{X'}^{(i)}|$.
- Semi-Markovian experiments (with latent variable $U_{X,B}$) further validate the effectiveness of the EA approach.
- Two real-world datasets are used: CDC Diabetes Health Indicators and bike-sharing demand prediction.
- The frontier-checking complexity of FRA caching scales linearly with coalition size rather than exponentially.
- **Value**: ⭐⭐⭐⭐ Provides a practical tool for causal explainable AI.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[NeurIPS 2025\] It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation](its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](performative_validity_of_recourse_explanations.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)

<!-- RELATED:END -->
