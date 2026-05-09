---
title: >-
  [Paper Note] ComPO: Preference Alignment via Comparison Oracles
description: >-
  [NeurIPS 2025][LLM Evaluation][Preference Alignment] To address likelihood displacement and verbosity caused by noisy preference pairs (where preferred and dispreferred responses are highly similar) in DPO, this paper proposes ComPO, a zeroth-order preference alignment method based on comparison oracles. The approach partitions data into clean and noisy subsets, applying DPO to the clean subset and ComPO to extract alignment signals from the noisy subset, achieving consistent improvements in LC win rate on benchmarks such as AlpacaEval 2.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Preference Alignment
  - DPO
  - Comparison Oracle
  - Likelihood Displacement
  - Zeroth-Order Optimization
date: 2026-05-08
content_hash: a2900718ecdf3799
---

# ComPO: Preference Alignment via Comparison Oracles

**Conference**: NeurIPS 2025  
**arXiv**: [2505.05465](https://arxiv.org/abs/2505.05465)  
**Code**: [https://github.com/PeterLauLukChen/ComparisonPO](https://github.com/PeterLauLukChen/ComparisonPO)  
**Area**: LLM Evaluation  
**Keywords**: Preference Alignment, DPO, Comparison Oracle, Likelihood Displacement, Zeroth-Order Optimization

## TL;DR
To address likelihood displacement and verbosity caused by noisy preference pairs (where preferred and dispreferred responses are highly similar) in DPO, this paper proposes ComPO, a zeroth-order preference alignment method based on comparison oracles. The approach partitions data into clean and noisy subsets, applying DPO to the clean subset and ComPO to extract alignment signals from the noisy subset, achieving consistent improvements in LC win rate on benchmarks such as AlpacaEval 2.

## Background & Motivation

**State of the Field**: Direct preference alignment methods (DPO and its variants) have been widely adopted for LLM alignment due to their simplicity and efficiency. DPO optimizes a policy by maximizing the log-likelihood margin between preferred and dispreferred responses.

**Limitations of Prior Work**: DPO exhibits two critical failure modes: (a) **Likelihood displacement**: the absolute probability of preferred responses decreases during training, with probability mass shifting to semantically opposite responses (e.g., from "No" to "Yes"), potentially causing originally safe models to become unsafe; (b) **Verbosity**: models tend to generate longer responses that are not necessarily better. Both issues are primarily caused by noisy preference pairs—when preferred and dispreferred responses are highly similar, the DPO surrogate objective fails to extract meaningful information.

**Root Cause**: Noisy preference pairs (similar response pairs) do contain useful information, but the log-likelihood margin surrogate objective of DPO cannot effectively extract it. Simply discarding noisy data (e.g., via CHES score filtering) wastes potentially valuable signal.

**Paper Goals**: To design a method that extracts useful alignment signals from noisy preference pairs without relying on an indirect surrogate objective.

**Starting Point**: Drawing on the concept of **comparison oracles** from optimization theory—which require only a binary judgment of "is A better or worse than B?" without needing objective function values or gradients—the paper treats preference pairs as outputs of a comparison oracle and uses the binary comparison signal to directly guide parameter updates.

**Core Idea**: Rather than defining an explicit surrogate objective, ComPO treats preference pairs as comparison oracle outputs and employs zeroth-order optimization to directly guide model parameters toward higher preferred likelihood and lower dispreferred likelihood.

## Method

### Overall Architecture

ComPO follows a three-stage pipeline:
1. A reference model partitions the dataset by log-likelihood margin into a **clean subset** (margin > δ) and a **noisy subset** (margin ≤ δ).
2. Standard DPO (DPO_clean) is applied to the clean subset.
3. Starting from the policy output by DPO_clean, ComPO (Algorithm 2) is applied to the noisy subset.

### Key Designs

1. **Preference Comparison Oracle**:

    - **Function**: Defines a novel parameter comparison mechanism that does not rely on an explicit objective function.
    - **Mechanism**: Given parameters $\theta$ and $\theta'$, the comparison oracle $\mathcal{C}_\pi(\theta, \theta')$ returns $-1$ if and only if $\pi_{\theta'}(\mathbf{y}^+|\mathbf{x}) > \pi_\theta(\mathbf{y}^+|\mathbf{x})$ and $\pi_{\theta'}(\mathbf{y}^-|\mathbf{x}) < \pi_\theta(\mathbf{y}^-|\mathbf{x})$, i.e., $\theta'$ simultaneously achieves higher preferred likelihood and lower dispreferred likelihood.
    - **Design Motivation**: DPO optimizes only the likelihood margin (difference), which may increase the margin while both likelihoods decrease. The comparison oracle requires both likelihoods to move in the correct direction simultaneously, strictly preventing likelihood displacement.

2. **Basic Algorithm (Algorithm 1) + Convergence Guarantee**:

    - **Function**: A foundational zeroth-order optimization framework based on comparison oracles.
    - **Mechanism**: Generate $m$ random perturbations $\mathbf{z}_i$, query the oracle $y_i = \mathcal{C}_\pi(\theta_t, \theta_t + r\mathbf{z}_i)$, recover a normalized sparse gradient estimate via 1-bit compressed sensing $\hat{\mathbf{g}}_t = \arg\max_{\|\mathbf{g}\|_1 \leq \sqrt{s}, \|\mathbf{g}\| \leq 1} \sum_i y_i \mathbf{z}_i^\top \mathbf{g}$, and update $\theta_{t+1} = \theta_t - \eta \hat{\mathbf{g}}_t$.
    - Convergence is guaranteed under non-convex smooth function assumptions, with oracle query complexity $O(\frac{\ell \Delta}{\epsilon^2}(s\log(\frac{2d}{s}) + \log(\frac{\ell\Delta}{\Lambda\epsilon^2})))$, which depends on gradient sparsity $s$ rather than ambient dimension $d$.
    - **Design Motivation**: Although the true alignment objective is unknown (neither evaluable nor differentiable), comparison signals suffice to estimate the normalized gradient direction.

3. **Practical Algorithm (Algorithm 2)**:

    - **Function**: Adapts the theoretical framework to billion-parameter scale.
    - Three key approximations: (a) **Perturb only output layer weights**—reduces the perturbation dimension from $d$ (billions) to $d^o$ (output layer dimension), substantially reducing computation and memory overhead; (b) **Approximate sparse gradient estimation via normalization and clipping**—compute $\hat{\mathbf{g}}^o = \frac{\sum y_i \mathbf{z}_i}{\|\sum y_i \mathbf{z}_i\|}$, then zero out components with magnitude below $\lambda_g$; (c) **Adaptive step size**—scale the step size by the "improvement" ratio $\frac{|\{i: y_i=-1\}|}{m}$, skipping updates when the signal is insufficient.
    - **Design Motivation**: Makes zeroth-order optimization feasible at LLM scale while preserving the core comparison oracle principle.

4. **Data Partitioning Strategy**:

    - Noise criterion: $|\log\pi_{ref}(\mathbf{y}^+|\mathbf{x}) - \log\pi_{ref}(\mathbf{y}^-|\mathbf{x})| \leq \delta$, with $\delta=3$.
    - Although the authors acknowledge that the log-likelihood margin is less precise than the CHES score, it is computationally simple, and experiments demonstrate that ComPO effectively leverages the otherwise discarded noisy data.

### Loss & Training
- Models: Mistral-7B, Llama-3-8B, Gemma-2-9B (both base and instruct variants).
- ComPO hyperparameters: $r=0.0005$–$0.00075$, $m=1600$–$1800$, $\lambda_g=0.00008$–$0.00022$, $\lambda=0.2$.
- Evaluation: AlpacaEval 2, Arena-Hard, MT-Bench.
- Hardware: 30× NVIDIA A40 GPUs.

## Key Experimental Results

### Main Results (Mistral-Instruct-7B)

| Method | AlpacaEval 2 LC% | AlpacaEval 2 WR% | Arena-Hard WR% | MT-Bench Avg |
|--------|------------------|-------------------|-----------------|-------------|
| DPO | 24.14 | 16.71 | 14.4 | 5.86 |
| DPO_clean | 23.89 | 16.15 | 14.2 | 5.73 |
| **DPO_clean + ComPO** | **26.17** | **18.32** | 10.5 | **7.69** |

ComPO consistently improves LC (length-controlled) win rate (+2.03–2.28%), indicating a reduction in verbosity.

### Ablation Study: Likelihood Displacement Mitigation

| Method | Likelihood Displacement Rate | Notes |
|--------|------------------------------|-------|
| DPO (full data) | High | Noisy data causes severe displacement |
| DPO_clean | Reduced | Filtering noise alleviates the issue |
| DPO_clean + ComPO | **Lowest** | ComPO effectively leverages noisy data without inducing displacement |

### Key Findings
- **Noisy data is valuable but requires appropriate treatment**: Directly filtering noisy data (DPO_clean) may slightly hurt performance, while processing noisy data with ComPO yields gains.
- **Consistent LC win rate improvement**: ComPO particularly excels at improving LC relative to raw WR across all models and benchmarks, indicating reduced verbosity.
- **Comparison oracles eliminate the root cause of likelihood displacement**: The oracle requires preferred/dispreferred likelihoods to simultaneously move in the correct direction, preventing the "widening margin with both probabilities decreasing" failure mode.
- Perturbing only output layer weights proves sufficient, substantially reducing the cost of zeroth-order optimization.

## Highlights & Insights
- **Introducing comparison oracles from optimization theory into LLM alignment** is a highly novel cross-domain idea: rather than defining a surrogate objective, it directly uses binary comparison signals from preference pairs for optimization, theoretically circumventing the inaccuracies of DPO's surrogate function.
- **The divide-and-conquer strategy for clean/noisy data** is practically effective: DPO handles easy data while ComPO handles difficult data, with the two methods being complementary.
- **The convergence proof extends to the non-convex setting**, representing a theoretical contribution beyond the original comparison oracle framework (which addressed only the convex case).

## Limitations & Future Work
- The noise/clean partitioning threshold $\delta=3$ is set manually and may require tuning for different datasets.
- The practical algorithm perturbs only the output layer—whether this provides sufficient information for alignment at deeper representation levels remains an open question.
- Only the best results across 5 trials are reported, without mean±std, suggesting potentially high variance.
- On Arena-Hard, ComPO occasionally underperforms DPO (e.g., on Mistral-Instruct), indicating the method does not consistently improve across all dimensions.
- Future work could explore combining ComPO with CHES scores, as CHES-based noise partitioning may be more accurate than log-likelihood margin thresholding.

## Related Work & Insights
- **vs. DPO**: DPO optimizes a log-likelihood margin surrogate; ComPO bypasses the surrogate and uses comparison signals directly. The two are complementary—DPO for clean data, ComPO for noisy data.
- **vs. Razin et al. (CHES score filtering)**: CHES directly discards noisy data, whereas ComPO attempts to mine the value of noisy data.
- **vs. SimPO**: SimPO removes the reference model; ComPO can be further combined with SimPO as a foundation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Introducing comparison oracles into preference alignment is a genuinely new direction with substantial theoretical and methodological contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-benchmark evaluation, but lacking mean±std and experiments on larger models.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, but the gap between the basic algorithm and the practical algorithm is large.
- **Value**: ⭐⭐⭐⭐ Provides a new perspective on handling noisy preference data, with meaningful implications for the RLHF/DPO community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)

<!-- RELATED:END -->
