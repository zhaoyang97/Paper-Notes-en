---
title: >-
  [Paper Note] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects
description: >-
  [AAAI 2026][Robotics][threshold manipulation] This paper proposes the BMTM/HBMTM Bayesian mixture model framework. In scenarios where consumers strategically manipulate spending to reach reward thresholds…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "threshold manipulation"
  - "Bayesian mixture model"
  - "heterogeneous causal effects"
  - "regression discontinuity design"
  - "consumer behavior"
date: 2026-05-08
content_hash: 2262762f1ff9a53a
---

# Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects

**Conference**: AAAI 2026
**arXiv**: [2509.19814](https://arxiv.org/abs/2509.19814)  
**Code**: Not available  
**Area**: Robotics
**Keywords**: threshold manipulation, Bayesian mixture model, heterogeneous causal effects, regression discontinuity design, consumer behavior

## TL;DR

This paper proposes the BMTM/HBMTM Bayesian mixture model framework. In scenarios where consumers strategically manipulate spending to reach reward thresholds, the framework decomposes the observed distribution into bunching and non-bunching sub-distributions to accurately estimate threshold causal effects and heterogeneous treatment effects across subgroups.

## Background & Motivation

**Background**: Marketing strategies such as credit card incentive programs and loyalty programs commonly set spending thresholds to incentivize incremental consumption. Regression Discontinuity Design (RDD) is the standard method for causal inference in such settings and is widely accepted due to its relatively weak and credible identification assumptions.

**Limitations of Prior Work**: When consumers are aware of the threshold and strategically adjust their spending to obtain rewards, the core assumption of RDD—the local randomization (continuity) condition—is violated. Bunching estimation methods in the economics literature assume a "sharp bunching" region of zero density below the threshold, but this rarely holds in marketing contexts where consumers have imprecise control over their spending and bunching is diffuse.

**Key Challenge**: Standard RDD fails in the presence of manipulation; existing bunching methods require an overly strong sharp bunching assumption. Neither approach is applicable to fuzzy manipulation behavior in real marketing settings. Furthermore, practical marketing decisions require subgroup-level heterogeneous causal effects rather than global average effects.

**Goal**: To estimate causal effects under threshold manipulation while stably estimating heterogeneous causal effects across consumer subgroups, even when subgroup sample sizes are small.

**Key Insight**: The problem is reformulated as a density mixture decomposition—observed spending distribution = $\pi \times$ bunching distribution $+ (1-\pi) \times$ non-bunching distribution—using Bayesian inference to distinguish between the two types of consumers.

**Core Idea**: By decomposing the consumer spending distribution into threshold-influenced and uninfluenced components via a Bayesian mixture model, causal effects can be identified even in the presence of manipulation.

## Method

### Overall Architecture

| Step | Data | Estimation Target | Distributional Model |
|------|------|-------------------|----------------------|
| Step 1 | Outside threshold neighborhood $\mathcal{D}_{K^c}$ | Non-bunching distribution $g(\cdot\|\theta)$ | Singh-Maddala distribution |
| Step 2 | Inside threshold neighborhood $\mathcal{D}_K$ | Mixture model $\pi f(\cdot\|\gamma) + (1-\pi) g(\cdot\|\theta)$ | Skew-normal + fixed $g$ |

The average treatment effect on the treated (ATT) is defined as the difference in conditional means between the bunching and non-bunching distributions within the threshold neighborhood:
$$\Delta = \mathbb{E}_f[Y\|Y \in N_K] - \mathbb{E}_g[Y\|Y \in N_K]$$

### Key Designs

**BMTM (Bayesian Modeling of Threshold Manipulation via Mixtures)**

- **Function**: Estimates causal effects for the overall sample.
- **Mechanism**: Step 1 estimates the non-bunching distribution on $\mathcal{D}_{K^c}$ using an adjusted likelihood; Step 2 fixes the estimated parameters $\hat{\theta}$ (posterior mean) and fits a mixture model on $\mathcal{D}_K$, simultaneously inferring the bunching distribution parameters $\gamma$ and the mixing proportion $\pi$.
- **Design Motivation**: The two-step approach avoids the difficulty of joint estimation arising from the different supports of the bunching and non-bunching distributions. The Singh-Maddala distribution flexibly fits right-skewed, heavy-tailed spending data.

**HBMTM (Hierarchical BMTM)**

- **Function**: Estimates heterogeneous causal effects $\Delta_g$ for each of $G$ subgroups.
- **Mechanism**: A random effects structure is introduced: $\theta_g \sim H_\theta(\alpha_\theta)$, $\gamma_g \sim H_\gamma(\alpha_\gamma)$, $\text{logit}(\pi_g) \sim \mathcal{N}(\mu_\pi, \sigma_\pi^2)$, connecting subgroup-level parameters through shared hyperparameters.
- **Design Motivation**: When subgroup sample size $n_g$ is small, the hierarchical structure enables information borrowing (borrowing strength) across groups. The shrinkage effect adapts to sample size—weak shrinkage when $n_g$ is large and strong shrinkage when $n_g$ is small.

**Theoretical Guarantee**

- Theorem 1 establishes a posterior contraction rate of $O(n^{-1/2})$, meaning the posterior distribution concentrates around the true value as sample size grows. This provides a theoretical foundation for information borrowing in the hierarchical model.

### Loss & Training

Posterior inference is implemented via Stan (HMC/NUTS) using CmdStanPy. Four MCMC chains are run, each with 3,000 post-warmup samples (3,000 burn-in discarded), yielding 12,000 effective posterior samples in total. Point estimates (posterior mean) and 90% highest density intervals (HDI) for $\Delta(\Psi)$ are computed from posterior samples.

## Key Experimental Results

### Main Results

100 subgroups, 4 clusters (sample sizes 50, 100, 200, 300 respectively), averaged over 100 Monte Carlo replications:

| Scenario | Method | MAE↓ | CP (ideal 0.90) | AL↓ | IS↓ |
|----------|--------|------|-----------------|-----|-----|
| A (moderate bunching + low heterogeneity) | RDD | 3.31 | — | — | — |
| A | BMTM | 0.78 | 0.91 | 4.03 | 4.65 |
| A | **HBMTM** | **0.33** | 0.84 | 1.20 | **1.85** |
| B (weak bunching + high heterogeneity) | RDD | 3.50 | — | — | — |
| B | BMTM | 1.79 | 0.94 | 9.05 | 9.88 |
| B | **HBMTM** | **0.37** | 0.88 | 1.62 | **2.15** |

HBMTM reduces MAE by approximately 10× compared to RDD.

### Ablation Study

Real marketing data application (spending thresholds of 30,000 / 50,000 / 70,000 JPY, $G = 21$ subgroups):

| Subgroup Type | Threshold | Causal Effect $\Delta$ | Interpretation |
|---------------|-----------|------------------------|----------------|
| Prior-month spending below threshold | $K_1 = 30{,}000$ | Positive (significant) | Consumers increase spending to reach threshold and earn rewards |
| Prior-month spending slightly above threshold | $K_1 = 30{,}000$ | Weakly positive | Mild incremental effect |
| Prior-month spending far above threshold | $K_1 = 30{,}000$ | Negative | Anchoring effect—low threshold suppresses spending of high-value consumers |
| All thresholds | $K_1/K_2/K_3$ | Consistent trend | Above patterns observed consistently across all three thresholds |

### Key Findings

- In Scenario B (weak signal + high heterogeneity), BMTM degrades severely (MAE: 0.78→1.79), whereas HBMTM is largely unaffected (0.33→0.37), demonstrating the robustness of the hierarchical structure.
- HBMTM yields substantially more precise interval estimates (AL reduced from 9.05 to 1.62; IS reduced from 9.88 to 2.15).
- Real marketing data reveals an anchoring effect: for high-spending consumers, setting a relatively low threshold may paradoxically reduce their spending.
- Substantial variation in the Singh-Maddala distribution shape across subgroups validates the necessity of heterogeneous modeling.

## Highlights & Insights

- **Precision of problem formulation**: Recasting the "consumer threshold manipulation" problem from an RDD framework into a mixture distribution decomposition yields a conceptually clear formulation with weaker assumptions.
- **Novelty**: To the best of the authors' knowledge, this is the first method to estimate heterogeneous causal effects under threshold manipulation.
- **Practical value of findings**: The discovery of the anchoring effect directly informs marketing strategy design—improperly set low thresholds may be counterproductive.
- **Theory–experiment balance**: The paper provides both a theoretical proof of posterior contraction (Theorem 1) and extensive validation via simulation and real data.
- **Distributional appropriateness**: The Singh-Maddala distribution offers flexibility for fitting spending data, while the skew-normal captures the asymmetric peak of bunching behavior.

## Limitations & Future Work

- The method relies on parametric assumptions (Singh-Maddala + skew-normal) and may lack flexibility for complex data distributions; an extension to nonparametric Bayesian models (e.g., Dirichlet Process Mixture) is a natural direction.
- The selection of the threshold neighborhood $N_K$ requires prior knowledge or experimentation; in this paper it is fixed at ±10,000 JPY.
- The multi-threshold setting assumes independence across thresholds, whereas in practice consumers may simultaneously strategize around multiple thresholds.
- MCMC inference is computationally slow; large-scale applications would require approximate methods such as variational inference.
- Validation is limited to a single marketing dataset; cross-domain generalizability remains to be examined.

## Related Work & Insights

- **vs. Standard RDD**: Does not require the local randomization assumption; reduces MAE by approximately 10×.
- **vs. Bunching Estimation**: Does not assume sharp bunching (zero-density region), making it applicable to real marketing scenarios where consumers have imprecise spending control.
- **vs. Sugasawa et al. (2023) hierarchical RDD**: First approach to support heterogeneous effect estimation under manipulation.
- **Insights**: Mixture distribution decomposition is a general strategy for handling latent subgroups in observational data and can be applied to other policy evaluation settings (tax policy, educational thresholds, credit approval). The cross-group information borrowing of hierarchical Bayes offers broadly applicable guidance for small-sample problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — The problem formulation and methodological design are precise and highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple scenarios with both simulation and real data.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and conceptual diagrams are intuitive.
- Value: ⭐⭐⭐ — Primarily a statistical methodology contribution; not directly related to the reviewer's work, but the hierarchical Bayes and mixture model ideas are transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](../../NeurIPS2025/robotics/autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[AAAI 2026\] More Than Irrational: Modeling Belief-Biased Agents](more_than_irrational_modeling_belief-biased_agents.md)
- [\[ICML 2026\] From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](../../ICML2026/robotics/from_imagined_futures_to_executable_actions_mixture_of_latent_actions_for_robot_.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[ICML 2026\] Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](../../ICML2026/robotics/embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)

</div>

<!-- RELATED:END -->
