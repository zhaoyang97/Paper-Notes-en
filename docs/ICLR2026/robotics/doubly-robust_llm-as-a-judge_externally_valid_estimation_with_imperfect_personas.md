---
title: >-
  [Paper Note] Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas
description: >-
  [ICLR2026][Robotics][LLM-as-a-Judge] This paper proposes a doubly-robust estimation framework that combines imperfect LLM persona ratings with human annotations subject to sampling bias, yielding statistically valid estimates of GenAI system quality in the simultaneous presence of covariate shift and selection bias.
tags:
  - ICLR2026
  - Robotics
  - LLM-as-a-Judge
  - Doubly-Robust Estimation
  - External Validity
  - Persona Prompting
  - Evaluation Sampling Bias
date: 2026-05-08
content_hash: 6489f48034d855f9
---

# Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas

**Conference**: ICLR2026  
**arXiv**: [2509.22957](https://arxiv.org/abs/2509.22957)  
**Code**: [lguerdan/doubly-robust-llm-judge](https://github.com/lguerdan/doubly-robust-llm-judge)  
**Area**: Robotics  
**Keywords**: LLM-as-a-Judge, Doubly-Robust Estimation, External Validity, Persona Prompting, Evaluation Sampling Bias  

## TL;DR
This paper proposes a doubly-robust estimation framework that combines imperfect LLM persona ratings with human annotations subject to sampling bias, yielding statistically valid estimates of GenAI system quality in the simultaneous presence of covariate shift and selection bias.

## Background & Motivation
As generative AI systems are deployed at scale, **external validity** of evaluation has become a central concern — specifically, whether laboratory evaluation results generalize to real-world deployment scenarios.

Existing evaluation pipelines face two types of **evaluation sampling bias**:

1. **Covariate shift**: The annotator population used during evaluation (e.g., MTurk crowd workers, skewing younger and more educated) differs in distribution from the target deployment population (e.g., medical chatbot users, skewing older and female).
2. **Selection bias**: Annotators tend to abstain from rating sensitive content (i.e., rating completion depends on annotator/content characteristics), violating the Missing Completely at Random (MCAR) assumption.

Existing statistical frameworks such as PPI++ and RePPI assume i.i.d. sampling from source and target distributions and completely random missingness; when these assumptions are violated, severe coverage failures result. This paper aims to propose an estimation method that produces valid confidence intervals under sampling bias.

## Core Problem
How can one leverage cheap but imperfect LLM persona ratings alongside biased but genuine human annotations to obtain statistically valid estimates of system quality parameters over a target distribution, in the simultaneous presence of covariate shift and selection bias?

## Method

### Problem Formulation
System quality estimation is modeled as a tuple of random variables $Z = (X, V, C, Y, \hat{Y})$:

- $X$: annotator features (age, gender, region, etc.)
- $V$: content to be evaluated (embedding representations of system inputs and outputs)
- $C$: rating completion indicator ($C=1$ denotes a completed rating)
- $Y$: human rating (observable only when $C=1$)
- $\hat{Y}$: LLM persona rating

A source distribution $P_s$ and a target distribution $P_t$ are defined, and the goal is to estimate the quality parameter $\theta_t$ over the target distribution (e.g., mean rating $\mathbb{E}_t[Y]$).

### Limitations of Two Baseline Approaches
1. **Persona-Augmented Regression (PAR)**: Trains a model $\hat{\mu}(W, \hat{Y})$ on source data to predict human ratings, then performs inference on target data. Convergence is slow when the correlation between persona ratings and human ratings is insufficient.
2. **Inverse Propensity Weighting (IPW)**: Re-weights source samples via the density ratio $\omega_0(w)$ and completion probability $\pi_0(w)$. Variance is extremely high in high-dimensional text spaces.

### Doubly-Robust Estimator
The core idea is to combine the regression and re-weighting approaches into a doubly-robust form:

$$\hat{\theta} = \frac{1}{N_t}\sum_{i=1}^{N_t}\hat{\mu}(W_i^t, \hat{Y}_i^t) + \frac{1}{N_s}\sum_{j=1}^{N_s}\hat{\alpha}(W_j^s, C_j^s)\{Y_j^s - \hat{\mu}(W_j^s, \hat{Y}_j^s)\}$$

- **Left term**: Computes the predicted mean using the regression model over target samples, reducing variance by exploiting unlabeled data.
- **Right term**: Corrects residuals via the re-weighting function $\hat{\alpha}$, simultaneously correcting for persona rating bias and sampling bias.

**Double robustness condition**: It suffices that the product of estimation errors of the two nuisance functions decays at a parametric rate:
$$\|\hat{\alpha} - \alpha_0\|_{L^2} \cdot \|\hat{\mu} - \mu_0\|_{L^2} = o_\mathbb{P}(N_t^{-1/2})$$

This implies that the estimator remains valid as long as either $\hat{\mu}$ or $\hat{\alpha}$ is of sufficient quality (each may individually converge at the nonparametric rate $N_t^{-1/4}$).

### Riesz Loss Approach
Traditional methods separately estimate the density ratio $\hat{\omega}$ and completion probability $\hat{\pi}$ and then take their ratio, resulting in high variance in high-dimensional text spaces. This paper instead adopts the Riesz loss to directly learn the ratio $\beta_0(w) = \omega_0(w)/\pi_0(w)$:

$$\beta_0 = \arg\min_\beta \{\mathbb{E}_s[C \cdot \beta(W^s)^2] - 2\mathbb{E}_t[\beta(W^t)]\}$$

Combined with sentence transformer (MiniLM-L6-v2) embeddings and UMAP dimensionality reduction to 15-dimensional content representations, this approach enables effective estimation of the re-weighting function even in high-dimensional text spaces.

### K-Fold Cross-Fitting
$K$-fold cross-fitting is employed to maximize data efficiency: nuisance models are trained on the remaining folds for each fold, debiased estimates are computed on the held-out fold, and results are averaged across folds.

## Key Experimental Results

### Persona Simulation Framework (PSF)
Three experimental settings of increasing realism are proposed:

| Dataset | Type | Rating Task | Scale |
|--------|------|----------|------|
| Fully Synthetic | Fully synthetic | — | Nuisance functions known |
| Semi-Synthetic PRISM | Real dialogues + LLM ratings | Helpfulness (1–100) | 1,000 dialogues × 50 ratings |
| Semi-Synthetic DICES | Real dialogues + human ratings | Harmfulness (1–4) | 300 dialogues × 25 ratings |

### Main Results (averaged over 40 trials)
Performance of DR (Riesz) across three datasets:

- **Coverage**: Synthetic 1.00, PRISM 0.93, DICES 0.86 — substantially outperforming the next best method RePPI (0.56/0.66/0.40).
- **Bias (MAE)**: Synthetic 0.03, PRISM 0.46, DICES 0.02 — lowest across all methods.
- DR (Riesz) achieves valid coverage on PRISM and DICES when persona quality $\rho \geq 0.65$.
- Persona ratings from real LLMs (GPT-5, Claude Sonnet 3.5, etc.) also effectively improve estimation quality.

### Key Findings
1. DR (Riesz) achieves the lowest bias and highest coverage among all baselines.
2. Riesz loss substantially outperforms the conventional approach of separately estimating $\hat{\omega}$ and $\hat{\pi}$, particularly in high-dimensional text spaces.
3. Even moderate correlation between persona and human ratings ($\rho \approx 0.4$) yields improved estimation.

## Highlights & Insights
- **Solid theoretical contributions**: The doubly-robust estimator is extended to an M-estimation framework that simultaneously handles covariate shift and selection bias, supporting not only mean estimation but also richer statistics such as variance and quantiles.
- **Elegant application of Riesz loss**: The need to separately estimate density ratios and propensity scores in high-dimensional spaces is avoided by directly learning the required re-weighting function.
- **Rigorous experimental design**: The PSF framework systematically manipulates persona quality, covariate shift, and selection bias along three dimensions, and is open-sourced for community use.
- **Clear practical relevance**: The paper addresses the real-world problem of insufficient representativeness of annotator populations in current AI safety evaluation.

## Limitations & Future Work
- The framework relies on the **no concept drift** assumption ($P_s(Y|W) = P_t(Y|W)$), i.e., annotators with identical characteristics give the same rating distribution for identical content, which may not hold in practice.
- Content embedding uses MiniLM-L6-v2 with UMAP reduction to 15 dimensions; the impact of information loss on estimation quality warrants further analysis.
- Human annotation scales in the experiments are limited (DICES: only 300 dialogues × 25 ratings); performance at larger scales remains to be validated.
- Persona rating generation still relies on manually designed prompts; sensitivity to prompt design choices is not fully explored.

## Related Work & Insights

| Method | Handles Covariate Shift | Handles Selection Bias | Uses Persona Ratings | Coverage Guarantee |
|------|:-:|:-:|:-:|:-:|
| PPI++ | ✗ | ✗ | ✓ | i.i.d. only |
| RePPI | ✗ | ✗ | ✓ | MCAR only |
| IPW | ✓ | ✓ | ✗ | High variance |
| **DR (Riesz) (Ours)** | **✓** | **✓** | **✓** | **Doubly-robust** |

Compared to PPI++/RePPI, this work relaxes the MCAR assumption. Compared to conventional IPW, the Riesz loss substantially reduces variance in high-dimensional spaces. Compared to pure persona-based evaluation, the framework provides theoretically guaranteed bias correction.

The idea of directly learning density ratios via Riesz loss generalizes to other settings requiring importance weighting (e.g., domain adaptation, off-policy evaluation). The experimental design methodology of the PSF framework — systematically controlling bias magnitude — is worth emulating in other evaluation methodology research. For AI safety evaluation practice, this paper argues that relying solely on crowd-sourced annotators or solely on LLM-as-Judge is insufficient; a principled combination of the two is the way forward.

## Rating
- Novelty: ⭐⭐⭐⭐ — Integrates doubly-robust estimation with LLM persona ratings and formally characterizes evaluation sampling bias.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The PSF framework is elegantly designed, with synthetic and semi-synthetic experiments complementing each other, though the real human annotation scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical development is clear, problem motivation is well-articulated, and experimental visualizations are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a theoretically rigorous bias correction tool for GenAI evaluation with clear practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](on_entropy_control_in_llm-rl_algorithms.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)

</div>

<!-- RELATED:END -->
