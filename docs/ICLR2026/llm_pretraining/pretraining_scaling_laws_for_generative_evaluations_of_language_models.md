---
title: >-
  [Paper Note] Pretraining Scaling Laws for Generative Evaluations of Language Models
description: >-
  [ICLR 2026][Pretraining][pass@k] This paper proposes and systematically compares three sets of pretraining scaling laws for "generative evaluation" (tasks with verifiable binary rewards like math problem-solving, scored via pass@k). These laws use **pretraining compute**, **parameters + training tokens**, and **log-likelihood of gold reference solutio
tags:
  - ICLR 2026
  - Pretraining
  - pass@k
  - Pythia
date: 2026-05-08
content_hash: c37f2fc55fe9bd9d
---
# Pretraining Scaling Laws for Generative Evaluations of Language Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Ym33xJYINV](https://openreview.net/forum?id=Ym33xJYINV)  
**Code**: TBD  
**Area**: LLM Pretraining / Scaling Laws  
**Keywords**: Scaling Laws, Generative Evaluation, pass@k, Compute-Optimal, Pythia

## TL;DR
This paper proposes and systematically compares three sets of pretraining scaling laws for "generative evaluation" (tasks with verifiable binary rewards like math problem-solving, scored via pass@k). These laws use **pretraining compute**, **parameters + training tokens**, and **log-likelihood of gold reference solutions** as independent variables to fit and extrapolate pass@k. It reveals that the sampling count $k$ is a new lever for controlling scaling behavior and predictability, discovers that the parameters of the "gold reference likelihood" law are exceptionally stable across nearly five orders of magnitude, and theoretically proves that the compute law is the "compute-optimal envelope" of the parameter + token law.

## Background & Motivation

**Background**: Neural scaling laws (Kaplan, Hoffmann/Chinchilla, etc.) have established the predictable mapping of "compute/parameters/data → performance" as fundamental infrastructure for frontier LLM R&D. However, most existing works only characterize scaling for two types of metrics: **pretraining loss** (cross-entropy) and **discriminative downstream evaluations** (discrete accuracy in multiple-choice questions or QA).

**Limitations of Prior Work**: Many critical capabilities are **generative**—writing proofs, formalizing mathematics, or software engineering. Generative evaluation is fundamentally different from discriminative evaluation: performance is calculated from the model's **open-ended sampling**, introducing new dimensions (sampling temperature, algorithms, number of attempts, and scoring metrics). Scaling laws for such tasks have rarely been systematically characterized (the authors note that among 46 tasks studied by Gadre et al., none were generative).

**Key Challenge**: Discriminative accuracy is a "discrete function of correct/incorrect options," while the commonly used pass@k in generative tasks is a "continuous probability derived from the model's generation distribution." Their mathematical properties differ, meaning existing loss/discriminative scaling laws cannot be directly applied. More importantly, generative evaluation introduces a hyperparameter $k$ (number of attempts per problem) that does not exist in discriminative tasks; does this $k$ alter the scaling laws themselves?

**Goal**: To answer three sub-questions in a focused setting: (1) How to **fit** the changes in pass@k with pretraining resources; (2) How to use cheap models to **extrapolate and predict** the pass@k of the most expensive models; (3) What role $k$ plays in this process. The setting focuses on benchmarks with verifiable binary rewards, multiple attempts per problem, and pass@k scoring (GSM8K, MATH).

**Key Insight**: Borrowing the observation from the GPT-4 technical report (OpenAI 2024) that "negative log pass rate scales as a power law with compute," the authors extend this into a rigorous backtesting framework and explicitly treat $k$ as a primary variable—as $k$ is the simplest and most effective lever for scaling inference-time compute.

**Core Idea**: Instead of proposing a single law, the paper simultaneously compares **three scaling laws with different independent variables** and explicitly parameterizes $k$ into all law parameters. This reveals which law is most stable, how $k$ modifies scaling, and why compute laws are equivalent to parameter + token laws.

## Method

### Overall Architecture

The "method" is an analytical framework of **measurement-fitting-extrapolation-proof** rather than a trainable model. The pipeline is: take intermediate checkpoints from 8 scales of the Pythia family (14M–12B parameters, up to 300B tokens); perform sampling at temperature $\tau=1.0$ on 128 problems each from GSM8K / MATH; calculate the $\text{pass}_B@k$ for each checkpoint using an unbiased estimator; fit $-\log(\text{pass}_B@k)$ against three types of independent variables using power-law regression to obtain three scaling laws; evaluate the extrapolation capability of each law using "backtesting" (similar to cross-validation) to predict the pass@k of the most expensive model (Pythia-12B / 300B tokens, $\approx 2.16\times10^{22}$ FLOPs); and finally, theoretically link the compute law to the parameter + token law.

Pass@k uses the unbiased, low-variance estimator from Chen et al. (2021): for problem $i$, draw $n_i > k$ samples, count $s_i$ successes, and average over all subsets of size $k$:

$$\text{pass}_i@k \;\overset{\text{def}}{=}\; 1 - \frac{\binom{n_i-s_i}{k}}{\binom{n_i}{k}}, \qquad \text{pass}_B@k \;\overset{\text{def}}{=}\; \frac{1}{|B|}\sum_{i\in B}\text{pass}_i@k$$

Pythia was chosen because it is the **only** public model family providing dense sampling of both parameters $N$ and tokens $D$ across multiple orders of compute with public token budgets for every checkpoint. Compute is approximated as $C \approx 6ND$.

### Key Designs

**1. $k$ as a Control Lever for Scaling** 
While $k$ is irrelevant in discriminative evaluation, it directly changes the definition of performance in generative tasks. A key insight is that $k$ should not be a fixed constant but should be **explicitly parameterized** into every scaling law coefficient, written as functions of $k$: $E_0(k), C_0(k), \alpha(k)$. As $k$ increases, the irreducible error $E_0(k)$ decays exponentially and nearly vanishes at $k \approx 100$, causing the law to degenerate into a **pure power law**. Meanwhile, the power-law exponent $\alpha(k)$ steepens (from $\approx 0.12$ at $k=1$ to $\approx 0.38$ at $k=10^4$ on GSM8K). **Conclusion: Larger sampling budgets eliminate performance plateaus and make pass@k scale steeper with compute.**

**2. Three Parallel Scaling Laws with Different Variables**
The authors compare three "resource proxies":
- **Compute Law** (Variable $C$): $-\log(\text{pass}_B@k)(C,k) = E_0(k) + C_0(k)\,C^{-\alpha(k)}$.
- **Parameter + Token Law** (Variables $N,D$): $-\log(\text{pass}_B@k)(N,D,k) = E_0(k) + N_0(k)\,N^{-\beta(k)} + D_0(k)\,D^{-\gamma(k)}$. It fits better across the full range but can have larger relative errors on the largest scale.
- **Gold Reference Likelihood Law** (Variable = Avg Log-Likelihood of Gold Solutions): First calculate $\text{GoldProb}_B \overset{\text{def}}{=} \frac{1}{|B|}\sum_{i\in B} p_\theta(\text{Gold Reference}_i \mid \text{Problem}_i)$, then fit $-\log(\text{pass}_B@k) = \xi_0(k) + K_0(k)\cdot\big[-\log \text{GoldProb}_B\big]^{\kappa(k)}$. It uses an easily computable metric to predict sampling success.

**3. Hyper-stability of the Gold Reference Likelihood Law**
The most counter-intuitive finding is that the parameters $\xi_0(k), K_0(k), \kappa(k)$ **converge to their full-fit values when using models nearly 5 orders of magnitude cheaper than the target**. In contrast, Compute and Parameter+Token laws only stabilize within $\sim 1.5$–$2.5$ orders of magnitude. This makes the Gold Reference law a robust signal for long-range extrapolation.

**4. Compute Law as the "Compute-Optimal Envelope" of Parameter + Token Law**
The authors theoretically demonstrate that the compute law is the result of minimizing the parameter + token law under a fixed compute budget $C \approx cND$. The exponents map as:

$$\alpha(k) = \Big(\tfrac{1}{\beta(k)} + \tfrac{1}{\gamma(k)}\Big)^{-1}, \qquad E_0(k)=E_0(k)$$

Deviations from the optimal $(N^*, D^*)$ introduce a dimensionless **mismatch penalty** $\Phi(r) \ge 1$ (where $r$ is the mismatch ratio). This extends over-training scaling laws to generative evaluations and $k \ge 1$.

### Loss & Training
The paper does not train models. "Fitting" refers to curve regression of the 5 (or 3) parameters of the laws. "Prediction" is defined by a backtesting protocol: fit using checkpoints where $C \le C_{\max}$, extrapolate to $C_{\text{target}}$, and measure the absolute relative error:

$$\text{RelativeError}(k, C_{\max}) = \frac{\big|-\log(\text{pass}_B@k_{\text{target}}) - \hat E_0(k) - \hat C_0(k)\cdot C_{\text{target}}^{-\hat\alpha(k)}\big|}{-\log(\text{pass}_B@k_{\text{target}})}$$

## Key Experimental Results

### Main Results

Setting: 8 Pythia scales, ~5 orders of compute, GSM8K / MATH, $\tau=1.0$ sampling. Target: Pythia-12B/300B tokens.

| Dimension | Compute Law | Parameter + Token Law | Gold Reference Likelihood Law |
| :--- | :--- | :--- | :--- |
| Independent Variable | $C$ | $N,D$ | $-\log\text{GoldProb}_B$ |
| In-distribution Fit | Moderate | Tightest (lowest residual) | Tight, closest to pure power law |
| Stability Required Range| $\sim$1.5–2.5 orders | $\sim$1.5–2.5 orders | **$\sim$5 orders** (much more stable) |
| Prediction Error (Small $k$) | Slightly Higher | Slightly Lower | Slightly Lower |
| Prediction Error (Large $k$) | Equal | Equal | Slightly Higher |

### Ablation Study: Impact of $k$ on GSM8K Compute Law Parameters

| $k$ | Irreducible Error $E_0(k)$ | Compute Exponent $\alpha(k)$ | Note |
| :--- | :--- | :--- | :--- |
| 1 | $\approx 2.0$ | $\approx 0.121$ | Obvious performance plateau |
| 100 | $\approx 0$ | Intermediate | $E_0$ essentially vanished |
| 10000 | $\approx 0$ | $\approx 0.375$ | Degenerates into a steeper pure power law |

### Key Findings
- **$k$ is a true scaling control variable**: Increasing $k$ eliminates irreducible error, steepens the power law, and changes predictability.
- **Stability $\neq$ Fit Tightness $\neq$ Prediction Accuracy**: The Parameter+Token law fits tightest but doesn't extrapolate better than others. The Gold Reference law wins on **parameter convergence**.
- **Irreducible error quantifies benchmark difficulty**: $E_0$ vanishes quickly for GSM8K but remains high for MATH.
- **Compute law lacks independence**: it is an optimized "shadow" of the Parameter+Token law.

## Highlights & Insights
- **Promoting $k$ to a first-class citizen of scaling laws**: $k$ smoothly transforms a "saturated curve with a plateau" into a "steep pure power law."
- **Predicting sampling success via gold solution likelihood**: $\text{GoldProb}_B$ is computable via a single forward pass without sampling, yet provides the most stable long-range prediction signal.
- **Theoretical unification**: The harmonic mapping $\alpha = (1/\beta + 1/\gamma)^{-1}$ links Chinchilla-style allocation, compute power laws, and overtraining penalties into a single logical chain.

## Limitations & Future Work
- **Ours is validated only on the Pythia family**: Necessary for dense checkpoint constraints but generalizability to Llama/Qwen is unverified.
- **Covers only pass@k**: Scaling for partial scores or process rewards remains unknown.
- **Sampling dimensions are frozen**: Temperature $\tau$ is fixed at 1.0; how temperature or top-p interacts with scaling is left for future work.
- **Theoretical gap for Gold Likelihood stability**: The mechanism of why specific path likelihoods correlate so strongly with pass rates remains to be explained.

## Related Work & Insights
- **vs. Kaplan / Hoffmann (Chinchilla)**: They characterize pretraining loss; ours focuses on **generative pass@k** and proves compute laws as envelopes.
- **vs. Discriminative Scaling (Schaeffer, Gadre)**: Unlike discrete accuracy, pass@k is a continuous probability from the distribution.
- **vs. Gadre et al. Overtraining Laws**: Extends overtraining analysis from loss to generative pass@k with $k \ge 1$.
- **vs. GPT-4 Tech Report**: Extends the single-point observation into a rigorous backtesting framework with theoretical proofs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] What Scales in Cross-Entropy Scaling Law?](what_scales_in_cross-entropy_scaling_law.md)
- [\[ICLR 2026\] Soft-Masked Diffusion Language Models](soft-masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
