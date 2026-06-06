---
title: >-
  [Paper Note] Understanding the Role of Training Data in Test-Time Scaling
description: >-
  [ICLR2026][LLM Reasoning][test-time scaling] This paper theoretically analyzes how training data properties affect test-time scaling, proves that CoT reasoning is equivalent to pseudo-Newton method iteration…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "test-time scaling"
  - "chain-of-thought"
  - "in-context learning"
  - "task hardness"
  - "overthinking"
  - "training data selection"
date: 2026-05-08
content_hash: fe65d80569e9d411
---

# Understanding the Role of Training Data in Test-Time Scaling

**Conference**: ICLR2026
**arXiv**: [2510.03605](https://arxiv.org/abs/2510.03605)  
**Code**: N/A  
**Area**: LLM Reasoning
**Keywords**: test-time scaling, chain-of-thought, in-context learning, task hardness, overthinking, training data selection

## TL;DR
This paper theoretically analyzes how training data properties affect test-time scaling, proves that CoT reasoning is equivalent to pseudo-Newton method iteration, proposes a task hardness measure based on the minimum eigenvalue of feature covariance, reveals the mechanism behind the "more thinking is not always better" overthinking phenomenon, and derives an optimal task selection strategy for multi-task training — training sets should be diverse, relevant, and difficult.

## Background & Motivation

**Background**: Test-time scaling (e.g., OpenAI o1, DeepSeek R1) enhances reasoning by allocating more computation at inference time to generate longer CoTs, achieving notable success on mathematical competitions and coding tasks.

**Core Problem**: Despite strong empirical performance, it remains unclear under what conditions training data supports test-time scaling — specifically:
   - Does increasing test-time compute **always** improve downstream reasoning performance?
   - Can increasing test-time compute reduce training-time computational requirements?
   - What constitutes a "difficult" training sample, and why does it benefit test-time scaling?

**Limitations of Prior Work**: Prior studies on training data diversity and difficulty are largely empirical and lack a rigorous theoretical framework explaining the mechanism of test-time scaling.

**Key Insight**: The paper addresses these three questions both theoretically and empirically within the in-context learning framework for linear regression.

## Method

### Theoretical Framework: ICL Weight Prediction + Linear Self-Attention

**Task Setup**: Each prompt $P_\tau = (x_{\tau,1}, y_{\tau,1}, \ldots, x_{\tau,n}, y_{\tau,n})$, where $y_{\tau,i} = \langle w_\tau, x_{\tau,i} \rangle$, $x_{\tau,i} \sim \mathcal{N}(0, \Lambda)$, $w_\tau \sim \mathcal{N}(0, I_d)$. The model estimates the weight vector $w_\tau$ from the prompt.

**Model**: A single-layer linear self-attention (LSA) module, with input embedding matrix containing data and weight estimation slots:

$$E_\tau = \begin{bmatrix} X_\tau & 0 \\ y_\tau & 0 \\ 0_{d \times n} & \hat{w}_0 \\ 0_{1 \times n} & 1 \end{bmatrix}$$

**Training Loss**: Minimizes the mean squared error of weight prediction:

$$L(\theta) = \frac{1}{2} \mathbb{E}\left[\left\| f_{\text{LSA}}(E_\tau;\theta)_{[:,-1]} - (0_d, 0, w_\tau, 1) \right\|^2\right]$$

**Theorem 3.1 (Gradient Descent Converges to Global Optimum)**: Under appropriate initialization, gradient descent with a constant step size converges to the global optimum $V_* = -\Gamma^{-1}/c$, where:

$$\Gamma := \left(1 + \frac{1}{n}\right)\Lambda + \frac{1}{n}\text{tr}(\Lambda) I_d$$

### Test-Time CoT as Pseudo-Newton Method

**Proposition 3.2**: Performing $k$ steps of CoT at test time yields the following recursive weight update:

$$w_{i+1} = w_i - \frac{1}{m} \Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top (w_i - w_{\text{test}})$$

This is equivalent to a **pseudo-Newton method** applied to the quadratic loss $\ell(w) = \frac{1}{2m}\|y_{\text{test}} - X_{\text{test}}^\top w\|^2$, using $\Gamma^{-1}$ as an approximation to the inverse Hessian $\Lambda^{-1}$. After $k$ steps:

$$w_{k+1} = \left(I - \left(I - \frac{1}{m}\Gamma^{-1} X_{\text{test}} X_{\text{test}}^\top\right)^k\right) w_{\text{test}}$$

### Task Hardness Measure

**Theorem 3.3** provides an upper bound on the estimation error of direct ICL (without CoT):

$$\mathbb{E}\|\hat{w} - w_{\text{test}}\|^2 \leq \frac{d}{n^2}\left(1 + \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}\right)^2 + \frac{d}{m}\left(1 + \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}\right)$$

This motivates the following definition of **task hardness**:

$$\text{Hard}(\Lambda) := \frac{\text{tr}(\Lambda)}{\lambda_{\min}(\Lambda)}$$

**Intuition**: Each eigenvector of $\Lambda$ represents a "skill," and the corresponding eigenvalue measures its strength. Easy tasks rely on a few balanced skills (eigenvalues close to each other), while hard tasks depend on many skills with a long-tailed distribution (containing very small eigenvalues).

### Test-Time Scaling Law

**Corollary 3.5**: The estimation error after $k$ steps of CoT is:

$$\mathbb{E}\|w_{k+1} - w_{\text{test}}\|^2 \leq d \left(1 + \frac{n}{1 + \text{Hard}(\Lambda)}\right)^{-2k} (1 + o(1))$$

**Key Implications**:
- For a fixed target error $\varepsilon$, increasing $k$ (more test-time compute) can reduce the required training prompt length $n$
- Harder tasks require longer CoTs
- Computational complexity is $O(kd^2)$

### Overthinking Mechanism

**Remark 4.1**: The effect of test-time thinking is governed by $\text{tr}((I - \Gamma^{-1/2}\Sigma\Gamma^{-1/2})^{2k})$. When certain skill directions of the target task (eigenvectors of $\Sigma$) are **insufficiently covered** in the training data (i.e., $\Gamma$ is weak in those directions), this term increases with $k$ — more thinking **hurts** performance.

### Optimal Task Selection

**Proposition 4.3**: In multi-task training, the optimal task sampling probabilities $\{\pi_\ell\}$ should assign at least 50% of total probability mass to "hard" tasks (those with small $\sigma_{\min}(\Lambda_\ell)$).

The optimal selection can be formulated as an efficiently solvable quadratic optimization problem:

$$\min_{\{\pi_\ell\}} \left\| I - \Sigma^{-1} \sum_{\ell} \Lambda_\ell \pi_\ell \right\|_F^2 \quad \text{s.t.} \sum \pi_\ell = 1, \pi_\ell \geq 0$$

Core conclusion: Training sets should simultaneously exhibit **diversity** (covering all directions of the target task), **relevance** (alignment with the target task's feature distribution), and **difficulty** (inclusion of hard samples with long-tailed skill distributions).

## Key Experimental Results

### LSA Model Validation

| Setting | Conclusion |
|---------|------------|
| Training prompt length $n=10,20,30$ | Increasing $k$ compensates for shorter training context; $n=10$ at $k=20$ reaches the error level of $n=30$ direct prediction |
| Skewed training covariance ($\lambda_i \propto 1/i$) | Under train/test distribution mismatch, test error first decreases then increases as $k$ grows — overthinking emerges |
| Larger $n$ performs worse under overthinking | Opposite to the non-overthinking case — longer training context becomes "more biased" under skewed distributions |

### GPT-2 (9.5M Parameters) Validation

| Experiment | Result |
|-----------|--------|
| Training $n=20,30,40$, varying $k$ | Consistent with LSA trends: longer CoT allows shorter training context to match equivalent performance |
| Skewed covariance + isotropic test | GPT-2 also exhibits overthinking: error increases after approximately $k>10$ |

### Task Selection Experiment

| Task Type | $(\alpha, B)$ | Avg. Sampling Probability |
|-----------|--------------|--------------------------|
| Easy-Short | (0.2, 20) | Lowest |
| Hard-Short | (0.8, 20) | Medium |
| Easy-Long | (0.2, 100) | Medium-low |
| **Hard-Long** | **(0.8, 100)** | **Highest** |

### Real Reasoning Benchmarks (Qwen 2.5-7B)

| Model | CoT Length [0, 1k) | CoT Length [1k, 2k] |
|-------|--------------------|---------------------|
| Qwen-Base | 30.39% | 27.2% |
| Qwen-GCD (aligned training) | **75%** (+44.6) | **38.4%** (+11.2) |
| Qwen-Poly (misaligned training) | 29% (−1.4) | 20.83% (**−6.4**) |

More thinking helps when training data is aligned with the target task, and hurts when it is not — perfectly validating the theoretical predictions.

## Highlights & Insights
- **CoT = Pseudo-Newton Method**: Establishes a precise mathematical correspondence between test-time CoT and an optimization algorithm, offering a novel perspective for understanding the reasoning process.
- **Theoretical Explanation of Overthinking**: Provides the first theoretical account of why more reasoning can sometimes degrade performance — skill directions not covered by training data are amplified through iteration.
- **Spectral Definition of Task Hardness**: $\text{Hard}(\Lambda) = \text{tr}(\Lambda)/\lambda_{\min}(\Lambda)$ is a concise and insightful measure.
- **Substitutability of Training and Test-Time Compute**: Rigorously proves that test-time compute can compensate for insufficient training context length, offering theoretical guidance for practical resource allocation.

## Limitations & Future Work
- **Theory Limited to Linear Models**: The main analysis is confined to linear regression with LSA; extension to nonlinear tasks and deep Transformers requires further work.
- **GPT-2 Experiments Remain on Synthetic Data**: The real reasoning benchmark experiments (Qwen) cover only two specific tasks (GCD and polynomial roots), limiting generalizability.
- **Task Hardness Definition Relies on Covariance Spectrum**: In real NLP tasks, "skills" and "feature distributions" are difficult to measure directly, resulting in a notable gap between theory and practice.
- **RL Training Scenarios Not Considered**: The current analysis is based on SFT/ICL; whether the theory holds under the RL training paradigm of o1/R1-style models remains unknown.

## Related Work & Insights
- **vs. Snell et al. (2024) / Muennighoff et al. (2025)**: Prior work empirically demonstrates test-time scaling effects; this paper provides the theoretical foundation.
- **vs. Su et al. (2025) (overthinking)**: Prior work empirically observes that LLMs overthink on simple problems; this paper provides a mathematical explanation.
- **vs. Huang et al. (2025a)**: Prior analysis is restricted to isotropic features ($\Lambda = I$) and uses CoT during training; this paper extends to general covariance and analyzes purely test-time CoT.
- **Practical Implication**: In real system design, test-time compute should be dynamically adjusted based on training data coverage — allocating more compute to tasks well covered by training data, and moderating compute for tasks with insufficient coverage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic theoretical analysis of the relationship between training data and test-time scaling; the theories on overthinking and task selection are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ LSA/GPT-2 synthetic experiments thoroughly validate the theory; Qwen real-benchmark experiments are a highlight, though broader task coverage would strengthen the work.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear, with well-motivated intuition; the paper progresses logically from single-task to multi-task settings.
- Value: ⭐⭐⭐⭐⭐ Provides important guidance for understanding and improving test-time scaling; the task selection strategy is directly applicable to RL-based reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](../../CVPR2026/llm_reasoning/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](../../ICML2026/llm_reasoning/ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)

</div>

<!-- RELATED:END -->
