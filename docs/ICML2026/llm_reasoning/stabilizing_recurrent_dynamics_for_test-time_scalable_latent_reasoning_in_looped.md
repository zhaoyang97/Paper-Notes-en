---
title: >-
  [Paper Note] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models
description: >-
  [ICML 2026][LLM Reasoning][Looped LM] This paper diagnoses the root cause of the "rise-then-collapse" phenomenon in Looped Language Models (LoopLM) when scaling depth at test-time from a dynamical systems perspective—a "…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Looped LM"
  - "Test-time Scaling"
  - "Jacobian Spectral Radius"
  - "Dynamical System Stability"
  - "Random Loop Sampling"
date: 2026-05-08
content_hash: df5d80e365d7a9d3
---

# Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.26733](https://arxiv.org/abs/2605.26733)  
**Code**: https://github.com/njuyxw/STARS (Available)  
**Area**: LLM Reasoning / Latent Reasoning / Recurrent Transformers  
**Keywords**: Looped LM, Test-time Scaling, Jacobian Spectral Radius, Dynamical System Stability, Random Loop Sampling

## TL;DR
This paper diagnoses the root cause of the "rise-then-collapse" phenomenon in Looped Language Models (LoopLM) when scaling depth at test-time from a dynamical systems perspective—a "stability-effectiveness" dilemma caused by normalization positioning. The authors propose STARS, which utilizes Jacobian Spectral Radius Regularization (JSRR) and random loop sampling to steer latent trajectories toward "asymptotically stable and effective fixed points." On GSM8K, STARS reduces the performance drop at 8-step recursion from 20.47% to 8.26% while improving peak performance by 4.01%.

## Background & Motivation

**Background**: The mainstream path for LLM test-time scaling involves explicitly extending output length (CoT, majority voting, ToT, MCTS), which is limited by the bandwidth and efficiency of natural language sequences. Emerging Looped Language Models (LoopLM, e.g., Huginn, Ouro) take an alternative route: by deeply recursing through a shared set of Transformer blocks, they reflect "thinking" within a continuous latent space. Theoretically, more iterations lead to more refined representations without increasing context length.

**Limitations of Prior Work**: The authors observe that the assumption of "thinking longer yields higher accuracy" does not hold. On GSM8K, Ouro-1.4B's accuracy collapses sharply after reaching a peak at a certain iteration depth; after standard SFT, the peak is higher, but performance drops from 70.46% to 52.97% at step 8. This implies that LoopLMs fail to learn "scalable latent reasoning capabilities" and instead overfit the fixed iteration depth used during training.

**Key Challenge**: Analyzing the recurrent block as a discrete-time map $\mathbf{h}^{(t+1)}=\Phi_\theta(\mathbf{h}^{(t)})$, the authors identify an overlooked fundamental dilemma—**effectiveness and stability are determined by the position of LayerNorm, and the two are mutually exclusive**:
- **Internal Normalization** (Pre-Norm / Pre-Sandwich): Residuals bypass normalization, maintaining an "information highway" (effective), but the update vectors accumulate directly onto the backbone, causing the hidden state norm to explode exponentially and the trajectory to deviate from the data manifold, leading to performance collapse.
- **External Normalization** (Post-Norm / Post-Sandwich): Normalization encloses the residuals, keeping the hidden states bounded (stable), but the reasoning depth remains shallow, leading to poor training performance.

Common remedies—such as non-recurrent Prelude/Coda layers, L2 regularization, and random loop sampling—**all fail to resolve this deadlock simultaneously**.

**Goal**: To enable LoopLMs with truly test-time scalable latent reasoning capabilities: where deeper iterations lead to more convergent latent states and more robust performance.

**Key Insight**: Reasoning is conceptualized as an iterative process of uncertainty reduction. In dynamical terms, this means hidden states should converge to a fixed point that is both "stable" (non-divergent, non-oscillatory) and "effective" (positioned to solve the task). Stability without depth or effectiveness without stability both fail.

**Core Idea**: Leveraging Lyapunov’s linearization theorem—where fixed-point stability is determined by the Jacobian spectral radius—the authors explicitly formulate "asymptotic stability" as a training regularizer $\rho(J) < 1$. This is combined with random loop sampling to generalize the constraint across the entire trajectory, teaching the model to converge to "stable and accurate" fixed points.

## Method

### Overall Architecture

STARS (STAbility-driven Recurrent Scaling) is a training framework applicable to any existing LoopLM (the paper fine-tunes Ouro-1.4B). The overall pipeline is nearly identical to standard LoopLM training: input token sequences pass through a prelude embedding into a recurrent block $\Phi_\theta = \mathcal{M}^L$, iterate $t$ times, and are output via a coda and lmhead. The modification lies in the **loss function**: for each batch, a loop depth $t$ is randomly sampled from a distribution $\mathcal{P}$ (log-normal). Two losses are computed: the standard SFT cross-entropy $\mathcal{L}_{SFT}^{(t)}$ and the Jacobian Spectral Radius Regularization (JSRR) $\mathcal{L}_{JSRR}^{(t)}$, which are combined for backpropagation. Inference remains unchanged with zero additional overhead.

### Key Designs

1.  **Dynamical System Diagnosis + Post-Sandwich Normalization Choice**:
    *   **Function**: Determining which architecture is viable before applying training improvements.
    *   **Mechanism**: Through exhaustive training of 12 normalization structures (LayerNorm/RMSNorm/SimpleNorm × Pre/Post/Pre-Sandwich/Post-Sandwich) on a 4-digit addition task, the authors observe the evolution of latent trajectory "scales" and accuracy via PCA. They conclude that normalization "position" dictates dynamics, while "type" has negligible impact. STARS selects **Post-Sandwich LayerNorm**—naturally bounded and prone to attractor convergence—as the foundation for further stability steering.
    *   **Design Motivation**: Internal norms explode in PCA scale plots; external norms remain compact but fail to maintain accuracy. Since Prelude/Coda, L2, and random sampling alone cannot break the deadlock, the authors choose to build upon external normalization and actively inject stability constraints.

2.  **Jacobian Spectral Radius Regularization (JSRR)**:
    *   **Function**: Explicitly suppressing the Jacobian spectral radius $\rho(J)$ of the map $\Phi_\theta$ at the current hidden state during training to move fixed points toward the "asymptotically stable" side.
    *   **Mechanism**: Stability at a fixed point $\mathbf{h}^\star$ is governed by $\rho(J(\mathbf{h}^\star)) = \max_i |\lambda_i|$, where $\rho<1$ ensures exponential decay of perturbations. Since $J\in\mathbb{R}^{D\times D}$ is too large for explicit eigenvalue computation, the authors use **single-step power iteration + Jacobian-vector product (JVP)**. Using PyTorch’s JVP with a random vector $\mathbf{v}$ yields $J\mathbf{v}$ without constructing $J$, estimating the spectral radius as $\rho(J)\approx \|J\mathbf{v}\|_2$. The loss is defined as $\mathcal{L}_{JSRR}^{(t)} = \frac{1}{N}\sum_i \|J^{(t,i)} \mathbf{v}^{(t,i)}\|_2^2$.
    *   **Design Motivation**: DEQ models (Bai 2019, 2021) use the Frobenius norm $\|J\|_F$, which is an over-restrictive upper bound that hampers model expressivity. Directly targeting the spectral radius is mathematically precise and only constrains the most unstable direction. Single-step iteration is chosen to avoid complex second-order gradient dependencies while remaining statistically valid across batches with minimal overhead.

3.  **Trajectory-level Regularization via Random Loop Sampling × JSRR**:
    *   **Function**: Upgrading stability from a single point $t$ to the global trajectory while preventing the model from overfitting a single training depth.
    *   **Mechanism**: A loop step $t$ is sampled from $\mathcal{P}$ (log-normal $\mu=1.7, \sigma=0.4$, range $[1,16]$), and the model is optimized via $\mathcal{L}_{STARS} = \mathbb{E}_{t\sim\mathcal{P}}[(1-\lambda)\cdot\mathcal{L}_{SFT}^{(t)} + \lambda\cdot\mathcal{L}_{JSRR}^{(t)}]$ ($\lambda=0.1$).
    *   **Design Motivation**: JSRR at a single point does not guarantee stability for deeper iterations. Random sampling alone might延后 degradation but cannot stop state drift in internal norms or ensure "beneficial" attractors in external norms. The combination achieves both bounded and effective trajectories.

### Loss & Training

The final objective is expressed as (Eq. 4):

$$\mathcal{L}_{STARS} = \mathbb{E}_{t\sim\mathcal{P}}\left[(1-\lambda)\cdot\mathcal{L}_{SFT}^{(t)} + \lambda\cdot\mathcal{L}_{JSRR}^{(t)}\right]$$

For math reasoning: Fine-tuned Ouro-1.4B on a 400K NuminaMath subset for 1 epoch using 4×A800, AdamW, cosine schedule, and a starting lr of $1\times10^{-6}$. Random loop sampling used log-normal $\mu=1.7, \sigma=0.4$ in range $[1, 16]$ with $\lambda=0.1$.

## Key Experimental Results

### Main Results (Math Reasoning, Ouro-1.4B Fine-tuned)

| Model | Loop Steps | GSM8K | MATH500 | ASDiv | SVAMP | AMC23 | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ouro-1.4B (base) | 4 | 75.21 | 59.60 | 76.57 | 75.67 | 50.00 | 67.41 |
| Ouro-1.4B (base) | 8 | 58.23 | 40.80 | 70.07 | 66.33 | 40.00 | 55.09 |
| Ouro-1.4B-SFT | 4 | 80.06 | 64.60 | 83.47 | 76.67 | 47.50 | 70.46 |
| Ouro-1.4B-SFT | 8 | 60.05 | 39.20 | 75.10 | 68.00 | 22.50 | 52.97 |
| **Ouro-1.4B-STARS** | 4 | **81.96** | **67.40** | **84.73** | **84.33** | **52.50** | **74.18** |
| **Ouro-1.4B-STARS** | 8 | 74.45 | 54.80 | 82.52 | 81.00 | 35.00 | 65.55 |

Key comparison: The relative drop from peak (step 4) to step 8 on GSM8K is 20.47% for Ouro, 25.0% for SFT, and only 8.26% for STARS. Furthermore, STARS' peak at step 4 (81.96%) outperforms SFT by 1.90%.

### Ablation Study (Average of 4 math benchmarks)

| Configuration | Trend Feature |
| :--- | :--- |
| Ouro-1.4B (base) | Sharp decline after 4 steps |
| + Random Loop only | Slower decline, but significant decay remains |
| + JSRR only | Slower decline, complementary to Random Loop |
| **Full STARS (RL+JSRR)** | Slowest decline and highest peak; both are essential |

### Key Findings
*   **Normalization position is the "Achilles' heel" of LoopLM**: Exhaustive search shows normalization type is irrelevant, but Pre vs. Post determines if the latent space diverges or converges.
*   **Common remedies fail systematically**: Prelude/Coda layers, L2, and pure random sampling cannot achieve both stability and effectiveness simultaneously.
*   **JSRR and Random Loop are perfectly complementary**: JSRR provides local stability, while Random Loop generalizes it globally.

## Highlights & Insights
*   **Re-examining LoopLM as Dynamical Systems**: Translating "thinking longer yields worse results" into "trajectories fail to converge to fixed points" and visualizing them via PCA provides a rigorous diagnostic framework.
*   **Leveraging Control Theory via JSRR**: Applying mature stability conditions from the 1970s to Transformer training using $O(D)$ JVP power iterations is an elegant, low-cost engineering solution reusable for DEQs or Neural ODEs.
*   **"Stable and Accurate Fixed Points" as a Philosophy**: Formalizing reasoning as fixed-point convergence suggests that all latent reasoning methods (Coconut, SIM-CoT, CODI) can be evaluated by their proximity to stable latent trajectories.

## Limitations & Future Work
*   **Proxy Points vs. True Fixed Points**: JSRR constrains the spectral radius at the current state $\mathbf{h}^{(t)}$ rather than the true $\mathbf{h}^\star$.
*   **Scale Limits**: Validation was limited to 1.4B models and loop depths up to 16. The decay at step 8 is mitigated but not eliminated.
*   **Performance Gap**: A 8.62 point gap still exists between step 8 and the peak, indicating that the ultimate goal of "monotonic improvement with depth" is not yet fully achieved.
*   **Future Directions**: Combining JSRR with explicit fixed-point solvers (DEQ-style implicit differentiation) or applying it to sequential latent reasoning like Coconut.

## Related Work & Insights
*   **vs. Huginn / Ouro**: These rely on Prelude/Coda layers, which only slightly mitigate drift. STARS provides an explicit stability constraint.
*   **vs. DEQ**: DEQ uses Frobenius norms which over-constrain the model; STARS uses the spectral radius for higher precision and less interference with expressivity.
*   **vs. Coconut / SIM-CoT**: These expand along the sequence dimension, while LoopLM expands depth. STARS makes the depth-wise route truly scalable.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First to systematically diagnose LoopLM scaling failures from a dynamical perspective and inject differentiable spectral constraints.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive controlled experiments and benchmark testing, though scaling to larger base models is pending.
*   Writing Quality: ⭐⭐⭐⭐⭐ A logical progression from diagnosis to philosophy to method, supported by excellent visualizations.
*   Value: ⭐⭐⭐⭐⭐ Directly applicable to all "recurrent depth" latent reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)

</div>

<!-- RELATED:END -->
