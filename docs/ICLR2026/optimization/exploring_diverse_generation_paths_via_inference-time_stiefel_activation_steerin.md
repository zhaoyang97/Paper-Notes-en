---
title: >-
  [Paper Note] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering
description: >-
  [ICLR 2026][Optimization][activation steering] This paper proposes STARS (Stiefel-based Activation Steering for Diverse ReaSoning), a training-free inference-time activation steering method that jointly optimizes $N$ orthogonal steering directions on the Stiefel manifold at each token decoding step, maximizing the geometric volume of hidden states to promote divergent activation trajectories. STARS consistently outperforms temperature sampling in diversity on test case generation (TestEval) and scientific discovery (LiveIdeaBench) with negligible latency overhead and without sacrificing output quality.
tags:
  - ICLR 2026
  - Optimization
  - activation steering
  - Stiefel manifold
  - Riemannian optimization
  - diverse generation
  - inference-time intervention
date: 2026-05-08
content_hash: 5ee4e61ad0a3e2a6
---

# Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering

**Conference**: ICLR 2026
**arXiv**: [2601.22010](https://arxiv.org/abs/2601.22010)
**Code**: [https://github.com/lythk88/STARS](https://github.com/lythk88/STARS)
**Area**: Optimization
**Keywords**: activation steering, Stiefel manifold, Riemannian optimization, diverse generation, inference-time intervention

## TL;DR
This paper proposes STARS (Stiefel-based Activation Steering for Diverse ReaSoning), a training-free inference-time activation steering method that jointly optimizes $N$ orthogonal steering directions on the Stiefel manifold at each token decoding step, maximizing the geometric volume of hidden states to promote divergent activation trajectories. STARS consistently outperforms temperature sampling in diversity on test case generation (TestEval) and scientific discovery (LiveIdeaBench) with negligible latency overhead and without sacrificing output quality.

## Background & Motivation
**State of the Field**: The Best-of-N paradigm—sampling multiple candidate solutions and selecting the best—has been widely adopted in reasoning, coding, and planning tasks. However, its effectiveness is fundamentally limited by the diversity of the candidate pool. When multiple parallel generation paths converge to the same high-probability regions in the latent space, the outputs are merely paraphrases of the same idea, and increasing the sampling budget fails to break the performance ceiling.

**Limitations of Prior Work**:
   - Temperature sampling, nucleus sampling, and beam search perturb the token-level probability distribution only locally, lack coordination across parallel runs, and have no global diversity objective.
   - Training-time approaches (e.g., modifying the objective via reinforcement learning) require a full training pipeline, incur high computational costs, and may not generalize across domains.
   - Existing activation steering techniques are designed for **convergence**—steering a single generation toward a fixed, predefined direction—and are ill-suited for **divergence** objectives.

**Root Cause**: Inference-time diversification must simultaneously satisfy two conflicting requirements: (a) lightweight enough to intervene at each token in real time without introducing significant latency; and (b) powerful enough to induce meaningful divergence in the latent space, rather than merely superficial variation.

**Starting Point**: The paper reframes activation steering from "converging toward a fixed direction" to "pushing multiple paths away from each other"—transforming it from a control tool into an exploration engine. Orthogonal steering directions are jointly optimized on the Stiefel manifold to maximize the geometric volume of the modified hidden states.

## Method

### Overall Architecture
Given a query, the model generates $N$ output sequences in parallel. At each decoding step $\tau$, hidden states $h_{\tau,1}^{(l)}, \ldots, h_{\tau,N}^{(l)} \in \mathbb{R}^d$ are extracted from a designated layer $l$ across all paths. The corresponding steering vectors $v_{\tau,i}^{(l)}$ are computed, and the modified hidden states $h_i + v_i$ are fed into subsequent layers to continue decoding. Paths that reach EOS exit early until all paths are complete.

### Key Designs

1. **Attention Head-Level Steering Intervention**:

    - A steering vector is added to the output of each attention head $j$ at layer $l$: $\text{Attn}_j(x^{(l)}) + v^{(l,j)}$
    - Attention head outputs are chosen over the residual stream because different heads exhibit functional specialization (e.g., syntactic dependencies, coreference resolution), enabling more targeted interventions.
    - Steering vectors across all $M$ heads are concatenated to form $v^{(l)} \in \mathbb{R}^d$, where $d = d_h \times M$.

2. **Volume Maximization + Orthogonality Constraint (Core Optimization Problem)**:

    - Objective: maximize the volume of the parallelepiped spanned by the modified hidden states $\{h_i + v_i\}_{i=1}^N$.
    - Equivalent optimization problem: $\min_{V^\top V = \alpha I} -\log\det((H+V)^\top(H+V))$
    - $V^\top V = \alpha I$ is a scaled Stiefel manifold constraint: (a) orthogonality $v_i^\top v_j = 0$ ensures each path receives a distinct direction; (b) fixed norm $\|v_i\|_2^2 = \alpha$ prevents uncontrolled steering magnitudes from corrupting hidden state information.
    - Setting of $\alpha$: $\alpha = C \cdot \|H\|_2^2$, where $C > 0$ is a manual hyperparameter; experiments use $C \in \{0.1, 0.5\}$.

3. **Full Riemannian Gradient Descent (Algorithm 2, Theoretical Guarantee)**:

    - Riemannian optimization is performed on the Stiefel manifold $\text{St}(d, N, \alpha)$.
    - Each step: (a) compute the Euclidean gradient $\nabla\ell(V_k) = -2(H+V_k)[(H+V_k)^\top(H+V_k)]^{-1}$; (b) project onto the tangent space to obtain the Riemannian gradient; (c) map back to the manifold via polar decomposition retraction; (d) apply Armijo backtracking line search to ensure sufficient descent.
    - **Convergence guarantee (Theorem 1)**: $\min_{0 \le k \le K} \|\text{grad}\ \ell(V_k)\|_F^2 = O(1/K)$
    - **Practical issue**: Each step requires matrix inversion, square roots, and backtracking line search, with complexity $O(N^3)$; multi-step iteration is unacceptable for real-time inference.

4. **Lightweight Single-Step Update with Closed-Form Step Size (Algorithm 3, Core Practical Algorithm)**:

    - **Initialization (Algorithm 1)**: Compute the SVD of $H$: $H = Q\Sigma W^\top$. Randomly select $N$ columns from the last $d-r$ columns of $Q$ (the null space basis of $H$), scaled to $\sqrt{\alpha}$ to obtain $V_0$. Proposition 1 provides a constructive proof that $H + V_0$ is full rank.
    - **Search direction**: Directly set $S = H$ (i.e., use the original activation matrix as the descent direction). Proposition 3 proves that $H$ is a Riemannian descent direction at $V_0$.
    - **Closed-form step size**: A second-order Taylor approximation to the exact line search (Proposition 2) yields $\eta^\star = D_1 / D_2$, where:
        - $D_1 = 2\sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2 + \alpha}$, $D_2 = 4\sum_{i=1}^r \frac{\sigma_i^4}{(\sigma_i^2 + \alpha)^2}$
        - These depend only on the singular values of $H$; since SVD is already computed during initialization, the step size incurs zero additional cost.
    - **Update**: $V_1 = \sqrt{\alpha}(V_0 + \eta^\star H) W (\alpha I + (\eta^\star \Sigma)^2)^{-1/2} W^\top$, computed efficiently by reusing the SVD.
    - **Empirical validation**: Compared with multi-step Algorithm 2, single-step Algorithm 3 achieves approximately 2% suboptimality while using only about 3% of the runtime.

### A Key Insight
The geometric meaning of the Stiefel manifold constraint $V^\top V = \alpha I$: steering vectors are constrained to an "orthonormal frame" in $d$-dimensional space. When $d \gg N$ (e.g., $d = 1536$ for Qwen-2.5-1.5B with $N = 4 \sim 20$), the manifold dimension far exceeds the number of generation paths. The orthogonality constraint is thus easily satisfied while providing each path with an essentially distinct intervention direction.

## Key Experimental Results

### Test Case Generation (TestEval, $N = 20$)

| Model | Temp. | Method | Syntax Correctness | Total Line Cov. | Total Branch Cov. |
|-------|-------|--------|-------------------|-----------------|-------------------|
| Gemma-1.1-2B | 0.2 | Sampling | 95.64% | 1.44% | 1.41% |
| | 0.2 | **STARS_0.5** | 78.93% | **39.03%** | **35.05%** |
| Qwen3-1.7B | 0.2 | Sampling | 8.17% | 4.71% | 4.16% |
| | 0.2 | **STARS_0.5** | 73.40% | **91.35%** | **87.13%** |

- STARS_0.5 consistently and substantially improves coverage (the diversity metric) across all temperature settings.
- On Qwen3-1.7B, STARS simultaneously raises execution correctness from ~3% to ~42%, suggesting that diversified steering helps the model explore higher-quality generation paths.
- The advantage is especially pronounced at low temperature ($T = 0.2$), where standard sampling yields near-zero diversity while STARS still produces high diversity.

### Scientific Discovery (LiveIdeaBench, $N = 4$)

| Model | Temp. | Method | Fluency (Diversity) | Avg. Score |
|-------|-------|--------|---------------------|------------|
| Qwen2.5-3B | 0.2 | Sampling | 2.68 | 5.01 |
| | 0.2 | **STARS_0.5** | **5.09** | **5.59** |
| Llama-3.2-3B | 0.2 | Sampling | 3.27 | 5.27 |
| | 0.2 | **STARS_0.5** | **4.04** | **5.48** |

- Fluency (measuring diversity): at $T = 0.2$, STARS scores nearly twice that of standard sampling (5.09 vs. 2.68).
- Key finding: standard sampling degrades sharply as temperature decreases (5.71 → 5.01), whereas STARS remains stable across all temperatures.
- Quality metrics including Originality, Feasibility, and Clarity are largely unaffected—diversity gains are not purchased at the cost of quality.

### Runtime

| Task | Model | Standard Sampling | Algorithm 3 | Overhead |
|------|-------|-------------------|-------------|----------|
| TestEval | Gemma-1.1-2B | 4.53s | 4.63s | +0.1s |
| TestEval | Qwen3-1.7B | 9.01s | 9.97s | +0.96s |
| LiveIdeaBench | Qwen2.5-3B | 3.02s | 5.01s | +1.99s |
| LiveIdeaBench | Llama-3.2-3B | 4.21s | 4.33s | +0.12s |

- At most ~2 additional seconds per query—negligible overhead for practical deployment.

## Highlights & Insights
- **Elegant problem reformulation**: The paper formalizes the vague objective of "inference-time diversity" as a volume maximization problem on the Stiefel manifold, establishing a rigorous mathematical framework.
- **Principled balance between theory and practicality**: Algorithm 2 provides convergence guarantees but is impractical; Algorithm 3 combines SVD initialization, $H$ as the search direction, and a closed-form step size to trade 2% optimality for a 97% speedup.
- **Using the activation matrix itself as the search direction is an inspired design**: $S = H$ is not only a valid descent direction (Proposition 3) but also incurs zero additional computational cost and yields a step size formula determined entirely by singular values.
- **Training-free**: No modification of model weights, no contrastive samples required; the method intervenes only at inference time and is plug-and-play for any pretrained model.

## Limitations & Future Work
- Syntax correctness decreases under STARS_0.5 (Gemma: 95% → 79%)—strong steering may disrupt generation fluency.
- The hyperparameter $\alpha$ requires tuning ($C = 0.1$ or $0.5$), and the optimal value may differ across tasks and models.
- Experiments are limited to small models (1.5B–3B); performance and overhead on larger models (7B+) remain unreported.
- The choice of which layer to apply steering requires additional experimentation (the paper uses layer 20 for LiveIdeaBench).
- The orthogonality constraint may become overly restrictive when $N$ approaches $d$ (though in practice $d \gg N$).
- Algorithm 3 lacks a theoretical convergence guarantee—it is validated empirically only.

## Related Work & Insights
- **vs. temperature / nucleus sampling**: These methods locally perturb the token-level distribution; STARS globally coordinates multiple paths in the latent space—an upgrade from "stochastic perturbation" to "structured divergence."
- **vs. training-time diversity methods (DPP, RL)**: STARS is a training-free inference-time method that requires no modification to the training pipeline.
- **vs. traditional activation steering**: Traditional methods extract fixed directions from contrastive samples for convergent control; STARS uses orthogonal volume maximization for divergent exploration—a completely opposite design philosophy.
- **Inspiration**: The orthogonality constraint on the Stiefel manifold can be generalized to other settings requiring "structured diversity," such as diversification in ensemble learning and strategy differentiation in multi-agent collaboration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A novel combination of Stiefel manifold and activation steering that transforms steering from a convergence tool into an exploration engine.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two benchmarks with comprehensive multi-model, multi-temperature comparisons, but lacks evaluation on large models.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and complete, progressing systematically from problem formulation to practical algorithm.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play inference-time diversity enhancement tool for the Best-of-N paradigm.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[ICLR 2026\] The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)

<!-- RELATED:END -->
