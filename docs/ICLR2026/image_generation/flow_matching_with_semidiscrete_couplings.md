---
title: >-
  [Paper Note] Flow Matching with Semidiscrete Couplings
description: >-
  [ICLR 2026][Image Generation][Flow Matching] Replaces the "batch-wise $n \times n$ optimal transport" in OT-guided Flow Matching with a one-time fit of an $N$-dimensional dual potential vector and an online Maximum Inner Product Search (MIPS) to assign noise to data points. By removing the quadratic dependence on batch size $n$ found in OT-FM, this method consist
tags:
  - ICLR 2026
  - Image Generation
  - Flow Matching
  - Optimal Transport
  - Semidiscrete OT
  - Generative Models
  - MIPS
date: 2026-05-08
content_hash: a4960294b9e56a67
---
# Flow Matching with Semidiscrete Couplings

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4EGjzT6w80](https://openreview.net/forum?id=4EGjzT6w80)  
**Code**: [OTT-JAX (ott-jax/ott)](https://github.com/ott-jax/ott)  
**Area**: Image Generation / Flow Matching / Optimal Transport  
**Keywords**: Flow Matching, Optimal Transport, Semidiscrete OT, Generative Models, MIPS  

## TL;DR
Replaces the "batch-wise $n \times n$ optimal transport" in OT-guided Flow Matching with a one-time fit of an $N$-dimensional dual potential vector and an online Maximum Inner Product Search (MIPS) to assign noise to data points. By removing the quadratic dependence on batch size $n$ found in OT-FM, this method consistently outperforms standard FM and OT-FM across multiple datasets, conditional/unconditional tasks, and even single-step mean-flow generation.

## Background & Motivation

**Background**: Flow Matching (FM) trains generative models by sampling noise/data pairs $(x_0, x_1)$ and aligning the velocity field $v_\theta(t, x_t)$ with the direction $x_1 - x_0$, where $x_t = (1-t)x_0 + t x_1$. A key degree of freedom in FM is the coupling between noise and data: the default independent coupling $\pi_I = \mu_0 \otimes \mu_1$ is simple and cheap, but results in high-curvature ODE paths, requiring many Function Evaluations (NFE) during inference to integrate high-quality samples.

**Limitations of Prior Work**: To "straighten" paths, OT-FM (Pooladian 2023, Tong 2024) proposes using Optimal Transport couplings instead of independent couplings—calculating the optimal matching for $n$ noise and $n$ data points via Hungarian/Sinkhorn algorithms per batch. However, OT-FM gains are limited in practice. Zhang et al. (2025) identified the root cause as small $n$: optimal matching on small samples is unstable and fails to approximate the true matching that emerges only as $n \to \infty$ (curse of dimensionality). Their solution was to scale $n$ from 256 to $\approx 10^6$ using multi-GPU Sinkhorn, but this introduces a precomputation cost of $O(n^2/\varepsilon^2)$—higher quality results require larger $n$ and smaller $\varepsilon$, making it "more expensive the more you want to use it."

**Key Challenge**: The theoretical promise of OT-FM (straight paths $\to$ low NFE, high quality) is only fulfilled with massive batch sizes, but the $O(n^2/\varepsilon^2)$ scaling of Sinkhorn makes this approach engineering-prohibitive. The fundamental issue is treating OT as a discrete problem of "repeatedly matching $n$ i.i.d. noise and data samples."

**Goal**: Fulfill the benefits of OT-FM without relying on batch size $n$.

**Core Idea**: By noting that the training dataset consists of a finite $N$ points, the OT problem can be framed as **Semidiscrete Optimal Transport (SD-OT)** between continuous noise and discrete data. This requires fitting a dual potential vector $g$ of size $N$ using SGD (one-time precomputation). During training, each newly sampled noise point is assigned to a data point using a **Maximum Inner Product Search (MIPS)** with cost $O(N)$. This completely eliminates the quadratic dependence on $n/\varepsilon$ found in OT-FM.

## Method

### Overall Architecture
SD-FM consists of two steps: the **Precomputation Stage** solves the SD-OT between the noise distribution $\mu_0$ and the finite data distribution $\mu_1 = \sum_j b_j \delta_{y_j}$, fitting and storing the dual potential $g^\star \in \mathbb{R}^N$ via SGD; the **FM Training Stage** pairs each new noise sample $x_0$ with a data point $x_1^{(k)}$ according to $k \sim s_{\varepsilon, g^\star}(x_0)$, while keeping the rest of the FM process unchanged.

```mermaid
flowchart LR
    subgraph Precomputation["Precomputation (One-time, K SGD steps)"]
        A[Noise μ0 + Data μ1] --> B["Solve Semidiscrete OT<br/>Fit dual potential g* ∈ R^N"]
        B --> C["χ² Convergence Criterion<br/>Monitor m(g)≈b"]
    end
    subgraph Training["FM Training Loop"]
        D[Sample noise x0~N] --> E["MIPS: k*=argmax g*_k + ⟨x0, x1^k⟩"]
        E --> F[Pair (x0, x1^k*)]
        F --> G["FM Regression Loss<br/>‖x1-x0 - vθ(t,xt)‖²"]
    end
    C -.Store g*.-> E
```

### Key Designs

**1. Semidiscrete Dual + SGD Fitting: Replacing Batch-OT with One-time Precomputation**. SD-OT leverages the finiteness of the target measure to express entropy-regularized OT as a concave semi-dual problem parameterized by an $N$-dimensional potential vector $g$: $\max_{g} F_\varepsilon(g) = \mathbb{E}_{X\sim\mu}[f_{g,\varepsilon}(X)] + \langle b, g\rangle$, where the soft c-transform reduces to $f_{g,\varepsilon}(x) = -\max_j [g_j - c(x, y_j)]$ when $\varepsilon=0$. Following Genevay et al. (2016), $g^\star$ is solved using averaged SGD. This compresses the repeated $n \times n$ matching overhead into a **one-time** precomputation (converging in $\approx 12$ hours on 8xH100) that depends only on the dataset itself. Complexity comparison (Table 1): OT-FM adds $O(dn/\varepsilon^2)$ to each training pair, while SD-FM adds only the $dN$ lookup cost, with $\Theta$ (the FM loss gradient itself) dominating the total cost.

**2. Unbiased $\chi^2$ Convergence Criterion: Enabling Monitoring of Large-scale SD-OT**. Existing SD-OT literature lacks convergence criteria—$g$ is the optimal solution if and only if the second marginal $m(g) = b$. Directly monitoring the TV distance $\frac{1}{2}\|m(g)-b\|_1$ is biased because the expectation in $m(g)$ is inside the norm; debiasing would require a sample size scaling linearly with $N$, which is too costly. This paper adopts the $\chi^2$ divergence $\chi^2(p\|q) = \sum_j (p_j/q_j)^2 q_j - 1$, proving it can be written as a double integral over $x, x'$ (Fact 1). This yields an **unbiased estimator** computable with a single batch in $O(NB)$ time: $\hat\chi^2 = \frac{1}{B(B-1)}\sum_j \frac{1}{b_j}\big[(\sum_i [s_{\varepsilon,g}(x_i)]_j)^2 - \sum_i [s_{\varepsilon,g}(x_i)]_j^2\big] - 1$. Experiments (Fig 2, 3) confirm that lower $\hat\chi^2$ leads to lower FID and curvature, with benefits saturating at $\hat\chi^2 \approx 0.05$, providing a clear stopping point for precomputation.

**3. Convergence Analysis covering $\varepsilon=0$: Supporting Non-regularized Usage**. Genevay et al. (2016) only analyzed $\varepsilon>0$, but SD-FM is most efficient at $\varepsilon=0$ (where pairing is exact MIPS, allowing fast retrieval, and avoiding the expensive $N$-dimensional softmax sampling required for $\varepsilon>0$). Under additional regularity assumptions for $\varepsilon=0$ (point separation $\delta = \min_{j\ne j'}\|y_j - y_{j'}\| > 0$, $\mu$ having a density, and bounded surface area), and defining $L_0 = C_\mu^{\max}/\delta$, Theorem 2 proves that for any $\varepsilon\ge 0$, SGD iterations satisfy $\mathbb{E}[\chi^2(m(g_t)\|b)] \lesssim \frac{1}{\min_j b_j}\sqrt{L_\varepsilon \Delta / K}$, and the entropy transport cost approaches optimality at $O((1/K)^{1/4})$. This unifies the treatments of $\varepsilon=0$ and $\varepsilon>0$, theoretically justifying the use of $\varepsilon=0$ throughout the paper.

**4. Generalized Tweedie Formula: Enabling Score Estimation and Guidance for SD Couplings**. Under independent coupling, the FM velocity field and score relate via the Tweedie formula $\nabla\log\rho_t(x) = \frac{t v_t(x) - x}{1-t}$. This relationship fails under general couplings due to the dependence between $X_0$ and $X_1$. Proposition 3 provides a generalized version: $\nabla\log\rho_t(x) = \frac{t v_t(x) - x + (1-t)\delta_\varepsilon}{(1-t)^2}$, where the correction term $\delta_\varepsilon$ vanishes as $\varepsilon\to\infty$ (independent coupling) **and** as $\varepsilon\to 0$ ($\|\delta_\varepsilon\| \lesssim e^{-1/\varepsilon}/\varepsilon$; intuitively, $X_1$ is almost determined by $X_0$ as $\varepsilon\to 0$). This means SD-FM models with $\varepsilon=0$ can recover the score directly from the velocity field, supporting corrected sampling like CFG/autoguidance (Proposition 4 provides a weight-based resampling corrector).

## Key Experimental Results

### Main Results: ImageNet-64 / PetFace Unconditional + Class-Conditional (FID ↓)

| Dataset | Method | Euler 4 | Euler 8 | Euler 16 | Dopri5 | Class-Cond Euler 4 | Class-Cond Dopri5 |
|---|---|---|---|---|---|---|---|
| ImageNet-64 | I-FM | — | 79.95 | 37.90 | 9.10 | 34.51 | 3.91 |
| ImageNet-64 | SD-FM (PCA500, ε=0) | **45.62** | **23.75** | **15.02** | **8.42** | 26.04 | **3.63** |
| PetFace | I-FM | — | 56.53 | 26.85 | 1.26 | 47.66 | 1.09 |
| PetFace | SD-FM (full d=12k, ε=0) | **20.54** | **12.77** | **7.50** | 1.26 | **19.10** | **1.05** |

The improvements are most significant at low NFE (Euler 4/8)—the exact "inference budget savings" promised by OT straightening.

### Ablation Study and Cost Comparison

| Dimension | Finding |
|---|---|
| Potential Quality $\to$ FID (Fig 3) | Lower $\hat\chi^2$ correlates with simultaneous drops in FID and curvature, saturating at $\hat\chi^2 \approx 0.05$. |
| Pairing Time (Fig 4, Table 1) | SD-FM pairing overhead is negligible compared to the FM gradient $\Theta$; OT-FM with large $n$ takes >10 days on 64x64. |
| $\varepsilon \in \{0, 0.01, 0.1\}$ | Minimal performance difference; $\varepsilon=0$ is recommended for MIPS compatibility and faster pairing. |
| CelebA Super-res (Table 3, Cont. Cond) | 4x SR PSNR 21.17 $\to$ 21.41, 8x SR PSNR 17.52 $\to$ 17.94; SD-FM outperforms I-FM in both. |
| Mean-Flow Single-step (ImgN-256 latent, Fig 6) | SD-MF outperforms I-MF in FID at low NFE, proving SD coupling benefits extend beyond standard FM. |
| Guidance (Fig 5) | Correction via generalized Tweedie improves precision by increasing resampling count $r$ (at the cost of recall). |

### Key Findings
- Investing precomputation into "fitting the potential vector $g$ accurately" reliably yields lower FID and straighter flows, with a clear saturation point.
- SD-FM consistently outperforms I-FM across all inference budgets, datasets, and conditions (unconditional, conditional, mean-flow) with negligible pairing overhead, and is orders of magnitude cheaper than OT-FM.
- $\varepsilon=0$ is the recommended configuration as it allows for MIPS acceleration with almost no performance loss.

## Highlights & Insights
- **Changing the Problem, Not Just the Algorithm**: Instead of trying to optimize $n \times n$ Sinkhorn for larger batches, the paper shifts to a semidiscrete formulation. This converts "per-batch online calculation" into "one-time precomputation + online lookup," removing the quadratic dependence on $n$.
- **Utility of Unbiased $\chi^2$**: Provides the first convergence criterion for stochastic SD-OT that can be estimated unbiasedly from a single batch and correlates strongly with downstream FID, making the precomputation phase measurable and terminable.
- **Elegance of $\delta_\varepsilon$ Vanishing at Both Ends**: The Tweedie correction term vanishes at both $\varepsilon\to 0$ and $\varepsilon\to \infty$. This implies that the cheapest configuration ($\varepsilon=0$, enabling MIPS) preserves score estimation capabilities, providing guidance functionality at no extra cost.
- **Clarity of the Cost Table (Table 1)**: Highlights that while $\Theta$ dominates training, the marginal pairing cost $dN$ of SD-FM is effectively free, whereas the $dn/\varepsilon^2$ of OT-FM is computationally expensive.

## Limitations & Future Work
- **Precomputation scales with $N$**: For datasets with billions of points, fitting $g$ remains a challenge (though still much smaller than the FM training itself). This may require batching, momentum, or $\varepsilon$-tempering.
- **Categorical Sampling is Expensive for $\varepsilon>0$**: Sampling from an $N$-dimensional softmax is slow, which is why the author recommends $\varepsilon=0$.
- **Exact MIPS**: The current implementation uses exact MIPS; utilizing approximate MIPS for further speedup is left for future work.
- **Interactions with Advanced Methods**: Combining SD couplings with Reflow or other orthogonal methods (as well as reverse-lookup of noise within Laguerre cells from a dataloader perspective) remains to be explored.

## Related Work & Insights
- **OT-FM Lineage**: Pooladian (2023) and Tong (2024) introduced batch-OT couplings; Davtyan (2025) proposed LOOM-CFM to cache Hungarian pairings but still costs $O(n^3)$ per batch; Zhang et al. (2025) used ultra-large $n$ and multi-GPU Sinkhorn to show that "$n$ matters"—this work breaks the cost barrier of that trend.
- **Semidiscrete OT**: The theories of SD-OT from Oliker-Prussner, Mérigot (2011), Cuturi-Peyré (2018), and An (2020, AE-OT), combined with Genevay et al. (2016)'s stochastic dual optimization, form the foundation of this method.
- **Flow/Diffusion Unification**: Stochastic interpolants (Albergo 2023), FM (Lipman 2023), and the score-velocity bridge via Tweedie are sources for the generalized Tweedie formula.
- **Concurrent Work**: Kong et al. (2026) also use SD-OT to improve flow models (focusing on unconditional/conditional generation). This paper distinguishes itself with the unbiased convergence criterion, the analysis covering $\varepsilon=0$, and the generalized Tweedie for guidance.
- **Insight**: When the cost of "making a sub-problem larger and more accurate" explodes, consider if the problem can be reformulated into an equivalent but more structurally friendly representation (here, reducing continuous-continuous OT to continuous-discrete by leveraging data finiteness). This often allows amortizing online calculations into one-time precomputations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Introduces SD-OT to Flow Matching with an unbiased convergence criterion and generalized Tweedie. Clear perspective and solid breakthrough point; minor deduction as SD-OT is an existing tool and for the existence of concurrent work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers unconditional, class-conditional, continuous-conditional super-resolution, guidance, and mean-flow. Multiple datasets and solvers, with direct pairing time comparisons. A minor regret is the lack of a full high-resolution pixel-space comparison against OT-FM due to the latter's cost.
- **Writing Quality**: ⭐⭐⭐⭐ — Comparison of three couplings in Fig 1 and the complexity Table 1 explain the motivation and value very clearly. Smooth transition between theory and experiments.
- **Value**: ⭐⭐⭐⭐ — Makes OT-guided FM truly affordable and usable. Highly relevant for engineering practices seeking high-quality generation at low NFE. Already integrated into OTT-JAX.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Carré du champ Flow Matching: 用几何感知噪声改善生成模型的质量-泛化权衡](carré_du_champ_flow_matching_better_quality-generalisation_tradeoff_in_generativ.md)
- [\[ICLR 2026\] Edit-Based Flow Matching for Temporal Point Processes](edit-based_flow_matching_for_temporal_point_processes.md)
- [\[ICLR 2026\] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference](meancache_from_instantaneous_to_average_velocity_for_accelerating_flow_matching_.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
