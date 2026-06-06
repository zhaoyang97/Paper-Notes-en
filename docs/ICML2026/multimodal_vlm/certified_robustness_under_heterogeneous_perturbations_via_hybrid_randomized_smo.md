---
title: >-
  [Paper Note] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing
description: >-
  [ICML 2026][Multimodal VLM][Randomized Smoothing] This paper extends Randomized Smoothing (RS) from scenarios supporting only single continuous or discrete inputs to hybrid perturbation scenarios involving "discrete toke…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Randomized Smoothing"
  - "Neyman–Pearson"
  - "Multimodal Safety Filtering"
  - "Hybrid Perturbation Certification"
  - "prompt injection"
date: 2026-05-08
content_hash: 276a3799fc2371f9
---

# Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing

**Conference**: ICML 2026  
**arXiv**: [2605.12876](https://arxiv.org/abs/2605.12876)  
**Code**: Not explicitly disclosed  
**Area**: Multimodal VLM / Adversarial Robustness / Certified Robustness  
**Keywords**: Randomized Smoothing, Neyman–Pearson, Multimodal Safety Filtering, Hybrid Perturbation Certification, prompt injection

## TL;DR
This paper extends Randomized Smoothing (RS) from scenarios supporting only single continuous or discrete inputs to hybrid perturbation scenarios involving "discrete tokens + continuous images." Through a hybrid Neyman–Pearson analysis, it derives a **one-dimensional, continuous, and invertible** likelihood ratio CDF. This transforms the combinatorial knapsack problem into a solvable root-finding problem and provides the first model-agnostic certificate against "jointly unsafe image-text" inputs for the LLaVA-Guard multimodal safety filter.

## Background & Motivation
**Background**: Randomized Smoothing is currently the most prominent model-agnostic robustness certification method. The continuous side (Cohen 2019) provides closed-form $\ell_2$ certificates for Gaussian noise, while the discrete side (Ye 2020, Chen 2025) requires fractional knapsack to determine the worst-case likelihood ratio; the two systems have historically evolved independently.

**Limitations of Prior Work**: Attacks on modern multimodal systems (VLM, agents, robot safety) are **cross-modal**—the image alone and text alone may be safe, but their combination is unsafe (e.g., Hateful Memes, prompt injection). Simply concatenating single-mode certificates is mathematically incorrect as it lacks a unified joint likelihood ratio framework.

**Key Challenge**: Purely discrete likelihood ratios are atomic, leading to non-invertible NP decision rules that cannot yield closed-form radii. Conversely, pure Gaussian NP only supports continuous inputs. The optimal rejection region of the joint NP is not simply the "Cartesian product of two single-mode thresholds" (disproven by the counterexample in Prop. 4.1).

**Goal**: (i) Provide rigorous NP closed-form certificates under hybrid discrete + continuous perturbations; (ii) develop monotone and conservative engineering algorithms; (iii) verify the certificate's utility on multimodal safety filtering tasks involving interaction-level unsafety.

**Key Insight**: It is observed that as long as the joint likelihood ratio $\gamma(z_1,z_2)=\gamma_1(z_1)\cdot\gamma_2(z_2)$ contains a Gaussian factor, $\log\gamma$ is strictly monotonic in continuous coordinates. This effectively uses continuous noise to "smooth out" the atomic structure of the discrete likelihood ratio, collapsing the joint NP problem into one dimension.

**Core Idea**: Continuous Gaussian smoothing is used as a "regularizer" to fuse the discrete knapsack problem into a continuous, invertible 1D CDF $F(t;r)$. The NP threshold $t^\star(r)$ is then solved via 1D bisection, followed by a worst-case aggregation over the discrete attack space.

## Method

### Overall Architecture
For an input $x=(x_1,x_2)$ (text + image), two independent smoothing kernels are applied: text $Z_1\sim p_1(\cdot\mid x_1)$ (uniform/absorbing substitution) and image $Z_2\sim\mathcal{N}(x_2,\sigma^2 I)$. The base classifier $f$ is smoothed into $g(x)=\mathbb{E}[f(Z_1,Z_2)]$. Given a joint perturbation budget $(d,\epsilon)$ ($\ell_0$ + $\ell_2$), the hybrid worst-case probability $p_{\mathrm{adv}}(d,\epsilon)$ is defined. The overall algorithm: ① Estimate the Clopper-Pearson lower bound of the clean $p_A$ via Monte Carlo → ② Enumerate/analyze the worst-case discrete adversary using kernel symmetry → ③ Solve for the 1D NP threshold $t^\star$ for each candidate $x_{1,\mathrm{adv}}$ → ④ Calculate $V_k$ → ⑤ Take the minimum as the final conservative certification value.

### Key Designs

1.  **1D CDF of the Joint Likelihood Ratio $F(t;r)$**:
    - **Function**: Expresses the hybrid NP capacity constraint (comprising discrete atomic LR and continuous Gaussian LR) as a univariate, continuous, and strictly increasing function.
    - **Mechanism**: Defines $F(t;r)=\sum_{z_1} p_1(z_1\mid x_1)\,\Phi\!\big(\tfrac{r^2/2+\sigma^2(\log t-\log\gamma_1(z_1))}{\sigma r}\big)$, where $\Phi$ is the standard Gaussian CDF and $r$ is the continuous perturbation radius. Using the additive decomposition $\log\gamma(z_1,z_2)=\log\gamma_1(z_1)+rz_2-r^2/2$, taking the Gaussian expectation over $z_2$ yields the formula. For each $r>0$, there exists a unique $t^\star(r)$ such that $F(t^\star;r)=p_A$ (NP capacity constraint).
    - **Design Motivation**: Pure discrete NP cannot exactly match $p_A$ with a threshold rule due to likelihood ratio atomization (requiring fractional allocation). Introducing a continuous dimension stretches $\log t$ into a continuous scalar via Gaussian noise, collapsing the "combinatorial search + fractional knapsack" into a "bisection over $u=\log t$," solvable on a CPU in $< 1$ second.

2.  **Closed-form Worst-case Probability $V(x_{1,\mathrm{adv}};r)$ and $r=\epsilon$ Monotonicity**:
    - **Function**: Directly calculates the worst-case smoothed value given a discrete adversary $x_{1,\mathrm{adv}}$ and continuous radius $r$.
    - **Mechanism**: $V(x_{1,\mathrm{adv}};r)=\sum_{z_1} p_1(z_1\mid x_{1,\mathrm{adv}})\,\Phi\!\big(\tfrac{r^2/2+\sigma^2(\log t^\star(r)-\log\gamma_1(z_1))}{\sigma r}-\tfrac{r}{\sigma}\big)$. It is proven that $V$ is monotonically non-increasing regarding $r$; thus, the continuous worst-case automatically occurs at $r=\epsilon$. Finally, $p_{\mathrm{adv}}(d,\epsilon)=\min_{D_1(x_1,x_{1,\mathrm{adv}})\le d}V(x_{1,\mathrm{adv}};\epsilon)$.
    - **Design Motivation**: Monotonicity collapses the dual-layer inf (minimizing over all $(x_{1,\mathrm{adv}},x_{2,\mathrm{adv}})$) into an enumeration over discrete attacks + 1D root-solving, avoiding actual searches in $\mathbb{R}^D$ space. Monotonicity also provides a certification invariant for $d$, facilitating visualization.

3.  **Kernel Symmetry for Discrete Space + Conservative 1D Root-Solving**:
    - **Function**: Ensures that "minimizing over all discrete adversaries" does not require combinatorial enumeration under common text smoothing kernels (uniform/absorbing), keeping the algorithm only ~3x slower than image-only RS.
    - **Mechanism**: Under suffix or $\ell_0$ attacks, $p_1(\cdot\mid x_{1,\mathrm{adv}})$ for uniform/absorbing kernels depends only on the budget $d$ rather than specific token identities (kernel symmetry). This allows a canonical adversarial input to represent the entire attack set. The NP threshold is solved via monotone bisection on $u=\log t$, while the clean $p_A$ uses a conservative lower bound via Clopper-Pearson. Floating-point errors are managed by the numerical precision strategy in Appendix A.7.
    - **Design Motivation**: The original NP formula involves a discrete combinatorial space of $O(|\mathcal{V}|^d)$, which is the primary bottleneck for practicality. The authors choose the uniform kernel over the absorbing kernel (the latter decays exponentially as $\beta^d$ under suffix attacks) to ensure the certificate is both conservative and non-trivial.

### Loss & Training
Ours is a **pure certification algorithm** that does not train the base classifier, applying directly to frozen models like LLaVA-Guard or linear SVMs. Hyperparameters: $\alpha=0.01$ (CP risk), $n=10^4$ (MC samples), $\beta=0.25$ (token substitution probability), $\sigma\in\{0.5,1.0\}$ (Gaussian variance). The certification threshold $\tau=4.6\times 10^{-5}$ follows Chen 2025a.

## Key Experimental Results

### Main Results

| Method | Image radius $\bar{r}$ | Text budget $\bar{d}$ |
|--------|----------------------|---------------------|
| Image-only RS | 3.99 | 0 |
| Text-only RS | 0 | 3.26 |
| **Hybrid RS (ours)** | **3.76** (at $d=1$) | **3.07** |

At a text budget of $d=1$, the Hybrid certificate's image radius is only 5.8% lower than the image-only certificate, and its text budget is only 5.8% lower than the text-only certificate. Crucially, it provides a **joint** guarantee—whereas single-modality certificates are unsound on interaction-only datasets. External validation on MM-SafetyBench (1680 samples, 7.5% passing interaction-only filter) yielded $\bar{d}=3.62$ / $\bar{r}=3.37$.

### Ablation Study

| $\beta$ (corruption rate) | Certified examples (%) | Mean $d_{\max}$ | Mean $r^\star(d_{\max})$ |
|--------------------------|-----------------------|-----------------|--------------------------|
| 0.1 | 82.35 | 2.29 | 4.99 |
| **0.25** | **70.59** | **3.07** | **3.21** |
| 0.5 | 58.82 | 4.00 | 3.24 |
| 1.0 | 41.18 | 8.00 | 4.57 |

| Setting | Time/datapoint | Effect |
|---------|---------------|------|
| Image-only RS | ≈156s | Single image radius |
| Hybrid RS, default | ≈500s | Complete $(d,\epsilon)$ frontier |
| Hybrid RS + FlashAttention/batching | ≈0.7× | Same certificate |
| One-shot suffix / $\ell_0$, $d_{\max}=8$ | ≈44s | Slight radius drop (2.07→1.55) |

### Key Findings
- $\beta$ controls the coverage-budget trade-off: small $\beta$ certifies more samples but only for small $d$, while large $\beta$ broadens the text budget at the cost of coverage. $\beta=0.25$ is the default balance.
- Increasing Gaussian variance $\sigma$ (0.5 to 1.0) sacrifices certification precision at small $\epsilon$ but allows for larger certified image radii. For high text budgets ($d > 3$), $\sigma=1.0$ leads to near-total certification failure.
- Adaptive attack experiments (Sec 5.3) show a gap between empirical attack success and the theoretical $p_{\mathrm{adv}}$ bound, indicating the certificates are not vacuous. MMCert-style subsampling produces **zero certification** on interaction-only data, highlighting the necessity of the joint NP framework.

## Highlights & Insights
- **"Continuous smoothing regularizing discrete knapsack" is the core insight**: Gaussian noise provides $\ell_2$ radius and eliminates atomic ties in discrete likelihood ratios, turning non-invertible NP decision rules into a 1D invertible CDF. Here, $\sigma$ acts as both a continuous radius controller and a discrete regularizer.
- **The joint certificate strictly generalizes two special cases**: When $x_{1,\mathrm{adv}}=x_1$, it degenerates into the classic Cohen Gaussian certificate; as $\sigma\to\infty$, it degenerates into the fractional knapsack discrete certificate (Appendix A.3). This "lossless generalization" is rare in multimodal certification literature.
- **Well-designed interaction-only evaluation**: The authors construct a 400-sample subset of Hateful Memes ("image safe + text safe but combination unsafe"), turning the qualitative claim that "single-modality certificates are unsound" into a measurable experimental fact.

## Limitations & Future Work
- Currently only supports binary (safe/unsafe) outputs and $\ell_2$ + $\ell_0$ geometries; NP analysis would need to be reworked for multi-classification, $\ell_\infty$, or semantic perturbations.
- The text side uses a uniform kernel (to avoid the exponential decay of absorbing kernels), but uniform substitution significantly damages semantics, leading to high clean accuracy loss for long prompts (Appendix A.9 Table 5).
- Certification is nearly impossible at large discrete budgets ($d \ge 5$ under $\sigma=1.0$), remaining ineffective against real long-suffix prompt injections. $\bar{d}_{\mathrm{hybrid}}=0.33$ under $\ell_0$ attacks is significantly lower than $\bar{d}_{\mathrm{txt}}=1.02$, suggesting the hybrid certificate is conservative in $\ell_0$ scenarios.
- A single certification takes 500s ($10^4$ MC), which is acceptable offline but difficult for real-time deployment. Future work explores confidence sequence early stopping and input-adaptive sampling.

## Related Work & Insights
- **vs. Cohen 2019 / Salman 2019 (Gaussian RS)**: This work strictly generalizes their continuous certificates, exactly reproducing the $\Phi^{-1}(p_A)-\Phi^{-1}(\tau)$ formula when no discrete perturbation exists.
- **vs. Chen 2025a (fractional knapsack for LLM safety)**: That work only solves the pure discrete side via 0-1/fractional knapsack solvers; this work proves that adding Gaussian noise collapses the knapsack into a 1D equation, reducing complexity to $O(\log\epsilon^{-1})$.
- **vs. MMCert (Wang 2024)**: MMCert uses independent subsampling for each modality before aggregation (essentially an $\ell_0$ cross-modal threshold). Its zero certification on interaction-only data underscores the necessity of the joint NP framework.
- **vs. COMMIT / CertTA**: These ad-hoc multi-sensor/network certifications are not based on classical NP analysis. This work provides the first principled joint Neyman-Pearson certificate for heterogeneous discrete-continuous threats.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First closed-form joint NP certificate for hybrid perturbations; beautiful insight on collapsing knapsack into a 1D equation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers tabular data, multimodal safety, empirical attacks, external benchmarks, and extensive $\beta/\sigma$ ablations. Could be improved with larger $d$ and more base models.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theorems, propositions, and counterexamples; clearly states limitations and structural parameters.
- Value: ⭐⭐⭐⭐ Provides the first theoretically rigorous model-agnostic certificate for multimodal safety and prompt injection, with direct implications for high-stakes deployments (medical VLM, robotics).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[ICML 2026\] Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework](hierarchical_synthetic_tabular_data_generation_a_hybrid_top-down_and_bottom-up_f.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)

</div>

<!-- RELATED:END -->
