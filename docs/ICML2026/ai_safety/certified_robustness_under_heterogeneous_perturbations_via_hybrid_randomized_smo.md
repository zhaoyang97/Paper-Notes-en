---
title: >-
  [Paper Note] Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing
description: >-
  [ICML 2026][AI Safety][Randomized Smoothing] This work extends randomized smoothing (RS) from "supporting only single continuous or discrete input" to the hybrid perturbation setting of "discrete tokens + continuous imag…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Randomized Smoothing"
  - "Neyman–Pearson"
  - "Multimodal Safety Filtering"
  - "Hybrid Perturbation Certification"
  - "Prompt Injection"
date: 2026-05-08
content_hash: 2ab9491c47f40ba5
---

# Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing

**Conference**: ICML 2026  
**arXiv**: [2605.12876](https://arxiv.org/abs/2605.12876)  
**Code**: Not explicitly released  
**Area**: Multimodal VLM / Adversarial Robustness / Certified Robustness  
**Keywords**: Randomized Smoothing, Neyman–Pearson, Multimodal Safety Filtering, Hybrid Perturbation Certification, Prompt Injection

## TL;DR
This work extends randomized smoothing (RS) from "supporting only single continuous or discrete input" to the hybrid perturbation setting of "discrete tokens + continuous images." Through a hybrid Neyman–Pearson analysis, it derives a **one-dimensional, continuous, invertible** likelihood ratio CDF, transforming the originally combinatorial discrete knapsack problem into a solvable root-finding problem. The first model-agnostic certificate for "joint image-text insecurity" is provided on LLaVA-Guard multimodal safety filtering.

## Background & Motivation
**Background**: Randomized Smoothing is currently the most mainstream model-agnostic robustness certification method: on the continuous side (Cohen 2019), there is a closed-form Gaussian $\ell_2$ certificate; on the discrete side (Ye 2020, Chen 2025), a fractional knapsack is needed to compute the worst-case likelihood ratio; both are well-established in their respective domains.

**Limitations of Prior Work**: Attacks on modern multimodal systems (VLMs, agents, robot safety) are **cross-modal**—safe when considering image or text alone, but unsafe in combination (e.g., Hateful Memes, prompt injection). Simply combining unimodal certificates is mathematically incorrect; there is no unified joint likelihood ratio framework.

**Key Challenge**: Purely discrete likelihood ratios are atomic, making NP decision rules non-invertible and preventing closed-form radii; pure Gaussian NP only supports continuous inputs; the joint NP optimal rejection region is fundamentally not a "Cartesian product of two unimodal thresholds" (Prop. 4.1 provides a direct counterexample).

**Goal**: (i) Provide a strict NP closed-form certificate under discrete + continuous hybrid perturbations; (ii) Offer a monotone, conservative engineering algorithm; (iii) Validate the certificate's practicality on multimodal safety filtering tasks with interaction-level insecurity.

**Key Insight**: Observing that as long as the joint likelihood ratio $\gamma(z_1,z_2)=\gamma_1(z_1)\cdot\gamma_2(z_2)$ contains a Gaussian factor, $\log\gamma$ is strictly monotonic in the continuous coordinate—meaning "continuous noise smooths out the atomic structure of the discrete likelihood ratio," collapsing the joint NP problem to one dimension.

**Core Idea**: Use continuous Gaussian smoothing as a "regularizer" to fuse the discrete knapsack problem into a continuous, invertible one-dimensional CDF $F(t;r)$, then solve the NP threshold $t^\star(r)$ via one-dimensional bisection, and aggregate over the worst-case discrete attack space.

## Method

### Overall Architecture
Given input $x=(x_1,x_2)$ (text + image), two independent smoothing kernels are used: text $Z_1\sim p_1(\cdot\mid x_1)$ (uniform/absorbing replacement), image $Z_2\sim\mathcal{N}(x_2,\sigma^2 I)$. The base classifier $f$ is smoothed into a smoothed classifier $g(x)=\mathbb{E}[f(Z_1,Z_2)]$. Given joint perturbation budget $(d,\epsilon)$ ($\ell_0$ + $\ell_2$), define the hybrid worst-case probability $p_{\mathrm{adv}}(d,\epsilon)$. The overall algorithm: ① Monte Carlo estimate of the Clopper-Pearson lower bound for clean $p_A$ → ② Use kernel symmetry to enumerate/analyze the worst-case discrete adversary → ③ For each candidate $x_{1,\mathrm{adv}}$, solve the one-dimensional NP threshold $t^\star$ → ④ Compute $V_k$ → ⑤ Take the minimum as the final conservative certification value.

### Key Designs

1. **One-dimensional CDF $F(t;r)$ of the Joint Likelihood Ratio**:

    - **Function**: Expresses the hybrid NP capacity constraint of "discrete atomic likelihood ratio + continuous Gaussian likelihood ratio" as a single-variable, continuous, strictly increasing function.
    - **Mechanism**: Define $F(t;r)=\sum_{z_1} p_1(z_1\mid x_1)\,\Phi\!\big(\tfrac{r^2/2+\sigma^2(\log t-\log\gamma_1(z_1))}{\sigma r}\big)$, where $\Phi$ is the standard Gaussian CDF and $r$ is the continuous perturbation radius. Using the additive decomposition $\log\gamma(z_1,z_2)=\log\gamma_1(z_1)+rz_2-r^2/2$, taking the Gaussian expectation over $z_2$ yields the above; for each $r>0$, there exists a unique $t^\star(r)$ such that $F(t^\star;r)=p_A$ (NP capacity constraint).
    - **Design Motivation**: Purely discrete NP cannot exactly match $p_A$ with a threshold rule due to atomic likelihood ratios (requiring fractional allocation); introducing a continuous dimension allows the Gaussian to stretch $\log t$ into a continuous scalar, collapsing the original "combinatorial search + fractional knapsack" into "bisection over $u=\log t$," solvable on CPU in under 1 second.

2. **Closed-form Worst-case Probability $V(x_{1,\mathrm{adv}};r)$ and Monotonicity at $r=\epsilon$**:

    - **Function**: Directly computes the worst-case smoothed value for a given discrete adversary $x_{1,\mathrm{adv}}$ and continuous radius $r$.
    - **Mechanism**: $V(x_{1,\mathrm{adv}};r)=\sum_{z_1} p_1(z_1\mid x_{1,\mathrm{adv}})\,\Phi\!\big(\tfrac{r^2/2+\sigma^2(\log t^\star(r)-\log\gamma_1(z_1))}{\sigma r}-\tfrac{r}{\sigma}\big)$, and it is proven that $V$ is monotonically non-increasing in $r$, so the continuous worst-case is always at $r=\epsilon$; finally, $p_{\mathrm{adv}}(d,\epsilon)=\min_{D_1(x_1,x_{1,\mathrm{adv}})\le d}V(x_{1,\mathrm{adv}};\epsilon)$.
    - **Design Motivation**: The double-layer infimum over all $(x_{1,\mathrm{adv}},x_{2,\mathrm{adv}})$ is folded into "enumerate discrete attacks + solve a one-dimensional equation" via monotonicity, avoiding actual search over the continuous space $\mathbb{R}^D$; monotonicity also provides a monotone-in-$d$ certification invariant, facilitating plotting.

3. **Structurally Symmetric Discrete Kernel + Conservative Implementation via One-dimensional Root-finding**:

    - **Function**: Ensures that "taking the worst over all discrete adversaries" under common text smoothing (uniform/absorbing) does not require combinatorial enumeration; the entire algorithm is only about 3 times slower than image-only RS.
    - **Mechanism**: For suffix attack or $\ell_0$ attack, the $p_1(\cdot\mid x_{1,\mathrm{adv}})$ of the uniform/absorbing kernel depends only on the edit budget $d$ rather than specific token identities (kernel symmetry), so a canonical adversarial input can represent the entire attack set; the NP threshold is solved via monotone bisection over $u=\log t$; clean $p_A$ uses one-sided Clopper-Pearson for a conservative lower bound; floating-point errors are controlled by the numerical precision strategy in Appendix A.7.
    - **Design Motivation**: The original NP formula involves $O(|\mathcal{V}|^d)$ discrete combinatorial space, which is the key bottleneck for the method's practicality; the authors deliberately choose the uniform kernel over the absorbing kernel (the latter degenerates to a two-point distribution with exponential decay $\beta^d$ under suffix attack), ensuring the certificate is both conservative and non-trivial.

### Loss & Training
This is a **pure certification algorithm**; the base classifier is not trained, but directly applied to existing frozen models such as LLaVA-Guard or linear SVM. Hyperparameters: $\alpha=0.01$ (CP risk), $n=10^4$ (MC sample size), $\beta=0.25$ (token replacement probability), $\sigma\in\{0.5,1.0\}$ (Gaussian variance), certification threshold $\tau=4.6\times 10^{-5}$ as in Chen 2025a.

## Key Experimental Results

### Main Results

| Method | Image radius $\bar{r}$ | Text budget $\bar{d}$ |
|--------|----------------------|---------------------|
| Image-only RS | 3.99 | 0 |
| Text-only RS | 0 | 3.26 |
| **Hybrid RS (ours)** | **3.76** (at $d=1$) | **3.07** |

The hybrid certificate, at text budget $d=1$, has an image radius only 5.8% lower than the pure image certificate, and a text budget only 5.8% lower than the pure text certificate, but **simultaneously** provides joint image-text guarantees—whereas unimodal certificates are unsound on interaction-only datasets. On the MM-SafetyBench external validation (1680 samples, 7.5% pass interaction-only filter), it achieves $\bar{d}=3.62$ / $\bar{r}=3.37$.

### Ablation Study

| $\beta$ (corruption rate) | Certified examples (%) | Mean $d_{\max}$ | Mean $r^\star(d_{\max})$ |
|--------------------------|-----------------------|-----------------|--------------------------|
| 0.1 | 82.35 | 2.29 | 4.99 |
| **0.25** | **70.59** | **3.07** | **3.21** |
| 0.5 | 58.82 | 4.00 | 3.24 |
| 1.0 | 41.18 | 8.00 | 4.57 |

| Setting | Time/datapoint | Effect |
|---------|---------------|--------|
| Image-only RS | ≈156s | Single image radius |
| Hybrid RS, default | ≈500s | Full $(d,\epsilon)$ frontier |
| Hybrid RS + FlashAttention/batching | ≈0.7× | Same certificate |
| One-shot suffix / $\ell_0$, $d_{\max}=8$ | ≈44s | Slightly lower radius (2.07→1.55) |

### Key Findings
- $\beta$ controls the coverage-budget tradeoff: small $\beta$ certifies more samples but only up to small $d$, large $\beta$ broadens the text budget but reduces coverage; $\beta=0.25$ is the default balance.
- Increasing Gaussian variance $\sigma$ (0.5→1.0) sacrifices certification accuracy at small $\epsilon$, but increases the maximum certifiable image radius; for high text budgets $d>3$, $\sigma=1.0$ almost always fails to certify.
- Adaptive attack experiments (Sec 5.3) show that the empirical attack success rate is below the theoretical $p_{\mathrm{adv}}$ bound, indicating the certificate is not vacuous; MMCert-style subsampling achieves **zero certification** on interaction-only data, further demonstrating the necessity of the joint NP certificate for this task.

## Highlights & Insights
- **"Continuous smoothing regularizes discrete knapsack" is the core insight**: Gaussian noise not only provides the $\ell_2$ radius, but also smooths out the atomic ties of the discrete likelihood ratio, turning the originally non-invertible NP decision rule into a one-dimensional invertible CDF—$\sigma$ here serves as both a "continuous radius controller" and a "discrete regularizer."
- **The joint certificate strictly generalizes two special cases**: When $x_{1,\mathrm{adv}}=x_1$, it reduces to the classic Cohen Gaussian certificate; as $\sigma\to\infty$, it reduces to the fractional knapsack discrete certificate (Appendix A.3); such "lossless generalization" is rare in the multimodal certification literature.
- **Interaction-only evaluation is well designed**: The authors construct a 400-sample subset on Hateful Memes where "image safe + text safe but combination unsafe," turning the qualitative statement "unimodal certificates are unsound" into a measurable experimental fact—single MMCert achieves zero certification on this subset, providing a strong contrast.

## Limitations & Future Work
- Only supports binary (safe/unsafe) outputs and $\ell_2$ + $\ell_0$ geometries; extension to multiclass, $\ell_\infty$, or semantic-level perturbations requires new NP analysis.
- The text side uses a uniform kernel (avoiding the exponential degeneration of absorbing), but uniform replacement significantly damages semantics, leading to substantial clean accuracy loss for long prompts (Appendix A.9 Table 5).
- For large discrete budgets ($d\ge 5$ at $\sigma=1.0$), certification is almost impossible; it remains weak against real long-suffix prompt injection. $\bar{d}_{\mathrm{hybrid}}=0.33$ under $\ell_0$ attack is much lower than $\bar{d}_{\mathrm{txt}}=1.02$, indicating the hybrid certificate is conservative in the $\ell_0$ setting.
- Each certification takes 500s ($10^4$ MC), acceptable offline but difficult for real-time deployment; the authors suggest confidence sequence early stopping + input-adaptive sampling as future directions.

## Related Work & Insights
- **vs Cohen 2019 / Salman 2019 (Gaussian RS)**: This work strictly generalizes their continuous certificate, exactly recovering the $\Phi^{-1}(p_A)-\Phi^{-1}(\tau)$ formula when there is no discrete perturbation.
- **vs Chen 2025a (fractional knapsack for LLM safety)**: They only solve the pure discrete NP via 0-1/fractional knapsack solver; this work proves that "adding Gaussian collapses the knapsack to a one-dimensional equation," reducing combinatorial complexity to $O(\log\epsilon^{-1})$.
- **vs MMCert (Wang 2024)**: MMCert independently subsamples each modality and aggregates, essentially an $\ell_0$-cross-modal threshold; it achieves zero certification on interaction-only data, highlighting the irreplaceability of the joint NP framework in this work.
- **vs COMMIT / CertTA**: These ad-hoc multi-sensor/network certifications are not based on classical NP analysis; this work provides the first principled joint Neyman–Pearson certificate for heterogeneous discrete-continuous threats.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a closed-form joint NP certificate for discrete + continuous hybrid perturbations, with an elegant insight collapsing the combinatorial knapsack into a one-dimensional equation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers tabular data, multimodal safety, empirical attacks, external benchmarks, and multiple $\beta/\sigma$ ablations; could be improved with larger $d$ and more base model coverage.
- Writing Quality: ⭐⭐⭐⭐ Theorems, propositions, and counterexamples are rigorous and self-consistent, with clear identification of each limitation (absorbing degeneracy, numerical safety), and clear structure.
- Value: ⭐⭐⭐⭐ Provides the first theoretically rigorous model-agnostic certificate for multimodal safety filtering and prompt injection, directly relevant for high-stakes deployments (medical VLMs, robotics).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](../../CVPR2026/multimodal_vlm/pop_proof_of_perception_conformal_reasoning.md)
- [\[ICML 2026\] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)

</div>

<!-- RELATED:END -->
