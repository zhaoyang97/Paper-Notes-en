---
title: >-
  [Paper Note] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States
description: >-
  [ICML 2026][AI Safety][Differential Privacy] The authors provide the first convergent **hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)". By designing a hybrid "directional + isotropic" noise mechanism and constructing an auxiliary process between two adjacent trajectories, they bypass the technical barrier of zeroth-order updates lacking global Lipschitz continuity. This reveals a previously unknown DP algorithm design princi…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Zeroth-Order Optimization"
  - "PABI"
  - "hidden-state analysis"
  - "coupling analysis"
date: 2026-05-08
content_hash: fc1a1e42a6d3053c
---

# Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States

**Conference**: ICML 2026  
**arXiv**: [2506.00158](https://arxiv.org/abs/2506.00158)  
**Code**: None (Theoretical paper)  
**Area**: LLM Security / Differential Privacy / Zeroth-Order Optimization  
**Keywords**: Differential Privacy, Zeroth-Order Optimization, PABI, hidden-state analysis, coupling analysis

## TL;DR
The authors provide the first convergent **hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)". By designing a hybrid "directional + isotropic" noise mechanism and constructing an auxiliary process between two adjacent trajectories, they bypass the technical barrier of zeroth-order updates lacking global Lipschitz continuity. This reveals a previously unknown DP algorithm design principle: "increasing the number of sampling directions $K$ per step actually reduces privacy loss."

## Background & Motivation

**Background**: As model sizes explode to hundreds of billions of parameters, the per-sample gradient clipping in first-order DP methods like DP-SGD incurs massive memory overhead. Recent works such as MeZO-DP (Zhang et al. 2024a) and Tang et al. (2024) have introduced zeroth-order optimization (ZO, using only forward passes to evaluate loss) into the DP framework, enabling fine-tuning of 60B+ models with performance close to DP-LoRA. However, their privacy analysis still relies on composition theorems—where the privacy budget accumulates linearly with the number of training steps $T$, requiring careful control of the stopping point.

**Limitations of Prior Work**: First-order DP-SGD already has the "privacy amplification by iteration (PABI)" theory—where treat intermediate iterates are treated as hidden and only the final parameters are published, allowing $\varepsilon$ to saturate as $T \to \infty$. However, this analysis requires two conditions: (i) isotropic noise (to ensure controllable shifted Rényi divergence); (ii) global Lipschitz continuity of the update mapping. Zeroth-order methods fail both: the noise is scalar Gaussian along a random direction $u$ (anisotropic), and the global Lipschitz constant for ZOGD across all $u$ is significantly larger than in first-order methods. While adding isotropic noise to the entire $\mathbb{R}^d$ could allow the application of existing analysis, it would severely degrade the utility-privacy trade-off (noise added in the $u^\perp$ direction contributes nothing to privacy yet wastes utility).

**Key Challenge**: There is a structural contradiction between the noise shape and the analysis framework—utility requires noise to be as directional as possible (scalar, directional), but PABI's shifted-divergence analysis requires isotropy and global Lipschitz. Thus, the question becomes: "Can a hybrid noise be designed to satisfy both utility and hidden-state analysis?"

**Goal**: (i) Propose a unified noisy update rule for DP-ZOGD that supports both directional and isotropic noise; (ii) Derive the first convergent hidden-state DP bound (where $\varepsilon$ does not explode as $T \to \infty$); (iii) Reveal algorithm design degrees of freedom previously ignored in the literature (the role of the update dimension $K$ and the necessity of orthogonal directions).

**Key Insight**: The authors observe that while the zeroth-order update mapping is not globally Lipschitz, it is **pointwise Lipschitz with high probability**—meaning for a single fixed point relative to nearby points, the Lipschitz constant is much smaller than the global one. This provides an entry point to bypass shifted-divergence: instead of pursuing "controllable Rényi divergence between two adjacent trajectories under the original update," they explicitly construct a third auxiliary process $\widetilde W$ that lies between the two adjacent trajectories.

**Core Idea**: Utilizing a "hybrid noise + coupled auxiliary process" toolkit—the former addresses utility issues, while the latter bypasses Lipschitz obstacles, proving that ZO can also benefit from PABI-style privacy amplification.

## Method

### Overall Architecture
The authors perform projected zeroth-order GD on the ERM problem $L(w;\mathcal D)=\frac1n\sum_i \ell_i(w)$ over a convex bounded domain $\mathcal B_R$. The goal is to prove a privacy upper bound for this forward-pass-only optimizer that does not explode as training steps $T$ increase. Each update incorporates three components: first, calculating two-point ZO gradients using $K$ orthogonal directions $\{u_{t,k}\}_{k=1}^K$ (uniformly sampled from the Stiefel manifold $V_K(\mathbb R^d)$); next, adding a scalar Gaussian noise along each of these directions; and finally, supplementing the entire space with a small isotropic Gaussian noise. The ratio of the two noises is controlled by a continuous knob $\beta_t\in[0,1]$ (directional vs. isotropic). For the analysis, a third auxiliary trajectory $\widetilde W_t$ is explicitly inserted between two adjacent trajectories $W_t,W_t'$, splitting the privacy analysis into two parts: a TV segment for $W_t\leftrightarrow\widetilde W_t$ and a Rényi segment for $\widetilde W_t\leftrightarrow W_t'$, thereby bypassing the lack of global Lipschitz in zeroth-order updates.

### Key Designs

**1. Hybrid Directional + Isotropic Noisy-ZOGD Mechanism: Unifying Two Old Schemes with a Continuous Knob**

Previously, zeroth-order DP literature featured two separate noise mechanisms: mechanism (a), which adds scalar noise along the update direction (good utility but hard to analyze), and mechanism (b), which adds isotropic noise to the full space (easy to analyze but poor utility). The authors parameterize these into a single continuous family, with the update written as $w_{t+1}=\Pi_{\mathcal B_R}[w_t-\frac{\eta}{K}\sum_k \hat g_t(w_t;u_{t,k})+\frac{\eta}{\sqrt K}\sum_k G_{t,k}^{(1)} u_{t,k}+\frac{\eta}{\sqrt d}G_t^{(2)}]$, where the two-point ZO gradient is $\hat g_t(w_t;u_{t,k})=\frac1n\sum_i \mathsf{clip}(\frac{\ell_i(w_t+\xi u_{t,k})-\ell_i(w_t-\xi u_{t,k})}{2\xi};\Delta)\,u_{t,k}$, directional noise $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$, and isotropic noise $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$. The directions $\{u_{t,k}\}$ are chosen to be orthogonal rather than i.i.d. uniform on $\mathbb S^{d-1}$. When $\beta_t=1$, it degrades to mechanism (a); when $\beta_t=0$, it degrades to mechanism (b); intermediate values represent a hybrid. This parameterization maintains an equivalent total noise variance for all $\beta_t$ and $K$, keeping the utility upper bound unchanged while varying the privacy degrees of freedom. Both parts are kept because the directional part concentrates noise on the privacy-sensitive components (utility-friendly), while the isotropic part provides space for the vector shift $v_t$ in the subsequent coupling analysis to be "absorbed"—without this isotropic noise, the shifted Gaussian mechanism segment would fail.

**2. Coupled Auxiliary Process $\widetilde W$: Bypassing Global Lipschitz with Pointwise Lipschitz**

The Lipschitz constant of the zeroth-order update mapping, $c_1=\sqrt{1-\sum_k\upsilon_k+c^2\sum_k\gamma_k}$ (where $\upsilon_k,\gamma_k\sim\mathsf{Beta}(K/2,(d-K)/2)$ are random), is almost certainly not globally $\le c$. Thus, the path used in first-order PABI—where shifted Rényi divergence is controlled along the original trajectory—is blocked. The authors' solution is to insert an auxiliary trajectory $\widetilde W$ between adjacent trajectories $W_t,W_t'$ (corresponding to datasets $\mathcal D,\mathcal D'$), evolving from time $\tau$ as $\widetilde W_{t+1}\stackrel{d}{=}\Pi_{\mathcal B_R}[\hat\psi_t(\widetilde W_t)+Y_t+Z_t+v_t]$, where the shift is $v_t:=\min(a_t,(\|d_t\|-z_{t+1})_+)\frac{d_t}{\|d_t\|}$ and $d_t:=\hat\psi_t(W_t)-\hat\psi_t(\widetilde W_t)$. This insertion splits the analysis into two manageable parts: the TV distance between $W$ and $\widetilde W$ only requires the high-probability pointwise Lipschitz condition (bad event probability $\delta_f$), while the relationship between $\widetilde W$ and $W'$ follows a standard shifted Gaussian mechanism, allowing Rényi divergence to accumulate as in classic PABI. Finally, Lemma 3.7 uses forward tracking $W_\infty(w_t,w_t')\le \min(2R,2\eta\Delta t/\sqrt K)$ to close the bound. The essence is "divide and conquer": discard bad events where global Lipschitz fails into the TV term, and handle good events in the Rényi term—as long as the Beta tail bound proves $c_1\le \bar c_1$ holds with high probability, pointwise Lipschitz is sufficient.

**3. Privacy Gains from Orthogonal Directions + Many-Dimensional $K$: More Directions Improve Privacy**

Combining these components, the authors obtain a closed-form DP upper bound in Theorem 3.2 / Corollary 3.3: $\varepsilon=O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$. From this, a new design principle emerges: the term $MRn\sqrt d/(K\Delta)$ inside the $\min$ is inversely proportional to the number of directions $K$. Thus, when $T \to \infty$ and the bound saturates, $\varepsilon$ decreases as $K$ increases. This contradicts the intuition of standard composition—which suggests that each additional direction exposes more sensitive information, worsening privacy by $\tilde O(\sqrt K)$. Hidden-state analysis changes the rules: under fixed utility constraints, using more directions dilutes the sensitivity of each individual direction. Using orthogonal directions $\{u_{t,k}\}$ instead of i.i.d. uniform samples on $\mathbb S^{d-1}$ further tightens the bound, as the Beta tail bound in Lemma 3.6 is tighter under orthogonality, reflecting no overlapping privacy leaks between orthognoal directions. This phenomenon is unique to zeroth-order methods and absent in first-order analysis.

### Loss & Training
The objective function remains unchanged; the paper provides a tighter privacy accountant for existing ZO DP optimizers. Specifically: (i) Choose $\eta\le K/M$ (strongly convex) or $\le 2K/M$ (convex); (ii) Choose $\xi\le 2\Delta/(n\eta M\sqrt{2d})$; (iii) $K$ must satisfy $\max(20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil MRn\sqrt{2d}/\Delta\rceil),1)\le K\le d/2$. For non-convex cases, a numerical accountant is provided instead of a closed-form one.

## Key Experimental Results

### Main Results
This paper is primarily theoretical, with numerical validation in Figure 1 comparing the $\varepsilon$ curves of hidden-state bound, standard composition, and output perturbation on a smooth strongly convex loss with a bounded domain.

| Method | $\varepsilon$ Trend with $T$ | Remarks |
|:-------|:---------------------------|:--------|
| Standard composition (Theorem 3.1, $\beta_t=1$) | $O(\sqrt{\Delta^2\log(1/\delta)T/(n^2\sigma^2)})$ | Grows unboundedly with $T$ |
| Output perturbation | $O(\sqrt{R^2\log(1/\delta)/\sigma^2})$ | Independent of $T$, but with large constant |
| **Hidden-state DP (Ours, Corollary 3.3)** | $O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$ | Saturates with $T$; inversely proportional to $K$ |

Once $K \ge K_{\min}$ (the lower bound from Corollary 3.3), the **Ours** bound is strictly superior to standard composition and output perturbation in the medium-to-large $T$ regime.

### Ablation Study

| Configuration | Key Finding | Description |
|:--------------|:------------|:------------|
| $\beta_t=1$ (Fully directional) | Best utility, hardest analysis | Prior mechanism (a) |
| $\beta_t=0$ (Fully isotropic) | Easy analysis, worst utility | Prior mechanism (b) |
| $\beta_t\in(0,1)$ | **Gain** in utility-privacy trade-off | Sweet spot of the hybrid scheme |
| $K=1$, i.i.d. spherical | Classic ZO configuration | Loosest privacy bound |
| $K>1$, orthogonal directions | **Gain** in privacy tightness and convergence | New discovery of this work |

### Key Findings
- The first convergent hidden-state DP bound reveals that $\varepsilon$ saturates as $T \to \infty$ to an $O(MRn\sqrt d/(K\Delta))$ magnitude—independent of further training steps and determined only by domain radius, Lipschitz constant, and the number of directions.
- Increasing the sampling direction count $K$ per step actually **decreases** privacy loss in hidden-state analysis—directly contradicting prior work (standard composition) which assumed larger $K$ worsened privacy.
- Replacing i.i.d. spherical sampling with orthogonal $\{u_{t,k}\}$ (Stiefel manifold sampling) further reduces privacy loss by eliminating redundant information leakage between directions.

## Highlights & Insights
- "Constructing an auxiliary process between two adjacent trajectories to split the analysis into TV + Rényi segments" is a highly general coupling analysis technique. Any process lacking global Lipschitz but possessing pointwise Lipschitz can benefit from this. This is potentially applicable to SGD-on-manifolds or Langevin dynamics in non-convex landscapes.
- Parameterizing the noise mechanism as a continuous family $\beta_t\in[0,1]$ rather than a binary switch is an effective paradigm for surfacing hidden design choices, allowing the analysis to find an optimal point between extremes.
- The use of Stiefel manifold sampling over i.i.d. spherical sampling provides practical guidance for ZO optimization: the cost is minimal (a single QR orthogonalization of i.i.d. Gaussian directions), yet it yields a strictly better privacy bound.

## Limitations & Future Work
- The combination of strong convexity/convexity + bounded domain + smoothness + per-sample Lipschitz represents a fairly strong set of assumptions; non-convex cases lack a closed-form bound.
- The lower bound for $K$ in Corollary 3.3 can be quite large as $c \to 1$ (weak convexity), meaning the practical usable interval for $K$ may be compressed, requiring mini-batches to mitigate.
- The paper lacks end-to-end LLM fine-tuning experiments (e.g., 60B models) to verify the actual budget savings of hidden-state DP in practice, relying instead on theory and synthetic curves.
- The shift sequences $\{a_t, z_t\}$ in the coupling analysis are derived from a constrained optimization problem without a closed-form solution; while a numerical accountant is provided, the tuning cost is not negligible.

## Related Work & Insights
- **vs. DP-SGD + PABI (Feldman 2018, Altschuler-Talwar 2022/2023, Chien-Li 2025)**: First-order PABI uses shifted Rényi divergence directly on the original process, relying on isotropic noise and global Lipschitz. **Ours** performs "reverse engineering" for ZO: directional noise + auxiliary process + pointwise Lipschitz, proving the crucial PABI property that $\varepsilon$ saturates with $T$.
- **vs. MeZO-DP (Zhang et al. 2024a)**: They proposed mechanism (a) (directional noise) which is empirically better for utility, but their privacy analysis was limited to composition. **Ours** provides a tighter hidden-state bound for the same update and discovers the counter-intuitive benefit of increasing $K$.
- **vs. Tang et al. 2024**: Similar setting but still uses composition; **Ours** essentially provides a free upgrade (hybrid noise + orthogonal directions) to their algorithm, maintaining utility while lowering the privacy budget.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First convergent ZO hidden-state DP bound + counter-intuitive $K$ relationship).
- Experimental Thoroughness: ⭐⭐⭐ (Large focus on theory; numerical validation limited to synthetic settings).
- Writing Quality: ⭐⭐⭐⭐ (Deconstructs the three main challenges clearly; explains why first-order PABI cannot be directly applied).
- Value: ⭐⭐⭐⭐ (Provides a tighter accountant for all DP-ZO based LLM fine-tuning, allowing for more training steps or tighter $\varepsilon$).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](../../NeurIPS2025/ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](../../NeurIPS2025/ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)

</div>

<!-- RELATED:END -->
