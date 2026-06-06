---
title: >-
  [Paper Note] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States
description: >-
  [ICML 2026][LLM Safety][Differential Privacy] The authors prove the first **convergent hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)." By designing a "directional + iso…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Differential Privacy"
  - "Zeroth-Order Optimization"
  - "PABI"
  - "hidden-state analysis"
  - "coupling analysis"
date: 2026-05-08
content_hash: 1054e16f0be6e42d
---

# Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States

**Conference**: ICML 2026  
**arXiv**: [2506.00158](https://arxiv.org/abs/2506.00158)  
**Code**: None (Theoretical paper)  
**Area**: LLM Security / Differential Privacy / Zeroth-Order Optimization  
**Keywords**: Differential Privacy, Zeroth-Order Optimization, PABI, hidden-state analysis, coupling analysis

## TL;DR
The authors prove the first **convergent hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)." By designing a "directional + isotropic" mixed noise mechanism and constructing an auxiliary process between two adjacent trajectories, they bypass the technical barrier where zeroth-order updates lack global Lipschitz continuity. This reveals a previously unknown design criterion: increasing the number of sampling directions per step $K$ actually reduces privacy loss.

## Background & Motivation

**Background**: As model sizes swell to hundreds of billions of parameters, the per-sample gradient clipping in first-order DP training methods like DP-SGD incurs massive memory overhead. Recent works such as MeZO-DP (Zhang et al. 2024a) and Tang et al. (2024) have introduced zeroth-order optimization (ZO, evaluating loss only via forward passes) into the DP framework, enabling the fine-tuning of 60B+ models with performance close to DP-LoRA. However, their privacy analysis still relies on the composition theorem—where the privacy budget accumulates linearly with the number of training steps $T$, requiring careful control of the stopping point.

**Limitations of Prior Work**: First-order DP-SGD already possesses "privacy amplification by iteration (PABI)" theory—treating intermediate iterates as hidden and releasing only the final parameters, proving that $\varepsilon$ saturates as $T \to \infty$. However, this analysis requires two components: (i) isotropic noise (to ensure controlled shifted Rényi divergence); and (ii) global Lipschitz continuity of the update mapping. Zeroth-order methods break both: the noise is a scalar Gaussian along a random direction $u$ (anisotropic), and the global Lipschitz constant of ZOGD updates across all $u$ is significantly larger than in first-order methods. Simply adding isotropic noise to the entire $\mathbb{R}^d$ could allow for existing analysis but would severely degrade the utility-privacy trade-off (noise in the $u^\perp$ direction contributes to privacy but is wasted for utility).

**Key Challenge**: There is a structural contradiction between (noise shape vs. analysis framework)—utility requires noise to be concentrated along update directions (scalar, directional), but PABI's shifted-divergence analysis requires isotropy and global Lipschitz. The problem becomes: "Can a hybrid noise be constructed to satisfy both utility and hidden-state analysis?"

**Goal**: (i) Propose a unified noisy update rule for DP-ZOGD supporting both directional and isotropic noise; (ii) Derive the first convergent hidden-state DP bound ($\varepsilon$ does not explode as $T \to \infty$); (iii) Reveal degrees of freedom in algorithm design ignored by prior literature (the role of the dimension $K$ and whether directions need to be orthogonal).

**Key Insight**: The authors observe that while zeroth-order update mappings are not globally Lipschitz, they are **pointwise Lipschitz with high probability**—the Lipschitz constant for a single fixed point relative to its neighborhood is much smaller than the global one. This provides an entry point to bypass shifted-divergence: rather than pursuing "Rényi divergence between two adjacent trajectories controlled via the original update," they explicitly construct a third auxiliary process $\widetilde W$ situated between the two adjacent trajectories, splitting the analysis into two segments.

**Core Idea**: Utilizing a "mixed noise + coupled auxiliary process" toolkit—the former addresses utility, while the latter bypasses the Lipschitz barrier, proving that ZO can also enjoy PABI-style privacy amplification.

## Method

### Overall Architecture
The authors consider the ERM problem $L(w;\mathcal D)=\frac1n\sum_i \ell_i(w)$ and run projected zeroth-order GD on a convex bounded domain $\mathcal B_R$. Each step update combines three elements: (1) Computing a two-point zeroth-order gradient $\hat g_t(w_t;u_{t,k})=\frac1n\sum_i \mathsf{clip}(\frac{\ell_i(w_t+\xi u_{t,k})-\ell_i(w_t-\xi u_{t,k})}{2\xi};\Delta)\,u_{t,k}$ using $K$ *orthogonal* directions $\{u_{t,k}\}_{k=1}^K$ (sampled uniformly from the Stiefel manifold $V_K(\mathbb R^d)$); (2) Adding a scalar Gaussian $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$ along each of these directions; (3) Adding a small isotropic Gaussian $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$ across the entire space. The mixing coefficient $\beta_t\in[0,1]$ parameterizes the "directional vs. isotropic" trade-off. $\beta_t=1$ reduces to purely directional noise (Mechanism (a) of Zhang et al. 2024a), while $\beta_t=0$ reduces to purely isotropic noise (Mechanism (b)). Analytically, an auxiliary process $\widetilde W_t$ is placed between $W_t$ and $W_t'$, applying a TV bound for $W_t\leftrightarrow \widetilde W_t$ and a Rényi bound for $\widetilde W_t\leftrightarrow W_t'$.

### Key Designs

1. **Hybrid Directional + Isotropic Noisy-ZOGD Mechanism**:
    - **Function**: Unifies common ZO noise mechanisms (a)/(b) into a continuous family, allowing hidden-state analysis to find a tighter bound than pure (a).
    - **Mechanism**: $w_{t+1}=\Pi_{\mathcal B_R}[w_t-\frac{\eta}{K}\sum_k \hat g_t(w_t;u_{t,k})+\frac{\eta}{\sqrt K}\sum_k G_{t,k}^{(1)} u_{t,k}+\frac{\eta}{\sqrt d}G_t^{(2)}]$, where $\{u_{t,k}\}$ are orthogonal, $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$, and $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$. This parameterization maintains equivalent total noise variance for all $\beta_t, K$, thus the utility upper bound remains unchanged.
    - **Design Motivation**: The directional part $G_{t,k}^{(1)} u_{t,k}$ is utility-friendly (noise falls on directions carrying privacy-sensitive information), while the isotropic part $G_t^{(2)}$ provides "maneuvering space" for the shifted Gaussian mechanism—the vector shift $v_t$ in the coupling analysis is absorbed by this isotropic noise component.

2. **Coupling Auxiliary Process $\widetilde W$ to Bypass Global Lipschitz**:
    - **Function**: Reconstructs hidden-state DP analysis from "shifted-divergence along original trajectories" to a two-part "TV(coupling failure) + Rényi(shifted Gaussian)" approach, requiring only pointwise Lipschitz continuity.
    - **Mechanism**: For two adjacent trajectories $W_t, W_t'$ (corresponding to data $\mathcal D, \mathcal D'$), a third process $\widetilde W$ is constructed. From some time $\tau$, $\widetilde W_{t+1}\stackrel{d}{=}\Pi_{\mathcal B_R}[\hat\psi_t(\widetilde W_t)+Y_t+Z_t+v_t]$, where the shift $v_t:=\min(a_t,(\|d_t\|-z_{t+1})_+)\frac{d_t}{\|d_t\|}$ and $d_t:=\hat\psi_t(W_t)-\hat\psi_t(\widetilde W_t)$. The analysis splits: (i) the TV distance between $W$ and $\widetilde W$ is controlled by the high-probability pointwise Lipschitz constant from Lemma 3.6 (failure event probability $\delta_f$); (ii) the Rényi divergence between $\widetilde W$ and $W'$ follows standard PABI accumulation via the shifted Gaussian mechanism. Finally, Lemma 3.7 is used to bound the forward $W_\infty$ distance $W_\infty(w_t,w_t')\le \min(2R,2\eta\Delta t/\sqrt K)$.
    - **Design Motivation**: The Lipschitz constant $c_1=\sqrt{1-\sum_k\upsilon_k+c^2\sum_k\gamma_k}$ for ZO updates involves random variables $\upsilon_k,\gamma_k\sim\mathsf{Beta}(K/2,(d-K)/2)$, making almost sure global $\le c$ impossible. However, Beta distribution tail bounds prove $c_1\le \bar c_1=\sqrt{1-(1-c^2)K/d+\vartheta(1+c^2)K/d}$ holds with high probability. Separating the "bad event" into the TV term and "good event" into the Rényi term allows for a "divide and conquer" strategy.

3. **Privacy Gains from Orthogonal Update Directions + Multi-dimensional $K$**:
    - **Function**: Provides a design criterion absent in previous literature—selecting $K>1$ with orthogonal directions simultaneously reduces privacy loss and utility error.
    - **Mechanism**: Theorem 3.2 / Corollary 3.3 provide a closed-form DP bound $\varepsilon=O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$. Crucially, $1/K$ appears in the $\min$ term: as $T\to\infty$ and the bound saturates, $\varepsilon$ is inversely proportional to $K$. Furthermore, Lemma 3.6 shows the Beta distribution tail is tighter under orthogonal $\{u_{t,k}\}$ relative to independent uniform sampling.
    - **Design Motivation**: Under standard composition, increasing $K$ degrades privacy by $\tilde O(\sqrt K)$ (exposing more sensitive directions), leading previous works to avoid large $K$. Hidden-state analysis changes the rules—under fixed utility constraints, using more directions "dilutes" the sensitivity of each direction. Orthogonal directions further eliminate overlapping privacy contributions. This phenomenon is unique to ZO and absent in first-order methods.

### Loss & Training
The work does not modify training objectives; it provides a tighter privacy accountant for existing ZO DP optimizers. Specifically: (i) Set $\eta\le K/M$ (strongly convex) or $\le 2K/M$ (convex); (ii) Set $\xi\le 2\Delta/(n\eta M\sqrt{2d})$; (iii) $K$ satisfies $\max(20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil MRn\sqrt{2d}/\Delta\rceil),1)\le K\le d/2$. For non-convex cases, a numerical accountant is provided instead of a closed form.

## Key Experimental Results

### Main Results
The paper is primarily theoretical. Numerical validation focuses on Figure 1: comparing the hidden-state bound, standard composition, and output perturbation $\varepsilon$ curves over $T$ under a smooth strongly convex loss and bounded domain.

| Method | Trend of $\varepsilon$ with $T$ | Notes |
|------|--------------------------|------|
| Standard composition (Theorem 3.1, $\beta_t=1$) | $O(\sqrt{\Delta^2\log(1/\delta)T/(n^2\sigma^2)})$ | Grows unboundedly with $T$ |
| Output perturbation | $O(\sqrt{R^2\log(1/\delta)/\sigma^2})$ | Independent of $T$, but constant is large |
| **Hidden-state DP (Ours, Corollary 3.3)** | $O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$ | Saturates over $T$, inverse to $K$ |

For $K \ge K_{\min}$ (the lower bound given in Corollary 3.3), our bound is strictly superior to standard composition and output perturbation in the medium-to-large $T$ regime.

### Ablation Study

| Configuration | Key Conclusion | Explanation |
|------|---------|------|
| $\beta_t=1$ (Pure directional) | Best utility, most difficult analysis | Mechanism (a) from prior work |
| $\beta_t=0$ (Pure isotropic) | Easiest analysis, worst utility | Mechanism (b) from prior work |
| $\beta_t\in(0,1)$ | Improved utility-privacy trade-off | Sweet spot for mixed scheme |
| $K=1$, i.i.d. spherical | Classic ZO configuration | Loose privacy analysis |
| $K>1$, Orthogonal directions | Tighter privacy bound, better convergence | Novel discovery of this work |

### Key Findings
- The first convergent hidden-state DP bound reveals that $\varepsilon$ saturates to the order of $O(MRn\sqrt d/(K\Delta))$ as $T\to\infty$—independent of additional training steps and determined by domain radius, Lipschitz constants, and direction count.
- Increasing the sampling direction count $K$ per step reduces privacy loss under hidden-state analysis—contrary to prior works (standard composition) which suggest larger $K$ worsens privacy. This is a vital algorithmic insight gained by bringing "PABI ideas" to ZO.
- Using orthogonal $\{u_{t,k}\}$ (Stiefel manifold sampling) instead of i.i.d. spherical sampling further reduces privacy by eliminating redundant sensitivity leaks between directions.

## Highlights & Insights
- "Constructing an auxiliary process between two adjacent trajectories to split the analysis into TV + Rényi segments" is a highly versatile coupling analysis trick. As long as the original process lacks global Lipschitz but possesses pointwise Lipschitz continuity, this technique can be applied (e.g., SGD-on-manifold, Langevin dynamics on non-convex landscapes).
- Expressing the noise mechanism as a continuous family $\beta_t\in[0,1]$ rather than a binary switch brings implicit design choices to light, allowing for the discovery of optimal points between extremes.
- "Sampling directions from the Stiefel manifold is tighter than i.i.d. spherical" is a practical guide for ZO optimization with low implementation cost (requiring only QR orthogonalization of i.i.d. Gaussian directions) but yielding a direct improvement in the privacy bound.

## Limitations & Future Work
- The combination of strong convexity/convexity + bounded domain + smoothness + per-sample Lipschitz is a strong set of assumptions; non-convex cases lack a closed-form bound.
- The lower bound for $K$ in Corollary 3.3 can be quite large as $c\to 1$ (weak convexity), meaning the practical range for $K$ might be compressed, necessitating mini-batching to mitigate this.
- The paper lacks end-to-end experimental validation on LLMs (fine-tuning 60B models) to show actual budget savings from hidden-state DP in practice, providing only theoretical and numerical curves.
- The shift sequence $\{a_t,z_t\}$ in the coupling analysis stems from a constrained optimization problem without a closed-form solution, making the numerical accountant somewhat costly to tune.

## Related Work & Insights
- **vs. DP-SGD + PABI (Feldman 2018, Altschuler-Talwar 2022/2023, Chien-Li 2025)**: First-order PABI uses shifted Rényi divergence directly on the original process, relying on isotropic noise + global Lipschitz. This work reverse-engineers ZO: directional noise + auxiliary process + pointwise Lipschitz to prove the same saturation of $\varepsilon$.
- **vs. MeZO-DP (Zhang et al. 2024a)**: They proposed mechanism (a) (directional noise) with better empirical utility but limited privacy analysis to composition. This work provides a tighter hidden-state bound for the same update and discovers the counter-intuitive benefit of larger $K$.
- **vs. Tang et al. 2024**: Similar setting but restricted to composition; this work essentially provides a "free upgrade" (mixed noise + orthogonal directions) to their algorithm, maintaining utility while reducing the privacy budget.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergent hidden-state DP bound for ZO + counter-intuitive "$K$ effect." The analysis technique is also original.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper; numerical validation is limited to synthetic settings and Figure 1 curve comparisons without large-scale LLM validation.
- Writing Quality: ⭐⭐⭐⭐ Deconstructs three challenges (noise shape, Lipschitz, composition vs. hidden-state) clearly.
- Value: ⭐⭐⭐⭐ Provides a tighter accountant for all DP-ZO based model fine-tuning, allowing for more training steps or tighter $\varepsilon$ for the same steps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/llm_safety/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
