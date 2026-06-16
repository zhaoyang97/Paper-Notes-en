---
title: >-
  [Paper Note] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States
description: >-
  [ICML 2026][LLM Safety][PABI] The authors prove the first **convergent hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)". By designing a "directional + isotropic" mixed noise mechanism and constructing an auxiliary process between two adjacent trajectories, they bypass the technical obstacle of lacki
tags:
  - ICML 2026
  - LLM Safety
  - PABI
date: 2026-05-08
content_hash: c7bdd2c965cdb342
---
# Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States

**Conference**: ICML 2026  
**arXiv**: [2506.00158](https://arxiv.org/abs/2506.00158)  
**Code**: None (Theoretical paper)  
**Area**: LLM Security / Differential Privacy / Zeroth-Order Optimization  
**Keywords**: Differential Privacy, Zeroth-Order Optimization, PABI, hidden-state analysis, coupling analysis

## TL;DR
The authors prove the first **convergent hidden-state DP upper bound** for "Differentially Private Zeroth-Order Gradient Descent (DP-ZOGD)". By designing a "directional + isotropic" mixed noise mechanism and constructing an auxiliary process between two adjacent trajectories, they bypass the technical obstacle of lacking global Lipschitz continuity in zeroth-order updates. This reveals a previously unknown DP algorithm design principle: "increasing the number of sampling directions $K$ per step actually reduces privacy loss."

## Background & Motivation

**Background**: As model sizes expand to tens or hundreds of billions of parameters, the per-sample gradient clipping in first-order DP training methods like DP-SGD incurs massive memory overhead. Recent works such as MeZO-DP (Zhang et al. 2024a) and Tang et al. (2024) have introduced Zeroth-Order (ZO, using only forward passes to evaluate loss) optimization into the DP framework, enabling fine-tuning of 60B+ models with performance close to DP-LoRA. However, their privacy analysis still relies on the composition theorem—where the privacy budget accumulates linearly with the number of training steps $T$, necessitating careful control of the stopping point.

**Limitations of Prior Work**: First-order DP-SGD already possesses "privacy amplification by iteration (PABI)" theory—treating intermediate iterates as hidden and only releasing the final parameters, proving that $\varepsilon$ saturates as $T$ increases. However, this analysis requires two elements: (i) isotropic noise (to ensure shifted Rényi divergence is controllable) and (ii) global Lipschitz continuity of the update mapping. Zeroth-order methods violate both—the noise is scalar Gaussian along a random direction $u$, making it anisotropic; and the global Lipschitz constant of the ZOGD update mapping across all $u$ is much larger than that of first-order methods. Adding isotropic noise directly to the entire $\mathbb{R}^d$ could allow the use of existing analysis but would severely degrade the utility-privacy trade-off (noise added in the $u^\perp$ direction contributes to privacy but entirely wastes utility).

**Key Challenge**: There is a structural contradiction between the noise shape and the analysis framework—utility requires noise to be aligned with the update direction (scalar, directional), but the shifted-divergence analysis of PABI requires isotropy and global Lipschitz. The problem becomes: "Can a mixed noise be designed to satisfy both utility and hidden-state analysis?"

**Goal**: (i) Propose a unified noisy update rule for DP-ZOGD that supports both directional and isotropic noise; (ii) derive the first convergent hidden-state DP bound (where $\varepsilon$ does not explode as $T\to\infty$); (iii) reveal algorithmic design freedoms ignored in previous literature (the role of the update dimension $K$ and whether directions need to be orthogonal).

**Key Insight**: The authors observe that while the zeroth-order update mapping is not globally Lipschitz, it is **pointwise Lipschitz with high probability**—for a single fixed point relative to nearby points, the Lipschitz constant is much smaller than the global one. This provides an entry point to bypass shifted divergence: instead of pursuing "controllability of Rényi divergence between two adjacent trajectories under the original update," they explicitly construct a third auxiliary process $\widetilde W$ "between" the two adjacent trajectories, splitting the analysis into two segments.

**Core Idea**: Use a "mixed noise + coupled auxiliary process" toolkit—the former addresses the utility issue, while the latter bypasses the Lipschitz obstacle, thereby proving that ZO can also enjoy PABI-style privacy amplification.

## Method

### Overall Architecture
The authors perform projected zeroth-order GD on an ERM problem $L(w;\mathcal D)=\frac1n\sum_i \ell_i(w)$ over a convex bounded domain $\mathcal B_R$, aiming to prove a privacy upper bound for this forward-pass-only optimizer that does not explode as training steps $T$ increase. Each update step consists of three parts: first, calculate a two-point zeroth-order gradient using $K$ orthogonal directions $\{u_{t,k}\}_{k=1}^K$ (uniformly sampled from the Stiefel manifold $V_K(\mathbb R^d)$); second, add scalar Gaussian noise along each of these directions; and finally, add a small amount of isotropic Gaussian noise to the full space. The ratio of the two noises is controlled by a continuous knob $\beta_t\in[0,1]$ (directional vs. isotropic). For the analysis, a third auxiliary trajectory $\widetilde W_t$ is explicitly inserted between two adjacent trajectories $W_t,W_t'$, splitting the privacy analysis into a TV segment for $W_t\leftrightarrow\widetilde W_t$ and a Rényi segment for $\widetilde W_t\leftrightarrow W_t'$, bypassing the lack of global Lipschitz in zeroth-order updates.

### Key Designs

**1. Mixed Directional + Isotropic Noisy-ZOGD Mechanism: Unifying Two Old Schemes with a Continuous Knob**

Prior zeroth-order DP literature contained only two disjoint noise mechanisms: mechanism (a), which adds scalar noise along the update direction (good utility but hard to analyze), and mechanism (b), which adds isotropic noise to the full space (easy to analyze but poor utility). The authors parameterize these into a single continuous family, writing the update as:  
$w_{t+1}=\Pi_{\mathcal B_R}[w_t-\frac{\eta}{K}\sum_k \hat g_t(w_t;u_{t,k})+\frac{\eta}{\sqrt K}\sum_k G_{t,k}^{(1)} u_{t,k}+\frac{\eta}{\sqrt d}G_t^{(2)}]$  
where the two-point zeroth-order gradient is $\hat g_t(w_t;u_{t,k})=\frac1n\sum_i \mathsf{clip}(\frac{\ell_i(w_t+\xi u_{t,k})-\ell_i(w_t-\xi u_{t,k})}{2\xi};\Delta)\,u_{t,k}$, directional noise $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$, and isotropic noise $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$. Directions $\{u_{t,k}\}$ are chosen to be orthogonal rather than i.i.d. uniform on $\mathbb S^{d-1}$. Setting $\beta_t=1$ reduces to mechanism (a), and $\beta_t=0$ reduces to mechanism (b), with intermediate values being a hybrid. This parameterization maintains an equivalent total noise variance across all $\beta_t$ and $K$, so the utility bound remains unchanged while adjusting $\beta_t$ shifts the degrees of freedom on the privacy side. Both parts are necessary: the directional part concentrates noise on directions carrying privacy-sensitive information (utility-friendly), while the isotropic part provides "space" for the vector shift $v_t$ in the subsequent coupling analysis—without this isotropic noise, the shifted Gaussian mechanism segment would have no foundation.

**2. Coupled Auxiliary Process $\widetilde W$: Bypassing Global Lipschitz with Pointwise Lipschitz**

The Lipschitz constant $c_1=\sqrt{1-\sum_k\upsilon_k+c^2\sum_k\gamma_k}$ of the zeroth-order update mapping involve random variables $\upsilon_k,\gamma_k\sim\mathsf{Beta}(K/2,(d-K)/2)$, making it almost impossible to satisfy global Lipschitz $\le c$. Consequently, the standard PABI path—"shifted Rényi divergence controllable along original trajectories"—is blocked. The authors' solution is to insert an auxiliary trajectory $\widetilde W$ between adjacent trajectories $W_t,W_t'$ (corresponding to datasets $\mathcal D,\mathcal D'$), evolving from some time $\tau$ according to $\widetilde W_{t+1}\stackrel{d}{=}\Pi_{\mathcal B_R}[\hat\psi_t(\widetilde W_t)+Y_t+Z_t+v_t]$, where the shift is defined as $v_t:=\min(a_t,(\|d_t\|-z_{t+1})_+)\frac{d_t}{\|d_t\|}$ and $d_t:=\hat\psi_t(W_t)-\hat\psi_t(\widetilde W_t)$. This insertion splits the analysis into two segments: the TV distance between $W$ and $\widetilde W$ only requires the high-probability pointwise Lipschitz property from Lemma 3.6 (bad event probability $\delta_f$), while the segment between $\widetilde W$ and $W'$ follows the classic shifted Gaussian mechanism with Rényi divergence accumulating according to standard PABI. Finally, Lemma 3.7 uses forward tracking $W_\infty(w_t,w_t')\le \min(2R,2\eta\Delta t/\sqrt K)$ to close the bound. Essentially, this is "divide and conquer": throw bad events where global Lipschitz fails into the TV term, and handle good events via the Rényi term—it suffices that $c_1\le \bar c_1=\sqrt{1-(1-c^2)K/d+\vartheta(1+c^2)K/d}$ holds with high probability, which is proven using Beta tail bounds.

**3. Privacy Gains from Orthogonal Directions + Multi-dimensional $K$: More Directions Equal Better Privacy**

Combining these toolkit elements, the authors obtain a closed-form DP upper bound in Theorem 3.2 / Corollary 3.3: $\varepsilon=O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$. From this, they derive a design rule previously absent in the literature: the term $MRn\sqrt d/(K\Delta)$ in the $\min$ is inversely proportional to the number of directions $K$. Thus, as $T\to\infty$ and the bound saturates, $\varepsilon$ decreases as $K$ increases. This contradicts the intuition from standard composition, which suggests that every additional direction exposes more sensitivity, worsening privacy by $\tilde O(\sqrt K)$. Hidden-state analysis changes the rules: under a fixed utility constraint, using more directions dilutes the sensitivity across each direction. Choosing orthogonal directions (sampling from the Stiefel manifold, which only requires one QR decomposition of i.i.d. Gaussian directions) further tightens the bound—the Beta tail bound in Lemma 3.6 is tighter under orthogonal directions than i.i.d. spherical sampling because there is no overlapping privacy leakage between orthogonal directions. This is a phenomenon unique to zeroth-order methods and not seen in first-order methods.

### Loss & Training
The training objective is not modified—this work provides a tighter privacy accountant for existing ZO DP optimizers. Specifically: (i) choose $\eta\le K/M$ (strongly convex) or $\le 2K/M$ (convex); (ii) choose $\xi\le 2\Delta/(n\eta M\sqrt{2d})$; (iii) $K$ must satisfy $\max(20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil MRn\sqrt{2d}/\Delta\rceil),1)\le K\le d/2$. For non-convex cases, a numerical accountant is provided instead of a closed-form solution.

## Key Experimental Results

### Main Results
The paper is primarily theoretical, with numerical validation focused on Figure 1: comparing the hidden-state bound, standard composition, and output perturbation $\varepsilon$ curves over $T$ on a smooth strongly convex loss with a bounded domain.

| Method | Trend of $\varepsilon$ with $T$ | Remarks |
|------|--------------------------|------|
| Standard composition (Theorem 3.1, $\beta_t=1$) | $O(\sqrt{\Delta^2\log(1/\delta)T/(n^2\sigma^2)})$ | Grows unboundedly with $T$ |
| Output perturbation | $O(\sqrt{R^2\log(1/\delta)/\sigma^2})$ | Independent of $T$, but with a large constant |
| **Hidden-state DP (Ours, Corollary 3.3)** | $O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$ | Saturates with $T$, inversely proportional to $K$ |

For $K\ge K_{\min}$ (the lower bound given in Corollary 3.3), the proposed hidden-state bound is strictly superior to standard composition and output perturbation in the medium-to-large $T$ regime.

### Ablation Study

| Configuration | Key Conclusion | Description |
|------|---------|------|
| $\beta_t=1$ (Fully directional) | Best utility but hardest to analyze | Previous mechanism (a) |
| $\beta_t=0$ (Fully isotropic) | Easy to analyze but poor utility | Previous mechanism (b) |
| $\beta_t\in(0,1)$ | Improved utility-privacy trade-off | The sweet spot of the proposed hybrid scheme |
| $K=1$, i.i.d. spherical directions | Classic ZO configuration | Loose privacy analysis |
| $K>1$, orthogonal directions | Tighter privacy bound, better convergence rate | New discovery of this paper |

### Key Findings
- The first convergent hidden-state DP bound reveals that $\varepsilon$ saturates to the scale of $O(MRn\sqrt d/(K\Delta))$ as $T\to\infty$—independent of the number of additional training steps, depending only on the domain radius, Lipschitz constant, and number of directions.
- Increasing the number of sampling directions $K$ per step reduces privacy loss in the hidden-state setting—contradicting prior work (standard composition) which suggested larger $K$ worsens privacy. This is a major algorithmic insight from bringing "PABI ideas" to ZO.
- Replacing i.i.d. spherical sampling with orthogonal $\{u_{t,k}\}$ (Stiefel manifold sampling) further reduces privacy loss by eliminating redundant leakage between directions.

## Highlights & Insights
- "Constructing an auxiliary process between two adjacent trajectories to split the analysis into TV + Rényi segments" is a highly versatile coupling analysis trick—applicable to any process lacking global Lipschitz but possessing pointwise Lipschitz. This could be used for hidden-state analysis of SGD-on-manifold or Langevin dynamics in non-convex landscapes.
- Treating the noise mechanism as a continuous family $\beta_t\in[0,1]$ rather than a binary switch is itself a paradigm of "bringing hidden design choices to the surface," allowing the analysis to find the optimal point between two extremes.
- Using Stiefel manifold sampling instead of i.i.d. spherical sampling is a practical guide for ZO optimization with low implementation cost (just one QR orthogonalization of i.i.d. Gaussian directions) that yields a better privacy bound directly.

## Limitations & Future Work
- The combination of strong convexity/convexity + bounded domain + smoothness + per-sample Lipschitz is a fairly strong set of assumptions; closed-form solutions are absent for the non-convex case, although a numerical accountant is provided.
- The lower bound for $K$ in Corollary 3.3, $K\ge 20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil\cdot\rceil)$, can be quite large as $c\to 1$ (weakly convex), meaning the practical usable range for $K$ might be compressed, necessitating mini-batches to alleviate this.
- The paper lacks end-to-end experimental validation of actual budget savings from hidden-state DP in practical LLM fine-tuning (e.g., 60B models), providing only theory and numerical curves; large-scale experiments on an improved MeZO-DP version would be more compelling.
- The shift sequences $\{a_t,z_t\}$ in the coupling analysis are derived from a constrained optimization problem without a closed-form solution; while a numerical accountant is provided, the tuning cost is not low.

## Related Work & Insights
- **vs. DP-SGD + PABI (Feldman 2018, Altschuler-Talwar 2022/2023, Chien-Li 2025)**: First-order PABI uses shifted Rényi divergence directly on the original process, relying on isotropic noise and global Lipschitz. This paper performs reverse engineering for ZO: directional noise + auxiliary process + pointwise Lipschitz, equivalently proving the critical PABI property that "$\varepsilon$ saturates with $T$."
- **vs. MeZO-DP (Zhang et al. 2024a)**: They proposed mechanism (a) (adding noise along the direction), which is empirically better for utility, but their privacy analysis stopped at composition. This paper provides a tighter hidden-state bound for the same update and discovers the counter-intuitive benefit of larger $K$.
- **vs. Tang et al. 2024**: Similar setting but still uses composition; this work is effectively a free upgrade (mixed noise + orthogonal directions) to their algorithm, maintaining utility while reducing the privacy budget.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergent hidden-state DP bound for ZO + counter-intuitive "$K$ role," analysis techniques are also new.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper, numerical validation limited to synthetic settings + curve comparisons in Figure 1, lacks end-to-end LLM validation.
- Writing Quality: ⭐⭐⭐⭐ Deconstructs the three challenges (noise shape, Lipschitz, composition vs. hidden-state) one by one; the introduction clearly explains why first-order PABI cannot be directly applied.
- Value: ⭐⭐⭐⭐ Provides a tighter accountant for all researchers taking the DP-ZO path for LLM fine-tuning, allowing for more training steps under the same $\varepsilon$ or a tighter $\varepsilon$ for the same steps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/llm_safety/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
