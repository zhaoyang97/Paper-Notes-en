---
title: >-
  [Paper Note] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States
description: >-
  [ICML 2026][LLM Safety][Differential Privacy] The authors provide the first **convergent hidden-state DP upper bound** for "differentially private zeroth-order optimization (DP-ZOGD)"—by designing a "directional + isotro…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Differential Privacy"
  - "Zeroth-Order Optimization"
  - "PABI"
  - "hidden-state analysis"
  - "coupling analysis"
date: 2026-05-08
content_hash: 440a20ba07325845
---

# Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States

**Conference**: ICML 2026  
**arXiv**: [2506.00158](https://arxiv.org/abs/2506.00158)  
**Code**: None (theoretical paper)  
**Area**: LLM Security / Differential Privacy / Zeroth-Order Optimization  
**Keywords**: Differential Privacy, Zeroth-Order Optimization, PABI, hidden-state analysis, coupling analysis

## TL;DR
The authors provide the first **convergent hidden-state DP upper bound** for "differentially private zeroth-order optimization (DP-ZOGD)"—by designing a "directional + isotropic" hybrid noise mechanism and constructing an auxiliary process between two neighboring trajectories, they circumvent the technical barrier that zeroth-order updates lack global Lipschitzness. This reveals a previously unknown DP algorithmic principle: **increasing the number of sampled directions per step $K$ can actually reduce privacy loss**.

## Background & Motivation

**Background**: As model sizes balloon to tens or hundreds of billions of parameters, per-sample gradient clipping in first-order DP training methods like DP-SGD incurs huge memory overhead. Recent works such as MeZO-DP (Zhang et al. 2024a) and Tang et al. (2024) have brought zeroth-order optimization (ZO, using only forward passes to evaluate loss) into the DP framework, enabling fine-tuning of 60B+ models with performance close to DP-LoRA. However, their privacy analyses still rely on the composition theorem—privacy budget accumulates linearly with training steps $T$, necessitating careful control of the stopping point.

**Limitations of Prior Work**: First-order DP-SGD already has the "privacy amplification by iteration (PABI)" theory—by treating intermediate iterates as hidden and only releasing the final parameter, it can be shown that $\varepsilon$ saturates with $T$. This analysis requires two things: (i) isotropic noise (to control shifted Rényi divergence); (ii) globally Lipschitz update mapping. Zeroth-order methods break both: noise is only along random direction $u$ as scalar Gaussian, i.e., anisotropic; the global Lipschitz constant of ZOGD update mapping over all $u$ is much larger than first-order. Adding isotropic noise over the entire $\mathbb{R}^d$ allows existing analysis but severely worsens the utility-privacy trade-off (noise added in $u^\perp$ directions does not contribute to privacy but wastes utility).

**Key Challenge**: There is a structural conflict between (noise shape vs analysis framework)—utility requires noise to be as directional as possible (scalar, directional), but PABI's shifted-divergence analysis requires isotropy + global Lipschitzness. The problem thus becomes: "Can a hybrid noise be constructed to satisfy both utility and hidden-state analysis?"

**Goal**: (i) Propose a unified noisy update rule for DP-ZOGD that supports both directional and isotropic noise; (ii) derive the first convergent hidden-state DP bound (where $\varepsilon$ does not blow up as $T\to\infty$); (iii) reveal previously overlooked algorithmic degrees of freedom (the role of update dimension $K$, whether directions need to be orthogonal).

**Key Insight**: The authors observe that although the zeroth-order update mapping is not globally Lipschitz, it is **pointwise high-probability Lipschitz**—for a fixed point and its neighborhood, the Lipschitz constant is much smaller than the global one. This provides a way to circumvent shifted-divergence: instead of controlling the Rényi divergence between two neighboring trajectories along the original update, explicitly construct a third auxiliary process $\widetilde W$ "between" the two, splitting the analysis into two segments.

**Core Idea**: Use a "hybrid noise + coupled auxiliary process"—the former solves the utility issue, the latter bypasses the Lipschitz barrier, thus proving that ZO can also enjoy PABI-style privacy amplification.

## Method

### Overall Architecture
The authors consider the ERM problem $L(w;\mathcal D)=\frac1n\sum_i \ell_i(w)$, running projected zeroth-order GD over a convex bounded domain $\mathcal B_R$. Each update step consists of three parts: (1) Compute two-point zeroth-order gradients using $K$ *orthogonal* directions $\{u_{t,k}\}_{k=1}^K$ (sampled uniformly from the Stiefel manifold $V_K(\mathbb R^d)$): $\hat g_t(w_t;u_{t,k})=\frac1n\sum_i \mathsf{clip}(\frac{\ell_i(w_t+\xi u_{t,k})-\ell_i(w_t-\xi u_{t,k})}{2\xi};\Delta)\,u_{t,k}$; (2) Add a scalar Gaussian $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$ along each direction; (3) Add a small isotropic Gaussian $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$ over the whole space. The mixing coefficient $\beta_t\in[0,1]$ parameterizes the "directional vs isotropic" trade-off: $\beta_t=1$ degenerates to only directional noise (i.e., mechanism (a) of Zhang et al. 2024a), $\beta_t=0$ to fully isotropic (mechanism (b)). On the analysis side, an auxiliary $\widetilde W_t$ between $W_t,W_t'$ is constructed, so that the TV bound goes via $W_t\leftrightarrow \widetilde W_t$, and the Rényi bound via $\widetilde W_t\leftrightarrow W_t'$.

### Key Designs

1. **Hybrid Directional + Isotropic Noisy-ZOGD Mechanism**:

    - **Function**: Unifies the two common ZO noise mechanisms (a)/(b) into a continuous family, allowing hidden-state analysis to find a tighter bound than pure (a).
    - **Mechanism**: $w_{t+1}=\Pi_{\mathcal B_R}[w_t-\frac{\eta}{K}\sum_k \hat g_t(w_t;u_{t,k})+\frac{\eta}{\sqrt K}\sum_k G_{t,k}^{(1)} u_{t,k}+\frac{\eta}{\sqrt d}G_t^{(2)}]$, where $\{u_{t,k}\}$ are orthogonal (not i.i.d. uniform on $\mathbb S^{d-1}$), $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$, $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$. This parameterization keeps the total noise variance equivalent for all $\beta_t,K$, so the utility upper bound is unchanged.
    - **Design Motivation**: The directional part $G_{t,k}^{(1)} u_{t,k}$ is utility-friendly (noise only in directions carrying privacy-sensitive information), while the isotropic part $G_t^{(2)}$ provides "room" for the shifted Gaussian mechanism in the coupling analysis—the vector shift $v_t$ is absorbed by this isotropic noise.

2. **Coupled Auxiliary Process $\widetilde W$ to Bypass Global Lipschitzness**:

    - **Function**: Reconstructs hidden-state DP analysis from "shifted-divergence along the original trajectory" into "TV (coupling failure) + Rényi (shifted Gaussian)" two-stage analysis, requiring only pointwise Lipschitzness.
    - **Mechanism**: For two neighboring trajectories $W_t,W_t'$ (corresponding to datasets $\mathcal D,\mathcal D'$), construct a third $\widetilde W$, starting from some time $\tau$, with $\widetilde W_{t+1}\stackrel{d}{=}\Pi_{\mathcal B_R}[\hat\psi_t(\widetilde W_t)+Y_t+Z_t+v_t]$, where shift $v_t:=\min(a_t,(\|d_t\|-z_{t+1})_+)\frac{d_t}{\|d_t\|}$, $d_t:=\hat\psi_t(W_t)-\hat\psi_t(\widetilde W_t)$. The analysis is split: (i) TV distance between $W$ and $\widetilde W$ is controlled by Lemma 3.6's high-probability pointwise Lipschitz (bad event probability $\delta_f$); (ii) Between $\widetilde W$ and $W'$ is the classic shifted Gaussian mechanism, with Rényi divergence accumulated as in standard PABI. Lemma 3.7's forward $W_\infty$ tracking $W_\infty(w_t,w_t')\le \min(2R,2\eta\Delta t/\sqrt K)$ is used to close the argument.
    - **Design Motivation**: The Lipschitz constant $c_1=\sqrt{1-\sum_k\upsilon_k+c^2\sum_k\gamma_k}$ for zeroth-order updates involves random variables $\upsilon_k,\gamma_k\sim\mathsf{Beta}(K/2,(d-K)/2)$, so almost sure global $c_1\le c$ cannot be guaranteed. However, using Beta tail bounds, $c_1\le \bar c_1=\sqrt{1-(1-c^2)K/d+\vartheta(1+c^2)K/d}$ holds with high probability. Bad events are absorbed into the TV term, good events go into the Rényi term—divide and conquer.

3. **Privacy Gains from Orthogonal Update Directions and Multi-dimensional $K$**:

    - **Function**: Provides a previously unrecognized algorithmic guideline—choosing $K>1$ and orthogonal directions can simultaneously reduce privacy loss and utility error.
    - **Mechanism**: Theorem 3.2 / Corollary 3.3 give a closed-form DP upper bound $\varepsilon=O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$. The key is the $1/K$ in the $\min$: as $T\to\infty$ and the bound saturates, $\varepsilon$ is inversely proportional to $K$. The Beta tail bound Lemma 3.6 is also tighter for orthogonal $\{u_{t,k}\}$ (i.e., Stiefel manifold sampling) than for i.i.d. spherical sampling.
    - **Design Motivation**: Under standard composition, increasing $K$ worsens privacy as $\tilde O(\sqrt K)$ (since more sensitive directions are exposed), so prior work avoided large $K$. Hidden-state analysis changes the game—under fixed utility constraints, using more directions dilutes sensitivity per direction. Orthogonal directions further eliminate redundant privacy contributions. This is unique to ZO and not seen in first-order methods.

### Loss & Training

No modification to training objectives—the paper provides a tighter privacy accountant for existing ZO DP optimizers. Specifically: (i) choose $\eta\le K/M$ (strongly convex) or $\le 2K/M$ (convex); (ii) choose $\xi\le 2\Delta/(n\eta M\sqrt{2d})$; (iii) $K$ satisfies $\max(20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil MRn\sqrt{2d}/\Delta\rceil),1)\le K\le d/2$. For non-convex cases, a numerical accountant is provided instead of a closed form.

## Key Experimental Results

### Main Results
This is a theory-focused paper; numerical validation is concentrated in Figure 1: comparing the $\varepsilon$ curves of hidden-state bound, standard composition, and output perturbation as $T$ increases, under smooth strongly convex loss + bounded domain.

| Method | $\varepsilon$ Trend with $T$ | Remarks |
|--------|------------------------------|---------|
| Standard composition (Theorem 3.1, $\beta_t=1$) | $O(\sqrt{\Delta^2\log(1/\delta)T/(n^2\sigma^2)})$ | Grows unbounded with $T$ |
| Output perturbation | $O(\sqrt{R^2\log(1/\delta)/\sigma^2})$ | Independent of $T$, but large constant |
| **Hidden-state DP (this work, Corollary 3.3)** | $O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$ | Saturates with $T$, inversely proportional to $K$ |

For $K\ge K_{\min}$ (lower bound from Corollary 3.3), the bound in this work strictly outperforms standard composition and output perturbation for moderate to large $T$.

### Ablation Study

| Configuration | Key Conclusion | Notes |
|---------------|---------------|-------|
| $\beta_t=1$ (fully directional) | Best utility but hardest to analyze | Old method mechanism (a) |
| $\beta_t=0$ (fully isotropic) | Easy to analyze but poor utility | Old method mechanism (b) |
| $\beta_t\in(0,1)$ | Improved utility-privacy trade-off | Sweet spot of this hybrid scheme |
| $K=1$, i.i.d. spherical directions | Classic ZO setup | Loose privacy analysis |
| $K>1$, orthogonal directions | Tighter privacy bound, better convergence | New finding of this work |

### Key Findings
- The first convergent hidden-state DP bound shows: as $T\to\infty$, $\varepsilon$ saturates at $O(MRn\sqrt d/(K\Delta))$—independent of "how many more steps are trained," depending only on domain radius, Lipschitz constant, and number of directions.
- Increasing the number of sampled directions per step $K$ actually reduces privacy loss under hidden-state analysis—contrary to prior work (standard composition), which believed larger $K$ worsens privacy. This is a key algorithmic insight in bringing "PABI thinking" to ZO.
- Using orthogonal $\{u_{t,k}\}$ (Stiefel manifold sampling) instead of i.i.d. spherical sampling further reduces privacy loss, as there is no redundant leakage between orthogonal directions.

## Highlights & Insights
- "Constructing an auxiliary process between two neighboring trajectories and splitting the analysis into TV + Rényi" is a highly general coupling analysis technique—whenever the original process lacks global Lipschitzness but has pointwise Lipschitzness, this can be adopted. It is also applicable to SGD-on-manifold, hidden-state analysis of Langevin in non-convex landscapes, etc.
- Writing the noise mechanism as a continuous family $\beta_t\in[0,1]$ rather than a binary switch is itself a paradigm of "making implicit design choices explicit" (similar to the SRR paper's "splitting the rank budget"), allowing the analysis to find optima between two extremes.
- "Stiefel manifold sampling is tighter than i.i.d. spherical sampling" is a practical guideline specific to ZO optimization, with minimal implementation cost (just QR orthogonalization of i.i.d. Gaussian directions) but directly yielding better privacy bounds.

## Limitations & Future Work
- The combination of strong convexity/convexity + bounded domain + smoothness + per-sample Lipschitz is a rather strong set of assumptions; for non-convex cases, only a numerical accountant (not closed-form) is provided.
- The lower bound on $K$ in Corollary 3.3, $K\ge 20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil\cdot\rceil)$, can be large as $c\to 1$ (weak convexity), meaning the practical range of usable $K$ may be compressed, requiring mini-batching to mitigate.
- The paper does not provide end-to-end experimental validation of hidden-state savings in LLM practice (e.g., fine-tuning 60B models), only theory + numerical curves; future work could be more convincing if large-scale experiments are conducted on improved MeZO-DP.
- The shift sequence $\{a_t,z_t\}$ in the coupling analysis comes from a constrained optimization problem with no closed-form solution; although a numerical accountant is provided, tuning cost is nontrivial.

## Related Work & Insights
- **vs DP-SGD + PABI (Feldman 2018, Altschuler-Talwar 2022/2023, Chien-Li 2025)**: First-order PABI uses shifted Rényi divergence directly on the original process, relying on isotropic noise + global Lipschitzness. This work reverse-engineers for ZO: directional noise + auxiliary process + pointwise Lipschitz, equivalently proving the key PABI property that "$\varepsilon$ saturates with $T$".
- **vs MeZO-DP (Zhang et al. 2024a)**: They proposed mechanism (a) (directional noise), which empirically has better utility but privacy analysis stops at composition; this work provides a tighter hidden-state bound for the same update and discovers the counterintuitive result that larger $K$ is better.
- **vs Tang et al. 2024**: Similar setting but still uses composition; this work is essentially a free upgrade (hybrid noise + orthogonal directions) on their algorithm, reducing privacy budget while maintaining utility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergent hidden-state DP bound for ZO + counterintuitive "$K$ larger, privacy better"; new analysis technique.
- Experimental Thoroughness: ⭐⭐⭐ Theory paper, numerical validation limited to synthetic setting + Figure 1 curve comparison, lacks end-to-end LLM validation.
- Writing Quality: ⭐⭐⭐⭐ Three challenges (noise shape, Lipschitz, composition vs hidden-state) are deconstructed one by one; introduction clearly explains "why first-order PABI cannot be directly applied".
- Value: ⭐⭐⭐⭐ Provides a tighter accountant for all DP-ZO-based large model fine-tuning, enabling more steps under the same $\varepsilon$ budget or tighter $\varepsilon$ for the same number of steps.

## Related Papers

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[ACL 2025\] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](../../ACL2025/llm_safety/are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)
- [\[ACL 2026\] Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge](../../ACL2026/llm_safety/beyond_end-to-end_dynamic_chain_optimization_for_private_llm_adaptation_on_the_e.md)

</div>

<!-- RELATED:END -->

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
