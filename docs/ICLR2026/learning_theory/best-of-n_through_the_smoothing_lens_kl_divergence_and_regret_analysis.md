---
title: >-
  [Paper Note] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis
description: >-
  [ICLR 2026][learning_theory][Best-of-N] This paper analyzes the commonly used Best-of-N (BoN) inference-time alignment through a "softened" framework, Soft Best-of-N (SBoN). It provides KL divergence upper bounds relative to the reference policy and upper/lower regret bounds relative to the optimal policy. The authors prove that when the proxy reward model q
tags:
  - ICLR 2026
  - learning_theory
  - Best-of-N
date: 2026-05-08
content_hash: b09d3101285d5bf3
---
# Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=tCv1D3M7Lb](https://openreview.net/forum?id=tCv1D3M7Lb)  
**Code**: To be confirmed  
**Area**: Learning Theory / LLM Alignment  
**Keywords**: Best-of-N, Inference-time Alignment, KL Divergence, Regret Bound, Reward Overoptimization  

## TL;DR
This paper analyzes the commonly used Best-of-N (BoN) inference-time alignment through a "softened" framework, Soft Best-of-N (SBoN). It provides KL divergence upper bounds relative to the reference policy and upper/lower regret bounds relative to the optimal policy. The authors prove that when the proxy reward model quality is poor and overoptimization occurs, a finite inverse temperature $\beta$ makes the regret bound for SBoN tighter than that for BoN, thereby mitigating reward hacking.

## Background & Motivation
**Background**: Post-processing methods for LLM alignment include RLHF, DPO, SLiC, controlled decoding, and Best-of-N (BoN), which selects the answer during the sampling phase without training. These methods essentially approximate the solution to KL-regularized reward maximization $\max_\pi \mathbb{E}_{y\sim\pi}[r(y,x)] - \frac{1}{\beta}\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$, whose optimal solution is a "tilted policy" $\pi_{\beta,r}(y|x)\propto \pi_{\mathrm{ref}}(y|x)\exp(\beta r(y,x))$, balancing high rewards and proximity to the reference model.

**Limitations of Prior Work**: Existing theories mostly analyze BoN under **ideal assumptions**, assuming access to the true reward $r^\star$ without proxy error. Beirami et al. proved $\mathrm{KL}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})\le \log N - 1 + \tfrac1N$ and showed BoN is near-optimal on the reward-vs-KL curve. However, in reality, BoN selects answers based on a **learned proxy reward** $\hat r$, which is only an approximation. Once $\hat r$ deviates from $r^\star$, the "greedy max" rule of BoN leads to **overoptimization** of the proxy reward, selecting answers of lower true quality—known as reward hacking.

**Key Challenge**: BoN takes the highest proxy score among $N$ candidates (the limit as $\beta\to\infty$), fully trusting the ranking of $\hat r$. When $\hat r$ is unreliable, more aggressive selection (larger $N$, more greedy) can be counterproductive, as minor proxy ranking errors are amplified. Existing regret analysis (Huang et al. 2025) provides bounds that grow with the $L_\infty$ norm of the reward estimation error and do not necessarily converge cleanly when the error vanishes.

**Goal**: Quantify how two factors affect alignment quality in realistic settings with **proxy error**: (a) the KL divergence between the alignment policy and the reference policy; (b) the regret, i.e., the difference in true reward between the optimal policy and the alignment policy. The paper seeks to answer: when should BoN be "softened," and by how much?

**Key Insight**: Instead of analyzing hard BoN directly, the authors utilize its smooth version, **Soft Best-of-N (SBoN)**—replacing "taking the maximum" with "softmax sampling based on proxy scores," introducing an adjustable inverse temperature $\beta$. As $\beta\to\infty$, it reverts to BoN; as $\beta\to0$, it reverts to direct sampling from the reference model. With this continuous dial, BoN can be treated as an endpoint of SBoN to derive unified scaling behaviors with respect to $N$, $\beta$, and reward quality.

**Core Idea**: Viewing BoN through a "smoothing lens"—proving that in overoptimization scenarios, moderate softening (finite $\beta$) achieves a balance between "KL tradeoff gains" and "proxy estimation errors," resulting in a regret bound for SBoN that is strictly superior to that of hard BoN.

## Method

### Overall Architecture
This is a purely theoretical analysis paper. The "method" is an analytical framework centered on SBoN: defining sampling policies and metrics, deriving KL divergence bounds, then deriving upper and lower regret bounds, and finally explaining "when to soften" via the structure of these bounds.

The logic flow is: ① For a fixed prompt $x$, sample $N$ candidates $Y_{1:N}$ independently from $\pi_{\mathrm{ref}}$; ② SBoN avoids strictly taking the highest proxy score, instead sampling one according to the softmax probability $P_Z(i)\propto\exp(\beta\hat r(Y_i,x))$, where the closed-form policy is $\pi^{\mathrm{SBoN}}_{\hat r}(y|x)=\pi_{\mathrm{ref}}(y|x)\exp(\beta \hat r(y,x))/Z_{N,\beta}$; ③ To characterize "proxy unreliability," two quantities are defined: **tilted error** $\varepsilon_{\beta,r}$ and **coverage** $C_{\beta,r}$; ④ KL divergence upper bounds (Lemma 4.1/4.2) and regret bounds (Theorem 5.2/5.6) are derived using these quantities plus $N, \beta$, where BoN bounds are derived as the limit $\beta\to\infty$; ⑤ Comparing SBoN and BoN bounds characterizes the overoptimization regime where an optimal $\beta$ exists and SBoN outperforms BoN.

The two main threads are KL divergence (measuring deviation from the reference) and regret (measuring distance from true optimality); all theorems quantify how "proxy error + N + $\beta$" pull these two threads.

### Key Designs

**1. Soft Best-of-N: Replacing "Hard Max" with Temperature-Adjustable Softmax Sampling**

The fundamental issue with BoN is its absolute trust in the proxy reward—deterministically taking $i^\star=\arg\max_i\hat r(Y_i,x)$. The authors re-derive the selection rule from the perspective of KL-regularized reward maximization: maximizing the expected proxy reward $\mathbb{E}_Z[\hat r(Y_Z,x)]$ on the simplex $\Delta_N$ of $N$ candidate indices, plus an entropy regularization term $\frac1\beta H(P_Z)$. The unique solution is the softmax distribution:

$$P_Z(i)=\frac{\exp(\beta\hat r(Y_i,x))}{\sum_{j=1}^N\exp(\beta\hat r(Y_j,x))},$$

from which $Z$ is sampled and $Y_Z$ returned. Its closed-form policy (Mayrink Verdun et al. 2025, Lemma 1) is $\pi^{\mathrm{SBoN}}_{\hat r}(y|x)=\pi_{\mathrm{ref}}(y|x)\exp(\beta\hat r(y,x))/Z_{N,\beta}$, which is precisely the **finite-sample version of the tilted optimal policy** $\pi_{\beta,\hat r}\propto\pi_{\mathrm{ref}}\exp(\beta\hat r)$. The inverse temperature $\beta$ acts as a continuous dial: $\beta\to\infty$ yields BoN, $\beta\to-\infty$ yields worst-of-N, and $\beta\to0$ yields uniform sampling. This design embeds the discrete selection problem into a continuous family.

**2. Tilted Error and Coverage: Scalars Characterizing Proxy Quality and Reference Sparsity**

To discuss overoptimization, the distance between the proxy reward $\hat r$ and the true reward $r^\star$ must be quantified. The authors define the **tilted error**:

$$\varepsilon_{\beta,r}(x):=\frac1\beta\log\Big(\mathbb{E}_{Y\sim\pi_{\mathrm{ref}}}\big[e^{\beta(r^\star(Y,x)-\hat r(Y,x))^2}\big]\Big),$$

which averages the squared estimation error according to the tilted weights: it reduces to MSE at $\beta=0$ and the supremum norm squared $\|r^\star-\hat r\|_\infty^2$ as $\beta\to\infty$. It is monotonically increasing in $\beta$, where $\varepsilon_{\beta,r}>0$ denotes the "mismatch/overoptimization" regime. Notably, this uses **calibrated** rewards (range $[0,1]$, following $\mathrm{Unif}(0,1)$ under the reference model), so if the proxy is a strictly monotonic transformation of the true reward, $\varepsilon=0$ and no overoptimization occurs.

The companion metric **coverage** $C_{\beta,r}(x):=\sum_y \pi_{\beta,r}^2(y|x)/\pi_{\mathrm{ref}}(y|x)$ (i.e., the $\chi^2$ divergence of the tilted policy vs. reference + 1) has a limit $C_{\infty,r}(x)=1/\sum_i\pi_{\mathrm{ref}}(y^{\max}_{i,r}(x)|x)$ that measures how difficult it is for the reference model to generate the optimal response.

**3. KL Divergence Bounds: Scaling with $N, \beta$ and the Extra Cost of Proxy Error**

Real reward gains are constrained by KL divergence—by Pinsker's inequality, $\mathbb{E}_{\pi^{\mathrm{SBoN}}_{r^\star}}[r^\star]\le 0.5+\sqrt{\tfrac12\mathrm{KL}(\pi^{\mathrm{SBoN}}_{r^\star}\|\pi_{\mathrm{ref}})}$, meaning real reward improvement cannot exceed the square root of KL. First, under true reward:

$$\mathrm{KL}\big(\pi^{\mathrm{SBoN}}_{r^\star}(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big)\le \log\!\Big(\frac{N}{1+(N-1)\exp(-\beta)}\Big)\quad(\text{Lemma 4.1}),$$

which yields $\log N$ as $\beta\to\infty$. While looser than $\log N-1+\tfrac1N$ from prior work, it holds for any $\beta$ and is tight at $\beta=0$. Crucially, Lemma 4.2 quantifies the cost of using a proxy:

$$\mathrm{KL}\big(\pi^{\mathrm{SBoN}}_{r^\star}\,\|\,\pi^{\mathrm{SBoN}}_{\hat r}\big)\le \frac{N\beta\sqrt{\varepsilon_{\beta,r}(x)}}{1+(N-1)\exp(-\beta)}\Big(\frac{N\exp(2\beta)}{(N-1)^2}+1\Big),$$

which is zero when $\varepsilon=0$. Remark 4.4 highlights the **core tension**: for a fixed $N$, Lemma 4.1 prefers large $\beta$ (better KL tradeoff), while Lemma 4.2 prefers small $\beta$ (more accurate estimation). An optimal $\beta$ must balance the two.

**4. Regret Bounds: Proving "Soften when Overoptimizing, Select Hard Otherwise"**

The final goal is the regret $\Delta J_{r^\star}=J_{r^\star}(\pi^\star_{r^\star})-J_{r^\star}(\pi^{\mathrm{SBoN}}_{\hat r})$. Theorem 5.2 gives the SBoN regret upper bound:

$$\Delta J_{r^\star}\le \sqrt{\varepsilon_{\beta,r}(x)}\Big(\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}\Big)+2\sqrt{\tfrac12\log\!\Big(1+\tfrac{C_{\infty,\hat r}-1}{N}\Big)}+\frac{\log C_{\infty,r^\star}(x)}{\beta},$$

where the three terms correspond to proxy error, finite sample statistical error, and softening bias ($\propto 1/\beta$). The BoN bound (Proposition 5.3) is the $\beta\to\infty$ limit. Unlike Huang et al. 2025, this bound **remains finite** even as $N$ grows or error vanishes. Theorem 5.6 further provides **lower bounds** under a Margin assumption $\gamma(x)=1-\sup_{y\notin Y^\star}r^\star(y,x)\in(0,1)$.

Remark 5.8/5.9 concludes the logic: defining $g(\beta)=\beta(\sqrt{\varepsilon_\infty}-\sqrt{\varepsilon_\beta})$, since $g(0)=g(\infty)=0$ and $g\ge0$, there must exist a maximum $\beta^\star\in(0,\infty)$. **When overoptimizing** ($\varepsilon>0$), if $\frac{\log C_{\infty,r^\star}}{\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}}\le g(\beta^\star)$, then SBoN at $\beta^\star$ is strictly better than BoN. **When not overoptimizing** ($\varepsilon=0$), BoN ($\beta\to\infty$) is superior as the $1/\beta$ term always increases regret.

## Key Experimental Results

### Main Results
Experiments validate the theory: Olmo-2 1B generator, Attaq dataset (harmful prompts) for harmlessness performance, and LLM-as-a-Judge as the true reward $r^\star$. Rewards are empirically calibrated using 256 samples. Comparison: Strong proxy RM (ArmoRM 8B) vs. Weak proxy RM (Beaver 7B).

| Setting | Proxy Reward Model | Phenonmenon | Theoretical Correspondence |
|------|------------|------|-----------|
| Strong RM | ArmoRM 8B | Harmlessness increases monotonically with $N$; hard BoN performs well. | $\varepsilon\approx0$, no overoptimization, BoN is superior (Remark 5.9). |
| Weak RM | Beaver 7B | At large $N$, BoN degrades due to reward hacking; SBoN with appropriate $\beta$ stabilizes/outperforms. | $\varepsilon>0$, overoptimization, finite $\beta^\star$ exists (Remark 4.4 / 5.8). |

### Ablation Study
- **Inverse Temperature $\beta$ (Weak RM)**: The harmlessness curve for medium $\beta$ is higher than both $\beta\to\infty$ (BoN) and $\beta\to0$. This confirms the existence of a finite optimal $\beta^\star$.
- **Sample Count $N$**: KL upper bound grows with $N$, consistent with the $\log\frac{N}{1+(N-1)e^{-\beta}}$ scaling in Lemma 4.1.
- **RM Quality**: As the RM weakens, BoN overoptimizes earlier, and the gain from softening increases ($\varepsilon_{\beta,r}$ increases).

### Key Findings
- **Overoptimization is a prerequisite for softening gains**: SBoN only outperforms BoN when the proxy reward quality is low ($\varepsilon>0$).
- **A finite optimal temperature exists**: Under weak RM, performance is non-monotonic with respect to $\beta$.
- **Reward calibration is crucial**: It ensures that proxies that are monotonic transformations of the true reward result in $\varepsilon=0$, a property not captured by raw rewards.

## Highlights & Insights
- **Embedding BoN into a continuous family**: By treating BoN as the $\beta\to\infty$ limit of SBoN, the analysis covers both algorithms. This "soften then take limit" approach is applicable to other greedy inference algorithms.
- **Elegant Tilted Error design**: $\varepsilon_{\beta,r}$ bridges MSE and $L_\infty$ error using the same $\beta$, allowing proxy error and sampling temperature to be discussed within the same framework.
- **Actionable criteria**: Remark 5.8 provides an explicit condition $\frac{\log C_{\infty,r^\star}}{\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}}\le g(\beta^\star)$ to judge whether to soften.

## Limitations & Future Work
- The KL upper bound for BoN ($\log N$) is **looser** than the $\log N-1+\tfrac1N$ bound from Beirami et al.; tighter BoN-specific bounds are needed.
- **Coupled Temperatures**: The tilted error and the SBoN policy share the same $\beta$. Decoupling these temperatures is a promising future direction.
- Strong assumptions: Requires **calibrated rewards**, achievable maximum reward, finite prompt/response sets, and Margin assumptions.
- Small experimental scale (1B generator, one dataset).

## Related Work & Insights
- **vs Beirami et al. 2024 / Mroueh 2024**: They provide tighter KL bounds under ideal true reward/bijection assumptions. Ours allows for **proxy error**, yielding looser BoN bounds but covering overoptimization analysis.
- **vs Yang et al. 2024**: They showed BoN is equivalent to KL-constrained RL solutions under true rewards; this work removes that ideal assumption.
- **vs Huang et al. 2025**: Similar regret study but their bounds grow with $L_\infty$ error of uncalibrated rewards; this work's bounds remain finite and characterize the $O(\exp(-N))$ decay when errors vanish.

## Rating
- Novelty: ⭐⭐⭐⭐  
- Experimental Thoroughness: ⭐⭐⭐  
- Writing Quality: ⭐⭐⭐⭐  
- Value: ⭐⭐⭐⭐  

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions](a_sharp_kl_convergence_analysis_for_diffusion_models_under_minimal_assumptions.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)

</div>

<!-- RELATED:END -->
