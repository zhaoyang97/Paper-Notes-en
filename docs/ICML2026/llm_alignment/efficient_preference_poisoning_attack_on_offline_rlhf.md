---
title: >-
  [Paper Note] Efficient Preference Poisoning Attack on Offline RLHF
description: >-
  [ICML 2026][LLM Alignment][Preference Poisoning] Based on the key observation that "flipping one preference label = adding a fixed vector independent of policy parameters to the loss gradient" for log-linear DPO…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Preference Poisoning"
  - "DPO"
  - "Label Flipping"
  - "Sparse Recovery"
  - "Lattice Basis Reduction"
date: 2026-05-08
content_hash: 5b4860d6aeeb9de2
---

# Efficient Preference Poisoning Attack on Offline RLHF

**Conference**: ICML 2026  
**arXiv**: [2605.02495](https://arxiv.org/abs/2605.02495)  
**Code**: None  
**Area**: LLM Security / Preference Poisoning / RLHF Alignment  
**Keywords**: Preference Poisoning, DPO, Label Flipping, Sparse Recovery, Lattice Basis Reduction

## TL;DR
Based on the key observation that "flipping one preference label = adding a fixed vector independent of policy parameters to the loss gradient" for log-linear DPO, this work reduces the targeted poisoning attack to a binary sparse approximation problem. The paper proposes two algorithms, BAL-A (based on LLL lattice basis reduction) and BMP-A (based on matching pursuit), along with provable recovery and impossibility conditions.

## Background & Motivation

**Background**: Offline RLHF has become the mainstream path for aligning LLMs. DPO trains directly on pre-collected paired preference datasets, bypassing the need for an explicit reward model. Two typical attack models have emerged around DPO security: label flipping and data injection.

**Limitations of Prior Work**: Data injection attacks (Nika et al., 2025) have a relatively complete theoretical characterization but are expensive—the number of injected samples must grow linearly with the original dataset size for the attack to succeed. Label flipping is more "economical" and realistic (attackers usually can only modify existing labels rather than create new pairs), but current research primarily relies on empirical observations and **lacks a theoretical characterization of "how many and which ones to flip"** to push the policy in a specific direction.

**Key Challenge**: An attacker performing label flipping faces two fundamental difficulties. First, the manipulable set is constrained—one can only flip a subset $\mathcal{F}$ of the $n$ existing comparisons. Second, the impact of a single label flip on the final learned policy $\hat\theta$ is parameter-dependent in general non-linear models and cannot be accurately predicted beforehand, making "which one is most effective" a combinatorial search problem.

**Goal**: For the log-linear policy class, provide (i) a first-order characterization of what a label flip actually changes; (ii) a formalization of targeted poisoning as a solvable optimization problem; and (iii) two provable algorithms with recovery/impossibility guarantees.

**Key Insight**: The authors notice that in log-linear DPO, the gradient increment $\Delta g_i$ caused by flipping a label $o_i\to-o_i$ in the per-sample loss $\ell_i(\theta)$ **is entirely independent of the current $\theta$**—it is simply a constant vector $o_i\beta(\psi(s_i,a_i)-\psi(s_i,a_i'))$. This transforms an apparently policy-dependent attack into "performing binary sparse approximation over a fixed dictionary $V$."

**Core Idea**: Rewrite "finding the minimum label flips to make the trained policy close to $\pi^\dagger$" as $\min_{x\in\{0,1\}^n}\|x\|_0$ s.t. $Vx=-g^\dagger$, where each column of dictionary $V$ is a gradient atom of a single sample flip, and target $g^\dagger$ is the gradient of the clean DPO loss at $\theta^\dagger$.

## Method

### Overall Architecture
The attack pipeline consists of three steps: **(1) Reduction**: Based on the "parameter-independent gradient increment" in Theorem 3.1, the targeted poisoning problem over $\{0,1\}^n$ is written as a binary sparse approximation $\min \mathbf{1}^\top x$ s.t. $\|Vx+g^\dagger\|_2\le\varepsilon$, where $V=[v_1,\dots,v_n]\in\mathbb{R}^{d\times n}$ and $v_i=o_i\beta\Delta\psi_i$. **(2) Residual-to-Policy Distance Bridge** (Lemma 3.2): Under $m$-strong convexity, if $\|Vx+g^\dagger\|_2\le\varepsilon$, then $\|\hat\theta-\theta^\dagger\|_2\le\varepsilon/m$. Log-linear properties then yield an $\ell_1$ policy distance bound, showing that controlling the residual ensures the policy approaches the target. **(3) Solving**: Since the objective is NP-hard, the authors provide two algorithms based on the attack scenario—BAL-A for minimum-flip without budget, and BMP-A for sparse $K$-flip with budget $K$—along with corresponding recovery/impossibility conditions.

### Key Designs

1. **Flip = Fixed Dictionary Atom (Theorem 3.1)**:
    - **Function**: Reduces the "impact of flipping one label on training results" to a constant vector independent of $\theta$.
    - **Mechanism**: For log-linear $\pi_\theta(a|s)\propto\exp(\psi(s,a)^\top\theta)$, the derivative of the single-sample DPO loss $\ell_i(\theta)$ with respect to $\theta$ involves a coupling of $o_i$ and $\sigma(\cdot)$ as $o_i\bigl(1-\sigma(o_i\beta\Delta\psi_i^\top\theta)\bigr)\beta\Delta\psi_i$, which is "sigmoid-symmetric" regarding $o_i$. Flipping $o_i$ to $-o_i$ and subtracting the sigmoid terms **exactly cancels out the $\theta$ portion**, leaving only $\Delta g_i=o_i\beta\Delta\psi_i$.
    - **Design Motivation**: This structural observation is the backbone of the paper—it transforms the bi-level challenge of "retraining and observing policy changes" into "finding a binary linear combination in a fixed vector set $\{v_i\}$ to approximate $-g^\dagger$," placing the attack into a standard sparse recovery context.

2. **Binary-Aware Lattice Embedding BAL-A (§4)**:
    - **Function**: Finds the minimum number of flips such that $Vx+g^\dagger=0$ for the budget-free case.
    - **Mechanism**: Applying integer relaxation to $\min_x\|Vx+g^\dagger\|^2$ yields a CVP, but CVP solutions may not be in $\{0,1\}$ and do not directly minimize the flip count. The authors construct a $(d+n)\times(n+1)$ embedding basis $B_{\mathrm{bin}}=\begin{pmatrix}V&-g^\dagger\\MI_n&0\end{pmatrix}$, such that the squared length of a lattice point corresponding to integer coefficient vector $z$ is $\|y(z)\|^2=\|Vz+g^\dagger\|^2+M^2\|z\|^2$, punishing both residual and coefficient magnitude. For $\{0,1\}$ solutions, $\|x\|_2^2=\mathbf{1}^\top x$, automatically replacing the $\ell_2$ penalty with the flip count. LLL ($\delta=0.75$) + Babai nearest-plane is used to find integer $z$, which is then truncated to $\{0,1\}$.
    - **Design Motivation**: When $M$ is sufficiently large (Lemma 4.1: $M_0\approx (B\sqrt{K^\star}+\sqrt{B^2K^\star+6BR+3B^2})/3$), it forces coefficients $\in\{-1,0,1\}$ and subsequently $\{0,1\}$ (under $z\ge 0$). The "separation condition" $\rho_k^2>M^2(K^\star-k)$ in Theorem 4.3 further ensures that the global minimum falls on the $K^\star$-flip feasible solution. This **first aligns LLL-CVP continuous/integer relaxation tools with $\{0,1\}$ binary minimum flip semantics**.

3. **Binary Matching Pursuit BMP-A (§5)**:
    - **Function**: Greedily identifies the $K^\star$ most effective flips under budget $K$ and provides a certificate for when an **attack is completely impossible**.
    - **Mechanism**: Adapts OMP/BMP to the unnormalized dictionary $V$—using normalized correlation scores $|\langle v_i,r\rangle|/\|v_i\|_2$ to select atoms at each step, while residual updates use original columns $r\leftarrow r-v_{i_t}$. The process stops after $K$ steps or if $\|r\|_2\le\varepsilon$. Complexity is extremely low (approx. $5000\times$ faster than BAL-A in experiments).
    - **Design Motivation**: Defining mutual coherence $\mu(V)=\max_{i\ne j}|\langle v_i,v_j\rangle|/(\|v_i\|\|v_j\|)$, Theorem 5.3 guarantees that BMP-A correctly identifies the support and achieves exact recovery in $K^\star$ steps if $\mu(V)<b/((2K^\star-1)B)$. Conversely, Theorem 5.4 provides two impossibility conditions: $\|g^\dagger\|_2-\varepsilon>\sqrt{K}\|V\|_2$ or $(\|g^\dagger\|_2-\varepsilon)^2>B^2(K+\mu(V)K(K-1))$. These conditions are algorithm-independent and provide a **spectral and geometric description of DPO's inherent robustness**—the smaller the column norms and the more divergent the directions, the harder the attack.

### Loss & Training
The attacker does not train a new model but solves the optimization problem above directly. BAL-A contains a single hyperparameter $M$ (scanned over 25 log-spaced values in experiments); BMP-A includes $K$ and $\varepsilon$. Downstream DPO training follows the standard recipe of log-linear + $\ell_2$ regularization: $L_{\mathrm{DPO}}(\theta;\mathcal{D})+\tfrac{\lambda}{2}\|\theta-\theta_\mu\|^2$.

## Key Experimental Results

### Main Results

| Dataset | Method | Setting | TPR | Residual/Distance |
|--------|------|------|------|----------|
| Synthetic Gaussian ($d=64,n=20,K^\star=5$) | BAL-A | $M<M_{\text{all sep}}\approx0.68$ | ≈1.0 | 0 |
| Synthetic Gaussian ($d=64,n=20,K^\star=5$) | BAL-A | $M>M_{\text{all sep}}$ | Rapid drop | Large |
| Synthetic low-coherence ($\mu\approx0.197,n=200$) | BMP-A | $K^\star\le K_{\text{coh}}=3$ | 1.0 | 0 |
| Synthetic low-coherence ($\mu\approx0.197,n=200$) | BMP-A | $K^\star>K_{\text{coh}}$ | High, slow decay | Small |
| SHP Real Data ($n=50,K^\star=7$, common feasible) | BAL-A | TP/FP/FN = 7/0/0 | 1.0 | $\|\pi_{\theta^\dagger}-\pi_{\hat\theta}\|_1\approx0.012$ |
| SHP Real Data ($n=50,K^\star=7$, common feasible) | BMP-A | TP/FP/FN = 7/0/0 | 1.0 | Same as above |

The clean-vs-attacked distance is $\|\pi_{\mathrm{clean}}-\pi_{\hat\theta}\|_1\approx 0.224$, which is approximately 19 times the attack-vs-groundtruth distance (0.012)—indicating that the recovered flipping pattern not only reproduces the constructed attack but **truly pushes the policy away from the clean baseline**.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| BAL-A, $M\to0$ | TPR≈1 | Smaller $M$ approaches pure residual minimization but loses binary enforcement |
| BAL-A, $M=M_0\approx1.69$ (Theoretical bound) | TPR drops significantly | The binary sufficiency bound in Lemma 4.1 is too conservative |
| BMP-A on low-coherence subset | TPR↑ | Matching pursuit is more accurate as dictionary geometry becomes more divergent |
| BMP-A on random subset | TPR↓ | High-coherence dictionaries cause greedy selection of incorrect atoms |
| BAL-A runtime (SHP, $n=50$) | 0.6865 s | Dominated by LLL preprocessing |
| BMP-A runtime (SHP, $n=50$) | $1.37\times10^{-4}$ s | Approx. $5\times10^3$ times faster than BAL-A |

### Key Findings
- BAL-A's success exhibits a **sharp transition at the separation threshold** $M_{\text{all sep}}$. The theoretical binary sufficiency bound $M_0$ is usually too conservative; in practice, smaller $M$ can still guarantee binary solutions.
- The mutual coherence sufficiency condition $K_{\text{coh}}$ for BMP-A is also conservative—TPR degrades slowly rather than crashing once $K^\star$ exceeds $K_{\text{coh}}$.
- Dictionary geometry (especially mutual coherence $\mu(V)$) is almost the sole determinant of attack success—on SHP, BMP-A performs significantly better on low-coherence subsets than on random subsets. Conversely, on high-coherence dictionaries, attacks may fail even if $K\gg K^\star$.
- The impossibility certificate (Theorem 5.4) is algorithm-independent—if $\|g^\dagger\|_2-\varepsilon>\sqrt{K}\|V\|_2$, **any** algorithm will fail to complete the attack within $K$ flips.

## Highlights & Insights
- **"Parameter-independent gradient increment" is the critical pivot for reducing poisoning from a bi-level problem to sparse recovery**. It relies on the symmetry of the log-linear + DPO sigmoid loss and represents an intrinsic vulnerability of this combination rather than an implementation detail.
- **The transplanting of number-theoretic tools like LLL + Babai into ML poisoning is ingenious**: The binary-aware embedding $\begin{pmatrix}V&-g^\dagger\\MI_n&0\end{pmatrix}$ encodes both "small residual" and "small coefficients." Controlling a single $M$ allows scanning between these objectives—this paradigm of "scheduling NP-hard binary constraints with a scalar" could be applied to many sparse selection problems beyond OMP/Lasso.
- **Impossibility conditions provide a geometric characterization of DPO robustness**: $\sqrt{K}\|V\|_2$ and $\mu(V)$ suggest that real defense lies not in post-hoc cleaning, but in **proactively designing preference datasets such that $V$ columns are small and divergent** (e.g., via diversity sampling/deduplication). This provides direct guidance for preference dataset construction.
- The "economy" of the attack—flipping only 7/50 labels on SHP pushes the policy $\ell_1$ distance to 0.224—represents a substantial efficiency gain over data injection, highlighting the RLHF annotation pipeline as an attack surface rather than just a "quality issue."

## Limitations & Future Work
- Strong assumptions: Covers only **log-linear policies + DPO**. In general neural network policies, gradient increments are $\theta$-dependent, requiring approximation or online dictionary updates.
- White-box: The attacker must know the feature map $\psi$, reference policy $\mu$, and target $\theta^\dagger$. Black-box versions or those with partial data knowledge are not discussed.
- The target policy $\pi^\dagger$ is specified by the attacker and assumed feasible (Assumption 3.3)—the work does not address "how to select a target policy that is both dangerous and reachable."
- BAL-A's LLL preprocessing becomes computationally expensive as $n$ reaches medium scales (>100), forcing the real-world SHP experiments to use $n=50$ subsets. Although BMP-A is fast, it requires the feasible subset to be low-coherence to avoid greedy errors.
- Experiments are validated only on the SHP dataset and do not cover more mainstream RLHF benchmarks like Anthropic-HH or UltraFeedback.
- Defense remains open: The authors provide impossibility certificates as "robustness conditions" but do not propose proactive defense algorithms.

## Related Work & Insights
- **vs. Nika et al. 2025 (Data Injection)**: Also performs theoretical analysis under log-linear DPO, but focuses on adding unconstrained sample pairs, concluding that injection volume grows linearly with $n$. This work targets "flipping subsets within a fixed $n$," where feasibility is constrained by $V$'s geometry, and efficiency is far higher.
- **vs. Rando & Tramèr 2024 / Wu et al. 2025a**: These works are more empirical; this paper provides the first provable characterization of label flipping under log-linear DPO.
- **vs. Wen & Li 2021 (Binary Matching Pursuit)**: Borrows the BMP support recovery framework, with the main technical contribution being the integration of "unnormalized columns + norm ratio $\rho=B/b$" into the coherence bound $\mu(V)<b/((2K^\star-1)B)$.
- **vs. Lenstra et al. 1982 / Babai 1986 (LLL & CVP)**: The first application of lattice basis reduction from number theory/cryptography as a "binary sparse selection" solver, utilizing the dual role of $M$ (residual regularization + binary penalty).
- **Insights**: (i) "Flip = Fixed Atom" suggests similar reductions might exist for other symmetric losses (contrastive/InfoNCE); (ii) The impossibility certificate paradigm can be used inversely to design robust data collection; (iii) Using LLL outside of ML poisoning, such as for sparse feature selection or integer neural network quantization, holds significant potential.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "parameter-independent gradient increment" + LLL grafting + impossibility certificates are all firsts in DPO poisoning.
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic + SHP dataset only; small scale ($n\le 401$) and lacks mainstream RLHF benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous and clear, with extensive appendices. The structure is well-explained by four tables/figures, though the LLL motivation for BAL-A is somewhat academic.
- **Value**: ⭐⭐⭐⭐ Directly relevant to understanding RLHF annotation security and designing robust datasets. The theoretical framework is easily extensible to other DPO variants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](../../AAAI2026/llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](../../ACL2026/llm_alignment/alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)
- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)
- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](../../NeurIPS2025/llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[ICML 2026\] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)

</div>

<!-- RELATED:END -->
