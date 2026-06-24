---
title: >-
  [Paper Note] Convergent Differential Privacy Analysis for General Federated Learning
description: >-
  [ICLR2026][AI Safety][Differential Privacy] This paper utilizes the f-DP framework and shifted interpolation techniques to prove, for the first time, that the "worst-case privacy" of two classic Federated Learning methods (Noisy-FedAvg / Noisy-FedProx) under non-convex smooth objectives **converges to a constant lower bound** as the number of communication rounds $T \to \infty$ rather than diverging. This theoretically refutes the long-standing perception that "long-term FL-D…
tags:
  - "ICLR2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Federated Learning"
  - "f-DP"
  - "Convergent Privacy"
  - "shifted interpolation"
date: 2026-05-08
content_hash: dccb1d6fe8b99393
---

# Convergent Differential Privacy Analysis for General Federated Learning

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=7Zbe5ad3eX](https://openreview.net/forum?id=7Zbe5ad3eX)  
**Code**: To be confirmed  
**Area**: Federated Learning / Differential Privacy / Learning Theory  
**Keywords**: Differential Privacy, Federated Learning, f-DP, Convergent Privacy, shifted interpolation

## TL;DR
This paper utilizes the f-DP framework and shifted interpolation techniques to prove, for the first time, that the "worst-case privacy" of two classic Federated Learning methods (Noisy-FedAvg / Noisy-FedProx) under non-convex smooth objectives **converges to a constant lower bound** as the number of communication rounds $T \to \infty$ rather than diverging. This theoretically refutes the long-standing perception that "long-term FL-DP training inevitably exhausts the privacy budget."

## Background & Motivation
**Background**: Federated Learning (FL) allows multiple clients to train collaboratively without uploading raw data. However, models/gradients can still indirectly leak privacy through reconstruction attacks or membership inference attacks. Embedding Differential Privacy (DP) into FL by adding Gaussian noise before uploading parameters results in the mainstream FL-DP privacy protection paradigm.

**Limitations of Prior Work**: Almost all privacy analyses of FL-DP are built upon the **composition theorem** and "privacy amplification by iteration." The essence of the composition theorem is the linear accumulation of privacy budgets—adding up $(\epsilon_t, \delta_t)$ for each round—which leads to the requirement that the noise variance $\sigma^2$ must be proportional to $T$ (or even $TK$) for $(\epsilon, \delta)$-DP. This leads to a counter-intuitive conclusion: the privacy bound **diverges arbitrarily** as the number of rounds $T \to \infty$, implying that FL-DP completely loses privacy protection capabilities under long-term training.

**Key Challenge**: This theoretical conclusion contradicts experimental observations: in practice, models still maintain significant privacy protection after many rounds of training with **constant-level noise**. The root cause is that composition analysis is too loose; it treats each round as an independent privacy loss accumulation, failing to exploit the fact that "local-global optimization dynamics naturally contract sensitivity." In convex or strongly convex Noisy Gradient Descent, existing works (Chourasia 2021, Altschuler & Talwar 2022, etc.) have proved **convergent privacy** using RDP, but these analysis techniques have not yet been migrated to FL due to multi-step local updates and heterogeneous data resulting in biased local models.

**Goal**: (1) Establish a **tight and convergent** worst-case privacy bound for FL-DP under non-convex smooth objectives; (2) Clarify the actual impact of key hyperparameters such as $K$ (local steps), $m$ (client scale), $V$ (clipping norm), and $\alpha$ (FedProx regularization coefficient) on privacy.

**Key Insight**: The authors replace the composition theorem with **f-DP** (proposed by Dong et al. 2022)—an information-theoretically lossless DP definition based on the trade-off curve between Type I/II errors in hypothesis testing. Combined with the **shifted interpolation** technique from Bok et al. 2024, they replace "amplifying $T$ times round-by-round" with "amplifying only $T-t_0$ times along an interpolation path," thereby compressing the diverging bound into a convergent one.

**Core Idea**: By using the f-DP trade-off function and an optimizable interpolation sequence, the global sensitivity is decomposed into "data sensitivity + model sensitivity," which are scaled term-by-term using interpolation coefficients $\lambda_t$. Proving that the weighted sensitivity and $H_0$ converge to constants ensures the convergence of the privacy lower bound.

## Method

### Overall Architecture
This is a purely theoretical analysis paper that does not propose a new algorithm but rather provides tighter privacy proofs for two existing methods. The analyzed methods are standard:

- **Noisy-FedAvg**: Each client performs $K$ steps of clipped gradient descent $w_{i,k+1,t}=w_{i,k,t}-\eta_{k,t}g_{i,k,t}$ (where $g$ is the gradient clipped to norm $V$), adds isotropic Gaussian noise $n_i\sim\mathcal{N}(0,\sigma^2 I_d)$ before uploading, and the server aggregates these to obtain the global model.
- **Noisy-FedProx**: A proximal regularization term $\frac{\alpha}{2}\|w-w_t\|^2$ is added to the local objective, with the iteration form $w_{i,k+1,t}=w_{i,k,t}-\eta_{k,t}[g_{i,k,t}+\alpha(w_{i,k,t}-w_t)]$.

The analysis targets **global model privacy**: while local privacy at the time of upload is directly guaranteed by noise, the challenge lies in whether an attacker can infer from the global model—after the server broadcasts aggregated parameters—whether a specific sample participated in training. The proof logic is: write the global update on adjacent datasets $C,C'$ as $w_{t+1}=\phi(w_t)+n_t$ ($\phi$ is the accumulation of $K$ local steps, $n_t\sim\mathcal{N}(0,\sigma^2 I_d/m)$ is the average noise) $\to$ construct a shifted interpolation sequence to lower-bound the trade-off function $T(w_T;w_T')$ as a sum of weighted global sensitivities (Theorem 1) $\to$ decompose global sensitivity into data/model parts and provide recurrence coefficients (Theorem 2) $\to$ solve the minimization problem for $\lambda_t, t_0$, using $t_0=0$ relaxation for a solvable form (Sec 4.3) $\to$ derive convergent bounds for four learning rate strategies (Theorem 3, 4).

### Key Designs

**1. Replacing Composition Theorems with f-DP Trade-off Functions: From "Loose Addition" to "Lossless Characterization"**

Previous analyses used $(\epsilon, \delta)$-DP or RDP, where privacy budgets are linearly added during composition ($(\epsilon_1,\delta_1)+(\epsilon_2,\delta_2)\to(\epsilon_1+\epsilon_2,\delta_1+\delta_2)$), which is the source of divergence. This paper uses f-DP: modeling whether an attacker can distinguish between adjacent datasets $C$ ($H_0$) and $C'$ ($H_1$) as a hypothesis test, characterized by the optimal trade-off curve between Type I error $E_I=\mathbb{E}_{M(C)}[\chi]$ and Type II error $E_{II}=1-\mathbb{E}_{M(C')}[\chi]$. The trade-off function is defined as $T(P;Q)(\gamma)=\inf\{1-\mathbb{E}_Q[\chi]\mid \mathbb{E}_P[\chi]\le\gamma\}$, which is convex, continuous, and non-increasing. For Gaussian distributions, this specializes to Gaussian-DP: $T_G(\mu)(\gamma)=\Phi(\Phi^{-1}(1-\gamma)-\mu)$, where larger $\mu$ implies weaker privacy. f-DP is information-theoretically lossless, avoiding the "excessive relaxation for the sake of summability" inherent in composition analysis, and its conclusions can be converted back to $(\epsilon, \delta)$-DP and RDP (Table 3).

**2. Shifted Interpolation: Compressing "T Amplifications" into "T-t₀ Amplifications"**

Traditional privacy amplification requires an evaluation of the relationship between $w$ and $w'$ at every round, leading to divergence over $T$ rounds. Borrowing from Bok et al. 2024, the authors construct an interpolation sequence to reduce the number of amplifications from $T$ to $T-t_0$:

$$\tilde{w}_{t+1}=\lambda_{t+1}\phi(w_t)+(1-\lambda_{t+1})\phi'(\tilde{w}_t)+n_t,\quad t=t_0,\dots,T-1$$

Let $\lambda_T=1$ so that $\tilde{w}_T=w_T$, and use $\tilde{w}_{t_0}=w'_{t_0}$ as the starting point, where $0\le\lambda_t\le1$ are coefficients to be optimized. Theorem 1 lower-bounds the trade-off function as:

$$T(w_T;w_T')\ge T_G\!\left(\frac{\sqrt{m}}{\sigma}\sqrt{\textstyle\sum_{t=t_0}^{T-1}\lambda_{t+1}^2\|\phi(w_t)-\phi'(\tilde{w}_t)\|^2}\right)$$

The core idea is: the diverging global sensitivity $\|\phi(w_t)-\phi'(\tilde{w}_t)\|$ can be **suppressed term-by-term** by the interpolation coefficients $\lambda_t \le 1$. If $\lambda_t$ is chosen correctly, the weighted sum converges—this is the source of convergent privacy.

**3. Binary Decomposition of Global Sensitivity: Data Sensitivity + Model Sensitivity**

Global sensitivity $\|\phi(w_t)-\phi'(\tilde{w}_t)\|$ is influenced by both "data difference" and "initial state difference," making direct analysis extremely difficult. The authors introduce an auxiliary sequence $\phi'(w_t)$ to decompose the sensitivity (Theorem 2):

$$\|\phi(w_t)-\phi'(\tilde{w}_t)\|\le \underbrace{\rho_t\|w_t-\tilde{w}_t\|}_{\text{Model Sensitivity}}+\underbrace{\gamma_t}_{\text{Data Sensitivity}}$$

**Data sensitivity** $\gamma_t$ measures the error generated by training on different datasets starting from the same initialization (purely caused by data differences); **model sensitivity** $\rho_t\|w_t-\tilde{w}_t\|$ measures the error generated by training on the same dataset starting from different initializations (determined by the similarity of the two initial states). $\rho_t, \gamma_t$ vary with learning rate strategies (e.g., in Noisy-FedAvg with constant LR, $\rho_t=(1+\mu L)^K$ and $\gamma_t=\frac{2\mu V}{m}K$). Remark 2.1 notes that $\rho_t$ is always greater than 1 (typical for non-convexity), so the upper bound of sensitivity itself diverges with $t$—but with the scaling of $\lambda_t \le 1$, the final sum still converges.

**4. Relaxation with $t_0=0$ and Solving the Minimization Problem**

The bound in Theorem 1 depends on the parameters $\lambda_t$ and $t_0$. The authors denote the weighted sensitivity as $H(\lambda_t,t_0)=\sum_{t=t_0}^{T-1}\lambda_{t+1}^2(\rho_t\|w_t-\tilde{w}_t\|+\gamma_t)^2$ and seek to minimize it for the tightest bound. There is a dilemma: a small $t_0$ introduces a small stability gap but allows sensitivity to skyrocket over $T-t_0$ rounds; a large $t_0$ results in small accumulated errors but still diverges due to unbounded global sensitivity. The authors relax the problem by setting $t_0=0$—here, the stability error is exactly 0 (since $\tilde{w}_0=w'_0$ are aligned), avoiding divergence. Since $H_0 \ge H^\star$, then $T(w_T;w_T')\ge T_G(\sqrt{m}H^\star/\sigma)\ge T_G(\sqrt{m}H_0/\sigma)$. Crucially, even the relaxed $H_0$ converges to a constant form, thus ensuring the convergence of the privacy lower bound.

The two core theorems:

- **Theorem 3 (Noisy-FedAvg)**: Convergent bounds are given for four learning rate strategies (Constant C / Cyclical Decay CD / Step Decay SD / Inverse Decay ID). Taking step decay $\eta_{k,t}=\frac{\mu}{t+1}$ as an example: $T(w_T;w_T')>T_G\!\left(\frac{2\mu V K}{\sqrt{m}\sigma}\sqrt{2-\frac{1}{T}}\right)$, where the term in parentheses approaches the constant $\frac{2\mu V K}{\sqrt{m}\sigma}\sqrt{2}$ as $T\to\infty$. This is the **first convergent privacy analysis for non-convex FL-DP**. Remark 3.1 concludes: privacy is mainly determined by clipping norm $V$, local steps $K$, scale $m$, and noise $\sigma$. Larger $V$ increases the gap, while larger $m$ strengthens privacy (sensitivity $O(1/\sqrt{m})$). Constant noise can achieve convergent privacy.

- **Theorem 4 (Noisy-FedProx)**: When $\alpha > L$ and $\eta < \frac{1}{\alpha-L}$, $T(w_T;w_T')\ge T_G\!\left(\frac{2V}{\sqrt{m\alpha}\sigma}\sqrt{\frac{2\alpha-L}{L}\left(1-\frac{2}{(\frac{\alpha}{\alpha-L})^T+1}\right)}\right)$. The proximal regularization term brings a key advantage: privacy **no longer depends on local steps $K$**, even with a constant learning rate. Remark 4.1 notes that larger $\alpha$ strengthens privacy (sensitivity $O(1/\sqrt{\alpha})$), though excessive $\alpha$ slows training—thus $\alpha$ is a fine trade-off between optimization and privacy. This yields a "win-win" insight: **well-designed local regularization can simultaneously improve optimization and privacy**.

### Loss & Training
No new training objectives are proposed. The analysis is established under Assumption 1 (each local objective $f_i$ is $L$-smooth: $\|\nabla f_i(w_1)-\nabla f_i(w_2)\|\le L\|w_1-w_2\|$), which is a standard non-convex smooth assumption.

## Key Experimental Results

The purpose of the experiments is not to beat the SOTA, but to **verify the theory**: global sensitivity is indeed bounded and its variation with $m/K/V/\alpha$ matches theoretical predictions.

### Main Results
Accuracy of Noisy-FedAvg on MNIST (LeNet-5)/CIFAR-10 (ResNet-18) (Dir-0.1 high heterogeneity, fixed $TK=30000$, step decay LR):

| Setting | Noise $\sigma$ | $m=50, K=50$ | $m=100, K=50$ |
| --- | --- | --- | --- |
| MNIST | $10^{-1}$ | 95.40 | 97.32 |
| MNIST | $10^{-3}$ | 98.41 | 98.94 |
| CIFAR-10 | $10^{-1}$ | 53.76 | 62.02 |
| CIFAR-10 | $10^{-3}$ | 70.98 | 75.38 |

When $\sigma=1.0$, training diverges directly (marked as "-" in the paper). More clients reduce the impact of noise (more noise averaged $\to$ approaches noise mean $\to$ approximately noiseless), consistent with $O(1/\sqrt{m})$. Increasing $\sigma$ from $10^{-3}$ to $10^{-1}$ causes a drop of 5.57%/1.62% on MNIST ($m=20/100$) and 14.19%/11% on CIFAR-10.

### Ablation Study
Noisy-FedAvg vs Noisy-FedProx ($T=600$, performance and sensitivity comparison; lower sensitivity indicates stronger privacy):

| Configuration | Accuracy | Global Sensitivity |
| --- | --- | --- |
| Noisy-FedAvg | 60.67 | 31.33 |
| Noisy-FedProx $\alpha=0.01$ | 60.69 | 30.97 |
| Noisy-FedProx $\alpha=0.1$ | 60.94 | 18.52 |
| Noisy-FedProx $\alpha=1$ | 56.33 | 6.34 |

### Key Findings
- **Sensitivity is bounded and does not diverge with $T, K$**: Fig.2 shows that although increasing $K$ raises sensitivity during the process, the upper bound after optimization convergence remains unchanged. this is experimental evidence that the "privacy lower bound exists and is independent of $T, K$," debunking the divergence predicted by composition analysis.
- **Impact of $m, V$ aligns with theory**: Sensitivity decreases as $m$ increases ($O(1/\sqrt{m})$) and increases as $V$ increases ($O(V)$).
- **FedProx proximal term is a "free lunch" for privacy**: At $\alpha=0.01$, sensitivity is nearly the same as FedAvg but accuracy is slightly higher; at $\alpha=0.1$, sensitivity is nearly halved (31.33 $\to$ 18.52) while accuracy actually increases to 60.94; at $\alpha=1$, sensitivity drops to 6.34 (significant privacy enhancement) but accuracy falls to 56.33. This indicates a suitable $\alpha$ can significantly reduce sensitivity and enhance privacy with almost no loss in accuracy.

## Highlights & Insights
- **Overturning long-standing perceptions**: Proves that the pessimistic conclusion that "FL-DP privacy is inevitably exhausted during long-term training"—based on composition theorems—is false. Constant noise is sufficient to maintain a convergent privacy lower bound. This is of great practical significance: one does not need to add infinite noise for long-term training.
- **Effective combination of f-DP and shifted interpolation**: f-DP provides a lossless privacy metric, while shifted interpolation reduces the amplification rounds from $T$ to $T-t_0$. Migrating this set of techniques from convex Noisy Gradient Descent to the non-convex + multi-step local update + heterogeneous data FL scenario is the core technical contribution.
- **Sensitivity decomposition as a reusable tool**: Decomposing global sensitivity into "data sensitivity + model sensitivity" using an auxiliary sequence is a clear approach that can be migrated to the privacy analysis of other multi-step local update algorithms (e.g., SCAFFOLD, FedDyn).
- **"Regularization as Privacy" insight**: The proximal term decouples privacy from $K$, suggesting that many FL optimization methods based on local regularization (e.g., FedDyn) inherently possess privacy advantages.

## Limitations & Future Work
- **Covers only two classic methods**: The analysis is limited to Noisy-FedAvg / Noisy-FedProx. It has not yet provided convergent privacy bounds for more advanced FL optimizers with momentum correction, variance reduction, or primal-dual methods.
- **Relies on L-smoothness assumption**: Non-convexity is allowed but smoothness is required; the rigor for non-smooth objectives (like actual networks with ReLU non-differentiable points or $\ell_1$ regularization) is questionable. The bias introduced by the clipping operation itself is also not fully discussed.
- **Worst-case lower bound rather than exact privacy**: The $t_0=0$ relaxation introduces a gap $H_0 \ge H^\star$, making the bound conservative; actual privacy might be stronger. Tightness is only briefly discussed in Appendix F.
- **Small experimental scale**: MNIST/CIFAR-10 + LeNet/ResNet-18 lacks verification in scenarios with large models, large-scale clients, or cross-device real-world heterogeneity. While accuracy isn't the focus, whether sensitivity measurement remains bounded on real large models deserves further testing.

## Related Work & Insights
- **vs. Composition-based analysis (Wei 2020 / Shi 2021 / Zhang 2021 / Noble 2022)**: These works use $(\epsilon, \delta)$-DP/RDP + composition, requiring $\sigma^2 \propto T$ or $TK$ to maintain privacy, leading to divergence as $T \to \infty$. This paper uses f-DP + shifted interpolation to get an $O(2-1/T)$ convergent bound with constant noise. Table 3 converts these results back to $(\epsilon, \delta)$-DP and RDP for comparison.
- **vs. Convex/Strongly Convex convergent privacy (Chourasia 2021 / Altschuler & Talwar 2022 / Bastianello 2024)**: They proved convergent privacy under convex or $\beta$-strongly convex assumptions using RDP, but were limited to single-machine Noisy Gradient Descent or strong convexity. This paper extends convergent privacy to **non-convex + federated multi-step local updates** for the first time without requiring strong convexity.
- **vs. Bok et al. 2024 (Shifted Interpolation original work)**: Bok introduced the interpolation technique for single-machine convex/strongly convex scenarios. This paper extends it to the local-global dynamics of FL and adds the binary decomposition of data/model sensitivity to handle multi-step heterogeneous updates.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergent privacy analysis for non-convex FL-DP, overturning long-standing perceptions with significant technical migration efforts.
- Experimental Thoroughness: ⭐⭐⭐ Experiments solely verify the theory at a small scale, which is sufficient for a theoretical paper.
- Writing Quality: ⭐⭐⭐⭐ The proof logic is clear, theorems are well-compared in tables, and even with high formula density, it remains readable.
- Value: ⭐⭐⭐⭐⭐ Provides more credible privacy guarantees for FL-DP and offers practical insights like "regularization is privacy."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Federated Learning of Quantile Inference under Local Differential Privacy](federated_learning_of_quantile_inference_under_local_differential_privacy.md)
- [\[ICLR 2026\] Fairness via Independence: A General Regularization Framework for Machine Learning](fairness_via_independence_a_general_regularization_framework_for_machine_learnin.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICLR 2026\] Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy](beyond_membership_limitations_of_addremove_adjacency_in_differential_privacy.md)
- [\[ICLR 2026\] RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility](resfl_an_uncertainty-aware_framework_for_responsible_federated_learning_by_balan.md)

</div>

<!-- RELATED:END -->
