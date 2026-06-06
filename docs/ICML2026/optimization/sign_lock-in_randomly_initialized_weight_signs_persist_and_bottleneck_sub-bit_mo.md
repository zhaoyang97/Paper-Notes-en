---
title: >-
  [Paper Note] Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression
description: >-
  [ICML2026][Optimization][Sub-bit compression] This paper reveals that post-training weight sign matrices are indistinguishable from i.i.d. Rademacher noise across all architectures…
tags:
  - "ICML2026"
  - "Optimization"
  - "Sub-bit compression"
  - "sign bit"
  - "lock-in theory"
  - "geometric tail law"
  - "low-rank templates"
date: 2026-05-08
content_hash: 6a682e3887465aea
---

# Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression

**Conference**: ICML2026  
**arXiv**: [2602.17063](https://arxiv.org/abs/2602.17063)  
**Code**: TBD  
**Area**: Model Compression / Quantization / Optimization Dynamics  
**Keywords**: Sub-bit compression, sign bit, lock-in theory, geometric tail law, low-rank templates  

## TL;DR
This paper reveals that post-training weight sign matrices are indistinguishable from i.i.d. Rademacher noise across all architectures, forming a "one-bit wall" for sub-bit compression. Using stopping time analysis, it proves this pseudo-randomness is actually the "locking" of initialization signs. Based on this, it proposes a from-scratch training scheme using low-rank sign templates + gap initialization + boundary log-barrier regularization to amortize sign bits to nearly 0 bit/weight.

## Background & Motivation

**Background**: Model compression mainstream follows the "few-bit" path, combining 2~4 bit quantization, low-rank decomposition, pruning, and entropy coding. Compressing the magnitude $A=|W|$ to below 1 bit per weight is not difficult. At this scale, the sign bit $S=\mathrm{sign}(W)$ is treated as a relatively small fixed overhead and is rarely discussed.

**Limitations of Prior Work**: Once the target drops to the "sub-bit" regime (average <1 bit per weight), the sign bit becomes an incompressible bottleneck. The authors systematically tested three things on MLP-Mixer / ResNet18 / TinyLlama-1.1B: (i) the best rank-$r$ Frobenius approximation error $E_r(S)$ of the sign matrix $S$ decays significantly slower than $E_r(A)$; (ii) Two-sample Kolmogorov–Smirnov tests show normalized singular values of $S$ are nearly indistinguishable from i.i.d. Rademacher; (iii) the entropy rate proxy $\widehat{H}_{\mathrm{RD}}\approx 1$ from Shannon rate-distortion lower bounds implies signs have almost no redundancy.

**Key Challenge**: Post-training sign matrices "look like noise," yet from a single-coordinate perspective, most weights maintain their initialization signs, with flip ratios consistently below 0.5. These two facts—"marginal distribution like i.i.d. Rademacher" and "highly persistent coordinate-wise trajectories"—must coexist.

**Goal**: (1) Identify a mathematical mechanism explaining "pseudo-random distribution + strong persistence"; (2) Use this mechanism to transform sign bits from a bottleneck into a controllable variable.

**Key Insight**: View a single scalar weight as a 1D stochastic process $(w_t)$. A sign change can only occur by crossing the zero boundary. If training dynamics keep weights in "outer regions" far from 0, sign flips are necessarily triggered by rare boundary-crossing events—a Freidlin–Wentzell type stopping/first-passage problem.

**Core Idea**: Replace step-by-step flip counts with an effective flip count $K_T^{\mathrm{eff}}(\rho)$ of "outer region → boundary neighborhood → outer region" transitions, proving it follows a geometric tail distribution. Since initial signs are locked, the "initial signs" can be set as low-rank reproducible templates, turning the lock-in from a bug into a feature.

## Method

### Overall Architecture
The paper consists of two parts: diagnosis and intervention. The diagnosis phase performs sign–magnitude decomposition $W=S\odot A$, measuring SVD compressibility, spectral randomness, and drift rates of $S$ and $A$; it then derives the sign lock-in theorem using 1D stopping time analysis. The intervention phase treats two key quantities from the diagnosis—initial hit probability $h_T$ and re-entry probability $g_T$—as control knobs. It proposes a from-scratch training pipeline: "low-rank sign template initialization + gap sampling + outer-region log-barrier regularization," ensuring the post-training sign matrix remains a shallow perturbation of the initial template, requiring only $(G,H,\text{rank})$ for storage.

### Key Designs

1.  **1D Stopping Time Framework and Sign Lock-in Theorem**:
    - **Function**: Reframe the "post-training sign flip" event as a stopping time event and provide a tail probability upper bound.
    - **Mechanism**: Fix an outer threshold $\rho>0$ and boundary radius $\epsilon=\max\{\epsilon_0,\Delta\}$. Recursively define stopping times $\sigma_0=\inf\{t:|w_t|\ge\rho\}$, $\tau_k=\inf\{t>\sigma_{k-1}:|w_t|\le\epsilon\}$, and $\sigma_k=\inf\{t>\tau_k:|w_t|\ge\rho\}$. Under the "bounded update assumption" (step increment $|w_{t+1}-w_t|\le\Delta$ with probability $\ge 1-\delta_{\mathrm{upd}}$) and "re-entry rate assumption" ($\mathbb{P}[\tau_{k+1}\le T\mid\mathcal{F}_{\sigma_k}]\le g_T$), the effective flip count satisfies $\mathbb{P}[K_T^{\mathrm{eff}}(\rho)\ge k]\le h_T g_T^{k-1}+\delta_{\mathrm{upd}}$, where $h_T=\mathbb{P}[\tau_1\le T]$.
    - **Design Motivation**: Capture rare events that functional asymptotic analysis might miss. Boundary crossing is a rare event; stopping times are the only language to reconcile "apparent randomness" with "persistence." The resulting geometric tail law $(\hat h,\hat g)$ is experimentally verifiable. Proposition 3.5 for SGD links $g_T^{\mathrm{SGD}}$ to the margin $\rho-\epsilon$, the sum of squared learning rates $\sum_t\eta_t^2$, and batch noise, providing actionable metrics for stronger lock-in.

2.  **Low-Rank Sign Template $T=\mathrm{sign}(GH^\top)$**:
    - **Function**: Replace "random signs" with reproducible structured signs to reduce storage costs to nearly zero.
    - **Mechanism**: For each layer $W^{(l)}\in\mathbb{R}^{m\times n}$, sample $G\in\mathbb{R}^{m\times r}$ and $H\in\mathbb{R}^{n\times r}$ (i.i.d. standard normal, $r\ll\min(m,n)$, $r=2$ in paper). Set $T^{(l)}=\mathrm{sign}(GH^\top)$. Sample magnitudes $A^{(l)}$ from any positive distribution; initialize $W^{(l)}=T^{(l)}\odot A^{(l)}$. At inference, store only $(G,H,r)$ and magnitude SVD factors, making sign bits per weight approach $0$.
    - **Design Motivation**: Sign lock-in ensures the final sign matrix remains close to $T^{(l)}$. Moving low-rank structure from $W$ directly to "$\mathrm{sign}(GH^\top)$" bypasses the fundamental "one-bit wall" where sign matrices typically resist low-rank approximation.

3.  **Gap Initialization + Outer-Region Log-Barrier Regularization**:
    - **Function**: Actively minimize $h_T$ and $g_T$ to enforce template locking.
    - **Mechanism**: (a) Gap initialization uses rejection sampling $z\sim\mathcal{N}(0,\sigma_{\mathrm{init}}^2)$; if $|z|<a_{\mathrm{init}}=c_{\mathrm{gap}}\sigma_{\mathrm{init}}$, it is resampled, creating a truncated Gaussian on $\mathbb{R}\setminus[-a_{\mathrm{init}},a_{\mathrm{init}}]$ to reduce $h_T$. (b) Log-barrier $R_{\mathrm{LB}}(W)=\frac{1}{mn}\sum_{i,j}\log\max\{1,\frac{a_{\mathrm{init}}}{|W_{ij}|+\epsilon_{\mathrm{lb}}}\}$ is added during early training. It is zero in outer regions and increases near the boundary. The total loss $\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{task}}+\lambda(t)\sum_{l\in\mathcal{M}}R_{\mathrm{LB}}(W^{(l)})$ uses a $\lambda(t)$ that warms up and decays to 0, suppressing early re-entry probability $g_T$.
    - **Design Motivation**: Independent knobs for $h_T$ (initialization) and $g_T$ (early dynamics) allow "lock-in" to be upgraded from a default behavior to a tunable training parameter.

### Loss & Training
In addition to the task loss, only one term $\lambda(t)\sum_{l\in\mathcal{M}}R_{\mathrm{LB}}(W^{(l)}; a_{\mathrm{init}}, \epsilon_{\mathrm{lb}})$ is added. $\lambda(t)$ remains constant during warmup then decays to 0. The template $T^{(l)}$ is used only at initialization; signs are not explicitly constrained during training, relying entirely on the geometric tail law for preservation.

## Key Experimental Results

### Main Results

| Task / Data | Metric | Vanilla SVD on raw $W$ | Hashing / 1-bit baselines | **SVD on $\lvert W_{\mathrm{lockin}}\rvert$ (Ours)** |
|:---|:---|:---|:---|:---|
| CharLM | PPL (Lower is better, $<1$ bpw) | Sharp rise / Stalls at 1 bpw | Stalls near 1 bpw | Continues to drop, significantly best in sub-bit |
| Text8-Char | PPL | Same as above | Same as above | Same as above |
| DBPedia14 | Classification Acc (Higher is better) | Collapses near 1 bpw | Cap at 1 bpw | Remains competitive in sub-bit |

(Numerical values in Figure 8; $\hat h$ and $\hat g$ decrease monotonically with model scale from 30M to 10B. $\hat g$ approaches 0 on the largest model, meaning LLMs have naturally stronger lock-in.)

### Ablation Study

| Configuration | mean flip rate | Var. PPL Change | Note |
|:---|:---|:---|:---|
| baseline (Std Init + No Reg) | $\sim 10^{-1}$ | Reference | Sign matrix nearly impossible to low-rank approx |
| Gap init only ($a_{\mathrm{init}}$ moderate) | Moderate drop | Nearly unchanged | $h_T$ reduced, $g_T$ unchanged |
| Log-barrier only ($\lambda$ large) | Significant drop | Slight increase | $g_T$ reduced, $h_T$ unchanged |
| **gap + log-barrier (Pareto Front)** | $\sim 10^{-3}$ | Only $\approx +1$ ppl | Combination locks sign structure |

### Key Findings
- The geometric tail law $\mathbb{P}[K_T^{\mathrm{eff}}\ge k]\approx \hat h\hat g^{k-1}$ is verified by semi-log tail plots across different learning rates; varying lr changes the effective step size $\Delta$, affecting tail coefficients but not the geometric shape.
- Lock-in strength order: inverse decay $\to$ cosine $\to$ exponential $\to$ constant learning rate (decreasing strength). ReLU homogeneity, normalization layers, and larger batch/model scales all enhance lock-in.
- After template + gap + log-barrier training, the magnitude matrix low-rank structure remains consistent with the baseline (Figure 7), but the sign matrix shifts from "nearly irreducible" to "clearly low-rank," proving no loss in magnitude compressibility.

## Highlights & Insights
- Resolves the paradox of "sign matrices looking like noise" vs "stationary coordinate-wise sign trajectories" using stopping times. Marginal distributions are averages of rare boundary events; trajectories are locked by the $\sigma_k$ sequence.
- Mapping geometric tail parameters $(h_T, g_T)$ to engineering knobs (rejection sampling for $h_T$, log-barrier for $g_T$) shifts the "post-training quantization" paradigm toward "pre-training template selection + training boundary clamping."
- Naturally stronger lock-in in large models (Figure 4 shows $\hat g\to 0$ for 10B) suggests sub-bit compression becomes more feasible as scale increases, contrary to the intuition that larger models are harder to compress.

## Limitations & Future Work
- **From-scratch only**: Templates must be chosen before optimization; cannot be directly applied to existing checkpoints. A post-training version remains an open problem.
- **Sign floating mode**: Under extremely strong magnitude regularization (Appendix D.6), weights are sucked toward 0, and the geometric tail law no longer holds.
- **Enforcement strategies**: Only log-barriers were tested; other strategies like triangular barriers, stop-gradients, or sign-STE were not compared.
- **Expressivity**: The potential contribution of sign bits as "degrees of freedom" for representation is not discussed; fixing signs might limit the performance ceiling for certain tasks.

## Related Work & Insights
- **vs. Classic Post-Training Quantization (GPTQ / AWQ etc.)**: They attempt to manipulate post-training $S$ and $A$ but hit the "one-bit wall" where $S$ behaves like noise. This work intervenes pre-training to bypass the wall.
- **vs. 1-bit / Sign-SGD**: These treat signs as the core carriers of information. This work suggests training rarely changes signs, so "preserving signs" is more efficient than "training signs."
- **vs. SDE / Diffusion views of SGD (Mandt et al.)**: Prior works analyze average trajectories; this work focuses on first-passage and rare events, providing a new way to induce or suppress specific dynamic events in engineering.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Connects stopping time analysis, spectral randomness, and sub-bit compression for the first time with actionable knobs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers MLP/CNN/Transformer and 30M to 10B scales, though downstream tasks are mostly language/text-based.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous definitions and propositions; main logic is clear, though the appendix is dense.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the "one-bit wall" in sub-bit compression with a first-principles explanation, holding high engineering value for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[ICML 2026\] Automatic Unsupervised Ensemble Outlier Model Selection–Extended Version](automatic_unsupervised_ensemble_outlier_model_selection--extended_version.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)

</div>

<!-- RELATED:END -->
