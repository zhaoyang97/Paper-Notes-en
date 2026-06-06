---
title: >-
  [Paper Note] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions
description: >-
  [ICML 2026][Image Generation][Diffusion Models] This paper proves that when the data distribution is supported on a Union of Subspaces (UoS) consisting of $M$ low-dimensional linear subspaces with sub-gaussian distributi…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Sample Complexity"
  - "Union of Subspaces"
  - "Multi-modal Distributions"
  - "Minimax Optimality"
date: 2026-05-08
content_hash: 60d388e8ae29e82c
---

# Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions

**Conference**: ICML 2026  
**arXiv**: [2605.30153](https://arxiv.org/abs/2605.30153)  
**Code**: None (Theoretical paper, contains synthetic numerical validation)  
**Area**: Diffusion Models / Statistical Theory of Generative Models  
**Keywords**: Diffusion Models, Sample Complexity, Union of Subspaces, Multi-modal Distributions, Minimax Optimality  

## TL;DR
This paper proves that when the data distribution is supported on a Union of Subspaces (UoS) consisting of $M$ low-dimensional linear subspaces with sub-gaussian distributions within each, a kernel-density-based score estimator allows the score-based diffusion sampler to reach a 1-Wasserstein error $\varepsilon$ using $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ samples (where $k$ is the maximum intrinsic dimension). This achieves a minimax optimal rate matching the intrinsic dimension for the first time under multi-modal settings without assumptions of smoothness, bounded density, or log-concavity, effectively overcoming the curse of dimensionality.

## Background & Motivation

**Background**: Score-based diffusion models have achieved SOTA performance in high-dimensional generation tasks such as images, video, and language. The mechanism involves adding noise in a forward OU process and then denoising using a learned score function $s_t^\star=\nabla\log p_t$. Recent theoretical work has investigated the sample complexity required for a diffusion sampler to approximate a target distribution within $\varepsilon$ accuracy. Current state-of-the-art results for $\beta$-Hölder smooth $d$-dimensional distributions provide a sample complexity of $\varepsilon^{-(d+2\beta)/\beta}$ (Zhang 2024, Cai & Li 2025), which is minimax optimal but clearly dominated by the curse of dimensionality.

**Limitations of Prior Work**: This worst-case rate fails to explain the actual performance of diffusion models on real-world high-dimensional data. Existing works leveraging "intrinsic low-dimensional structures" (Chen 2023, Azangulov 2024, Tang & Yang 2024, etc.) improve the rate to depend only on the intrinsic dimension, but at the cost of requiring the distribution to satisfy: (i) support on a single low-dimensional subspace/manifold; (ii) density bounded away from zero on the support; (iii) global smoothness or log-concavity of the score. These conditions naturally exclude multi-modal distributions, where the density between modes necessarily approaches zero and the score diverges.

**Key Challenge**: Real high-dimensional data (e.g., image or text clusters) is almost certainly multi-modal, with different modes often residing on different low-dimensional structures ("union of manifolds hypothesis"). Existing theories either suffer from the curse of dimensionality or adopt assumptions that exclude multi-modality to gain intrinsic dimensionality—neither path explains empirical phenomena.

**Goal**: Remove assumptions regarding density smoothness, boundedness, or log-concavity, allowing the diffusion sample complexity rate to depend only on the intrinsic dimension $k$ while accommodating multi-modal geometry.

**Key Insight**: Data distributions are modeled as a UoS—$\mathsf{supp}(p^\star)\subseteq \cup_{i=1}^M V_i$, where each $V_i$ is a $k_i$-dimensional linear subspace and the restricted distribution $p_i^{\mathsf{low}}$ on $V_i$ is $\sigma$-subgaussian. This provides a minimal model that is both multi-modal and low-dimensional. The key technical observation is that on each subspace, the smoothed score can be decomposed into "normal (analytic Gaussian) + tangential ($k_i$-dimensional sub-problem)" components—the former is computable in closed form, and the latter is a low-dimensional score estimation in $k_i$ dimensions.

**Core Idea**: Perform subspace clustering to assign samples to each $V_i$, then construct low-dimensional score estimators within each $V_i$ using kernel density estimation (KDE) with thresholding. Finally, use mixture weights to reassemble the components back into $d$ dimensions. This estimator is purely analytic and does not depend on neural networks; the objective is to "prove achievability," thereby establishing the statistical limit of diffusion sampling at $\varepsilon^{-(k\vee 2)}$.

## Method

### Overall Architecture

**Input**: $n$ i.i.d. samples from $p^\star$. The process consists of two stages:

1.  **Subspace Recovery**: Use $n_0=C_{\mathsf{sc}}M^2k\log n$ samples to run standard subspace clustering (SSC, thresholded clustering, or greedy clustering) to recover the orthonormal bases $\{A_i\}_{i=1}^M$ ($A_i\in\mathbb{R}^{d\times k_i}$, $A_i^\top A_i=I_{k_i}$) and a classification function $c:\mathbb{R}^d\to[M]$. Under UoS + standard separation conditions, this step succeeds with high probability, and the required sample size is negligible relative to the total budget.
2.  **Score Estimation**: Use the remaining $N=n-n_0$ samples to construct a score estimator $\widehat s_t$ to approximate the target $s_t^\star=\nabla\log p_t$ (where $p_t=p^\star * \mathcal{N}(0,tI_d)$) for all $t>0$.

After obtaining $\widehat s_t$, it is plugged into a standard reverse SDE sampling algorithm (Algorithm 1: initialized at $\mathcal{N}(0,I_d)$, integrated backward to an early-stopping time $\tau$). The theoretical analysis relies on the equivalence mapping between the OU process $X_t$ and the variance exploding process $Z_t$ via $h(t)=\sigma_t^2/c_t^2$.

### Key Designs

1.  **Normal-Tangential Decomposition and Mixture Representation**:
    - **Function**: Reduces $d$-dimensional score estimation into $M$ sub-problems of $k_i$ dimensions.
    - **Mechanism**: Starting from the UoS assumption $p^\star=\sum_i p_i^\star$, the smoothed density is $p_t(x)=\sum_i\int_{V_i}\varphi_t(x-y;d)p_i^\star(\mathrm{d}y)$. The corresponding score decomposes as $s_t^\star(x)=\sum_i w_t(i,x)\cdot s_t(i,x)$, where $w_t(i,x)$ is the posterior weight that $x$ originates from the $i$-th subspace. A key lemma (following Chen et al. 2023) is $s_t(i,x)=-\tfrac{1}{t}(x-\mathsf{proj}_i(x))+A_i\,s_t^{\mathsf{low}}(i,A_i^\top x)$: the first term is the normal displacement from the subspace (closed form), and the second is the score of the $k_i$-dimensional distribution $p_i^{\mathsf{low}}*\mathcal{N}(0,tI_{k_i})$.
    - **Design Motivation**: This decomposition moves all "high-dimensional difficulties" to the closed-form normal part and compresses "statistical difficulties" into intrinsic $k_i$-dimensional sub-problems, avoiding the curse of dimensionality for the $d$-dimensional score.

2.  **KDE + Adaptive Thresholding Low-Dimensional Score Estimator**:
    - **Function**: Estimates $s_t^{\mathsf{low}}(i,\cdot)$ on each $V_i$ with a minimax optimal rate, robust across all time scales $t$.
    - **Mechanism**: First, a Gaussian KDE $\widehat g_t(i,x)=\frac{1}{|\mathcal{C}_i|}\sum_{j\in\mathcal{C}_i}\varphi_t(x-A_i^\top X^{(j)};k_i)$ is applied to subsets $\mathcal{C}_i=\{j:X^{(j)}\in V_i\}$. A plug-in ratio $\nabla\widehat g_t/\widehat g_t$ is then used. Crucially, two layers of regularization are added: an indicator threshold $\psi(\widehat g_t;\eta_t)$ zeros the estimate in low-density regions ($\eta_t=\frac{\log N}{N(2\pi t)^{k_i/2}}$, adaptive to $t$), and a radius $R=\sqrt{2\log N/t}$ is used to clip the values. This yields $\widehat s_t^{\mathsf{low}}(i,x)=\mathsf{clip}_R\!\big(\tfrac{\nabla\widehat g_t}{\widehat g_t}\psi(\widehat g_t;\eta_t)\big)$.
    - **Design Motivation**: Plug-in score estimates oscillate violently in low-density regions where the denominator is near zero—a pathology typical of the "gaps" in multi-modal distributions. Thresholding declares that the estimator "gives up" when data is too sparse, controlling the second moment and aligning the estimator complexity with the minimax bound. Under this, $\mathbb{E}[\|\widehat s_t-s_t^\star\|_2^2]=\widetilde{O}(\tfrac{1}{N}(\tfrac{1}{t}+\tfrac{\sigma^{k\vee 2}}{t^{(k\vee 2)/2+1}}))$.

3.  **Geometric Gating for Mixture Weight Estimation**:
    - **Function**: Controls weight estimation error to depend only on the intrinsic dimension when reassembling the $d$-dimensional score.
    - **Mechanism**: For $w_t(i,x)=q_t(i,x)/p_t(x)$, the numerator and denominator are estimated via KDE. A "geometric gate" $\mathds{1}_{\{x\in\mathcal{G}_t(i)\}}$ is applied, where $\mathcal{G}_t(i)=\{x:\|x-\mathsf{proj}_i(x)\|_2\le R_t(i)\}$. Lemma 1 proves that the MSE bound for this point-wise $x$ estimate (containing an $e^{-\|x-\mathsf{proj}_i(x)\|^2/(2t)}$ decay factor) integrates over the sub-gaussian band $\mathcal{B}_t$ such that the $d$-dimensional Gaussian weights are eliminated by the normal decay of $V_j$, leaving only terms related to intrinsic dimensions.
    - **Design Motivation**: Naive weight estimation MSE would include the $t^{-d/2}$ factor from $d$-dimensional KDE, ruining the intrinsic dimension benefits. Geometric gating and point-wise bounds compress the contribution of $x$ far from $V_i$ exponentially, ensuring the rate is independent of $d$.

### Loss & Training

The method requires no neural network training or gradient descent; all estimators are explicit formulas. The required "hyperparameters" are the threshold $\eta_t$ (determined by $N, t, k_i$), the clip radius $R$, and the geometric gate $R_t(i)$. The sampling stage uses the reverse SDE from Algorithm 1 with early-stopping $\tau=n^{-2/k}$ and total time $T=\log n$, providing an end-to-end $W_1$ error. The paper notes that NN-based score training should follow the path of "subspace clustering followed by score fitting on low-dimensional latents," for which this construction provides a theoretical target.

## Key Experimental Results

### Main Results: Theoretical Rate Comparison

| Setting | Prev. SOTA Theory | Ours | Key Difference |
| :--- | :--- | :--- | :--- |
| $d$-dim $\beta$-Hölder smooth density | $\varepsilon^{-(d+2\beta)/\beta}$ (Zhang 2024 / Cai & Li 2025) | — | Baseline, dominated by $d$ |
| Single $k$-dim subspace + density lower bound | $\varepsilon^{-O(k)}$ (with extra conditions) | $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ | Removed density bound and smoothness |
| UoS Multi-modal + intrinsic sub-gaussian | None (multi-modality breaks density bounds) | $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ | First to reach minimax optimal intrinsic rate |
| Minimax lower bound ($k$-dim KDE) | $n^{-1/(k\vee 2)}$ | Matches | Ours is (near-)minimax optimal |

Two specific main bounds:
- **Score Estimation (Theorem 1)**: $\mathbb{E}[\|\widehat s_t(X)-s_t^\star(X)\|_2^2]\le C\cdot\tfrac{dM^3}{N}\big(\tfrac{1}{t}+\tfrac{\sigma^{k\vee 2}}{t^{(k\vee 2)/2+1}}\big)\mathsf{polylog}\,N$. The $t$-dependence follows $k$, while $d$ only appearing as a prefactor.
- **Sampling (Theorem 2)**: For $T=\log n, \tau=n^{-2/k}$, $\mathbb{E}[W_1(p^\star,p_{\widehat Y_{T-\tau}})]\le C\cdot dM^{3/2}n^{-1/(k\vee 2)}\mathsf{polylog}\,n$. Achieving $\varepsilon$ accuracy requires only $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ samples, decoupled from the ambient dimension $d$.

### Ablation Study (Synthetic Data)

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full Method | $d=48, M=128, k=3, N=50{,}000$ | Empirical $L^2$ score error vs $t$ curve aligns exactly with $1/N\cdot(1/t+\sigma^k/t^{k/2+1})$. |
| No Normal-Tangential decomp. | Degenerates to $d$-dim KDE | $t^{-d/2}$ factor absorbs all signals; intrinsic rate $t^{-k/2}$ is impossible. |
| No thresholding $\psi$ | Plug-in $\nabla\widehat g/\widehat g$ | Second moment explodes in multi-modal gaps; bounds fail. |
| No geometric gate $\mathcal{G}_t(i)$ | Naive KDE ratio | Weight MSE carries $t^{-d/2}$; prefactor degrades from $d$ to $\exp(d)$. |

### Key Findings
- On synthetic data with $d=48$ but $k=3$, the empirical $L^2$ score error dependence on $t$ is much milder than $t^{-d/2}$ and matches the theoretical $t^{-(k\vee 2)/2-1}$, refuting the worst-case argument that high ambient dimensions necessitate high score errors.
- The rate $n^{-1/(k\vee 2)}$ where $k\vee 2$ appears stems from the fact that for $k=1$, the minimax lower bound for density estimation is $n^{-1/2}$—this is a statistical limit, not a weakness of the proof.
- The prefactors $dM^{3/2}$ and $dM^3$ are acknowledged as artifacts of the analysis; finer union bounds could potentially remove the $M$ dependence.

## Highlights & Insights
- The combination of "Normal-Tangential decomposition + Mixture Weights + Thresholded KDE" marks the first time the statistical limit of diffusion has been fixed to the intrinsic dimension in UoS multi-modal scenarios. It is particularly noteworthy that it requires only sub-gaussianity, without any smoothness or lower bounds on density.
- The paper clarifies that the kernel-based estimator is a theoretical proof device to establish achievability, not a replacement for NN-based estimators. However, the low-dimensional approximation targets it provides can serve as specific goals for future analysis of NN capacity and approximation error.
- Adaptive thresholding $\psi(\widehat g_t;\eta_t)$—giving up in low-density regions—could inspire practical score network training: instead of forcing the network to learn in regions with low SNR/unstable KDE, one could explicitly ignore these regions in the loss or adjust the noise schedule.

## Limitations & Future Work
- The results assume exact subspace recovery; while a sketch is provided for noisy/approximate subspaces in Section 6, the end-to-end rate is not yet formalized.
- The linear dependence on $d$ and $M$ in the prefactors are artifacts that make constants large when $M$ is substantial.
- The theory covers the OU forward process and continuous-time reverse SDE; numerical discretization errors (DDPM/DDIM) are not yet integrated into the end-to-end $W_1$ bound.
- Extending from linear "subspace unions" to "manifold unions" is the natural next step.

## Related Work & Insights
- **vs. Zhang 2024 / Cai & Li 2025**: They achieve $\varepsilon^{-(d+2\beta)/\beta}$ (minimax optimal for general density); this work achieves $\varepsilon^{-(k\vee 2)}$ (intrinsic dimension optimal) by trading generality for UoS structure.
- **vs. Chen et al. 2023**: Inherits the decomposition tool but removes the "density lower bound" and "smooth score" requirements while extending from a single subspace to a union.
- **vs. Azangulov 2024 / Tang & Yang 2024**: They handle manifolds but require density bounds; this work trades manifold geometry for UoS to bypass density bounds and include multi-modality.
- **Insight**: Thresholded KDE's "zeroing low-density areas" can migrate to score distillation and handling sparse categories in conditional generation. The UoS hypothesis can serve as a standard toy model for analyzing diffusion on datasets with "intra-class low-dimensionality + inter-class separation."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to accommodate multi-modality, achieve minimax intrinsic optimality, and remove density smoothness/lower bound assumptions.
- **Experimental Thoroughness**: ⭐⭐⭐ As a theoretical paper, it lacks extensive empirical validation beyond synthetic $L^2$ tests.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Assumptions, theorems, and proof sketches are clearly articulated.
- **Value**: ⭐⭐⭐⭐⭐ Advances the theoretical explanation of why diffusion works in high dimensions to the level of multi-modality and low-dim structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](../../NeurIPS2025/image_generation/understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](../../NeurIPS2025/image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)

</div>

<!-- RELATED:END -->
