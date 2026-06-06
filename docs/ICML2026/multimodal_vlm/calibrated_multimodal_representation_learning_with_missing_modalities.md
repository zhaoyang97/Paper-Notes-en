---
title: >-
  [Paper Note] Calibrated Multimodal Representation Learning with Missing Modalities
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Alignment] Addressing the practical scenario of training unified multimodal alignment using partial data (e.g., V-T, A-T)…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Alignment"
  - "Missing Modality"
  - "anchor shift"
  - "Probabilistic PCA"
  - "EM Algorithm"
date: 2026-05-08
content_hash: 050c46c820981cd8
---

# Calibrated Multimodal Representation Learning with Missing Modalities

**Conference**: ICML 2026  
**arXiv**: [2511.12034](https://arxiv.org/abs/2511.12034)  
**Code**: https://github.com/Xiaohao-Liu/CalMRL (Available)  
**Area**: Multimodal VLM / Representation Learning / Missing Modality  
**Keywords**: Multimodal Alignment, Missing Modality, anchor shift, Probabilistic PCA, EM Algorithm

## TL;DR
Addressing the practical scenario of training unified multimodal alignment using partial data (e.g., V-T, A-T), this paper provides theoretical upper and lower bounds for "anchor shift" caused by missing modalities using singular value perturbation. It proposes CalMRL: using a Probabilistic PCA-style generative model to perform closed-form EM imputation for missing modalities in the representation layer, then feeding the combined observed and imputed representations into the SVD alignment objective of GRAM/PMRL. It pushes the cross-modal average Recall@1 on VAST from 44.8 to 54.2 (+9.4).

## Background & Motivation

**Background**: Multimodal alignment, starting from CLIP, has recently evolved into the ImageBind / LanguageBind / VAST / GRAM / TRIANGLE / PMRL lineage—the latter using "the maximum singular value of the GRAM matrix" or similar geometric tools to align all modalities simultaneously to a virtual anchor, achieving stronger multimodal synergy than pair-wise alignment.

**Limitations of Prior Work**: All these "simultaneous alignment" methods assume that **all modalities are present** in the training samples. However, in reality, most public datasets contain only two modalities: ImageNet has only vision+text, AudioSet has only audio+text, and while VAST has four modalities, it only contains 150K samples. To utilize "incomplete data" like V-T or A-T, one must follow the ImageBind approach by fixing one anchor (vision or text) and binding all other modalities to it—which caps the alignment performance at the capacity of the anchor modality.

**Key Challenge**: When all modalities are present, the alignment anchor is a "virtual center" in the modality space. When a modality is missing, the observed modalities can only align to a **local anchor**, leading to an inevitable deviation from the full-modality anchor—which the authors call **anchor shift**. This is essentially a "geometric center bias due to uneven sampling."

**Goal**: To find a **computationally inexpensive, theoretically guaranteed, and provably convergent** way to impute a reasonable representation for missing modalities in training data, minimizing the anchor shift as much as possible.

**Key Insight**: Humans can roughly imagine missing senses based on priors even without sight. This inspired the authors to use a generative model leveraging "observed modalities + intrinsic cross-modal correlations" to perform representation-level imputation for missing modalities, rather than complex pixel-level or token-level synthesis.

**Core Idea**: Model the probability distribution of missing modalities in the representation space as a Probabilistic PCA form with a shared latent variable $\beta$ + modality-specific noise $\rightarrow$ Optimize via two-step iteration (closed-form E-step posterior, closed-form M-step parameters) $\rightarrow$ Perform closed-form completion during inference via $\widehat{\mathbf z}^{m'}=\mathbf W^{m'}\mathbf m+\boldsymbol\mu^{m'}$ $\rightarrow$ Concatenate completed and observed representations into the SVD alignment target of PMRL.

## Method

### Overall Architecture
A two-layer structure: **(1) Generative Model**: For each modality $m$, assume $\mathbf{z}^m=\mathbf{W}^m\bm{\beta}+\bm{\mu}^m+\bm{\epsilon}^m$ ($\bm{\beta}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, $\bm{\epsilon}^m\sim\mathcal{N}(\mathbf{0},(\sigma^m)^2\mathbf{I})$), where all modalities share the latent variable $\bm{\beta}$, with parameters $\widehat{\bm\theta}=\{\mathbf{W}^m, \bm{\mu}^m, \sigma^m\}_{m\in\mathcal{M}}$; **(2) Representation Learning**: Observed modalities are encoded by their respective encoders $\phi^m_{\bm\theta}$, and missing modalities are completed in closed-form via (1) as $\widehat{\mathbf{z}}^{m'}=\mathbf{W}^{m'}\mathbf{m}+\bm{\mu}^{m'}$. Subsequently, $[\mathbf{Z}^\Omega;\widehat{\mathbf{Z}}^{\mathcal{M}/\Omega}]$ are concatenated and fed into the SVD to take the maximum singular value $\lambda_1$ as the alignment objective (PMRL style).

### Key Designs

1.  **Theoretical Characterization of Anchor Shift (Theorem 1)**:
    - **Function**: For the first time, it provides a calculable upper and lower bound for "how detrimental missing modality alignment is," turning the intuition of "why imputation is necessary" into a mathematical fact.
    - **Mechanism**: Let $\mathbf{u}_1, \mathbf{u}_1^\Omega$ be the maximum left singular vectors of the full modality matrix $\mathbf{Z}$ and the observed sub-matrix $\mathbf{Z}^\Omega$ respectively; define $\eta=\sqrt{\sum_{m\in\bar\Omega}\langle\mathbf{u}_1^\Omega,\mathbf{z}^m\rangle^2}$. The anchor shift $\|\mathbf{\Delta}\|=\|\mathbf{u}_1-\mathbf{u}_1^\Omega\|$ is bounded between $\sqrt{2(1-(\sigma_1^\Omega+\eta^2)/\sigma_1)}$ (lower) and $\sqrt{2}\|\mathbf{Z}^{\bar\Omega}\|_2/(\sigma_1-\sigma_2)$ (upper). Corollary 3 further provides a sufficient condition for "strictly smaller shift after imputation": when each imputation error is $\|\widehat{\mathbf{z}}^{m'}-\mathbf{z}^{m'}\|_2\le\varepsilon$, it only requires $\varepsilon<(\sigma_1-\sigma_2)/\sqrt{|\bar\Omega|}\cdot\sqrt{1-(\sigma_1^\Omega+\eta^2)/\sigma_1}$.
    - **Design Motivation**: Elevation of the engineering intuition that "missing modalities are harmful" to the level of SVD perturbation theory, providing clear thresholds that "imputation is useful as long as it is not too poor," endorsing the validity of the entire method.

2.  **Probabilistic PCA-style Shared Latent Variable Generative Model**:
    - **Function**: Uses a lightweight generative model to compress "cross-modal commonalities" into $\bm{\beta}$ and "modality-specific bias" into $\bm{\mu}^m$, allowing any missing modality to be recovered from others.
    - **Mechanism**: $\mathbf{z}^m=\mathbf{W}^m\bm{\beta}+\bm{\mu}^m+\bm{\epsilon}^m$ with the independence assumption $\mathbf{x}^m\perp\mathbf{x}^{m'}|\bm{\beta}$. The model capacity is merely $\{\mathbf{W}^m, \bm{\mu}^m, \sigma^m\}$, which is nearly zero compared to encoders, allowing joint training.
    - **Design Motivation**: Traditional multimodal generation (Diffusion / Flow models) requires retraining large models for imputation. Since this paper only seeks to fill gaps in the **representation layer**, a simple and analyzable Gaussian latent variable model suffices—enabling both closed-form E/M-steps and closed-form imputation formulas $\widehat{\mathbf{z}}^{m'}=\mathbf{W}^{m'}\mathbf{m}+\bm{\mu}^{m'}$.

3.  **Bi-step (EM) Closed-form Optimization + Observed-only Updates**:
    - **Function**: Enables closed-form solutions for each step even when shared latent variables $\bm{\beta}$ couple all modality parameters, using only observed modalities for parameter updates (consistent with real-world data constraints).
    - **Mechanism**: **E-step**: Fix $\widehat{\bm\theta}$, solve for the posterior $p(\bm{\beta}\mid\mathbf{z},\widehat{\bm\theta})=\mathcal{N}(\mathbf{m},\mathbf{V})$, where $\mathbf{V}=[\mathbf{I}+\sum_{m\in\Omega}(\sigma^m)^{-2}\mathbf{W}^{m\top}\mathbf{W}^m]^{-1}$ and $\mathbf{m}=\mathbf{V}\sum_{m\in\Omega}(\sigma^m)^{-2}\mathbf{W}^{m\top}(\mathbf{z}^m-\bm{\mu}^m)$—summing only over observed modalities. **M-step**: Given the posterior, update $\bm{\mu}^m, \mathbf{W}^m, (\sigma^m)^2$ in closed-form (Eq. 6). Corollary 4 uses EM monotonicity to prove $L(\widehat{\bm{\theta}}^{(t+1)})\ge L(\widehat{\bm{\theta}}^{(t)})$, ensuring convergence.
    - **Design Motivation**: Since naive Probabilistic PCA cannot handle parameter coupling caused by "shared $\bm{\beta}$ across different modalities," the authors introduce a variational lower bound + EM-style two-step optimization to bypass this difficulty, with closed-form solutions for each step ensuring negligible training cost.

### Loss & Training
The final loss for the encoders (Eq. 9) is: $\mathcal{L}_{\text{rep}}=-\frac{1}{N}\sum_i[\text{exp}(\lambda_1/\tau)/\sum_j\text{exp}(\lambda_j/\tau)+\text{instance-uniformity}] +\alpha\cdot \text{BCE matching loss}$. The first term maximizes the largest singular value of the GRAM matrix for "global alignment," the second is an instance-level regularizer for $\mathbf{u}_1$, and the matching loss ($\alpha=0.1$) is calculated only on observed modalities. Backbone = VAST (vision+caption+audio+subtitle 4 modalities). Training flow: Warm-up on full-modality VAST-150K, then continue training on **modality-incomplete** datasets MSR-VTT (V-T) and AudioCaps (A-T).

## Key Experimental Results

### Main Results (Table 1, Recall@1, ↑ indicates continued training on incomplete data)

| Method | MSR-VTT (T→V/V→T) | AudioCaps (T→A/A→T) | Avg. |
|---|---|---|---|
| VAST (baseline) | 50.5 / 49.0 | 33.7 / 32.2 | 44.8 |
| GRAM↑ | 59.7 / 57.2 | 49.1 / 51.7 | 52.9 |
| TRIANGLE↑ | 57.6 / 58.4 | 48.3 / 51.7 | 51.6 |
| PMRL↑ | 60.1 / 59.2 | 50.4 / 52.0 | 53.8 |
| **CalMRL↑** | **61.1 / 61.1** | **50.1 / 51.0** | **54.2 (+9.4)** |

(Classification tasks Table 2: CalMRL averages 45.19, performing better than PMRL 44.04 and ImageBind 42.08.)

### Ablation Study (Simplified, based on MSR-VTT V-T continued training)

| Configuration | Key Metrics | Description |
|---|---|---|
| Observed modality alignment only (PMRL↑) | Avg 53.8 / large shift | No imputation, baseline |
| CalMRL Imputation (Full) | **Avg 54.2 / small shift** | Complete method |
| Only $S_{\text{param}}$ or $S_{\text{task}}$ missing | – | (Table 3: CalMRL achieves +5.9–10.6 Recall@1 on single dataset) |
| Random noise imputation (Random) | Significantly higher MSE | Validates that imputation is not just random padding |
| Full modal (oracle) | 5↑ | Provided "ideal" upper bound for reference |

Figure 4 visualizes the anchor shift $\|\mathbf{\Delta}\|$ comparison (w/o calibration vs. w/ calibration): CalMRL significantly reduces the shift, and Figure 5 shows CalMRL $\approx$ the ideal upper bound of full-modality training.

### Key Findings
- Under both V-T and A-T continued training settings where "only one type of modality is supplemented," CalMRL significantly outperforms PMRL/GRAM/TRIANGLE, indicating that the gain of imputation for SVD alignment is real and not a side effect of increased data.
- Figure 3 (MSE between real and imputed): The MSE of imputed representations is significantly lower than the "random" baseline, verifying that the generative model successfully learns cross-modal mappings.
- Figure 4: Anchor shift narrows significantly after calibration; Figure 5 shows calibrated performance is close to the full-modality "ideal," confirming Theoretical Corollary 3.

## Highlights & Insights
- For the first time, "why missing modality alignment fails" is articulated using SVD perturbation theory (Davis–Kahan style)—the anchor shift bounds provide an analytical framework directly usable by future works.
- Using an "old tool" like Probabilistic PCA for imputation seems naive but perfectly addresses the need for "representation-level imputation without training large generative models." The closed-form E/M-steps allow training with almost zero additional overhead.
- The EM logic of "calculating posterior with observed, imputing missing with posterior" can be directly transferred to any scenario where "alignment anchors are contaminated by sampling bias," such as unbalanced contrastive learning or multi-task representation learning.

## Limitations & Future Work
- The model assumes $\bm{\beta}$ and $\bm{\epsilon}^m$ are Gaussian, which is a strong simplification for high-dimensional semantic representations; imputation quality may be limited if real cross-modal relationships are highly non-linear.
- Solving for the posterior $\mathbf{m}, \mathbf{V}$ requires summation of $(\sigma^m)^{-2}\mathbf{W}^{m\top}\mathbf{W}^m$ across all observed modalities. When the number of modalities $k$ or dimensionality $d$ is very large, the cost of inverting $\mathbf{V}^{-1}$ cannot be ignored.
- Experiments only cover V/T/A/Subtitle 4 modalities; applicability to heterogeneous modalities like IMU, point clouds, or 3D is not yet verified.
- The imputation error bound $\varepsilon$ is based on average MSE; there is no explicit protection against alignment collapse caused by imputation failure on individual "abnormal prompts."

## Related Work & Insights
- **vs ImageBind / LanguageBind**: They fix one anchor modality (vision/text) and freeze its encoder, restricted by the anchor modality's capacity; CalMRL does not fix an anchor and uses imputation to complete missing modalities for full alignment.
- **vs PMRL / GRAM**: Also performs "maximum singular value of SVD" alignment, but the former requires all modalities to be complete; CalMRL extends this to missing modality scenarios with a +0.4–2 Recall@1 gain in V-T training.
- **vs Traditional Multiview Methods (CCA)**: CCA also uses SVD but is pair-wise; CalMRL performs joint alignment for all modalities + missing completion, theoretically covering any $|\Omega|<k$.

## Rating
- Novelty: ⭐⭐⭐⭐ SVD perturbation analysis of anchor shift is a new perspective; using Probabilistic PCA for representation imputation is simple but symptomatic.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 retrieval datasets + 4 classification datasets, including both single and dual modality continued training settings; however, scalability to more modalities (e.g., adding IMU) is not verified.
- Writing Quality: ⭐⭐⭐⭐⭐ From anchor shift intuition $\rightarrow$ Theorem $\rightarrow$ EM $\rightarrow$ Convergence proof, the logical chain is complete. One image in Figure 1 explains the core problem clearly.
- Value: ⭐⭐⭐⭐ Enables a massive amount of "only two-modality" existing datasets to be used by simultaneous alignment methods, potentially leveraging huge public data like ImageNet and AudioSet to expand the scale of unified multimodal pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](../../ICCV2025/multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICML 2026\] AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning](aoept_breaking_the_implicit_modality-reduction_bottleneck_in_modality-missing_pr.md)
- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](../../CVPR2026/multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[ICML 2026\] DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement](diva_harnessing_the_representation_divergence_in_unified_multimodal_models_for_m.md)

</div>

<!-- RELATED:END -->
