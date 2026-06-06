---
title: >-
  [Paper Note] Calibrated Multimodal Representation Learning with Missing Modalities
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Alignment] Addressing the practical scenario of "training unified multimodal alignment with partial modality data such as V-T, A-T…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Alignment"
  - "Missing Modalities"
  - "Anchor Shift"
  - "Probabilistic PCA"
  - "EM Algorithm"
date: 2026-05-08
content_hash: 9b469dd552897f33
---

# Calibrated Multimodal Representation Learning with Missing Modalities

**Conference**: ICML 2026  
**arXiv**: [2511.12034](https://arxiv.org/abs/2511.12034)  
**Code**: https://github.com/Xiaohao-Liu/CalMRL (available)  
**Area**: Multimodal VLM / Representation Learning / Missing Modalities  
**Keywords**: Multimodal Alignment, Missing Modalities, Anchor Shift, Probabilistic PCA, EM Algorithm

## TL;DR
Addressing the practical scenario of "training unified multimodal alignment with partial modality data such as V-T, A-T," this work theoretically establishes upper and lower bounds for "anchor shift caused by missing modalities" via singular value perturbation, and proposes CalMRL: a probabilistic PCA-style generative model performs closed-form EM imputation for missing modalities at the representation level, then feeds both observed and imputed representations into the SVD alignment objective of GRAM/PMRL. On VAST, cross-modal average Recall@1 is improved from 44.8 to 54.2 (+9.4).

## Background & Motivation

**Background**: Multimodal alignment, starting from CLIP, has recently evolved into the ImageBind / LanguageBind / VAST / GRAM / TRIANGLE / PMRL family—these later works use "maximum singular value of the GRAM matrix" or similar geometric tools to align all modalities to a virtual anchor simultaneously, achieving stronger multimodal synergy than pair-wise alignment.

**Limitations of Prior Work**: All these "simultaneous alignment" methods assume **all modalities are present** in every training sample. However, in reality, most public datasets only have two modalities: ImageNet has only vision+text, Audioset only audio+text, and VAST has four modalities but only 150K samples. To leverage more "incomplete" data like V-T or A-T, one must fix an anchor (vision or text) as in ImageBind and align all other modalities to it—this limits the alignment upper bound to the anchor modality's capacity.

**Key Challenge**: When all modalities are present, the alignment anchor is a "virtual center" in the modality space; if a modality is missing, the observed modalities can only align to a **local anchor**, resulting in an unavoidable offset from the full-modality anchor—termed **anchor shift** by the authors. This is essentially a "sampling-induced geometric center bias."

**Goal**: Given training data with missing modalities, find a **computationally efficient, theoretically guaranteed, and provably convergent** way to impute reasonable representations for missing modalities, minimizing anchor shift.

**Key Insight**: Humans can roughly infer missing information based on priors even without direct observation—this inspires the use of generative models leveraging "observed modalities + intrinsic inter-modal relationships" to impute missing modalities at the representation level, rather than complex pixel- or token-level synthesis.

**Core Idea**: Model the probability distribution of missing modalities in representation space as a probabilistic PCA form with shared latent variable $\beta$ + modality-specific noise → optimize via two-step iteration (E-step closed-form posterior, M-step closed-form parameters) → at inference, use $\widehat{\mathbf z}^{m'}=\mathbf W^{m'}\mathbf m+\boldsymbol\mu^{m'}$ for closed-form imputation → concatenate imputed and observed representations and feed into the SVD alignment objective of PMRL.

## Method

### Overall Architecture
Two-layer structure: **(1) Generative Model** assumes for each modality $m$, $\mathbf{z}^m=\mathbf{W}^m\bm{\beta}+\bm{\mu}^m+\bm{\epsilon}^m$ ($\bm{\beta}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, $\bm{\epsilon}^m\sim\mathcal{N}(\mathbf{0},(\sigma^m)^2\mathbf{I})$), with all modalities sharing latent variable $\bm{\beta}$ and parameters $\widehat{\bm\theta}=\{\mathbf{W}^m, \bm{\mu}^m, \sigma^m\}_{m\in\mathcal{M}}$; **(2) Representation Learning**: observed modalities are encoded by their respective encoders $\phi^m_{\bm\theta}$, missing modalities are imputed via (1) closed-form $\widehat{\mathbf{z}}^{m'}=\mathbf{W}^{m'}\mathbf{m}+\bm{\mu}^{m'}$, then $[\mathbf{Z}^\Omega;\widehat{\mathbf{Z}}^{\mathcal{M}/\Omega}]$ are concatenated and the largest singular value $\lambda_1$ is taken as the alignment objective (PMRL style).

### Key Designs

1. **Theoretical Characterization of Anchor Shift (Theorem 1)**:

    - Function: For the first time, provides computable upper and lower bounds for "how bad missing modality alignment can be," turning the intuition for imputation into a mathematical fact.
    - Mechanism: Let $\mathbf{u}_1, \mathbf{u}_1^\Omega$ be the leading left singular vectors of the full modality matrix $\mathbf{Z}$ and observed submatrix $\mathbf{Z}^\Omega$ respectively; define $\eta=\sqrt{\sum_{m\in\bar\Omega}\langle\mathbf{u}_1^\Omega,\mathbf{z}^m\rangle^2}$. Then anchor shift $\|\mathbf{\Delta}\|=\|\mathbf{u}_1-\mathbf{u}_1^\Omega\|$ is bounded below by $\sqrt{2(1-(\sigma_1^\Omega+\eta^2)/\sigma_1)}$ and above by $\sqrt{2}\|\mathbf{Z}^{\bar\Omega}\|_2/(\sigma_1-\sigma_2)$. Corollary 3 further gives a sufficient condition for "imputation strictly reduces shift": if each imputation error $\|\widehat{\mathbf{z}}^{m'}-\mathbf{z}^{m'}\|_2\le\varepsilon$, then $\varepsilon<(\sigma_1-\sigma_2)/\sqrt{|\bar\Omega|}\cdot\sqrt{1-(\sigma_1^\Omega+\eta^2)/\sigma_1}$ suffices.
    - Design Motivation: Elevates the engineering intuition that "missing modalities are harmful" to the level of SVD perturbation theory, and provides a clear threshold for "imputation is guaranteed to help as long as it's not too poor," supporting the method's validity.

2. **Probabilistic PCA-Style Shared Latent Variable Generative Model**:

    - Function: Uses a lightweight generative model to compress "inter-modal commonality" into $\bm{\beta}$ and "modality-specific bias" into $\bm{\mu}^m$, enabling recovery of any missing modality from others.
    - Mechanism: $\mathbf{z}^m=\mathbf{W}^m\bm{\beta}+\bm{\mu}^m+\bm{\epsilon}^m$ with independence assumption $\mathbf{x}^m\perp\mathbf{x}^{m'}|\bm{\beta}$. Model capacity is only $\{\mathbf{W}^m, \bm{\mu}^m, \sigma^m\}$, negligible compared to the encoder, and can be trained jointly.
    - Design Motivation: Traditional multimodal generation (diffusion/flow models) requires retraining large models for imputation; here, only representation-level imputation is needed, so a simple, analytically tractable Gaussian latent variable model suffices—enabling closed-form E/M-steps and imputation formula $\widehat{\mathbf{z}}^{m'}=\mathbf{W}^{m'}\mathbf{m}+\bm{\mu}^{m'}$.

3. **Bi-step (EM) Closed-form Optimization + Update Using Only Observed Modalities**:

    - Function: Despite parameter coupling via shared latent variable $\bm{\beta}$, each step remains closed-form, and parameter updates use only observed modalities (matching real data constraints).
    - Mechanism: **E-step** fixes $\widehat{\bm\theta}$, computes posterior $p(\bm{\beta}\mid\mathbf{z},\widehat{\bm\theta})=\mathcal{N}(\mathbf{m},\mathbf{V})$, where $\mathbf{V}=[\mathbf{I}+\sum_{m\in\Omega}(\sigma^m)^{-2}\mathbf{W}^{m\top}\mathbf{W}^m]^{-1}$, $\mathbf{m}=\mathbf{V}\sum_{m\in\Omega}(\sigma^m)^{-2}\mathbf{W}^{m\top}(\mathbf{z}^m-\bm{\mu}^m)$—summing only over observed modalities. **M-step** updates $\bm{\mu}^m, \mathbf{W}^m, (\sigma^m)^2$ in closed form (Eq. 6). Corollary 4 uses EM monotonicity to prove $L(\widehat{\bm{\theta}}^{(t+1)})\ge L(\widehat{\bm{\theta}}^{(t)})$, ensuring convergence.
    - Design Motivation: Naive probabilistic PCA cannot handle parameter coupling from shared $\bm{\beta}$; the authors introduce a variational lower bound + EM-style bi-step optimization to circumvent this, with closed-form solutions for each step → negligible training cost.

### Loss & Training

Final encoder loss (Eq. 9): $\mathcal{L}_{\text{rep}}=-\frac{1}{N}\sum_i[\text{exp}(\lambda_1/\tau)/\sum_j\text{exp}(\lambda_j/\tau)+\text{instance-uniformity}] +\alpha\cdot \text{BCE matching loss}$, where the first term maximizes the largest singular value of the GRAM matrix for "full alignment," the second is instance-level regularization on $\mathbf{u}_1$, and $\alpha=0.1$ matching loss is computed only on observed modalities. Backbone = VAST (vision+caption+audio+subtitle, 4 modalities); training pipeline: VAST-150K full-modality warm-up, then continued training on MSR-VTT (V-T) and AudioCaps (A-T), both **missing modality** datasets.

## Key Experimental Results

### Main Results (Table 1, Recall@1, ↑ indicates continued training on missing modality data)

| Method | MSR-VTT (T→V/V→T) | AudioCaps (T→A/A→T) | Avg. |
|---|---|---|---|
| VAST (baseline) | 50.5 / 49.0 | 33.7 / 32.2 | 44.8 |
| GRAM↑ | 59.7 / 57.2 | 49.1 / 51.7 | 52.9 |
| TRIANGLE↑ | 57.6 / 58.4 | 48.3 / 51.7 | 51.6 |
| PMRL↑ | 60.1 / 59.2 | 50.4 / 52.0 | 53.8 |
| **CalMRL↑** | **61.1 / 61.1** | **50.1 / 51.0** | **54.2 (+9.4)** |

(Classification task Table 2: CalMRL average 45.19, outperforming PMRL 44.04 and ImageBind 42.08.)

### Ablation Study (based on MSR-VTT V-T continued training, simplified)

| Configuration | Key Metric | Description |
|---|---|---|
| Observed modalities only (PMRL↑) | Avg. 53.8 / shift large | No imputation, baseline |
| CalMRL with imputation (Full) | **Avg. 54.2 / shift small** | Full method |
| Only $S_{\text{param}}$ or $S_{\text{task}}$ missing | – | (Table 3: CalMRL gains +5.9–10.6 Recall@1 in V-T continued training)|
| Random noise imputation (Random) | MSE significantly higher | Shows imputation is not trivially beneficial |
| Full modalities (oracle) | 5↑ | Provides "ideal" upper bound for reference |

Figure 4 directly compares anchor shift $\|\mathbf{\Delta}\|$ (w/o calibration vs. w/ calibration): CalMRL significantly reduces shift, and Figure 5 shows CalMRL ≈ full-modality training's ideal upper bound.

### Key Findings
- In both V-T and A-T continued training (imputing only one modality), CalMRL clearly outperforms PMRL/GRAM/TRIANGLE, indicating the SVD alignment gain is genuine and not a side effect of more data.
- Figure 3 (MSE between real and imputed): imputed representations have much lower MSE than "random" baseline, confirming the generative model learns inter-modal mappings.
- Figure 4: anchor shift is significantly reduced after calibration; Figure 5 shows calibrated performance approaches the full-modality "ideal," confirming Corollary 3.

## Highlights & Insights
- For the first time, "why missing modality alignment fails" is rigorously explained via SVD perturbation theory (Davis–Kahan style)—the anchor shift bounds provide a reusable analytical framework for future work.
- Using probabilistic PCA, though seemingly simple, precisely meets the need for "representation-level imputation without training large generative models"; closed-form E/M-steps make training nearly cost-free.
- The EM approach of "posterior from observed, impute missing from posterior" can be directly transferred to any scenario where "alignment anchor is contaminated by sampling bias," such as imbalanced contrastive learning or multi-task representation learning.

## Limitations & Future Work
- The model assumes both $\bm{\beta}$ and $\bm{\epsilon}^m$ are Gaussian, which is a strong simplification for high-dimensional semantic representations; if inter-modal relationships are highly nonlinear, imputation quality may be limited.
- Solving for posterior $\mathbf{m}, \mathbf{V}$ requires summing over all observed modalities $(\sigma^m)^{-2}\mathbf{W}^{m\top}\mathbf{W}^m$; when the number of modalities $k$ or dimension $d$ is large, inverting $\mathbf{V}^{-1}$ can be costly.
- Experiments only cover V/T/A/Subtitle (4 modalities); applicability to heterogeneous modalities such as IMU, point cloud, or 3D remains untested.
- The imputation error bound $\varepsilon$ is based on average MSE, with no explicit protection against alignment collapse from imputation failures on rare "outlier prompts."

## Related Work & Insights
- **vs ImageBind / LanguageBind**: These fix an anchor modality (vision/text) and freeze its encoder, limited by the anchor's capacity; CalMRL does not fix the anchor, imputes missing modalities, and performs full alignment.
- **vs PMRL / GRAM**: Also use "SVD largest singular value" alignment, but require all modalities to be present; CalMRL extends this to missing modality scenarios and achieves +0.4–2 Recall@1 in V-T continued training.
- **vs CCA-style traditional multi-view methods**: CCA also uses SVD but pair-wise; CalMRL performs joint alignment and imputation for any $|\Omega|<k$.

## Rating
- Novelty: ⭐⭐⭐⭐ The singular value perturbation analysis of anchor shift is a new perspective; using probabilistic PCA for representation-level imputation is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 retrieval and 4 classification datasets, with both single- and dual-modality continued training; however, extensibility to more modalities (e.g., IMU) is untested.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical chain from anchor shift intuition → theorem → EM → convergence proof is complete; Figure 1 clearly illustrates the core problem.
- Value: ⭐⭐⭐⭐ Enables simultaneous alignment methods to leverage large "two-modality-only" datasets, potentially unlocking massive public datasets like ImageNet/AudioCaps and expanding unified multimodal pretraining data scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](../../ICCV2025/multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](../../CVPR2026/multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[ICML 2026\] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)
- [\[ICML 2026\] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)

</div>

<!-- RELATED:END -->
