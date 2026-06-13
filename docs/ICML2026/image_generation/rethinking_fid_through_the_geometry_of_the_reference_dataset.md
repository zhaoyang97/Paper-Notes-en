---
title: >-
  [Paper Note] Rethinking FID Through the Geometry of the Reference Dataset
description: >-
  [ICML 2026][Image Generation][FID] This paper identifies that the "lower is better" assumption of FID systematically fails across different reference datasets. By introducing two geometric descriptors—distribution densit…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "FID"
  - "Generative Evaluation"
  - "Reference Set Geometry"
  - "Distribution Density"
  - "Effective Rank"
date: 2026-05-08
content_hash: c5ba2cc930f7dcb9
---

# Rethinking FID Through the Geometry of the Reference Dataset

**Conference**: ICML 2026  
**arXiv**: [2605.29335](https://arxiv.org/abs/2605.29335)  
**Code**: TBD  
**Area**: Image Generation / Generative Model Evaluation  
**Keywords**: FID, Generative Evaluation, Reference Set Geometry, Distribution Density, Effective Rank

## TL;DR
This paper identifies that the "lower is better" assumption of FID systematically fails across different reference datasets. By introducing two geometric descriptors—distribution density $\langle -\log d_k\rangle$ and effective rank $\mathrm{erank}(A)$—and applying hierarchical linear models, the authors demonstrate that these descriptors explain ~70% of the cross-dataset variance in the "sample quality → FID" slope. This work provides the first quantitative attribution of FID's fragility to the reference set itself.

## Background & Motivation

**Background**: FID, which utilizes Inception-v3 features and the Fréchet distance to measure the discrepancy between generative and reference distributions, has become the de facto standard for evaluating image generation. It serves as the primary benchmark for nearly all diffusion, GAN, and autoregressive model papers.

**Limitations of Prior Work**: Numerous counterexamples have emerged in recent years. Choi et al. (2025) showed that increasing compute for clearer images on COCO unexpectedly worsened FID. Lee et al. (2025) found that tuning hyperparameters for minimum FID yielded samples with the poorest ImageReward. Jayasumana et al. (2024) demonstrated that stronger image perturbations could actually improve FID. These findings suggest that "FID decrease = quality increase" no longer holds reliably in practice.

**Key Challenge**: Previous explanations have primarily blamed the fragility of Inception-v3 features (Kynkäänniemi et al. 2022, Parmar et al. 2022) or the instability of the Fréchet distance in long-tail estimation (Chong & Forsyth 2020). However, the reference dataset itself—a core component of FID—has rarely been scrutinized. Questions remain: How are reference sets selected? Why does FID work for CelebA-HQ but fail for COCO? Are there quantifiable "geometric" differences between them?

**Goal**: To formalize how the reference set shapes FID's behavior and to identify a small set of geometric descriptors that can predict FID's response across different reference sets.

**Key Insight**: FID, as a distribution distance, naturally focuses on two aspects of the reference set: its tightness in the feature space (density) and the number of principal directions it spans (effective dimensionality). Single-modal datasets like CelebA-HQ and multi-category open-domain datasets like COCO occupy different regions of the feature space, which likely results in different FID response directions.

**Core Idea**: Characterize reference set geometry using two scalars: mean kNN log-density and effective rank. Employ a two-level hierarchical linear model to explicitly model the "sample quality → FID" slope as a function of the reference set's geometry, followed by cross-dataset statistical testing of this hypothesis.

## Method

### Overall Architecture
The study fixes a generator (Stable Diffusion 1.5 + DDIM) and generates images across six reference sets with vastly different semantic spans (FFHQ, CelebA-HQ, MJHQ-30K, ImageNet, Flickr30K, COCO). Sample quality is controlled by scanning denoising steps $N \in \{15, 20, \dots, 50\}$, using ImageReward as a quality proxy. The methodology involves: (1) calculating two geometric descriptors for each dataset; (2) using a hierarchical linear model to determine if slope differences are explained by these descriptors. Finally, FID is decomposed into precision/recall to identify the dominant side, and ablations are performed by replacing Inception-v3 with DINOv2 and Fréchet distance with MMD/KID.

### Key Designs

1.  **Two Geometric Descriptors: Density + Effective Rank**:
    - **Function**: Characterize the "shape of the reference set in feature space" using a minimal number of scalars.
    - **Mechanism**: Distribution density is estimated via a log-version of $k$-NN: $\langle -\log d_k\rangle = \frac{1}{n}\sum_i -\log d_k(x_i)$, where $d_k(x_i)$ is the Euclidean distance from the $i$-th sample to its $k$-th nearest neighbor ($k=80$). Standard density estimation $\hat p(x) \propto d_k(x)^{-D}$ is unusable in $D=2048$ dimensions due to extreme numerical ranges, so the average of the log is used. Effective rank $\mathrm{erank}(A) = \exp(H(\bm\sigma/\|\bm\sigma\|_1))$ is the exponential of the Shannon entropy of normalized singular values of the centered feature matrix $A$. It generalizes the rank to a continuous "weighted dimension."
    - **Design Motivation**: To make "geometry" interpretable, it is necessary to distinguish between inherently different distributions like CelebA-HQ (density $-2.36$, erank $1220$) and COCO (density $-2.67$, erank $1337$) using one or two numbers directly computable without semantic labels.

2.  **Hierarchical Linear Model with Cross-Level Interaction**:
    - **Function**: Regress the within-dataset regression variable (slope $\beta_d$, representing how much FID $Y$ changes per unit quality $X$) onto dataset-level geometric descriptors $Z$.
    - **Mechanism**: Level-1 fits $Y = \alpha_d + \beta_d X + \epsilon$ for each dataset. Level-2 models $\beta_d$ as $\gamma_{00} + \gamma_{11} Z_d + u_d$. An Omnibus test uses the likelihood ratio to test $H_0: \beta_d$ are all equal; a moderation test uses the Wald test for $H_0: \gamma_{11} = 0$ and reports $R^2_{\mathrm{slope}}$ (the variance in slopes explained by $Z$). $X$ is either $N$ or ImageReward, and $Y$ is FID (or KID, $\mathrm{FD_{DINOv2}}$).
    - **Design Motivation**: Plotting $X$–$Y$ scatter plots for each dataset lacks rigor. The hierarchical model allows for independent p-value reporting for cross-dataset slope differences and the explanatory power of geometry, leading to more credible conclusions.

3.  **Precision / Recall Attribution + Two Ablations**:
    - **Function**: Further localize whether FID bias on different reference sets leans toward "fidelity" or "coverage."
    - **Mechanism**: Decompose FID into precision (proportion of generated samples near the real manifold) and recall (proportion of real samples covered by the generated distribution). Use OLS to calculate $R^2(\text{Precision}, \text{FID})$ and $R^2(\text{Recall}, \text{FID})$ for each dataset. Ablations involve replacing Inception-v3 with DINOv2 and replacing Fréchet distance with MMD (KID) to repeat the tests.
    - **Design Motivation**: While FID only indicates improvement or degradation, P/R decomposition explains why—for instance, FID reverses on COCO because it is recall-dominated, and recall drops as quality increases. Ablations rule out the possibility that findings are artifacts of a specific backbone.

### Loss & Training
Ours is an evaluation study rather than a training study, so no loss functions are involved. Generation uses SD 1.5 + DDIM, CFG=7.5, 512×512, with fixed seeds per prompt. The only variable is denoising steps $N$. For each (dataset, $N$), samples equal in size to the reference set are generated to calculate FID, KID, FD$_{\text{DINOv2}}$, precision, recall, and ImageReward.

## Key Experimental Results

### Main Results
Geometric descriptors for the six reference sets:

| Dataset | $\langle -\log d_k\rangle$ | $\mathrm{erank}(A)$ | Type |
| :--- | :--- | :--- | :--- |
| FFHQ | $-2.48$ | $1243$ | Concentrated (single-domain face) |
| CelebA-HQ | $-2.36$ | $1220$ | Concentrated (single-domain face) |
| MJHQ-30K | $-2.74$ | $1341$ | Intermediate |
| ImageNet | $-2.68$ | $1431$ | Dispersed |
| Flickr30K | $-2.80$ | $1341$ | Dispersed |
| COCO | $-2.67$ | $1337$ | Dispersed |

Main conclusion: For CelebA-HQ / FFHQ, FID $\downarrow$ as $N \uparrow$ (aligned with quality). For COCO / Flickr30K / ImageNet, FID $\uparrow$ as $N \uparrow$ (reversed). MJHQ-30K is intermediate. Omnibus tests $D = 44.3, p < .001$ (using $X = N$) and $D = 90.9, p < .001$ (using $X = \text{ImageReward}$) strongly reject the hypothesis that all dataset slopes are equal.

### Ablation Study
Moderation tests and ablation results:

| $X$ | $Y$ | $Z$ | $\gamma_{11}$ | $p$ | $R^2_{\mathrm{slope}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $N$ | FID | $\langle -\log d_k\rangle$ | $-0.0323$ | $<.001$ | $0.707$ |
| $N$ | FID | $\mathrm{erank}$ | $0.0314$ | $.002$ | $0.661$ |
| IR | FID | $\langle -\log d_k\rangle$ | $-0.120$ | $.007$ | $0.548$ |
| IR | FID | $\mathrm{erank}$ | $0.119$ | $.010$ | $0.530$ |
| $N$ | KID | $\langle -\log d_k\rangle$ | $-0.0343$ | $<.001$ | $0.763$ |
| $N$ | KID | $\mathrm{erank}$ | $0.0315$ | $.005$ | $0.596$ |
| $N$ | FD$_{\text{DINOv2}}$ | $\langle -\log d_k\rangle$ | $-0.0108$ | $<.001$ | $0.827$ |
| $N$ | FD$_{\text{DINOv2}}$ | $\mathrm{erank}$ | $0.0110$ | $<.001$ | $0.837$ |

Precision / Recall attribution $R^2$: FFHQ 0.989 / 0.672, CelebA-HQ 0.951 / 0.001, MJHQ 0.734 / 0.025, ImageNet 0.690 / **0.949**, Flickr30K 0.314 / **0.850**, COCO 0.676 / **0.833**. FID follows precision on concentrated datasets but is dominated by recall on dispersed datasets.

### Key Findings
- Density coefficients are consistently negative, while effective rank coefficients are consistently positive: the denser the reference set, the more likely FID aligns with quality; the wider the spread, the more likely the reverse.
- Switching the backbone to DINOv2 increased $R^2_{\mathrm{slope}}$ to $0.83$, confirming that this is not an Inception-v3 artifact. Replacing Fréchet with MMD also preserved significant effects, ruling out estimator-specific issues.
- FID is strongly correlated with recall on dispersed datasets (COCO recall $R^2 = 0.833$). The mechanism for FID abnormality in these scenarios is: "more steps → finer samples but narrowed mode → decreased recall → worsened FID."
- Practical Conclusion: FID can be used confidently with concentrated reference sets (FFHQ, CelebA-HQ). For dispersed sets, geometric descriptors must be reported alongside FID, or alternative metrics should be used.

## Highlights & Insights
- **Formulating Metric Fragility as a Statistical Problem**: While previous counterexamples were anecdotal, this work uses hierarchical linear models to test how much geometric descriptors explain slope variance, moving the field toward empirical science.
- **Two Low-Dimensional Descriptors Provide 70%+ Explanatory Power**: Density and effective rank explain over half of FID's behavioral variance, suggesting that benchmark selection should be guided by these metrics rather than experience or intuition.
- **Improved Reporting Standards**: The authors suggest using concentrated datasets for FID or reporting $\langle -\log d_k\rangle$ and $\mathrm{erank}$ alongside FID—a practice the community can adopt immediately.
- **Cross-Metric Generality**: Stronger effects on KID and FD$_{\text{DINOv2}}$ suggest that this is a common issue for "distributional metrics on a reference set," rather than being limited to FID.

## Limitations & Future Work
- The study only evaluated 6 reference sets and 1 generator (SD 1.5). Future work should replicate conclusions across more text-to-image, class-conditional, and unconditional settings.
- Geometric descriptors were selected post-hoc; there is no learnable solution provided to directly correct geometric bias when calculating FID.
- Quality proxies still rely on ImageReward, which is inherently biased; incorporating human evaluation into the regression would be ideal.
- The work does not yet discuss how to integrate geometric sensitivity into leaderboard ranking methods, which remains a normative future task.

## Related Work & Insights
- **vs. Kynkäänniemi et al. (2022, 2024)**: While they blame the Inception-v3 feature space, this work proves the backbone is not the primary cause (effects are stronger with DINOv2).
- **vs. Chong & Forsyth (2020)**: While they blame finite sample estimation bias of the Fréchet distance, this work shows the effect persists with MMD, indicating the estimator is not the root cause.
- **vs. Jayasumana et al. (2024) CMMD**: CMMD proposes changing the backbone and distance to "fix" FID; this work provides a more upstream diagnosis: reference set geometry must be considered before discussing metric changes.
- **vs. Precision/Recall Series (Kynkäänniemi 2019, Sajjadi 2018)**: While P/R are traditionally seen as supplementary, this work uses them to deconstruct FID's bias under different geometries, repositioning P/R as a diagnostic tool for FID anomalies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](../../CVPR2026/image_generation/garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[ICML 2026\] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](../../CVPR2026/image_generation/learnability-guided_diffusion_for_dataset_distillation.md)

</div>

<!-- RELATED:END -->
