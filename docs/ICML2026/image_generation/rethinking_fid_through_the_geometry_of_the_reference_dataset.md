---
title: >-
  [Paper Note] Rethinking FID Through the Geometry of the Reference Dataset
description: >-
  [ICML 2026][Image Generation][FID] This paper points out that the "lower is better" assumption of FID systematically fails across different reference datasets. Using two geometric descriptors—distribution density $\langle -\log d_k\rangle$ and effective rank $\mathrm{erank}(A)$—the authors prove via a hierarchical linear model that these descriptors exp
tags:
  - ICML 2026
  - Image Generation
  - FID
date: 2026-05-08
content_hash: 085c86457788fc89
---
# Rethinking FID Through the Geometry of the Reference Dataset

**Conference**: ICML 2026  
**arXiv**: [2605.29335](https://arxiv.org/abs/2605.29335)  
**Code**: To be confirmed  
**Area**: Image Generation / Generative Model Evaluation  
**Keywords**: FID, Generative Evaluation, Reference Set Geometry, Distribution Density, Effective Rank

## TL;DR
This paper points out that the "lower is better" assumption of FID systematically fails across different reference datasets. Using two geometric descriptors—distribution density $\langle -\log d_k\rangle$ and effective rank $\mathrm{erank}(A)$—the authors prove via a hierarchical linear model that these descriptors explain ~70% of the cross-dataset variance in the "sample quality → FID" slope, quantitatively attributing FID's fragility to the reference set itself for the first time.

## Background & Motivation

**Background**: FID uses Inception-v3 features + Fréchet distance to measure the difference between generative and reference distributions. It has become the de facto standard for evaluating image generation, used as a primary benchmark across nearly all diffusion, GAN, and autoregressive model papers.

**Limitations of Prior Work**: Numerous counterexamples have emerged in recent years—Choi et al. (2025) showed that spending more compute on COCO to obtain clearer images actually worsened FID; Lee et al. (2025) found that tuning hyperparameters by minimum FID resulted in samples with the worst ImageReward; Jayasumana et al. (2024) demonstrated that stronger image perturbations could paradoxically "improve" FID. These indicate that "FID decrease = quality increase" no longer consistently holds in practice.

**Key Challenge**: Previous explanations either blamed the fragility of Inception-v3 features (Kynkäänniemi et al. 2022, Parmar et al. 2022) or the instability of the Fréchet distance for long-tail estimation (Chong & Forsyth 2020). However, the other core component of FID—the reference dataset itself—has rarely been scrutinized. How are reference sets selected? Why does FID work on CelebA-HQ but fail on COCO? Are there quantifiable "geometric" differences between them?

**Goal**: To formalize "how the reference set shapes FID behavior" and identify a small set of geometric descriptors capable of predicting FID's sensitivity across different reference sets.

**Key Insight**: FID is essentially a distribution distance, naturally concerning two properties of the reference set: how tightly it clusters in feature space (density) and how many principal directions it spans (effective dimension). Single-modality face datasets like CelebA-HQ and multi-category open-domain datasets like COCO naturally occupy different regions of the feature space, which should cause the response direction of FID to differ.

**Core Idea**: Describe reference set geometry using two scalars—mean kNN log-density and effective rank—and then use a two-layer hierarchical linear model (HLM) to explicitly model the "sample quality → FID slope" as a function of reference set geometry, followed by cross-dataset statistical testing.

## Method

### Overall Architecture
A fixed generator (Stable Diffusion 1.5 + DDIM) is used to generate images for six reference sets with vastly different semantic spans (FFHQ, CelebA-HQ, MJHQ-30K, ImageNet, Flickr30K, COCO). Sample quality is controlled by scanning denoising steps $N \in \{15, 20, \dots, 50\}$, with ImageReward serving as a quality proxy. Two main tasks are performed: (1) calculating two geometric descriptors for each dataset; (2) using an HLM to test whether slope differences can be explained by these geometric quantities. Finally, FID is decomposed using precision/recall to attribute dominant drivers, and ablations are performed by replacing Inception-v3 (with DINOv2) and Fréchet distance (with MMD/KID).

```mermaid
graph TD
    A["Fix Generator: SD1.5 + DDIM<br/>6 Reference Sets, Scan Denoising Steps N"] --> B["Generate Equal-sized Samples per (Dataset, N)<br/>ImageReward as Quality Proxy X"]
    B --> C["Two Geometric Descriptors Z<br/>Density ⟨−log d_k⟩ + Effective Rank erank"]
    B --> D["Distance Metric Y<br/>FID / KID / FD_DINOv2"]
    subgraph HLM["Hierarchical Linear Model + Cross-level Interaction"]
        direction TB
        E["Level-1: Fit Quality→FID Slope β_d per Dataset"]
        E --> F["Level-2: β_d = γ00 + γ11·Z + u_d"]
        F --> G["Omnibus Test:<br/>Does Slope Vary Across Sets?"]
        F --> H["Moderation Test:<br/>Variance explained by Geometry Z (R²)"]
    end
    C --> F
    D --> E
    G --> ATTRIN
    H --> ATTRIN
    subgraph ATTR["Precision / Recall Attribution + Ablations"]
        direction TB
        ATTRIN["P/R Decomposition: Locate FID Driver Side"]
        ATTRIN --> J["Replace DINOv2 Backbone / MMD Distance<br/>Effect Persists → Not Solely Due to Estimator"]
    end
```

### Key Designs

**1. Two Geometric Descriptors: Characterizing Reference "Shape" via Density and Effective Rank**

To clarify the reference set's role in shaping FID, the distribution in feature space must be compressed into comparable scalars. This paper selects two complementary metrics: distribution density using a logarithmic version of $k$-NN:

$$\langle -\log d_k\rangle = \frac{1}{n}\sum_i -\log d_k(x_i)$$

where $d_k(x_i)$ is the Euclidean distance from the $i$-th sample to its $k$-th nearest neighbor ($k=80$). The log-average is used because Loftsgaarden-Quesenberry estimates $\hat p(x)\propto d_k(x)^{-D}$ span dozens of orders of magnitude in $D=2048$ dimensions. Effective rank is defined as the exponent of the Shannon entropy of normalized singular values: $\mathrm{erank}(A)=\exp(H(\bm\sigma/\|\bm\sigma\|_1))$, where $A$ is the centered feature matrix. This provides a continuous generalization of "weighted dimensionality." Together, these separate distinct distributions—CelebA-HQ is a compact single-modality set (density $-2.36$, erank $1220$), while COCO is a spread-out open-domain set (density $-2.67$, erank $1337$).

**2. Hierarchical Linear Model + Cross-level Interaction: Testing Geometric Explanations**

To rigorously quantify differences rather than just plotting scatters, a two-layer HLM is used. Level-1 fits $Y=\alpha_d+\beta_d X+\epsilon$ within each dataset $d$, yielding the intra-dataset slope $\beta_d$ (how much FID changes per unit quality). Level-2 regresses these slopes on dataset-level geometric descriptors: $\beta_d=\gamma_{00}+\gamma_{11}Z_d+u_d$. Two independent tests are then conducted: an Omnibus test using a likelihood ratio test to check if $\beta_d$ are equal (answering if slopes vary across sets), and a Moderation test using a Wald test for $H_0:\gamma_{11}=0$ (reporting $R^2_{\mathrm{slope}}$ to show how much variance geometry explains).

**3. Precision / Recall Attribution + Ablations: Locating Bias and Eliminating Backbone Suspicions**

FID is decomposed into Precision (proportion of generated samples near the real manifold) and Recall (proportion of the real distribution covered by generated samples). The authors calculate $R^2(\text{Precision},\text{FID})$ and $R^2(\text{Recall},\text{FID})$ for each dataset to see which side dominates—explaining mechanisms where higher $N$ on COCO leads to finer samples but narrower modes, causing recall to drop and FID to worsen. To rule out "Inception-v3 or Fréchet bias," backbones are replaced with DINOv2 and the Fréchet distance with MMD (KID), re-running the HLM tests.

### Loss & Training
This study focuses on evaluation; no training occurs. Generation uses SD 1.5 + DDIM, CFG=7.5, 512×512, with fixed seeds per prompt. The sole variable is the denoising steps $N$. For each (dataset, $N$), samples equal in size to the reference set are generated to calculate FID, KID, FD$_{\text{DINOv2}}$, precision, recall, and ImageReward.

## Key Experimental Results

### Main Results
Geometric descriptors for the six reference sets:

| Dataset | $\langle -\log d_k\rangle$ | $\mathrm{erank}(A)$ | Type |
|--------|------|------|------|
| FFHQ | $-2.48$ | $1243$ | Centralized (Single-domain face) |
| CelebA-HQ | $-2.36$ | $1220$ | Centralized (Single-domain face) |
| MJHQ-30K | $-2.74$ | $1341$ | Intermediate |
| ImageNet | $-2.68$ | $1431$ | Dispersed |
| Flickr30K | $-2.80$ | $1341$ | Dispersed |
| COCO | $-2.67$ | $1337$ | Dispersed |

Main Findings: For CelebA-HQ / FFHQ, FID decreases as $N$ increases (consistent with quality). For COCO / Flickr30K / ImageNet, FID increases as $N$ increases (contradicts quality). Omnibus test $D = 44.3, p < .001$ (with $X = N$) and $D = 90.9, p < .001$ (with $X = \text{ImageReward}$) strongly reject the hypothesis of identical slopes across datasets.

### Ablation Study
Moderation test results:

| $X$ | $Y$ | $Z$ | $\gamma_{11}$ | $p$ | $R^2_{\mathrm{slope}}$ |
|------|------|------|------|------|------|
| $N$ | FID | $\langle -\log d_k\rangle$ | $-0.0323$ | $<.001$ | $0.707$ |
| $N$ | FID | $\mathrm{erank}$ | $0.0314$ | $.002$ | $0.661$ |
| IR | FID | $\langle -\log d_k\rangle$ | $-0.120$ | $.007$ | $0.548$ |
| IR | FID | $\mathrm{erank}$ | $0.119$ | $.010$ | $0.530$ |
| $N$ | KID | $\langle -\log d_k\rangle$ | $-0.0343$ | $<.001$ | $0.763$ |
| $N$ | FD$_{\text{DINOv2}}$ | $\langle -\log d_k\rangle$ | $-0.0108$ | $<.001$ | $0.827$ |

Precision / Recall Attribution ($R^2$): FFHQ 0.989 / 0.672; CelebA-HQ 0.951 / 0.001; ImageNet 0.690 / **0.949**; COCO 0.676 / **0.833**. FID is driven by precision in centralized datasets but by recall in dispersed ones.

### Key Findings
- Density coefficients are consistently negative, while effective rank coefficients are positive: denser reference sets favor quality-FID alignment, whereas broader sets induce contradiction.
- Replacing the backbone with DINOv2 increases $R^2_{\mathrm{slope}}$ to $0.83$, confirming that Inception-v3 is not the primary cause.
- In dispersed datasets, FID correlates strongly with recall (COCO Recall $R^2 = 0.833$), signifying that "increased steps → finer samples but mode collapse → reduced recall → worse FID" is the mechanism of FID anomalies.
- Practical Conclusion: Use FID with confidence on centralized sets (FFHQ, CelebA-HQ); on dispersed sets, geometric descriptors must be reported or alternative metrics used.

## Highlights & Insights
- **Metric Fragility as a Statistical Problem**: Moves beyond anecdotal counterexamples to a hierarchical linear model approach, advancing the field from "disputing counterexamples" to "empirical science."
- **Low-dimensional Explanatory Power**: Just two scalars (density + effective rank) explain over half of the FID behavioral variance, allowing benchmark selection to be guided by quantitative metrics rather than intuition.
- **Reporting Standards**: Proposes a practical norm—either use centralized datasets for FID or report $\langle -\log d_k\rangle$ and $\mathrm{erank}$ alongside FID scores.
- **Cross-metric Universality**: The effects are even stronger for KID and FD$_{\text{DINOv2}}$, indicating this is a fundamental issue for all "distributional metrics on a reference set."

## Limitations & Future Work
- The study only used 6 reference sets and 1 generator (SD 1.5); results need replication across more T2I, class-conditional, and unconditional settings.
- Geometric descriptors are selected post-hoc; a learnable solution to directly correct for geometric bias in FID calculation is missing.
- Quality proxies still rely on ImageReward, which itself may be biased; human evaluation should ideally be integrated into the regression.

## Related Work & Insights
- **vs Kynkäänniemi et al. (2022, 2024)**: While they blame the Inception-v3 feature space, this paper proves the backbone is not the primary driver (effect is stronger on DINOv2).
- **vs Chong & Forsyth (2020)**: They blame Fréchet estimator bias on finite samples, but the effect persists with MMD, suggesting the estimator is not the main cause.
- **vs Jayasumana et al. (2024) CMMD**: CMMD proposes changing backbones and distances; this paper provides an "upstream" diagnosis—one must analyze reference set geometry before debating metrics.
- **vs Precision/Recall Series**: Historically viewed as auxiliary, P/R is repositioned here as a diagnostic tool for understanding FID anomalies under different geometries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](../../CVPR2026/image_generation/garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[ICML 2026\] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)
- [\[CVPR 2026\] Refaçade: Editing Object with Given Reference Texture](../../CVPR2026/image_generation/refacade_editing_object_with_given_reference_texture.md)

</div>

<!-- RELATED:END -->
