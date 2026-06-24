---
title: >-
  [Paper Note] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy
description: >-
  [NEURIPS2025][Image Restoration][fluorescence microscopy] This paper proposes scSplit, which introduces a severity-cognizant input normalization module (SCIN) and a regression network (Reg) to endow an InDI-based iterative image decomposition framework with awareness of the mixing severity of two overlapping structures in fluorescence microscopy images. The method unifies image splitting and bleedthrough removal across five public datasets under a single framework.
tags:
  - "NEURIPS2025"
  - "Image Restoration"
  - "fluorescence microscopy"
  - "image decomposition"
  - "severity cognizance"
  - "mixing ratio"
  - "bleedthrough removal"
date: 2026-05-08
content_hash: 39572dd06d688bee
---

# scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy

**Conference**: NEURIPS2025  
**arXiv**: [2503.22983](https://arxiv.org/abs/2503.22983)  
**Code**: [juglab/scSplit](https://github.com/juglab/scSplit/)  
**Area**: Image Restoration  
**Keywords**: fluorescence microscopy, image decomposition, severity cognizance, mixing ratio, bleedthrough removal  

## TL;DR
This paper proposes scSplit, which introduces a severity-cognizant input normalization module (SCIN) and a regression network (Reg) to endow an InDI-based iterative image decomposition framework with awareness of the mixing severity of two overlapping structures in fluorescence microscopy images. The method unifies image splitting and bleedthrough removal across five public datasets under a single framework.

## Background & Motivation
- **Background**: Fluorescence microscopy is a critical imaging modality in life sciences, enabling separate imaging of distinct cellular structures across channels via different fluorescent markers, yet the number of structures that can be simultaneously imaged remains limited.
- **Limitations of Prior Work**: Computational multiplexing methods (e.g., μSplit, denoiSplit, MicroSplit) allow multiple structures to be mixed into a single channel and subsequently separated via deep learning, but they assume the input is a simple average of two structures (t=0.5), neglecting variation in the mixing ratio.
- **Key Challenge**: In practice, sample properties, labeling density, and microscope configurations vary substantially, causing the mixing ratio to fluctuate widely. When the mixing characteristics of test images differ from those seen during training, existing methods degrade significantly.
- **Bleedthrough Problem**: Bleedthrough is a related phenomenon in which structures leak into a dedicated imaging channel due to imprecise optical filtering, which is intrinsically a decomposition problem with varying mixing ratios.
- **Difficulty of Unification**: Only by effectively addressing mixing ratio variation can a single network handle both image splitting (t≈0.5) and bleedthrough removal (t close to 0 or 1).
- **Rationale for InDI**: InDI models degradation as a linear mixture of clean and degraded images, whose inductive bias naturally aligns with the linear superposition model in fluorescence microscopy, making it an ideal backbone framework.

## Method

### Overall Architecture
Given two structures $c_0 \in C_0$ and $c_1 \in C_1$, the mixed input is defined as $c_t = (1-t)c_0 + tc_1$, where $t \in [0,1]$ is the mixing ratio. scSplit comprises three key components: (1) the severity-cognizant input normalization module SCIN, (2) two generation networks $\text{Gen}_0$ and $\text{Gen}_1$ predicting each channel independently, and (3) a regression network Reg that estimates the mixing ratio $t$. During training, $t$ is sampled randomly; at inference, Reg estimates $t$, which is further refined through aggregation.

### Key Designs

#### Module 1: Severity-Cognizant Input Normalization (SCIN)
- **Function**: Applies $t$-specific mean-variance normalization to mixed inputs $c_t$ at varying mixing ratios, ensuring $\mathbb{E}[\mu(t)]=0$ and $\mathbb{E}[\sigma^2(t)]=1$ for all $t$.
- **Mechanism**: The interval $[0,1]$ is partitioned into $n=100$ equally spaced sub-intervals. For each sub-interval, the mean and standard deviation of mixed patches are precomputed and stored in a lookup table $D[i]=(\mu_i, \sigma_i)$. During training, normalization parameters are retrieved from the table according to the sub-interval containing $t$; at inference, because $\mathbb{E}[\mu(t)]=0$ and $\mathbb{E}[\sigma^2(t)]=1$ are guaranteed for all $t$, the test image set statistics can be used directly for normalization.
- **Design Motivation**: Theoretical derivation shows that, after standardizing $C_0$ and $C_1$ individually, the expected variance of the mixed image $c_t$ satisfies $\mathbb{E}[\sigma^2(t)] = t^2 + (1-t)^2 + 2t(1-t)\text{Cov}(p_0, p_1)$, which is a function of $t$. Without $t$-specific normalization, inputs at different $t$ values present different statistical properties to the network, and correct normalization at inference is infeasible when $t$ is unknown, resulting in distribution mismatch.

#### Module 2: Regression Network Reg
- **Function**: Given the normalized mixed input $x_t$, predicts an estimate of the mixing ratio $t$.
- **Mechanism**: Reg shares the SCIN normalization module with the generation networks, ensuring consistent input statistics. At inference, domain knowledge is exploited: all images within the same microscopy session share the same laser power and fluorophore type, and therefore the same mixing ratio. Estimates of $t$ across all images in a session are averaged (aggregated) to improve prediction accuracy.
- **Design Motivation**: Accurate $t$ estimation requires correct normalization, while correct normalization depends on $t$, forming a circular dependency. SCIN breaks this cycle. The aggregation operation leverages prior knowledge specific to fluorescence microscopy; single-image $t$ estimates may be noisy, but averaging over a batch substantially reduces variance.

#### Module 3: Generation Networks Gen₀ and Gen₁
- **Function**: Predict the normalized versions of channels $c_0$ and $c_1$ respectively. The input is the normalized mixed image $x_t = c_t^{\text{Norm}} + t\varepsilon n$ ($n \sim \mathcal{N}(0, I)$, $\varepsilon=0.01$) conditioned on the mixing ratio.
- **Mechanism**: Following InDI's training procedure, the mixing ratio $t$ is incorporated as an additional conditioning input. The prediction for channel $i$ is $\hat{c}_i^{\text{Norm}} = \text{Gen}_i(x_t,\, t\delta_i + (1-t)\delta_{1-i})$, where the severity for separating $c_0$ is $t$ and for separating $c_1$ is $1-t$.
- **Design Motivation**: The InDI framework naturally supports linear mixture degradation modeling. By conditioning on $t$, the network can adapt its decomposition strategy to different mixing severities.

### Loss & Training
- During training, the mixing ratio $t$ is sampled from $p(t) = \frac{1}{1+a}\mathcal{U}[0,1] + \frac{a}{1+a}\delta_{0.5}$ ($a=1$), assigning greater weight to $t=0.5$ relative to InDI, since image splitting inputs typically contain both structures.
- Single-step inference is used at test time to minimize distortion.
- A Gaussian noise perturbation with $\varepsilon=0.01$ serves as regularization.

## Key Experimental Results

### Table 1: Quantitative Evaluation on 5 Datasets (PSNR, Grouped by Mixing Severity)

| Method | Hagen-Dom | Hagen-Bal | Hagen-Weak | HTLIF24-Dom | HTLIF24-Bal | HTLIF24-Weak |
|--------|-----------|-----------|------------|-------------|-------------|--------------|
| U-Net | 31.8 | 28.2 | 22.0 | 45.9 | 44.6 | 36.0 |
| μSplit_D | 33.1 | 32.4 | 23.4 | 45.9 | 44.9 | 36.4 |
| InDI | 33.1 | 32.1 | 24.2 | 45.2 | 43.9 | 37.6 |
| scSplit₀.₅ | 34.1 | 33.7 | 25.0 | 45.9 | 45.1 | 37.4 |
| **scSplit** | **40.9** | **33.9** | **29.3** | **51.8** | **45.5** | **39.9** |

### Table 2: Supplementary Results on BioSR and HTT24 Datasets (PSNR)

| Method | BioSR-Dom | BioSR-Bal | BioSR-Weak | HTT24-Dom | HTT24-Bal | HTT24-Weak |
|--------|-----------|-----------|------------|-----------|-----------|------------|
| MicroSplit | 38.5 | 34.3 | 26.6 | 36.9 | 36.6 | 30.3 |
| InDI | 35.9 | 33.4 | 26.3 | 37.6 | 36.5 | 30.5 |
| scSplit₀.₅ | 37.3 | 35.0 | 26.4 | 38.1 | 38.6 | 31.4 |
| **scSplit** | **40.1** | **35.3** | **28.7** | **44.5** | **39.1** | **34.7** |

### Key Findings
- **Large Gains in the Dominant Regime**: scSplit achieves 2–8 dB PSNR improvement over the best baseline in the Dominant regime ($w=0.7$–$0.9$, i.e., bleedthrough removal scenarios), demonstrating that severity cognizance is critical for bleedthrough removal.
- **Substantial Contribution of SCIN**: Comparing scSplit₀.₅ vs. InDI (differing only in normalization) yields an average PSNR improvement of 1.2 dB.
- **Effectiveness of Aggregation**: scSplit consistently outperforms scSplit$_{-\text{agg}}$ on all datasets across all PSNR metrics.
- **Downstream Task Validation**: In BioSR segmentation, scSplit achieves the lowest DICE dissimilarity (e.g., at $w=0.9$, Ch1: 0.035 vs. InDI 0.044).
- **Robustness to Distribution Shift**: Under large train–test mixing ratio discrepancy (Table 3), scSplit achieves 35.8 dB, substantially outperforming U-Net at 30.1 dB.
- **Augmented μSplit Still Falls Short**: Adding mixing ratio augmentation to μSplit_D improves its performance, yet scSplit still exceeds it by an average of 2.4 dB PSNR.

## Highlights & Insights
1. **Precise Problem Formulation**: The authors clearly identify the overlooked yet critical issue of varying mixing severity and demonstrate that it is the central obstacle to unifying image splitting and bleedthrough removal.
2. **Theoretically Grounded Normalization**: The dependence of the expected variance of $c_t$ on $t$ is derived mathematically, providing rigorous theoretical motivation for SCIN rather than a purely empirical design choice.
3. **Elegant Use of Domain Knowledge**: The aggregation module cleverly exploits the microscopy-specific prior that all images within the same acquisition session share the same mixing ratio, thereby improving $t$ estimation accuracy.
4. **Comprehensive Ablation Study**: Through variants such as scSplit₀.₅ and scSplit$_{-\text{agg}}$, the contribution of each component is clearly quantified.

## Limitations & Future Work
- The method handles only two-structure decomposition and is not extended to the case $k>2$.
- The linear superposition assumption may not hold strictly under certain imaging conditions.
- Single-step inference, while reducing distortion, may limit the model's capacity to handle complex mixing patterns.
- Improvements in the Weak regime on the PaviaATN dataset are relatively modest (24.3 vs. 21.8 for InDI), suggesting that the method's effectiveness may be constrained under certain data characteristics.
- The regression network's $t$ estimates remain imperfect; Figure 3 shows notable performance degradation when the assumed $w$ deviates substantially from the true value.

## Related Work & Insights
- **InDI** (Delbracio & Milanfar, 2023): An iterative restoration method that models degradation as a linear mixture; scSplit inherits its inductive bias and substantially extends its scope.
- **μSplit** (Ashesh et al.): A GPU-efficient meta-architecture for image splitting that lacks awareness of mixing severity.
- **MicroSplit**: Quantifies the impact of mixing variability but proposes no solution; scSplit fills this gap.
- **denoiSplit**: Combines unsupervised denoising with supervised image splitting but similarly ignores variation in the mixing ratio.
- **Insight**: For any image restoration problem involving unknown degradation severity (e.g., noise level, blur degree), introducing degradation-aware normalization and degradation level prediction may be an effective strategy.
- **Broader Perspective**: Image splitting can be viewed as a special case of image translation, related to reflection removal, dehazing, and deraining, but fundamentally distinct in its linear superposition model and ground truth availability.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of SCIN normalization and severity cognizance constitutes an innovative contribution tailored to fluorescence microscopy)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 datasets, 3 scenario categories, extensive ablation studies, downstream segmentation validation)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear and problem motivation is thoroughly articulated)
- Value: ⭐⭐⭐⭐ (Practically valuable for the fluorescence microscopy community, unifying two important tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Recover Cell Tensor: Diffusion-Equivalent Tensor Completion for Fluorescence Microscopy Imaging](../../ICLR2026/image_restoration/recover_cell_tensor_diffusion-equivalent_tensor_completion_for_fluorescence_micr.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](../../ECCV2024/image_restoration/denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[CVPR 2025\] A Flag Decomposition for Hierarchical Datasets](../../CVPR2025/image_restoration/a_flag_decomposition_for_hierarchical_datasets.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](../../ICCV2025/image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
