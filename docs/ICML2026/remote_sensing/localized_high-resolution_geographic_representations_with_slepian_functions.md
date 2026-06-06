---
title: >-
  [Paper Note] Localized, High-resolution Geographic Representations with Slepian Functions
description: >-
  [ICML 2026][Remote Sensing][Slepian functions] This paper constructs a geographic positional encoder using spherical Slepian functions to concentrate representation capacity on a Region of Interest (ROI). It proposes a S…
tags:
  - "ICML 2026"
  - "Remote Sensing"
  - "Slepian functions"
  - "Spherical harmonics"
  - "Positional encoding"
  - "Local high-resolution"
  - "Geographic machine learning"
date: 2026-05-08
content_hash: b3511c02f6d342df
---

# Localized, High-resolution Geographic Representations with Slepian Functions

**Conference**: ICML 2026  
**arXiv**: [2602.00392](https://arxiv.org/abs/2602.00392)  
**Code**: https://github.com/arjunarao619/SlepianPosEnc (Available)  
**Area**: Remote Sensing / Geographic Representation / Positional Encoding  
**Keywords**: Slepian functions, Spherical harmonics, Positional encoding, Local high-resolution, Geographic machine learning

## TL;DR
This paper constructs a geographic positional encoder using spherical Slepian functions to concentrate representation capacity on a Region of Interest (ROI). It proposes a Slepian-spherical harmonic hybrid encoding to simultaneously account for local high-resolution and global coarse-grained context, consistently outperforming mainstream baselines such as SH, Wavelet, and RFF across five classification, regression, and image enhancement prediction tasks.

## Background & Motivation
**Background**: Embedding latitude and longitude $(\lambda, \phi) \in S^2$ into a continuous function $\Phi(x)$, followed by an MLP for downstream prediction, is the standard practice in geographic machine learning. Typical choices include grid cell-style multi-scale sinusoidal encodings (Space2Vec), the Double Fourier Sphere series (SphereC/M), Random Fourier Features (RFF), and Spherical Harmonics (SH) defined natively on the sphere. SatCLIP uses SH for large-scale pre-training and is currently the recognized global universal positional encoder.

**Limitations of Prior Work**: All these encodings distribute the "resolution budget" uniformly across the entire Earth. To see details at a city level ($\sim$ several kilometers), the global resolution must be increased simultaneously, causing the feature dimension to expand quadratically at $(L+1)^2$, which leads to prohibitive memory and computational costs. More critically, the recursive associated Legendre polynomials of SH are numerically unstable, with normalization constants $N_{\ell m} \sim \ell^{-m}$ decaying extremely fast. In FP32, numerical errors occur when $L \gtrsim 40$, and the threshold for mixed-precision training is even lower. Thus, the resolution that SH can reliably use is stuck at roughly $20000/40 = 500$ km, which is far from sufficient for local tasks like California housing prices or Japanese prefectural analysis at a scale of tens of kilometers.

**Key Challenge**: A structural trade-off exists between "global completeness" and "local high-resolution" on the sphere. Global applicability requires a complete orthogonal basis covering the whole sphere, where each basis function spans the entire domain; local high-resolution requires exponential dimensions to achieve refinement.

**Goal**: (1) Find a basis function that concentrates the vast majority of energy within a user-specified ROI under a fixed bandwidth; (2) Ensure this basis can stitch seamlessly with global SH to retain global context; (3) Be pole-safe (no collapse at the poles); (4) Be computationally scalable to resolutions like $L_r \sim 256$ that are unattainable for SH.

**Key Insight**: There is a classic concentration problem in signal processing (Slepian & Pollak, 1961): among all band-limited functions, which ones concentrate their energy maximally within a given interval? Extending this to the sphere (Simons et al. 2006) yields spherical Slepian functions, which have been used for "localized signal analysis" in geophysical tasks like Earth's gravity field and ice sheet mass variations. The authors' shift is: rather than using Slepian functions to analyze observed signals, use them as a positional encoding basis to directly learn local representations.

**Core Idea**: Project the band-limited SH subspace $\mathcal{H}_{L_r}$ onto the ROI to obtain a concentration matrix $K$, and take its first $K = \lceil N(R,L_r) \rceil$ eigenvectors as the basis for positional encoding. By concatenating this with a low-bandwidth global SH basis, one can achieve "high-resolution Slepian locally and low-resolution SH globally."

## Method
The core of the paper is reformulating the spherical concentration problem as an eigenvalue problem, making it computable at high bandwidths using spherical caps, and merging local Slepian with global SH into a hybrid encoding for an MLP.

### Overall Architecture
The input consists of spherical coordinates $x = (\lambda, \phi) \in S^2$; the encoder $\Phi(x)$ provides a $D$-dimensional feature, which is then passed through an arbitrary NN (MLP / GLU / bottleneck fused with image embeddings) to output the label $y$. This paper does not change the NN but only replaces $\Phi$. The pipeline consists of four steps: (A) Select one or more ROI spherical caps $R_c$ and a high bandwidth $L_r$, pre-compute the spherical cap Slepian eigenfunctions $\{g_n\}$, sort them by concentration eigenvalue $\mu_n$, and truncate them according to the Shannon number; (B) Select a low bandwidth $L_g \ll L_r$ to calculate the global SH basis $\Phi_{\text{SH}}$; (C) During online inference, concatenate the Slepian evaluations for each ROI with the global SH evaluations as $\Phi_{\text{Hybrid}}(x)$; (D) Pass the result into a downstream NN for classification, regression, or fusion with image features.

### Key Designs

1.  **Slepian Concentration Problem and Basis Construction**:
    - Function: Select orthogonal basis functions in a band-limited subspace that have "nearly all energy falling within the ROI," compressing representation capacity into the region of interest.
    - Mechanism: Define the concentration ratio $\mu = \int_R |h(x)|^2 ds / \int_{S^2} |h(x)|^2 ds \in [0,1)$. Maximizing this is equivalent to solving the eigenvalue problem $K h = \mu h$ for a $D_{L_r} \times D_{L_r}$ symmetric concentration matrix $K$, where $K_{\ell m, \ell' m'} = \int_R Y_\ell^m Y_{\ell'}^{m'} ds$. Once sorted in descending order, the eigenvalues show a sharp transition from $\mu_n \approx 1$ to $\mu_n \approx 0$ at the Shannon number $N(R,L_r) = \mathrm{tr}(K) \approx \frac{\text{area}(R)}{4\pi}(L_r+1)^2$. The first $K = \lceil N(R,L_r)\rceil$ eigenfunctions $\{g_n\}$ yield the positional encoding $\Phi_{\text{Slep}}(x) = [g_1(x), \dots, g_K(x)]^\top$.
    - Design Motivation: The Shannon number precisely translates "region + bandwidth" into the "number of independent modes the region can accommodate," serving as a natural intrinsic dimension upper bound. For a small area like Sri Lanka ($f_R \approx 1.29 \times 10^{-4}$), only $K \approx 9$ Slepian modes are needed at $L_r = 256$, whereas the corresponding global SH would require $D_{L_r} \approx 6.6\times 10^4$ dimensions—sparsity is intrinsic, not manually pruned.

2.  **Spherical Cap Slepian and High-Bandwidth Computability**:
    - Function: Reduce the dimension of the $D_{L_r} \times D_{L_r}$ eigenvalue problem into small block-diagonal problems based on order $m$, allowing $L_r$ to reach $256$ via offline computation.
    - Mechanism: Restrict the ROI to a spherical cap with center point and angular radius $\Theta$. In this axisymmetric setting, the concentration matrix becomes block-diagonal by order $m$, with each block size not exceeding $L_r$. There is an explicit formula for the number of well-concentrated modes $N_\Theta(L_r) = \frac{1-\cos\Theta}{2}(L_r+1)^2$, so $\Theta$ directly controls the "local information budget." Implementation-wise: compute the spherical cap Slepian once at a standard pole position, and then translate it to any target center via spherical rotation, retaining the first $N_\Theta(L_r)$ modes.
    - Design Motivation: Directly solving the dense $K$ matrix (~$6.6\times 10^4$ dimensions) for $L_r = 256$ on an arbitrary $R$ is infeasible in terms of memory and numerical precision. Block-diagonalization splits the $O(D_{L_r}^3)$ cost into $L_r$ small $O(L_r^3)$ problems, making "high-resolution local encoding" a practical tool rather than just a theory.

3.  **Hybrid Slepian-SH Encoding and Pole-Safety**:
    - Function: Provide global coarse-grained context outside the ROI while inheriting the analytical properties of SH at the poles, allowing a single encoder to handle both "California housing prices" and "global species distribution."
    - Mechanism: Define $\Phi_{\text{Hybrid}}(x) = \mathrm{Concat}(\Phi_{\text{Slep}}(x), \Phi_{\text{SH}}(x))$, where $\Phi_{\text{SH}}$ is calculated using low bandwidth $L_g \ll L_r$ to carry coarse "where globally" information. For multiple ROIs, $\{\Phi_{\text{Slep}}^{(c)}\}$ are concatenated side-by-side; since Slepian energies do not overlap, cross-region interference is negligible. Pole-safety comes from the fact that each $g_n = \sum_{\ell,m} h_{\ell m}^{(n)} Y_\ell^m$ is a finite linear combination of $Y_\ell^m$, and since spherical harmonics are analytical at the poles, $g_n$ is also analytical. When $R = S^2$, the concentration matrix becomes the identity matrix, and Slepian reverts to SH.
    - Design Motivation: Pure Slepian is nearly zero outside the ROI; tasks like cross-domain species distribution or global pre-training require global signals. Conversely, pure SH is limited by numerical issues and resolution. The hybrid scheme assigns specific scales to each component, splitting the "local vs. global" trade-off into two parallel components. The authors also extended this to the temporal dimension: using DPSS discrete Slepian sequences for time encoding, with the spatio-temporal Shannon number $k_t \approx 2 N_t W$ controlling the temporal frequency budget, resulting in $\Phi_{\text{ST}}(x,t) = \mathrm{Concat}(\Phi_{\text{SH}}(x), \Phi_{\text{Time}}(t))$.

### Training Strategy
The positional encoding itself is entirely non-parametric (Slepian bases are pre-computed offline); training occurs only within the downstream NN. Classification/regression uses a 3-layer MLP + ReLU + dropout 0.1; building density regression uses a 2-layer bottleneck to concatenate positional encoding with frozen AlphaEarth/Galileo image features; species distribution follows the SINR framework for presence-only training with global pseudo-negative sampling, using positional predictions as spatial priors to weight the Xception output element-wise during inference. All tasks are compared using a unified positional encoding replacement protocol.

## Key Experimental Results

### Main Results
Covering five tasks and over a dozen baselines, the most critical comparisons are summarized below.

| Dataset | Metric | Ours (Hybrid Slepian) | Strong Baseline SphereC / Theory | Gain |
| :--- | :--- | :--- | :--- | :--- |
| California Housing (Regression) | $R^2 \uparrow$ | Significantly Optimal (see Table 1) | SphereC 0.53 / SH(L=40) Weak | Sig. ahead of dense RFF |
| Japan Prefectures (47 classes) | Acc $\uparrow$ | Best | Space2Vec 0.84 | Sig. wins on 2 km boundary hard samples |
| Arctic MSS (Sea Surface Height) | $R^2 \uparrow$ | Best | SphereM 0.91 | Simultaneously verifies pole-safe |
| OpenBuildings Density Regression (4 regions) | $R^2$ under $\sigma$ 0–40 km | Leads in all | SH degrades severely at small $\sigma$ | High-freq details provided by Slepian |
| Species (eBird S&T / IUCN) | mAP $\uparrow$ | Best / Superior out-of-cap | Pure Slepian collapses outside cap | Demonstrates criticality of hybrid architecture |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full Hybrid Slepian ($L_r=120, L_g=10$) | best | Complete model |
| Slepian only (No global SH) | Significant drop | Zero outside cap in global tasks/pseudo-negative scenarios; IUCN species distribution degrades |
| SH only, high $L$ | Numerical collapse | $L \gtrsim 40$ in FP32 results in NaNs, proving the SH path is unviable |
| High-dim Planar RFF | Cal 0.42 / Japan 0.59 | Merely increasing dimensions cannot replace "spatio-spectral concentration" prior |
| Different NN backbones (MLP / GLU / SIREN) | Stable ranking | Gains stem from the encoder itself rather than the downstream network |

### Key Findings
- The true source of improvement is spatio-spectral concentration rather than high dimensionality: Planar RFF with 2000 dimensions still loses significantly to Slepian with 9-200 dimensions, showing that "putting capacity in the right place" is more critical than "adding more capacity."
- The physical meaning of $N(R, L_r)$ is empirically validated: taking Slepian modes beyond the Shannon number introduces noise, whereas truncation at $K$ precisely aligns with the region's intrinsic dimensionality.
- On polar tasks, hybrid encoding performs on par with pure SH, proving that when $R = S^2$, it reverts to SH without introducing new polar pathologies.
- The temporal dimension extension with DPSS also outperforms Fourier time encoding in ACE climate simulations, suggesting the concentration framework is reusable across modalities.

## Highlights & Insights
- Translating "local concentration bases used for decades in geophysics" into "positional encoding" is a rare example in geographic ML of a hardcore idea introduced from signal processing. The cost is simply inverting the analytical perspective: instead of using Slepian to decompose signals, use Slepian to represent coordinates.
- The chain "Region → Shannon Number → Dimension" provides a very clean semantic for capacity allocation: dimension is no longer a hyperparameter for trial and error, but an intrinsic quantity determined by area multiplied by bandwidth. This "prior-informed dimension" design is worth emulating for other spatial data (point clouds, maps, meshes).
- The engineering trick of spherical caps + rotation reduces $O(D^3)$ to $L_r$ instances of $O(L_r^3)$, serving as a bridge from "theoretically elegant" to "engineered for production." The multi-cap concatenation for species distribution also benefits from this.

## Limitations & Future Work
- ROIs currently require manual specification of the spherical cap center and radius. For tasks that are globally uniform but locally sparse (e.g., randomly distributed rare events), there is no obvious ROI partition. Adaptive or learnable ROI selection is a natural next step.
- The spherical cap assumption simplifies geometry but sacrifices fit for irregular shapes (e.g., elongated coastlines, complex administrative boundaries). Generalizing the ROI would break the block-diagonalization property.
- Hybrid dimensionality requires separate tuning of $L_r$ and $L_g$; no principled guidance for the optimal combination across tasks was provided, which could potentially be searched automatically using Shannon numbers and task priors.
- Temporal DPSS was only verified on climate data with 6-hour resolution over one year; performance across longer timescales (interannual/decadal) or with irregular sampling remains to be tested.

## Related Work & Insights
- **vs. SatCLIP / Pure SH (Klemmer 2025a)**: Both are natively spherical and pole-safe, but this paper allows $L_r$ to reach 256 without numerical collapse, significantly outperforming SatCLIP on local tasks like California housing.
- **vs. Spherical Wavelets (Cai & Balestriero 2025)**: Wavelets based on stereographic projection degrade at the poles; Slepian is a finite linear combination of SH, inheriting analyticity and providing a more "spherical-native" multi-resolution solution.
- **vs. Random Fourier Features**: RFF stacks capacity through raw dimensionality (2000 dims in the paper) but cannot "point" capacity toward specific regions; Slepian matches the performance of tens of thousands of SH dimensions in a small region like Sri Lanka using only 9 dimensions.
- **vs. Sphere2Vec / Space2Vec (Mai 2020/2023)**: The DFS series has discontinuities at the poles and uniform global resolution; Slepian addresses both polar and local refinement issues, serving as a natural next step for this lineage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Repositioning spherical Slepian from a "signal analysis tool" to a "positional encoding basis" is a unique and robust crossover from signal processing to geographic ML.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five tasks cover regression/classification/polar/image fusion/spatio-temporal analysis, though complex ROIs beyond spherical caps and cross-continental transfer remain unexplored.
- Writing Quality: ⭐⭐⭐⭐ The explanation of Shannon numbers and concentration problems is clear, and the mapping between formulas and figures 1/2 is intuitive. The appendix is comprehensive, though engineering details in the main text are somewhat brief.
- Value: ⭐⭐⭐⭐⭐ Provides the community with an open-source, plug-and-play, and mathematically interpretable local high-resolution encoder, which is an immediately applicable tool for city-level remote sensing and ecological distribution tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](../../ICLR2026/remote_sensing/measuring_the_intrinsic_dimension_of_earth_representations.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[ICML 2026\] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)
- [\[ICML 2026\] Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](any2any_unified_arbitrary_modality_translation_for_remote_sensing.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](../../CVPR2026/remote_sensing/geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)

</div>

<!-- RELATED:END -->
