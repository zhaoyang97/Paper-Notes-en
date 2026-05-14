---
title: >-
  [Paper Note] VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair
description: >-
  [CVPR 2026][Self-Supervised Learning][Intrinsic image decomposition] VT-Intrinsic exploits the physical complementarity between visible and thermal infrared images—unreflected light is absorbed as heat—to derive ordinal…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Intrinsic image decomposition"
  - "visible-thermal imaging"
  - "reflectance estimation"
  - "illumination decomposition"
  - "ordinal constraints"
date: 2026-05-08
content_hash: 0951435155a4b35c
---

# VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair

**Conference**: CVPR 2026
**arXiv**: [2509.10388](https://arxiv.org/abs/2509.10388)
**Code**: [https://vt-intrinsic.github.io](https://vt-intrinsic.github.io)
**Area**: Self-Supervised
**Keywords**: Intrinsic image decomposition, visible-thermal imaging, reflectance estimation, illumination decomposition, ordinal constraints

## TL;DR
VT-Intrinsic exploits the physical complementarity between visible and thermal infrared images—unreflected light is absorbed as heat—to derive ordinal relationships between visible-thermal intensities that directly correspond to ordinal relationships in reflectance and shading. These ordinal relations serve as dense self-supervised signals to drive neural network optimization, achieving high-quality intrinsic image decomposition without any pre-training data.

## Background & Motivation

1. **Background**: Intrinsic image decomposition (IID) aims to factorize an image into reflectance (albedo) and illumination (shading) components. This is a classical problem in computer vision and graphics. Mainstream approaches include optimization-based methods (e.g., Retinex, relying on strong prior assumptions) and learning-based methods (trained on synthetic data, suffering from a sim-to-real gap).

2. **Limitations of Prior Work**:
    - Obtaining ground-truth reflectance and illumination for real scenes is extremely difficult, requiring specialized equipment and controlled procedures.
    - Learning-based methods are constrained by synthetic training data, often producing over-smoothed results or hallucinated textures in real scenes—particularly severe for diffusion-based approaches.
    - Optimization methods depend on strong prior assumptions (smooth illumination, chromaticity invariance, etc.) and generalize poorly to complex real-world scenes.
    - Methods using NIR auxiliary images are limited by the fact that NIR reflectance still exhibits significant material variation, and LED illumination lacks NIR components.

3. **Key Challenge**: IID is an inherently under-constrained inverse problem—a single visible image cannot uniquely determine the albedo-shading decomposition. Existing methods either rely on insufficiently reliable priors or require large amounts of annotated data.

4. **Goal**: Leverage a single additional thermal infrared image to provide physically meaningful constraints, enabling high-quality IID without pre-training data or controlled illumination.

5. **Key Insight**: A key physical insight—for opaque objects, the portion of incident light not reflected is absorbed as heat. Consequently, low-reflectance regions appear dark in the visible image but bright in the thermal image (absorbing more heat), whereas illumination variation manifests in the same direction in both modalities. This "ordinal relationship" can directly distinguish reflectance edges from shading edges.

6. **Core Idea**: Use the intensity ordinal relationships between visible and thermal infrared images—co-directional implies illumination-dominant; opposing implies reflectance-dominant—as dense self-supervised signals to decompose reflectance and shading.

## Method

### Overall Architecture
The input is a registered pair of a visible image $I_v$ and a thermal infrared image $I_t$. Physically derived ordinal constraints—both local (edge-based) and non-local (pairwise)—are combined with a reconstruction loss to drive the optimization of a Double Deep Image Prior (DDIP) network, which outputs a 3-channel albedo $\hat{\rho}$ and a single-channel shading $\hat{\eta}$. The entire process requires no pre-training or external data.

### Key Designs

1. **Visible-Thermal Ordinality Theory**:

    - **Function**: Derive constraints from physical first principles that can distinguish reflectance variation from illumination variation.
    - **Mechanism**: In a Lambertian scene, visible intensity $I_v = g\rho\eta$ and absorbed heat $\mathcal{H} = (1-\rho)\eta$. For two pixels $x_i, x_j$: if $I_v(x_i) > I_v(x_j)$ and $I_t(x_i) > I_t(x_j)$ (co-directional), then $\eta(x_i) > \eta(x_j)$ (illumination-dominant); if $I_v(x_i) > I_v(x_j)$ and $I_t(x_i) < I_t(x_j)$ (opposing), then $\rho(x_i) > \rho(x_j)$ (reflectance-dominant). A critical intermediate step: the thermal image $I_t$ serves as a monotonic proxy for absorbed heat $\mathcal{H}$ under thermal equilibrium (after neglecting thermal conduction, $\mathcal{H} = c_1 I_t - c_3$).
    - **Design Motivation**: This is the paper's central theoretical contribution—translating the unobservable ordinal relationships of albedo/shading into ordinal relationships of visible/thermal intensities that can be directly measured from images.

2. **Extension to Broadband Light Sources**:

    - **Function**: Extend the theory from purely visible light sources to sources with infrared components (e.g., sunlight, incandescent lamps).
    - **Mechanism**: When the light source contains invisible components, the heat source becomes $\mathcal{H} = (\beta - \rho_v)\eta$, where $\beta = 1 + (1-\rho_i)l_i/l_v$. Key assumption: infrared-band reflectance $\rho_i$ is approximately constant within local regions, since inter-material variation in infrared reflectance is far smaller than in the visible band. Under this assumption, $\beta$ is locally constant and the ordinal relationships still hold.
    - **Design Motivation**: The authors validate this assumption statistically using the USGS spectral reflectance database across 427 materials, finding that 94.2% of material pairs satisfy ordinal consistency.

3. **Local (Edge) Loss**:

    - **Function**: Use the relative gradient directions between the visible and thermal images to classify edges as either albedo edges or shading edges.
    - **Mechanism**: The cosine similarity between $\nabla I_v$ and $\nabla I_t$ is computed—opposing gradients (cosine $< -\epsilon_p$) indicate albedo edges, co-directional gradients ($> \epsilon_p$) indicate shading edges. Albedo edges penalize $\|\nabla\hat{\eta}\|^2$ (illumination should not change), while shading edges penalize $\|\nabla\bar{\rho}\|^2$ (reflectance should not change).
    - **Design Motivation**: Edges are the most intuitive signal for distinguishing albedo from shading; gradient direction comparison provides a highly robust discriminative criterion.

4. **Non-Local (Pairwise) Loss**:

    - **Function**: Provide long-range constraints across the image, capturing information that cannot be covered by edges alone.
    - **Mechanism**: Random point pairs $(x_i, x_j)$ are generated via Poisson disk sampling and classified into four categories ($S_+, S_-, A_+, A_-$) based on the signs of normalized intensity differences $\delta I_v$ and $\delta I_t$. A hinge loss then enforces the predicted albedo/shading to satisfy the corresponding ordinal relationships. For example, if classified as $S_+$ (both positive → illumination-dominant), the penalty is $\max(\hat{\eta}_j - \hat{\eta}_i + \varepsilon_m, 0)$.
    - **Design Motivation**: Edge constraints are local and cannot determine absolute values. Pairwise constraints supply global ordering information, extending the coverage of ordinal supervision.

5. **Double Deep Image Prior Regularization**:

    - **Function**: Provide structural priors for albedo and shading, preventing overfitting to noise.
    - **Mechanism**: Two randomly initialized encoder-decoder networks separately parameterize albedo and shading, leveraging the implicit regularization of the network architecture (DIP) to constrain the solution space. Albedo output is bounded to $[0,1]$ via sigmoid; a non-negativity penalty is applied to shading.
    - **Design Motivation**: Ordinal constraints only restrict relative ordering and cannot fully determine absolute values. The frequency preference of DIP (fitting low frequencies before high frequencies) provides natural regularization.

### Loss & Training
The total loss is $\mathcal{L} = \|\hat{\rho} \cdot \hat{\eta} - I_v\|_2 + \lambda_1 \mathcal{L}_{edge} + \lambda_2 \mathcal{L}_{ord}$, where the first term is the reconstruction loss and the latter two are ordinal constraints. The thermal image participates only in label generation for the edge and pairwise losses, not in reconstruction.

## Key Experimental Results

### Main Results (si-MSE × $10^{-2}$, ↓ lower is better)

| Method | Category | Painted Mask Albedo | Color Chart Albedo | White LED Albedo | Incandescent Albedo | Daylight Albedo |
|--------|----------|--------------------|--------------------|-----------------|--------------------|--------------------|
| RGB-Retinex | Optimization | 25 | 3.4 | 2.42 | 2.33 | 2.73 |
| Intrinsic-v2 | Learning | 27 | 2.8 | 1.25 | 4.36 | 4.17 |
| CRefNet | Learning | 38 | 8.8 | 1.79 | 2.29 | 1.98 |
| JoLHT-Video | Physics | 8.4 | 2.0 | N/A | ✗ | ✗ |
| **VT-Intrinsic** | Physics | **11** | **2.7** | **0.37** | **1.06** | **1.19** |

### Ablation Study

| Validation Scenario | Accuracy |
|--------------------|----------|
| 20 material patches + daylight | 98.59% (albedo 99.37%, shading 97.01%) |
| 20 material patches + white LED | 96.82% (albedo 94.62%, shading 100%) |
| 100 real scenes, 1063 annotated pairs | 98.95% (albedo 96.96%, shading 99.62%) |
| USGS 427-material spectral statistics | 94.2% of material pairs satisfy ordinal consistency |

### Key Findings
- VT-Intrinsic outperforms all learning-based methods across all illumination conditions without any pre-training data.
- Performance is comparable to JoLHT-Video (which requires thermal video, controlled illumination, and calibration) using only a single thermal image.
- Expert-annotated ordinal accuracy exceeds 98%, confirming the high reliability of the theory across real materials and scenes.
- Learning-based methods tend to over-smooth albedo/shading (e.g., flattening illumination on grass), while diffusion-based methods hallucinate textures.
- Incandescent and daylight experiments demonstrate robustness to light sources containing infrared components.

## Highlights & Insights
- **Elegant Exploitation of Physical Complementarity**: Visible imaging captures reflected light; thermal imaging captures absorbed heat—this complementary pair of signals naturally encodes the discriminative information needed to separate albedo from shading, a deeply elegant insight.
- **Derivation Chain from Heat Transfer to Ordinal Proxy**: Energy conservation → heat transport equation → thermal equilibrium → thermal image as monotonic proxy for absorbed heat: the theoretical derivation forms a coherent and physically intuitive chain.
- **Zero-Shot Superiority over Pre-Trained Models**: Physical constraints derived from a single image pair surpass models trained on large-scale datasets, demonstrating that the correct inductive physical bias can outperform statistical learning.

## Limitations & Future Work
- Assumes Lambertian reflectance; metallic, transparent, and specular surfaces cause failures.
- Assumes heat originates primarily from light absorption; non-optical heat sources such as engines or human bodies interfere with the model.
- Multi-chromatic illumination is not supported.
- Relies on low-cost microbolometer thermal cameras, which exhibit insufficient SNR under weak illumination or in dynamic scenes.
- Thermal camera resolution is lower than that of visible cameras, potentially limiting fine detail recovery.
- Future directions: (1) use VT-Intrinsic's high-quality pseudo ground truth to supply training data for large-scale learning methods; (2) extend the ordinal theory to multispectral imaging.

## Related Work & Insights
- **vs. JoLHT-Video**: JoLHT-Video uses the transient process in thermal video to directly estimate absorbed light intensity, requiring controlled illumination and thermal video; VT-Intrinsic relies only on ordinal relationships in steady-state thermal images, greatly broadening its applicability.
- **vs. NIR-Priors**: NIR-based methods treat NIR reflectance as a shading proxy under the assumption of small material variation, but NIR reflectance still exhibits significant inter-material variation and LED illumination lacks NIR components; VT-Intrinsic's use of the complementary thermal absorption relationship is more physically fundamental.
- **vs. Intrinsic-v2**: Although the latest learning-based method performs competitively in some indoor scenarios, it degrades under incandescent and daylight illumination (si-MSE 4.17–4.36), indicating that learned priors are insufficiently robust to illumination variation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work to leverage thermal infrared ordinal constraints for IID; the physical theory is original and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple illumination conditions, materials, and scenes with thorough ordinal theory verification; large-scale quantitative evaluation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Physical derivations are clear and rigorous; the Roger Shepard illusion example provides an exceptionally intuitive motivation.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new paradigm for exploiting light-thermal complementarity in vision; can provide large-scale real-world annotations for learning-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](../../ICLR2026/self_supervised/snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[AAAI 2026\] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](../../AAAI2026/self_supervised/neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)

</div>

<!-- RELATED:END -->
