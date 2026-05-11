---
title: >-
  [Paper Note] Reimagining Anomalies: What if Anomalies Were Normal?
description: >-
  [AAAI2026][anomaly detection] This paper proposes the first counterfactual explanation framework for unsupervised image anomaly detection. By training a generator to modify anomalous samples into multiple disentangled co…
tags:
  - "AAAI2026"
  - "anomaly detection"
  - "counterfactual explanation"
  - "explainable AI"
  - "GAN"
  - "diffusion model"
date: 2026-05-08
content_hash: 0b96e7ac8fdb5029
---

# Reimagining Anomalies: What if Anomalies Were Normal?

**Conference**: AAAI2026
**arXiv**: [2402.14469](https://arxiv.org/abs/2402.14469)
**Code**: [liznerski/counterfactual-xad](https://github.com/liznerski/counterfactual-xad)
**Area**: Other
**Keywords**: anomaly detection, counterfactual explanation, explainable AI, GAN, diffusion model

## TL;DR

This paper proposes the first counterfactual explanation framework for unsupervised image anomaly detection. By training a generator to modify anomalous samples into multiple disentangled counterfactuals perceived as normal by the detector, the framework answers at the semantic level: "What would an anomaly look like if it were normal?" This provides a depth of interpretability far exceeding traditional heatmap-based approaches.

## Background & Motivation

Deep learning-based anomaly detection (AD) has achieved remarkable success on image benchmarks (error rates as low as ~1%), yet its black-box nature makes it difficult for users to understand why a particular sample is flagged as anomalous. This is especially concerning in safety-critical domains and settings where trust must be established.

**Limitations of Prior Work**:
- **Feature attribution methods** (heatmaps, e.g., FCDD, PaDiM, PatchCore): only localize the regions attended to by the detector, without explaining higher-level semantics.
- **Heatmaps fail to capture multi-dimensional anomalies**: for example, on Colored-MNIST, heatmaps can highlight digit shape differences but are entirely unable to explain color anomalies.
- **Existing counterfactual explanation (CE) methods**: primarily target tabular data and time series; image-based CE methods (e.g., DISSECT) apply only to supervised classifiers; diffusion-model-based CE methods for medical imaging (e.g., Sanchez 2022, Wolleb 2022) require normal/anomalous annotations and cannot be used for unsupervised AD.

**Core Idea**: Generate counterfactual samples—minimally modified versions of anomalies that are perceived as normal by the detector—thereby explaining detector decisions at the semantic level. Multiple "disentangled" counterfactuals are generated simultaneously to capture different dimensions of anomalousness (e.g., shape anomaly vs. color anomaly).

## Method

### Problem Formulation

Given an anomaly detector $\phi: \mathbb{R}^D \rightarrow [0,1]$ and a sample $\bm{x}^*$ flagged as anomalous ($\phi(\bm{x}^*) \gg 0$), a counterfactual $\bar{\bm{x}}^*$ must satisfy:
1. **Normality**: $\phi(\bar{\bm{x}}^*) \approx 0$ (perceived as normal by the detector)
2. **Minimal modification**: $\|\bar{\bm{x}}^* - \bm{x}^*\|_1 \leq \epsilon$

A concept dimension $k \in \{1, \dots, K\}$ is further introduced, requiring sufficient diversity between counterfactuals of different concepts: $\|\bar{\bm{x}}^*_k - \bar{\bm{x}}^*_{k'}\|_1 \geq \epsilon'$, enabling disentanglement.

### GAN-based Counterfactual Generation

A generator $G: \mathbb{R}^D \times [0,1] \times \{1,\dots,K\} \rightarrow \mathbb{R}^D$ is trained, taking as input the original image, a target anomaly score $\alpha$, and a concept index $k$. The overall optimization objective is:

$$\min_{G,R} \max_{\mathcal{D}} \mathbb{E}_{\bm{x} \sim P_X} \mathbb{E}_{\alpha,k} \left[ \lambda_{gan}(L_\mathcal{D} + L_G) - \lambda_\phi L_\phi + \lambda_{rec}(L_{rec} + L_{cyc}) + \lambda_r L_{con} \right]$$

This comprises five loss groups:
1. **GAN loss** ($L_\mathcal{D}, L_G$): Adversarial loss between discriminator and generator, using spectral normalization + hinge loss to ensure perceptual realism.
2. **Anomaly score loss** ($L_\phi$): Continuous binary cross-entropy driving the generated sample's anomaly score toward target $\alpha$ (set to $\alpha=0$ at inference).
3. **Reconstruction loss** ($L_{rec}$): When the target score equals the true score, the generator should reproduce the original sample, encouraging minimal modification.
4. **Cycle-consistency loss** ($L_{cyc}$): Starting from a generated sample and targeting the original score should recover the original sample, further constraining modification magnitude.
5. **Concept loss** ($L_{con}$): A concept classifier $R$ classifies counterfactuals of different concepts, driving semantic disentanglement.

### Diffusion Model Extension

For high-resolution images, DiffEdit (based on Stable Diffusion) is integrated, redefining the generator to operate in latent space:

$$G(\bm{x}, \alpha, k) = A_\Omega(G'(\psi(A_\mathcal{E}(\bm{x}), t), \alpha, k))$$

where $A_\mathcal{E}$/$A_\Omega$ are the autoencoder's encoder/decoder, $\psi$ is the DiffEdit model, and $t$ is the text prompt for the normal class. $G'$ is trained in latent space using the same loss framework as the GAN.

### Theoretical Guarantees

- **Theorem 4.2**: With only the GAN loss, the generator converges to the training data distribution $p_X$; upon adding reconstruction/cycle losses, approximate convergence still holds under the condition that the detector $\phi$ is approximately flat.
- **Theorem 4.3**: The anomaly score loss $L_\phi$ is the key factor causing the generated distribution to deviate from $p_X$; it is precisely this deviation that enables the generator to learn to map anomalies into normal space.

## Key Experimental Results

### Experimental Setup
- **Datasets**: MNIST, Colored-MNIST, CIFAR-10, GTSDB (traffic signs), ImageNet-Neighbors (INN), MVTec-AD
- **Detectors**: DSVDD, BCE (OE), HSC (OE); 80+ AD configurations in total
- **Number of concepts** $K=2$; GAN used for low-resolution datasets, diffusion model for INN

### Table 1: Normality Evaluation of Counterfactuals (AuROC, normal test set vs. counterfactuals; closer to 50% is better)

| Dataset | BCE OE | HSC OE | DSVDD |
|---|---|---|---|
| MNIST (single) | 72.0 ± 4.0 | 80.8 ± 5.3 | 75.2 ± 9.2 |
| CIFAR-10 (single) | 47.5 ± 10.0 | 49.9 ± 4.4 | 54.6 ± 3.4 |
| INN (single) | 69.1 ± 18.1 | 67.9 ± 13.2 | × |
| C-MNIST (multi) | **55.6 ± 1.5** | **55.8 ± 4.7** | 61.5 ± 4.3 |
| CIFAR-10 (multi) | **49.0 ± 8.5** | **44.4 ± 6.7** | 50.7 ± 3.3 |
| GTSDB (multi) | **50.2 ± 8.0** | **48.6 ± 14.4** | 53.1 ± 4.8 |

AuROC values on CIFAR-10 and GTSDB are very close to 50%, indicating that the counterfactuals are perceived as almost entirely normal by the detector.

### Table 2: Fidelity of Counterfactuals (FIDN, normalized to anomalous sample FID as 100%; 50–100% is a reasonable range)

| Dataset | BCE OE | HSC OE | DSVDD |
|---|---|---|---|
| MNIST (single) | 43 ± 8.1 | 68 ± 14.6 | 100 ± 8.8 |
| CIFAR-10 (single) | 116 ± 20.8 | 300 ± 90.0 | 116 ± 12.0 |
| INN (single) | 85.0 ± 28.6 | 85.4 ± 24.6 | × |
| C-MNIST (multi) | **56 ± 12.4** | 95 ± 30.5 | 83 ± 8.7 |
| MNIST (multi) | 78 ± 26.0 | 96 ± 25.0 | 100 ± 10.7 |
| GTSDB (multi) | 110 ± 101 | 95 ± 73.5 | 131 ± 118 |

Counterfactuals from BCE and HSC are comparably realistic to anomalous samples on most datasets, and even more realistic on MNIST and C-MNIST.

### Qualitative Highlights

- **Colored-MNIST** (normal = cyan digit + digit 1): BCE counterfactuals either change the digit to 1 (without altering color) or change the color to cyan (without altering the digit), achieving perfect disentanglement.
- **GTSDB** (normal = speed limit signs): All triangular anomalous signs are converted to circular shapes, revealing that the detector relies on shape features.
- **CIFAR-10** (normal = ships): Counterfactuals preserve the anomalous object's color but replace the background with water, revealing that the detector primarily relies on background features.
- **ImageNet-Neighbors** (normal = zebras): Anomalous animals such as horses and wild boars are converted to zebras while preserving pose and background.

### Revealing Classifier Bias

A supervised classifier trained only on blue anomalies is compared against an unsupervised detector trained with OE:
- Unsupervised BCE+OE AuROC: **98%**
- Supervised BCE (blue anomalies only) AuROC: **75%**
- Counterfactual visualizations clearly reveal that the supervised classifier generates unreasonable counterfactuals for anomalies of unseen colors, intuitively exposing the source of bias.

## Highlights & Insights

- **Semantic-level anomaly explanation**: The first systematic application of counterfactual explanations to unsupervised image anomaly detection, elevating interpretation from "why is this anomalous" to "what changes would make this anomaly normal"—far surpassing heatmap-based methods.
- **Multi-concept disentanglement**: Multiple counterfactuals are generated per anomaly, each capturing a different dimension of anomalousness (shape vs. color, foreground vs. background), providing comprehensive and structured explanations.
- **Theoretical and empirical guarantees**: The conditions under which the generator converges to the training distribution at Nash equilibrium are formally proven, and the anomaly score loss is identified as the critical driving force for generating effective counterfactuals.
- **Revealing detector bias**: Counterfactuals enable intuitive discovery of supervised classifier bias on anomaly subsets, providing a new tool for model auditing.
- **Flexible framework**: Supports both GAN (low-resolution) and diffusion model (high-resolution) pathways, compatible with a variety of deep AD methods.

## Limitations & Future Work

- **Limited generation quality**: On high-resolution natural images (e.g., CIFAR-10, INN), the visual quality of counterfactuals still has room for improvement, with some counterfactuals exhibiting artifacts.
- **Dependence on AD model quality**: For weaker detectors such as DSVDD, counterfactual quality degrades accordingly; the framework's effectiveness is coupled to the underlying detector's performance.
- **Fixed number of concepts**: $K$ must be specified in advance; the optimal number of concepts may vary across datasets, and an automatic determination mechanism is lacking.
- **Applicable only to semantic anomalies**: In low-level anomaly scenarios such as industrial defect detection (e.g., MVTec-AD), counterfactuals are technically correct but provide no additional insight.
- **Training overhead**: Additional training of the GAN/diffusion model generator and discriminator incurs increased computational cost.
- **Disentanglement local optima**: Under certain conditions with HSC, counterfactuals simultaneously alter both color and digit, suggesting the optimization may converge to local optima.

## Related Work & Insights

- **Feature attribution explanations**: Methods such as FCDD (fully convolutional heatmaps), PaDiM (patch distribution modeling), and PatchCore (coreset memory bank) only localize spatial anomaly regions and cannot provide semantic-level explanations.
- **Image counterfactual explanations**: DISSECT (multi-concept disentangled counterfactuals) forms the methodological foundation of this work but applies only to supervised classifiers; diffusion-model-based CE methods for medical imaging (Sanchez 2022, Wolleb 2022) require anomaly annotations.
- **AD counterfactual explanations**: Prior work on tabular/time-series data includes Angiulli 2023 and Datta 2022; for images, AR-Pro (Ji 2024) addresses local defects via masked inpainting but cannot handle semantic anomalies.
- **Positioning of this work**: The first multi-concept disentangled counterfactual explanation framework for unsupervised semantic image AD, requiring no anomaly annotations and supporting both GAN and diffusion model pathways.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic application of counterfactual explanations to unsupervised image anomaly detection; the multi-concept disentanglement framework represents a clear methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 80+ AD configurations, 5 datasets, 3 detectors, with comprehensive qualitative and quantitative analysis; however, direct quantitative comparison with other CE methods is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, theoretically rigorous, and with intuitive qualitative illustrations; notation is dense but necessary.
- Value: ⭐⭐⭐⭐ — Opens a new paradigm for AD interpretability; the progression from feature attribution to semantic counterfactuals represents an important directional advance with practical implications for model auditing and bias discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](../../CVPR2026/others/novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](../../ICCV2025/others/hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [\[CVPR 2026\] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](../../CVPR2026/others/what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)
- [\[ICCV 2025\] Processing and Acquisition Traces in Visual Encoders: What Does CLIP Know About Your Camera?](../../ICCV2025/others/processing_and_acquisition_traces_in_visual_encoders_what_does_clip_know_about_y.md)

</div>

<!-- RELATED:END -->
