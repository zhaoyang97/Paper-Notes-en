---
title: >-
  [Paper Note] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection
description: >-
  [AAAI 2026][anomaly detection] This paper proposes a Recursive Convolutional Autoencoder (RcAE) that progressively suppresses anomalies while preserving normal details through multi-step iterative reconstruction with sha…
tags:
  - "AAAI 2026"
  - "anomaly detection"
  - "autoencoder"
  - "recursive reconstruction"
  - "industrial defect detection"
  - "unsupervised learning"
date: 2026-05-08
content_hash: 88e39b784a417e67
---

# RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection

**Conference**: AAAI 2026
**arXiv**: [2512.11284](https://arxiv.org/abs/2512.11284)
**Code**: None
**Area**: Other
**Keywords**: anomaly detection, autoencoder, recursive reconstruction, industrial defect detection, unsupervised learning

## TL;DR

This paper proposes a Recursive Convolutional Autoencoder (RcAE) that progressively suppresses anomalies while preserving normal details through multi-step iterative reconstruction with shared parameters. Combined with a Cross-Recursive Detection module (CRD) that exploits multi-step reconstruction dynamics for robust anomaly localization, the method achieves performance comparable to state-of-the-art approaches using only 10% of the parameters required by diffusion models.

## Background & Motivation

Unsupervised industrial anomaly detection requires models trained exclusively on normal samples to identify defects at test time. Reconstruction-based methods—particularly autoencoders (AE)—have attracted broad interest due to their conceptual simplicity: a model trained on normal data should theoretically fail to reconstruct anomalous regions, thereby exposing defects through reconstruction error.

However, conventional single-pass decoding AEs suffer from four fundamental limitations:

**Overfitting**: Models tend to overfit on limited and homogeneous normal data, leading to poor generalization;

**Anomaly reconstruction**: A highly expressive latent space may reconstruct anomalous regions as well, reducing detection contrast;

**Detail loss**: Single-pass decoding over-smooths high-frequency details, producing false positives in normal regions;

**Fixed-scale limitation**: Fixed-scale architectures struggle to handle anomalies of varying sizes and severities.

Recent methods based on GANs, Transformers, and diffusion models have improved reconstruction quality, but often at the cost of high computational overhead or complex preprocessing pipelines. For instance, diffusion models require tens to hundreds of denoising steps, resulting in slow inference; pretrained-DINO-based methods (e.g., GLAD) rely on additional large vision models.

The core motivation of this paper is: **can high-quality anomaly reconstruction be achieved without complex or resource-intensive designs?** The authors propose replacing deep stacking with a recursive paradigm, achieving progressive anomaly suppression through multi-step iteration with shared parameters.

## Method

### Overall Architecture

As shown in Figure 2 of the paper, the overall framework consists of three core components trained independently in three stages:

1. **Recursive Convolutional Autoencoder (RcAE)**: Performs multi-step compression-reconstruction with shared encoder and decoder parameters to progressively suppress anomalies;
2. **Detail Preservation Network (DPN)**: Recovers high-frequency texture details lost during recursive reconstruction to reduce false positives;
3. **Cross-Recursive Detection module (CRD)**: Employs 3D convolutions to capture dynamics across multi-step reconstructions and generate pixel-level anomaly maps.

### Key Designs

1. **Recursive Convolutional Autoencoder (RcAE)**

   The core idea is to replace traditional deep stacking with parameter-sharing recursion. A conventional deep ConvAE uses $N$ independent encoding/decoding blocks, with parameter count growing linearly with $N$. RcAE instead uses a single encoder $E$ and decoder $D$, applying them recursively $N$ times to emulate the effect of a deep AE.

   **Compression stage**: The input image is recursively encoded $N$ times, with each step reducing spatial resolution via strided convolution:
   $\mathbf{I}_C^i = E(\mathbf{I}_C^{i-1}; \boldsymbol{\theta}_E), \quad i \in \{1, 2, \ldots, N\}$

   **Reconstruction stage**: Starting from the deepest compressed representation, $N$ recursive decoding steps progressively restore resolution:
   $\mathbf{I}_R^j = D(\mathbf{I}_R^{j-1}; \boldsymbol{\theta}_D), \quad j \in \{1, 2, \ldots, N\}$

   **Design Motivation**: Early iterations retain low-level details but may preserve residual anomalies; later iterations better suppress anomalies but may over-smooth. This progressive refinement allows the model to simultaneously suppress anomalies and preserve normal structure without increasing parameter count. During training, the recursion depth is randomly sampled from $[1, N]$ to prevent shortcut learning.

2. **Detail Preservation Network (DPN)**

   While recursive reconstruction effectively suppresses anomalies, it also accumulates detail loss in normal regions. DPN is a lightweight 4-layer ConvAE that takes as input the concatenation of the recursive reconstruction result $\mathbf{I}_R^n$ and the first-order gradient of the original image $\mathbf{I}'$, and predicts a residual map to recover missing details:
   $\mathbf{Res}_D^n = f_{\text{DPN}}((\mathbf{I}_R^n \oplus \mathbf{I}'); \boldsymbol{\theta}_{\text{DPN}})$
   $\mathbf{I}_D^n = \mathbf{I}_R^n + \mathbf{Res}_D^n$

   **Key Design**: RcAE is frozen during DPN training, and only normal samples are used. This forces the network to learn residuals corresponding to recursion-induced detail degradation rather than anomaly-related bias. At inference time, residuals produced in anomalous regions fall outside DPN's learned distribution, so DPN naturally fails to recover them, thereby preserving the anomaly suppression effect.

3. **Cross-Recursive Detection Module (CRD)**

   The recursive design naturally produces a reconstruction sequence, where inter-step differences reflect regional stability: normal regions stabilize quickly, while anomalous regions continue to fluctuate due to reconstruction difficulty. CRD is a 4-layer 3D ConvAE that predicts an anomaly map from the concatenation of the original image $\mathbf{I}$ and all enhanced reconstructions $\mathbf{I}_D^n$:
   $\mathbf{M}_A = f_{\text{CRD}}((\mathbf{I}_D^n \oplus \mathbf{I}); \boldsymbol{\theta}_{\text{CRD}}), \quad n \in \{1, 2, \ldots, N\}$

   3D convolutions enable CRD to extract features jointly along spatial dimensions and the recursion-step dimension, capturing temporal patterns across recursive steps. The key advantage is that reconstructions at different recursion steps are complementary: early steps preserve details but retain residual defects, while later steps suppress anomalies but lose texture. Jointly exploiting all steps provides more reliable localization.

### Loss & Training

The framework adopts a three-stage independent training strategy. All components are trained from scratch without relying on pretrained models.

**Stage 1 – RcAE Training**: A dual-term $\ell_1$ loss on intensity and edges:
$$\mathcal{L}_{\text{rec}} = \|\mathbf{I} - \mathbf{I}_R^N\|_1 + \|\mathbf{I}' - \mathbf{I}_R'^N\|_1$$
Trained for 1500 epochs with recursion depth randomly sampled per batch.

**Stage 2 – DPN Training**: RcAE is frozen; a dual-term $\ell_1$ loss is used to recover intensity and edge details:
$$\mathcal{L}_{\text{DPN}} = \|(\mathbf{Res}_D^n + \mathbf{I}_R^n) - \mathbf{I}\|_1 + \|(\mathbf{Res}_D^n + \mathbf{I}_R^n)' - \mathbf{I}'\|_1$$
Trained for 400 epochs.

**Stage 3 – CRD Training**: RcAE and DPN are frozen; pseudo-anomaly masks (color patches, random lines, copy-paste) and a dual-term $\ell_2$ loss are used:
$$\mathcal{L}_{\text{CRD}} = \|\mathbf{M}_A - \mathbf{M}_P\|_2 + \|\mathbf{M}_A' - \mathbf{M}_P'\|_2$$
Trained for 300 epochs.

Training uses the Adam optimizer with learning rate $10^{-4}$, input resolution $1024 \times 1024$, and recursion depth $N=5$.

## Key Experimental Results

### Main Results

Evaluated on the two standard industrial anomaly detection benchmarks, MVTec AD and VisA:

| Dataset | Metric | Ours (RcAE) | EfficientAD | GLAD (Diffusion+DINO) | DiffAD (Diffusion) |
|---------|--------|-------------|-------------|----------------------|-------------------|
| MVTec AD | I-AUROC | 98.9% | 99.1% | 99.3% | 98.7% |
| MVTec AD | P-AUROC | **98.7%** | 96.9% | 98.6% | 98.3% |
| VisA | I-AUROC | **99.2%** | 98.1% | 99.5% | 89.5% |
| VisA | P-AUROC | **98.6%** | 99.1% | 98.6% | - |
| Overall Avg. | I-AUROC | **99.0%** | 98.6% | 99.4% | 94.6% |
| Overall Avg. | P-AUROC | **98.7%** | 97.8% | 98.6% | - |

Key finding: RcAE achieves the best pixel-level P-AUROC of 98.7% across all methods, and the best image-level detection score among non-diffusion methods that use no pretrained models. Compared to GLAD, comparable performance is achieved without DINO or diffusion models.

### Ablation Study

| Configuration | I-AUROC / P-AUROC | Note |
|---------------|-------------------|------|
| ConvAE baseline | 82.4% / 90.8% | No recursion, no DPN, no CRD |
| + RcAE | 94.1% / 95.8% | Recursive reconstruction: +11.7% / +5.0% |
| + RcAE + DPN | 95.7% / 96.6% | Detail preservation: +1.6% / +0.8% |
| + RcAE + DPN + CRD | **98.9% / 98.7%** | Cross-recursive detection: +3.2% / +2.1% |

| Recursion Depth N | MVTec I/P-AUROC | VisA I/P-AUROC |
|-------------------|-----------------|----------------|
| N=1 | 86.2 / 87.4 | 89.3 / 88.7 |
| N=3 | 96.3 / 96.8 | 97.3 / 96.4 |
| N=5 | **98.9 / 98.7** | **99.2 / 98.6** |
| N=6 | 98.7 / 98.4 | 99.2 / 98.1 |

| CRD Input Steps $N_R$ | I-AUROC / P-AUROC |
|-----------------------|-------------------|
| 1 (final reconstruction only) | 95.7 / 96.6 |
| 3 (steps 1, 3, 5) | 98.0 / 97.1 |
| 5 (all steps) | **98.9 / 98.7** |

### Key Findings

1. **Effect of recursion depth**: The largest gains occur from $N=1$ to $N=3$ (approximately 10%); $N=5$ achieves peak performance, with slight degradation thereafter, indicating diminishing returns and mild over-smoothing risk.
2. **Importance of weight sharing**: Removing weight sharing causes performance to drop drastically from 98.9% to 71.3%, demonstrating that the shared-parameter constraint is critical for preventing anomaly memorization.
3. **Data efficiency**: RcAE trained on only 10% of the training data (84.1% / 93.4%) already surpasses the full-data ConvAE (82.4% / 90.8%), which has significant practical value in industrial scenarios.
4. **Role of skip connections in recursion**: In conventional AEs, skip connections tend to cause shortcut learning; in RcAE, however, the recursive compression-reconstruction suppresses shortcuts, allowing skip connections to effectively enhance shallow feature propagation.
5. **Computational efficiency**: The parameter count is approximately 10% of that of diffusion models, with substantially faster inference, making the method well-suited for industrial deployment.

## Highlights & Insights

1. **Elegant recursive parameter sharing**: Using multiple recursive passes of a single encoder-decoder pair to emulate a deep network achieves significant improvements in reconstruction quality without increasing parameter count—a particularly elegant architectural design.
2. **Exploiting cross-recursive dynamics**: Treating the multi-step reconstruction sequence produced during recursion as a "time series" and using 3D convolutions to capture the unstable fluctuations of anomalous regions across steps is a highly novel perspective.
3. **Modular three-stage training**: Decoupling anomaly suppression (RcAE), detail recovery (DPN), and anomaly detection (CRD) into independent training stages avoids multi-objective optimization conflicts and enhances training stability.
4. **Training from scratch**: Achieving performance comparable to pretrained diffusion-based methods without relying on any pretrained model (e.g., DINO, ImageNet) demonstrates the strong expressive capacity of the recursive mechanism itself.

## Limitations & Future Work

1. **Limited semantic-level anomaly detection**: The current design primarily targets appearance-level defects (texture, structure) and is less effective for logical anomalies that require semantic reasoning (e.g., a correctly appearing but misplaced component).
2. **Fixed recursion depth**: Although recursion depth is randomly sampled during training, it is fixed to $N=5$ at test time, with no adaptive adjustment based on sample difficulty.
3. **Dependence on pseudo-anomalies**: CRD training relies on simple geometric augmentations to generate pseudo-anomaly masks, which may not cover all real-world defect patterns.
4. **Single-modality input**: The method processes only RGB images and does not leverage multimodal information commonly available in industrial settings, such as 3D point clouds or infrared imaging.

## Related Work & Insights

- Compared to **EfficientAD**: EfficientAD uses pretrained features and knowledge distillation, whereas this work trains entirely from scratch yet achieves comparable performance, demonstrating that a well-designed inductive bias (recursive architecture) can substitute for large-scale pretraining.
- Compared to **GLAD** (Diffusion + DINO): GLAD requires a large pretrained vision model and a diffusion process, while this work achieves similar results using simple recursive ConvAEs with an order-of-magnitude fewer parameters.
- **Insights from the recursive paradigm**: The core observation underlying recursive reconstruction—"normal regions converge rapidly, anomalous regions fluctuate persistently"—is generalizable to other reconstruction-based anomaly detection scenarios, such as time-series data and point clouds.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of recursive reconstruction and cross-recursive detection constitutes a novel framework design
- **Technical Depth**: ⭐⭐⭐⭐ — The three-component co-design is thorough and well-motivated, with comprehensive ablation studies
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full evaluation on two major benchmarks with ablations covering all core design choices
- **Practicality**: ⭐⭐⭐⭐⭐ — Lightweight, trained from scratch, fast inference—highly suitable for industrial deployment
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich figures and tables

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](../../CVPR2026/others/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](../../CVPR2026/others/novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)
- [\[AAAI 2026\] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast](radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)
- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](../../NeurIPS2025/others/an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)

</div>

<!-- RELATED:END -->
