---
title: >-
  [Paper Note] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line
description: >-
  [CVPR2025][Object Detection][Anomaly Detection] A semi-supervised anomaly detection framework based on GAN and Dual-stage Residual Autoencoder (DRAE) deployed for real-time online quality inspection on high-speed pharmaceutical BFS lines. Trained exclusively on normal samples, it achieves a single-patch inference time of 0.17 ms, optimizing reconstruction quality via Perlin noise augmentation and Noise Loss.
tags:
  - "CVPR2025"
  - "Object Detection"
  - "Anomaly Detection"
  - "GAN"
  - "Residual Autoencoder"
  - "Industrial Deployment"
  - "BFS Production Line"
  - "Real-time Inference"
date: 2026-05-08
content_hash: f6b6cce5774b314a
---

# Integration of deep generative Anomaly Detection algorithm in high-speed industrial line

**Conference**: CVPR2025  
**arXiv**: [2603.07577](https://arxiv.org/abs/2603.07577)  
**Code**: None (Industrial closed-source)  
**Area**: Others  
**Keywords**: Anomaly Detection, GAN, Residual Autoencoder, Industrial Deployment, BFS Production Line, Real-time Inference

## TL;DR
A semi-supervised anomaly detection framework based on GAN and Dual-stage Residual Autoencoder (DRAE) deployed for real-time online quality inspection on high-speed pharmaceutical BFS lines. Trained exclusively on normal samples, it achieves a single-patch inference time of 0.17 ms, optimizing reconstruction quality via Perlin noise augmentation and Noise Loss.

## Background & Motivation
**Background**: BFS (Blow-Fill-Seal) lines in the pharmaceutical industry require non-destructive visual quality inspection. Currently, they still heavily rely on manual visual inspection, which is limited by fluctuations in operator attention and throughput bottlenecks.

**Limitations of Prior Work**: Classic rule-based computer vision algorithms (thresholding + template matching) are highly dependent on specific product configurations, feature excessive parameters, scale poorly, and struggle to distinguish process noise (e.g., bubbles in liquid) from genuine defects (e.g., foreign particles).

**Limitations of Supervised Learning**: In industrial scenarios, conforming products vastly outnumber defective ones, leading to severe class imbalance where supervised classification methods cannot be directly applied.

**Key Challenge**: High-precision anomaly detection and spatial localization must be achieved under strict hardware constraints (Industrial PC + A4500 GPU) and time constraints (500 ms acquisition interval).

**Key Insight**: A semi-supervised approach is adopted where a GAN is trained solely on normal samples to detect anomalies via reconstruction residuals, with engineering optimization tailored for industrial production lines to achieve a truly deployable system.

## Method

### Overall Architecture
The system improves upon GRD-Net and consists of a GAN network:
- **Generator G**: An Encoder-Decoder-Encoder structure, at the core of which is a Dual-stage Residual Autoencoder (DRAE) using a fully-connected bottleneck layer (64-dimensional features). The encoder uses a 4-stage ResNet v2 architecture (outputting 16×16×1024), and the decoder is a symmetric transposed convolutional structure.
- **Discriminator C**: A convolutional encoder + fully connected layer, performing true/false binary classification.

### Loss & Training
1. **Adversarial Loss** $\mathcal{L}_{adv}$: Matches feature activations in the final convolutional layer of the discriminator.
2. **Context Loss** $\mathcal{L}_{con}$: A weighted combination of Huber Loss (replacing the original L1 for improved stability) and SSIM Loss.
3. **Encoder Consistency Loss** $\mathcal{L}_{enc}$: Restricts the latent representations of the original and reconstructed images to be consistent.
4. **Noise Loss** $\mathcal{L}_{nse}$ (newly introduced in this work): Explicitly supervises the network's reconstruction behavior on Perlin noise regions, preventing entropy increase from the augmentation from destabilizing optimization.

Total generator objective: $\mathcal{L}_{gen} = w_1 \cdot \mathcal{L}_{adv} + w_2 \cdot \mathcal{L}_{con} + w_3 \cdot \mathcal{L}_{enc} + w_4 \cdot \mathcal{L}_{nse}$

### Perlin Noise Augmentation
- Overlays Perlin noise on normal samples with a probability of $q=0.75$ to formulate a denoising task.
- The noise blending factor $\beta \sim \mathcal{U}(0.5, 1.0)$ controls noise intensity.
- Additional random rotations in $[-\pi/8, \pi/8]$ and vertical flips are applied, excluding horizontal flips that could generate unrealistic patterns.

### Anomaly Scoring and Localization
- Anomaly score: $\phi = 1 - \text{SSIM}(X, \hat{X})$
- Heatmap: min-max normalization of $H = |X - \hat{X}|$
- Thresholds are tuned on an independent calibration set and configured differently for each region.

### Preprocessing Pipeline
- Each BFS strip contains 5 ampoules, with each ampoule further divided into 4 logical regions (label region, upper body, liquid level region, bottom).
- 20 gray-scale patches of size 256×256 are extracted from each frame.
- Training set: 782 strips × 10 acquisitions × 16 frames × 20 patches = 2,815,200 images.

## Key Experimental Results

### Detection Performance (Real Industrial Test Set: 141 Defective + 120 Conforming)

| Level | Accuracy | TPR | TNR | Balanced Accuracy |
|------|--------|-----|-----|-----------|
| Patch (Overall) | 99.19%~99.91% | 99.66%~99.94% | 90.44%~99.73% | 95.15%~99.84% |
| Product Level | 95.93% | 96.94% | 94.67% | 95.81% |
| Run Level (7/10 Voting) | 96.41% | 96.76% | 95.99% | 96.38% |

### Inference Speed
- Single-patch inference time: **0.1689 ms** (A4500 GPU)
- Single-product inference time: **0.4873 ms** (<500 ms acquisition interval, satisfying real-time constraints)

### Training Details
- Training set of 2.815 million patches, trained for 10 epochs.
- Learning rate of 1.5×10⁻⁴ with cosine decay restarts, batch size of 32.
- Training hardware: A100 40GB; Inference hardware: A4500 20GB.

### Industrial Deployment Details
- The inference side is integrated into the machine control software via the C++ TensorFlow API.
- A rotary online quality inspector grabs products from the conveyor belt into a carousel, performing automated sorting post-detection.
- Telecentric lenses are utilized to eliminate geometric distortion, particularly at the sides and bottom.
- A rank filter is applied during training to generate the brightest/darkest images, increasing the variability of normal samples.
- A run-level voting strategy (at least 7 out of 10 consistent acquisitions) is designed to simulate the decision logic of human inspectors.

### Position relative to related methods
- Compared to embedding similarity methods like **PaDiM/PatchCore**: This work opts for a reconstruction-based approach because it provides intuitive, explainable heatmaps, matching the industrial customers' requirement for visual feedback on anomaly locations.
- Compared to **DRÆM**: Although sharing the Perlin noise augmentation concept, DRÆM relies on an extra U-Net segmentation network. This work removes the segmentation branch and retains only the generator to meet inference time constraints.
- Compared to **EfficientAD**: While EfficientAD aims for lightweight execution, it is still based on embedding distances. This work provides pixel-level explanations based on reconstruction residuals.

## Highlights & Insights
- **Genuine Industrial Deployment**: Rather than a theoretical laboratory paper, this work details the entire workflow from data acquisition and model training to production line integration (C++ TensorFlow API), including GMP compliance validation.
- **Practical Innovation in Noise Loss**: By explicitly constraining reconstruction behavior in noisy regions, it resolves the training instability issue introduced by Perlin noise augmentation.
- **Awareness of Engineering Constraints**: Design trade-offs are made under strict hardware (Industrial PC + A4500) and time (500 ms slot) limitations, compressing the model significantly using a fully connected bottleneck layer (64-dimensional).
- **Multi-Level Evaluation Protocol**: A three-level aggregation protocol (patch $\rightarrow$ product $\rightarrow$ run level) is established, adhering to industrial acceptance standards.
- **Heatmap Explainability**: Anomaly region heatmaps are automatically generated via reconstruction discrepancies, providing operators with intuitive defect localization on the HMI.

## Limitations & Future Work
- Validated only on a single product (BFS vials), without comparative evaluation on public benchmarks such as MVTec.
- Restricted by NDA, the dataset and full pipeline cannot be publicly released, making verification or follow-up difficult for academia.
- Quantitative comparison with current SOTA methods like PatchCore or EfficientAD is absent, failing to evaluate competitive performance under general scenarios.
- TNR is relatively low in certain regions (R0: 90.93%, R1: 90.44%), where an approximately 10% false positive rate could cause a massive amount of false rejections on high-speed lines.
- Threshold tuning relies on manual configuration on a validation/calibration set, lacking an adaptive mechanism to handle batch-to-batch product variations.
- The liquid level region exhibits high variability due to bubble movement; the paper does not thoroughly discuss the robustness in this region.
- Employs TensorFlow 2.x instead of PyTorch, limiting ecosystem compatibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The method itself is an engineering improvement on GRD-Net/DRÆM, though Noise Loss exhibits certain novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation on real-world industrial data is compelling, but public benchmark comparisons are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Engineering details are abundant, but mathematical notation is occasionally disorganized, and the structure is slightly redundant.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable reference for industrial anomaly detection deployment, bridging the gap between academic research and production line deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/object_detection/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](../../AAAI2026/object_detection/rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](../../NeurIPS2025/object_detection/multimodal_generative_flows_for_lhc_jets.md)
- [\[ECCV 2024\] Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](../../ECCV2024/object_detection/self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)

</div>

<!-- RELATED:END -->
