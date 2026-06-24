---
title: >-
  [Paper Note] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering
description: >-
  [ICML 2025][Object Detection][Unsupervised Anomaly Detection] By introducing the core concept of **cost volume filtering** from stereo matching and optical flow estimation into Unsupervised Anomaly Detection (UAD), this work constructs a matching cost volume between the input and templates. It utilizes a 3D U-Net with dual-stream attention guidance for denoising and filtering. Designed as a general plug-and-play post-processing module, it simultaneously boosts the performance…
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Unsupervised Anomaly Detection"
  - "Matching Cost Volume Filtering"
  - "Multi-class Anomaly Detection"
  - "Plug-and-play Module"
  - "Dual-stream Attention Guidance"
date: 2026-05-08
content_hash: 85d0e1137c6094c8
---

# CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering

**Conference**: ICML 2025  
**arXiv**: [2505.01476](https://arxiv.org/abs/2505.01476)  
**Code**: [https://github.com/ZHE-SAPI/CostFilter-AD](https://github.com/ZHE-SAPI/CostFilter-AD)  
**Area**: LLM Evaluation  
**Keywords**: Unsupervised Anomaly Detection, Matching Cost Volume Filtering, Multi-class Anomaly Detection, Plug-and-play Module, Dual-stream Attention Guidance

## TL;DR

By introducing the core concept of **cost volume filtering** from stereo matching and optical flow estimation into Unsupervised Anomaly Detection (UAD), this work constructs a matching cost volume between the input and templates. It utilizes a 3D U-Net with dual-stream attention guidance for denoising and filtering. Designed as a general plug-and-play post-processing module, it simultaneously boosts the performance of both reconstruction-based and embedding-based UAD methods, achieving state-of-the-art (SOTA) results on MVTec-AD and VisA.

## Background & Motivation

Unsupervised Anomaly Detection (UAD) is crucial for industrial quality inspection. Its core concept is to train a model using only normal samples and identify anomalous regions by **matching** the input with normal templates. Existing methods are categorized into two major classes:

**Reconstruction-based methods**: Reconstruct the normal counterpart of the image using UNet/Transformer/Diffusion, and detect anomalies via residuals or similarity. These methods suffer from the "identical shortcut" problem (where reconstruction maintains the anomaly) and spatial misalignment.

**Embedding-based methods**: Extract features using pre-trained models and identify anomalous features deviating from the normal distribution through feature distance or memory bank matching.

**Core Problem**: Regardless of the approach, the final step invariably involves matching the input with templates to obtain an anomaly score map. However, existing methods generally **ignore noise in the matching process**—calculating anomaly scores directly using L2 norm or cosine similarity, which leads to:

- Blurry boundaries between normal and abnormal regions
- Frequent false positives and false negatives
- Poor performance under simple threshold segmentation

The authors observe that this matching noise is qualitatively identical to the noise in classic visual matching tasks (e.g., stereo matching and optical flow estimation). Consequently, they propose to systematically tackle this overlooked issue by drawing inspiration from **cost volume filtering**.

## Method

### Overall Architecture

CostFilter-AD reformulates UAD as a three-stage pipeline:

1. **Image Feature Extraction**: Extract multi-layer features of the input and templates using a pre-trained DINO encoder.
2. **Anomaly Cost Volume Construction**: Construct multi-layer matching cost volumes through global similarity matching.
3. **Anomaly Cost Volume Filtering**: Denoise the cost volume using a 3D U-Net with dual-stream attention guidance.

The entire module is designed as a **general post-processing plugin** that can be seamlessly integrated into either reconstruction-based or embedding-based methods.

### Key Designs

#### 1. Template Construction Strategy

**Templates for Reconstruction-based Methods**: Instead of relying solely on the reconstructed image from the final denoising step of diffusion models, $N$ templates are sampled from the intermediate steps of the reverse denoising process. The intermediate reconstructions retain low-frequency normal information, thereby providing complementary cues:

$$I_{t \to 0} = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( I_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(I_t, t) \right)$$

**Templates for Embedding-based Methods**: Only a small number ($N$) of normal images are randomly selected as templates. By substituting giant memory banks with global matching and cost volume filtering, this approach eliminates the dependency on large storage footprints.

#### 2. Anomaly Cost Volume Construction

Perform **global similarity matching** (non-local matching) for each template feature, calculating the cosine similarity across all spatial positions between the input and template features:

$$\mathcal{V}(j, n, l, i) = \frac{f_\mathcal{S}^{i,l} \cdot f_\mathcal{T}^{n,j,l}}{\|f_\mathcal{S}^{i,l}\| \cdot \|f_\mathcal{T}^{n,j,l}\|}$$

The similarity volume is converted into an anomaly cost volume $\mathcal{C}(j,n,l,i) = 1 - \mathcal{V}(j,n,l,i)$, where larger values indicate a higher likelihood of anomaly. The final cost volume has a dimension of $\mathcal{C} \in \mathbb{R}^{(DN) \times L \times H' \times W'}$, where $DN$ denotes the matching dimension, $L$ represents the number of feature layers, and $H' \times W'$ is the spatial dimension.

An initial anomaly map $\bar{\mathcal{M}}$ is obtained via global min pooling along the matching dimension, providing a rough estimate of the anomalies.

#### 3. Dual-stream Attention-guided Cost Volume Filtering

The core innovation—the **Residual Channel-Spatial Attention (RCSA) module**—incorporates two types of guidance streams:

- **Spatial Guidance (SG)**: Input image features $f_\mathcal{S}$ provide spatial details, retaining the edge structures of subtle anomalies.
- **Matching Guidance (MG)**: The initial anomaly map $\bar{\mathcal{M}}$ focuses the model's attention on matching dimensions that are most likely to detect anomalies.

Specific implementation:

$$x_l' = \text{cat}(x_l, h(\bar{\mathcal{M}}), h(f_s^l))$$

$$x_l^{ca} = \sigma(\text{conv}(\text{MP}(x_l')) + \text{conv}(\text{AP}(x_l'))) * x_l' + x_l'$$

$$x_l^{sa} = \sigma(\text{conv}(\text{cat}(\mu(x_l^{ca}), \text{max}(x_l^{ca})))) * x_l^{ca} + x_l^{ca}$$

where $h$ is a guidance projector for channel transformation and spatial resolution alignment, and $\sigma$ is the sigmoid activation function. Channel attention first performs global feature aggregation, followed by pixel-wise refinement through spatial attention. Both components integrate residual connections to preserve delicate anomaly details.

#### 4. Class-aware Adaptor

To improve the generalization capability for multi-class anomaly detection, a Class-aware Adaptor is designed: it aggregates deep cost volume features via spatial average pooling and projects them into multi-class classification logits through fully connected layers, dynamically regularizing the segmentation loss to prioritize hard samples.

### Loss & Training

Joint loss function:

$$\mathcal{L} = \mathcal{L}_{\text{Focal}}(\mathcal{M}, \mathcal{M}_s, \sigma(\hat{Y}_c)) + \mathcal{L}_{\text{CE}}(\hat{Y}_c, Y) + \alpha \cdot (\mathcal{L}_{\text{Soft-IoU}}(\mathcal{M}, \mathcal{M}_s) + \mathcal{L}_{\text{SSIM}}(\mathcal{M}, \mathcal{M}_s))$$

| Loss Term| Function | Explanation |
|---|---|---|
| $\mathcal{L}_{\text{Focal}}$ | Handle normal/anomalous sample imbalance | $\gamma$ is dynamically adjusted by the Class-aware Adaptor: $\gamma = \gamma_0 - \sigma(\hat{Y}_c)$ for correct classification, otherwise $\gamma = \gamma_0$ |
| $\mathcal{L}_{\text{CE}}$ | Multi-class classification | Classification loss for the Class-aware Adaptor |
| $\mathcal{L}_{\text{Soft-IoU}}$ | Anomaly localization | Improve region-level accuracy |
| $\mathcal{L}_{\text{SSIM}}$ | Structural consistency | Retain structural details of anomalies |

Training details: Trained from scratch for 40 epochs with a batch size of 8, using the Adam optimizer, learning rate $1 \times 10^{-3}$, and $\alpha=0.1$. During inference, a weighted fusion is applied between the plugin's output and the baseline output: $\lambda \cdot \mathcal{M} + (1-\lambda) \cdot \mathcal{M}_{\text{baseline}}$.

## Key Experimental Results

### Main Results

Performance on MVTec-AD multi-class anomaly detection (Image AUROC / Pixel AUROC):

| Method | Image AUROC | Pixel AUROC | Gain |
|---|---|---|---|
| GLAD | 97.5 | 97.3 | — |
| **GLAD+Ours** | **98.7** | **98.2** | +1.2 / +0.9 |
| HVQ-Trans | 98.0 | 97.3 | — |
| **HVQ-Trans+Ours** | **99.0** | **98.0** | +1.0 / +0.7 |
| AnomalDF | 96.8 | 98.1 | — |
| **AnomalDF+Ours** | **98.5** | **98.8** | +1.7 / +0.7 |
| Dinomaly | 99.6 | 98.3 | — |
| **Dinomaly+Ours** | **99.7** | **98.4** | +0.1 / +0.1 |

Performance on VisA multi-class anomaly detection (Image AUROC / Pixel AUROC):

| Method | Image AUROC | Pixel AUROC | Gain |
|---|---|---|---|
| GLAD+Ours | 93.2 | 98.1 | +3.1 / +0.7 |
| HVQ-Trans+Ours | 93.4 | 98.6 | +2.1 / +0.1 |
| AnomalDF+Ours | 94.3 | 99.2 | +3.8 / +1.7 |

### Ablation Study

| Configuration | I-AUROC / P-AUROC | Explanation |
|---|---|---|
| DN→depth (stereo matching style) | 87.8 / 89.0 | Global matching mapped to depth dimension leads to feature pollution |
| DN→channel + $\mathcal{C}_0$ | 96.2 / 96.8 | Using only the final denoising step template |
| + $\mathcal{C}_{N-1}$ (intermediate templates) | 96.7 / 97.3 | Multi-step templates provide a gain of +0.5 / +0.5 |
| + SG (Spatial Guidance) | 97.8 / 97.5 | Spatial attention brings significant improvement |
| + MG (Matching Guidance) | 98.3 / 97.8 | Dual-stream guidance further enhances performance |
| + $\mathcal{L}_{CE}$ (Class-aware) | 98.5 / 98.0 | Adaptor improves multi-class generalization |
| + $\mathcal{L}_{S}$ (Full Loss) | **98.7 / 98.2** | All components collaborate to achieve optimal performance |

### Key Findings

1. **Matching noise is a neglected bottleneck**: Even state-of-the-art (SOTA) UAD methods (such as Dinomaly) still improve after integrating CostFilter-AD, indicating that matching noise is ubiquitous.
2. **Global matching should be mapped to the channel dimension**: Directly copying the depth-dimension mapping from stereo matching leads to severe performance degradation (87.8% vs. 96.2%), as the global matching patterns in anomaly detection differ from local pixel-wise matching.
3. **Templates from intermediate denoising steps are valuable**: They preserve low-frequency normal information and form a complement with the final reconstruction.
4. **Acceptable computational overhead**: The plugin adds only about 43M parameters and 26-33G FLOPs, with inference latency increasing by 0.04-0.37s/image.
5. **Hybrid models can match or even outperform single-type models**: A unified model alternately trained with reconstruction-based and embedding-based cost volumes exhibits robust performance.

## Highlights & Insights

- **Novelty of Perspective**: Reformulating UAD into a three-step pipeline ("feature extraction $\to$ cost volume construction $\to$ cost volume filtering") establishes a unified perspective with stereo matching and optical flow estimation, introducing well-established matching optimization methodologies to UAD.
- **General and Plug-and-Play**: Without modifying the architectures and training workflows of base methods, it serves purely as a post-processing module to yield performance gains. Its effectiveness has been validated across 5 distinct baselines (UniAD, GLAD, HVQ-Trans, AnomalDF, Dinomaly) and 4 datasets.
- **Exquisite Dual-Stream Attention Design**: Combining Spatial Guidance for edge preservation with Matching Guidance for focusing on anomalies. These two streams complement each other, as proved necessary by the ablation study.
- **Intelligent Multi-Template Strategy**: For diffusion models, it utilizes reconstructions from different denoising steps as multiple templates. For embedding-based methods, only a small set of normal samples is required, effectively replacing massive memory banks.

## Limitations & Future Work

1. **Reliance on the presence of anomalous signals in the cost volume**: When input resolution is extremely low or feature extraction is insufficient, the cost volume may lack anomaly-related signals, rendering the filtering process unable to recover them.
2. **Additional computational overhead**: Although relatively acceptable, for heavy diffusion-based baselines like GLAD, the addition of 2GB VRAM and 0.37s latency remains non-negligible.
3. **Room for improvement in unified multi-class models**: Performance gains on certain categories (e.g., Screw, Capsules) remain limited, and the diversity of cross-class anomalies continues to pose a challenge.
4. **Unexplored model compression**: Whether the 43M parameters of the 3D U-Net can be compressed via distillation or pruning remains unexplored.

## Related Work & Insights

- **Cost volume filtering in stereo matching/optical flow** (Hosni et al., 2012; Kendall et al., 2017): The core inspiration of this paper, demonstrating the value of cross-domain methodology transfer.
- **Dinomaly** (Guo et al., 2025): One of the strongest current multi-class UAD baselines, upon which CostFilter-AD can still yield improvements.
- **GLAD** (Yao et al., 2024): A diffusion-model-based UAD method, serving as a primary target for integration in this work.
- **AnomalyDINO / AnomalDF** (Damm et al., 2025): Embedding-based methods, where this paper demonstrates that their matching noise can likewise be mitigated via filtering.
- **Insights**: The concept of denoising matching noise can be extended to other tasks dependent on feature matching (e.g., few-shot segmentation, image retrieval), illustrating the general applicability of the "cost volume construction $\to$ filtering" paradigm.

## Rating

| Dimension | Score (1-5) | Explanation |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Introducing cost volume filtering into UAD is a novel perspective, though the core technique is borrowed from stereo matching |
| Practicality | ⭐⭐⭐⭐⭐ | Plug-and-play design, validated across multiple methods and datasets, high industrial deployment value |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Solid experimental design with 5 baselines, 4 datasets, 7 metrics, and detailed ablation studies |
| Writing Quality | ⭐⭐⭐⭐ | Clear arguments, rich figures and tables, well-articulated motivation |
| Overall | ⭐⭐⭐⭐ | A solid ICML-level paper presenting an elegant solution to an overlooked problem |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](../../CVPR2025/object_detection/mulsen_ad_multi_sensor_anomaly_detection.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](../../CVPR2025/object_detection/aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/object_detection/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)

</div>

<!-- RELATED:END -->
