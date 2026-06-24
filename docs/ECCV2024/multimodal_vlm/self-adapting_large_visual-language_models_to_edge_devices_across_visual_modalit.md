---
title: >-
  [Paper Note] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities
description: >-
  [ECCV 2024][Multimodal VLM][Edge deployment] The EdgeVL framework is proposed to adapt large-scale VLMs (such as CLIP) to edge devices through a two-stage adaptation (dual-modality knowledge distillation + quantization-aware contrastive learning), achieving open-vocabulary cross-modality (RGB and non-RGB) classification without requiring human annotations. This achieves up to a 15.4% accuracy improvement and a 93x model compression.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Edge deployment"
  - "Knowledge distillation"
  - "Quantization"
  - "Cross-modality transfer"
  - "Contrastive learning"
date: 2026-05-08
content_hash: a26ccb44c2ee6ee7
---

# Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities

**Conference**: ECCV 2024  
**arXiv**: [2403.04908](https://arxiv.org/abs/2403.04908)  
**Code**: [GitHub](https://github.com/ramdrop/edgevl)  
**Area**: LLM/NLP  
**Keywords**: Edge deployment, Knowledge distillation, Quantization, Cross-modality transfer, Contrastive learning

## TL;DR
The EdgeVL framework is proposed to adapt large-scale VLMs (such as CLIP) to edge devices through a two-stage adaptation (dual-modality knowledge distillation + quantization-aware contrastive learning), achieving open-vocabulary cross-modality (RGB and non-RGB) classification without requiring human annotations. This achieves up to a 15.4% accuracy improvement and a 93x model compression.

## Background & Motivation
Deploying vision-language models to edge devices faces three core challenges:

**Multi-Modality Generalization**: Edge devices are typically equipped with sensors other than RGB (depth, infrared, etc.). However, the vision encoders of VLMs like CLIP are primarily trained on RGB images, and their zero-shot classification performance drops drastically on non-RGB images (e.g., the depth image accuracy is only 1/8 of RGB on the ScanNet dataset).

**Label Scarcity**: Edge devices continuously generate a massive volume of unlabeled images, and human annotation costs are prohibitively high.

**Resource Constraints**: The computational demands of large vision encoders (e.g., ViT) far exceed the memory and compute limits of edge devices.

Limitations of Prior Work:
- Cross-modality knowledge transfer methods (such as CMKD): Require labeled data and are trained only for a single modality.
- Model compression methods (quantization, pruning, distillation): Used in isolation and not co-designed with cross-modality transfer.
- Brute-force integration of these two components leads to severe performance degradation.

Our core innovation lies in: **systematically adapting large-scale VLMs to edge devices for the first time, addressing both cross-modality generalization and model compression simultaneously** without any human annotations throughout the process.

## Method

### Overall Architecture
EdgeVL is a two-stage adaptation framework:
- **Stage-1**: Dual-Modality Knowledge Distillation ($\Phi_{img} \rightarrow \Phi_{img}^{stu}$) — Transfers knowledge from a large teacher encoder (ViT-G) to a lightweight student encoder (Swin-T/DAT-T/ViT-S), enabling it to process both RGB and non-RGB images.
- **Stage-2**: Quantization-Aware Contrastive Learning ($\Phi_{img}^{stu} \rightarrow \Phi_{img}^{edge}$) — Quantizes the full-precision student model to an Int8 low-bit model and maintains or even enhances feature discriminability after quantization through contrastive learning.

### Key Designs

1. **Automatic Dataset Curation**:

    - **Function**: Automatically filters out noisy samples before distillation without human intervention.
    - **Mechanism**: Leverages ChatGPT-4 to generate a label superset $\mathcal{S}$ of scenes (e.g., indoor or satellite categories), and utilizes the image-text matching ability of CLIP to calculate a confidence score $c_i = \max_k \frac{e^{\Phi_{img}(x_i)^\top \Phi_{text}(y_k)}}{\sum_k e^{\Phi_{img}(x_i)^\top \Phi_{text}(y_k)}}$ for each RGB image.
    - Low-confidence images (containing noise or uninformative features) are discarded, preserving only those above a threshold $\tau_c = 0.25$ along with their paired non-RGB images.
    - **Design Motivation**: The teacher model also has failure cases, and using noisy samples as supervision signals would degrade distillation performance.

2. **Dual-Modality Feature Distillation**:

    - **Function**: Trains a unified student encoder capable of handling both RGB and non-RGB images.
    - **Mechanism**: For each pair of RGB and non-RGB images, aligns the student's features from both modalities with the teacher's features on the RGB image.
    - Distillation Loss: $\mathcal{L}_d = d(\Phi_{img}(x), \Phi_{img}^{stu}(x')) + d(\Phi_{img}(x), \Phi_{img}^{stu}(x))$ where $x$ represents the RGB image, $x'$ represents the corresponding non-RGB image, and $d$ is the L1 distance.
    - **Novelty**: **Weight-shared dual-modality encoder** — Instead of training separate models for each modality, a single unified encoder processes both inputs through weight sharing, reducing memory requirements by at least half.
    - **Design Motivation**: Both modalities describe the same scene, so their features should be consistent; the RGB image acts as a data augmentation for the non-RGB image.

3. **Quantization-Aware Contrastive Learning (QAT + Contrastive Learning)**:

    - **Function**: Maintains and enhances feature discriminability during the quantization process through contrastive learning.
    - **Background Observation**: Applying PTQ directly leads to a significant drop in feature discriminability, characterized by an increased angle between image features and text labels (misalignment).
    - **Mechanism**: During fake-quantization training in QAT, in addition to using traditional distillation loss, a contrastive learning loss is introduced to enhance the discriminability of quantized features.
    - **Key Finding**: QAT coupled with contrastive learning not only recovers the discriminability loss caused by quantization but also makes the quantized features superior to those of the full-precision model (with the image-text feature angle $\theta_3 < \theta_1$).
    - **Design Motivation**: Traditional distillation loss only seeks alignment with the teacher's features, which may not fully exploit the discriminative potential of quantized models in the feature space. Contrastive learning is naturally suited for learning invariant representations that are robust to quantization perturbations.

4. **Semi-Hard Triplet Sampling**:

    - **Function**: Selects effective positive and negative samples for contrastive learning.
    - **Pseudo-Label Generation**: Leverages a pre-trained VLM to match maximum similarity on the label superset to obtain pseudo-labels.
    - **Positive Sample Selection**: Samples closest in feature distance within the same pseudo-label.
    - **Negative Sample Selection**: Samples satisfying the semi-hard condition — where the negative sample distance is greater than the positive sample distance but smaller than the positive distance plus a margin $m$.
    - Contrastive Loss: $\mathcal{L}_c = \frac{1}{J}\sum_{j=1}^{J} d(f(x_i), f(p_{i,k^*})) - d(f(x_i), f(n_{i,j})) + m$
    - Hyperparameters: margin $m = 0.3$, number of negative samples $J = 3$.
    - **Design Motivation**: Semi-hard sampling is more stable than hard sampling and can better improve feature robustness.

### Loss & Training
- **Stage-1**: AdamW, learning rate $10^{-4}$ decaying to $5 \times 10^{-6}$ using a cosine scheduler, 120 epochs.
- **Stage-2**: Learning rate reduced to $10^{-6}$, utilizing per-channel weight quantization + per-tensor activation quantization.
- **The two stages must be run sequentially**: Contrastive learning requires the high-quality feature space provided by Stage-1 as a starting point. Training them in a single stage leads to severe performance degradation (50.0% vs. 30.0%).
- Teacher Model: CLIP ViT-G-14 (OpenCLIP).
- Student Model: ViT-S / DAT-T / Swin-T + feature projection heads.

## Key Experimental Results

### Main Results: Accuracy on ScanNet and EuroSAT

| Method | Precision | ScanNet Non-RGB/RGB/Avg | EuroSAT Non-RGB/RGB/Avg |
|------|------|----------------------|----------------------|
| CLIP-B | F32 | 4.5/36.2/20.4 | 16.8/40.4/28.6 |
| CLIP-G | F32 | 6.2/47.3/26.8 | 16.9/54.0/35.5 |
| SKD | F32 | 31.2/37.8/34.5 | 22.9/50.3/36.6 |
| CQD | F32 | 40.1/6.7/23.4 | 62.4/36.4/49.4 |
| **EdgeVL (DAT-T)** | **Int8** | **47.9/52.0/49.9** | **61.0/65.7/63.3** |
| **EdgeVL (Swin-T)** | **Int8** | **46.0/48.7/47.4** | **61.3/67.1/64.2** |

### Ablation Study: Quantization Strategy Comparison (ScanNet, DAT-T)

| Method | Precision | Non-RGB | RGB | Avg |
|------|------|------|-----|------|
| Stage-1 only | F32 | 38.6 | 40.6 | 39.6 |
| +PTQ | Int8 | 33.0 | 36.5 | 34.8 |
| +QAT | Int8 | 39.4 | 41.2 | 40.3 |
| +QViT | Int8 | 35.0 | 38.0 | 36.5 |
| **+Stage-2 (EdgeVL)** | **Int8** | **47.9** | **52.0** | **50.0** |

### Efficiency Comparison

| Method | Model Size | AGX Latency | Nano Latency | 4090 Throughput |
|------|---------|---------|---------|---------|
| CLIP-G | 5213 MB | / | / | / |
| CLIP-B | 330 MB | 9.5 ms | 20.2 ms | 772 img/s |
| EdgeVL (ViT-S) | 86 MB | 4.6 ms (↓52%) | 9.9 ms (↓51%) | 1492 img/s (↑93%) |
| EdgeVL (Swin-T) | 56 MB | 5.2 ms (↓46%) | 11.4 ms (↓44%) | 1098 img/s (↑42%) |

### Key Findings
- **Int8 quantization outperforms FP32 baselines**: The average accuracy of EdgeVL (Int8) far exceeds all F32 baselines (ScanNet: 49.9% vs. 34.5%; EuroSAT: 64.8% vs. 49.4%).
- **Contrastive learning is key to quantization success**: Stage-2 outperforms standard QAT by roughly 10% (50.0% vs. 40.3%) and PTQ by 15.2%.
- **Gain of dual-modality training**: Compared to single RGB or single non-RGB training, dual-modality training improves the average accuracy by 15.0% and 13.1%, respectively.
- **Cross-dataset generalization**: Training on ScanNet and transferring to NYU2 improves the depth image accuracy from 25.7% (CLIP-G) to 51.1% (EdgeVL).
- **Influence of threshold $\tau_c$**: $\tau_c = 0.25$ is optimal. Too small (0.10) leads to insufficient training, while too large (0.50) introduces noise.
- **The two stages cannot be merged**: Training in a single stage causes severe degradation (49.9% vs. 30.0%) as contrastive learning requires a high-quality feature space as a starting point.

## Highlights & Insights
- **First to systematically solve VLM edge deployment and cross-modality issues**: Integrates knowledge transfer and model compression organically rather than simply splicing them together.
- **Fully automatic without labels**: Fully labor-free throughout, spanning from dataset curation to feature distillation and quantization training.
- **Quantization outperforms full precision**: Through contrastive learning, quantization becomes a means of feature enhancement rather than degradation. This finding is highly aligned with Paper 1 (QPrompt).
- **Unified dual-modality encoder**: The weight-sharing architecture halves the memory requirement, and using RGB images as an "augmentation" for non-RGB images improves the performance of both modalities.
- **Real-world deployment validation**: Conducted actual inference latency testing via TensorRT on Jetson AGX, Nano, and RTX 4090.

## Limitations & Future Work
- **RGB performance trade-off**: RGB accuracy drops slightly in cross-dataset scenarios (due to the drastically reduced model size), which may require more adaptation data.
- **Evaluated only on scene classification**: Has not been evaluated on more complex tasks like semantic segmentation or object detection.
- **Limited adaptation data scale**: Uses only about 4,725 image pairs for adaptation; expanding the data scale might further improve generalization.
- **Label superset dependency on ChatGPT**: The quality of the label superset may affect dataset curation and pseudo-label accuracy.
- **Evaluated only on depth and SWIR non-RGB modalities**: The efficacy on other modalities like thermal infrared and near-infrared remains unverified.

## Related Work & Insights
- **CLIP**: Serves as the foundation VLM and teacher model. Its deficiency in zero-shot capabilities on non-RGB modalities is the starting point of this work.
- **CMKD**: A cross-modality knowledge distillation approach, which requires labels and only supports a single modality. EdgeVL's Stage-1 extends this to label-free dual-modality distillation.
- **LSQ/EWGS**: Gradient improvement methods in QAT that inspired the design of EdgeVL during the quantization stage.
- **Insight**: The combination of quantization and contrastive learning might be applicable to more scenarios (e.g., other edge-based downstream tasks of VLMs). The concept of "constraints as enhancement" is a promising avenue for deeper exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ The first framework to systematically solve cross-modality edge deployment of VLMs, with a highly unique two-stage design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, backbones, and GPU platforms, featuring extensive ablation studies and cross-dataset generalization.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, self-consistent two-stage logic, and substantial supplementary materials.
- Value: ⭐⭐⭐⭐ Edge deployment and cross-modality adaptation are critical bottlenecks for the practical application of VLMs, making this work highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[ECCV 2024\] Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)
- [\[ECCV 2024\] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sq-llava_self-questioning_for_large_vision-language_assistant.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[ECCV 2024\] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)

</div>

<!-- RELATED:END -->
