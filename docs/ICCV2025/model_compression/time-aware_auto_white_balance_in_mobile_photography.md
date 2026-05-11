---
title: >-
  [Paper Note] Time-Aware Auto White Balance in Mobile Photography
description: >-
  [ICCV 2025][Model Compression][Auto White Balance] This paper proposes a lightweight illumination estimation method (~5K parameters) that leverages contextual metadata from mobile devices (timestamps and geolocation) alo…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Auto White Balance"
  - "Illumination Estimation"
  - "Lightweight Model"
  - "Mobile Photography"
  - "Contextual Metadata"
date: 2026-05-08
content_hash: c9c67563e65a59f3
---

# Time-Aware Auto White Balance in Mobile Photography

**Conference**: ICCV 2025
**arXiv**: [2504.05623](https://arxiv.org/abs/2504.05623)
**Code**: None
**Area**: Model Compression
**Keywords**: Auto White Balance, Illumination Estimation, Lightweight Model, Mobile Photography, Contextual Metadata

## TL;DR
This paper proposes a lightweight illumination estimation method (~5K parameters) that leverages contextual metadata from mobile devices (timestamps and geolocation) alongside image color information. The method achieves performance on par with or superior to much larger models on a newly collected dataset of 3,224 smartphone images, and runs in under 0.25ms on a flagship mobile DSP.

## Background & Motivation
Auto white balance (AWB) is a critical step in the camera pipeline for correcting color casts induced by scene illumination. Traditional methods rely solely on image color information (raw images or color histograms) to estimate illumination chromaticity.

Mobile devices, however, provide rich contextual metadata — capture timestamps and geolocation — that can substantially constrain the illumination estimation search space. For example:
- Photos taken at **noon** exhibit a different correlated color temperature (CCT) range than those taken at **sunset**.
- **Indoor** and **outdoor** scenes have markedly different illumination conditions.
- Capture parameters such as ISO and shutter speed can help distinguish ambient brightness levels.

The root cause of the problem is that existing white balance datasets (NUS, Cube++, etc.) are predominantly collected with DSLR cameras and lack contextual information such as timestamps and geolocation; moreover, prior methods have not explored integrating such contextual cues into illumination estimation.

The paper's starting point is to convert capture time into a solar-event probability vector (dawn, sunrise, noon, sunset, dusk, midnight), and combine it with capture parameters (ISO, shutter speed, flash state) and image color histograms to train an extremely lightweight model.

## Method

### Overall Architecture
The model takes two input branches: (1) a **time-capture feature** branch that processes contextual and capture metadata to produce $\mathbf{v}_t \in \mathbb{R}^{16}$; and (2) a **histogram feature** branch that processes R/G and B/G chromaticity histograms of the raw image to produce $\mathbf{v}_h \in \mathbb{R}^{16}$. The two features are concatenated and passed through an MLP to output the illumination chromaticity $\ell_c \in \mathbb{R}^2$, which is then normalized into an RGB illumination vector.

### Key Designs
1. **Time-Capture Feature**

    - **Function**: Converts mobile device timestamps and geolocation into a solar-event probability vector useful for the model.
    - **Mechanism**: Solar event times (dawn, sunrise, noon, sunset, dusk, midnight) are computed from geolocation. The probability of each event given the capture time is: $p_g = 1 - \frac{|t_c - t_g|}{t_s}$, where $t_s = 86400$ seconds. Probabilities are square-rooted to balance the distribution, and a one-hot vector is appended to indicate whether the capture time falls before or after each event.
    - **Additional capture features**: ISO ($i$), shutter speed ($s$), flash state ($f$), and optional noise information (noise stats or SNR stats).
    - **Final feature**: $\mathbf{c} = [\mathbf{p}^T, \log(i), \log(s), f, \mathbf{w}^T]^T$, mapped to $\mathbf{v}_t \in \mathbb{R}^{16}$ via a linear layer.
    - **Design Motivation**: Converting clock time into solar-event probabilities rather than using raw hour values enables cross-timezone generalization.

2. **Histogram Feature**

    - **Function**: Compactly represents the color distribution of the raw image.
    - **Mechanism**: A 2D chromaticity histogram $\mathbf{H}_c \in \mathbb{R}^{48 \times 48}$ is computed over R/G and B/G chromaticity, weighted by pixel brightness. An edge-image chromaticity histogram $\mathbf{H}_e$ is also computed, and square roots are applied to enhance feature representation.
    - **Final feature**: $\mathbf{H} \in \mathbb{R}^{48 \times 48 \times 4}$ (color histogram + edge histogram + u/v coordinate channels), mapped to $\mathbf{v}_h \in \mathbb{R}^{16}$ via a convolutional network + ELU + adaptive average pooling + linear layer.
    - **Design Motivation**: R/G and B/G chromaticity spaces are used (rather than conventional log-uv) to maintain consistency with the model output space.

3. **Illumination Estimation Head**

    - **Function**: Fuses features from both branches and outputs illumination color.
    - **Mechanism**: $\mathbf{v} = [\mathbf{v}_t; \mathbf{v}_h] \in \mathbb{R}^{32}$ is passed through a multi-layer MLP with BatchNorm to output R/G and B/G chromaticity, which is then converted to a normalized RGB illumination vector.
    - The entire model contains only approximately 5K parameters.

### Loss & Training
- Angular error is used as the loss function.
- Adam optimizer, 400 epochs, with learning rate warmup from $10^{-6}$ to $10^{-3}$ followed by cosine annealing.
- Progressive batch size schedule: starting from 8 and doubling every 100 epochs.
- The dataset provides two types of ground truth: neutral white balance (color chart) and user-preference white balance (expert-annotated and validated via user study).

## Key Experimental Results

### Main Results
Results on the proposed test set (neutral / user-preference format):

| Method | Mean↓ | Median↓ | Best25%↓ | Worst25%↓ | Parameters |
|--------|-------|---------|----------|-----------|------------|
| GW | 6.23/5.54 | 6.01/4.52 | 1.02/1.00 | 12.43/11.91 | - |
| FFCC | 2.62/1.50 | 1.46/0.81 | 0.37/0.24 | 6.89/3.99 | 12K |
| C4 | 1.92/1.49 | 1.30/0.90 | 0.36/0.24 | 4.64/3.82 | 5,116K |
| **Ours (w/o n,r)** | **1.93/1.26** | **1.35/0.77** | **0.38/0.23** | **4.56/3.13** | **4.83K** |
| **Ours (w/ n, w/ r)** | **1.84/1.20** | **1.24/0.77** | **0.35/0.19** | **4.41/2.95** | **5.03K** |

The method also achieves state-of-the-art or near state-of-the-art results on the Simple Cube++ DSLR dataset, demonstrating cross-device generalization.

### Ablation Study

| Configuration | Mean↓ | Median↓ | Best25%↓ | Notes |
|---------------|-------|---------|----------|-------|
| Ours (w/o n, w/o r) | 1.93/1.26 | 1.35/0.77 | 0.38/0.23 | Time + basic capture features only |
| Ours (w/o n, w/ r) | 1.89/1.23 | 1.18/0.79 | 0.32/0.24 | + SNR stats |
| Ours (w/ n, w/o r) | 1.87/1.20 | 1.24/0.72 | 0.37/0.24 | + noise stats |
| Ours (w/ n, w/ r) | 1.84/1.20 | 1.24/0.77 | 0.35/0.19 | Both noise stats and SNR stats |

### Key Findings
- With only 4.83K parameters (without noise features), the proposed method matches or surpasses C4, which uses 5,116K parameters.
- The time-capture feature substantially improves estimation accuracy for outdoor scenes.
- Noise stats and SNR stats each provide complementary gains when incorporated.
- The model runs in 0.25ms on a flagship mobile DSP and 0.80ms on CPU.
- User-preference ground truth was selected in 71.95% of trials, validating the value of user-preference annotations.

## Highlights & Insights
- **Extreme lightweight design**: Achieving SOTA performance with only 5K parameters is highly appealing for on-device deployment.
- **Novel use of contextual metadata**: This work is the first to systematically incorporate mobile device timestamps and geolocation into illumination estimation.
- **Dataset contribution**: A new dataset of 3,224 smartphone raw images with both neutral and user-preference ground truth fills a gap in mobile AWB benchmarks.
- **Solar-event probability encoding**: Converting absolute time into relative probabilities with respect to solar events elegantly enables timezone-agnostic generalization.

## Limitations & Future Work
- Acquisition of noise information depends on denoised reference images or specific ISP pipelines, limiting generalizability.
- The dataset was collected with a single device model (Samsung S24 Ultra); sensor-specific characteristics may affect generalization.
- Time-based cues offer limited benefit for extreme artificial lighting conditions (e.g., colored LEDs).
- The method assumes a single global illuminant; multi-illuminant scenes require additional handling via masking.

## Related Work & Insights
- FFCC previously explored using metadata to condition the model, but did not exploit timestamps or geolocation in depth.
- The solar-event probability vector encoding introduced in this work may inspire other vision tasks that leverage temporal cues, such as night scene enhancement and scene understanding.
- The methodology for collecting user-preference white balance ground truth is a valuable reference for other image quality assessment tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The use of contextual metadata is a fresh perspective, though the overall methodology is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive baselines, ablation studies, cross-dataset evaluation, and user studies are all included.
- **Writing Quality**: ⭐⭐⭐⭐ Well-organized with detailed dataset description.
- **Value**: ⭐⭐⭐⭐ Directly applicable to mobile AWB systems in industry; dataset contribution is significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](../../NeurIPS2025/model_compression/correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](scheduling_weight_transitions_for_quantization-aware_training.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](../../NeurIPS2025/model_compression/inference-time_hyper-scaling_with_kv_cache_compression.md)

</div>

<!-- RELATED:END -->
