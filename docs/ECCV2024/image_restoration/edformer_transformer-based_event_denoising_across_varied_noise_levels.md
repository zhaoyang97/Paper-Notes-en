---
title: >-
  [Paper Note] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels
description: >-
  [ECCV 2024][Image Restoration][Event camera] EDformer proposes an event-by-event denoising model based on Transformer, which handles event camera noise under varied noise levels by learning spatiotemporal correlations among events. It also establishes ED24, the first real-world event denoising dataset containing 21 noise levels.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Event camera"
  - "event denoising"
  - "Transformer"
  - "background activity noise"
  - "real-world dataset"
date: 2026-05-08
content_hash: a507ed6f789c17b8
---

# EDformer: Transformer-Based Event Denoising Across Varied Noise Levels

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Event camera, event denoising, Transformer, background activity noise, real-world dataset

## TL;DR

EDformer proposes an event-by-event denoising model based on Transformer, which handles event camera noise under varied noise levels by learning spatiotemporal correlations among events. It also establishes ED24, the first real-world event denoising dataset containing 21 noise levels.

## Background & Motivation

**Background**: Event cameras (e.g., DAVIS346) are novel bio-inspired vision sensors that record pixel brightness changes asynchronously, offering benefits like high temporal resolution, high dynamic range, and low power consumption. However, event cameras generate significant Background Activity Noise (BA noise) during practical operation, which severely degrades the performance of downstream tasks (such as optical flow estimation, depth estimation, and object recognition).

**Limitations of Prior Work**: Current event denoising research faces two major challenges. First, existing methods lack robustness to noise under different illumination conditions—the characteristics of BA noise change significantly with ambient lighting, with noise becoming more severe in low-light conditions, yet existing algorithms are typically trained and tested under fixed noise conditions. Second, real-world event denoising datasets are extremely scarce, and most methods rely on synthetic noise data, leading to a substantial performance drop upon actual deployment.

**Key Challenge**: The BA noise of event cameras exhibits strong spatiotemporal correlation and illumination dependency. However, existing denoising methods either consider only a single noise level or adopt simple spatiotemporal filtering strategies, failing to capture the complex spatiotemporal relationships among events across different noise levels. Furthermore, the lack of real labeled data covering diverse noise conditions makes it difficult for algorithms to learn robust noise features.

**Goal**: (1) Construct a real-world event denoising dataset that covers multiple noise levels; (2) Design an event denoising model capable of operating effectively across different noise levels; (3) Validate the effectiveness of the method in real-world application scenarios (such as low-light microscopic imaging).

**Key Insight**: The authors first systematically capture and analyze the characteristics of DAVIS346 BA noise under different illumination conditions, discovering that the noise rate is positively correlated with illumination intensity, and events at different noise levels exhibit distinct statistical characteristics in their spatiotemporal distribution. Based on this, the authors argue that a model capable of modeling long-range spatiotemporal dependencies is required to distinguish signal events from noise events. The global attention mechanism of Transformers is naturally suited for this task.

**Core Idea**: Build a real multi-noise level dataset ED24, and design a Transformer architecture EDformer to learn the spatiotemporal correlations in the event stream, achieving unified denoising across various noise levels.

## Method

### Overall Architecture

EDformer adopts an event-by-event processing approach to perform binary signal/noise classification for every event in the input event stream. The input is an original event sequence $\{(x_i, y_i, t_i, p_i)\}$, where $(x, y)$ represent pixel coordinates, $t$ is the timestamp, and $p$ is the polarity. The model learns the spatiotemporal relationships between events through a Transformer encoder, and finally outputs a confidence score for each event, indicating the probability of it being a true signal event. The pipeline consists of three stages: event representation construction, Transformer spatiotemporal encoding, and event classification.

### Key Designs

1. **Real-World Dataset ED24**:

    - **Function**: Provide the first real-world labeled dataset covering multiple noise levels for event denoising research.
    - **Mechanism**: A DAVIS346 event camera is used to collect event stream data under 21 different illumination intensities. Through a carefully designed experimental protocol, signal and noise events are separated under static scenes and known motion patterns to generate pixel-level noise annotations. The dataset covers a complete noise spectrum from extremely low illumination to bright light and includes various scene types.
    - **Design Motivation**: Existing datasets are either synthetic (meaning the noise models are inaccurate) or contain only a single noise condition, making it impossible to evaluate the robustness of algorithms under real and variable environments. ED24 fills this gap.

2. **Transformer Spatiotemporal Encoder**:

    - **Function**: Capture long-range spatiotemporal dependencies in the event stream to distinguish signal events from noise events.
    - **Mechanism**: The spatiotemporal information of events (coordinates, timestamps, polarity) is encoded into high-dimensional feature vectors, and then a multi-layer Transformer encoder is used to learn global attention relationships among events. Real signal events typically exhibit correlated spatiotemporal patterns (e.g., triggering continuously along moving edges), whereas noise events display a more random distribution. The self-attention mechanism of the Transformer can effectively capture these differences in spatiotemporal correlation. To accommodate the specific data format of event streams, the authors design a dedicated positional encoding scheme to integrate the $(x, y, t)$ position information of events into the attention computation.
    - **Design Motivation**: Traditional event denoising methods (such as neighbor-based filters or CNNs) can only capture local spatiotemporal relationships, leading to a restricted receptive field for sparsely distributed event streams. The global attention of Transformers can model connections among events over a larger spatiotemporal scope, making it highly suitable for handling the asynchronous and sparse nature of events.

3. **Noise-Level Adaptive Mechanism**:

    - **Function**: Enable the model to adaptively adjust its denoising strategy under different noise levels.
    - **Mechanism**: A noise-level-aware feature modulation is introduced into the model, allowing a single model to handle various scenarios from mild noise to severe noise. During inference, the model can either accept an estimated noise level as a conditional input or automatically infer the current noise level from the input event stream. This design avoids the overhead of training separate models for each noise level.
    - **Design Motivation**: In practical applications, environmental illumination conditions and camera noise levels constantly change. A practical denoising model must be able to cope with these dynamic changes rather than operating under a single noise condition.

### Loss & Training

Binary cross-entropy loss is employed to supervise the learning of signal/noise labels for each event. Since the ratio of signal to noise events varies significantly across different noise levels (noise events far outnumber signal events in high-noise conditions), a class balancing strategy is adopted to address label imbalance. During training, samples are uniformly drawn from data representing different noise levels to ensure the model acquires denoising capabilities across all noise levels.

## Key Experimental Results

### Main Results

| Dataset | Metric | EDformer | Prev. SOTA | Gain |
|:---:|:---:|:---:|:---:|:---:|
| ED24 (Low Noise) | AUC / F1 | Best | Second-best method | Significant |
| ED24 (High Noise) | AUC / F1 | Best | Second-best method | Substantial improvement |
| Open-source Dataset | Denoising Accuracy | SOTA | Baseline method | Consistent improvement |
| Zebrafish Vascular Imaging | Imaging Quality | Best | Traditional method | Obvious improvement |

Comprehensive comparisons on the ED24 dataset and existing open-source datasets demonstrate that EDformer achieves optimal denoising performance across various noise levels, with a particularly pronounced advantage under high-noise and low-light conditions.

### Ablation Study

| Configuration | Key Metric Change | Description |
|:---:|:---:|:---:|
| Replacing Transformer with CNN | Performance degradation | Global attention of Transformer outperforms local convolution |
| Ground truth noise level condition removed | Performance drops under high noise | The noise-adaptive mechanism is especially important under extreme noise |
| Training only with synthetic noise | Poor performance in real-world scenes | Validates the necessity of the real-world dataset ED24 |
| Joint training with single fixed noise level | Poor cross-level generalization | Multi-level joint training is critical for robustness |

### Key Findings

- The BA noise of event cameras presents distinctly different spatiotemporal statistical properties across different illumination conditions, and existing single-noise-level denoising methods fail to generalize effectively.
- The global attention of the Transformer offers clear advantages over CNNs and traditional filtering methods in event denoising, particularly in high-noise rate scenarios.
- The 21 noise levels of the ED24 dataset cover the vast majority of practical scenarios, providing a standardized evaluation benchmark for subsequent research.
- In low-light scenarios (such as zebrafish vascular imaging), the denoising of EDformer significantly improves subsequent imaging quality.

## Highlights & Insights

1. **The contribution of the ED24 dataset may be more valuable than the model itself**—it is the first real-world event denoising dataset covering multiple noise levels, which will drive denoising research in the entire event vision community.
2. The event-by-event processing design preserves the asynchronous nature of event cameras, eliminating the need to aggregate events into frames and aligning better with the philosophy of event-driven processing.
3. Validating the method in real-world application scenarios, such as zebrafish vascular imaging, enhances the practical value of this work.

## Limitations & Future Work

1. Event-by-event Transformer processing incurs high computational overhead, which may present real-time challenges in scenarios with high event rates.
2. The ED24 dataset is currently collected using only one camera (DAVIS346); the noise characteristics of different event camera models may vary.
3. Unsupervised or self-supervised methods could be introduced to reduce reliance on large amounts of labeled noise data.
4. Joint optimization with downstream tasks (e.g., optical flow estimation, SLAM) could bring further performance gains.
5. The computational complexity of the Transformer is $O(n^2)$. For event streams with high data rates, introducing efficient variants such as linear attention may be necessary.

## Related Work & Insights

- **Event Denoising Methods**: Traditional methods like kNN filtering and temporal correlation filtering are computationally simple but have limited effectiveness; deep learning methods like EventDenoisingNet introduce neural networks but are constrained by training data.
- **Event Vision Transformer**: Works like EvT introduce Transformers to event vision but primarily focus on high-level tasks (e.g., detection, segmentation). This work is the first to apply them to low-level denoising tasks.
- **Event Camera Datasets**: Datasets like DSEC and MVSEC serve different tasks; ED24 fills the data gap in the direction of denoising.
- **Insights**: The paradigm of joint training across multiple noise levels combined with conditional modeling can be extended to traditional image denoising (different ISO levels) and point cloud denoising (different sensor noise).

## Rating

- **Novelty**: ⭐⭐⭐⭐ First real multi-noise level dataset + Combination of Transformer with event denoising
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset comparisons, ablation studies, and real-world application validation
- **Writing Quality**: ⭐⭐⭐ Clear explanation of problem motivation, systematic dataset construction
- **Value**: ⭐⭐⭐⭐ High long-term value of the ED24 dataset to the community, practical utility of the method in real-world scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](../../CVPR2026/image_restoration/nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[ECCV 2024\] Blind Image Deblurring with Noise-Robust Kernel Estimation](blind_image_deblurring_with_noise-robust_kernel_estimation.md)

</div>

<!-- RELATED:END -->
