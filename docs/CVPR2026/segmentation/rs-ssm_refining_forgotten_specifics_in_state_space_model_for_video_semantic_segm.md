---
title: >-
  [Paper Note] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Video Semantic Segmentation] RS-SSM is proposed to extract channel-wise specific information distribution features (CwAP) via frequency domain analysis and adaptively invert the forget gate matr…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Video Semantic Segmentation"
  - "State Space Model"
  - "Forget Gate Refinement"
  - "Frequency Domain Analysis"
  - "Mamba"
date: 2026-05-08
content_hash: 94a3e7ab3a8777d3
---

# RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.24295](https://arxiv.org/abs/2603.24295)
**Code**: [https://github.com/zhoujiahuan1991/CVPR2026-RS-SSM](https://github.com/zhoujiahuan1991/CVPR2026-RS-SSM)
**Area**: Semantic Segmentation / Video Understanding
**Keywords**: Video Semantic Segmentation, State Space Model, Forget Gate Refinement, Frequency Domain Analysis, Mamba

## TL;DR

RS-SSM is proposed to extract channel-wise specific information distribution features (CwAP) via frequency domain analysis and adaptively invert the forget gate matrix (FGIR) to complementarily refine spatiotemporal details lost during SSM state space compression, achieving state-of-the-art performance on four video semantic segmentation benchmarks while maintaining high efficiency.

## Background & Motivation

1. **Background**: Video semantic segmentation (VSS) requires assigning semantic labels to every pixel in each frame while maintaining temporal consistency. Early methods model inter-frame motion via optical flow, which is computationally expensive and noise-prone; Transformer-based methods aggregate spatiotemporal information through global attention but incur quadratic complexity. Recent SSM-based methods efficiently compress and propagate spatiotemporal semantic information with linear complexity.

2. **Limitations of Prior Work**: SSMs compress sequence information via a fixed-size state space, which is effective for retaining common semantics (global structures, smooth regions) but inevitably discards specific information (boundaries, textures, local variations), resulting in segmentation outputs that only coarsely localize objects with blurred details.

3. **Key Challenge**: The essence of state space compression is representing an arbitrarily long sequence with a finite-dimensional hidden state. The forget gate $\bar{A}_d = \exp(\Delta_d A_d)$ governs the degree of decay applied to historical information. Smaller forget gate values lead to aggressive compression, while values close to 1 retain more information—for pixel-level VSS, the standard forget gate's decay strategy systematically discards high-frequency details.

4. **Goal**: To compensate for the spatiotemporal specific information lost during SSM state space compression while preserving the linear complexity advantage of SSMs.

5. **Key Insight**: Different channels of the hidden state encode varying amounts of specific information. The proportion of high-frequency energy per channel can be quantified via frequency domain analysis, enabling targeted refinement of channels rich in high-frequency information by inverting the forget gate.

6. **Core Idea**: Identify channels that are "rich in specific information" via frequency domain analysis, and adaptively invert the forget gate to compensatorily recover spatiotemporal details suppressed by compression.

## Method

### Overall Architecture

Input video frames $\{I_t\}_{t=1}^T$ are processed by an image encoder to extract features $\{M_t\}$, which are then fed into $L$ dual-path SSM layers. Each layer contains two SSM modules $\theta_1, \theta_2$: $\theta_2$ is a standard SSM for extracting common semantics, while $\theta_1$ employs CwAP and FGIR modules to refine the specific information forgotten by $\theta_2$. The outputs of both paths are concatenated and fused via an MLP, then passed to a linear segmentation decoder to generate per-frame segmentation masks.

### Key Designs

1. **Channel-wise Amplitude Perceiver (CwAP)**:

    - **Function**: Quantifies the amount of specific information (high-frequency details) contained in each channel.
    - **Mechanism**: A 2D FFT is applied to the projected features $H_t \in \mathbb{R}^{D \times H \times W}$ to obtain the amplitude spectrum $H_t^m$. The frequency domain is partitioned into $K$ frequency bands by normalized frequency radius, where low-frequency bands correspond to common semantics and high-frequency bands correspond to specific information. The normalized energy ratio of each channel within the top $k_h$ high-frequency bands is computed, yielding a spectral feature $F_t \in \mathbb{R}^D$ that reflects the amount of specific information per channel.
    - **Design Motivation**: Different channels of the hidden state encode information at different semantic granularities. Extracting the high-frequency energy ratio in the frequency domain provides an unsupervised, label-free metric for localizing "detail-rich" channels.

2. **Channel Information Loss ($\mathcal{L}_{ci}$)**:

    - **Function**: Aligns the channel distribution of specific information across samples.
    - **Mechanism**: Spectral features are L2-normalized, and the cosine similarity matrix $\mathbf{S}_{i,j}$ is computed over all frame pairs within a batch. The loss is defined as $\mathcal{L}_{ci} = 1 - \frac{1}{|\mathcal{B}|^2} \sum_i \sum_j \mathbf{S}_{i,j}$, i.e., maximizing the mean cosine similarity.
    - **Design Motivation**: If specific information is scattered across different channels for different samples, FGIR cannot perform consistent refinement. By aligning channel distributions, specific information is concentrated in similar channel subsets across samples, enabling more efficient and consistent refinement.

3. **Forget Gate Information Refiner (FGIR)**:

    - **Function**: Adaptively inverts the forget gate to guide the SSM to focus on forgotten specific information.
    - **Mechanism**: The spectral features $F_t$ extracted by CwAP are used to adaptively modify the forget gate matrix of $\theta_1$. The core operation is "forget gate inversion"—channels that are most heavily decayed (i.e., most forgotten) in the standard SSM $\theta_2$ are correspondingly assigned higher retention in $\theta_1$. Specifically, the forget gates of $\theta_2$ indicate which channels undergo the most aggressive compression; FGIR inverts these forget gate values weighted by the spectral features, enabling $\theta_1$ to perform compensatory refinement of $\theta_2$ on these channels.
    - **Design Motivation**: The dual-path design assigns $\theta_2$ the role of standard SSM modeling for common semantics, while $\theta_1$ specializes in recovering details lost by $\theta_2$. Forget gate inversion ensures the two SSMs are complementary rather than redundant in their focus.

### Loss & Training

The total loss comprises the standard cross-entropy loss for semantic segmentation and the channel information loss $\mathcal{L}_{ci}$. The image encoder is initialized with SegFormer pretrained weights. GFLOPs and FPS are computed at an input resolution of 480×853. The number of dual-path SSM layers $L$ is a hyperparameter. The number of frequency bands $K$ and high-frequency bands $k_h$ are key hyperparameters of CwAP.

## Key Experimental Results

### Main Results

RS-SSM is compared against existing methods on three benchmarks: VSPW, NYUv2, and CamVid. As described in the abstract and method, RS-SSM achieves the best or second-best segmentation accuracy while maintaining high efficiency (linear complexity):

| Dataset | Metric | RS-SSM | Note |
|---------|--------|--------|------|
| VSPW | mIoU | SOTA | Large-scale VSS benchmark |
| NYUv2 | mIoU | SOTA | Indoor scene segmentation |
| CamVid | mIoU | SOTA | Driving scene video segmentation |
| 4 benchmarks overall | mIoU + GFLOPs + FPS | Best accuracy with sustained efficiency | Linear vs. quadratic Transformer complexity |

(Note: The experimental data tables in the cached file are truncated; refer to Table 1 in the original paper for specific values.)

### Ablation Study

Based on the method design, the ablation study should include:

| Configuration | Description |
|---------------|-------------|
| Full RS-SSM | Complete model with CwAP + FGIR + $\mathcal{L}_{ci}$ |
| w/o CwAP | Remove frequency-domain channel analysis; FGIR receives no spectral feature guidance |
| w/o FGIR | Remove forget gate inversion; reduce to standard dual-path SSM only |
| w/o $\mathcal{L}_{ci}$ | Remove channel alignment loss |
| Single-path SSM | Remove dual-path design; degenerate to standard SSM (e.g., TV3S) |

### Key Findings

- **Forget gate inversion is the core contribution**: Visualizations of the update gate $\bar{B}_d$ show that information in detail regions is severely attenuated in the standard SSM $\theta_2$, while $\theta_1$ with inverted forget gates effectively recovers these forgotten boundary and texture details.
- **CwAP frequency domain analysis provides an effective channel selection signal**: Channels with high proportions of high-frequency energy indeed correspond to more boundary and texture information, making the spectral feature an unsupervised and computationally efficient measure of channel importance.
- **Effect of the channel information loss alignment**: Cross-sample alignment enables FGIR to consistently refine similar channel subsets, avoiding training instability caused by inconsistent refinement directions across samples.
- **Linear complexity advantage**: RS-SSM preserves the linear efficiency of SSMs, avoiding the quadratic complexity bottleneck of Transformer methods on long videos.

## Highlights & Insights

- **The idea of "the complement of forgetting is refinement" is highly elegant**: Rather than attempting to prevent SSMs from forgetting—which would undermine their compression advantage—a complementary SSM is employed to specifically recover what has been forgotten. This "division of labor" dual-path design is more elegant than simply increasing the state space dimensionality.
- **Frequency domain as a channel analysis tool**: Using the high-frequency energy ratio of the FFT amplitude spectrum to quantify "specific information content" is intuitive in principle (high frequency = boundaries/textures), efficient in computation (FFT is $O(n \log n)$), and label-free. This frequency domain analysis is transferable to any scenario requiring differentiation between coarse-grained and fine-grained information.
- **Systematic improvement of SSMs for vision**: The paper identifies the key bottleneck of SSMs in pixel-level tasks (forgetting details) and provides a targeted solution, offering guidance for the further development of Mamba in the visual domain.

## Limitations & Future Work

- The dual-path design increases parameter count and computation (though still linear complexity); it remains to be verified whether the efficiency advantage holds at higher resolutions or for longer videos.
- The number of frequency bands $K$ and high-frequency bands $k_h$ in CwAP require manual specification and may need adjustment for different datasets.
- Validation is limited to four predefined VSS benchmarks; performance on panoptic/instance-level video segmentation has not been tested.
- It is unclear whether forget gate inversion may cause over-retention of certain long-range dependency information, resulting in "information overload."
- A learnable frequency-domain selection mechanism could be explored to replace the fixed frequency band partition.

## Related Work & Insights

- **vs. TV3S**: TV3S is the first method to apply SSMs to VSS, but overlooks detail loss caused by state space compression. RS-SSM directly addresses this issue via forget gate inversion for compensatory refinement.
- **vs. Transformer-based VSS (CFFM/MRCFA)**: Transformer methods naturally preserve details via global attention but incur quadratic complexity. RS-SSM achieves comparable or superior accuracy at linear complexity through dual-path design with frequency-domain guidance.
- **vs. VideoMamba**: VideoMamba is a general-purpose video backbone; RS-SSM designs a task-specific forget gate refinement strategy tailored for pixel-level segmentation, representing a task-specific improvement over general SSMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of forget gate inversion and frequency-domain channel analysis is novel; the perspective of "refining forgotten details within SSMs" is thought-provoking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 4 benchmarks is broad, though specific numerical values are truncated in the cache; update gate visualizations provide intuitive validation.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, and visualizations of the dual-path architecture and frequency domain analysis are clear.
- Value: ⭐⭐⭐⭐ Provides a systematic solution for SSM application in pixel-level vision tasks, advancing the development of Mamba in the visual domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](../../ICCV2025/segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)

</div>

<!-- RELATED:END -->
