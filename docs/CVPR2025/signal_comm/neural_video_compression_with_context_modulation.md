---
title: >-
  [Paper Note] Neural Video Compression with Context Modulation
description: >-
  [CVPR 2025][Signal & Communication][Neural Video Compression] Proposed the DCMVC framework, which modulates temporal context in two steps: flow orientation and context compensation. By fully utilizing reference information in both the pixel domain and the feature domain, it achieves compression performance that saves an average of 22.7% bitrate compared to H.266/VVC and 10.1% bitrate compared to the previous SOTA, DCVC-FM.
tags:
  - "CVPR 2025"
  - "Signal & Communication"
  - "Neural Video Compression"
  - "Temporal Context Modeling"
  - "Conditional Coding"
  - "Flow Guidance"
  - "Feature Compensation"
date: 2026-05-08
content_hash: 63ead5f70a70d3eb
---

# Neural Video Compression with Context Modulation

**Conference**: CVPR 2025  
**arXiv**: [2505.14541](https://arxiv.org/abs/2505.14541)  
**Code**: [https://github.com/Austin4USTC/DCMVC](https://github.com/Austin4USTC/DCMVC)  
**Area**: Signal and Communication  
**Keywords**: Neural Video Compression, Temporal Context Modeling, Conditional Coding, Flow Guidance, Feature Compensation

## TL;DR
Proposed the DCMVC framework, which modulates temporal context in two steps: flow orientation and context compensation. By fully utilizing reference information in both the pixel domain and the feature domain, it achieves compression performance that saves an average of 22.7% bitrate compared to H.266/VVC and 10.1% bitrate compared to the previous SOTA, DCVC-FM.

## Background & Motivation

**Background**: The current mainstream paradigm for Neural Video Compression (NVC) is the conditional coding framework, represented by the DCVC series. This framework removes temporal redundancy by extracting temporal context from propagated reference features as coding/decoding conditions.

**Limitations of Prior Work**: Propagated reference features accumulate irrelevant information over long prediction chains, leading to a frame-by-frame degradation in temporal context quality. Although DCVC-FM proposed a periodic refresh mechanism (switching to reference frames within a fixed period), manual switching with a fixed period fails to fully utilize reference information.

**Key Challenge**: Propagated reference features contain more information but also carry more irrelevant noise, whereas reference frames are constrained by distortion loss and are thus "cleaner," but directly using reference frame information is not as rich as using feature-domain information. The two reference sources have their own strengths and weaknesses, and existing methods fail to merge them effectively.

**Goal**: Design a context modulation scheme that can simultaneously utilize reference information in both the pixel domain and the feature domain to generate high-quality temporal contexts and alleviate error propagation.

**Key Insight**: The authors observe that the information provided by propagated reference features and reference frames is complementary—reference frames have clearer edges and smaller prediction errors, while propagated features contain richer high-level semantics.

**Core Idea**: Generate an additional "oriented temporal context" using reference frames, and then fuse it with the propagated temporal context through a global-local collaborative mechanism to remove irrelevant information and generate a superior compensated context.

## Method

### Overall Architecture
DCMVC is built upon the conditional coding framework DCVC-DC. Inputting the current frame $x_t$ and the reference frame $\hat{x}_{t-1}$, motion estimation yields optical flow $v_t$, which is compressed to obtain the decoded optical flow $\hat{v}_t$. Utilizing $\hat{v}_t$ and the propagated reference feature $F_{t-1}$, multi-scale temporal context mining is performed to obtain propagated temporal contexts at three scales: $C_t^0, C_t^1, C_t^2$. The core innovation lies in: modulating the largest-scale context $C_t^0$—first generating the oriented context $\check{C}_t^0$ from the reference frame through flow orientation, and then fusing it with the propagated context $C_t^0$ via context compensation to ultimately obtain the compensated context $\bar{C}_t^0$. This compensated context, along with contexts of other scales, serves as conditions fed into the encoder, entropy model, and decoder.

### Key Designs

1. **Flow Orientation**:

    - **Function**: Extract additional inter-frame correlation information from the reference frame to generate oriented temporal context.
    - **Mechanism**: First, warp the reference frame using the decoded optical flow $\hat{v}_t$ to obtain the predicted frame $\check{x}_t$. Then, use SpyNet as a pyramid inter-frame correlation extractor to search for the "oriented optical flow" $\check{v}_t$ between the reference frame and the predicted frame. This oriented optical flow can capture the temporal correlations missed by the estimated/decoded optical flow under bitrate constraints. Finally, align the reference frame using the oriented optical flow and extract the oriented temporal context $\check{C}_t^0$.
    - **Design Motivation**: Estimated and decoded optical flows are constrained by RD trade-offs (rate-distortion trade-offs), limiting their representation capability. Oriented optical flow does not need to be transmitted (adding no extra bit overhead), thus allowing for more thorough exploitation of inter-frame correlations.

2. **Context Compensation**:

    - **Function**: Effectively fuse the two types of temporal contexts from different reference sources to remove irrelevant information from the propagated context.
    - **Mechanism**: A global-local compensation network is adopted. The two contexts first pass through a shared shallow feature extractor, and are then respectively routed through a global extractor and a local extractor to obtain global features (structure/background) and local features (texture/edges). Corresponding global/local features are added and fed into a fusion network, and finally output the compensated context $\bar{C}_t^0$ through a shared fusion layer. The local extractor and fusion network employ an Invertible Neural Network (INN) coupled with affine decoupling layers to preserve detailed information of both contexts as much as possible.
    - **Design Motivation**: The two contexts originate from similar sources (reference frames and propagated features); global features exhibit more similarities, while local features exhibit more differences. Thus, they need to be processed separately to achieve better complementarity.

3. **Decoupling Loss**:

    - **Function**: Constrain the division of labor between global and local features during training to facilitate collaborative complementarity between the two contexts.
    - **Mechanism**: $L_{decouple} = \frac{Cor(\check{L}_t^0, L_t^0)^2}{Cor(\check{G}_t^0, G_t^0)^2 + \delta}$, where $Cor$ denotes cosine similarity. This loss encourages global features to be more correlated (increasing the denominator) and local features to be less correlated (decreasing the numerator), thereby forcing the global branch to focus on commonalities (structure/background) and the local branch to focus on differences (texture/edges).
    - **Design Motivation**: Without explicit constraints, it is difficult for the network to automatically learn a reasonable global-local division of labor. Visualization results confirm that this loss indeed guides the global features to focus more on the background and the local features to focus more on textures.

### Loss & Training
The total training loss is $L = \lambda \cdot D + \alpha \cdot L_{decouple} + R$, where $D$ represents the MSE distortion, $R$ represents the coding rate, $\lambda$ controls the distortion weight (values: 85/170/380/840), and $\alpha$ is set to 0.2. A hierarchical quality structure is adopted, in which the $\lambda$ weight is adjusted periodically. Training consists of two stages: first training on the Vimeo-90k 7-frame sequences, followed by a 32-frame cascaded training on 9000 sequences from the original Vimeo videos (using $256 \times 384$ patches), utilizing gradient checkpointing (FRB) to alleviate GPU memory pressure.

## Key Experimental Results

### Main Results

| Dataset | Metric | DCMVC | DCVC-FM | vs VTM |
|--------|------|-------|---------|--------|
| UVG | BD-Rate (IP=32) | -30.6% | -20.4% | Saves 30.6% |
| MCL-JCV | BD-Rate (IP=32) | -17.3% | -8.1% | Saves 17.3% |
| HEVC B | BD-Rate (IP=32) | -14.5% | -10.3% | Saves 14.5% |
| Average (IP=32) | BD-Rate | -19.4% | -9.9% | Saves 19.4% |
| Average (IP=-1) | BD-Rate | -22.7% | -12.6% | Saves 22.7% |

### Ablation Study

| Configuration | BD-Rate Change | Description |
|------|------------|------|
| Ma (Baseline DCVC-DC) | 0.0% | Baseline |
| Mb (+ Flow Orientation) | -1.9% | Flow Orientation only (direct concatenation) |
| Md (+ FO + CC) | -4.4% | Flow Orientation + Context Compensation |
| Me (+ FO + CC + Decouple) | -5.4% | Decoupling loss saves an additional 1.0% |
| Mf (+ Long-seq Training) | -4.3% | Long sequence training (32 frames) only |
| Mg (All combined) | -10.3% | Combination of all methods |

### Key Findings
- The collaborative effect of the flow orientation and context compensation modules (-4.4%) is superior to using them individually (-1.9% and -3.5%), indicating that oriented optical flow indeed provides better input for context compensation.
- The decoupling loss yields an additional 1.0% bitrate savings without increasing model complexity, while also enhancing model interpretability.
- Long-sequence training (32 frames) makes a significant contribution (-4.3%), and its effect compounds when combined with the proposed method (totaling -10.3%).
- The advantage is even more pronounced under a long prediction chain (IP=-1): it saves 10.1% bitrate compared to DCVC-FM, confirming that context modulation effectively alleviates error propagation.

## Highlights & Insights
- **Oriented optical flow requires no transmission**: By estimating the oriented optical flow between the reference frame and the predicted frame at the decoder side, additional temporal information is obtained without increasing bitrate overhead. This is a clever "free lunch" design that leverages the advantages of end-to-end NVC training.
- **Feature fusion concept with global-local decoupling**: Extracting and then fusing temporal contexts from two sources separately based on global (structure) and local (texture) properties is a clear approach that can be transferred to other multi-source feature fusion scenarios.
- **Invertible neural networks for local feature processing**: Utilizing an INN with affine decoupling layers to preserve detailed information is a relatively novel practice in the field of video compression.

## Limitations & Future Work
- High computational complexity: The MACs reach 4131G. The encoding time of 932ms and decoding time of 810ms are both higher than those of DCVC-FM, leaving a significant gap to real-time applications.
- Modulation is only performed on the largest-scale context, without exploring the possibility of unified multi-scale modulation.
- Performance on the USTC-TD dataset is subpar (saving only 1.9% compared to VTM at IP=-1), which may be related to the characteristics of this dataset; robustness needs to be enhanced.
- The oriented optical flow uses a fixed SpyNet, without exploring more advanced optical flow estimation methods or learnable correlation extractors.

## Related Work & Insights
- **vs DCVC-FM**: DCVC-FM uses a fixed period to switch between reference frames and propagated features to mitigate error propagation. In contrast, this paper utilizes both reference sources simultaneously for each frame and fuses them through learning, offering more flexibility and thoroughness.
- **vs DCVC-DC**: DCVC-DC relies solely on propagated reference features to generate context. This paper adds oriented context from the pixel domain on top of this, providing stronger complementarity.
- **vs SDD**: SDD has more parameters (21.77M vs 20.98M) and comparable MACs (3830G vs 4131G), but its compression performance is far inferior to the proposed method, demonstrating that the architectural design of this paper is more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of the two-step modulation framework (flow orientation + context compensation) is novel, and the decoupling loss is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple datasets with various settings, comprehensive ablation studies, and qualitative visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, abundant diagrams/tables, and detailed description of the methods.
- Value: ⭐⭐⭐⭐ Achieves significant performance breakthroughs in the neural video compression field, and the methodological insights are transferable.

---
title: >-
  [Paper Notes] Neural Video Compression with Context Modulation
description: >-
  [CVPR 2025][Video Compression][Context Modulation] Proposes a context modulation mechanism to enhance the temporal context utilization capability of neural video codecs
tags:
  - CVPR 2025
  - Neural Video Compression
  - Context Modulation
  - Temporal Redundancy
  - Conditional Coding
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Continuous Space-Time Video Resampling with Invertible Motion Steganography](continuous_space-time_video_resampling_with_invertible_motion_steganography.md)
- [\[ICLR 2026\] Synchronizing Probabilities in Model-Driven Lossless Compression](../../ICLR2026/signal_comm/synchronizing_probabilities_in_model-driven_lossless_compression.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ICLR 2026\] Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametic Invertible Transformation](../../ICLR2026/signal_comm/advancing_spatiotemporal_representations_in_spiking_neural_networks_via_parametr.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)

</div>

<!-- RELATED:END -->
