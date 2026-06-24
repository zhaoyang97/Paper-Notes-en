---
title: >-
  [Paper Note] SpikeStereoNet: A Brain-Inspired Stereo Depth Estimation Framework for Spike Streams
description: >-
  [ICLR 2026][3D Vision][Spike Camera] This paper proposes SpikeStereoNet, which estimates stereo depth directly from a pair of raw spike streams (binary high-frequency streams from spike cameras). It employs a three-layer Recurrent Spiking Neural Network (RSNN) as an iterative refinement operator to repeatedly update disparity. Accompanied by large-scale synthetic and real spike stereo datasets, the method outperforms existing frame-based and event-based stereo matching method…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Spike Camera"
  - "Stereo Depth Estimation"
  - "Spiking Neural Network"
  - "Iterative Refinement"
  - "Neuromorphic Computing"
date: 2026-05-08
content_hash: 5c7d7f6d228d084b
---

# SpikeStereoNet: A Brain-Inspired Stereo Depth Estimation Framework for Spike Streams

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lPMPFeioCZ](https://openreview.net/forum?id=lPMPFeioCZ)  
**Code**: https://github.com/Criticality-Cognitive-Computation-Lab/SpikeStereoNet  
**Area**: 3D Vision  
**Keywords**: Spike Camera, Stereo Depth Estimation, Spiking Neural Network, Iterative Refinement, Neuromorphic Computing

## TL;DR
This paper proposes SpikeStereoNet, which estimates stereo depth directly from a pair of raw spike streams (binary high-frequency streams from spike cameras). It employs a three-layer Recurrent Spiking Neural Network (RSNN) as an iterative refinement operator to repeatedly update disparity. Accompanied by large-scale synthetic and real spike stereo datasets, the method outperforms existing frame-based and event-based stereo matching methods on both datasets while maintaining high accuracy with only 10% of the training data.

## Background & Motivation
**Background**: Stereo depth estimation has long been established on conventional frame-based cameras. Mainstream approaches prioritize constructing a cost volume followed by 3D CNN regularization, or utilizing iterative refinement via recurrent units (ConvGRU), such as RAFT-Stereo. These methods perform well in static and slow-moving scenes.

**Limitations of Prior Work**: Frame-based cameras suffer from motion blur and latency in high-speed, dynamic scenes, leading to blurred depth maps. Bio-inspired **spike cameras** can asynchronously emit binary pulses at temporal resolutions up to 40,000 Hz to capture rich luminance information, making them ideal for extreme scenarios. However, there are currently **no dedicated stereo algorithms for spike streams and no associated benchmarks**. Converting spike streams into frames for existing methods introduces temporal quantization errors, motion blur, and significant computational overhead, negating the sensor's inherent advantages.

**Key Challenge**: Spike cameras output **asynchronous, binary, high-throughput** spike streams, whereas frame-based algorithms expect **synchronous, intensity-valued** image pairs—the data modalities are fundamentally mismatched. While stereo methods for event cameras exist, they utilize differential sampling (recording changes in brightness), whereas spike cameras use integral sampling (accumulating photons until a threshold is reached), meaning their processing methods cannot be directly adopted.

**Goal**: (1) Design an end-to-end stereo depth network that consumes raw spike streams directly without frame conversion; (2) Provide the missing data benchmarks for this new research direction.

**Key Insight**: The authors noted that key properties of biological neurons (firing threshold, resting potential, membrane time constant) are not fixed but change dynamically with neuronal states and context. Integrating such "adaptive" mechanisms into an iterative refinement operator may both match the temporal characteristics of spike data and stabilize the convergence of the iteration process.

**Core Idea**: Replace the ConvGRU in RAFT-Stereo with an **Adaptive Recurrent Spiking Neural Network (ALIF-RSNN)** as the iterative disparity update operator, allowing the network to refine depth progressively within the spike domain.

## Method

### Overall Architecture
SpikeStereoNet receives raw spike streams $S_l, S_r \in \{0,1\}^{N\times H\times W}$ from two spike cameras and outputs a full-resolution depth map. The overall framework follows the RAFT-Stereo paradigm of "Feature Extraction → Correlation Pyramid Construction → Iterative Disparity Refinement → Upsampling," but replaces the core iterative update operator with a brain-inspired RSNN. A specialized network extracts multi-scale features from spike streams at the front end. Training involves two steps: supervised pre-training on synthetic data followed by domain adaptation fine-tuning for real spike data.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Left/Right Raw Spike Streams<br/>S ∈ {0,1}^(N×H×W)"] --> B["Spike Feature Extraction<br/>Multi-scale + Context Features"]
    B --> C["Correlation Pyramid<br/>4-level 1D Pooled Cost Volume"]
    C --> D["RSNN Iterative Update Operator<br/>ALIF Neurons Refine Disparity"]
    D -->|Residual Update d_t = d_(t-1)+Δd_t| D
    D --> E["Convex Combination Upsampling<br/>1/4 → Full Resolution"]
    E --> F["Disparity → Depth"]
```

### Key Designs

**1. Multi-scale Feature Extraction from Raw Spike Streams: Converting binary spikes into matchable dense features**

The output of spike cameras is a high-frequency sequence of binary frames; direct stereo matching is both sparse and noisy. This paper uses two feature networks and one context network for processing. The feature networks receive the left and right spike streams $S_{l(r)}\in\mathbb{R}^{N\times H\times W}$, first reducing them to half-resolution via $7\times7$ convolutions, followed by residual blocks to extract features and downsample to 1/4 resolution. This further generates features at three scales: 1/4, 1/8, and 1/16 $\{f_{l,i}, f_{r,i}\}$. The 1/4 scale features $f_{l,4}, f_{r,4}$ are used to construct the cost volume, while all scales guide the 3D regularization network. The context network shares a similar backbone to produce multi-scale context features, which initialize and inject context into the RSNN at each iteration. This design converts spike streams into multi-scale dense representations early, providing sufficient spatial detail for subsequent iterations without lossy "frame conversion."

**2. 1D Correlation Pyramid: Compressing the all-pairs cost volume into a multi-level structure along the disparity dimension**

Given left and right feature maps $f_l, f_r$, an all-pairs correlation cost volume $C_{ijk}=\sum_h f_{l,hij}\cdot f_{r,hik}$ is constructed. Then, 1D average pooling is applied along the last dimension (disparity dimension) to obtain a 4-level correlation pyramid $\{C_i\}_{i=1}^4$. Each level downsamples the disparity dimension using a kernel size of 2 and a stride of 2. This follows the RAFT-Stereo approach of restricting 2D optical flow to a 1D disparity dimension: instead of building a full 4D volume, the pyramid provides local-to-global matching cues across scales. During iteration, local costs are queried from the pyramid based on current disparity, retaining a large receptive field while controlling computation.

**3. ALIF-RSNN Iterative Update Operator: Replacing ConvGRU with adaptive spiking neurons for disparity refinement (Core Contribution)**

This is the most critical design of the paper. A three-layer RSNN serves as the update module, with each layer containing intra-layer recurrent connections and inter-layer feedforward connections, forming hierarchical temporal processing. The 1/8 and 1/16 resolution layers receive context features and postsynaptic currents derived from weights applied to spike states of the same/previous layers. The 1/4 resolution layer additionally inputs the current disparity estimate and the local cost volume sampled from the correlation pyramid. Each RSNN unit first calculates three adaptive variables using sigmoid convolutions:

$$\alpha_t=\sigma(\mathrm{Conv}([s_{t-1},x_t],W_\alpha)+c_\alpha),\quad \beta_t=\sigma(\cdots W_\beta\cdots),\quad \gamma_t=\sigma(\cdots W_\gamma\cdots)$$

where $c_\alpha,c_\beta,c_\gamma$ are context embeddings from the context network. These are fed into the proposed Adaptive Leak-Integreate-and-Fire (ALIF) neuron:

$$h_t=\alpha_t\cdot v_{t-1}+(1-\alpha_t)\cdot(W_{rec}s_{t-1}+W_f s_t^{(l-1)}),\quad v_t^{th}=\beta_t\cdot v_{peak}$$
$$s_t=\theta(h_t-v_t^{th}),\quad v_t=h_t-\gamma_t\cdot s_t\cdot v_t^{th}$$

Here $v_t$ is the membrane potential, $v_t^{th}$ is the firing threshold, $\theta(\cdot)$ is the Heaviside step function, and $W_{rec},W_f$ are recurrent/feedforward synaptic weights implemented via convolution kernels. The three adaptive variables $\alpha,\beta,\gamma\in[0,1]$ control **membrane potential retention, firing threshold, and soft reset strength**, respectively. This maps the biological fact that neuronal properties change dynamically with state and context into the model. compared to standard LIF neurons with fixed parameters, adaptivity allows neurons to regulate dynamically based on the temporal characteristics of spike data, better matching asynchronous spike streams. The highest resolution RSNN layer outputs the residual disparity, updating $d_t=d_{t-1}+\Delta d_t$ at each step. The authors also analyze network dynamics: hidden state differences decrease over time (convergence), eigenvalues of the weight Jacobian lie within the unit circle (stability), and PCA shows hidden states diverge over time (expressive power)—providing a dynamical explanation for the effectiveness of iterative refinement.

**4. Convex Combination Upsampling: Recovering full-resolution depth from 1/4 resolution**

Iterations are performed at 1/4 resolution to save computation. Finally, based on the hidden states of the last RSNN layer, two convolutional layers predict the residual disparity to update the current disparity. A convex combination strategy is then used to upsample the 1/4 disparity to full resolution, which is eventually converted to depth. This step ensures that the precision of iterative refinement is not lost during upsampling, resulting in dense depth maps with sharp boundaries.

### Loss & Training
The total loss consists of three terms:

$$L=L_{stereo}+\lambda_f L_{rate\_reg}+\lambda_v L_{v\_reg}=\sum_{t=1}^T \eta^{T-t}\|d_{gt}-d_t\|_1+\lambda_f\sum_{i=1}^N(r_i-r_0)^2+\lambda_v\sum_{i=1}^N\sum_{t=1}^T v_i(t)^2,\quad \eta=0.9$$

The main loss $L_{stereo}$ is the L1 distance between the predicted disparity $\{d_t\}$ at all iteration steps and the ground truth $d_{gt}$, with weights increasing per step (heavier weighting for later steps). The second term is a firing rate regularization that forces the average firing rate $r_i$ of the $i$-th neuron toward a target $r_0$, promoting temporal sparsity. The third term is a voltage regularization constraining membrane potential magnitude. Both regularization terms encourage sparsity and improve performance. Training uses AdamW with a one-cycle learning rate (initial $2\times10^{-4}$), gradient clipping to $[-1,1]$, 16 iterations per sample, batch size 8, for 300k steps, complemented by random horizontal/vertical flip augmentation.

## Key Experimental Results

### Main Results
On the synthetic dataset, where all methods take spike streams as input, SpikeStereoNet achieves the best performance across almost all metrics:

| Dataset | Method | bad 2.0 (%)↓ | bad 3.0 (%)↓ | AvgErr (px)↓ | FLOPs (B) |
|--------|------|------|------|------|------|
| Synthetic | RAFT-Stereo | 4.64 | 2.76 | 0.48 | 798 |
| Synthetic | Selective-Stereo | 4.57 | 2.66 | 0.45 | 957 |
| Synthetic | MonSter | 4.64 | 2.72 | 0.46 | 1567 |
| Synthetic | **Ours** | **4.13** | **2.38** | **0.42** | **473** |

On the real spike dataset (pre-trained on synthetic then fine-tuned via domain adaptation), it also leads:

| Dataset | Method | bad 2.0 (%)↓ | bad 3.0 (%)↓ | AvgErr↓ |
|--------|------|------|------|------|
| Real | DLNR | 5.64 | 3.38 | 0.61 |
| Real | Selective-Stereo | 5.50 | 3.43 | 0.58 |
| Real | **Ours** | **5.33** | **3.19** | **0.56** |

Notably, the FLOPs of SpikeStereoNet (473B) are among the lowest in the table, and the parameter count (12.15M) is superior to most models, achieving a good balance between performance and efficiency.

### Ablation Study
Ablation of network structure and regularization (Synthetic data, lower AvgErr is better):

| Configuration | bad 2.0 (%)↓ | AvgErr (px)↓ | Description |
|------|------|------|------|
| Ours (full) | 4.13 | 0.42 | Full model |
| w/o RC & FFC | 12.07 | 1.29 | Removing both recurrent and feedforward connections; worst performance |
| w/o RC | 11.05 | 0.83 | Removing intra-layer recurrent connections |
| w/o FFC | 5.86 | 0.68 | Removing inter-layer feedforward connections |
| w/o GN module | 7.49 | 0.58 | Removing group norm on adaptive variables |
| w/o regularization | 7.77 | 0.61 | Removing all regularization terms |

Ablation of update operator replacement (same neuron scale):

| Update Operator | bad 2.0 (%)↓ | AvgErr (px)↓ | Description |
|------|------|------|------|
| Vanilla RNN | 7.28 | 0.66 | Standard RNN |
| GRU | 4.53 | 0.48 | Unit used by RAFT-Stereo |
| LSTM | 4.77 | 0.49 | |
| Raw SNN | 11.05 | 0.83 | Naive spiking network |
| LIF (Fixed α,β,γ) | 7.05 | 0.69 | Non-adaptive spiking neuron |
| **ALIF RSNN (Ours)** | **4.13** | **0.42** | Adaptive spiking neuron |

### Key Findings
- **Recurrent + Feedforward connections are the skeleton**: Removing both RC and FFC caused AvgErr to jump from 0.42 to 1.29, the most significant performance drop. This indicates that information flow across time and layers is crucial for capturing spike data dynamics; recurrent connections (RC) are more critical than feedforward (FFC).
- **Adaptivity is the core of the spiking solution's success**: Replacing ALIF with fixed-parameter LIF saw AvgErr regress from 0.42 to 0.69, which is even worse than the standard GRU (0.48). Only with the introduction of adaptive thresholds/resets did the ALIF-RSNN outperform the GRU, proving that "dynamically changing neuronal properties" is the key advantage of spiking solutions over classic recurrent units.
- **Outstanding Data Efficiency**: When using only 10%–50% of training data, Ours outperforms RAFT-Stereo / Selective-Stereo across all ratios, with the gap widening as data decreases, showing stronger generalization in data-scarce scenarios.
- **Stable Convergence of Iteration**: Hidden state differences decreased over iterations, and weight eigenvalues remained within the unit circle, providing a dynamical explanation for why iterative refinement works.

## Highlights & Insights
- **Implementation of "Biological Neuron Adaptivity" as a trainable ALIF operator**: $\alpha, \beta, \gamma$ correspond to membrane potential retention, firing threshold, and soft reset, and are generated dynamically from context rather than being fixed constants. This is a clean mapping from neuroscientific facts to an engineering module, proven by ablation to be the key to beating GRUs.
- **The Data Benchmark is a significant contribution**: Synthetic spike streams were created using Blender + video frame interpolation (50 frames between adjacent pairs) + a brain-inspired spike generator. Additionally, dual spike cameras + Kinect were used to collect 3000+ real stereo spike pairs (300k+ frames), filling the gap of missing standard datasets in this field.
- **Transferrable Concepts**: The paradigm of replacing recurrent units in frame-based iterative refinement frameworks with adaptive spiking units can be transferred to other low-level vision tasks requiring temporal refinement, such as optical flow or event-based stereo.

## Limitations & Future Work
- **Limited Real Data Scale and Diversity**: The real dataset was collected only indoors with a resolution of $400\times250$. Generalization in outdoor, long-range, or complex lighting scenarios has not been fully verified.
- **Dependency on Domain Adaptation**: Performance on real data requires synthetic pre-training followed by fine-tuning; the gap between synthetic and real is still bridged by domain adaptation. The effectiveness of purely end-to-end real training is unknown.
- **Efficiency advantage not yet realized at the Neuromorphic Hardware level**: While FLOPs are lower, models were trained and evaluated on an RTX 4090 using PyTorch. Real power consumption/latency gains on neuromorphic chips have not yet been demonstrated.
- **Future Directions**: Exploring larger scale real-world spike data, interpretability analysis of adaptive variables, and deploying ALIF-RSNN to neuromorphic hardware to realize the low-power promises of spiking computation.

## Related Work & Insights
- **vs RAFT-Stereo**: Both use iterative refinement and correlation pyramids. However, RAFT-Stereo uses ConvGRU for frame inputs, while Ours uses ALIF-RSNN for raw spike streams. Ablations show ALIF (0.42) outperforms GRU (0.48) with better data efficiency.
- **vs Fixed-parameter LIF / Raw SNN**: Naive spiking networks (Raw SNN, 0.83) and fixed LIF (0.69) perform significantly worse than Ours, indicating the key to integrating spiking networks into stereo matching is "adaptivity" rather than just "using spikes."
- **vs Event-based Stereo (e.g., ZEST)**: Event cameras use differential sampling for brightness changes. Ours is designed for the integral sampling of spike cameras and achieves an AvgErr of 0.42 on synthetic sets, significantly better than ZEST’s 0.62.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first brain-inspired framework for stereo depth directly from raw spike streams; the ALIF-RSNN update operator design is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both synthetic and real datasets, extensive ablation, data efficiency, and dynamics analysis. Real dataset scale is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ Method and dynamics analysis are clear; some module details (e.g., GN, context injection) are relatively brief.
- Value: ⭐⭐⭐⭐⭐ Contributes both an algorithm and a scarce spike stereo benchmark, significantly advancing the neuromorphic vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Nope-SGS: 3D Gaussian Reconstruction from Unposed Spike Streams](../../CVPR2026/3d_vision/3d_gaussian_splatting_from_unposed_spike_stream.md)
- [\[CVPR 2026\] SCE-Depth: A Spherical Compound Eye Framework for Wide FOV Depth Estimation](../../CVPR2026/3d_vision/sce-depth_a_spherical_compound_eye_framework_for_wide_fov_depth_estimation.md)
- [\[ICLR 2026\] StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams](streamsplat_towards_online_dynamic_3d_reconstruction_from_uncalibrated_video_str.md)
- [\[CVPR 2025\] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](../../CVPR2025/3d_vision/efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)
- [\[ICLR 2026\] PatchRefiner V2: Fast and Lightweight Real-Domain High-Resolution Metric Depth Estimation](patchrefiner_v2_fast_and_lightweight_real-domain_high-resolution_metric_depth_es.md)

</div>

<!-- RELATED:END -->
