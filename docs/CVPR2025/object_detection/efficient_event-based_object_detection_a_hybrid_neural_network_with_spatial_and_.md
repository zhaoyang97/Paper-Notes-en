---
title: >-
  [Paper Note] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention
description: >-
  [CVPR 2025][Object Detection][Event Camera] This work proposes the first hybrid SNN-ANN object detection model targeting large-scale benchmarks. An Attention-Squeeze Bridging (ASAB) block is designed to convert sparse spike representations from the SNN into dense features for the ANN via spatio-temporal attention. With only 6.6M parameters, it significantly outperforms SNN methods and approaches the accuracy of ANN/RNN methods on the Gen1/Gen4 datasets…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Event Camera"
  - "Hybrid SNN-ANN"
  - "Attention Bridging"
  - "Neuromorphic Hardware"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 23ced190b8ee0e09
---

# Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention

**Conference**: CVPR 2025  
**arXiv**: [2403.10173](https://arxiv.org/abs/2403.10173)  
**Code**: None  
**Area**: Object Detection / Event Vision  
**Keywords**: Event Camera, Hybrid SNN-ANN, Attention Bridging, Neuromorphic Hardware, Efficient Inference

## TL;DR

This work proposes the first hybrid SNN-ANN object detection model targeting large-scale benchmarks. An Attention-Squeeze Bridging (ASAB) block is designed to convert sparse spike representations from the SNN into dense features for the ANN via spatio-temporal attention. With only 6.6M parameters, it significantly outperforms SNN methods and approaches the accuracy of ANN/RNN methods on the Gen1/Gen4 datasets, while the SNN component can be deployed on the Intel Loihi 2 neuromorphic chip for low-power inference.

## Background & Motivation

**Background**: Event cameras asynchronously capture pixel-level illumination changes with a temporal resolution of ~10μs, offering advantages such as low latency, high dynamic range (140 dB), and no motion blur, which are ideal for fast-motion and low-light scenarios. Object detection is a crucial application of event vision. Currently, there are three main paradigms: (1) ANN-based methods directly reuse traditional detection architectures but suffer from large model sizes, which are unfriendly for edge deployment; (2) SNN-based methods naturally match the sparsity of event data and run with low power on neuromorphic hardware, but their accuracy lags significantly behind ANNs; (3) RNN-based methods excel in temporal modeling but are computationally heavy.

**Limitations of Prior Work**: ANN-based methods (>60M parameters) are too colossal to be deployed on edge or neuromorphic devices. SNN-based methods experience severe degradation in accuracy (mAP of only 0.15-0.31) when directly converting ANN backbones (e.g., VGG, ResNet) to SNNs. Existing hybrid SNN-ANN approaches have only been verified on simple tasks (classification, tracking) instead of large-scale detection benchmarks, and their SNN-to-ANN bridging designs are overly simplistic (e.g., direct accumulation), which discards critical spatio-temporal information.

**Key Challenge**: A representation gap exists between the sparse spike representations of SNNs and the dense feature maps required by ANNs. Naive accumulation or averaging operations lead to severe loss of spatio-temporal relationships in event data.

**Goal**: How to design an efficient SNN-to-ANN bridging mechanism that converts sparse spikes into dense representations while preserving the spatio-temporal information of events, thereby enabling compact hybrid networks to achieve near-ANN accuracy on large-scale detection benchmarks?

**Key Insight**: It is observed that the sparse spatial distribution of event data is irregular (well-suited for deformable convolutions), and temporal relationships between spikes at different timesteps require explicit modeling (well-suited for attention mechanisms). Furthermore, the event firing rate itself can serve as a guiding signal for spatial attention, as high event-rate regions typically correspond to moving objects and should be prioritized.

**Core Idea**: Design a bridging module containing Spatio-Temporal Aware Temporal Attention (SAT) and Event-Rate Spatial Attention (ERS) to effectively convert SNN spikes into ANN dense features, achieving high accuracy with a small model footprint.

## Method

### Overall Architecture

The hybrid backbone network consists of three parts: (1) SNN module $f_{snn}$: composed of multiple Conv-BN-PLIF blocks to process the event tensor with high temporal resolution (5ms bins), outputting sparse spike features $\mathbf{E}_{spike} \in \mathbb{R}^{T \times C \times H' \times W'}$; (2) ASAB bridging module $\beta_{asab}$: converts spikes into dense feature maps $\mathbf{F}_{out} \in \mathbb{R}^{C \times H' \times W'}$ via SAT and ERS attention; (3) ANN module $f_{ann}$: standard convolutional blocks to extract high-level spatial features, which are then fed into a YOLOX detection head. A DWConvLSTM can be optionally integrated to form a multi-time-scale RNN variant.

### Key Designs

1. **Spatio-Temporal Aware Temporal Attention (SAT)**:

    - **Function**: Captures temporal relationships from sparse spikes and transforms them into spatial features.
    - **Mechanism**: It first performs Channel-wise Temporal Grouping, rearranging the $T \times C$ dimensions to $C \times T$ for channel-wise processing. Then, Temporal-Separated Deformable Convolution (TSDC) is employed to independently extract the local spatial context at each timestep. Since spikes possess irregular spatial distributions that are difficult for fixed-grid convolution kernels to capture, deformable convolution adaptively adjusts sampling positions by learning offsets. Finally, a softmax temporal attention computes correlation weights across different timesteps: $\mathbf{A}_{score} = \text{softmax}(\mathbf{A}_q \mathbf{A}_k)$, and a weighted sum is aggregated along the temporal dimension using a 1x1 convolution to yield a single-frame output.
    - **Design Motivation**: Deformable convolution matches the irregular spatial structure of spikes; temporal-separated processing preserves the independence of spatial information across timesteps; temporal attention explicitly models the correlation strength between timesteps.

2. **Event-Rate Spatial Attention (ERS)**:

    - **Function**: Utilizes the event firing rate as spatial attention weights to highlight active motion regions.
    - **Mechanism**: Composing the event rate map $\mathbf{S}_{rate} \in \mathbb{R}^{C \times H' \times W'}$ by summing the spike tensor $\mathbf{E}_{spike}$ along the temporal dimension, which is then normalized via sigmoid and applied as a Hadamard product with the SAT output: $\mathbf{E}_{feature} = \text{sigmoid}(\mathbf{S}_{rate}) \odot \mathbf{A}_{out}$. Regions with high event rates (typically corresponding to moving object boundaries) receive larger weights.
    - **Design Motivation**: The core characteristic of event cameras is "motion-triggered." The event rate directly reflects motion information in the scene, serving as an inherent spatial saliency signal. Utilizing it as attention weights aligns with the physical characteristics of event data.

3. **Multi-Time-Scale RNN Variant**:

    - **Function**: Integrates DWConvLSTM into the ANN component to capture slow dynamics.
    - **Mechanism**: The SNN processes fast dynamics at a short timestep of 5ms, while DWConvLSTM handles long-term temporal dependencies on a larger timestep (50ms) output by the ASAB. Depthwise separable convolutional LSTM reduces parameter size.
    - **Design Motivation**: Autonomous driving scenarios involve both fast motion (sudden pedestrians) and slow variations (distant slow vehicles); multi-time-scale modeling is able to cover both types of dynamics simultaneously.

### Loss & Training

The detection head utilizes the YOLOX framework, with the loss containing IoU loss, classification loss, and regression loss. On the Gen1 dataset, training with a batch size of 24 and lr of 2e-4 on 4 RTX 3090 GPUs takes approximately 8 hours (50 epochs). The RNN variant uses a sequence length of 21 and is trained for 400 thousand steps, taking about 6 days. The SNN component is trained end-to-end jointly using the surrogate gradient method.

## Key Experimental Results

### Main Results

Comparison of mAP on Gen1 and Gen4 autonomous driving detection datasets:

| Method | Type | Parameters | Gen1 mAP | Gen4 mAP |
|------|------|--------|----------|----------|
| EMS-RES34 (Best SNN) | SNN | 14.4M | 0.31 | - |
| Events-RetinaNet | ANN | 33M | 0.34 | 0.18 |
| RVT-B (w/o LSTM) | Transformer | 16.2M | 0.32 | - |
| **Proposed (Hybrid)** | **Hybrid** | **6.6M** | **0.35** | **0.27** |
| RVT-B | TF+RNN | 19M | 0.47 | - |
| **Proposed+RNN** | **Hybrid+RNN** | **7.7M** | **0.43** | - |

The hybrid model achieves the best accuracy among non-RNN methods with only 6.6M parameters; with RNN added, it approaches the 19M RVT-B with only 7.7M parameters.

### Ablation Study

| Configuration | mAP(.5) | mAP | Description |
|------|---------|-----|------|
| w/o ASAB (Naive Accumulation) | 0.53 | 0.30 | Naive bridging leads to severe information loss |
| w/o Temporal Attention Φ_ta | 0.57 | 0.33 | Lack of temporal modeling |
| w/o Deformable Convolution | 0.59 | 0.34 | Replaced with standard convolution |
| w/o ERS Spatial Attention | 0.59 | 0.34 | Loss of spatial saliency signal |
| **Full model** | **0.61** | **0.35** | All components complete |

### Key Findings

- Importance of the ASAB module: Replacing ASAB with naive accumulation drops the mAP from 0.35 to 0.30 (-14%), demonstrating that a carefully designed bridge is crucial for hybrid networks.
- Effectiveness of deformable convolution for spatial modeling of sparse spikes: Standard convolution on fixed grids fails to match the irregular distribution of spikes.
- The SNN component deployed on the Intel Loihi 2 consumes only 1.73W of power, with almost lossless accuracy after int8 quantization (0.348 -> 0.343) and an execution time of 2.06ms per step.
- Only <5% of the hybrid model's MACs come from attention and MLP (vs 67% in RVT), making it more suitable for edge/neuromorphic deployment.

## Highlights & Insights

- **Design philosophy of the ASAB bridging module**: Instead of simply accumulating SNN spikes into frames, it utilizes attention mechanisms to let the network learn "which timesteps and spatial locations of spikes are more important," achieving a balance between information preservation and representation conversion. This bridging concept can be generalized to any SNN-ANN hybrid architecture.
- **Event rate as an attention signal**: Directly using the intrinsic physical quantity of the event camera (event rate = motion intensity) for spatial attention incurs zero parameter overhead. This approach of leveraging sensor properties for attention is highly elegant.
- Practical deployment verification (Intel Loihi 2 hardware experiment) provides credible evidence for the real-world deployment of hybrid networks, which is rare in academic papers but extremely valuable.

## Limitations & Future Work

- There is still a 4 percentage point accuracy gap between the RNN variant and pure RNN methods (such as RVT-B's 0.47), suggesting that information loss after bridging might be a bottleneck.
- Currently verified only on automotive detection datasets; generalization to more categories remains to be explored.
- The SNN part has only 2-3 shallow convolutional layers. Whether deeper SNNs can extract better lower-level features remains a question; however, this would also increase deployment complexity on neuromorphic hardware.
- The complexity of the temporal attention in ASAB is $O(T^2)$, which could become a bottleneck as the number of event timesteps increases.

## Related Work & Insights

- **vs EMS-YOLO (EMS-RES)**: EMS converts ResNet to an SNN for detection but achieves limited accuracy (0.31). Ours achieves 0.35 with fewer parameters, indicating that SNNs do not need to do all the work alone.
- **vs RVT**: RVT achieves high accuracy using a Transformer+RNN but requires 19M parameters, with 67% of MACs coming from attention/MLP. Ours, with 6.6M parameters and <5% attention MACs, is much more suitable for hardware deployment.
- **vs DashNet**: DashNet also implements a hybrid SNN-ANN but targets simpler tasks like tracking. Ours validates the feasibility of the hybrid scheme on large-scale detection benchmarks for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark-oriented hybrid detection scheme + elegantly designed ASAB
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across three paradigms + hardware deployment verification + thorough ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure with persuasive ablation analysis
- Value: ⭐⭐⭐⭐ Provides a viable path for the efficient deployment of event-based object detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[CVPR 2025\] Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[ICML 2025\] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](../../ICML2025/object_detection/when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)
- [\[CVPR 2026\] Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection](../../CVPR2026/object_detection/beyond_duality_a_hybrid_framework_of_leveraging_shared_and_private_features_for_.md)

</div>

<!-- RELATED:END -->
