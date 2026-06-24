---
title: >-
  [Paper Note] ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration
description: >-
  [ECCV 2024][Model Compression][Event-driven inference] This work proposes ELSE, an event suppression method through line-based sparsity exploration, which utilizes the spatial correlation of adjacent lines in feature maps to reduce the count of non-zero activations (events), achieving $3.14\sim6.49\times$ computational savings on object detection and pose estimation tasks while staying complementary to existing event suppression methods.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Event-driven inference"
  - "line-based sparsity"
  - "activation map compression"
  - "embedded AI"
  - "neuromorphic computing"
date: 2026-05-08
content_hash: e56dba11cf3c14b3
---

# ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Event-driven inference, line-based sparsity, activation map compression, embedded AI, neuromorphic computing

## TL;DR

This work proposes ELSE, an event suppression method through line-based sparsity exploration, which utilizes the spatial correlation of adjacent lines in feature maps to reduce the count of non-zero activations (events), achieving $3.14\sim6.49\times$ computational savings on object detection and pose estimation tasks while staying complementary to existing event suppression methods.

## Background & Motivation

**Background**: Brain-inspired architectures achieve low-power, low-latency deep neural network inference through an event-driven mechanism, making them highly suitable for embedded AI applications. In these architectures, hardware performance directly depends on the number of non-zero activations (i.e., events) during inference—fewer events translate to more efficient computation.

**Limitations of Prior Work**: Existing event suppression methods are mainly categorized into spatial suppression and temporal suppression. Spatial suppression methods reduce the event count per frame directly through thresholding, but ignore the spatial redundancy structures within feature maps. Temporal suppression methods utilize inter-frame differences to process only changing pixels, but require storing the complete state of the previous frame, leading to a substantial state memory footprint that often exceeds the memory limits of resource-constrained embedded platforms.

**Key Challenge**: A large amount of spatial redundancy exists in feature maps—adjacent lines often exhibit high similarity—but existing methods do not systematically exploit this line-level spatial correlation for event compression.

**Goal**: (1) How to significantly reduce the number of events in event-driven inference without substantial accuracy loss? (2) How to complement existing spatial/temporal suppression methods? (3) How to reduce the state memory footprint of temporal suppression methods?

**Key Insight**: It is observed that natural spatial correlation exists between adjacent lines in feature maps of convolutional networks, which is determined by the local receptive fields of convolutional operations and the spatial continuity of images. If the activation values of two adjacent lines are highly similar, the computation of the redundant line can be bypassed and substituted with the result of the reference line.

**Core Idea**: The redundant events are suppressed via line-level sparsity exploration by exploiting spatial correlation between adjacent lines in feature maps, thereby significantly reducing the computational workload of event-driven inference.

## Method

### Overall Architecture

The core mechanism of ELSE is to introduce a line-level event suppression module into the pipeline of event-driven inference. For the feature map of each layer, ELSE scans line-by-line and compares the differences between adjacent lines. If the difference between two lines is below a certain threshold, one of the lines is marked as a bypassable redundant line, thereby reducing the triggered event computation. The input to the entire pipeline is a standard convolutional network model, and the output is a line-sparsity-optimized event stream that can run directly on event-driven hardware accelerators.

### Key Designs

1. **Line-based Spatial Correlation Analysis**:

    - **Function**: Detecting the degree of redundancy between adjacent lines in feature maps.
    - **Mechanism**: For each pair of adjacent lines in the feature map, a difference metric (such as the sum of absolute differences or mean squared error of element-wise differences) is computed. When the difference is below a preset threshold, the two lines are considered highly redundant, and the latter line can be approximated by the values of the former line. This detection is conducted layer-by-layer since the activation characteristics vary across different layers (shallower layers exhibit stronger spatial correlation, while deeper layers exhibit weaker correlation).
    - **Design Motivation**: Convolutional operations naturally introduce local correlation, making adjacent lines highly similar in most cases. Utilizing this prior knowledge can significantly reduce the number of events to be processed with minimal information loss.

2. **Line-level Event Suppression**:

    - **Function**: Deciding which lines of events can be safely suppressed based on the correlation analysis results.
    - **Mechanism**: For lines marked as redundant, ELSE suppresses all their non-zero activations to zero (or copies the values of the reference line), thereby removing the events at these positions from the computation queue. The threshold selection requires a trade-off between compression rate and accuracy—a larger threshold leads to more suppressed lines but potentially larger accuracy loss. The authors determine the optimal threshold ranges for different networks and tasks through experiments.
    - **Design Motivation**: The line-level scale of operation is coarser than pixel-wise operations but finer than channel-wide operations, striking a good balance between compression efficiency and accuracy preservation.

3. **Complementary Combination Strategy**:

    - **Function**: Combining ELSE with existing spatial/temporal suppression methods to achieve greater performance gains.
    - **Mechanism**: ELSE can serve as a plug-and-play module. When used in series with spatial suppression methods, spatial suppression is applied first, followed by line-based sparsity exploration, further reducing the event count synergistically. When combined with temporal suppression methods, since ELSE reduces the number of events per frame, the state information (activation values of the previous frame) that temporal suppression methods need to store is also significantly reduced, thereby lowering the state memory overhead by more than $2\times$.
    - **Design Motivation**: Different event suppression strategies capture redundancy along different dimensions—spatial suppression focuses on magnitude redundancy, temporal suppression focuses on inter-frame redundancy, while ELSE focuses on inter-line structural redundancy. These three strategies are orthogonally complementary.

### Loss & Training

ELSE is an inference-time event suppression method that does not involve extra training procedures or modifications to loss functions. Its key parameter is the threshold for inter-line difference, which can be determined by searching for the optimal accuracy-efficiency trade-off point on the validation set. The threshold needs to be tuned separately for different tasks and network architectures.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | ELSE Event Reduction Ratio | Note |
|:---:|:---:|:---:|:---:|
| Object Detection (Various Architectures) | Event-triggered Computation Reduction | $3.14\sim6.49\times$ | Compared to conventional processing |
| Pose Estimation (Various Architectures) | Event-triggered Computation Reduction | $2.43\sim5.75\times$ | Compared to conventional processing |

### Ablation Study

| Configuration | Key Metric | Note |
|:---:|:---:|:---:|
| ELSE used alone | $3\sim6\times$ event reduction | Only utilizing inter-line spatial correlation |
| ELSE + Spatial Suppression | Significantly enhanced computation savings | Complementary superposition of both methods |
| ELSE + Temporal Suppression | State memory reduction $>2\times$ | Resolving memory constraint issues on embedded platforms |

### Key Findings

- ELSE achieves higher compression ratios on object detection tasks compared to pose estimation tasks ($6.49\times$ vs $5.75\times$), which is related to the higher inter-line correlation in large background regions of object detection.
- The combination of ELSE and spatial suppression methods yields multiplicative compression effects.
- The major contribution when combined with temporal suppression methods lies not in further event reduction, but in substantially lowering state memory requirements, rendering temporal suppression deployable on resource-constrained embedded platforms.
- The compression effect of ELSE varies across different network architectures, indicating that inter-line correlation is related to network structures.

## Highlights & Insights

- Line-level sparsity is an overlooked but highly effective dimension, which complements existing pixel-level and channel-level methods.
- The plug-and-play nature of the method imparts high practical value without requiring model retraining.
- Addressing embedded deployment bottlenecks by reducing the state memory footprint of temporal suppression methods is a highly valuable contribution from an engineering perspective.
- From a brain-inspired computing perspective, line-level sparsity exploration also possesses certain biological plausibility.

## Limitations & Future Work

- The information on the ECVA page is limited, and specific accuracy loss data is incomplete—line-level suppression inevitably introduces some accuracy degradation, so extreme scenarios require closer attention.
- Automatic threshold selection strategies (e.g., adaptive thresholds based on target accuracy constraints) are worth exploring.
- The evaluation is restricted to object detection and pose estimation tasks, leaving actual performance on other vision tasks (such as semantic segmentation and image classification) unknown.
- Whether line-level granularity is optimal requires further analysis—block-based or diagonal spatial correlations could potentially be explored.
- The joint utilization of ELSE with other model compression methods, such as structured pruning, is worthy of investigation.

## Related Work & Insights

- Temporal suppression methods (such as delta networks) reduce events by exploiting inter-frame differences, to which this work is complementary.
- Spatial suppression methods reduce single-frame events through thresholding or quantization.
- The performance of event-driven hardware (such as Loihi and SpiNNaker) directly benefits from sparser event streams.
- Insight: In other computing paradigms (such as attention computation in transformers), does a similar exploitable "inter-line redundancy" exist?

## Rating

- **Novelty**: ⭐⭐⭐⭐ Line-level sparsity is a simple yet effective new perspective, though not a revolutionary innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ It covers object detection and pose estimation tasks, but lacks a broader task coverage and detailed accuracy comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear and the logical chain is complete.
- **Value**: ⭐⭐⭐⭐⭐ Large practical engineering value for embedded AI deployment, and the method is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](../../ACL2025/model_compression/cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)
- [\[ICLR 2026\] Beyond Student: An Asymmetric Network for Neural Network Inheritance](../../ICLR2026/model_compression/beyond_student_an_asymmetric_network_for_neural_network_inheritance.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/model_compression/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](../../ACL2025/model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[CVPR 2026\] Decompose, Mix, Adapt: A Unified Framework for Parameter-Efficient Neural Network Recombination and Compression](../../CVPR2026/model_compression/decompose_mix_adapt_a_unified_framework_for_parameter-efficient_neural_network_r.md)

</div>

<!-- RELATED:END -->
