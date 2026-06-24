---
title: >-
  [Paper Note] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective
description: >-
  [ECCV 2024][Video Understanding][Gait Recognition] This paper revisits the problem of occluded gait recognition from the perspective of action detection, proposing the GaitMoE method. GaitMoE adaptively constructs action anchors through Mixture of Temporal Experts (MTE) and generates action proposals using Mixture of Action Experts (MAE). Trained end-to-end using only ID labels, it effectively handles various occlusion scenarios. Additionally…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Gait Recognition"
  - "Occlusion Handling"
  - "Mixture of Experts"
  - "Action Detection"
  - "Gait Dataset"
date: 2026-05-08
content_hash: 619122989b314b2d
---

# Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective

**Conference**: ECCV 2024  
**Code**: [https://github.com/BNU-IVC/OccGait](https://github.com/BNU-IVC/OccGait)  
**Area**: Video Understanding / Person Identification  
**Keywords**: Gait Recognition, Occlusion Handling, Mixture of Experts, Action Detection, Gait Dataset

## TL;DR
This paper revisits the problem of occluded gait recognition from the perspective of action detection, proposing the GaitMoE method. GaitMoE adaptively constructs action anchors through Mixture of Temporal Experts (MTE) and generates action proposals using Mixture of Action Experts (MAE). Trained end-to-end using only ID labels, it effectively handles various occlusion scenarios. Additionally, the first unified occluded gait dataset, OccGait, is constructed.

## Background & Motivation

**Background**: Gait recognition is a biometric technology that identifies individuals by analyzing their walking patterns, offering unique advantages in long-distance, non-cooperative scenarios. Existing gait recognition methods (such as GaitSet, GaitGL, GaitPart, etc.) primarily achieve good performance in controlled environments by extracting discriminative identity features through spatial-temporal modeling of contours or skeleton features in gait sequences.

**Limitations of Prior Work**: Occlusion in real-world scenarios poses the greatest challenge to gait recognition. Occlusion leads to problems on three levels: (1) **Information loss**—occluded body parts cannot provide effective gait features; (2) **Noise introduction**—the appearance features of occlusions (such as backpacks, umbrellas, other pedestrians) interfere with gait representation; (3) **Alignment failure**—occlusion causes misalignment of body parts in terms of spatial position and scale, rendering part-based methods ineffective. Existing methods typically only consider simple occlusions (such as wearing a coat) and lack a systematic study of diverse occlusion types.

**Key Challenge**: Traditional methods treat gait sequences as sets of static frames (via temporal pooling or simple aggregation), ignoring that gait is inherently a periodic action sequence. Under occlusion, this static perspective fails to exploit temporal continuity and periodicity to recover occluded information. The key observations are: (1) gait continuity between adjacent frames allows inferring information of occluded frames from complete frames; (2) gait periodicity allows information integration between complete and occluded actions.

**Goal**: (1) How to leverage temporal dynamics in gait sequences to handle occlusions; (2) How to adaptively handle different types and degrees of occlusion without occlusion annotations; (3) How to establish a unified benchmark for evaluating occluded gait.

**Key Insight**: The authors analogize gait sequences to action videos, treating a complete gait sequence as a combination of multiple "actions"—each action corresponding to a phase in the gait cycle (e.g., stepping with the left leg, stepping with the right leg, both feet on the ground, etc.). Occlusion essentially destroys the observability of certain "actions." From the perspective of action detection, it is necessary to accurately locate and identify each action phase in the sequence, and then selectively use high-quality action segments for identity recognition.

**Core Idea**: Transform occlusion handling in gait recognition into an action detection problem, using a mixture of experts mechanism to adaptively detect and aggregate gait actions to resist occlusion.

## Method

### Overall Architecture
The input is a sequence of gait silhouettes (temporal binary human contour maps), and the output is a discriminative gait feature vector. GaitMoE consists of two core modules connected in series: first, the Mixture of Temporal Experts (MTE) adaptively constructs "Action Anchors" (key temporal nodes in the gait cycle) in the temporal dimension; then, the Mixture of Action Experts (MAE) generates "Action Proposals" (feature representations of corresponding gait actions) based on these anchors. Finally, features from multiple action proposals are aggregated into an identity feature for recognition. The entire model is trained end-to-end using only ID classification labels, without requiring action or occlusion annotations.

### Key Designs

1. **Mixture of Temporal Experts (MTE)**:

    - **Function**: Adaptively locate key temporal nodes from the gait sequence as action anchors.
    - **Mechanism**: MTE contains multiple "temporal experts," each responsible for paying attention to different temporal intervals of the gait sequence. Specifically, each temporal expert is a temporal convolution module with a different temporal receptive field, and their weights are dynamically assigned by a gating network based on the features of the input sequence. The gating network learns to determine which time segments in the current sequence contain valid (unoccluded) gait information, assigning more weight to these experts. Finally, the outputs of each expert are weighted and combined to form action anchors—representative keyframe features in the gait cycle. Different gait speeds and occlusion patterns activate different expert combinations.
    - **Design Motivation**: Fixed temporal sampling strategies cannot handle diverse occlusion patterns (occlusions can occur at any position in the sequence and last for different durations), whereas the dynamic routing capability of MoE allows the model to adaptively skip occluded temporal segments.

2. **Mixture of Action Experts (MAE)**:

    - **Function**: Generate complete action proposals from action anchors and extract action-level features.
    - **Mechanism**: MAE receives the action anchors output by MTE. Each action expert is responsible for constructing an "action proposal" around an anchor—a compact representation of a gait segment covering a certain temporal range centered at the anchor. Different action experts have different temporal ranges and feature extraction strategies, similar to the design of multi-scale anchors in object detection. The gating network selects the most appropriate expert to handle each anchor based on anchor features. Finally, the generated multiple action proposals are aggregated into a unified gait descriptor. During aggregation, action proposals with higher quality (more complete, less occluded) receive higher weights.
    - **Design Motivation**: Different phases in a gait cycle experience different degrees of occlusion, requiring dynamic adjustment of each action segment's contribution to the final representation based on its quality. MAE avoids a "one-size-fits-all" treatment by processing action segments of different qualities in parallel using multiple experts.

3. **End-to-End Joint Training Strategy**:

    - **Function**: Formulate joint learning of action detection and identity recognition using only ID labels without extra annotations.
    - **Mechanism**: Action detection is trained jointly with gait recognition as a proxy task. The model does not require explicit action annotations—the backpropagation of the ID classification loss gradient naturally drives MTE and MAE to learn meaningful action decomposition: temporal decompositions and action aggregations that help final ID classification are reinforced. This is because, to accurately identify identity under occlusion, the model must learn to find unoccluded and valid segments in the sequence—which is equivalent to learning an implicit "action detection." Standard triplet loss and cross-entropy classification loss are used during training.
    - **Design Motivation**: Explicit action annotation is expensive and ambiguously defined (precise boundaries of gait cycles are hard to annotate). End-to-end training allows the model to automatically learn the most useful action decomposition method for occluded gait recognition.

### Loss & Training
The loss function consists of two parts: (1) cross-entropy classification loss $\mathcal{L}_{CE}$ for identity classification on the aggregated gait features; (2) triplet loss $\mathcal{L}_{triplet}$ to pull features of the same identity closer and push features of different identities further apart. The total loss is $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{triplet}$. During training, data augmentation is used to simulate various occlusion patterns (randomly occluded frames, randomly occluded regions) to enhance the model's robustness to occlusion.

## Key Experimental Results

### Main Results

| Dataset | Metric | GaitMoE (Ours) | GaitBase | GaitGL | GaitPart | Gain |
|--------|------|-------------|----------|--------|----------|------|
| OccGait (Severe Occlusion) | Rank-1 (%) | 72.5 | 61.3 | 58.7 | 55.2 | +11.2 |
| OccCASIA-B | Rank-1 (%) | 84.6 | 78.2 | 75.4 | 72.1 | +6.4 |
| Gait3D | Rank-1 (%) | 68.3 | 63.8 | 60.2 | 57.5 | +4.5 |
| GREW | Rank-1 (%) | 73.8 | 69.5 | 66.1 | 63.2 | +4.3 |
| OccGait (Light Occlusion) | Rank-1 (%) | 88.2 | 82.1 | 79.5 | 76.8 | +6.1 |

### Ablation Study

| Configuration | OccGait Rank-1 | OccCASIA-B Rank-1 | Description |
|------|---------------|-------------------|------|
| Full GaitMoE | 72.5 | 84.6 | Full model |
| w/o MTE | 66.8 | 80.1 | Remove temporal experts, use fixed sampling |
| w/o MAE | 68.2 | 81.5 | Remove action experts, use simple pooling |
| w/o MoE (Single Expert) | 65.3 | 78.9 | Replace all experts with a single network |
| w/o Occlusion Augmentation | 69.1 | 82.0 | No occlusion data augmentation during training |
| Number of Experts = 2 | 69.5 | 82.3 | Fewer experts |
| Number of Experts = 4 (Default) | 72.5 | 84.6 | Default configuration |
| Number of Experts = 8 | 72.8 | 84.7 | Diminishing returns with more experts |

### Key Findings
- The contributions of MTE and MAE are comparable (dropping 5.7% without MTE, and 4.3% without MAE), and they complement each other: MTE is responsible for temporal localization, while MAE extracts action features.
- Removing the entire MoE mechanism (replacing with a single network) causes a 7.2% drop, showing that dynamic selection through multiple experts is crucial for occlusion robustness.
- There is a sweet spot for the number of experts: 4 experts are sufficient, and increasing to 8 yields almost no improvement, suggesting limited diversity in gait actions.
- In severe occlusion scenarios (OccGait severe occlusion subset), GaitMoE shows a larger advantage over the baseline (+11.2% vs +6.1% in light occlusion), validating the method's effectiveness in challenging scenarios.
- The OccGait dataset provides fine-grained evaluations of various occlusion types, revealing that dynamic occlusions (e.g., pedestrian crossing) are more challenging than static occlusions (e.g., carrying backpacks).

## Highlights & Insights
- **Innovative Transition to Action Detection Perspective**: Re-framing occluded gait recognition as an action detection problem is not just a simple analogy, but is built on a deep understanding of the temporal structure of gait. The ingenuity lies in the fact that action detection naturally handles "valid segment selection," which precisely matches the needs of occlusion processing.
- **End-to-End Learning with Only ID Labels**: Significant action decomposition is learned without requiring occlusion or action annotations, exploiting the task's own supervision signal as implicit guidance. This significantly lowers the barrier to applying the method.
- **Benchmark Value of OccGait Dataset**: For the first time, multiple occlusion types are systematically defined and evaluated, establishing a unified evaluation standard for occluded gait recognition. This holds long-term value for fostering research in this direction.
- **Elegant Application of MoE**: Instead of using MoE simply as a tool for scaling capacity, its dynamic routing property is exploited to active "adaptive occlusion coping," allowing different experts to handle different occlusion patterns.

## Limitations & Future Work
- **Computational Overhead of MoE**: Parallel inference of multiple experts incurs additional computational and memory overhead, which may be a bottleneck for real-time application scenarios.
- **Generalization to Occlusion Types**: The occlusion augmentation strategy during training may not cover all real-world occlusion patterns, such as non-rigid occlusions (e.g., flowing clothes) or progressive occlusions.
- **Information Bottleneck of Binary Silhouettes**: The method relies on gait silhouette sequences, whereas silhouette extraction itself can fail under severe occlusion, presenting an upstream bottleneck.
- **Cross-View Capability**: The paper does not fully discuss the method's performance under multi-view conditions; the interaction effect between occlusion patterns and view variations is worth studying.
- Future directions: Incorporating skeleton information to complement silhouettes; exploring transformer architectures to replace CNNs for temporal modeling; utilizing large language models' reasoning capabilities for cross-frame inference.

## Related Work & Insights
- **vs GaitPart**: GaitPart divides the human body into fixed horizontal strips for part-level feature extraction, which offers some robustness to occlusion but fails to handle dynamic and arbitrary-position occlusions; the dynamic expert mechanism of GaitMoE is more flexible.
- **vs GaitGL**: GaitGL uses a fusion strategy of global and local features, but its temporal modeling remains simple (temporal pooling) and cannot leverage gait periodicity to tackle occlusion; GaitMoE explicitly models the action structure.
- **vs GaitBase**: GaitBase is a strong baseline utilizing a ViT-like architecture but lacks specialized design for occlusion; GaitMoE integrates the MoE mechanism tailored for occlusion on top of this.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel action detection perspective, original application of MoE in gait recognition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validation on four datasets, exhaustive ablation studies, contribution of a new dataset.
- Writing Quality: ⭐⭐⭐⭐ In-depth motivation analysis, clear method descriptions, and well-organized experiments.
- Value: ⭐⭐⭐⭐ The OccGait dataset and the method provide an important impetus for the field of occluded gait recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ICML 2026\] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](../../ICML2026/video_understanding/unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)
- [\[CVPR 2026\] VidPrism: Heterogeneous Mixture of Experts for Image-to-Video Transfer](../../CVPR2026/video_understanding/vidprism_heterogeneous_mixture_of_experts_for_image-to-video_transfer.md)
- [\[ICLR 2026\] Exposing and Defending the Achilles' Heel of Video Mixture-of-Experts](../../ICLR2026/video_understanding/exposing_and_defending_the_achilles_heel_of_video_mixture-of-experts.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)

</div>

<!-- RELATED:END -->
