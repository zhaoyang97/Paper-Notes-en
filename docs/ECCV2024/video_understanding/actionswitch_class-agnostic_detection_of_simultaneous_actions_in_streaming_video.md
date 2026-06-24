---
title: >-
  [Paper Note] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos
description: >-
  [ECCV 2024][Video Understanding][Online Temporal Action Localization] ActionSwitch is proposed—the first online temporal action localization (On-TAL) framework to detect overlapping action instances in streaming videos without category information. The core idea is to model multi-action detection as a state classification problem for a finite state machine, augmented by a conservativeness loss to reduce fragmented false positives. It achieves SOTA among OAD-extension methods…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Online Temporal Action Localization"
  - "Class-agnostic Detection"
  - "Overlapping Actions"
  - "Conservativeness Loss"
  - "Finite State Machine"
date: 2026-05-08
content_hash: eb9b484732068fe7
---

# ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos

**Conference**: ECCV 2024  
**arXiv**: [2407.12987](https://arxiv.org/abs/2407.12987)  
**Code**: [https://github.com/hyolim-kang/ActionSwitch](https://github.com/hyolim-kang/ActionSwitch)  
**Area**: Video Understanding / Online Temporal Action Localization  
**Keywords**: Online Temporal Action Localization, Class-agnostic Detection, Overlapping Actions, Conservativeness Loss, Finite State Machine  

## TL;DR
ActionSwitch is proposed—the first online temporal action localization (On-TAL) framework to detect overlapping action instances in streaming videos without category information. The core idea is to model multi-action detection as a state classification problem for a finite state machine, augmented by a conservativeness loss to reduce fragmented false positives. It achieves SOTA among OAD-extension methods on datasets such as THUMOS14, FineAction, and Epic-Kitchens 100.

## Background & Motivation
Online action detection (OAD) only performs frame-level classification, lacking instance-level understanding, whereas online temporal action localization (On-TAL) requires determining the start and end times of each action instance in video streams in real-time. Existing On-TAL methods face two core limitations: 
(1) **Inability to handle overlapping actions**: Class-agnostic OAD models output only 0/1, which causes multiple co-occurring actions to be merged into a single instance. Although class-aware methods can group tokens by class, they struggle when overlapping actions of the same class occur (e.g., heavily overlapping same-class actions in Epic-Kitchens), and manual threshold tuning becomes highly inflexible as the number of classes increases. 
(2) **Fragmented detections**: The inherent instability of frame-by-frame decision-making leads to a single long action being fragmented into multiple short instances. Moreover, pre-defining all action classes is impractical in open-world scenarios, making the decoupling of proposal generation from classification a more reasonable design.

## Core Problem
How can multiple co-occurring action instances (including same-class overlaps) be detected in real-time in streaming videos without relying on class information, while effectively suppressing fragmented false positives?

## Method
The overall mechanism is highly elegant: abstracting multi-action detection as a multi-switch machine, where each "switch" independently manages the detection of one action path. When multiple switches are turned on simultaneously, it represents the co-occurrence of multiple actions. The combinations of switches are encoded into state labels using a finite state machine, allowing the OAD model to directly predict frame-level states, which are subsequently decoded into action instances from the state sequence.

### Overall Architecture
- **Input**: Pre-extracted video feature sequence $f_t \in \mathbb{R}^D$
- **Core Model**: State-emitting OAD model—unidirectional GRU encoder + state classifier (MLP + residual), outputting the probability distribution of $S$ states $p_t = \text{softmax}(\text{SC}(g_t))$
- **State Decoding**: The current frame state is obtained via $s_t = \arg\max(p_t)$, and boundary transitions are determined instantly by comparing $s_t$ with $s_{t-1}$.
- **Output**: Online accumulated class-agnostic action proposals (start/end times), which can be fed into an independent classifier later to obtain category labels.
- **Inference Speed**: Thanks to the lightweight GRU + MLP design, the model runs at **over 500 fps**.

### Key Designs
1. **Finite State Machine Modeling (ActionSwitch)**: Taking a 2-switch configuration as an example, the 4 states represent "no action", "only switch 1 active", "only switch 2 active", and "both active". State transitions naturally correspond to action boundaries. Replacing manual thresholding with argmax completely avoids the threshold-tuning explosion seen in class-aware methods as class numbers grow. Two overlapping actions of the same class can be captured separately by different switches, which is impossible for class-aware methods.
2. **State Label Encoding (Encode Action)**: During training, GT action instances are encoded into frame-level state labels. To resolve ambiguity, it is assumed that switch 1 activates first and switch 2 activates second. In practice, the state label is the sum of the active switch IDs (e.g., when switch 1 and switch 2 are active, the state is 3).
3. **State Decoding (Decode State)**: During inference, a historical state queue is maintained, and action start/end times are inferred on-the-fly by comparing status changes of adjacent frames. The state sequence is in one-to-one correspondence with the action instances, eliminating the need for an extra boundary-matching module and fundamentally resolving boundary matching errors.

### Loss & Training
**Conservativeness Loss $\mathcal{L}_c$**—The most concise and refined design of this paper:

$$\mathcal{L}_c(p_t, s_{t-1}) = \begin{cases} -\log(p_t[s_{t-1}]), & \text{if } \arg\max(p_t) \neq s_{t-1} \\ 0, & \text{otherwise} \end{cases}$$

A penalty is applied only when the model predicts a state jump, forcing it to be more conservative when "changing its mind". The prediction of the previous frame $s_{t-1}$ is used as a pseudo-label to apply cross-entropy on the transition frame. The total loss is formulated as:

$$\mathcal{L} = CE(p_t, y_t) + \alpha \cdot \mathcal{L}_c(p_t, s_{t-1})$$

This is highly concise to implement (5 lines of PyTorch), requiring no architectural modifications and working plug-and-play. $\alpha$ controls the degree of conservativeness; a larger $\alpha$ improves precision at the expense of recall.

**Classifier**: A vanilla Transformer classifier is auxiliary-trained to predict category labels and confidence scores using the input feature sequence, which is used for mAP evaluation.

## Key Experimental Results

| Dataset | Metric | ActionSwitch | Prev. SOTA On-TAL | Gain |
|--------|------|-------------|----------------|------|
| THUMOS14 | F1@0.5 | **53.2** | CAG-QIL 45.8 | +7.4 |
| THUMOS14 | Avg mAP | **40.3** | CAG-QIL 33.1 | +7.2 |
| FineAction | F1@0.5 | **19.44** | CAG-QIL 15.67 | +3.77 |
| FineAction | Avg mAP | **5.36** | CAG-QIL 4.45 | +0.91 |
| Epic-Kitchens 100 | F1@0.5 | **32.44** | OAT 27.58 | +4.86 |
| Epic-Kitchens 100 | mAP@0.5 | **3.597** | OAT 3.296 | +0.30 |
| MultiTHUMOS | F1 (3-switch) | **32.76** | OAT 29.63 | +3.13 |
| THUMOS14 ODAS | p-mAP@offset=1 | **33.06** | SimOn 31.45 | +1.61 |

Note: OAT belongs to TAL-extension methods (which have looser constraints). ActionSwitch still leads comprehensively under the stricter OAD-extension constraints.

### Ablation Study
- **Number of Switches**: Increasing from 1-switch to 2-switch significantly improves recall (THUMOS14: 59.49→63.96), confirming that the extra switch captures overlapping actions. Under MultiTHUMOS (a dense overlap dataset), the optimal configuration is a 3-switch setup.
- **Conservativeness Loss Weight $\alpha$**: There is a precision-recall trade-off, where $\alpha=0.025$ yields the best F1 across most datasets. When $\alpha=0$ (no conservativeness loss), precision is low (THUMOS14: 35.75); including the loss improves it to 47.73 ($\alpha=0.025$).
- **Side Effects of Excessively Large $\alpha$**: At $\alpha=0.05$, recall drops significantly as the model becomes overly conservative, missing genuine action boundaries.
- **SimOn Collapses on Large-Category Datasets**: On Epic-Kitchens, SimOn generates 137k proposals (vs only 9.6k GT), leading to a precision of only 2.35%. This shows that class-aware grouping strategies fail completely when the category count is high.

## Highlights & Insights
- **Brilliant Finite State Machine Abstraction**: Translating the mutual-exclusion detection problem of multiple OAD models into a state classification problem for a single model greatly simplifies the architecture. The state sequence and action instances have an intuitive one-to-one correspondence, bypassing the need for boundary-matching.
- **Incredibly Simple yet Effective Conservativeness Loss**: Implemented via 5 lines of code without architectural changes, it directly injects the "sparse action boundary" prior into the loss function. This design concept is highly generalizable—suitable for any task requiring temporal smoothness.
- **Argmax Replacing Threshold Calibration**: Completely resolves the painful process of manual threshold tuning, which is particularly beneficial for datasets with large numbers of categories.
- **Inference Speed >500fps**: The lightweight GRU+MLP design makes real-time application fully viable.

## Limitations & Future Work
- **Preset Number of Switches**: The number of switches must be manually configured based on dataset characteristics, failing to adaptively determine the number of overlapping actions at the current frame. A dynamic switch allocation mechanism could be introduced.
- **Ambiguity in State Encoding**: Assuming switch 1 activates first and switch 2 activates second to resolve ambiguities during training may introduce systematic biases in certain scenarios.
- **Complete Decoupling of Classifier and Proposal Generation**: Although this is the core design philosophy of this paper, class information may benefit proposal quality, and exploring weak utilization of such info is a worthy direction.
- **Huge Gap Compared to Offline Methods**: Compared to offline methods like ActionFormer, there is still a noticeable mAP gap (THUMOS14: 40.3 vs 62.6); the online constraint naturally limits the performance ceiling.
- **Reliance on Pre-extracted Features**: The visual encoder is not trained end-to-end, making feature quality a potential bottleneck.

## Related Work & Insights
- **vs CAG-QIL**: CAG-QIL is also a class-agnostic OAD extension, but it requires two-stage training (OAD training followed by Q-imitation learning for the grouping module), and it cannot handle overlapping actions. ActionSwitch completes this in a single-stage model, and its conservativeness loss is simpler and more effective than CAG-QIL's grouping module.
- **vs SimOn**: SimOn is a class-aware method that groups actions class-by-class, leading to a proposal explosion under massive categories (137k proposals vs 9.6k GT on Epic-Kitchens) and an inability to detect same-class overlapping actions. ActionSwitch operates completely independent of class information, showing better scalability.
- **vs OAT**: OAT is a TAL-extension method that allows backtracking to determine start times at the end of an action (representing a looser constraint), thereby achieving higher mAP on smaller datasets (THUMOS14). However, its training is unstable on large datasets (requiring downsampling to work on FineAction), and it relies heavily on class info and manual thresholds. ActionSwitch remains competitive under stricter constraints and naturally supports the ODAS task.

## Related Work & Insights
- **Transfer Value of Conservativeness Loss**: It is highly applicable to any scenario mapping frame-level predictions to instance-level outputs, such as mask consistency in video object segmentation, or attention focus stability in VideoQA.
- **Class-agnostic + Late Classification Paradigm**: Consistent with the proposal-first, classify-later two-stage design philosophy in object detection. This is particularly valuable in open-vocabulary scenarios, where it can be combined with video-language models to realize open-vocabulary online temporal action localization.

## Rating
- Novelty: ⭐⭐⭐⭐ The finite state machine modeling is novel and natural, and the conservativeness loss is simple yet effective, though the overall contribution is somewhat lightweight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering 4 datasets (THUMOS14/FineAction/Epic-Kitchens/MultiTHUMOS) with thorough ablation, but lacking end-to-end experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logic, intuitive diagrams, and a very smooth problem-solution-experiment flow.
- Value: ⭐⭐⭐⭐ Establishes a strong baseline for On-TAL. The class-agnostic paradigm provides positive insights for open-world applications, although On-TAL itself is a relatively niche field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ECCV 2024\] Classification Matters: Improving Video Action Detection with Class-Specific Attention](classification_matters_improving_video_action_detection_with_class-specific_atte.md)
- [\[ECCV 2024\] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)
- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](../../CVPR2026/video_understanding/streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)

</div>

<!-- RELATED:END -->
