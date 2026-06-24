---
title: >-
  [Paper Note] Revisiting Model Stitching in the Foundation Model Era
description: >-
  [CVPR 2025][Multimodal VLM][model stitching] This paper systematically studies the stitchability between heterogeneous Vision Foundation Models (e.g., CLIP, DINOv2, SigLIP 2), finding that pre-training the stitch layer with Final Feature Matching enables reliable stitching, and proposes the VFM Stitch Tree architecture to achieve efficient multi-VFM sharing.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "model stitching"
  - "vision foundation model"
  - "VFM"
  - "representation alignment"
  - "multimodal LLM"
date: 2026-05-08
content_hash: 781ccae89c7fc28b
---

# Revisiting Model Stitching in the Foundation Model Era

**Conference**: CVPR 2025  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: model stitching, vision foundation model, VFM, representation alignment, multimodal LLM

## TL;DR

This paper systematically studies the stitchability between heterogeneous Vision Foundation Models (e.g., CLIP, DINOv2, SigLIP 2), finding that pre-training the stitch layer with Final Feature Matching enables reliable stitching, and proposes the VFM Stitch Tree architecture to achieve efficient multi-VFM sharing.

## Background & Motivation

**Background**: VFMs have become the default backbone for vision tasks, with different VFMs (such as CLIP, DINOv2, and SigLIP 2) excelling in different tasks. Modern multimodal systems increasingly deploy multiple VFMs simultaneously.

**Limitations of Prior Work**: Deploying $k$ VFMs concurrently incurs $k\times$ computational and memory overhead; it remains unclear whether the internal representations of these heterogeneous VFMs are compatible and reusable.

**Key Challenge**: Prior model stitching research was only validated on small models trained on the same dataset (e.g., ResNet-18 on CIFAR-10). It is unknown whether large models with entirely different training objectives, datasets, and mixtures of modalities can still be stitched in the VFM era.

**Goal**: Are heterogeneous VFMs stitchable? How to properly train the stitch layer? Can we obtain performance beyond a single model from stitching?

**Key Insight**: A systematic experimental protocol (stitch point × stitch layer family × training loss × downstream task) is designed to reveal the failure modes of existing methods and propose a new solution.

**Core Idea**: Use Final Feature Matching to match the features of the target model's final layer to initialize the stitch layer, enabling reliable stitching of heterogeneous VFMs and the fusion of complementary knowledge.

## Method

### Overall Architecture

Given a source VFM $f_\theta$ and a target VFM $f_\phi$, insert a stitch layer $S$ at the $n$-th layer to construct the stitched model:
$$F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$$
where $R_\theta^n$ takes the first $n$ layers of features from the source, $T_\phi^N$ takes the remaining $N-n$ layers from the target, and only $S$ is trainable.

### Key Designs

**1. Failure Analysis of Layer Feature Matching (LFM)**
- **Function**: Train the stitch layer to minimize the feature discrepancy at the stitching point: $\|S(R_\theta^n(x)) - R_\phi^n(x)\|_2^2$.
- **Mechanism**: Although the layer feature distance is very low (on the order of $10^{-3}$), the final feature distance is high, especially for stitching at shallower layers.
- **Design Motivation**: Small mismatches in the intermediate layers are accumulated and amplified by the subsequent frozen layers, leading to severe deviation in the output features.

**2. Final Feature Matching (FFM)**
- **Function**: Train the stitch layer to match the patch features of the final layer (pre-logit) of the target model.
- **Mechanism**: $\mathcal{L}_{\text{FFM}} = \frac{1}{M}\sum_{i=1}^{M} \|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$, note that FFM is label-free.
- **Design Motivation**: Directly constraining the final output eliminates the error accumulation problem; it is surprisingly found that FFM simultaneously maintains a low feature distance at the stitching point, showing that final-layer supervision can implicitly induce intermediate-layer alignment.

**3. Two-Stage Training Strategy**
- **Function**: (i) Pre-train the stitch layer with FFM $\rightarrow$ (ii) fine-tune using downstream task loss.
- **Mechanism**: Resolve the gradient vanishing problem of Task Loss Training (TLT) during shallow-layer stitching, where gradients must propagate through a large number of frozen layers to reach the stitch layer. FFM pre-training provides a good initialization to bypass this optimization difficulty.
- **Design Motivation**: TLT achieves only 25.1% accuracy at Layer 2 for DINOv2 $\rightarrow$ SigLIP2, which improves to 51.7% with FFM initialization.

**4. Self-Stitch Baseline Design**
- **Function**: Insert the same stitch layer within the same model (e.g., DINOv2 $\rightarrow$ DINOv2) to rule out pseudo-improvements brought about by the capacity of the stitch layer.
- **Mechanism**: If cross-model stitching outperforms self-stitching within the same model, it indicates that complementary knowledge fusion indeed has occurred.
- **Design Motivation**: A rigorous control experiment design that isolates stitch layer capacity from genuine knowledge fusion.

### Loss & Training

- Stage 1: $\mathcal{L}_{\text{FFM}}$ (label-free, features can be pre-extracted for offline training)
- Stage 2: $\mathcal{L}_{\text{task}}$ (cross-entropy classification or mIoU segmentation)
- The default stitch layer is a 2-layer MLP + ReLU

## Key Experimental Results

### Main Results

fMoW classification accuracy (%), two-stage FFM+TLT:

| Stitch Direction | Layer 2 | Layer 6 | Layer 10 | Layer 14 | Layer 18 | Layer 22 |
|---|---|---|---|---|---|---|
| DINOv2→SigLIP2 | 51.7 | 55.8 | 59.3 | 68.0 | 72.0 | 71.8 |
| SigLIP2→DINOv2 | 53.8 | 53.8 | 61.9 | 69.6 | 70.4 | 72.2 |

Multi-dataset validation (Layers 6/14/22):

| Direction | fMoW | iNaturalist | Aircraft | ADE20K(mIoU) |
|---|---|---|---|---|
| DINOv2→DINOv2 (self) | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| SigLIP2→SigLIP2 (self) | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| DINOv2→SigLIP2 | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| SigLIP2→DINOv2 | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

### Ablation Study

Stitch layer selection (fMoW, DINOv2→SigLIP2, Layer 22):

| Stitch Layer | Acc(%) |
|---|---|
| Linear | 69.6 |
| MLP | **71.8** |
| LoRA | 67.3 |

FFM initialization vs. no initialization (TLT, DINOv2→SigLIP2):

| Layer | Without FFM | With FFM |
|---|---|---|
| 2 | 25.1 | **51.7** |
| 6 | 39.4 | **55.8** |
| 22 | 68.6 | **71.8** |

### Key Findings

1. **Cross-model stitching consistently outperforms self-stitching**: Across all datasets and tasks, the stitched models outperform the self-stitch baselines by +0.7% to +5.5%, demonstrating the existence of complementary knowledge fusion.
2. **FFM initialization is critical**: During shallow-layer stitching, TLT fails severely (25.1%), whereas FFM pre-training completely restores and surpasses linear probing.
3. **Weak sources drag down performance**: When CLIP (a weaker VFM) is the source, the stitched model cannot match the performance of a strong target; however, it performs well when CLIP acts as the target.
4. **MLP outperforms LoRA**: Potentially because a moderate amount of mismatch actually aids the fusion of complementary information.
5. **Practicality of VFM Stitch Tree**: VST-22 (with 4.3% extra resources) recovers 45% of the dual-VFM gain; VST-14 (with 39% extra resources) recovers 84%.

## Highlights & Insights

- Upgrades model stitching from a diagnostic tool to a practical utility, proving that heterogeneous VFMs can be reliably stitched to fuse complementary knowledge.
- The insights on FFM are profound: final-layer supervision can implicitly induce intermediate-layer alignment, and it is entirely label-free.
- The self-stitch baseline is rigorously designed, effectively ruling out explanations based on pseudo-improvements from capacity.
- VFM Stitch Tree offers a continuous performance-efficiency trade-off knob for multi-VFM deployment.

## Limitations & Future Work

- VFM Stitch Tree is only validated on VQAv2 and MME; a broader multimodal evaluation remains to be confirmed.
- Only VFMs with the same architecture (ViT) are considered; stitching across different architectures remains unexplored.
- FFM requires running the full forward pass of the target model, leaving room for improvement in training efficiency.
- Dynamic selection of stitching points or adaptive usage of different VFM branches based on the input remains unexplored.

## Related Work & Insights

- Bansal et al. (2021) proposed the "Anna Karenina principle" of model stitching: successful models learn similar representations; this work validates and extends this conclusion at the VFM level.
- SN-Net designs a family of stitchable networks for elastic inference; this work focuses on the post-hoc stitching of independently trained heterogeneous VFMs.
- Insight: The shallower layers of VFMs might encode pre-training-specific features while deeper layers are more transferable, which can guide the design of efficient inference architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The discoveries regarding FFM and the two-stage training are profound; the VFM Stitch Tree holds practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly systematic—spanning multiple datasets, tasks, VFMs, and stitch layer types, with a rigorously designed self-stitch control.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, with a thorough analysis of failure modes and a progressively layered experimental design.
- **Value**: ⭐⭐⭐⭐ Holds significant guiding value for both the understanding of VFM representations and the efficient deployment of multiple VFMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Evaluating Model Perception of Color Illusions in Photorealistic Scenes](evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)

</div>

<!-- RELATED:END -->
