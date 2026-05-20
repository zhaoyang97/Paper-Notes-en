---
title: >-
  [Paper Note] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models
description: >-
  [CVPR 2026][Multimodal VLM][VLA models] This paper proposes HiF-VLA, a framework that uses Motion Vectors (MV) as compact temporal primitives to unify three temporal reasoning capabilities—Hindsight, Insight…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "VLA models"
  - "motion representation"
  - "temporal reasoning"
  - "long-horizon manipulation"
  - "world models"
date: 2026-05-08
content_hash: daa1d12c65836b55
---

# HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models

**Conference**: CVPR 2026
**arXiv**: [2512.09928](https://arxiv.org/abs/2512.09928)  
**Code**: [GitHub](https://github.com/OpenHelix-Team/HiF-VLA)  
**Area**: Multimodal VLM
**Keywords**: VLA models, motion representation, temporal reasoning, long-horizon manipulation, world models

## TL;DR

This paper proposes HiF-VLA, a framework that uses Motion Vectors (MV) as compact temporal primitives to unify three temporal reasoning capabilities—Hindsight, Insight, and Foresight—enabling bidirectional temporal extension of VLA models. HiF-VLA substantially outperforms baselines on long-horizon manipulation tasks with minimal computational overhead.

## Background & Motivation

Vision-Language-Action (VLA) models have achieved remarkable progress in robotic manipulation by mapping visual and linguistic inputs to action spaces in an end-to-end manner. However, most VLA models implicitly assume the Markov property—predicting actions solely from the current observation—without explicitly modeling temporal dependencies. This leads to **temporal myopia**, manifested as trajectory fragmentation and degraded task-level coherence in long-horizon manipulation.

Existing approaches to mitigate temporal myopia follow two directions:

**Historical frame stacking**: Methods such as TraceVLA and Octo concatenate multiple past observations as inputs, but suffer from severe redundancy—adjacent frames are highly similar—resulting in large computational overhead and high inference latency (Tab. 3 shows a 3.15× latency increase at history=4).

**Pixel-level subgoal prediction**: Methods such as CoT-VLA and Seer predict future visual subgoals, but are prone to local distortion and semantic drift.

The core argument of this paper is that **motion is a more suitable representation of temporal context than raw pixels**. Motion vectors capture the dynamic changes between states while filtering out static pixel noise, making them a natural bridge between history and future.

## Method

### Overall Architecture

HiF-VLA is built upon the OpenVLA-OFT architecture (Prismatic-7B VLM backbone) with three added components:

Original VLA inference: $\tilde{a}_{t:t+n} \sim P_\theta(a_{t:t+n} | o_t, l)$

HiF-VLA inference: $(\tilde{a}_{t:t+n}, \tilde{m}_{t:t+n}) \sim P'_\theta(a_{t:t+n}, m_{t:t+n} | o_t, l, m^{his}_{t-h:t})$

The model jointly predicts future motion and actions during training; motion decoding is optional at inference time.

### Key Designs

1. **Hindsight Prior Acquisition**:

    - MPEG-4 motion vectors (MV) replace raw frame stacking. MVs predict the displacement of macroblocks between adjacent frames, serving as a compressed representation inherent to video codecs.
    - MV format: $MV_{t-1:t}(x,y) = (x_t - x_{t-1}, y_t - y_{t-1})$, with shape $h \times (H/16) \times (W/16) \times 2$
    - A lightweight ViT encoder combined with shallow 3D convolutions encodes the MV sequence into compact hindsight tokens $M_h \in \mathbb{R}^{K_h \times d}$
    - Design motivation: MVs enable near-lossless reconstruction under video codec standards, making them inherently efficient and faithful representations of historical dynamics.

2. **Foresight Reasoning with Insight**:

    - $K_f$ learnable foresight query tokens and $K_a$ empty action tokens are introduced and fed into the VLM alongside the task instruction and current observation.
    - The VLM performs parallel inference via non-causal attention, outputting foresight motion tokens $M_f$ and action latent tokens $A_f$.
    - Motion vectors rather than raw pixels are predicted as the foresight target, avoiding the distortion and redundancy associated with pixel-level prediction.

3. **Hindsight-Modulated Joint Expert**:

    - Core innovation: rather than injecting historical motion into the VLM input—which would disrupt visual-language alignment—historical context is incorporated via AdaLN (Adaptive Layer Normalization) conditioning within the decoder layers.
    - Foresight motion tokens and action tokens form two parallel streams that interact through **cross-stream joint attention**, while maintaining independent FFNs to ensure complementary yet decoupled representations.
    - AdaLN: $\text{AdaLN}(z; h_c) = \gamma(h_c) \cdot \frac{z - \mu(z)}{\sigma(z)} + \beta(h_c)$
    - Design motivation: motion is the physical manifestation of action in visual space; jointly predicting both improves alignment between semantic understanding and underlying dynamics.

### Loss & Training

The total loss is a weighted sum of the action L1 loss and the motion reconstruction L1 loss:

$$\mathcal{L}_{all} = \mathcal{L}_A + \lambda \cdot \mathcal{L}_{MV}$$

where $\lambda = 0.01$. Training runs for 150K steps on LIBERO and 80K steps on CALVIN, using 8×A100 GPUs with a global batch size of 64.

## Key Experimental Results

### Main Results

**LIBERO-Long (10 tasks, 500 trials)**:

| Method | View | Avg. Success Rate |
|--------|------|-------------------|
| OpenVLA-OFT | Third-person | 91.0% |
| MemoryVLA | Third-person | 93.4% |
| **HiF-VLA** | **Third-person** | **94.4%** |
| OpenVLA-OFT | Multi-view | 94.0% |
| Seer | Multi-view | 87.7% |
| **HiF-VLA** | **Multi-view** | **96.4%** |

The third-person variant of HiF-VLA (94.4%) approaches the performance of the multi-view baseline.

**CALVIN ABC-D (trained on A–C, tested on unseen environment D)**:

| Method | View | Avg. Len. ↑ |
|--------|------|------------|
| VPP | Multi-view | 4.33 |
| Seer | Multi-view | 4.28 |
| **HiF-VLA** | **Multi-view** | **4.35** |
| HiF-VLA | Third-person | 4.08 |

### Ablation Study

**Efficiency comparison (LIBERO-Long, third-person, history=4)**:

| Configuration | GPU Memory | Latency | Success Rate |
|---------------|-----------|---------|--------------|
| Baseline | 30.8GB (1.00×) | 72.9ms (1.00×) | 91.0% |
| + Subgoal | 38.2GB (1.24×) | 115.9ms (1.59×) | 91.8% |
| + Foresight (HiF) | 31.8GB (1.03×) | 82.7ms (1.13×) | 92.2% |
| + History frames | 63.6GB (2.06×) | 229.5ms (3.15×) | 90.4% |
| + Hindsight (HiF) | 31.4GB (1.02×) | 117.7ms (1.61×) | 92.2% |
| + Hindsight+Foresight | 32.2GB (1.05×) | 121.6ms (1.67×) | 93.2% |

Historical frame stacking incurs a 3.15× latency increase while actually degrading performance; HiF-VLA's foresight component adds only 0.13× latency overhead.

**Hindsight injection position**: Conditioning hindsight information within the expert decoder outperforms direct injection into the VLM input, as motion information may interfere with the visual-language pretraining alignment.

**Hindsight length**: Peak performance is achieved at length 8 (94.4% third-person, 96.4% multi-view).

### Key Findings

1. Raw frame stacking not only incurs substantial computational overhead but can **degrade** performance (90.4% vs. 91.0%), as redundant pixel information dilutes task-relevant temporal cues.
2. Motion vectors as a historical representation are more efficient and more effective than raw frames—achieving a 1.2% absolute improvement with only 2% additional GPU memory.
3. HiF-VLA's inference latency increases only marginally with longer history lengths, whereas frame-stacking baselines scale nearly linearly (4.5× at history=8).
4. In real-world experiments, the baseline achieves only 17.4% on Press-Buttons-Order due to its inability to detect subtle visual differences between pressed and unpressed states; HiF-VLA successfully identifies fine-grained state transitions by virtue of its temporal receptive field.

## Highlights & Insights

- **Elegant borrowing of motion vectors**: Adopting MVs from the video codec domain as a historical representation has both theoretical grounding (near-lossless reconstruction) and practical advantages (compactness and efficiency)—a highly elegant cross-domain transfer.
- **"Think while acting" paradigm**: Jointly predicting motion and actions enables the VLA to reason about future dynamics simultaneously with action generation, analogous to human decision-making.
- **The ablation on hindsight injection position** is particularly instructive: it demonstrates that in pretrained multimodal models, the injection site of new modality information is critical—conditioning at the decoder or post-processing layer is safer than embedding it directly into the input.

## Limitations & Future Work

1. The current motion representation relies on estimation accuracy and may be sensitive to noise in highly dynamic scenes.
2. Large-scale pretraining on internet video to enhance motion understanding and generation has not been explored.
3. The hindsight window length may require adaptive tuning across different tasks; a fixed window is used in the current work.
4. Validation is limited to the LIBERO and CALVIN benchmarks; more complex real-world tasks (e.g., kitchen manipulation, warehouse logistics) remain unexplored.

## Related Work & Insights

- Compared to pixel-level subgoal prediction in CoT-VLA and UP-VLA, using motion vectors for foresight is more compact and less susceptible to distortion.
- Compared to frame-stacking approaches in TraceVLA and Octo, MV encoding substantially reduces redundancy while preserving informativeness.
- The AdaLN conditioning mechanism, originating from diffusion models (DiT), is creatively repurposed here for temporal modulation—a technique worth adopting more broadly.
- The proposed framework can be viewed as a motion-centric world model (World Action Model) that connects perception, dynamics, and control.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Motion vectors as temporal primitives + the hindsight-modulated joint expert design are highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Simulation + real-world, efficiency analysis, inference scalability, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, RQ-driven experimental design)
- Value: ⭐⭐⭐⭐⭐ (Provides an efficient and effective new paradigm for temporal modeling in VLA research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/multimodal_vlm/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)

</div>

<!-- RELATED:END -->
