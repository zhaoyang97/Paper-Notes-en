---
title: >-
  [Paper Note] MatAnyone: Stable Video Matting with Consistent Memory Propagation
description: >-
  [CVPR 2025][Segmentation][Video Matting] The MatAnyone framework is proposed, which achieves consistent propagation in memory space via a region-adaptive memory fusion mechanism (maintaining semantic stability in core regions and capturing fine alpha details in boundary regions). Together with a new dataset VM800 and a training strategy that directly supervises the matting head using segmentation data, it realizes robust and high-quality object-specified video matting.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video Matting"
  - "Memory Propagation"
  - "Region-Adaptive Fusion"
  - "alpha matte"
  - "Object Specification"
date: 2026-05-08
content_hash: 3ab7a17d3e8f8ebd
---

# MatAnyone: Stable Video Matting with Consistent Memory Propagation

**Conference**: CVPR 2025  
**arXiv**: [2501.14677](https://arxiv.org/abs/2501.14677)  
**Code**: [https://pq-yang.github.io/projects/MatAnyone](https://pq-yang.github.io/projects/MatAnyone)  
**Area**: Video Matting / Segmentation  
**Keywords**: Video Matting, Memory Propagation, Region-Adaptive Fusion, alpha matte, Object Specification

## TL;DR

The MatAnyone framework is proposed, which achieves consistent propagation in memory space via a region-adaptive memory fusion mechanism (maintaining semantic stability in core regions and capturing fine alpha details in boundary regions). Together with a new dataset VM800 and a training strategy that directly supervises the matting head using segmentation data, it realizes robust and high-quality object-specified video matting.

## Background & Motivation

**Background**: Auxiliary-free portrait video matting (such as RVM) relies solely on input frames, which easily fails in complex/ambiguous backgrounds (such as multiple characters). Object-specified video matting borrows the setting of VOS (semi-supervised video object segmentation) — requiring only the first frame's segmentation mask — to achieve stable cross-frame tracking through a memory matching paradigm.

**Limitations of Prior Work**: (1) When existing mask-guided methods (AdaM, FTP-VM, MaGGIe) fine-tune VOS priors for matting, the poor quality (holes in core regions, blurry boundary details) and limited scale of video matting data (VideoMatte240K) easily disrupt the semantic stability of the VOS priors; (2) Core and boundary regions have drastically different requirements for memory matching — core regions require stable propagation, while boundary regions require fine updates, yet existing memory frameworks treat all tokens equally; (3) When training with segmentation data, parallel head schemes are adopted, meaning the matting head itself cannot receive supervision from the ground-truth segmentation data.

**Key Challenge**: How to simultaneously ensure the semantic stability of core regions and the matting-level details of boundary regions under the condition of training with sub-optimal video matting data?

**Key Insight**: A region-adaptive mechanism is introduced in the memory propagation phase — predicting the alpha change probability of each token relative to the previous frame, where "large change" regions (boundaries) rely on current-frame information queried from the memory bank, and "small change" regions (cores) retain previous-frame memory, achieving selective memory fusion.

## Method

### Overall Architecture

Based on the memory matching paradigm (similar to STCN/Cutie), the input is the first-frame segmentation mask and a sequence of video frames to predict the alpha matte frame by frame. The current frame $t$ is encoded into features $F^t$ ($16\times$ downsampled), and is fused with memory bank information and previous-frame information via the Consistent Memory Propagation (CMP) module to obtain the pixel memory readout $P^t$. Object-level semantics are then extracted by an Object Transformer and fed into a decoder to predict the alpha matte $M^t$. The prediction is encoded as a memory value $V^t$ to update the alpha memory bank.

### Key Designs

1. **Consistent Memory Propagation (CMP) + Region-Adaptive Memory Fusion**:
    - **Alpha Memory Bank**: Stores alpha mattes (instead of segmentation masks/trimaps), allowing the memory paradigm to provide stability in boundary regions as well.
    - **Change Probability Prediction**: Uses a lightweight 3-layer convolutional module to predict the change probability $U_t$ of each token, using the binarized results of the inter-frame alpha difference $|M_{t-1}^{GT} - M_t^{GT}| \geq \delta$ as supervision.
    - **Soft Fusion**: $P_t = V_t^m \cdot U_t + V_{t-1} \cdot (1 - U_t)$, where high $U_t$ (boundary/changing regions) relies more on the current frame information queried from the memory bank, and low $U_t$ (core/stable regions) retains the memory from the previous frame.
    - **Design Motivation**: The alpha of the core region barely changes between frames; directly propagating the memory of the previous frame avoids matching noise. Boundary regions need to be updated according to the current frame to capture fine alpha transitions.

2. **Segmentation Data Supervision Strategy for Core Regions**:
    - **Novelty**: Directing segmentation data into the matting head (instead of a parallel segmentation head) and supervising it with region-split losses.
    - **Core Region**: Guided by segmentation labels, utilizing L1 loss $\mathcal{L}_{core}$ to ensure semantic stability.
    - **Boundary Region**: Lacking alpha GT, an improved Scaled DDC loss is used: $\mathcal{L}_{boundary} = |(\alpha_i - \alpha_j)(F-B) - \|I_i - I_j\|_2|$.
    - **Correction to original DDC loss**: The original assumption $\|\alpha_i - \alpha_j\| = \|I_i - I_j\|$ holds only when $|F-B|=1$. Introducing a foreground/background color difference scaling generates more natural edges.
    - **Design Motivation**: Direct supervision on the matting head with segmentation data leverages segmentation priors more thoroughly than parallel head schemes.

3. **First-Frame Recurrent Refinement during Inference**:
    - Treats the first frame as a sequence by repeating it $n$ times, exploiting the frame-by-frame refinement property of the memory paradigm, and taking only the $n$-th frame's output as the actual first-frame result.
    - Enhances robustness against the given segmentation mask, while boosting the first frame's quality to the level of image-level matting.
    - **Design Motivation**: The quality of the first-frame matte directly affects subsequent frames; recurrent refinement improves the first-frame quality with zero cost.

### Loss & Training

- **Matting Data**: L1 + Laplacian loss + Grad loss (standard matting losses)
- **Segmentation Data**: $\mathcal{L}_{core}$ (L1) + $\mathcal{L}_{boundary}$ (Scaled DDC)  
- **Change Probability Prediction**: $\mathcal{L}_{bin\_seg}$ (binary cross-entropy)
- **Three-stage training**: VM800 matting $\rightarrow$ incorporating core-region supervision from segmentation data $\rightarrow$ fine-tuning with image matting data

## Key Experimental Results

### Main Results

**VideoMatte 1080p:**

| Method | MAD↓ | MSE↓ | Grad↓ | dtSSD↓ |
|------|------|------|-------|--------|
| RVM-Large (AF) | 5.81 | 0.97 | 9.65 | 1.78 |
| MaGGIe† (per-frame mask) | 4.42 | 0.40 | 4.03 | 1.31 |
| **MatAnyone** | **4.24** | **0.33** | **4.00** | **1.19** |

**Real-world benchmark (Core region metrics):**

| Method | MAD↓ | MSE↓ | dtSSD↓ |
|------|------|------|--------|
| RVM-Large | 0.95 | 0.50 | 1.30 |
| MaGGIe | 1.94 | 1.53 | 1.63 |
| **MatAnyone** | **0.14** | **0.10** | **0.89** |

- On the real-world benchmark, the MAD is 85% lower than the second-best method RVM-Large (0.14 vs 0.95).
- MatAnyone requires only the first-frame mask, whereas MaGGIe requires per-frame mask guidance, yet MatAnyone still outperforms it.

### Ablation Study

| Component | MAD↓ | dtSSD↓ |
|------|------|--------|
| Baseline (w/o CMP, old data, old training) | High | High |
| +New Data (VM800) | Improved | Improved |
| +CMP (Memory propagation) | Significantly improved | Significantly improved |
| +New Training (Segmentation supervision) | **Globally optimal** | **Globally optimal** |

### Key Findings

- The CMP module simultaneously improves core region stability and boundary region details — the core regions directly propagate the previous frame to avoid matching noise, and boundary regions focus on alpha transitions.
- Scaled DDC loss vs. original DDC: The original version produces segmentation-like step-like edges, whereas the scaled version produces more natural matting transitions.
- The quality of the VM800 dataset contributes significantly to training — being 2x larger, more diverse, and having higher boundary quality than VideoMatte240K.
- First-frame recurrent refinement ($n=3$) can significantly improve robustness against coarse initial masks.

## Highlights & Insights

- **Elegant region-adaptive memory fusion**: A lightweight prediction module naturally combines the "core vs. boundary" characteristics of matting with the memory propagation mechanism, preserving the semantic stability of VOS while enhancing boundary detail.
- **Breakthrough in segmentation data supervision**: Prior methods using parallel heads could not fully leverage segmentation priors. Directly supervising the matting head and handling boundary regions with Scaled DDC is a key innovation.
- **High practicality**: Requires only the first-frame mask (which can be obtained by tools like SAM), supports instance-level matting, and maintains stability in long videos and complex backgrounds.

## Limitations & Future Work

- Generalization capability to non-human objects has not been fully verified (training data is mostly portraits).
- Extreme motion blur or extremely fast motion may cause the change probability prediction to be inaccurate.
- Relying on the memory matching of the VOS framework, large-scale long videos may face memory bank management issues.
- The domain gap between synthetic training data and the real world has not been completely eliminated.

## Related Work & Insights

- **Paradigm shift from VOS to Video Matting**: Memory matching is the core paradigm of VOS (STCN $\rightarrow$ XMem $\rightarrow$ Cutie). This paper demonstrates that this paradigm can be directly applied to finer matting tasks with appropriate adaptations.
- **Ideas for improving DDC loss**: For supervision methods on data without ground-truth alpha, analyzing default assumptions and applying corrections (adding foreground-background scaling) represents a methodology worth adopting.
- **Application prospects of instance-level video matting**: Based on the first-frame-specified setting, it can replace traditional green screen solutions in scenarios such as video editing, virtual backgrounds, and special effects production.

## Rating

⭐⭐⭐⭐ — The region-adaptive memory fusion is ingeniously designed, and the core-region segmentation supervision strategy is innovative and practical. The experiments are comprehensive, showing prominent advantages on real-world benchmarks. The dataset contributions (VM800 + YoutubeMatte) provide a better foundation for training and evaluation in the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](../../CVPR2026/segmentation/matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[ICLR 2026\] Matting Anything 2: Towards Video Matting for Anything](../../ICLR2026/segmentation/matting_anything_2_towards_video_matting_for_anything.md)
- [\[CVPR 2025\] MaSS13K: A Matting-level Semantic Segmentation Benchmark](mass13k_a_matting-level_semantic_segmentation_benchmark.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)

</div>

<!-- RELATED:END -->
