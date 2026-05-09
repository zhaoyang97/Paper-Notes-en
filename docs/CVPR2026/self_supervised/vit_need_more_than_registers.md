---
title: >-
  [Paper Note] Vision Transformers Need More Than Registers
description: >-
  [CVPR 2026][Self-Supervised Learning][ViT artifacts] This paper systematically analyzes the artifact phenomenon widely observed in ViTs across fully supervised, text-supervised, and self-supervised paradigms, revealing that the root cause is "lazy aggregation"—ViTs exploit semantically irrelevant background patches as shortcuts to represent global semantics. The authors propose LaSt-ViT (LazyStrike ViT), which anchors the CLS token to foreground regions via frequency-aware selective channel aggregation, consistently eliminating artifacts and improving performance across 12 benchmarks.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - ViT artifacts
  - lazy aggregation
  - patch score
  - foreground aggregation
  - register tokens
date: 2026-05-08
content_hash: be7c52c1f35f6052
---

# Vision Transformers Need More Than Registers

**Conference**: CVPR 2026
**arXiv**: [2602.22394](https://arxiv.org/abs/2602.22394)
**Code**: [GitHub](https://github.com/ChengShiest/LAST-ViT)
**Area**: Visual Representation Learning / Vision Transformer Analysis
**Keywords**: ViT artifacts, lazy aggregation, patch score, foreground aggregation, register tokens

## TL;DR

This paper systematically analyzes the artifact phenomenon widely observed in ViTs across fully supervised, text-supervised, and self-supervised paradigms, revealing that the root cause is "lazy aggregation"—ViTs exploit semantically irrelevant background patches as shortcuts to represent global semantics. The authors propose LaSt-ViT (LazyStrike ViT), which anchors the CLS token to foreground regions via frequency-aware selective channel aggregation, consistently eliminating artifacts and improving performance across 12 benchmarks.

## Background & Motivation

**Background**: ViTs have become the de facto standard for image recognition and, more importantly, serve as general-purpose feature extractors (frozen foundation models) for diverse downstream tasks. ViTs trained under different supervision paradigms each have their strengths: fully supervised and text-supervised models (e.g., CLIP) excel at open-vocabulary tasks and serve as visual encoders for LVLMs, while self-supervised models (e.g., DINO) are well-suited for unsupervised segmentation and object discovery.

**Limitations of Prior Work**:
1. DINO identifies attention deficit issues in fully supervised ViTs.
2. CLIPSelf finds that dense features from text-supervised ViTs are misaligned with textual cues.
3. The Register paper discovers that self-supervised ViTs (DINOv2) produce high-norm token artifacts that impair object localization.
4. These phenomena suggest a shared underlying problem in ViTs, yet no unified explanation or solution has been proposed.

**Key Challenge**: ViTs achieve excellent image-level classification performance, but patch-level dense feature quality is poor—top-scoring patches fall on background rather than foreground regions, and register tokens only suppress the high-norm phenomenon without addressing the fundamental issue (PiB even degrades).

**Goal**: To define, analyze, and resolve the artifact problem in ViTs across different supervision paradigms from first principles in a unified manner.

**Key Insight**: The authors introduce Patch Score (CLS-patch cosine similarity) and Point-in-Box (PiB) as unified metrics for quantifying artifacts, and identify the root cause as lazy aggregation—global attention combined with coarse-grained supervision leads ViTs to take shortcuts by encoding global semantics through background patches.

**Core Idea**: Distinguish foreground from background patches via frequency-domain stability analysis, and selectively aggregate stable patches into the CLS token to eliminate lazy aggregation.

## Method

### Overall Architecture

The core of LaSt-ViT is replacing the original CLS token aggregation mechanism in ViTs (Attention Pooling or GAP) with a channel-wise frequency-stability-based Top-K selective aggregation. The intuition is that foreground patch features are more homogeneous along the channel dimension (semantically consistent) and are therefore more stable under low-pass filtering.

### Key Designs

1. **Patch Score and Point-in-Box (PiB) as Unified Metrics**:
    - **Patch Score** is defined as the cosine similarity between each patch feature and the CLS token: $\mathcal{S}_p = \frac{\mathbf{x}_{\text{patch}} \cdot Q_{\text{CLS}}}{\|\mathbf{x}_{\text{patch}}\|_2 \|Q_{\text{CLS}}\|_2}$
    - **PiB** measures whether the highest-scoring patch falls within a foreground ground-truth bounding box, serving as an indicator of artifact severity.
    - Experiments show that ViT achieves a PiB of only 42.7, far below ResNet's 68.4; while register tokens eliminate the high-norm phenomenon, PiB actually drops to 41.5.
    - These metrics are agnostic to the supervision paradigm and apply uniformly across fully supervised, text-supervised, and self-supervised ViTs.

2. **Validating the Lazy Aggregation Hypothesis**:
    - **Masking probe**: Removing the top 50% of patches by Patch Score has negligible or slightly positive effect on ImageNet accuracy (+1.2%), whereas removing low-scoring patches causes a sharp accuracy drop (−60% at 70% masking), confirming that high-scoring patches are semantically irrelevant background.
    - **Training dynamics**: ViT's PiB remains low (~0.42) from the very beginning of training and hardly changes, indicating that lazy aggregation is an intrinsic behavior established early in training.
    - **Factor disentanglement**: (1) Increasing patch size (28→fewer background tokens, −10%) raises PiB from 0.44 to 0.52 but hurts accuracy; (2) Replacing global attention with window attention raises PiB to 59.8 but drops accuracy from 72.3 to 63.9.
    - Conclusion: coarse-grained supervision (image-level labels) combined with global dependencies (long-range attention) jointly induces lazy aggregation.

3. **LaSt-ViT: Frequency-Aware Selective Aggregation**:
    - **Stability Score computation**: Apply a channel-wise 1D FFT to patch features → multiply by a Gaussian low-pass filter $\mathbf{g}$ → apply inverse FFT to obtain low-frequency features $\hat{\mathbf{x}}_{\text{patch}}$.
    - Stability score: $\mathbf{S}_{i,j} = \frac{\hat{\mathbf{x}}_{\text{patch}}[i,j]}{|\hat{\mathbf{x}}_{\text{patch}}[i,j] - \mathbf{x}_{\text{patch}}[i,j]| + \varepsilon}$, where a higher score indicates that patch $i$ is more stable in channel $j$ (and thus more likely to be foreground).
    - **Channel-wise Top-K Pooling**: For each channel $j$, select the $K$ patches with the highest stability scores and compute their mean as the corresponding channel value of the CLS token: $\mathcal{Q}_{\text{CLS}}[j] = \frac{1}{K} \sum_{i \in \mathcal{I}_K(j)} \mathbf{x}_{\text{patch}}[i,j]$
    - **Vote Count** visualization: Define $v_i = \sum_{j=1}^D \mathbf{1}\{i \in \mathcal{I}_K(j)\}$; patches with high vote counts are strongly aligned with foreground regions.

### Loss & Training

- LaSt-ViT introduces no additional loss functions; it only replaces the CLS token aggregation mechanism.
- It is applicable to any ViT pre-training pipeline (fully supervised, CLIP, DINO) as a drop-in replacement.
- The hyperparameter $K$ controls the number of patches selected per channel; the optimal value is approximately 50% of the total patch count (e.g., 98 out of 196 patches for ViT-B/16).
- The training procedure is identical to that of the original ViT, requiring no additional data or hyperparameter tuning.

## Key Experimental Results

### Main Results

Artifact elimination (Patch Score / PiB):

| Method | High Norm | Point-in-Box (PiB) |
|------|:---------:|---:|
| ResNet | ✗ | 68.4 |
| ViT | ✓ | 42.7 |
| ViT + Register | ✗ | 41.5 |
| ViT + **LazyStrike** | ✗ | **55.1** (+12.4) |
| DINO-ViT | ✗ | 44.5 |
| DINO + **LazyStrike** | ✗ | **69.7** (+25.2) |
| CLIP-ViT | ✓ | 39.8 |
| CLIP + **LazyStrike** | ✗ | **50.1** (+10.3) |

Zero-shot semantic segmentation (mIoU %, CLIP ViT-L/14):

| Method | VOC20 | ADE20K | Cityscapes | COCO-Stf. |
|------|---:|---:|---:|---:|
| CLIP | 17.1 | 1.6 | 2.7 | 3.2 |
| CLIP + **LazyStrike** | **72.4** (+55.3) | **8.4** (+6.8) | **12.3** (+9.6) | **11.9** (+8.7) |

### Ablation Study

CLS aggregation method comparison (OpenCLIP ViT-B/16):

| Method | ImageNet Top-1 | VOC20 (seg) | COCO-Stf. (seg) |
|------|---:|---:|---:|
| Attention-Pool | 55.8 | 49.0 | 7.2 |
| Max-Pool | 53.1 | 71.9 | 12.2 |
| LazyStrike K=1 | 53.5 | 72.7 | 13.5 |
| LazyStrike K=49 | 55.8 | **75.8** | **18.5** |
| LazyStrike K=98 | **56.2** | 75.9 | 18.0 |
| LazyStrike K=196 (Full) | 55.3 | 13.5 | 4.8 |

Unsupervised object discovery (CorLoc, DINO ViT-S):

| Method | VOC07 | VOC12 | COCO | FPS |
|------|---:|---:|---:|---:|
| DINO-seg | 45.8 | 46.2 | 42.1 | 29.4 |
| LOST | 61.9 | 64.0 | 50.7 | 29.4 |
| DINO + **LazyStrike** | **64.4** | **67.6** | **51.6** | **55.9** |

### Key Findings

1. Register tokens merely relocate the high-norm phenomenon from the feature map to the register tokens; PiB actually decreases (41.5 < 42.7), substantiating the claim that "Vision Transformers Need More Than Registers."
2. LazyStrike simultaneously eliminates both the high-norm and patch score artifacts, as both are distinct manifestations of the same underlying lazy aggregation behavior.
3. On CLIP ViT-L/14, zero-shot segmentation mIoU on VOC20 jumps from 17.1% to 72.4% (+55.3%), demonstrating that dense feature quality improves substantially once artifacts are eliminated.
4. LazyStrike endows fully supervised ViTs with emergent segmentation capability (evidenced by PCA visualizations), a property previously considered exclusive to self-supervised models such as DINO.
5. Setting $K$ to the full patch count degrades to GAP and hurts performance; setting $K$ too small discards too much information; the optimal value is $K \approx N/2$.

## Highlights & Insights

1. **Analytical rigor**: The investigation proceeds from first principles through masking probes, training dynamics tracking, and factor disentanglement experiments, yielding a rigorous hypothesis-validation workflow.
2. **Unified perspective**: Diverse artifact phenomena observed under three supervision paradigms are attributed to a single root cause (lazy aggregation), offering a new conceptual framework for understanding ViT behavior.
3. **Simplicity and effectiveness**: Replacing only the CLS aggregation mechanism—without additional modules, data, or losses—consistently improves performance across 12 benchmarks.
4. **Counterintuitive finding**: Emergent segmentation is not exclusive to self-supervised learning; fully supervised ViTs exhibit the same capability once lazy aggregation is eliminated.

## Limitations & Future Work

1. The frequency-domain stability assumption (foreground patches are more stable) may not hold in certain scenarios (e.g., texture-rich foreground against a uniform background).
2. Channel-wise Top-K selection requires additional FFT/IFFT computation; while lightweight, it still introduces overhead for real-time inference.
3. Validation is limited to ViT-S/B/L; larger-scale models (e.g., ViT-G) have not been evaluated.
4. The value of $K$ is fixed during training; adaptive or learnable selection mechanisms remain unexplored.
5. The paper title implies that register tokens are insufficient, yet the combination of LazyStrike and register tokens is not thoroughly investigated.

## Related Work & Insights

- **Register Tokens (Darcet et al.)**: Absorb high-norm artifacts via additional tokens, but do not address the root cause.
- **CLIPSelf**: Repairs CLIP dense features through additional alignment training, constituting a post-hoc solution.
- **MaskCLIP**: First demonstrates that CLIP features can be applied to zero-shot semantic segmentation.
- **F-ViT**: Performs open-vocabulary detection with a frozen CLIP backbone, directly benefiting from LazyStrike.
- **Insight**: Shortcut learning in pre-trained models is a pervasive problem; understanding model-internal behavior enables simple yet highly effective remediation.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)

</div>

<!-- RELATED:END -->
