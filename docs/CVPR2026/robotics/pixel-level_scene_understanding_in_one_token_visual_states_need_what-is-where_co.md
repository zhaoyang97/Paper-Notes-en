---
title: >-
  [Paper Note] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition
description: >-
  [CVPR 2026][Robotics][Visual state representation] This paper proposes CroBo, a self-supervised framework that learns visual state representations via a global-local reconstruction objective: a global reference image is compressed into a single bottleneck token, which is then used to reconstruct a heavily masked (90%) local crop, compelling the bottleneck token to encode pixel-level "what-is-where" scene composition. CroBo achieves state-of-the-art performance on the Franka Kitchen and DMC robot policy learning benchmarks.
tags:
  - CVPR 2026
  - Robotics
  - Visual state representation
  - self-supervised learning
  - robot policy learning
  - bottleneck token
  - global-local reconstruction
date: 2026-05-08
content_hash: 9fd4d34805137b26
---

# Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition

**Conference**: CVPR 2026
**arXiv**: [2603.13904](https://arxiv.org/abs/2603.13904)
**Code**: [Project Page](https://seokminlee-chris.github.io/CroBo-ProjectPage)
**Area**: Robotics / Self-Supervised Learning
**Keywords**: Visual state representation, self-supervised learning, robot policy learning, bottleneck token, global-local reconstruction

## TL;DR
This paper proposes CroBo, a self-supervised framework that learns visual state representations via a global-local reconstruction objective: a global reference image is compressed into a single bottleneck token, which is then used to reconstruct a heavily masked (90%) local crop, compelling the bottleneck token to encode pixel-level "what-is-where" scene composition. CroBo achieves state-of-the-art performance on the Franka Kitchen and DMC robot policy learning benchmarks.

## Background & Motivation
1. **Background**: Self-supervised visual representation learning methods (MAE, DINO, DINOv2, etc.) achieve strong performance on image classification and semantic segmentation, yet none of these methods explicitly consider what a good visual state representation should encode. For sequential decision-making in robotics, raw visual observations must be compressed into compact visual states.
2. **Limitations of Prior Work**: (1) Contrastive learning methods (DINO) learn global semantic features that are insensitive to spatial location; (2) MAE learns local features via patch-level reconstruction but lacks global scene understanding; (3) SiamMAE/CropMAE learn patch-level correspondences, which are ill-suited for downstream tasks requiring a single compact representation; (4) ToBo introduces the bottleneck token concept but relies on temporal frame reconstruction without explicitly constraining spatial composition encoding.
3. **Key Challenge**: Robots must detect subtle scene changes (e.g., a robotic arm moving a few centimeters), which demands that visual states simultaneously encode "what objects are present" and "where they are." Existing SSL methods either discard spatial information (augmentation invariance in contrastive learning) or do not produce a single compact representation (patch-level methods output token sequences).
4. **Goal**: Learn a single-token visual state representation that encodes complete "what-is-where" scene composition—which semantic entities exist, where they are, and how they are arranged.
5. **Key Insight**: If a compact representation can recover the content of an arbitrary local crop from minimal cues, it must encode both the semantic and spatial information of the entire scene.
6. **Core Idea**: Use a single [CLS] token as a global bottleneck to reconstruct a 90%-masked local crop, forcing the bottleneck token to become a compressed representation of pixel-level scene composition.

## Method

### Overall Architecture
A single frame is sampled from a video to construct two views: a global source view $\mathbf{x}^g$ (large crop, scale $[0.5, 1.0]$) and a local target view $\mathbf{x}^l$ (small crop taken from within the source view, scale $[0.3, 0.6]$). A weight-sharing Siamese encoder processes both views: the source view is unmasked, while the target view is masked at 90%. A Transformer decoder then uses the [CLS] bottleneck token from the source view together with the small number of visible patch tokens from the target view to reconstruct the masked target patches.

### Key Designs

1. **Global-Local Reconstruction Objective**:

    - **Function**: Compels the [CLS] bottleneck token to encode the semantic-spatial composition of the entire scene.
    - **Mechanism**: The target view $\mathbf{x}^l$ is a spatial sub-region of the source view $\mathbf{x}^g$. After 90% masking, only approximately 14 visible patches remain in the target view. The decoder must reconstruct from two signals: (a) the [CLS] token of the source view, providing global scene context; and (b) the few visible patches of the target view, providing local cues. Because so few patches are visible, the decoder must rely heavily on the [CLS] token to infer masked content, thereby forcing the [CLS] token to encode "what each object is and where it is located in the scene."
    - **Design Motivation**: Conventional MAE reconstructs masked patches from visible patches within the same view, where 75% masking still leaves sufficient local context. CroBo's cross-view design combined with an extreme 90% masking rate renders reconstruction from local cues alone infeasible, necessitating the use of the global bottleneck.

2. **Siamese Encoder + High Masking Rate**:

    - **Function**: Weight sharing ensures a consistent representation space; the high masking rate enforces dependence on the bottleneck.
    - **Mechanism**: Both the source and target views are processed by the same ViT encoder. All patches of the source view are fed to the encoder, producing $N$ patch tokens and one [CLS] token; only the 10% visible patches of the target view are encoded. The masking rate is raised from 75% (the MAE standard) to 90%, and experiments further demonstrate that 95% yields better performance.
    - **Design Motivation**: At lower masking rates (e.g., 75%), the decoder may reconstruct target patches from visible patches alone, bypassing the bottleneck token. A 90% masking rate ensures that visible information is insufficient for independent reconstruction.

3. **Crop vs. Time Design Choice**:

    - **Function**: Identify the optimal strategy for constructing source-target view pairs.
    - **Mechanism**: Three construction strategies are compared—Time (different frames), Crop (different crops from the same frame), and Time+Crop (different frames with cropping). Crop consistently outperforms Time, as the target view is a spatial subset of the source view, making the reconstruction objective well-defined. Time introduces ambiguity from motion and illumination changes. Time+Crop performs worst, as the combination of both sources of variation makes the reconstruction objective excessively ill-posed.
    - **Design Motivation**: Although CroBo does not use temporal training data (only single frames), it is trained on video datasets to enable fair comparison with prior work.

### Loss & Training
The loss is the MSE reconstruction loss over masked patches:

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{\mathbf{x}}_i^l - \mathbf{x}_i^l \|_2^2$$

Normalized pixel targets are used. The encoder is ViT-S/16 with an 8-layer decoder (512-dim). Optimization uses AdamW with a batch size of 1536, trained on Kinetics-400 for 400 epochs.

## Key Experimental Results

### Main Results (Franka Kitchen success rate % + DMC normalized score)

| Method | Knob on | Light on | Sdoor | Ldoor | Micro | walker/stand | walker/walk | reacher/easy |
|--------|---------|----------|-------|-------|-------|-------------|-------------|-------------|
| MAE | 12.0 | 24.3 | 71.5 | 12.8 | 10.0 | - | - | - |
| DINOv2 | 25.0 | 46.6 | 87.8 | 17.6 | 21.8 | 81.2 | 38.2 | 87.6 |
| ToBo | 58.4 | 80.6 | 98.4 | 44.2 | 51.2 | 87.0 | 77.7 | 87.5 |
| **CroBo** | **65.6** | **87.6** | **99.4** | 41.2 | **64.8** | **92.0** | **80.8** | **95.8** |
| Δ SOTA | +7.2 | +7.0 | +1.0 | -3.0 | +13.6 | +5.0 | +3.1 | +8.3 |

### Ablation Study

| Configuration | Knob1 | Light | Sdoor | Ldoor | Micro |
|---------------|-------|-------|-------|-------|-------|
| Time (temporal frames) | 44.4 | 67.6 | 96.4 | 36.8 | 49.6 |
| **Crop (spatial crop)** | **57.6** | **81.6** | **98.6** | **36.8** | **50.4** |
| Time+Crop | 38.2 | 62.8 | 90.8 | 22.2 | 28.4 |
| Masking rate 75% | 41.4 | 70.6 | 94.0 | 22.6 | 35.0 |
| Masking rate 90% | 57.6 | 81.6 | 98.6 | 36.8 | 50.4 |
| Masking rate 95% | **59.0** | **86.6** | **99.4** | **41.2** | **58.0** |

### Key Findings
- **Crop significantly outperforms Time**: Same-frame cropping provides a well-defined reconstruction target, whereas temporal frames introduce excessive ambiguity. This challenges the intuition that video data is necessary to learn dynamic representations.
- **Higher masking rates are consistently better**: Performance improves monotonically from 75% to 95% across all tasks, indicating that fewer target cues force the model to more fully exploit the bottleneck token.
- **ViT-S surpasses all ViT-L baselines**: The smaller model (65.0% average success rate) outperforms all larger-model baselines (best: 63.3%), with gains attributable to superior representations rather than model scale.
- **Perceptual linearity analysis**: CroBo achieves a representation curvature of only 75.4° on DAVIS video, far below DINOv2's 103.28°, indicating that CroBo's representations are more temporally linear and better encode "what is moving where."
- **Reconstruction visualization**: Even when 90% of the target view is masked, CroBo correctly recovers fully occluded objects (e.g., two cyan spheres invisible in the CLEVR target view but correctly reconstructed in position), demonstrating that the bottleneck token indeed encodes complete what-is-where information.

## Highlights & Insights
- The finding that **"dynamic representations can be learned without video"** is highly counter-intuitive: CroBo is trained solely on spatial crops from single frames, yet outperforms temporal video-based methods on dynamic robot tasks. This suggests that understanding spatial composition is foundational to understanding dynamics.
- The **use of extreme masking rates as an information bottleneck** elegantly unifies two ideas: MAE-style reconstruction learning and the information bottleneck principle. The higher the masking rate, the tighter the bottleneck, and the more the representation is compelled to encode global information.
- **Perceptual linearity** offers a new dimension for evaluating representation quality: a good visual state representation should render observation trajectories approximately linear in representation space, facilitating prediction and control.

## Limitations & Future Work
- Evaluation is limited to two simulated environments (Franka Kitchen and DMC); real-robot experiments are absent.
- The bottleneck is a single token with limited information capacity, which may be insufficient for more complex scenes.
- Only ViT-S/B/L are evaluated; larger or more recent visual backbones (e.g., ViT-G, DINOv2-L+) are not explored.
- The pixel-level MSE reconstruction target may be suboptimal; perceptual losses or feature-level reconstruction could be more effective.
- CroBo underperforms ToBo on the Ldoor task, suggesting that tasks requiring strong temporal correlation may still benefit from temporal information.

## Related Work & Insights
- **vs. ToBo**: CroBo's direct predecessor. ToBo reconstructs from temporal frames; CroBo reconstructs from spatial crops, yielding a cleaner objective.
- **vs. CropMAE**: A similar spatial cropping idea, but CropMAE learns patch-level correspondences whereas CroBo compresses information into a single token.
- **vs. SiamMAE**: SiamMAE also employs a Siamese encoder with video frame pairs but lacks the bottleneck token design.
- **vs. DINOv2**: Provides general-purpose representations but does not preserve fine-grained spatial information, resulting in substantially lower performance on robot tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The what-is-where concept is clearly articulated; the global-local reconstruction design is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two benchmarks, comprehensive ablations, reconstruction visualizations, and perceptual linearity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Strong narrative logic, deriving the method design from the question "what constitutes a good visual state."
- **Value**: ⭐⭐⭐⭐ — Significant implications for visual representation learning in robotics; the finding that single-frame training surpasses temporal training has broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] Towards Training-Free Scene Text Editing](towards_training-free_scene_text_editing.md)
- [\[CVPR 2026\] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)

</div>

<!-- RELATED:END -->
