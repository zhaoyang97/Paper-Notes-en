---
title: >-
  [Paper Note] Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D
description: >-
  [CVPR 2026][LLM Reasoning][2D-to-3D conversion] A new paradigm called "Artistic Disparity Synthesis" (Art3D) is proposed, shifting the goal of 2D-to-3D conversion from geometric accuracy to artistic expression. A dual-pa…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "2D-to-3D conversion"
  - "artistic disparity synthesis"
  - "stereoscopic film"
  - "dual-path architecture"
  - "depth style"
date: 2026-05-08
content_hash: 8dd92e70783c6289
---

# Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D

**Conference**: CVPR 2026
**arXiv**: [2603.05906](https://arxiv.org/abs/2603.05906)
**Code**: None (not yet open-sourced)
**Area**: LLM Reasoning
**Keywords**: 2D-to-3D conversion, artistic disparity synthesis, stereoscopic film, dual-path architecture, depth style

## TL;DR
A new paradigm called "Artistic Disparity Synthesis" (Art3D) is proposed, shifting the goal of 2D-to-3D conversion from geometric accuracy to artistic expression. A dual-path architecture decouples global depth style from local artistic effects, learning directorial intent from professional 3D film data.

## Background & Motivation
**Background**: Current 2D-to-3D conversion methods (e.g., diffusion-based StereoCrafter, Eye2Eye) have achieved geometric accuracy but lack artistic immersion—a significant gap remains compared to the viewing experience of professional 3D films such as *Avatar*.

**Limitations of Prior Work**: Geometric reconstruction paradigms (MonoDepth, MiDaS, etc.) treat the artistic disparity adjustments found in professional 3D films as "noise" to be suppressed, resulting in an "artistic poverty" problem—geometrically correct but narratively barren outputs.

**Key Challenge**: The three principal artistic operations in professional 3D post-production—Global Depth control, Zero-Plane selection, and Local Sculpting—are all encoded in disparity maps, yet existing methods cannot learn these artistic intentions.

**Goal**: To generate disparity maps from 2D images that embody the director's artistic intent, rather than merely physically accurate disparity.

**Key Insight**: The disparity map is treated as a carrier of artistic expression, enabling indirect learning of global depth style and local pop-out effects from professional 3D films.

**Core Idea**: A dual-path supervision mechanism decouples the director's global macro-intent from local "artistic brushstrokes," learning artistic disparity style from professional 3D films through indirect supervision.

## Method

### Overall Architecture
Art3D employs a three-network architecture: a frozen DepthNet for geometric feature extraction (Depth Anything V2), a frozen StereoNet for extracting the target artistic blueprint (SEA-RAFT), and a trainable CameraNet (lightweight U-Net) for synthesizing virtual camera parameters.

The core formulation models the artistic blueprint as a linear transform of the geometric canvas:

$$\hat{d}^L = vs \cdot iz + vt$$

where $vs$ and $vt$ are per-pixel scale and offset tensors, and $iz$ is the inverse depth map.

### Key Designs
1. **Dual-Path Supervision Mechanism**: The mixed signal $d^L$ is decomposed into global style ($M_{global}$) and local effects ($M_{local}$). The global mask is obtained by taking valid regions from StereoNet's left-right consistency check and removing the local regions: $M_{global} = M_{valid} \cdot (1 - M_{local})$. The local mask is generated via Lang-SAM with text prompts (e.g., "foreground character popping out"). This design is highly robust to errors—undetected pop-out regions naturally degrade to global-path supervision.
2. **CameraNet Architecture**: A lightweight encoder-decoder structure (3 downsampling + 3 upsampling stages) that outputs only 3 channels ($vs$, $vt$, and the right-view disparity $\hat{d}^R$). It is the only trainable component in the entire framework.
3. **DDC-IoU Data Filtering**: A Depth-Disparity Consistency IoU metric is proposed to filter low-quality frames (those with overly simplistic depth layering), with a threshold of 0.8. This yields 90K high-quality stereo image pairs from 25 3D films.

### Loss & Training
The core loss $\mathcal{L}_{Art}$ is defined as the sum of dual-path masked least-squares residuals:

$$\mathcal{L}_{Art} = \mathcal{L}_{path}(M_{global}) + \mathcal{L}_{path}(M_{local}) + \mathcal{L}_{st}$$

where $\mathcal{L}_{path}(M) = \min_{s,t} \sum_k M_k \cdot \|d^L_k - (s \cdot \hat{d}^L_k + t)\|^2$.

The global style regularization $\mathcal{L}_{st} = \|s-1\|^2 + \|t\|^2$ encourages the synthesized disparity to directly reflect the global supervision signal. Auxiliary losses include a smoothness loss and a left-right consistency loss. Training runs for 50 epochs on a single A800 GPU, with batch size 32 and input resolution 512×512.

## Key Experimental Results

### Main Results: Global Depth Style Evaluation

| Method | Global Depth $s$ (mean/std) | Zero-Plane $t$ (mean/std) |
|---|---|---|
| Baseline (w/o $\mathcal{L}_{Art}$) | 0.030 / 0.018 | 6.98 / 2.35 |
| **Art3D (Ours)** | **0.020 / 0.009** | **6.08 / 1.80** |
| Ground Truth | 0.013~0.023 / 0.010~0.020 | 4.35~5.28 / 2.09~4.68 |

Art3D achieves significantly reduced standard deviation ($\sigma$), indicating that a stable and consistent artistic style has been learned rather than random geometric disparity.

### Ablation Study: Paradigm Comparison

| Method | Global Control (Zero-Plane) | Local Sculpting (Artistic) |
|---|---|---|
| StereoCrafter | Manual (global shift) | None |
| Eye2Eye | Physical (reproduced) | None |
| **Art3D (Ours)** | **Learned (global style)** | **Yes (learned)** |

### Geometric Consistency Validation (DDC-IoU)
Art3D consistently achieves DDC-IoU of 0.83–0.89 in the right-view coordinate system, demonstrating that artistic style learning does not compromise underlying geometric consistency. In contrast, raw 3D film data exhibits inconsistent quality—some frames have DDC-IoU of 0 (poor structural alignment)—underscoring the necessity of data filtering.

### Key Findings
- Removing $\mathcal{L}_{path}(M_{local})$ allows the model to learn only global style, with no local pop-out effects produced.
- Art3D consistently achieves DDC-IoU of 0.83–0.89, confirming that artistic style learning does not degrade geometric consistency.
- Professional 3D software Owl3D produces inconsistent 3D perception across different scenes, whereas Art3D maintains stable pop-out effects.

## Highlights & Insights
- **Paradigm Innovation**: The first work to explicitly propose a paradigm shift from "geometric reconstruction" to "artistic disparity synthesis," repositioning the disparity map as a vehicle for cinematic storytelling.
- **Elegant Indirect Supervision**: Rather than directly supervising at the pixel level with GT, style parameters $(s, t)$ are extracted via least-squares fitting to assess artistic consistency through distributional alignment.
- **Robust Design**: The dual-path masks are complementary—missed local detections degrade gracefully to global supervision, and sparse global masks act as a form of data augmentation.
- **Compelling Motivation via Avatar**: The Jake/Ikran flying sequence from *Avatar* is used to concretely illustrate the three-layer artistic intent, making the motivation highly persuasive.
- **Minimal CameraNet**: As the only trainable component, its 3-downsampling + 3-upsampling + 1-output-layer design demonstrates that the framework architecture, rather than network capacity, is the primary driver of performance.

## Limitations & Future Work
- The paper self-identifies as a "preliminary exploration"; the CameraNet architecture is relatively simple (only 6 layers), limiting generative capacity.
- Local pop-out data consists of only 201 clips (~15K frames), representing a limited data volume.
- Validation is confined to 3D film data; generalization to non-cinematic scenarios (e.g., AR/VR content) remains unexplored.
- Evaluation relies primarily on statistical distribution comparisons, with no user perceptual studies.
- Integration with existing diffusion-based generation pipelines (e.g., StereoCrafter) has not been explored.
- A unified model is used across all film types (animation, sci-fi, contemporary) rather than training style-specific models.

## Related Work & Insights
- Traditional heuristic disparity remapping methods (nonlinear remapping, saliency-based editing) require stereo pairs as input and cannot generalize to monocular settings.
- Geometric reconstruction paradigms (Deep3D, MonoDepth → StereoCrafter, Eye2Eye), despite incorporating diffusion models, remain geometry-driven.
- Art3D fills the gap between heuristic artistic editing and geometric reconstruction, enabling cross-film 3D style transfer from monocular input.
- StereoCrafter normalizes the zero-plane position during data processing, actively discarding the director's original artistic intent.
- Eye2Eye can produce pop-out effects, but these are learned from physically accurate VR180 data and thus represent reproductions of physical disparity rather than artistic design.
- The three-layer artistic intent framework defined in this work (global depth / zero-plane / local sculpting) provides a clear analytical framework for future research in 3D visual creation.

## Data Construction Details
- Data is drawn from 25 well-known 3D films (e.g., *Hugo*, *The Amazing Spider-Man*, *The Great Gatsby*), following the data protocol of Ranftl et al.
- After DDC-IoU ≥ 0.8 filtering, 90K pairs of 1080P stereo images are retained: 80K for training and 10K for testing.
- Local pop-out data is manually collected from YouTube (201 clips), yielding ~15K frames after processing, which are added to the training set.
- Both positive and negative disparities are extracted by StereoNet, preserving complete pop-out/push-in information.

## Rating ⭐
- **Novelty**: ⭐⭐⭐⭐⭐ — Paradigm-level innovation; the first work to incorporate "artistic intent" into 2D-to-3D conversion.
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablations are thorough, but quantitative comparisons against SOTA and perceptual user studies are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is articulated with exceptional persuasiveness; the *Avatar* case study is vivid and effective.
- **Value**: ⭐⭐⭐⭐ — Opens a new research direction, though as a preliminary exploration, practical deployment requires further development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](../../ICLR2026/llm_reasoning/scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)
- [\[ICLR 2026\] Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts](../../ICLR2026/llm_reasoning/beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](../../ACL2026/llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](../../ACL2026/llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)

</div>

<!-- RELATED:END -->
