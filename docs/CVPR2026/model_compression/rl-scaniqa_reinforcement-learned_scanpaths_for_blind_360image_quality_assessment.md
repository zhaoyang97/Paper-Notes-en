---
title: >-
  [Paper Note] RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360° Image Quality Assessment
description: >-
  [CVPR 2026][Model Compression][360 image quality] This paper proposes RL-ScanIQA, the first end-to-end reinforcement learning-based blind 360° image quality assessment (BIQA) framework. The core idea is to formulate scanpath generation as a sequential decision-making process, using a PPO policy to learn task-driven viewing strategies directly from quality assessment feedback, rather than relying on imitation learning from human gaze data. The framework consists of two jointly optimized modules—a scanpath generator and a quality assessor—augmented with multi-level rewards (step-level exploration, set-level diversity, and task-aligned perception) and distortion-space data augmentation. The method achieves state-of-the-art performance and strong cross-dataset generalization on three benchmarks: CVIQD, OIQA, and JUFE.
tags:
  - CVPR 2026
  - Model Compression
  - 360 image quality
  - reinforcement-learning
  - scanpath
  - blind IQA
  - PPO
  - active perception
date: 2026-05-08
content_hash: f3a0c03dc9dca247
---

# RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360° Image Quality Assessment

**Conference**: CVPR 2026
**arXiv**: [2603.14297](https://arxiv.org/abs/2603.14297)
**Code**: None
**Area**: Model Compression
**Keywords**: 360 image quality, reinforcement-learning, scanpath, blind IQA, PPO, active perception

## TL;DR

This paper proposes RL-ScanIQA, the first end-to-end reinforcement learning-based blind 360° image quality assessment (BIQA) framework. The core idea is to formulate scanpath generation as a sequential decision-making process, using a PPO policy to learn task-driven viewing strategies directly from quality assessment feedback, rather than relying on imitation learning from human gaze data. The framework consists of two jointly optimized modules—a scanpath generator and a quality assessor—augmented with multi-level rewards (step-level exploration, set-level diversity, and task-aligned perception) and distortion-space data augmentation. The method achieves state-of-the-art performance and strong cross-dataset generalization on three benchmarks: CVIQD, OIQA, and JUFE.

## Background & Motivation

1. **Viewport constraints in 360° imagery**: Panoramic images in immersive environments can only be experienced progressively through limited viewports; quality perception depends on the viewing trajectory rather than the full image.
2. **Decoupled scanpath and quality assessment in prior methods**: Existing scanpath methods treat path generation as an independent preprocessing step, precluding end-to-end optimization and misaligning paths with IQA objectives.
3. **Dependence on human gaze data**: Prior methods require human eye-tracking data for supervision, which is costly to collect and may bias toward salient content rather than quality-relevant regions.
4. **ERP projection distortion**: Directly analyzing equirectangular projections introduces spatial bias and ignores spherical geometry.
5. **Limitations of fixed sampling strategies**: Methods based on predefined viewports ignore the sequential and content-adaptive nature of user exploration.
6. **Poor cross-dataset generalization**: Large variation in distortion types across datasets causes fixed-strategy methods to degrade sharply in cross-domain scenarios.

## Method

### Scanpath Generator (PPO Policy Network)

The sphere is discretized into $8 \times 4 = 32$ candidate viewports ($90° \times 90°$ FOV), modeled as a finite-horizon MDP:
- **State**: $s_t = [h_{t-1}; g]$, where $h_{t-1}$ is the GRU history hidden state and $g$ is the global image descriptor extracted by DINOv2.
- **Action**: Attention scoring over candidate viewport features + Softmax selection of the next viewport.
- **Optimization**: PPO with clipped objective, GAE advantage estimation, and entropy regularization.

### Multi-Level Reward Design

**A. Step-Level Exploration Reward**: $r_t = \lambda_{\text{ent}} \cdot \mathcal{H}(x_t) + \lambda_{\text{ssim}} \cdot (1-\text{SSIM}) + \lambda_{\text{nov}} \cdot \delta_{\text{new}} + \lambda_{\text{eqb}} \cdot \mathcal{B}(x_t)$
- Information entropy encourages attention to texture-rich regions; SSIM dissimilarity promotes diverse exploration; a novelty signal prevents revisiting; an equatorial bias prior simulates human fixation habits.

**B. Scanpath Diversity Reward**: $\mathcal{R}_{\text{div}} = \beta_{\text{cov}} \cdot \frac{|\cup_k S_k|}{X} - \beta_{\text{jac}} \cdot \text{mean Jaccard similarity}$
- Encourages $K$ paths to cover a larger spherical area while penalizing inter-path overlap.

**C. Task-Aligned Perceptual Reward**: MSE negative reward $\mathcal{R}_{\text{mse}}$ + ranking reward $\mathcal{R}_{\text{rank}}$
- Feedback directly derived from IQA prediction error, aligning path generation with the quality prediction objective.

### Quality Assessor

- Attention-weighted aggregation of viewport features: $\alpha_t$ is computed via interaction between local feature $f_t$ and global feature $g$.
- The aggregated representation is concatenated with the global feature and passed through an MLP to regress the quality score.
- Final score is averaged over predictions from $K$ paths.

### Cross-Domain Augmentation

- **Consistency loss**: Predictions after weak augmentation should remain stable.
- **Triplet loss**: Score ordering constraint among clean / mildly distorted / heavily distorted images.
- **Cross-ranking loss**: Relative quality relationships between image pairs are preserved after augmentation.

## Key Experimental Results

### Table 1: In-Dataset Evaluation (SRCC / PLCC)

| Method | JUFE | OIQA | CVIQD |
|--------|------|------|-------|
| NIQE (handcrafted) | 0.552 / 0.592 | 0.745 / 0.736 | 0.893 / 0.872 |
| MC360IQA | 0.502 / 0.623 | 0.875 / 0.906 | 0.877 / 0.892 |
| Assessor360 | 0.489 / 0.510 | 0.979 / 0.945 | 0.958 / 0.963 |
| GSR-X | 0.843 / 0.857 | 0.922 / 0.937 | 0.805 / 0.957 |
| Q-Insight (LLM) | 0.557 / 0.412 | 0.643 / 0.795 | 0.872 / 0.801 |
| **RL-ScanIQA** | **0.816 / 0.902** | **0.941 / 0.967** | **0.970 / 0.970** |

> RL-ScanIQA achieves the highest PLCC on all datasets, and also the best SRCC on CVIQD. On JUFE, PLCC substantially outperforms the runner-up (0.902 vs. 0.857), demonstrating the advantage of the RL policy under real-world distortion distributions.

### Table 2: Cross-Dataset Evaluation (SRCC / PLCC)

| Method | Train: CVIQD → Test: OIQA / JUFE | Train: JUFE → Test: CVIQD / OIQA |
|--------|----------------------------------|----------------------------------|
| Assessor360 | 0.853/0.632 — 0.887/0.749 | 0.617/0.724 — 0.405/0.499 |
| GSR-X | 0.804/0.765 — 0.831/0.694 | 0.782/0.732 — 0.733/0.611 |
| F-VQA(A) | 0.772/0.621 — 0.604/0.509 | 0.665/0.679 — 0.683/0.732 |
| **RL-ScanIQA** | **0.901/0.800 — 0.913/0.822** | **0.771/0.755 — 0.802/0.833** |

> Cross-dataset generalization is significantly superior to all baselines, validating the effectiveness of distortion augmentation and ranking consistency constraints.

## Highlights & Insights

1. **First end-to-end RL-based 360° IQA framework**: Jointly optimizes scanpath generation and quality assessment without requiring human eye-tracking data.
2. **Sophisticated multi-level reward design**: From step-level to set-level to task-level, sparse IQA supervision is transformed into dense shaping signals.
3. **Counterintuitive yet valuable finding**: Human real fixation trajectories perform worse than RL-learned paths (Table 3: SRCC 0.724 → 0.816), suggesting that humans tend to fixate on salient rather than quality-critical regions.
4. **Strong cross-domain generalization**: Distortion-space augmentation combined with ranking consistency losses enables effective transfer across diverse distortion types.
5. **Intuitive and convincing visualizations**: Paths for high-quality images cover the sphere uniformly, while paths for low-quality images concentrate on distorted regions.

## Limitations & Future Work

1. **High computational overhead**: Inference requires $K=15$ paths $\times$ $T=7$ steps $= 105$ viewport feature extractions, limiting real-time applicability.
2. **Coarse viewport discretization**: 32 candidate viewports may be insufficient to precisely localize fine-grained distortion regions.
3. **Limited benchmark coverage**: Only three datasets are evaluated; panoramic IQA datasets are small in scale, with CVIQD and OIQA each containing only a few hundred images.
4. **Frozen DINOv2 feature extractor**: A frozen pretrained model may not be the most distortion-sensitive feature extraction solution.
5. **Dependence on MOS annotations**: Training still requires precise human subjective quality scores, incurring non-trivial annotation costs.
6. **Heavy reward hyperparameter burden**: Four step-level weights, two diversity weights, two task-alignment weights, and five loss function weights yield a heavy tuning burden.

## Related Work & Insights

- **2D BIQA**: BRISQUE (natural scene statistics), DBCNN, TreS, MANIQA (Transformer), Q-Insight (multimodal RL + LLM).
- **360° BIQA**: MC360IQA (multi-branch CNN with fixed viewports), VGCN (graph convolution over viewport relations), Assessor360 / GSR-X / F-VQA (scanpath modeling with decoupled training).
- **RL for visual tasks**: Viewpoint planning, video summarization, attention selection; PPO combined with variance reduction and value guidance demonstrates robustness under sparse rewards.
- **360° visual exploration**: Eye-tracking studies reveal human viewing characteristics such as equatorial bias and salient object preference.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|:------------:|-------|
| Novelty | 8 | First end-to-end RL integration into 360° IQA; the joint path + assessment optimization paradigm is novel. |
| Technical Contribution | 8 | Multi-level reward design is well-motivated; cross-domain augmentation strategy is effective. |
| Experimental Thoroughness | 7 | Three datasets and complete ablations, but dataset scale is limited. |
| Writing Quality | 8 | Clear structure, rich figures and tables, comprehensive comparisons. |
| Value | 7 | Demand for 360° IQA is growing, but inference overhead and hyperparameter count may hinder deployment. |
| **Overall** | **7.6** | **An excellent work introducing active perception into 360° quality assessment; the end-to-end RL paradigm offers meaningful inspiration.** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](../../AAAI2026/model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[CVPR 2026\] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)

</div>

<!-- RELATED:END -->
