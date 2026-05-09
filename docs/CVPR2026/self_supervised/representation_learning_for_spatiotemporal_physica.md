---
title: >-
  [Paper Note] Representation Learning for Spatiotemporal Physical Systems
description: >-
  [CVPR 2026][Self-Supervised Learning][JEPA] This paper systematically compares four self-supervised/physics-modeling methods on three PDE-based physical systems (active matter, shear flow, and Rayleigh-Bénard convection), finding that latent-space prediction (JEPA) consistently outperforms pixel-level prediction (VideoMAE) on physical parameter estimation tasks — achieving 28%–51% relative MSE reduction — and that JEPA trained with only 10% of fine-tuning data surpasses VideoMAE trained on 100% of the data. Notably, methods specifically designed for physical modeling are not always the optimal choice.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - JEPA
  - Physical Systems
  - Representation Learning
  - Parameter Estimation
  - VICReg
date: 2026-05-08
content_hash: 792f5d2106555484
---

# Representation Learning for Spatiotemporal Physical Systems

**Conference**: CVPR 2026
**arXiv**: [2603.13227](https://arxiv.org/abs/2603.13227)
**Code**: [GitHub](https://github.com/helenqu/physical-representation-learning)
**Area**: Self-Supervised / Representation Learning
**Keywords**: JEPA, Physical Systems, Representation Learning, Parameter Estimation, VICReg

## TL;DR

This paper systematically compares four self-supervised/physics-modeling methods on three PDE-based physical systems (active matter, shear flow, and Rayleigh-Bénard convection), finding that latent-space prediction (JEPA) consistently outperforms pixel-level prediction (VideoMAE) on physical parameter estimation tasks — achieving 28%–51% relative MSE reduction — and that JEPA trained with only 10% of fine-tuning data surpasses VideoMAE trained on 100% of the data. Notably, methods specifically designed for physical modeling are not always the optimal choice.

## Background & Motivation

**Background**: The dominant paradigm for machine learning on spatiotemporal physical systems is surrogate modeling via next-frame prediction, with the goal of learning an accurate simulator of system evolution. Representative works include physics foundation models such as MPP and Poseidon, as well as operator learning methods such as DISCO.

**Limitations of Prior Work**: Autoregressive surrogate models are expensive to train and suffer from accumulated prediction errors. More importantly, the practical needs of scientific research often center not on frame-by-frame prediction but on estimating physical parameters (e.g., Reynolds number, Prandtl number) that govern qualitative system behavior (laminar vs. turbulent flow). Which learning paradigm best preserves physically meaningful information has not been systematically studied.

**Key Challenge**: Pixel-level prediction objectives (MAE / autoregressive models) pursue precise reconstruction of visual detail, yet such low-level details may be irrelevant to high-level physical semantics. Physics-specific methods incorporate physical inductive biases, but whether they truly outperform general-purpose methods on downstream scientific tasks remains an open question.

**Goal**: To compare general-purpose self-supervised methods (JEPA vs. VideoMAE) and physics-modeling methods (MPP vs. DISCO) in terms of their effectiveness at learning physically meaningful representations, using physical parameter estimation as a quantitative evaluation criterion.

**Key Insight**: Physical parameters govern the temporal evolution of a system; therefore, parameter estimation error directly quantifies how much physical information is captured in a representation. This is a more faithful measure of "physical understanding" than next-frame prediction error.

**Core Idea**: JEPA's latent-space prediction objective naturally filters out low-level visual details while preserving high-level dynamical structure, enabling it to learn better physical representations than pixel-level prediction methods.

## Method

### Overall Architecture

All four methods are pretrained on three physical systems and then evaluated by freezing the encoder and training an attentive probe for physical parameter estimation. The three evaluation systems are drawn from The Well dataset: active matter (parameters $\alpha$, $\zeta$), shear flow (Reynolds number, Schmidt number), and Rayleigh-Bénard convection (Rayleigh number, Prandtl number).

### Key Designs

1. **JEPA — Dynamics Variant (Latent-Space Temporal Prediction)**:

    - **Function**: Given $k$ context frames $x_{t:t+k}$, learn to predict the latent representation of the next $k$ frames $x_{t+k:t+2k}$.
    - **Mechanism**: An encoder $f: \mathcal{X} \to \mathcal{Z}$ (ConvNeXt architecture) and a predictor $g: \mathcal{Z} \to \mathcal{Z}$ (inverted-bottleneck CNN) minimize the VICReg loss $\ell_{VICReg}(g(f(x_i)), f(x_{i+1})) = \lambda s + \mu[v(z_i)+v(z_{i+1})] + \nu[c(z_i)+c(z_{i+1})]$. The invariance term $s$ aligns predictions with targets; the variance term $v$ maintains per-dimension variance to prevent collapse; the covariance term $c$ decorrelates dimensions to remove redundancy. Hyperparameters: $\lambda=2, \mu=40, \nu=2$.
    - **Design Motivation**: Rather than reconstructing pixels, the model predicts future states in representation space. This forces the encoder to retain only information useful for dynamical prediction, naturally filtering out low-level visual textures irrelevant to physics.

2. **VideoMAE — Pixel-Level Masked Reconstruction (Baseline)**:

    - **Function**: Randomly mask spatiotemporal patches and reconstruct the masked pixels from the unmasked ones.
    - **Mechanism**: ViT-Small/16 architecture with temporal tube masking (the same spatial mask is shared across all frames); trained with pixel-level MSE reconstruction loss.
    - **Design Motivation**: Serves as a representative pixel-level self-supervised learning method to test whether pixel reconstruction also captures physical information.

3. **Physics-Modeling Baselines (DISCO and MPP)**:

    - **Function**: DISCO is an operator-learning-based in-context inference method that infers trajectory-specific evolution operators from short context windows; MPP is an autoregressive physics foundation model that predicts pixel values frame by frame.
    - **Mechanism**: DISCO combines the in-context learning capability of Transformers with the physical inductive biases of neural operators, using the inferred operator for integration; MPP is trained on large collections of physical simulation data to learn general spatiotemporal field prediction.
    - **Design Motivation**: These two methods represent distinct technical approaches designed specifically for physics — operator learning (latent space) and autoregressive foundation modeling (pixel space).

### Loss & Training

JEPA and VideoMAE are pretrained independently on each system for 6 epochs. MPP uses publicly released pretrained weights followed by end-to-end fine-tuning (as the three target datasets were not included in its original pretraining). DISCO is pretrained on The Well dataset. All models are fine-tuned for 100 epochs using AdamW with a cosine learning rate schedule.

## Key Experimental Results

### Main Results

| Method | Active Matter MSE↓ | Shear Flow MSE↓ | RB Convection MSE↓ |
|--------|--------------------|-----------------|-------------------|
| **JEPA** | **0.079** | **0.38** | **0.13** |
| VideoMAE | 0.160 | 0.67 | 0.18 |
| DISCO | 0.057 | 0.13 | 0.01 |
| MPP (end-to-end fine-tuning) | 0.230 | 0.59 | 0.08 |

### Data Efficiency Experiment (Shear Flow)

| Fine-tuning Data Fraction | JEPA | VideoMAE |
|--------------------------|------|---------|
| 10% (~3.2k) | 0.57 | 0.98 |
| 50% (~16k) | 0.40 | 0.75 |
| 100% (~32k) | 0.38 | 0.67 |

### Key Findings

- **JEPA consistently outperforms VideoMAE**: Relative improvements of 51% (active matter), 43% (shear flow), and 28% (RB convection) confirm that latent-space prediction preserves physical information better than pixel reconstruction.
- **JEPA is highly data-efficient**: With only 10% of fine-tuning data (~3.2k samples), JEPA achieves an MSE of 0.57, already surpassing VideoMAE trained on 100% of the data (0.67), indicating higher physical information density in JEPA's representations.
- **Latent-space methods consistently outperform pixel-level methods**: DISCO (latent-space operator learning) and JEPA (latent-space prediction) are the strongest methods in their respective categories, while MPP (pixel-level autoregressive) and VideoMAE (pixel-level reconstruction) are comparatively weaker — analogous to the BERT vs. GPT comparison in NLP on non-generative tasks.
- **Physics-specific methods are not always optimal**: Despite being designed specifically for physical modeling and benefiting from end-to-end fine-tuning, MPP underperforms frozen-encoder JEPA with an attentive probe on two systems, suggesting that autoregressive pixel prediction objectives may be misaligned with downstream physical understanding tasks.
- **System-specific behavior exists across methods**: DISCO performs exceptionally well on RB convection (0.01), while JEPA's relative advantage over VideoMAE is smallest on that system (0.13 vs. 0.18), suggesting that different physical systems may favor different inductive biases.

## Highlights & Insights

- **A shift in evaluation paradigm**: Moving from "predicting future frames" to "estimating physical parameters" as the criterion for representation quality is a perspective shift with far-reaching implications for scientific machine learning — yielding the key insight that pixel prediction accuracy does not equal physical understanding.
- **Latent-space prediction as a superior paradigm for physical representation learning**: Paradoxically, by forgoing pixel-level accuracy, JEPA learns better physical representations. A plausible explanation is that pixel-level objectives force the model to allocate capacity to encoding visual texture details, diluting the representation of high-level dynamical structures (e.g., convection patterns, vortex formation). Latent-space prediction bypasses pixel details and directs the model to focus on "what is necessary to predict the future" — which is precisely what correlates with physical parameters.
- **The three-component design of VICReg for collapse prevention**: The combination of variance regularization (preventing dimensional collapse), covariance regularization (preventing dimensional redundancy), and invariance constraint (aligning predictions with targets) provides stable training signals for JEPA.

## Limitations & Future Work

- **Limited evaluation systems**: Only three 2D PDE systems are considered; more complex scenarios such as 3D turbulence and multi-physics coupled systems are not addressed.
- **No controlled comparison between JEPA and DISCO**: DISCO incorporates physical inductive biases (operator learning framework), whereas JEPA is entirely general-purpose. Incorporating physical inductive biases into JEPA (e.g., physics-constrained losses) may further close the gap with DISCO.
- **Joint pretraining not explored**: All JEPA and VideoMAE models are pretrained independently on individual systems; the effect of cross-system joint pretraining (analogous to a foundation model approach) remains unknown.
- **Single downstream task**: Only parameter estimation is evaluated; other scientific tasks such as qualitative prediction (e.g., detecting laminar-to-turbulent transitions) and anomaly detection are not considered.
- **Architectural heterogeneity**: JEPA uses ConvNeXt while VideoMAE uses ViT-Small — architectural differences may confound the conclusions, necessitating controlled comparisons under the same architecture.

## Related Work & Insights

- **vs. VideoMAE**: VideoMAE reconstructs in pixel space, retaining abundant low-level visual information at the cost of diluting physical semantics. JEPA predicts in latent space, filtering out pixel details to preserve purer physical structure.
- **vs. MPP (autoregressive physics foundation model)**: Despite large-scale physics pretraining, MPP's autoregressive objective generalizes less effectively to parameter estimation than JEPA, echoing the BERT vs. GPT comparison on understanding tasks in NLP.
- **vs. DISCO (operator learning)**: DISCO achieves the strongest results overall but requires physical inductive biases. As a fully general-purpose method, JEPA approaches DISCO's performance, suggesting that the latent-space prediction paradigm itself may already capture part of the operator structure.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic comparison of self-supervised paradigms for physical parameter estimation
- Experimental Thoroughness: ⭐⭐⭐ Three systems and four methods, but evaluation tasks are limited
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation with insightful conclusions
- Value: ⭐⭐⭐⭐ Provides important guidance for the choice of representation learning paradigms in scientific machine learning

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)

<!-- RELATED:END -->
