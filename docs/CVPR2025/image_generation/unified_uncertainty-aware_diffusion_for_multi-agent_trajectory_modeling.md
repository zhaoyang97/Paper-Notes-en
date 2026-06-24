---
title: >-
  [Paper Note] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling
description: >-
  [CVPR 2025][Image Generation][diffusion model] U2Diff is proposed as a unified diffusion model framework capable of simultaneously handling multi-agent trajectory completion and prediction tasks. It provides state-wise uncertainty estimation through an augmented denoising loss and introduces a Rank Neural Network to rank the error probabilities of multi-modal predictions.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "diffusion model"
  - "trajectory completion"
  - "uncertainty estimation"
  - "multi-agent"
  - "error probability"
  - "sports analytics"
date: 2026-05-08
content_hash: 695fc487e31bfdcc
---

# Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling

**Conference**: CVPR 2025  
**arXiv**: [2503.18589](https://arxiv.org/abs/2503.18589)  
**Code**: None  
**Area**: Trajectory Prediction  
**Keywords**: diffusion model, trajectory completion, uncertainty estimation, multi-agent, error probability, sports analytics

## TL;DR

U2Diff is proposed as a unified diffusion model framework capable of simultaneously handling multi-agent trajectory completion and prediction tasks. It provides state-wise uncertainty estimation through an augmented denoising loss and introduces a Rank Neural Network to rank the error probabilities of multi-modal predictions.

## Background & Motivation

### Background

**Background**: Multi-agent trajectory modeling is crucial in domains such as sports analytics, autonomous driving, and surveillance. Existing research primarily focuses on trajectory prediction (predicting the future given the past), where diffusion models have recently become a mainstream solution due to their strong multi-modal modeling capabilities. However, trajectory completion—recovering complete trajectories when gaps exist at arbitrary positions—is also a major practical requirement (e.g., correcting occlusions and missed detections in tracking data).

**Limitations of Prior Work**: (1) Existing multi-agent trajectory models are typically restricted to a single task (either prediction or completion), failing to handle both within a unified framework; (2) Most prediction models only output point estimates without providing state-wise uncertainty metrics, meaning users cannot distinguish between reliable and unreliable predictions; (3) While multi-modal sampling generates multiple candidate trajectories, there is a lack of inference-time error probability estimation, making it impossible to effectively rank candidates and select the optimal prediction.

**Key Challenge**: Trajectory uncertainty possesses spatiotemporal heterogeneity (e.g., turning points are inherently more uncertain than straight segments), yet existing methods treat all states equally. Furthermore, although multi-modal generation covers possible future paths, it does not inform the user which mode is most likely to be correct.

**Goal**: To simultaneously achieve trajectory completion and prediction under a unified diffusion framework, provide state-wise uncertainty estimation, and offer error probability ranking for multi-modal generation results.

**Key Insight**: Leveraging the flexibility of diffusion models to handle arbitrary masking patterns for trajectory completion/prediction, augmenting denoising loss to extract latent space uncertainty and propagate it to the real state space, and employing a post-processing Rank Neural Network to estimate the error probability of each generated sample.

**Core Idea**: Embedding uncertainty into the diffusion process via a negative log-likelihood (NLL) augmented denoising loss, combined with a Rank Neural Network to achieve quality ranking of generated trajectories.

## Method

### Overall Architecture

U2Diff is based on a conditional diffusion model, taking partially observed trajectories of multiple agents as input (which can be the first $N$ frames for prediction, or arbitrary patterns of known frames for completion) and generating complete trajectories through iterative denoising. During training, masked trajectories are used to construct different completion/prediction tasks, achieving task unification. During inference, multiple candidate trajectories (multi-mode) can be sampled, which are then ranked by the Rank Neural Network to select the optimal one. The architecture is based on CSDI, using bidirectional MambaSSM instead of Transformer Encoder to enhance temporal processing capabilities.

### Key Designs

1. **Uncertainty-Aware Denoising Loss**: On top of the standard diffusion model's simple denoising loss (MSE of predicted noise), a Negative Log-Likelihood (NLL) term is added. The model predicts not only the denoising direction but also the variance representing noise (i.e., $\sigma$) on each state dimension. This allows the model to naturally learn during training which state points have more uncertain predictions—where $\sigma$ is larger for sharp dynamic changes like turns and acceleration. The NLL loss enables the model to tolerate larger prediction errors in highly uncertain regions while providing tighter estimates in certain regions, achieving adaptive, state-wise confidence.

2. **Uncertainty Propagation from Latent Space to State Space**: The uncertainty learned by the diffusion model in the latent space needs to be propagated to the final trajectory coordinate space. U2Diff utilizes analytical or approximate uncertainty propagation methods (similar to the Unscented Transform or first-order Taylor expansion) to pass the variance accumulated at each step of the denoising process to the output layer, ensuring that each finally predicted $(x,y)$ coordinate is accompanied by a standard deviation estimate. This provides an intuitive measure of reliability for downstream applications, where points with high uncertainty may require additional validation.

3. **Rank Neural Network (RankNet)**: After multi-modal sampling generates $K$ candidate trajectories, the one closest to the ground truth needs to be selected. RankNet takes each candidate trajectory concatenated with its uncertainty estimation as input to predict the error probability of that trajectory relative to the ground truth. During training, the error between actual generation results and the ground truth is used as supervision, allowing RankNet to learn the mapping between error probabilities and uncertainty patterns. Experiments demonstrate that the ranking produced by RankNet is highly correlated with the actual error ranking, enabling the effective selection of the optimal prediction during inference even without ground truth.

## Key Experimental Results

### Main Results

Comprehensive validation was conducted on four real-world sports datasets, including scene-level and agent-level metrics. The trajectory completion task used different missingness rates and masking patterns, while the trajectory prediction task used the standard observation-prediction split.

### Key Findings

- Outperformed SOTA on four sports trajectory datasets (**NBA**, **Basketball-U**, **Football-U**, **Soccer-U**).
- Trajectory Completion Task: U2Diff outperforms existing methods across different missingness rates, demonstrating the baseline's capability to handle completion in a unified framework.
- Trajectory Prediction Task: The identical model directly outperforms specifically designed prediction methods without requiring architecture switching.
- Quality of Uncertainty Estimation: Highly uncertain regions align closely with the actual error distribution—indicating the model successfully learns meaningful state-wise confidence.
- Strong correlation is observed between the error probabilities from RankNet and the ground truth errors (Spearman median of approximately 0.58 and 0.78), making mode selection at inference time possible for the first time.
- Ablation of NLL Loss: Removing the NLL term degrades the uncertainty estimation, leading to a drop in RankNet's ranking capability, which validates the coupled design of the two features.

## Highlights & Insights

- Genuinely achieves the unification of completion and prediction—elegantly incorporating both tasks into the same diffusion framework via a masking mechanism, avoiding the need to design specialized models for different tasks.
- Rank Neural Network addresses a critical gap in multi-modal trajectory prediction: previously, while multiple candidates were generated, no robust method existed to select the optimal one in the absence of ground truth.
- Uncertainty estimation is not an after-the-fact addition; instead, it is directly embedded into the training objective, organically integrating with the denoising process.
- Particularly well-suited for sports analytics scenarios, where athlete trajectory data frequently suffers from missingness due to occlusions/ID switches, making completion demands both real and urgent.
- The organic integration of the diffusion model's denoising process with uncertainty estimation represents the primary methodological innovation.

## Limitations & Future Work

- Current validation is concentrated on sports scenarios (fixed cameras, bird's-eye view), and the generalization to ego-centric scenarios such as autonomous driving remains to be verified.
- Multi-step denoising inference in diffusion models is slow, rendering it unsuitable for applications requiring real-time prediction (e.g., autonomous driving motion planning); exploring consistency distillation for acceleration is a viable direction.
- As a post-processing module, the Rank Neural Network requires additional training data and inference overhead; integrating the ranking capability directly into the diffusion model itself represents a potential future direction.
- Approximate methods in uncertainty propagation may introduce distortion in highly non-linear scenarios.
- The model currently only processes 2D trajectory positions; extending it to high-dimensional states containing velocity, acceleration, and heading requires adjusting the uncertainty propagation strategy.
- Current validation is concentrated on sports scenarios (fixed cameras, bird's-eye view), and the generalization to ego-centric scenarios such as autonomous driving remains to be verified.
- Multi-agent interaction modeling relies on the observational completeness of all agents, and robustness in partially occluded scenarios has not been evaluated.
- While the masking mechanism for trajectory completion and prediction is elegant, it is sensitive to the mask ratio, requiring different masking strategies for different scenarios.
- The scene-level ADE/FDE metrics on the NBA dataset both achieved SOTA, demonstrating the practical utility of the method on real-world data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](../../ICLR2026/image_generation/jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](../../ICCV2025/image_generation/aether_geometric-aware_unified_world_modeling.md)
- [\[CVPR 2025\] MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)

</div>

<!-- RELATED:END -->
