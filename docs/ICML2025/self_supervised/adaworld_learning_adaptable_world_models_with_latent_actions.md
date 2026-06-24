---
title: >-
  [Paper Note] AdaWorld: Learning Adaptable World Models with Latent Actions
description: >-
  [ICML 2025][Self-Supervised Learning][World Models] AdaWorld is proposed, which builds highly adaptable world models by performing action-aware pre-training through self-supervised extraction of latent actions from videos, supporting zero-shot action transfer and fast adaptation to new environments with few interactions.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "World Models"
  - "Latent Actions"
  - "Self-Supervised"
  - "Video Pre-training"
  - "Action Transfer"
date: 2026-05-08
content_hash: cea4b5002d8c69bd
---

# AdaWorld: Learning Adaptable World Models with Latent Actions

**Conference**: ICML 2025  
**arXiv**: [2503.18938](https://arxiv.org/abs/2503.18938)  
**Code**: [https://adaptable-world-model.github.io](https://adaptable-world-model.github.io)  
**Area**: Self-Supervised Learning  
**Keywords**: World Models, Latent Actions, Self-Supervised, Video Pre-training, Action Transfer

## TL;DR
AdaWorld is proposed, which builds highly adaptable world models by performing action-aware pre-training through self-supervised extraction of latent actions from videos, supporting zero-shot action transfer and fast adaptation to new environments with few interactions.

## Background & Motivation

**Background**: World models aim to learn action-controlled future prediction, which is crucial for agent development. Existing methods rely on massive action-annotated data and expensive training to achieve action controllability.

**Limitations of Prior Work**: (a) Action formats vary across different environments, making it difficult to define a unified format; (b) adapting to new environments requires re-collecting massive action labels and retraining; (c) world models pre-trained solely on action-free videos lack action controllability.

**Key Challenge**: How to introduce action information during the pre-training phase without relying on explicit action labels?

**Goal**: Build world models that can rapidly adapt to new environments.

**Key Insight**: Extract latent actions from video frame pairs in a self-supervised manner—using an information bottleneck to force the encoder to retain only the most critical transitions between frames (i.e., action information) while removing context (such as color, texture).

**Core Idea**: Latent actions are context-independent and can transfer across environments—enabling action transfer to new scenes given just a single demonstration.

## Method

### Overall Architecture
Two components:
1. **Latent Action Autoencoder**: Extracts latent actions from unlabeled videos.
2. **Autoregressive World Model**: Predicts the next frame conditioned on latent actions.

### Key Designs

1. **Latent Action Autoencoder**:

    - **Function**: Extracts compact latent actions $\tilde{a}$ from two consecutive frames $f_t, f_{t+1}$.
    - **Mechanism**: The encoder uses a spatiotemporal Transformer to extract latent actions from frame pairs, and the decoder predicts $f_{t+1}$ based on $\tilde{a}$ and $f_t$. An information bottleneck from $\beta$-VAE is employed to force $\tilde{a}$ to encode only the most critical transitions between frames.
    - **Design Motivation**: The information bottleneck allows latent actions to be automatically decoupled from the context—making latent actions extremely compact compared to the immense dimensionality of pixels.

2. **Action-aware Pre-training**:

    - **Function**: Uses a world model based on Stable Video Diffusion to predict the next frame conditioned on latent actions.
    - **Mechanism**: Concatenates latent actions with timestep embeddings and CLIP image embeddings as conditions for the diffusion model.
    - **Design Motivation**: By learning that "different latent actions lead to different state transitions" during pre-training, adapting to a new environment only requires finding the action mapping.

3. **Adaptation Mechanism**:

    - Zero-shot transfer: Extract latent actions from a demonstration video and reuse them in a new scene.
    - Few-interaction adaptation: When action labels are available, find the mapping using the latent action encoder, requiring only minimal fine-tuning.

### Loss & Training
- Latent action autoencoder: $\beta$-VAE objective (reconstruction + KL divergence).
- World model: EDM diffusion loss + noise augmentation (to mitigate long-term drift).
- Pre-trained on large-scale diverse video datasets.

## Key Experimental Results

### Main Results

| Task | AdaWorld | Action-free Pre-training Baseline | Gain |
|------|---------|---------------|------|
| Action Transfer (cross-scene) | Feasible | Infeasible | Qualitative breakthrough |
| 50-interaction adaptation | FVD 85.2 | FVD 142.3 | 40%↓ |
| Visual planning success rate | 72.4% | 48.1% | +24.3% |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| No latent actions (action-free pre-training) | Poor | Lacks action controllability |
| Discrete latent actions | Insufficient expressiveness | Unable to compose actions |
| Continuous latent actions (Ours) | Optimal | Supports interpolation and composition |
| Large $\beta$ | Good decoupling but weak expressiveness | Trade-off |
| Small $\beta$ | Strong expressiveness but poor decoupling | Trade-off |

### Key Findings
- Latent actions are context-independent—the same action can be transferred from one scene to completely different ones.
- Continuous latent space supports action composition (averaging two actions produces a composite effect).
- Adaptation with only 50 interactions significantly outperforms training from scratch.

## Highlights & Insights
- The design of **Information Bottleneck = Action Decoupling** is elegant—leveraging the compression properties of VAEs to automatically separate actions from context.
- The continuous space of latent actions supports semantic interpolation and composition, implying a more generalized representation of actions.
- Integration with SVD allows the model to inherit powerful video-generation priors.

## Limitations & Future Work
- Latent actions only capture local changes between two frames, limiting long-term planning capabilities.
- Pre-training data mainly comes from automatically generated game environments, and generalization to real-world scenarios remains to be validated.
- The selection of $\beta$ requires manual tuning.

## Related Work & Insights
- **vs UniSim/GameGen**: Require action labels for training, whereas AdaWorld extracts them in a self-supervised manner.
- **vs GAIA-1**: Requires massive annotated data, whereas AdaWorld automatically learns from videos.
- Provides valuable insights for generalized world models and embodied AI research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The latent action pre-training paradigm is novel and powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-environments, action transfer, and planning tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Exquisite diagrams, clear methodological description.
- Value: ⭐⭐⭐⭐⭐ An important breakthrough in the adaptability of world models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[CVPR 2025\] CheXWorld: Image World Modeling for Radiograph Representation Learning](../../CVPR2025/self_supervised/chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)
- [\[CVPR 2025\] OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad](../../CVPR2025/self_supervised/ocrt_boosting_foundation_models_in_the_open_world_with_object-concept-relation_t.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](../../ICML2026/self_supervised/numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)
- [\[ICML 2025\] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)

</div>

<!-- RELATED:END -->
