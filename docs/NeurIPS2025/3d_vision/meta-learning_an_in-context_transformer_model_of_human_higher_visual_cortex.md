---
title: >-
  [Paper Note] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex
description: >-
  [NeurIPS 2025][3D Vision][fMRI encoding models] This paper proposes BraInCoRL (Brain In-Context Representation Learning), a Transformer-based meta-learning framework that predicts voxel-level neural responses for new subjects directly from a small number of stimulus–response samples via in-context learning (ICL), requiring no fine-tuning to generalize to new subjects or stimuli. With only 100 images as context, it approaches the performance of a reference model fully trained on 9,000 images.
tags:
  - NeurIPS 2025
  - 3D Vision
  - fMRI encoding models
  - meta-learning
  - in-context learning
  - higher visual cortex
  - hypernetworks
date: 2026-05-08
content_hash: 79dc3db1e65b6a54
---

# Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex

**Conference**: NeurIPS 2025
**arXiv**: [2505.15813](https://arxiv.org/abs/2505.15813)
**Code**: [GitHub](https://github.com/leomqyu/BraInCoRL)
**Area**: 3D Vision
**Keywords**: fMRI encoding models, meta-learning, in-context learning, higher visual cortex, hypernetworks

## TL;DR

This paper proposes BraInCoRL (Brain In-Context Representation Learning), a Transformer-based meta-learning framework that predicts voxel-level neural responses for new subjects directly from a small number of stimulus–response samples via in-context learning (ICL), requiring no fine-tuning to generalize to new subjects or stimuli. With only 100 images as context, it approaches the performance of a reference model fully trained on 9,000 images.

## Background & Motivation

Understanding functional representations in the higher visual cortex is a central problem in computational neuroscience. Existing visual cortex encoding models typically regress pretrained deep features linearly onto subject-specific voxel responses, but face critical bottlenecks:

**Data acquisition bottleneck**: Fitting an encoding model for each subject requires hours of expensive fMRI scanning (e.g., the NSD dataset requires ~10,000 images), and collecting large amounts of data in clinical settings is often infeasible.

**Individual differences**: Although the coarse-scale functional organization of the visual cortex is consistent across individuals (e.g., face-selective region FFA), there are significant individual differences at fine scales in anatomical location, spatial extent, and response characteristics.

**Difficulty in cross-subject generalization**: Existing methods (e.g., the auto-decoder Transformer of Adeli et al.) leverage multi-subject data but still require fine-tuning on new subjects.

**Core Idea**: Inspired by the in-context learning (ICL) capabilities of language models, the paper frames voxel encoding as a function inference problem—given a small number of stimulus–response pairs from a new individual, a voxel encoding model is constructed directly without any gradient updates.

## Method

### Overall Architecture

BraInCoRL treats the response function of each voxel as a meta-training task. During training, voxels are randomly sampled from multiple subjects, and a Transformer learns shared functional principles across subjects. At inference time, a small number of context samples from a new subject are used to directly generate voxel encoder parameters (hypernetwork paradigm).

The training pipeline consists of three stages: (1) **Pre-training**—training on synthetic voxel data with a fixed context of 500 images; (2) **Context extension**—context size is randomly sampled from $\text{Uniform}(30, 500)$ to obtain length robustness; (3) **Supervised fine-tuning**—training on real fMRI data.

### Key Designs

1. **Voxel-level meta-learning**: Unlike conventional approaches that model the entire visual cortex as a unit, BraInCoRL operates at the level of individual voxels. For each voxel in the support set, a context token is defined as $c_i = [x_i; \beta_i]$ (concatenation of image embedding and voxel response). The Transformer $T$ directly outputs encoder parameters $\omega$:

$$\omega = T_\theta(c_1, c_2, \ldots, c_p), \quad \hat{\beta} = f_\omega(\mathcal{I})$$

Optimization objective: $\theta^* = \arg\min_\theta \mathbb{E}_{(I_q, \beta_q)} \|f_\omega(\mathcal{I}) - \beta_{\text{True}}\|_2^2$

**Design Motivation**: Voxel-level granularity naturally handles varying numbers of voxels across subjects and requires no assumption of overlapping stimuli across subjects.

2. **Test-time context scaling (Logit Scaling)**: To handle context sequences of varying length at inference time, log-scaled attention is employed:

$$\alpha_{\text{scaled}} = \frac{\log(l) \cdot q \cdot k}{\sqrt{d_k}}$$

where $l$ is the context length. Combined with randomly sampled context sizes during training, this enables robust inference over arbitrary context lengths.

3. **Hypernetwork-generated linear encoder**: Following the convention of linear encoding models in neuroscience, the final voxel response prediction is a simple linear mapping: $\hat{\beta} = f(\phi(\mathcal{I}); \omega) = \text{matmul}(x, \omega)$. The $\omega$ generated by the Transformer serves as the linear weight vector, preserving interpretability while benefiting from the data efficiency of meta-learning.

4. **Network architecture details**: 20-layer self-attention encoder (SwiGLU activation + pre-normalization), 10 attention heads. The CLIP version ($E=512$) has ~97.2M parameters, DINO ($E=768$) ~112M, and SIGLIP ($E=1152$) ~130M. The [CLS] token is passed through an MLP to produce the final hypernetwork weights.

### Loss & Training

- MSE loss + AdamW optimizer (initial learning rate $10^{-3}$, decayed to $10^{-5}$)
- ReduceLROnPlateau scheduler (factor 0.1, patience 5, cooldown 2)
- Pre-training employs a data-by-synthesis approach (synthetic weights + Gaussian noise)
- Test subject data is strictly held out during fine-tuning
- Batch size 80, early stopping with patience of 5 epochs

## Key Experimental Results

### Main Results (NSD dataset, explained variance across 5 category-selective regions)

| Method | Faces (S1/S2) | Places (S1/S2) | Bodies (S1/S2) | Words (S1/S2) | Food (S1/S2) | Mean (S1/S2) |
|--------|--------------|----------------|----------------|---------------|-------------|-------------|
| Fully Trained (9,000 images) | 0.19/0.16 | 0.20/0.27 | 0.28/0.24 | 0.11/0.11 | 0.16/0.17 | 0.18/0.19 |
| Ridge-100 | 0.10/0.07 | 0.08/0.14 | 0.16/0.12 | 0.02/0.03 | 0.05/0.07 | 0.07/0.08 |
| Ridge-300 | 0.13/0.10 | 0.13/0.20 | 0.22/0.16 | 0.06/0.06 | 0.10/0.11 | 0.11/0.12 |
| FsAverage map | 0.13/0.06 | 0.11/0.19 | 0.09/0.08 | 0.06/0.03 | 0.14/0.18 | 0.08/0.06 |
| **BraInCoRL-100** | **0.16/0.13** | **0.16/0.23** | **0.25/0.21** | **0.07/0.08** | **0.12/0.13** | **0.13/0.15** |

### Ablation Study / Cross-Dataset Experiments

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| BraInCoRL (test images held out) | Mean EV ≈ 0.15 | Default setting |
| BraInCoRL (test images not held out) | Slightly higher performance | No overfitting risk |
| Pre-training only (no fMRI fine-tuning) | Significant performance drop | Fine-tuning on real fMRI data is important |
| BOLD5000 cross-dataset generalization | Outperforms Ridge regression | Different scanner (3T vs. 7T), different parameters |
| Text-prompt classification accuracy | Bodies 63%/54%, Places 81%/88% | Language-driven cortical functional mapping is feasible |

### Key Findings

- **Exceptional data efficiency**: With only 100 context images, BraInCoRL approaches ~72–89% of the performance of the fully trained reference model (9,000 images), yielding ~87% improvement over Ridge-100.
- **Strong test-time scaling behavior**: Performance improves steadily as context size increases from 0 to 1,000 images, approaching the fully trained reference.
- **Cross-dataset generalization**: A model trained on NSD (7T) generalizes directly to BOLD5000 (3T) with different subjects, different stimulus presentation times, and different trial structures.
- UMAP dimensionality reduction of BraInCoRL-predicted weights reveals semantic clusters consistent with known visual areas (EBA, FFA, RSC, OPA, PPA, etc.).
- **Counterintuitive finding**: Overly category-specific support sets degrade performance; randomly sampled, diverse images yield better results.

## Highlights & Insights

- **Paradigm shift**: The paper transitions from "collect large amounts of data per subject → fit a linear model" to "cross-subject meta-learning → few-shot in-context adaptation," with the potential to substantially reduce data requirements in neuroimaging research.
- **Language-driven cortical mapping**: CLIP features combined with BraInCoRL enable zero-shot mapping from natural language queries to voxel selectivity, offering a powerful tool for understanding the semantic organization of the visual cortex.
- **Elegant three-stage training strategy**: Meta-learning capabilities are first acquired on synthetic data, then adapted to real data, effectively addressing the scarcity of large-scale fMRI datasets.

## Limitations & Future Work

- The current framework handles only static natural images; extending to dynamic stimuli (video) requires redesigning the encoder backbone.
- Training relies primarily on NSD—currently the largest 7T fMRI dataset—and dataset diversity may still be insufficient.
- Voxel-level modeling ignores spatial relationships among voxels and does not exploit the topological structure of the cortex.
- Prediction accuracy for word-selective regions is relatively low, possibly due to stronger individual differences in this region.

## Related Work & Insights

- The work has deep connections to in-context learning in large language models, serving as neuroscientific evidence for the hypothesis that "Transformers implicitly learn meta-learning algorithms."
- It applies the hypernetwork paradigm in meta-learning—outputting function parameters rather than predictions directly.
- The approach has potential applications in personalized adaptation for brain–computer interfaces (BCIs) and neural prosthetics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The fusion of ICL, meta-learning, and hypernetworks applied to neuroscience is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated on two datasets (NSD + BOLD5000), multiple backbones (CLIP/DINO/SIGLIP), with thorough cross-domain generalization and semantic analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Addresses a core data efficiency bottleneck in neuroimaging with strong cross-disciplinary implications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MaNGO: Adaptable Graph Network Simulators via Meta-Learning](mango_-_adaptable_graph_network_simulators_via_meta-learning.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](../../CVPR2026/3d_vision/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[NeurIPS 2025\] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)
- [\[NeurIPS 2025\] Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model](galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)

<!-- RELATED:END -->
