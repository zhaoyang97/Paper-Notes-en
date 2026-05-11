---
title: >-
  [Paper Note] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
description: >-
  [Image Generation] > This paper proposes VLN-PE, the first physically realistic vision-and-language navigation platform supporting humanoid, quadruped…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: e3cfa257520c4a3c
---

# Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities

| Info | Content |
|------|------|
| Conference | ICCV 2025 |
| arXiv | [2507.13019](https://arxiv.org/abs/2507.13019) |
| Code | [Project Page](https://crystalsixone.github.io/vln_pe.github.io) |
| Area | Embodied Navigation · Vision-and-Language Navigation · Cross-Embodiment |
| Keywords | VLN, cross-embodiment, physical simulation, diffusion policy, benchmark |

## TL;DR

> This paper proposes VLN-PE, the first physically realistic vision-and-language navigation platform supporting humanoid, quadruped, and wheeled robots. It systematically evaluates existing VLN methods under real physical constraints, revealing a 34% drop in success rate when transferring from simulation to physical deployment.

## Background & Motivation

### The Gap Between Simulation and Physical Deployment

Vision-and-language navigation (VLN) has evolved from discrete graph traversal (R2R) to continuous navigation (VLN-CE), yet existing methods face critical challenges in physical deployment:

**Idealized Assumptions**: Most VLN benchmarks support only idealized wheeled or point-mass agents, ignoring the physical embodiment constraints of real robots.

**Overly Idealized Test Conditions**: Key physical phenomena such as viewpoint shifts, falls, deadlocks, and motion errors are neglected.

**Lack of Cross-Embodiment Evaluation**: No systematic analysis exists of how different robot types (humanoid, quadruped, wheeled) affect VLN method performance.

**Absence of Motion Control**: Existing platforms employ pseudo-motion via navigation meshes, which does not reflect real physical dynamics.

### Core Problem

> To what extent do physical embodiment constraints and visual environment variations affect the performance of existing VLN methods?

## Method

### Overall Architecture: The VLN-PE Platform

Built upon the GRUTopia physics simulator, supporting three robot categories:
- **Humanoid**: Unitree H1, G1
- **Quadruped**: Unitree Aliengo
- **Wheeled**: Jetbot

#### Scene Support
- 90 Matterport3D scenes (with floor gaps repaired to prevent legged robots from getting stuck)
- 10 high-quality synthetic household scenes (GRScenes)
- 3DGS-rendered laboratory environments

#### New Metrics
- **Fall Rate (FR)**: Frequency of unintended falls
- **Stuck Rate (StR)**: Frequency of the agent becoming immobile

### Three Categories of Evaluated Methods

#### 1. End-to-End Classification Methods (Single-Step Prediction)

**Seq2Seq**: LSTM for instruction encoding + ResNet50 for RGB/Depth encoding + GRU for action prediction.

$$h_t = \text{GRU}([V_t, D_t, I], h_{t-1}), \quad a_t = \arg\min_a \text{softmax}(W_a h_t + b_a)$$

**CMA**: Extends Seq2Seq with cross-modal attention, employing two GRUs to separately process visual observations and instruction-guided decision-making.

**NaVid**: A 7B-parameter video multimodal large language model built on LLaMa-VID, performing RGB-only navigation.

#### 2. Multi-Step Diffusion Policy (RDP) — Newly Proposed Baseline

This work is the first to apply diffusion policy to VLN. LongCLIP encodes RGB and instruction inputs; ResNet50 encodes depth; cross-modal attention fuses the representations, with a Transformer serving as the diffusion decoder:

$$a_t^{k-1} = \alpha \cdot (a_t^k - \gamma \epsilon_\theta(c_t, a_t^k, k) + \mathcal{N}(0, \mu^2 I))$$

Key novelty: a GRU maintains historical observations, and an auxiliary MLP predicts a stopping progress score (0→1).

$$\mathcal{L}_{\text{RDP}} = \text{MSE}(\epsilon^k, \epsilon_\theta(c_t, a_t^0 + \epsilon^k, k)) + \lambda \cdot \text{MSE}(\mathcal{S}_{\text{stop}}(c_t), \hat{p}_{\text{stop}})$$

#### 3. Training-Free Map-Based Method (Enhanced VLMaps)

An LLM parses instructions into sub-goal code; LSeg localizes targets on a semantic map. A VLFM frontier exploration strategy is added to handle cases where targets are not visible.

## Experiments

### Main Results: Transfer from VLN-CE to VLN-PE (R2R Dataset, Humanoid H1)

| Method | Setting | Val-Seen SR↑ | Val-Seen SPL↑ | Val-Unseen SR↑ | Val-Unseen SPL↑ |
|------|------|:---:|:---:|:---:|:---:|
| Seq2Seq-Full | Zero-shot transfer | 13.83 | 11.17 | 15.00 | 11.99 |
| CMA-Full | Zero-shot transfer | 15.50 | 14.00 | 16.04 | 14.63 |
| NaVid | Zero-shot transfer | 21.58 | 17.45 | 22.42 | 18.58 |
| CMA+ | VLN-PE fine-tuned | **28.72** | **24.24** | 23.31 | 19.66 |
| RDP | VLN-PE trained | 23.86 | 17.35 | 21.98 | 16.44 |

Key findings:
- Direct transfer from VLN-CE to VLN-PE results in an SR drop of approximately **34%**.
- NaVid (7B) achieves the best zero-shot performance, with substantially lower StR and FR than smaller models, suggesting that world knowledge aids obstacle avoidance.
- CMA-CLIP fine-tuned on only 441 VLN-PE trajectories surpasses NaVid's zero-shot performance.

### Cross-Scene Generalization (GRU-VLN10 + 3DGS-Lab)

| Method | GRU-VLN10 Unseen SR↑ | 3DGS-Lab SR↑ |
|------|:---:|:---:|
| NaVid (zero-shot) | 18.64 | 5.81 |
| CMA-CLIP (fine-tuned) | 22.46 | 24.88 |
| RDP (fine-tuned) | **28.52** | **30.63** |

NaVid **completely fails** in 3DGS scenes (SR of only 5.81%), likely because 3DGS rendering artifacts interfere with the large model's RGB perception.

### Effect of Lighting Conditions

| Lighting Condition | CMA (RGB-D) SR | CMA (RGB-only) SR |
|----------|:---:|:---:|
| High brightness (50k) | 21.74 | 14.72 |
| Very low brightness (1k) | 19.84 | 3.36 |

RGB-only models experience a sharp performance drop under low-light conditions, whereas RGB+Depth models demonstrate greater robustness.

## Highlights & Insights

1. **First Systematic Cross-Embodiment VLN Evaluation**: Reveals unique challenges posed by humanoid, quadruped, and wheeled robots in VLN, including differences in viewpoint height and legged locomotion constraints.
2. **First Application of Diffusion Policy to VLN**: The proposed RDP baseline demonstrates that dense trajectory prediction outperforms CMA/Seq2Seq when trained from scratch.
3. **The Large Model Paradox**: NaVid (7B) exhibits superior zero-shot capability and obstacle avoidance, yet completely fails in novel scenes (3DGS) and exhibits repetitive rotation near targets.
4. **The Power of Small-Data Fine-Tuning**: Fine-tuning on as few as 441 in-domain samples significantly surpasses the zero-shot performance of a 7B large model.

## Limitations & Future Work

- The RL-based locomotion controller cannot yet reliably handle stair navigation, requiring episodes containing stairs to be filtered out.
- Fall rates for legged robots (humanoid, quadruped) remain high in complex environments.
- The evaluation scene scale is limited (the 3DGS setting covers only one laboratory environment).
- The stopping decision in the RDP diffusion policy still requires an auxiliary MLP.

## Related Work & Insights

- **Discrete VLN**: Graph-based navigation benchmarks such as R2R and REVERIE.
- **Continuous VLN**: Habitat-based methods including VLN-CE, NaVid, and CMA.
- **Diffusion Policy**: Successful application of diffusion policy in robotic manipulation.
- **Cross-Embodiment Benchmarks**: GRUTopia, BEHAVIOR-100, ARIO, and related platforms.

## Rating

| Dimension | Score |
|------|:----:|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](../../NeurIPS2025/image_generation/dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)

</div>

<!-- RELATED:END -->
