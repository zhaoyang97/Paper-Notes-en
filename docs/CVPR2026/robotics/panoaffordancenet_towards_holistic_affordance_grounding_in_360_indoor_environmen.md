---
title: >-
  [Paper Note] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments
description: >-
  [CVPR2026][Robotics][affordance grounding] This paper presents PanoAffordanceNet, the first holistic affordance grounding framework for 360° panoramic indoor environments. It systematically addresses ERP geometric distortion, sparse functional regions, and semantic drift via a Distortion-Aware Spectral Modulator (DASM) and an Omnidirectional Spherical Densification Head (OSDH), and introduces 360-AGD, the first panoramic affordance dataset.
tags:
  - CVPR2026
  - Robotics
  - affordance grounding
  - 360° panoramic
  - equirectangular projection
  - one-shot learning
  - embodied intelligence
date: 2026-05-08
content_hash: 72b39da35c8846aa
---

# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**Conference**: CVPR2026
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)
**Code**: [GitHub](https://github.com/GL-ZHU925/PanoAffordanceNet)
**Area**: Robotics
**Keywords**: affordance grounding, 360° panoramic, equirectangular projection, one-shot learning, embodied intelligence

## TL;DR

This paper presents PanoAffordanceNet, the first holistic affordance grounding framework for 360° panoramic indoor environments. It systematically addresses ERP geometric distortion, sparse functional regions, and semantic drift via a Distortion-Aware Spectral Modulator (DASM) and an Omnidirectional Spherical Densification Head (OSDH), and introduces 360-AGD, the first panoramic affordance dataset.

## Background & Motivation

### State of the Field

Existing visual affordance research predominantly follows an **object-centric modeling paradigm**, reasoning about functional properties from constrained perspective views. However, embodied agents such as service robots inherently operate in a 360° omnidirectional physical space, creating a severe mismatch between the restricted viewpoint assumption and the agent's full action space.

### Limitations of Prior Work

Directly extending existing methods to panoramic scenes introduces three unique challenges:

**ERP Geometric Distortion**: Equirectangular Projection introduces severe latitude-dependent geometric distortion, particularly near the poles, making it difficult for models to simultaneously preserve local interaction details and global functional structure.

**Semantic Dispersion and Sparse Activation**: Non-uniform sampling causes functional regions to be highly sparsely distributed, with scattered initial activations that are difficult to aggregate into semantically coherent, boundary-consistent affordance regions.

**Cross-Scale Alignment Difficulty**: The absence of dense pixel-level annotations makes it extremely challenging to precisely align abstract affordance semantics with multi-scale regions in complex 360° scenes, often inducing semantic drift.

### Paper Goals

This paper advocates shifting the research focus from isolated object-level affordance understanding to **panoramic scene-level functional reasoning**, proposing the new task of "holistic affordance grounding in 360° indoor environments." This paradigm shift is critical for global decision-making and task planning by embodied agents in real-world scenarios.

## Method

### Overall Architecture

PanoAffordanceNet is an end-to-end one-shot learning framework consisting of four core modules:

1. **Dual-Encoder Feature Extraction**: LoRA-based parameter-efficient adaptation for multimodal representation learning.
2. **Distortion-Aware Spectral Modulator (DASM)**: Isolates task-relevant geometric signals via dual-spectral distillation.
3. **Spherical-Aware Hierarchical Decoder**: Incorporates the Omnidirectional Spherical Densification Head (OSDH) to recover topologically continuous regions from sparse activations.
4. **Multi-Level Training Objectives**: Combines pixel-level, distribution-level, and region-text contrastive constraints.

### Feature Extraction

- **Visual Encoder**: DINOv2 (ViT-B/14) extracts patch-level features $\mathbf{F}_v \in \mathbb{R}^{B \times L \times D}$; LoRA low-rank matrices are inserted into Transformer attention layers to adapt to ERP distortion while preventing overfitting.
- **Text Encoder**: A pretrained CLIP text encoder (ViT-B/16) combined with a CoOp prompt learner generates context-aware text embeddings $\mathbf{F}_t \in \mathbb{R}^{B \times C \times D}$.

### Key Designs

#### Key Design 1: Distortion-Aware Spectral Modulator (DASM)

The core idea of DASM is latitude-adaptive distortion compensation in the frequency domain:

**Step 1 – Cross-Modal Semantic Injection**: Text-guided information is injected into visual features via multi-head attention:

$$\mathbf{F}'_v = \text{Softmax}\Big(\frac{(\mathbf{F}_v \mathbf{W}_Q)(\mathbf{F}_t \mathbf{W}_K)^\top}{\sqrt{d}}\Big)(\mathbf{F}_t \mathbf{W}_V)$$

**Step 2 – Dual-Frequency Decomposition**: Features are decomposed into high- and low-frequency components:
- High-frequency: $\mathbf{F}_h = \nabla^2 * \mathbf{F}'_v$ (Laplacian operator extracts boundary/interaction contours)
- Low-frequency: $\mathbf{F}_l = \mathcal{K}_\sigma * \mathbf{F}'_v$ (Gaussian smoothing captures global structure)

**Step 3 – Targeted Frequency Compensation**:
- **High-Frequency Enhancement Module (HFEM)**: Enhances interaction boundaries in equatorial regions while suppressing pole-amplified artifacts.
- **Low-Frequency Stabilization Module (LFSM)**: Maintains global structural consistency in polar regions, alleviating semantic fragmentation caused by stretching.

**Step 4 – Mixed Gated Fusion**: A language-driven channel gate $\mathbf{g}_{ch}$ emphasizes task-relevant semantics, while an adaptive spatial gate $\mathbf{g}_{sp}$ anchors salient regions:

$$\mathbf{F}_{freq} = \mathbf{F}'_v + \sum_{k \in \{h,l\}} \lambda_k (\mathbf{g}_{ch} \odot \mathbf{g}_{sp} \odot \mathbf{F}_k)$$

#### Key Design 2: Omnidirectional Spherical Densification Head (OSDH)

OSDH addresses the problem of sparse and fragmented affordance signals in panoramic scenes, using visual self-similarity as a structural inductive bias:

1. **Global Semantic Discovery**: A lightweight Transformer decoder uses text embeddings as queries to cross-attend to visual features, producing an initial affordance map $\mathbf{A}_{init}$.
2. **Spherical Affinity Matrix Construction**: Visual features are projected onto a unit hypersphere, and a symmetric affinity matrix $\mathcal{S} \in \mathbb{R}^{L \times L}$ is constructed via cosine similarity.
3. **Confidence-Guided Noise Suppression**: High-confidence seed points are selected via top-k selection, and spurious noise is filtered using a normalized Sigmoid function.
4. **Seed Propagation Densification**: Complete functional regions are recovered through seed activation propagation:

$$\mathbf{A}_{refined} = \mathbf{A}_{init} + \alpha \cdot \max_{j \in \mathcal{K}} (\mathcal{S}_{ij} \cdot \mathcal{C}_j)$$

### Loss & Training

The total training objective consists of three levels of constraints:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{BCE} + \lambda_2 \mathcal{L}_{KL} + \lambda_3 \mathcal{L}_{RTC}$$

| Loss Component | Level | Function |
|---|---|---|
| $\mathcal{L}_{BCE}$ | Pixel-level | Binary cross-entropy for precise activation localization |
| $\mathcal{L}_{KL}$ | Distribution-level | KL divergence to maintain panoramic topological continuity and global shape consistency |
| $\mathcal{L}_{RTC}$ | Semantic-level | InfoNCE region-text contrastive loss to establish semantic correspondence between visual regions and affordance concepts |

**Training Details**: AdamW optimizer, learning rate 1e-5 with cosine annealing, 2×A6000 GPUs, 20k iterations, batch size 4, input resolution 560×1120. Panorama-specific data augmentation includes ±3° random rotation, ±5% scaling, and horizontal wrap-around translation.

## Key Experimental Results

### 360-AGD Dataset

This paper constructs 360-AGD, the first panoramic affordance grounding dataset for indoor environments. It covers 19 affordance categories, divided into an Easy Split (sourced from 360-Indoor and Gibson, 512×1024) and a Hard Split (sourced from PanoContext and Sun360, up to 4552×9104). A keypoint-based annotation strategy is used to generate Gaussian heatmaps.

### Main Results: 360-AGD and AGD20K Comparison

| Method | Supervision | 360-AGD Easy KLD↓ | Easy SIM↑ | Easy NSS↑ | 360-AGD Hard KLD↓ | Hard SIM↑ | Hard NSS↑ |
|---|---|---|---|---|---|---|---|
| OOAL | One-shot | 2.868 | 0.117 | 1.267 | 3.067 | 0.097 | 1.484 |
| OS-AGDO | One-shot | 2.853 | 0.124 | 1.299 | 2.965 | 0.115 | 1.484 |
| **PanoAffordanceNet** | **One-shot** | **1.270** | **0.506** | **4.490** | **1.306** | **0.474** | **4.398** |

The method also remains competitive on the conventional perspective-view AGD20K benchmark, achieving KLD of 0.739 and SIM of 0.616 on the Seen Split, surpassing or matching OOAL.

### Ablation Study

**Module Ablation (Hard Split):**

| LoRA | DASM | OSDH | KLD↓ | SIM↑ | NSS↑ |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 1.475 | 0.416 | 4.196 |
| ✓ | ✗ | ✗ | 1.421 | 0.429 | 4.257 |
| ✓ | ✓ | ✗ | 1.380 | 0.450 | 4.317 |
| ✓ | ✗ | ✓ | 1.359 | 0.448 | 4.339 |
| ✓ | ✓ | ✓ | **1.306** | **0.474** | **4.398** |

**Loss Function Ablation (Hard Split):**

| $\mathcal{L}_{KL}$ | $\mathcal{L}_{RTC}$ | $\mathcal{L}_{BCE}$ | KLD↓ | SIM↑ | NSS↑ |
|---|---|---|---|---|---|
| ✓ | ✗ | ✗ | 1.596 | 0.395 | 3.891 |
| ✓ | ✓ | ✗ | 1.459 | 0.442 | 4.374 |
| ✓ | ✗ | ✓ | 1.430 | 0.450 | 4.041 |
| ✓ | ✓ | ✗ | 1.331 | 0.493 | 4.361 |
| ✓ | ✓ | ✓ | **1.306** | **0.474** | **4.398** |

### Key Findings

1. **Overwhelming Advantage**: PanoAffordanceNet comprehensively outperforms existing methods on 360-AGD — SIM improves from 0.124 to 0.506 (>4×) and NSS from 1.299 to 4.490 (3.5×).
2. **Complementary Module Gains**: LoRA, DASM, and OSDH each contribute incremental improvements; their combination yields the best performance.
3. **Critical Role of DASM**: DASM significantly reduces KLD error and is the key component for addressing ERP distortion.
4. **Hyperparameter Robustness**: KLD fluctuates by only 0.006 when top-k varies within [5, 20]; LoRA rank=16 is the optimal balance point.
5. **Cross-Domain Generalization**: The method remains highly competitive on the perspective-view AGD20K dataset.

## Highlights & Insights

1. **New Task Definition**: This paper is the first to define the task of panoramic affordance grounding in 360° indoor environments, advancing affordance research from the object level to the scene level, aligned with the practical needs of embodied agents.
2. **Novel Frequency-Domain Perspective**: DASM reformulates panoramic distortion as a frequency-domain problem, handling the distinct distortion patterns of equatorial and polar regions via separate high- and low-frequency processing — an elegant design.
3. **Self-Similarity-Driven Densification**: OSDH leverages visual feature self-similarity as an inductive bias to recover topological continuity through seed propagation, circumventing the need for dense annotations.
4. **Complete Dataset Contribution**: 360-AGD fills the gap in panoramic affordance data; the Easy/Hard split design enables systematic evaluation.
5. **Real-World Deployment Validation**: Real-scene testing with a head-mounted Insta360 X4 camera validates the practical applicability of the approach.

## Limitations & Future Work

1. **Static Scenes Only**: Temporal reasoning in dynamic scenes is not considered; the paper itself acknowledges plans to explore temporal reasoning in future work.
2. **Annotation Strategy Limitations**: Keypoint-based annotation may not precisely capture complete boundaries of complex affordance regions.
3. **One-Shot Ceiling**: Although the framework design is sophisticated, the one-shot learning setting inherently limits generalization to long-tail affordance categories.
4. **Limited Dataset Scale**: The scale and diversity of 360-AGD remain below those of mature datasets, and 19 affordance categories may be insufficient to cover real-world scenario demands.
5. **Lack of 3D Integration**: The paper acknowledges that panoramic images serve as an intermediate representation between 2D and 3D, but does not explore synergy with 3D spatial representations.

## Related Work & Insights

- **Evolution of Visual Affordance**: From fully supervised → weakly supervised (LOCATE, WSMA) → foundation model-driven (OOAL, AffordanceLLM); this paper opens a new dimension at the panoramic scene level.
- **Panoramic Perception Foundations**: The work inherits geometry-aware ideas from SphereNet and panoramic semantic segmentation, applying them specifically to affordance grounding.
- **Implications for Embodied Intelligence**: Panoramic affordance grounding provides critical functional priors for global robotic decision-making, with strong potential for integration with navigation and manipulation planning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — New task definition + new dataset + comprehensive technical solution
- Experimental Thoroughness: ⭐⭐⭐⭐ — Detailed ablations and hyperparameter analysis, though the number of baselines is limited (only 2)
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich illustrations, explicit problem-solution correspondence
- Value: ⭐⭐⭐⭐ — Opens a new track in panoramic affordance research; actual impact depends on community follow-up

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)
- [\[CVPR 2026\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)

<!-- RELATED:END -->
