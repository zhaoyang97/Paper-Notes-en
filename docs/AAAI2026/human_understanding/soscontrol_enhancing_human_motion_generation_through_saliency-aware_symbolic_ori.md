---
title: >-
  [Paper Note] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control
description: >-
  [AAAI2026][Human Understanding][Human Motion Generation] This paper proposes the Salient Orientation Symbolic (SOS) script — a programmable symbolic motion representation framework inspired by Labanotation — that extract…
tags:
  - "AAAI2026"
  - "Human Understanding"
  - "Human Motion Generation"
  - "Symbolic Control"
  - "Labanotation"
  - "Diffusion Models"
  - "Saliency Detection"
  - "ControlNet"
date: 2026-05-08
content_hash: fa880af6d4bf9a38
---

# SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control

**Conference**: AAAI2026
**arXiv**: [2601.14258](https://arxiv.org/abs/2601.14258)
**Authors**: Ho Yin Au, Junkun Jiang, Jie Chen (The Hong Kong Polytechnic University)
**Code**: [GitHub](https://github.com/asdryau/SOSControl)
**Area**: Human Understanding
**Keywords**: Human Motion Generation, Symbolic Control, Labanotation, Diffusion Models, Saliency Detection, ControlNet

## TL;DR

This paper proposes the Salient Orientation Symbolic (SOS) script — a programmable symbolic motion representation framework inspired by Labanotation — that extracts keyframe saliency via temporally-constrained agglomerative clustering, and introduces an SMS data augmentation strategy along with a gradient-optimization-based SOSControl framework for precise control over body-part orientation and motion timing. On HumanML3D, the method achieves an SOS-Acc of 0.988 with an FID of only 3.892.

## Background & Motivation

Text-driven human motion generation has attracted significant attention due to its applications in film content creation, robotics, and human-robot collaboration. However, textual descriptions are inherently subjective and ambiguous — when a user requests "squat down and punch forward," conventional text-to-motion frameworks cannot precisely control arm orientation (straight punch vs. uppercut) or motion timing (when the punch reaches its peak).

Existing work on enhanced controllability has explored using **joint keyframe positions** as additional conditioning signals (e.g., OmniControl, GMD). However, joint-position control has fundamental limitations: (1) it provides only positional guidance without specifying body-part **orientation**; (2) models may misinterpret target positions as intermediate waypoints, causing motion "overshoot" that disrupts timing; (3) manually specifying physically plausible keyframe positions in 3D space requires extensive adjustment and deep knowledge of motion dynamics, making it impractical for industrial animation pipelines.

Labanotation is a widely used symbolic notation system in dance, employing directional symbols on a staff to annotate body-part movements. Inspired by this, the paper designs the SOS script for orientation and timing control, abstracting motion control from "precise 3D coordinates" to "directional symbol + temporal position." It further introduces a saliency-based automatic extraction pipeline, yielding sparse, interpretable, and programmable symbolic representations.

## Core Problem

How to design an intuitive, programmable motion control interface that enables users to symbolically specify body-part orientation and motion timing in a precise manner, while ensuring the generated motion is both natural and highly aligned with the control signals.

## Method

### Overall Architecture

SOSControl consists of two major components: (1) the SOS extraction pipeline, which automatically extracts saliency-aware symbolic scripts from motion data; and (2) the SOSControl generation framework, which performs diffusion-based motion generation conditioned on SOS scripts.

### Key Design 1: SOS Script and Automatic Extraction

**Orientation Feature Extraction**: The Pairwise Relative Position Phrase (PRPP) is used to compute the orientation feature for each body part:

$$\mathbf{o}_t^J = \text{PRPP}(e^J, a^J)_t = (\mathbf{l}_t(e^J) - \mathbf{l}_t(a^J)) \cdot \mathbf{r}_t$$

where $e^J$ is the end joint, $a^J$ is the anchor joint, and $\mathbf{r}_t$ is the egocentric reference direction.

**Spatial Feature Quantization**: Twenty-six unit direction vectors $\mathbf{u} \in \mathbb{R}^{26 \times 3}$ are defined, and discrete directional symbols are obtained via differentiable softmax quantization:

$$\mathbf{q} = \text{softmax}\left(\frac{\mathbf{o}}{\|\mathbf{o}\|} \cdot \mathbf{u}^T\right) \cdot \mathbf{u}$$

**Temporal Saliency Detection**: Temporally-constrained agglomerative clustering is applied to the rate of change of orientation features for each body part (a connectivity matrix restricts merging to adjacent temporal segments only), constructing a bottom-up segmentation tree. The merge distance at each node serves as the saliency value of the corresponding frame — a larger merge distance indicates more significant motion change around that frame.

**Saliency Masking Scheme (SMS)**: A threshold is applied to filter low-saliency frames, retaining only the directional symbols of high-saliency keyframes to produce a sparse, interpretable SOS staff.

### Key Design 2: Periodic Latent Space Diffusion

ACTOR-PAE is adopted to encode motion into periodic parameters (frequency $\mathbf{f}$, amplitude $\mathbf{a}$, bias $\mathbf{b}$, phase shift $\mathbf{s}$), generating periodic signals:

$$\mathbf{p} = \mathbf{a}\sin(\mathbf{f} \cdot (N - \mathbf{s})) + \mathbf{b}$$

An MDM-style diffusion model $\mathcal{D}^-$ is trained in this periodic latent space with the loss:

$$\mathcal{L}_{\mathcal{D}^-} = \|\mathbf{p}^0 - \mathcal{D}^-(k, \mathbf{c}, \mathbf{p}^k)\|_2$$

### Key Design 3: SOS-Guided Injection

**ControlNet Adaptation**: The ControlNet architecture is applied to both the diffusion model and the ACTOR-PAE decoder. Original model parameters are frozen, and trainable copies are trained to inject the SOS condition $\mathbf{d}$ into the generation process:

$$\mathcal{L}_{\mathcal{D}^+} = \|\mathbf{p}^0 - \mathcal{D}^+(k, \mathbf{c}, \mathbf{d}, \mathbf{p}^k)\|_2$$

**Gradient-Based Iterative Optimization**: Through the differentiable orientation feature extraction pipeline, gradient descent is applied to the periodic latent variable at test time:

$$\mathbf{p}^* = \mathbf{p}^* - \nabla_{\mathbf{P}}\|\mathcal{M}_{\mathbf{d}}(\hat{\mathbf{q}}) - \mathbf{d}\|_2$$

where $\mathcal{M}_{\mathbf{d}}$ is the SOS mask, computing orientation discrepancy only at visible salient regions.

**SMS Data Augmentation**: During training, a saliency threshold $m^{J} \sim \mathcal{U}(0,1)$ is randomly sampled for each body part to generate SOS scripts at varying granularities, enabling the model to accommodate control signals of arbitrary sparsity as provided by users.

## Key Experimental Results

### Main Results: SOS-Conditioned Motion Generation on HumanML3D

| Method | SOS-Acc↑ | L2-Rot6D↓ | FID↓ | MMD↓ |
|--------|---------|----------|------|------|
| MDM (baseline, no SOS) | 0.151 | 0.351 | 2.592 | 6.001 |
| GMD 1-stage | 0.113 | 0.427 | 25.669 | 7.835 |
| GMD 2-stage | 0.120 | 0.402 | 21.278 | 7.823 |
| OmniControl | 0.873 | 0.325 | 3.975 | 6.095 |
| TLControl | 0.982 | 0.341 | 11.132 | 7.066 |
| **SOSControl (Ours)** | **0.988** | **0.325** | **3.892** | **6.199** |

SOSControl achieves the best performance simultaneously on control signal alignment (SOS-Acc) and motion quality (FID). GMD-based methods perform poorly due to their inability to directly impute SOS into motion signals.

### Ablation Study: Contribution of Each Module

| Configuration | SOS-Acc↑ | L2-Rot6D↓ | FID↓ | MMD↓ |
|---------------|---------|----------|------|------|
| w/o SMS data proc. | 0.991 | 0.499 | 13.494 | 6.893 |
| w/o ACTOR-PAE | 0.956 | 0.323 | 3.025 | 5.988 |
| w/o ControlNet (both) | 0.956 | 0.333 | 4.611 | 6.258 |
| w/o Iter. Opt. (both) | 0.531 | 0.329 | 5.570 | 6.382 |
| **Full** | **0.988** | **0.325** | **3.892** | **6.199** |

Key findings: (1) Removing SMS data augmentation causes FID to surge from 3.892 to 13.494, as the model fails to accommodate SOS inputs of varying granularity; (2) Removing iterative optimization causes SOS-Acc to drop sharply from 0.988 to 0.531, demonstrating that optimization is central to ensuring control precision.

### Iterative Optimization Ablation

| Method | No Opt. | Diff-time Opt. | Test-time Opt. | Both |
|--------|---------|---------------|---------------|------|
| OmniControl SOS-Acc | 0.674 | 0.873 | 0.956 | 0.956 |
| Ours SOS-Acc | 0.531 | 0.535 | 0.988 | 0.988 |
| Ours FID | 5.570 | 5.187 | 4.209 | 3.892 |

Test-time optimization contributes most significantly; diffusion-time optimization has limited effect, as its adjustments may be overwritten by subsequent diffusion steps.

## Highlights & Insights

- **A novel motion control paradigm**: Abstracting from "precise 3D coordinates" to "directional symbol + temporal position" substantially lowers the barrier for users to specify motion constraints, better fitting the practical demands of industrial animation pipelines.
- **Saliency-aware sparse representation**: Agglomerative clustering automatically detects keyframes, generating a sparse and interpretable SOS staff that avoids the burden of dense per-frame annotation.
- **Critical contribution of SMS data augmentation**: Random sampling of saliency thresholds enables the model to handle SOS inputs of varying sparsity; removing this component causes FID to increase by 3.5×.
- **Stability from the ACTOR-PAE decoder**: The regularization properties of the periodic latent space allow sparse guidance to propagate naturally to neighboring frames, avoiding the motion inconsistency observed in OmniControl where test-time optimization affects only individual frames.

## Limitations & Future Work

- **Limited expressiveness of 26 directional symbols**: Discretization inevitably discards fine-grained orientation information, which may be insufficient for applications demanding high-precision directional control (e.g., surgical robot motion planning).
- **Slow inference speed**: 100-step diffusion inference combined with 100-step test-time iterative optimization results in a total inference time of approximately 17 seconds per batch, with forward kinematics computation as the primary bottleneck.
- **Evaluation limited to HumanML3D**: Generalization to other datasets (e.g., BABEL, KIT-ML) has not been validated, and clustering hyperparameters in SOS extraction may require re-tuning for different datasets.

## Related Work & Insights

- **OmniControl (Xie et al. 2024)**: Uses joint keyframe positions + ControlNet + diffusion-time optimization; achieves only 0.873 SOS-Acc, and test-time optimization causes motion inconsistency as it affects only individual frames.
- **TLControl (Wan et al. 2024)**: Employs Transformer + VQ-VAE encoding + test-time optimization; achieves high SOS-Acc of 0.982 but FID = 11.132 — the limited codebook of VQ-VAE degrades motion expressiveness and smoothness.
- **GMD (Karunratanakul et al. 2023)**: Relies on directly imputing control signals into motion signals, which is not applicable to the abstract, non-directly-mapped nature of SOS control signals.
- **PriorMDM (Shafir et al. 2024)**: Similarly relies on imputation and noise zeroing, which is incompatible with SOS.
- **KP/PoseScript and similar motion descriptors**: Extract orientation features but do not address motion saliency detection; their representations are dense per-frame and entail high programming overhead.

The SOS script abstraction — encoding motion control as a concise triplet of "which body part + which direction + at what time" — can be extended to robot task planning and choreography. The agglomerative clustering approach for motion saliency detection is transferable to tasks requiring keyframe extraction such as video summarization and motion compression. Furthermore, the SMS data augmentation strategy — randomly masking during training to adapt to varying sparsity — is a general approach applicable to sparse control signal scenarios, echoing the masking philosophy of MAE and BERT.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A novel control paradigm from Labanotation to programmable SOS scripts; the first introduction of saliency-based clustering into motion generation control.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons, ablations, and optimization strategy analyses, though validation is limited to a single dataset (HumanML3D).
- Writing Quality: ⭐⭐⭐⭐ — Systematic and modular presentation, though the overall pipeline is complex and imposes a non-trivial learning curve on readers.
- Value: ⭐⭐⭐⭐ — Introduces a practical new motion control paradigm with open-sourced code, offering direct applicability to animation and human-computer interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](../../CVPR2026/human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](../../CVPR2026/human_understanding/idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[AAAI 2026\] mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)

</div>

<!-- RELATED:END -->
