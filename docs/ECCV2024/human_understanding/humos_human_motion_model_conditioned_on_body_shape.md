---
title: >-
  [Paper Note] HUMOS: Human Motion Model Conditioned on Body Shape
description: >-
  [ECCV 2024][Human Understanding][Human Motion Generation] This paper proposes HUMOS, a human motion generation model conditioned on body shape. It learns the correlation between body shape and motion without paired training data through cycle consistency loss and differentiable intuitive physics/dynamic stability constraints, generating physically plausible and dynamically stable human motions.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Motion Generation"
  - "Body Shape Conditioning"
  - "Cycle Consistency"
  - "Dynamic Stability"
  - "Motion Retargeting"
date: 2026-05-08
content_hash: d682162e67adb9aa
---

# HUMOS: Human Motion Model Conditioned on Body Shape

**Conference**: ECCV 2024  
**arXiv**: [2409.03944](https://arxiv.org/abs/2409.03944)  
**Code**: [GitHub](https://github.com/CarstenEpic/humos)  
**Area**: Human Understanding  
**Keywords**: Human Motion Generation, Body Shape Conditioning, Cycle Consistency, Dynamic Stability, Motion Retargeting

## TL;DR

This paper proposes HUMOS, a human motion generation model conditioned on body shape. It learns the correlation between body shape and motion without paired training data through cycle consistency loss and differentiable intuitive physics/dynamic stability constraints, generating physically plausible and dynamically stable human motions.

## Background & Motivation

**Background**: Human motion generation is crucial in gaming, AR/VR, and robotics simulation. Existing SOTA methods (e.g., TEMOS, MDM) can already generate realistic human motions based on text or action labels.

**Limitations of Prior Work**: Existing motion models almost **completely neglect body shape variations**. They typically normalize all training data and train/generate using a standardized average body shape. However, individuals with different body shapes (muscle distributions, limb proportions) exhibit distinctly different motion patterns when performing the exact same action.

**Key Challenge**: Training a body-shape-conditioned motion model requires **paired data** (different body shapes performing the exact same motion), but such paired data is extremely scarce in existing datasets like AMASS.

**Goal**: How to train a generative model that can produce corresponding motions based on input body shapes in the absence of paired data.

**Key Insight**: Inspired by unpaired image-to-image translation (CycleGAN), this work utilizes cycle consistency to achieve self-supervised training while introducing differentiable physical constraints to prevent trivial solutions.

**Core Idea**: The encoder maps motion into a **shape-invariant latent space**, and the decoder receives the latent code and target body shape parameters to generate the translation. Cycle consistency ensures semantic invariance, while physical constraints guarantee physical plausibility.

## Method

### Overall Architecture

HUMOS is a Transformer-based Conditional Variational Autoencoder (c-VAE). Given the motion $M_\mathcal{A}$ of identity $\mathcal{A}$ and their identity features $\mathcal{I}_\mathcal{A} = (\beta_\mathcal{A}, \mathcal{G}_\mathcal{A})$ (body shape parameters and gender), the encoder outputs a shape-invariant latent code $z_{M_\mathcal{A}}$. The decoder takes $z_{M_\mathcal{A}}$ and a randomly sampled target identity $\mathcal{I}_\mathcal{B}$ to generate the retargeted motion $\hat{M}_{\mathcal{A} \to \mathcal{B}}$. The model represents motions using the SMPL mesh, utilizing 6D rotation representations and relative root joint rotations.

### Key Designs

1. **Self-Supervised Cycle Consistency Training**:

    - **Function**: Achieves body-shape-conditioned training without paired data.
    - **Mechanism**: The motion is first encoded into a shape-invariant latent code $z_{M_\mathcal{A}} = \mathcal{E}(M_\mathcal{A}, \mathcal{I}_\mathcal{A})$, and then decoded using the target identity to get $\hat{M}_{\mathcal{A}\to\mathcal{B}} = \mathcal{D}(z_{M_\mathcal{A}}, \mathcal{I}_\mathcal{B})$. This motion is then re-encoded with identity $\mathcal{B}$ and decoded back using the original identity $\mathcal{A}$ to obtain $\hat{M}_{\mathcal{A}\to\mathcal{A}}$, which is constrained to match the original motion $M_\mathcal{A}$:
    $\mathcal{L}_{\text{cycle}} = \mathcal{L}_{\text{rot}} + \mathcal{L}_{\text{pos}}$
   where $\mathcal{L}_{\text{rot}}$ computes the geodesic distance of rotation matrices, and $\mathcal{L}_{\text{pos}}$ is the smooth L1 loss of the root joint position.
    - **Design Motivation**: Inspired by CycleGAN, self-supervised learning is achieved via cyclic reconstruction without paired data.

2. **Intuitive Physics (IP) Constraints**:

    - **Function**: Prevents trivial solutions (directly copying the source motion to the target body shape) in cycle-consistent training.
    - **Mechanism**: Three differentiable physics losses:
        - $\mathcal{L}_{\text{penetrate}}$: distance penalty when the lowest vertex goes below the ground.
        - $\mathcal{L}_{\text{float}}$: distance penalty when the lowest vertex is above the ground.
        - $\mathcal{L}_{\text{slide}}$: horizontal velocity penalty for foot joints in contact with the ground.
    $\mathcal{L}_{\text{physics}} = \mathcal{L}_{\text{penetrate}} + \mathcal{L}_{\text{float}} + \mathcal{L}_{\text{slide}}$
    - **Design Motivation**: If the network simply copies the source motion $M_\mathcal{A}$ to the target body shape $\mathcal{B}$, physical inconsistencies like ground penetration, floating, and foot sliding will occur due to body shape differences. These physical constraints force the network to adapt the motion according to the target body shape.

3. **Dynamic Stability Term (ZMP)**:

    - **Function**: Ensures that the generated dynamic motion sequences satisfy biomechanical dynamic stability.
    - **Mechanism**: Based on the Zero Moment Point (ZMP) concept. ZMP is the point on the ground where the horizontal component of the ground reaction force torque becomes zero. When the ZMP lies within the Base of Support (BoS), the motion is considered dynamically stable. The ZMP is calculated as:
    $\mathcal{Z} = \mathcal{C}_m - \frac{n \times \mathcal{M}_{\mathcal{C}_m}^{gi}}{\mathcal{F}^{gi} \cdot n}$
   where $\mathcal{C}_m$ is the ground projection of the center of mass (CoM), $\mathcal{F}^{gi} = mg - ma_\mathcal{G}$ is the inertial force, and $\mathcal{M}_{\mathcal{C}_m}^{gi}$ is the torque around the projected CoM on the ground. The dynamic stability loss is defined as the distance between the ZMP and the Center of Pressure (CoP):
    $\mathcal{L}_{\text{dyn}} = \rho(\|\mathcal{C}_P - \mathcal{Z}\|_2)$
   where $\rho$ is the Geman-McClure robust loss function. The total mass is estimated via the SMPL mesh volume, and the mass of each vertex is distributed proportionally according to the volume of the body part it belongs to.
    - **Design Motivation**: Static stability (e.g., the IPMAN method) is only suitable for static poses. Since human locomotion is inherently highly dynamic, factors such as acceleration and angular momentum must be considered. ZMP is widely used in robotic balance control.

### Loss & Training

The total loss is a weighted sum of the components:

$$\mathcal{L} = \lambda_{\text{cycle}}\mathcal{L}_{\text{cycle}} + \lambda_{\text{physics}}\mathcal{L}_{\text{physics}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}} + \lambda_{\text{E}}\mathcal{L}_{\text{E}}$$

where $\lambda_{\text{cycle}}=1$, $\lambda_{\text{physics}}=1$, $\lambda_{\text{dyn}}=10^{-4}$, $\lambda_{\text{KL}}=10^{-5}$, and $\lambda_{\text{E}}=10^{-2}$. The KL divergence loss $\mathcal{L}_{\text{KL}}$ regularizes the latent space distribution, and $\mathcal{L}_{\text{E}}$ encourages shape-invariant latent codes for the same motion across different identities.

The training data is from the AMASS dataset (480 identities), sampled at 20fps with $T=200$ frames. Models are trained for 1300 epochs using the AdamW optimizer with a learning rate of $10^{-5}$ and a batch size of 60. Both the encoder and decoder consist of a 6-layer Transformer.

## Key Experimental Results

### Main Results

| Method | Penetrate(cm)↓ | Float(cm)↓ | Skate(%)↓ | Dyn.Stability(%)↑ | BoSDist(cm)↓ |
|------|----------------|------------|-----------|-------------------|--------------|
| TEMOS-Simple | 6.82 | 6.55 | 27.07 | 45.85 | 16.94 |
| TEMOS-Rokoko | 4.14 | 3.85 | 20.05 | 55.92 | 16.58 |
| TEMOS-Rokoko-G | 0.75 | 4.44 | 20.05 | 55.92 | 16.58 |
| **HUMOS** | **1.23** | **1.04** | **7.37** | **71.9** | **14.62** |

### Ablation Study

| Configuration | Penetrate↓ | Float↓ | Skate(%)↓ | Dyn.Stability(%)↑ | BoSDist↓ |
|------|-----------|--------|-----------|-------------------|----------|
| $\mathcal{L}_{\text{cycle}}$ only | 2.74 | 2.62 | 15.04 | 64.00 | 16.96 |
| + $\mathcal{L}_{\text{physics}}$ | 1.55 | 1.44 | 7.93 | 67.82 | 16.41 |
| + $\mathcal{L}_{\text{dyn}}$ (Full) | **1.23** | **1.04** | **7.37** | **71.9** | **14.62** |

### Key Findings

- Using only cycle consistency brings a ~33% improvement in penetration and a ~25% improvement in foot sliding compared to the TEMOS-Rokoko baseline.
- Physics constraints yield the largest improvement in foot sliding (~47%), indicating that foot skating is the most severe artifact when body shapes do not match.
- The dynamic stability term boosts the ratio of stable frames from 67.82% to 71.9% while improving all other metrics.
- In the perceptual study, HUMOS achieves a score of 3.64/5, significantly outperforming TEMOS-Rokoko's score of 3.25/5.

## Highlights & Insights

- **Clever avoidance of paired data requirements**: The combination of cycle consistency and physics constraints is elegant—the former provides self-supervised signals while the latter prevents trivial solutions, and both are indispensable.
- **Differentiable dynamic stability**: The ZMP concept from robotics is introduced into data-driven motion generation as a fully differentiable component, enabling end-to-end training with neural networks.
- **High practical value**: Direct motion retargeting between characters is achieved without the traditional two-step pipeline (generation followed by retargeting).

## Limitations & Future Work

- Motion differences across body shapes remain somewhat subtle, likely limited by the diversity of body shapes in the training set.
- Self-penetration issues in motions are not addressed.
- The model only considers body shape conditioning, leaving other motion style factors like emotional states or physical impairments unaddressed.
- Minor motion artifacts are still present in the generated results.

## Related Work & Insights

- **vs TEMOS**: TEMOS generates motion for a standardized body shape and requires an extra retargeting step, whereas HUMOS directly generates body-adapted motions.
- **vs Physics-based Simulation Methods (RL-based)**: RL methods are physically plausible but computationally expensive and suffer from limited diversity. HUMOS incorporates physical constraints within a data-driven framework via differentiable physics terms.
- **vs CycleGAN**: It borrows the concept of cycle consistency but innovatively applies it to the motion-shape space, replacing adversarial training with physical constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of cycle consistency and physics constraints is novel in the motion domain, and the dynamic ZMP constraint is introduced for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation uses comprehensive physical metrics and includes a perceptual study, though comparisons with more recent methods are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical flow is clear, the mathematical derivations are thoroughly presented, and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ It provides a viable solution for shape-aware motion generation, showing strong practical value for gaming and virtual character applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] Modeling and Driving Human Body Soundfields through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)
- [\[CVPR 2025\] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](../../CVPR2025/human_understanding/shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)
- [\[ICLR 2026\] CLUTCH: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild](../../ICLR2026/human_understanding/clutch_contextualized_language_model_for_unlocking_text-conditioned_hand_motion_.md)
- [\[ECCV 2024\] Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences](alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)

</div>

<!-- RELATED:END -->
