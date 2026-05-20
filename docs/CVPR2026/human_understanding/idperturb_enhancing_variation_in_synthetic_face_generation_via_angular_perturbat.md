---
title: >-
  [Paper Note] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations
description: >-
  [CVPR 2026][Human Understanding][Synthetic faces] This paper proposes IDperturb, a geometry-driven sampling strategy that applies angular perturbations to identity embeddings on the unit hypersphere. Without modifying th…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Synthetic faces"
  - "identity perturbation"
  - "angular sampling"
  - "diffusion models"
  - "face recognition"
date: 2026-05-08
content_hash: e4a65b83a4838866
---

# IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations

**Conference**: CVPR 2026
**arXiv**: [2602.18831](https://arxiv.org/abs/2602.18831)  
**Code**: [GitHub](https://github.com/fdbtrs/IDiff-Face) (based on IDiff-Face)  
**Area**: Image Generation / Face Recognition
**Keywords**: Synthetic faces, identity perturbation, angular sampling, diffusion models, face recognition

## TL;DR

This paper proposes IDperturb, a geometry-driven sampling strategy that applies angular perturbations to identity embeddings on the unit hypersphere. Without modifying the generative model, it significantly enhances intra-class diversity in synthetic face datasets and improves downstream face recognition performance.

## Background & Motivation

- **Background**: Synthetic face data has emerged as a privacy-friendly alternative for training face recognition (FR) systems. Identity-conditioned diffusion models (e.g., IDiff-Face, DCFace) can generate realistic and identity-consistent face images.
- **Limitations of Prior Work**: These models commonly suffer from **insufficient intra-class variation**—images generated for the same identity tend to be overly similar in age, expression, and pose, leading to FR models with limited generalization. Existing methods introduce diversity via additional label conditioning (ID3), learned style modules (DCFace), or iterative embedding optimization (HyperFace), but these approaches require architectural modifications, auxiliary labels, or substantial computational cost.
- **Key Challenge**: The geometric structure of the identity embedding space itself has not been exploited for diversity enhancement.
- **Goal**: To introduce controlled intra-class variation through a purely geometric sampling strategy, requiring no modification to the generative model.

## Method

### Overall Architecture

IDperturb is a **purely geometry-driven** sampling strategy operating in the embedding space of a pretrained identity-conditioned diffusion model. Given a reference identity embedding $\mathbf{v}$, a set of perturbed embeddings $\{\tilde{\mathbf{v}}_k\}_{k=1}^K$ is generated within a constrained angular region (a $d$-dimensional cone) around $\mathbf{v}$. Each perturbed embedding serves as a conditioning input to generate one face image.

### Key Designs

1. **Angular Sampling**: The core idea is to apply controlled angular displacements to identity embeddings on the unit hypersphere. A target cosine similarity is sampled uniformly as $s \sim \mathcal{U}[\mathbf{lb}, 1]$, corresponding to angle $\theta = \cos^{-1}(s)$. Random noise $\mathbf{n} \sim \mathcal{N}(0, \mathbf{I})$ is projected onto the hyperplane orthogonal to $\mathbf{v}$ to obtain a unit vector $\mathbf{u}$. The perturbed embedding is then constructed as:

$$\tilde{\mathbf{v}} = \cos(\theta) \cdot \mathbf{v} + \sin(\theta) \cdot \mathbf{u}$$

This guarantees $\|\tilde{\mathbf{v}}\| = 1$ (norm preservation) and $\langle \tilde{\mathbf{v}}, \mathbf{v} \rangle = \cos(\theta) = s$ (exact angular control). The **Design Motivation** is to exploit the correspondence between cosine similarity and identity semantics in FR embedding spaces, introducing controllable variation while preserving identity.

2. **Lower Bound Constraint**: The parameter $\mathbf{lb}$ defines the maximum permissible angular displacement. A smaller $\mathbf{lb}$ yields greater variation but may compromise identity consistency. To **prevent identity overlap**, the lower bound is dynamically adjusted:

$$\mathbf{lb} \leftarrow \max\left(\mathbf{lb}, \max_{j \neq i} \cos\left(\frac{\angle(\mathbf{v}_i, \mathbf{v}_j)}{2}\right)\right)$$

This ensures that the perturbed embedding always remains angularly closer to the original identity than to any other identity (via angle bisection)—an elegant geometric guarantee.

3. **Integration with Pretrained Diffusion Models**: IDperturb integrates seamlessly with pretrained LDMs (e.g., IDiff-Face). For each identity, $K$ perturbed embeddings are generated; each is paired with a distinct initial noise $\mathbf{z}_T$ and decoded through the reverse diffusion process. DDIM with 50 steps and Classifier-Free Guidance (CFG) are used. The perturbation overhead is negligible (50 perturbations per identity in 0.01 seconds on an M3 CPU).

### Loss & Training

IDperturb involves **no training**—it is an inference-time sampling strategy. Downstream FR training uses ResNet50 with CosFace loss (margin=0.35, scale=64), SGD optimizer, 34 epochs, and an initial learning rate of 0.1.

## Key Experimental Results

### Main Results

FR verification accuracy (%) on top of the IDiff-Face (C-WF) baseline:

| Dataset | Metric | IDperturb (lb=0.6) | Baseline (no perturbation) | Gain |
|--------|------|-------|----------|------|
| LFW | Acc | 99.40 | 98.75 | +0.65 |
| AgeDB-30 | Acc | 93.20 | 88.85 | +4.35 |
| CFP-FP | Acc | 93.61 | 91.61 | +2.00 |
| CA-LFW | Acc | 93.50 | 90.90 | +2.60 |
| CP-LFW | Acc | 88.37 | 86.15 | +2.22 |
| **Average** | Acc | **93.62** | 91.25 | **+2.37** |

Under the same setting (DGM trained on C-WF), IDperturb achieves 93.62% average accuracy, surpassing all competing methods.

### Ablation Study

| Configuration | Avg. Accuracy | Note |
|------|---------|------|
| lb=0.9 | 92.68 | Small perturbation, limited gain |
| lb=0.8 | 93.31 | Moderate perturbation |
| lb=0.7 | 93.44 | Near optimal |
| **lb=0.6** | **93.62** | **Optimal trade-off** |
| lb=0.5 | 93.56 | Slight decline begins |
| lb=0.4 | 93.36 | Identity consistency degrades |
| Baseline | 91.25 | No perturbation |

CFG strength ablation (lb=0.6): $\omega=2$ achieves the best result (93.63%); excessively large $\omega$ constrains diversity.

### Key Findings

- Decreasing lb monotonically increases intra-class diversity ($D_{intra}$) but reduces identity consistency ($C_{intra}$); the optimal balance is at lb=0.6.
- At lb=0.6, entropy of age, expression, and standard deviation of head pose all approach those of the real dataset C-WF.
- Perturbations applied solely in embedding space implicitly promote diversity across pose, age, and expression.

## Highlights & Insights

1. **Extreme Simplicity**: The method reduces to a single geometric operation—angular sampling on the hypersphere—requiring no model modification, no auxiliary labels, no training, and virtually zero computational overhead.
2. **Mathematical Elegance**: Hypersphere geometry guarantees norm invariance and exact angular control; the angle-bisection strategy for avoiding identity overlap has a rigorous geometric interpretation.
3. **Strong Generalizability**: IDperturb is plug-and-play for any identity-conditioned diffusion model, validated on both FFHQ and C-WF baselines.

## Limitations & Future Work

1. At low lb values (e.g., 0.4), identity consistency of some samples degrades noticeably, with a significant rise in EER.
2. Validation is limited to IDiff-Face; stronger baselines such as Arc2Face have not been tested.
3. Angular sampling directions are uniformly random, leaving unexploited the semantic structure whereby different directions in embedding space correspond to different attribute variations.
4. The method is designed for 2D face synthesis; extension to 3D face generation or general image generation remains to be validated.

## Related Work & Insights

- **IDiff-Face / UIFace**: The baseline diffusion models; IDperturb provides plug-and-play improvements on top of these.
- **DCFace**: Increases diversity via learned style embeddings—more complex but potentially capturing richer variation.
- **HyperFace**: Iteratively optimizes embedding space sampling at higher computational cost.
- **Insights**: The geometric structure of hyperspherical embedding spaces warrants deeper exploitation—for example, non-uniform sampling along semantically meaningful directions (corresponding to age, pose, etc.), or extending this paradigm to other conditional generation tasks (e.g., style transfer, text-conditioned generation).

## Rating

- Novelty: ⭐⭐⭐⭐ Addresses the diversity problem from a purely geometric perspective; the approach is concise yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines, benchmarks, and multi-faceted ablations (diversity / consistency / attributes / separability); highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, figures are intuitive, and experiments are well-organized.
- Value: ⭐⭐⭐⭐ Zero-cost plug-and-play improvement of synthetic face data quality; directly applicable to privacy-preserving FR training scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[CVPR 2026\] Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)

</div>

<!-- RELATED:END -->
