---
title: >-
  [Paper Note] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence
description: >-
  [ICLR 2026][LLM Evaluation][optimal transport] This paper leverages the geometric singularity boundaries of semi-discrete optimal transport (OT) to locate semantically ambiguous regions in latent space, generates proxy OOD samples (OTIS) near these boundaries, and applies a confidence suppression loss during training to enforce uniform predictions in structurally uncertain regions, thereby systematically mitigating OOD overconfidence in DNNs.
tags:
  - ICLR 2026
  - LLM Evaluation
  - optimal transport
  - OOD overconfidence
  - singularity boundaries
  - confidence calibration
  - OTIS
date: 2026-05-08
content_hash: 2745657bfeca2ba6
---

# Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence

**Conference**: ICLR 2026
**arXiv**: [2601.21320](https://arxiv.org/abs/2601.21320)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: optimal transport, OOD overconfidence, singularity boundaries, confidence calibration, OTIS

## TL;DR

This paper leverages the geometric singularity boundaries of semi-discrete optimal transport (OT) to locate semantically ambiguous regions in latent space, generates proxy OOD samples (OTIS) near these boundaries, and applies a confidence suppression loss during training to enforce uniform predictions in structurally uncertain regions, thereby systematically mitigating OOD overconfidence in DNNs.

## Background & Motivation

**State of the Field**: Deep neural networks have achieved remarkable success in closed-set classification, yet they frequently produce high-confidence erroneous predictions when encountering out-of-distribution (OOD) inputs during open-world deployment. Existing mitigation strategies fall into two broad categories: test-time OOD detection (e.g., filtering via confidence scores such as MSP and ODIN) and training-time exposure to proxy OOD samples to suppress overconfidence.

**Limitations of Prior Work**: Test-time detection methods are merely symptomatic treatments—they filter out high-confidence errors but do not alter the model's inherent tendency to produce overconfident predictions. Training-time approaches are more proactive, yet proxy OOD sample construction typically relies on heuristic rules (e.g., external datasets, input corruptions, class mixups, or latent-space outlier synthesis), lacking theoretical grounding and failing to precisely cover semantically ambiguous regions where overconfidence most commonly occurs.

**Root Cause**: A fundamental disconnect exists between existing proxy OOD sample generation and the decision-boundary regions where classifiers are most error-prone. Heuristically generated samples may be scattered throughout feature space rather than concentrated in the structurally unstable regions where classifier behavior is least reliable.

**Paper Goals**: (1) How can one theoretically identify regions where classifiers are most susceptible to overconfidence? (2) How can theoretically guided proxy OOD samples be generated that genuinely reside near semantic boundaries? (3) How can these samples be effectively leveraged for training-time regularization?

**Starting Point**: The authors draw on the geometric theory of semi-discrete OT, observing that the singularity boundaries of OT maps—non-smooth points where the transport direction undergoes abrupt changes—naturally correspond to semantically ambiguous regions. Near these boundaries, classifier decision behavior is most unstable and overconfidence is most likely to arise.

**Core Idea**: By solving a semi-discrete OT problem to obtain a Laguerre partition of the latent space, the method interpolates near singularity boundaries where transport directions change most sharply to generate semantically ambiguous samples (OTIS), which are then used for confidence suppression during training.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) encoding training samples into a compact latent space via an autoencoder; (2) solving a semi-discrete OT problem in latent space to identify singularity boundaries and generate OTIS; (3) mixing OTIS with normal training data and training the classifier with a confidence suppression loss. The input is a set of training images, and the output is a classification model with better calibration in semantically ambiguous regions.

### Key Designs

1. **Latent Space Representation and Autoencoder**:

    - **Function**: Maps high-dimensional inputs to a low-dimensional compact space, making OT computation feasible.
    - **Mechanism**: An encoder $y = Enc(x)$ maps each training sample to a latent vector, and a decoder $x' = Dec(y)$ reconstructs the input-space representation from latent points. The set of latent vectors $\{y_i\}$ constitutes the support of the discrete target measure for OT.
    - **Design Motivation**: Performing OT directly in the original input space is computationally intractable due to high dimensionality and irregular structure; the latent space is more compact and regular, facilitating regularization and interpolation.

2. **Semi-Discrete OT Partition and Singularity Boundary Identification**:

    - **Function**: Partitions the latent space into convex regions (Laguerre cells) and identifies boundaries where transport directions change most abruptly.
    - **Mechanism**: Given a continuous source distribution $\mu$ (e.g., Gaussian) and discrete targets $\{y_i\}$, the Brenier potential $u_{\mathbf{h}}(z) = \max_i \{\langle y_i, z \rangle + h_i\}$ is solved, whose gradient yields the optimal transport map. Monte Carlo sampling estimates the volume of each Laguerre cell, and gradient descent optimizes offsets $\mathbf{h}$ to match target weights. For each pair of adjacent boundaries $\mathcal{S}_{ij}$, an angular deviation score $\text{score}(\mathcal{S}_{ij}) = \arccos(\langle y_i, y_j \rangle / \|y_i\| \|y_j\|)$ is computed; boundaries with the highest scores are retained to form the singularity boundary set $\mathcal{S}'$.
    - **Design Motivation**: Large angular deviation indicates a sharp directional change in the transport map, corresponding to a discontinuity (singularity) in the OT map. These regions are precisely the most semantically ambiguous—inputs therein may simultaneously exhibit features of multiple classes.

3. **OTIS Generation via Inverse-Distance Interpolation**:

    - **Function**: Synthesizes proxy OOD samples near singularity boundaries, targeting semantically ambiguous regions.
    - **Mechanism**: For each selected singularity boundary $\mathcal{S}_{ij}$, the centroids $\hat{c}_i, \hat{c}_j$ of adjacent Laguerre cells are estimated via Monte Carlo. A latent point $z \sim \mu$ is sampled, and inverse-distance interpolation weights $\lambda_i = (1/\|z - \hat{c}_i\|) / (1/\|z - \hat{c}_i\| + 1/\|z - \hat{c}_j\|)$ are computed to obtain a smooth interpolation $\hat{y} = \lambda_i T(\hat{c}_i) + \lambda_j T(\hat{c}_j)$, which is then decoded into input space as $\hat{x} = Dec(\hat{y})$.
    - **Design Motivation**: Inverse-distance interpolation provides a smooth extension of the OT map near singularity boundaries, avoiding artifacts from discrete jumps while ensuring that generated samples are semantically coherent and naturally situated in inter-class transition zones.

### Loss & Training

Each training batch consists of 50% in-distribution (ID) samples and 50% OTIS. ID samples are trained with standard cross-entropy loss, while OTIS are trained with a confidence suppression loss: $\mathcal{L}_{\text{sup}}(\hat{x}) = \sum_{i=1}^{K} \frac{1}{K} \log V_i(\hat{x})$, where $V_i(\hat{x})$ is the softmax probability for class $i$. This loss encourages the model to output uniform predictive distributions on OTIS—i.e., an "uncertain" response—thereby preventing overconfident predictions in structurally ambiguous regions.

## Key Experimental Results

### Main Results

Experiments are conducted with CIFAR-10/100, SVHN, MNIST, and FMNIST as ID datasets and various natural/adversarial/noise sources as OOD inputs, comparing against 8 methods. The primary metrics are OOD maximum mean confidence (OOD MMC↓, lower is better) and ID maximum mean confidence (ID MMC↑, higher is better).

| ID Dataset | OOD Dataset | Ours | CEDA | ACET | CCUs | CODES | VOS |
|------------|-------------|------|------|------|------|-------|-----|
| CIFAR-10 | SVHN | **13.18%** | 71.62% | 82.16% | 72.48% | 72.35% | 73.16% |
| CIFAR-10 | CIFAR-100 | **64.79%** | 80.18% | 82.36% | 75.95% | 74.69% | 81.04% |
| CIFAR-10 | Uniform | **10.00%** | 10.04% | 10.00% | 10.00% | 11.13% | 80.65% |
| CIFAR-10 | Adv. Noise | **26.42%** | 43.04% | 10.00% | 10.00% | 37.66% | 95.56% |
| CIFAR-100 | SVHN | **9.30%** | 63.03% | 62.85% | 65.49% | 66.11% | 58.76% |
| SVHN | CIFAR-10 | **61.37%** | 73.70% | 62.54% | 46.92% | 61.09% | 71.39% |

The proposed method substantially reduces OOD MMC across the vast majority of ID/OOD combinations while maintaining ID accuracy and ID MMC comparable to baselines.

### Ablation Study

| Configuration | CIFAR-10 TE | CIFAR-10 OOD MMC (SVHN) | Notes |
|---------------|-------------|-------------------------|-------|
| No regularization (Baseline) | 5.79% | 84.22% | No OOD exposure |
| OE (external data) | 6.80% | 55.82% | Requires additional OOD data |
| CCUd (external data) | 5.55% | 76.52% | Requires additional OOD data |
| **Ours (no external data)** | **7.52%** | **13.18%** | Requires no external data |

### Key Findings

- **OTIS substantially outperforms all methods that require no external data**: On benchmarks such as SVHN→CIFAR-10, OOD MMC drops from 84% to 13%, a reduction of over 70 percentage points.
- **Effectiveness against adversarial inputs**: On Adversarial Noise and Adversarial Samples, the proposed method's OOD MMC is also significantly lower than most competitors.
- **Controlled ID accuracy degradation**: Test error increases only modestly (e.g., CIFAR-10: 5.79% to 7.52%), indicating that OT-guided regularization does not excessively interfere with ID learning.
- **Consistent cross-dataset generalization**: Consistent advantages are demonstrated on substantially different datasets such as MNIST and FMNIST.

## Highlights & Insights

- **Theoretical correspondence between OT singularities and semantic ambiguity**: The paper's most central insight is establishing, for the first time, a theoretical link between geometric singularities of semi-discrete OT and classifier overconfidence regions, providing a principled alternative to heuristic proxy OOD sample generation.
- **High-quality proxy OOD generation without external data**: Unlike methods such as OE that require additional datasets, OTIS is derived entirely from the geometric structure of the training data itself, making it better suited for data-constrained settings.
- **Smooth transport extension via inverse-distance interpolation**: This elegantly resolves the discontinuity of OT maps at singularity boundaries while preserving the semantic coherence of generated samples.

## Limitations & Future Work

- **Computational overhead**: The pipeline requires sequentially training an autoencoder, solving an OT problem, and generating OTIS; scalability to large-scale datasets remains to be validated.
- **Sensitivity to latent space dimensionality**: The dimensionality and quality of the autoencoder's latent space directly affect the validity of the OT partition, yet sensitivity analysis on this factor is insufficient in the paper.
- **Restriction to classification tasks**: The current framework is designed for multi-class classification; extension to regression, segmentation, and other task types remains unexplored.
- **Hyperparameter for singularity boundary selection**: The proportion of high-scoring boundaries to retain is a hyperparameter requiring tuning, and poor selection may degrade sample quality.

## Related Work & Insights

- **vs. OE (Outlier Exposure)**: OE relies on external OOD datasets and lacks principled guidance for sample selection; the proposed method derives proxy OOD samples from the OT geometry of the training data itself, requiring no external data and offering stronger theoretical justification.
- **vs. VOS (Virtual Outlier Synthesis)**: VOS samples outliers in latent space as proxy OOD without considering their relationship to classification boundaries; OTIS precisely targets semantically ambiguous singularity boundaries, making the approach more targeted.
- **vs. CEDA/ACET**: These methods construct proxy OOD samples via input corruption, potentially placing them far from actual semantic boundaries; OTIS directly generates samples in class transition regions by exploiting the geometric structure of OT.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing the concept of singularity boundaries from optimal transport theory into OOD overconfidence mitigation is a unique perspective with rigorous theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage spans 6 ID datasets and multiple OOD types, but validation on large-scale datasets (e.g., ImageNet) is absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and rigorous, though the pipeline description is complex and presupposes substantial background knowledge in OT.
- Value: ⭐⭐⭐⭐ Provides a principled solution to OOD overconfidence, though computational complexity may limit practical adoption.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/llm_evaluation/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/llm_evaluation/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)

<!-- RELATED:END -->
