---
title: >-
  [Paper Note] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models
description: >-
  [AAAI 2026][AI Safety][Adversarial Defense] Ours proposes TopoReformer, a model-agnostic adversarial purification pipeline based on topological autoencoders. By leveraging persistent homology to enforce topological consistency within the latent space, the framework filters adversarial perturbations without requiring adversarial training, effectively safeguarding OCR systems against classic, adaptive, and OCR-specific watermark attacks.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Adversarial Defense"
  - "Topological Autoencoders"
  - "OCR Security"
  - "Persistent Homology"
  - "Manifold Purification"
date: 2026-05-08
content_hash: ce4cfb04adbe17b5
---

# TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models

**Conference**: AAAI 2026  
**arXiv**: [2511.15807](https://arxiv.org/abs/2511.15807)  
**Code**: [github](https://github.com/invi-bhagyesh/TopoReformer)  
**Area**: AI Safety  
**Keywords**: Adversarial Defense, Topological Autoencoders, OCR Security, Persistent Homology, Manifold Purification

## TL;DR

Ours proposes TopoReformer, a model-agnostic adversarial purification pipeline based on topological autoencoders. By leveraging persistent homology to enforce topological consistency within the latent space, the framework filters adversarial perturbations without requiring adversarial training, effectively safeguarding OCR systems against classic, adaptive, and OCR-specific watermark attacks.

## Background & Motivation

OCR systems are widely deployed in high-risk scenarios such as document automation, license plate recognition, and compliance auditing. However, deep-learning-driven OCR models inherit the adversarial vulnerabilities of DNNs, where imperceptible perturbations can trigger catastrophic transcription errors and even survive physical-world transitions (print-scan, photography).

Existing defense strategies generally fall into four categories:

**Preprocessing/Denoising Purification** (e.g., MagNet, PuVAE): Maps inputs back to the learned data manifold but often degrades the performance on clean inputs.

**Anomaly Detection**: Relies on autoencoder reconstruction errors or distribution tests but remains vulnerable to adaptive attacks.

**Adversarial Training**: Entails high computational costs and depends on specific attack types.

**Post-processing Correction**: Patching text outputs, which addresses the symptoms rather than the root cause.

The common limitations of these methods are that they are **model-specific, computationally expensive, and unable to resist unseen or adaptive attacks**. More critically, many defenses have been shown to provide a "false sense of security" (gradient obfuscation) and fail completely under adaptive attacks such as BPDA/EOT.

**Key Insight**: Adversarial perturbations typically alter only local pixel relationships, leaving the global topological structure of the data (such as connectivity, loops, and holes) unchanged. If topological invariance can be preserved during encoding, topologically unrelated perturbations can be naturally filtered out—amounting to "purification" rather than mere "denoising," which serves as the core innovation of this work.

## Method

### Overall Architecture

TopoReformer is a three-stage cascaded pipeline:

1. **Topological Autoencoder (TopoAE)**: Enforces topological consistency in the latent space using a persistent homology loss to achieve topology-level purification.
2. **Reformer (VAE)**: Aligns the output of the TopoAE to the manifold expected by the downstream classifier.
3. **Auxiliary Module**: Injects the latent vectors of the TopoAE into the bottleneck layer of the Reformer to provide complementary topological information.

The entire pipeline is **model-agnostic**—serving as a plug-and-play preprocessing module that can be placed in front of any OCR model without modifying or retraining the downstream model.

### Key Designs

#### 1. **Topological Autoencoder (TopoAE) and Persistent Homology Loss**

**Mechanism**: Calculate the persistence diagrams of the input space $X$ and latent space $Z$, penalizing the topological discrepancy between them.

Persistent homology tracks the birth and death of topological features (connected components, loops, and holes) as the scale parameter varies, generating a stable summary of the data structure that is robust to tiny perturbations.

The topological loss is defined as a bidirectional mapping:

$$L_t = L_{X \to Z} + L_{Z \to X}$$

Where:
- $L_{X \to Z} = \frac{1}{2} \|A_X^{\pi_X} - A_Z^{\pi_X}\|^2$: Demands that the latent space is consistent under the topological pairings of the input space.
- $L_{Z \to X} = \frac{1}{2} \|A_Z^{\pi_Z} - A_X^{\pi_Z}\|^2$: Demands that the input space is consistent under the topological pairings of the latent space.

The total loss is:

$$L = L_{rec}(X, \hat{X}) + \lambda L_t$$

**Design Motivation**: Unlike pixel-level denoising, the topological loss focuses on global structural consistency. Topologically irrelevant variations on the data manifold (such as adversarial perturbations) are naturally discarded during encoding. TopoAE is trained solely on clean data and does not require exposure to any adversarial samples.

#### 2. **Reformer (VAE) and Classifier Alignment**

Although the output of TopoAE is topologically purified, its manifold may mismatch the input distribution of the downstream OCR classifier. The Reformer is a lightweight VAE designed to align the purified images with the target manifold expected by the classifier.

The Reformer training objective:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$$

- $\mathcal{L}_{MSE}$: Pixel-level reconstruction loss between TopoAE outputs and VAE reconstruction outputs.
- $\mathcal{L}_{CE}$: Cross-entropy loss computed on the VAE reconstruction output through the classifier, ensuring classification utility.
- $\mathcal{L}_{KL}$: KL divergence regularization.

**Design Motivation**: Purely topological purification might retain the correct structure but deviate from the input distribution of the classifier; the Reformer bridges this manifold mismatch.

#### 3. **Freeze-Flow Training Paradigm and Auxiliary Module**

The auxiliary module receives the latent vector of the TopoAE (via a learned projection network) and leverages topology-aware latent space information to assist the Reformer in making more accurate predictions.

However, during direct joint training, the model tends to rely solely on the purified image path, neglecting the auxiliary path and leading to gradient starvation for the auxiliary module.

The Freeze-Flow training paradigm:
1. **Freeze Stage**: Freeze the encoder of the Reformer VAE to force the gradients to flow through the auxiliary module.
2. **Unfreeze Stage**: Unfreeze the VAE decoder after a warmup period to joint-train both pathways.

**Design Motivation**: Ensures that the auxiliary pathway establishes meaningful latent representations before the main pathway dominates, balancing the optimization of both pathways. Experiments show that Freeze-Flow provides an additional ~5% improvement in classification performance under Carlini-Wagner attacks.

### Loss & Training

- The TopoAE is pre-trained independently on clean samples until convergence, and its weights are frozen to perform inference only.
- The Reformer + Auxiliary Module are trained on the output of the TopoAE using the objective function $\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$.
- Training hyperparameters: $\lambda_1=1, \lambda_2=0.5, \lambda_3=0.5$, using the Adam optimizer with a learning rate of $0.001$.
- The classifier weights remain frozen throughout the entire process.

## Key Experimental Results

### Main Results

**Ablation Study under Classic Attacks (MNIST/EMNIST, F1-score %)**

| Attack Type | Defense Configuration | MNIST (Weak / Strong) | EMNIST (Weak / Strong) |
|---------|---------|--------------|---------------|
| Carlini (c=1e-2/1e+1) | No Defense | 30.41 / 4.30 | 36.54 / 33.85 |
| | + TopoAE | 53.92 / 48.51 | 30.87 / 27.71 |
| | + Reformer | 65.38 / 67.93 | 50.64 / 49.16 |
| | + Aux + Warmup | **65.86 / 75.15** | **69.66 / 68.82** |
| PGD (ε=0.005/0.01) | No Defense | 96.74 / 96.62 | 84.87 / 72.66 |
| | Full Pipeline | **97.70 / 97.62** | **84.53 / 83.79** |
| FGSM (ε=0.005/0.01) | No Defense | 96.87 / 96.61 | 90.83 / 72.59 |
| | Full Pipeline | **97.69 / 97.51** | **90.48 / 84.42** |

**Adaptive Attack Results (ASR↓ / F1↑)**

| Attack | MNIST ASR↓ / F1↑ | EMNIST ASR↓ / F1↑ |
|------|------------------|-------------------|
| EOT | 9.19 / 90.73 | 28.32 / 73.28 |
| EOT+BPDA | 36.59 / 64.71 | 44.26 / 58.92 |
| BPDA | 81.14 / 15.65 | 84.46 / 12.77 |

**OCR-Specific Attack (FAWA Watermark Attack)**

| OCR Model | No Defense ASR / Acc | With Defense ASR / Acc |
|---------|----------------|----------------|
| CRNN | 100 / 48.13 | 78.83 / 71.00 |
| Rosetta | 99.83 / 69.66 | **44.08 / 85.98** |
| TRBA | 99.83 / 46.68 | 60.75 / 80.26 |

### Ablation Study

| Component | C&W (MNIST, Strong) F1 | Gain |
|------|-------------------|------|
| No Defense | 4.30% | — |
| + TopoAE | 48.51% | +44.21 |
| + Reformer | 67.93% | +19.42 |
| + Auxiliary | 72.41% | +4.48 |
| + Freeze-Flow Warmup | **75.15%** | +2.74 |

Gradual ablation validates the contributions of each component, with the Freeze-Flow training paradigm providing a significant gain under strong attacks.

### Key Findings

1. **Most effective against C&W attacks**: The perturbations generated by C&W are fine and of low amplitude, which is precisely the type of noise that topological purification excels at weeding out, improving F1 from 4.30% to 75.15%.
2. **Robust against EOT adaptive attacks**: ASR drops from 99.05% to 9.19%, indicating that the defense mechanism is not merely obfuscating gradients.
3. **No performance degradation on clean samples**: Achieves ~98% accuracy on MNIST and ~94% on EMNIST under clean conditions, showing almost zero degradation.
4. **Generalizable across OCR architectures**: Effective for both CTC-based (CRNN, Rosetta, STAR-Net) and Attention-based (RARE, TRBA) models.

## Highlights & Insights

- **Paradigm Innovation**: Serves as the first attempt to apply topological autoencoders to adversarial defense/input purification. Unlike traditional denoising, it performs "topological purification" by discarding topologically irrelevant variations.
- **No Adversarial Training Required**: Trained strictly on clean data, naturally generalizing to unseen attacks.
- **Intricate Freeze-Flow Training Technique**: Controls gradient flow to resolve the common under-training issue of auxiliary branches in multi-path models.
- **Implicit Lipschitz Smoothness**: Topological constraints implicitly limit the sensitivity of latent representations to input perturbations, offering a source of robustness independent of explicit gradient regularization.

## Limitations & Future Work

1. **Vulnerable to BPDA Attacks**: The BPDA attack can exploit local curvature to bypass the global topological smoothness, resulting in an ASR of up to 81%.
2. **Limited Evaluation Datasets**: Validated only on simple datasets like MNIST/EMNIST; not tested on natural image datasets such as CIFAR-10/ImageNet.
3. **Omission of Reformer in OCR Evaluation**: To maintain deployment efficiency, the Reformer was intentionally omitted in OCR evaluations, which capped the ceiling of defense effectiveness in OCR scenarios.
4. **Lack of Computational Overhead Analysis**: The time complexity and latency of computing persistent homology were not reported, which could pose a bottleneck for large-scale deployment.
5. **No Comparison with Diffusion-based Purification**: Lacks a comparison with recent powerful baselines such as diffusion-based adversarial purification methods (e.g., DiffPure).

## Related Work & Insights

- The application of Topological Data Analysis (TDA) in adversarial robustness is an emerging direction, and this paper represents the first attempt to use TopoAE for adversarial purification.
- The Freeze-Flow training paradigm provides generalizable reference value for balancing optimization in multi-branch/multi-path models.
- The conceptual distinction of "topological purification vs. denoising" merits further research: denoising attempts to reconstruct the original signal, while purification maps the representation back to the correct manifold.
- Future research can consider integrating topological constraints directly into the training objectives of OCR models (a direction suggested by the authors).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of topological autoencoders and adversarial purification is highly novel, and the Freeze-Flow training scheme is creative.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers a wide range of attacks, but the datasets are relatively simple, lacking validation on natural images.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear presentation with proper background on topology, though some descriptions are verbose.
- **Value**: ⭐⭐⭐⭐ — Points out a new path using TDA for adversarial defense, though its practicality remains to be fully verified in more complex scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[CVPR 2026\] Mitigating Error Amplification in Fast Adversarial Training](../../CVPR2026/ai_safety/mitigating_error_amplification_in_fast_adversarial_training.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](../../NeurIPS2025/ai_safety/keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)

</div>

<!-- RELATED:END -->
