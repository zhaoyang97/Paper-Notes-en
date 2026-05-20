---
title: >-
  [Paper Note] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models
description: >-
  [AAAI 2026][AI Safety][Adversarial Defense] This paper proposes TopoReformer, a model-agnostic adversarial purification pipeline based on a topological autoencoder. By leveraging persistent homology to enforce topologica…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Adversarial Defense"
  - "Topological Autoencoder"
  - "OCR Security"
  - "Persistent Homology"
  - "Manifold Purification"
date: 2026-05-08
content_hash: 2dce223cc10a57ea
---

# TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models

**Conference**: AAAI 2026
**arXiv**: [2511.15807](https://arxiv.org/abs/2511.15807)  
**Code**: [github](https://github.com/invi-bhagyesh/TopoReformer)  
**Area**: AI Security
**Keywords**: Adversarial Defense, Topological Autoencoder, OCR Security, Persistent Homology, Manifold Purification

## TL;DR

This paper proposes TopoReformer, a model-agnostic adversarial purification pipeline based on a topological autoencoder. By leveraging persistent homology to enforce topological consistency in the latent space, the method filters adversarial perturbations without adversarial training, effectively protecting OCR systems against classical attacks, adaptive attacks, and OCR-specific watermark attacks.

## Background & Motivation

OCR systems are widely deployed in high-stakes scenarios such as document automation, license plate recognition, and compliance auditing. However, deep learning-based OCR models inherit the adversarial vulnerability of DNNs—imperceptible perturbations can cause severe transcription errors and even survive physical-world transformations (print-and-scan, photography).

Existing defense strategies fall into four categories:

**Preprocessing/Denoising Purification** (e.g., MagNet, PuVAE): maps inputs back to a learned data manifold, but often degrades performance on unperturbed inputs.

**Anomaly Detection**: relies on autoencoder reconstruction errors or distributional tests, but is fragile against adaptive attacks.

**Adversarial Training**: computationally expensive and tied to specific attack types.

**Post-processing Correction**: patches text outputs after the fact, addressing symptoms rather than causes.

A shared limitation across these approaches is that they are **model-specific, computationally costly, and unable to withstand unknown or adaptive attacks**. More critically, many defenses have been shown to provide only "false security" via gradient obfuscation, completely failing under adaptive attacks such as BPDA and EOT.

**Core Insight**: Adversarial perturbations typically alter only local pixel relationships while leaving the global topological structure of the data (connectivity, loops, holes) intact. Enforcing topological invariance during encoding naturally discards topologically irrelevant variations—a process of "purification" rather than "denoising"—which constitutes the central contribution of this paper.

## Method

### Overall Architecture

TopoReformer is a three-stage cascaded pipeline:

1. **Topological Autoencoder (TopoAE)**: constrains the latent space with a persistent homology loss to perform topology-level purification.
2. **Reformer (VAE)**: aligns the TopoAE output to the manifold expected by the downstream classifier.
3. **Auxiliary Module**: injects the TopoAE latent vector into the bottleneck layer of the Reformer to supply topological information.

The entire pipeline is **model-agnostic**—it serves as a plug-and-play preprocessing module that can be placed before any OCR model without modifying or retraining the downstream model.

### Key Designs

#### 1. Topological Autoencoder (TopoAE) and Persistent Homology Loss

**Mechanism**: Persistent homology diagrams are computed for both input space $X$ and latent space $Z$, and topological discrepancies between the two are penalized.

Persistent homology tracks the birth and death of topological features (connected components, loops, voids) across scale parameters, producing a stable, perturbation-robust structural summary of the data.

The topological loss is defined as a bidirectional matching:

$$L_t = L_{X \to Z} + L_{Z \to X}$$

where:
- $L_{X \to Z} = \frac{1}{2} \|A_X^{\pi_X} - A_Z^{\pi_X}\|^2$: requires the latent space to be consistent under topological pairings from the input space.
- $L_{Z \to X} = \frac{1}{2} \|A_Z^{\pi_Z} - A_X^{\pi_Z}\|^2$: requires the input space to be consistent under topological pairings from the latent space.

The total loss is:

$$L = L_{rec}(X, \hat{X}) + \lambda L_t$$

**Design Motivation**: Unlike pixel-level denoising, the topological loss enforces global structural consistency. Variations in the data that are topologically irrelevant—such as adversarial perturbations—are naturally discarded during encoding. TopoAE is trained exclusively on clean data and requires no exposure to adversarial examples.

#### 2. Reformer (VAE) and Classifier Alignment

While the TopoAE output is topologically purified, its manifold may not match the input distribution expected by the downstream OCR classifier. The Reformer is a lightweight VAE responsible for aligning the purified image to the classifier's expected manifold.

Reformer training objective:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$$

- $\mathcal{L}_{MSE}$: pixel-level reconstruction loss between the TopoAE output and the VAE reconstruction.
- $\mathcal{L}_{CE}$: cross-entropy loss on the classifier's prediction over the VAE reconstruction, ensuring classification utility.
- $\mathcal{L}_{KL}$: KL divergence regularization.

**Design Motivation**: Pure topological purification may preserve the correct structure while diverging from the classifier's input distribution; the Reformer bridges this manifold mismatch.

#### 3. Freeze-Flow Training Paradigm and Auxiliary Module

The auxiliary module receives the TopoAE latent vector via a learned projection network, using topology-aware latent information to assist the Reformer in producing more accurate predictions.

However, in naive joint training, the model tends to rely solely on the purified image path, causing the auxiliary path to receive insufficient gradients.

The Freeze-Flow training paradigm proceeds as follows:
1. **Freeze phase**: The Reformer VAE encoder is frozen, forcing gradient flow toward the auxiliary module.
2. **Unfreeze phase**: After a warm-up period, the VAE decoder is unfrozen and both paths are trained jointly.

**Design Motivation**: This ensures the auxiliary path establishes meaningful latent representations before the main path dominates optimization, balancing learning across both branches. Experiments show that Freeze-Flow yields an additional ~5% classification improvement under Carlini–Wagner attacks.

### Loss & Training

- TopoAE is pretrained independently on clean samples until convergence; its weights are then frozen for inference only.
- The Reformer and auxiliary module are trained on TopoAE outputs using the objective $\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$.
- Training hyperparameters: $\lambda_1=1, \lambda_2=0.5, \lambda_3=0.5$; Adam optimizer with lr=0.001.
- Classifier weights remain frozen throughout.

## Key Experimental Results

### Main Results

**Ablation Under Classical Attacks (MNIST/EMNIST, F1-score %)**

| Attack | Defense Config | MNIST (Weak/Strong) | EMNIST (Weak/Strong) |
|--------|---------------|---------------------|----------------------|
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
|--------|------------------|-------------------|
| EOT | 9.19 / 90.73 | 28.32 / 73.28 |
| EOT+BPDA | 36.59 / 64.71 | 44.26 / 58.92 |
| BPDA | 81.14 / 15.65 | 84.46 / 12.77 |

**OCR-Specific Attack (FAWA Watermark Attack)**

| OCR Model | No Defense ASR / Acc | With Defense ASR / Acc |
|-----------|----------------------|------------------------|
| CRNN | 100 / 48.13 | 78.83 / 71.00 |
| Rosetta | 99.83 / 69.66 | **44.08 / 85.98** |
| TRBA | 99.83 / 46.68 | 60.75 / 80.26 |

### Ablation Study

| Component | C&W (MNIST, Strong) F1 | Gain |
|-----------|------------------------|------|
| No Defense | 4.30% | — |
| + TopoAE | 48.51% | +44.21 |
| + Reformer | 67.93% | +19.42 |
| + Auxiliary | 72.41% | +4.48 |
| + Freeze-Flow Warmup | **75.15%** | +2.74 |

The stepwise ablation validates the contribution of each component; the Freeze-Flow training paradigm provides a significant gain under strong attacks.

### Key Findings

1. **Greatest effect against C&W attacks**: C&W generates fine-grained, low-magnitude perturbations that are precisely the type best filtered by topological purification, with F1 rising from 4.30% to 75.15%.
2. **Robust against EOT adaptive attacks**: ASR drops from 99.05% to 9.19%, indicating that the defense does not rely on simple gradient masking.
3. **No degradation on clean inputs**: Accuracy on unperturbed samples remains ~98% on MNIST and ~94% on EMNIST with negligible loss.
4. **Generalizes across OCR architectures**: Effective for both CTC-based (CRNN, Rosetta, STAR-Net) and attention-based (RARE, TRBA) models.

## Highlights & Insights

- **Paradigm Innovation**: This is the first application of a topological autoencoder to adversarial defense/input purification. Unlike conventional denoising—which attempts to recover the original signal—this approach performs "topological purification" by discarding topologically irrelevant variations.
- **No adversarial samples required during training**: The model is trained entirely on clean data and generalizes naturally to unseen attacks.
- **Elegant Freeze-Flow training**: By controlling gradient flow, the paradigm resolves the common problem of underfitting in auxiliary branches of multi-path models.
- **Implicit Lipschitz smoothness**: Topological constraints implicitly bound the sensitivity of latent representations to input perturbations, providing a source of robustness that does not rely on explicit gradient regularization.

## Limitations & Future Work

1. **Remaining vulnerability to BPDA**: BPDA can exploit local curvature to circumvent global topological smoothness, with ASR still reaching 81%.
2. **Limited evaluation datasets**: Validation is conducted only on simple datasets such as MNIST and EMNIST; no experiments on natural images (e.g., CIFAR-10, ImageNet) are included.
3. **Reformer omitted in OCR evaluation**: The Reformer is intentionally excluded from OCR deployment for efficiency, but this limits the defense ceiling in OCR scenarios.
4. **No computational cost analysis**: Runtime overhead of persistent homology computation is not reported, which may be a bottleneck in large-scale deployment.
5. **No comparison with diffusion-based purification**: Recent diffusion-based adversarial purification methods (e.g., DiffPure) represent strong baselines that are not evaluated against.

## Related Work & Insights

- The application of topological data analysis (TDA) to adversarial robustness is an emerging direction; this paper represents the first attempt to use TopoAE for adversarial purification.
- The Freeze-Flow training paradigm offers general reference value for balancing optimization in multi-branch and multi-path models.
- The conceptual distinction between "topological purification" and "denoising" merits deeper investigation: denoising aims to recover the original signal, whereas purification maps inputs to the correct manifold.
- Integrating topological constraints into OCR model training objectives is a promising direction proposed by the authors as future work.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of topological autoencoder and adversarial purification is highly original; the Freeze-Flow training is also creative.
- Experimental Thoroughness: ⭐⭐⭐ — Attack coverage is broad but datasets are simple; natural image validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear and well-structured with adequate topological background, though some passages are verbose.
- Value: ⭐⭐⭐⭐ — Opens a new direction for TDA-based adversarial defense, though practical utility requires validation in more complex settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)
- [\[AAAI 2026\] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)

</div>

<!-- RELATED:END -->
