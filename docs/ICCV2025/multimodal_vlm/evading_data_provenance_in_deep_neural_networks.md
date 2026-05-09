---
title: >-
  [Paper Note] Evading Data Provenance in Deep Neural Networks
description: >-
  [ICCV 2025][Multimodal VLM][Data Provenance] This paper exposes the false sense of security in existing Dataset Ownership Verification (DOV) methods. Through a unified evasion framework, Escaping DOV, task-relevant but identity-free knowledge is transferred from a teacher model to a surrogate student via OOD data, successfully bypassing all 11 evaluated DOV methods simultaneously.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Data Provenance
  - Dataset Ownership Verification
  - Evasion Attack
  - Knowledge Distillation
  - Backdoor Watermarking
date: 2026-05-08
content_hash: 50d33aa41d57a420
---

# Evading Data Provenance in Deep Neural Networks

**Conference**: ICCV 2025
**arXiv**: [2508.01074](https://arxiv.org/abs/2508.01074)
**Code**: [GitHub](https://github.com/dbsxfz/EscapingDOV)
**Area**: Data Security / Model Copyright
**Keywords**: Data Provenance, Dataset Ownership Verification, Evasion Attack, Knowledge Distillation, Backdoor Watermarking

## TL;DR

This paper exposes the false sense of security in existing Dataset Ownership Verification (DOV) methods. Through a unified evasion framework, Escaping DOV, task-relevant but identity-free knowledge is transferred from a teacher model to a surrogate student via OOD data, successfully bypassing all 11 evaluated DOV methods simultaneously.

## Background & Motivation

Modern deep learning relies heavily on large-scale datasets, many of which are copyright-protected or contain sensitive information. DOV has emerged as a post hoc mechanism to trace unauthorized model training, and has evolved into three major categories:

**Backdoor Watermarking**: Embeds "backdoor" samples into the dataset so that the trained model exhibits predefined behavior on trigger inputs.

**Poison-free Watermarking**: Fine-tunes samples to shift model confidence, verified through hypothesis testing.

**Dataset Fingerprinting**: Exploits inherent properties of training data (e.g., decision boundary distances) for verification.

The paper's central finding is that **prior work has evaluated evasion attacks under overly simplistic settings, creating a false sense of security**. Existing evasion methods are limited to simple regularization and generic backdoor defenses (e.g., fine-tuning, pruning), and are unable to counter advanced DOV schemes such as invisible watermarks, poison-free watermarks, and dataset fingerprints.

**Key Insight**: Verification behaviors across all DOV methods share two common properties — **exclusivity** (they are specific to a particular dataset and are not triggered by other data) and **stealthiness** (they do not interfere with the main task semantics, acting as side-channel signals). Based on this observation, a unified evasion strategy can be designed: using OOD data as an intermediary to transfer knowledge while naturally filtering out these exclusive and stealthy verification signals.

## Method

### Overall Architecture

The pipeline of Escaping DOV proceeds as follows:
1. **Teacher Training**: Train a teacher model $f_{\theta_t}$ directly on the copyrighted dataset $\mathcal{D}$ (inevitably marked in the process).
2. **Transfer Set Curation**: Select an optimal subset $\mathcal{T}$ from an OOD image gallery $\mathcal{G}$ (e.g., ImageNet).
3. **Selective Knowledge Transfer**: Use $\mathcal{T}$ as an intermediary to distill task-oriented but identity-free knowledge from the teacher into a surrogate student $f_{\theta_s}$.

### Key Designs

1. **Transfer Set Curation**: Leverages a VLM (MobileCLIP) and an LLM (GPT-4o mini) for reliable sample selection, following the core pipeline:

    - **Zero-Shot Classification**: The LLM generates per-class description sets to enhance the VLM's zero-shot classification capability (avoiding fine-tuning the VLM on copyrighted data, which would cause it to be marked). Gallery samples are assigned to $K$ class buckets according to VLM predictions.
    - **Distribution Distance Ranking**: The VLM image encoder projects the copyrighted dataset $\mathcal{D}$ into feature space, where per-class density centroids $\text{Cent}_t$ are computed. Samples within each bucket are ranked by distance to the corresponding centroid.
    - **Consensus Filtering**: Only samples on which the teacher model and the VLM agree are retained (ensuring the teacher's predictions are grounded in true semantics rather than verification behavior), until the per-class count reaches $|\mathcal{D}_t|$.

   This ensures the transfer set contains virtually no samples capable of triggering verification behaviors.

2. **Selective Knowledge Transfer (SKT)**: Standard distillation risks propagating verification behaviors through "dark knowledge" in soft labels. SKT suppresses this transfer by generating worst-case perturbations for the teacher model:

    - **Perturbation Pool Generation**: Universal adversarial perturbations $\delta$ are generated using the teacher model on $\mathcal{D}$, adaptively selecting the norm constraint ($L_0$, $L_2$, or $L_\infty$) that maximizes the loss. These are precomputed offline.
    - **Corruption Chain Generation**: A genetic algorithm searches over 15 corruption types from ImageNet-C to find the optimal combination and ordering, forming unconstrained perturbations.
    - **Adversarial Distillation**: With probability $\beta$, perturbations or corruptions are applied to transfer set samples before being fed to the student, while the teacher still produces outputs from clean samples, forcing the student to maintain invariance to verification behaviors:

$$\arg\min_{\theta_s} \mathcal{L}\left(\frac{f_{\theta_s}(x)}{\tau}, \frac{f_{\theta_t}(x)}{\tau}\right) + \beta \cdot \mathcal{L}\left(\frac{f_{\theta_s}(A(x))}{\tau}, \frac{f_{\theta_t}(x)}{\tau}\right)$$

### Loss & Training

Standard KL divergence distillation loss is used with temperature $\tau=1$. The core intuition behind SKT is that task knowledge and verification knowledge have different "boiling points" and can be naturally separated at low temperature. Both the perturbation pool and the corruption chain are generated offline, incurring only marginal overhead at training time.

## Key Experimental Results

### Main Results

Evasion results against 11 DOV methods on CIFAR-10 and Tiny ImageNet:

| DOV Method | Original VSR/p-value (CIFAR-10) | After Evasion VSR/p-value | After Evasion ACC |
|---|:---:|:---:|:---:|
| BadNets | 100.00% | **1.36%** | 93.46% |
| UBW (untargeted) | 95.54% | **1.74%** | 93.41% |
| Narcissus (invisible) | 87.34% | **4.59%** | 94.37% |
| Radioactive Data | 3.03e-3 | **9.45e-1** | 94.07% |
| ANW | 1.37e-9 | **1.00e+0** | 93.91% |
| Domain Watermark | 1.67e-22 | **1.00e+0** | 93.90% |
| Dataset Inference | 1.87e-3 | **4.76e-1** | 93.97% |
| MeFA | 2.62e-14 | **1.00e+0** | 93.93% |

All DOV methods are successfully evaded (VSR approaches random guessing; p-values far exceed the 0.01 threshold), with accuracy degradation below 1%.

### Ablation Study

Comparison with 9 state-of-the-art evasion methods on CIFAR-10 across the 4 most challenging DOV schemes:

| Method | BadNets ACC/VSR | Narcissus ACC/VSR | Isotope p-value | Dataset Inf. p-value |
|---|:---:|:---:|:---:|:---:|
| Fine-pruning | 86.21/99.37 | 86.58/43.75 | 0.0532 | 0.0095 |
| BCU | 92.75/1.56 | 92.33/61.72 | 0.1622 | 0.0773 |
| ABD | 84.94/7.07 | 85.42/22.10 | **0.0066** | 0.3696 |
| **Escaping DOV** | **93.46/1.36** | **94.37/4.59** | **0.2845** | **0.4759** |

Escaping DOV is the only attack capable of **simultaneously** evading all 4 DOV methods while consistently achieving the highest accuracy. Although ABD is also effective against Narcissus, it fails to evade Isotope (p-value = 0.0066 < 0.01).

### Key Findings

- A **loss barrier** exists between the teacher and student parameters, indicating that the student develops a fundamentally different prediction mechanism.
- The teacher's test/train loss ratio exceeds $10^3$ (making it easily detectable by fingerprinting), while the student's ratio remains below 2.
- Effect of distillation temperature $\tau$: without SKT, higher temperature causes VSR to rise sharply; with SKT, VSR remains stable across temperatures.
- The framework is effective on domain-specific datasets (facial emotion recognition RAFDB, medical imaging OrganCMNIST).
- Training the teacher with mixup further enhances evasion performance.

## Highlights & Insights

- **Deep insight from a unified perspective**: The verification signals of all DOV methods are "exclusive and stealthy," which is both the source of their strength and their fundamental vulnerability — OOD data naturally cannot trigger these signals.
- **Asymmetry between attack and defense**: Constructing a robust DOV scheme requires satisfying multiple simultaneous constraints (not disrupting legitimate training, being undetectable, being non-bypassable), whereas an adversary only needs to break one.
- **Practical value**: The framework serves as a security evaluation tool to help DOV developers more rigorously test the robustness of their methods.
- The finding that task knowledge and verification knowledge naturally separate at $\tau=1$ carries theoretical significance.

## Limitations & Future Work

- Successful evasion depends on access to a sufficiently large and diverse OOD image gallery.
- Robustness against adaptive defenders (e.g., semantic backdoors, anti-distillation backdoors) is limited.
- The study focuses on image classification and does not extend to other modalities such as LLMs or text-to-image generation.
- While valuable for security research, the findings could also be exploited by adversaries — a double-edged sword effect.
- Transfer set curation relies on the capabilities of CLIP and GPT-4o mini, which may themselves introduce biases.

## Related Work & Insights

- The application of knowledge distillation to backdoor defense (NAD, BCU, ABD) is the most closely related line of work; however, these methods distill using a small set of clean data rather than OOD transfer.
- The distinction from model watermark removal (IPRemoval) lies in the fact that this work targets dataset-level provenance rather than model-level provenance.
- The VLM + LLM-assisted transfer set selection strategy is extensible to areas such as dataset distillation.
- The proposed evasion framework establishes a more reliable evaluation standard for the DOV research community.

## Rating

- **Novelty**: 8/10 — First unified evasion framework with deep and well-motivated insights.
- **Technical Quality**: 8/10 — Solid experiments covering 11 DOV methods and 9 baselines.
- **Value**: 7/10 — Double-edged: serves as both a security evaluation tool and a potential attack enabler.
- **Writing Quality**: 8/10 — Problem motivation and method design are presented with clear logical structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](../../AAAI2026/multimodal_vlm/information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[NeurIPS 2025\] Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems](../../NeurIPS2025/multimodal_vlm/hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)

</div>

<!-- RELATED:END -->
