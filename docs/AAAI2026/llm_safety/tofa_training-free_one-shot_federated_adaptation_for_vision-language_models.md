---
title: >-
  [Paper Note] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models
description: >-
  [AAAI 2026][LLM Safety][Federated Learning] TOFA is a federated learning framework that adapts CLIP via hierarchical Bayesian inference of personalized visual prototype distributions…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "CLIP Adaptation"
  - "One-Shot FL"
  - "Hierarchical Bayes"
  - "Training-Free"
date: 2026-05-08
content_hash: 505ecdfedb421c86
---

# TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.16423](https://arxiv.org/abs/2511.16423)  
**Code**: To be confirmed  
**Area**: Multimodal VLM
**Keywords**: Federated Learning, CLIP Adaptation, One-Shot FL, Hierarchical Bayes, Training-Free

## TL;DR
TOFA is a federated learning framework that adapts CLIP via hierarchical Bayesian inference of personalized visual prototype distributions, globally aligned LLM-based text augmentation, and adaptive modality fusion — achieving training-free, single-round communication adaptation that outperforms one-shot baselines and even some multi-round training methods across 9 datasets.

## Background & Motivation
**Background**: Adapting VLMs such as CLIP in federated learning (FL) has attracted growing attention. Existing approaches primarily rely on prompt learning (PromptFL, pFedPrompt) or fine-tuning for downstream task adaptation, but require multiple rounds of client–server communication.

**Limitations of Prior Work**:
   - **High communication overhead**: Multi-round interaction incurs substantial communication costs and demands long-term system stability.
   - **Insufficient computational resources**: Many mobile clients and low-capacity servers cannot support model training.
   - **One-shot methods ill-suited for VLMs**: Existing one-shot FL methods (e.g., FedLPA, FENS) are designed for traditional models and do not effectively leverage the multimodal information in VLMs.
   - **Data heterogeneity**: Non-IID data distributions cause misalignment between local and global optimization objectives.

**Key Challenge**: How to fully exploit the multimodal information of VLMs while handling data heterogeneity under the strict constraints of zero training and a single communication round.

**Goal**: Design a training-free, one-shot FL framework for high-quality VLM adaptation.

**Key Insight**: Extract complementary information from two parallel pipelines — a visual pipeline using Bayesian inference for personalized prototypes, and a textual pipeline using LLM augmentation with global alignment for robust text representations.

**Core Idea**: Combine hierarchical Bayesian inference for personalized visual distribution estimation, LLM-augmented text with global alignment, and confidence-based modality fusion to achieve training-free, single-round VLM adaptation in FL.

## Method

### Overall Architecture
TOFA consists of three modules:
- **Visual Pipeline**: Each client computes local visual statistics → uploads to server → server derives global class prototype distributions using an uninformative prior → distributions are sent back to clients → each client uses the global distribution as a prior to infer a personalized local posterior → GDA-based classification.
- **Textual Pipeline**: Each client generates augmented text descriptions using a local LLM → computes importance scores for each text prompt → server performs global alignment to select robust text prompts → prompts are combined via weighted aggregation for classification.
- **Adaptive Fusion**: Sample-wise confidence-based fusion dynamically balances visual and textual predictions.

### Key Designs

1. **Collaborative Prototype Distribution Learning (Visual Pipeline)**:

    - **Function**: Learn personalized class-specific visual prototype distributions for each client.
    - **Mechanism**: CLIP visual features are assumed to follow a class-specific Gaussian distribution $\mathcal{N}(\mathbf{w}_c, \mathbf{\Sigma})$. A **hierarchical Bayesian model** is employed: the global posterior $q(\theta) = \pi(\theta|D)$ is computed by aggregating local statistics from all clients; this global posterior then serves as an informative prior, and combined with local data, yields a personalized posterior $q(\theta^k) \propto L(D^k|\theta^k)[L(D|\theta)]^\alpha \pi(\theta)$, where the power prior parameter $\alpha$ controls the influence of global information. A Normal-Inverse-Wishart conjugate prior ensures a closed-form posterior, eliminating the need for iterative optimization. GDA is used for final classification.
    - **Design Motivation**: Using only mean prototypes discards distributional variance information. The Bayesian framework naturally trades off between global generalization and local personalization, with $\alpha$ controlling this balance. The conjugate prior enables one-shot feasibility by avoiding iterative inference.

2. **Globally Aligned Text Augmentation (Textual Pipeline)**:

    - **Function**: Select robust and generalizable text prompts from LLM-generated class descriptions.
    - **Mechanism**: Each client uses an LLM to generate dataset-aware class descriptions $\{t_c^m\}_{m=1}^M$. Each client computes the classification confidence $p_c^k(t_c^m)$ of each text prompt per class. The server scores text prompts using a KL divergence-inspired importance metric $r(t_c^m) = \frac{1}{K}\sum_{k=1}^K u^k(t_c^0)\log\frac{u^k(t_c^m)}{u^k(t_c^0)}$, where $u^k$ measures the confidence in distinguishing the target class from others. High-scoring prompts exhibit stable performance across heterogeneous data environments.
    - **Design Motivation**: Manual templates such as "A photo of a {class}" are overly simplistic, while LLM-augmented descriptions introduce rich semantics but vary in quality. Global alignment ensures that selected text prompts are effective across heterogeneous clients.

3. **Adaptive Modality Fusion**:

    - **Function**: Fuse visual and textual predictions in a sample-wise manner.
    - **Mechanism**: The fusion rule is $f_M^k(\mathbf{z}) = \eta(\mathbf{z})f_V^k(\mathbf{z}) + (1-\eta(\mathbf{z}))f_T(\mathbf{z})$, with weight $\eta(\mathbf{z}) = \sigma(\log\frac{\max_j \text{softmax}(f_V^k(\mathbf{z}))_j}{\max_j \text{softmax}(f_T(\mathbf{z}))_j})$. Theorem 1 proves that when $\eta$ is proportional to the difference in loss between the two modalities, the generalization error of the fused classifier is minimized. Calibrated confidence is used as a proxy for accuracy.
    - **Design Motivation**: Different samples exhibit varying reliability across modalities — for some samples the visual modality is more accurate, while for others the textual modality is superior. A fixed weighting scheme cannot adapt to this sample-level variation.

### Loss & Training
TOFA is entirely training-free:
- The visual pipeline transmits only sufficient statistics (mean, scatter matrix, sample count).
- The textual pipeline transmits only importance scores.
- **Single communication round**: One client→server→client exchange completes the entire process.

## Key Experimental Results

### Main Results
CLIP Datasets (16-shot, 10 clients, label shift):

| Method | Training-free | One-shot | OxfordPets | Flowers102 | Food101 | Caltech101 | DTD |
|------|:---:|:---:|------|----------|--------|----------|-----|
| CoOp | ✗ | ✗ | 89.18 | 69.03 | 82.54 | 90.62 | 63.97 |
| pFedPrompt | ✗ | ✗ | 91.84 | 96.46 | 92.26 | 96.54 | 77.14 |
| Zero-Shot CLIP | ✓ | ✓ | 85.77 | 66.14 | 77.31 | 86.29 | 42.32 |
| CLIP-GDA | ✓ | ✓ | 88.81 | 91.23 | 79.05 | 92.55 | 60.64 |
| FedLPA+PromptFL | ✗ | ✓ | 83.42 | 78.60 | 74.74 | 88.69 | 52.75 |
| **TOFA** | **✓** | **✓** | **91.23** | **95.78** | **85.49** | **94.58** | **71.68** |

CIFAR-10/100 (100 clients, Dir(0.3)):

| Method | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| FedAvg | 75.10 | 42.52 |
| Zero-Shot CLIP | 87.71 | 64.92 |
| CoOp | 93.11 | 74.83 |
| **TOFA** | **93.18** | **76.63** |

### Ablation Study

| Configuration | Performance | Note |
|------|------|------|
| Visual only | Below full model | Lacks robust textual information |
| Textual only | Below full model | Lacks personalized visual information |
| w/o Global Alignment | Performance drop | LLM text quality is inconsistent |
| w/o Adaptive Fusion (fixed weight) | Performance drop | Cannot adapt to sample-level differences |
| Full TOFA | Best | Three modules are complementary |

### Key Findings
- As a training-free, one-shot method, TOFA surpasses multi-round training approaches such as CoOp and PromptFL on multiple datasets.
- TOFA remains effective under extreme heterogeneity (CIFAR-100, 100 clients, Dir(0.3)), demonstrating strong robustness.
- Competitive performance on the DomainNet feature-shift scenario confirms that the method generalizes to both label shift and feature shift settings.

## Highlights & Insights
- **Elegant use of hierarchical Bayes**: Using the global posterior as an informative local prior elegantly achieves personalization within a single communication round. The conjugate prior guarantees a closed-form solution, avoiding any iterative optimization — this is the key mathematical infrastructure enabling training-free adaptation.
- **Global alignment of text augmentation**: Rather than simply averaging text scores across clients, a KL divergence-inspired selection criterion identifies text prompts that are robust across heterogeneous environments, yielding substantially higher quality than directly using raw LLM outputs.
- **Theoretically grounded sample-wise fusion**: Theorem 1 connects the generalization error bound of the fused classifier to the mixing coefficient, providing principled justification for the confidence-based weighting rather than ad hoc design.

## Limitations & Future Work
- **Gaussian distribution assumption**: Whether CLIP features truly follow a class-conditional Gaussian distribution remains an open question; more flexible distributional assumptions may be needed in complex scenarios such as fine-grained recognition.
- **LLM consistency requirement**: All clients are required to use the same LLM version for text augmentation generation, which may be difficult to guarantee in practical FL deployments.
- **Classification-only applicability**: The GDA-based visual pipeline restricts the method to classification tasks, precluding extension to detection or segmentation.
- **Insufficient privacy analysis**: Although only statistics rather than raw data are transmitted, whether class-specific means and covariance matrices could leak private information warrants deeper investigation.

## Related Work & Insights
- **vs. PromptFL/pFedPrompt**: These multi-round training methods require repeated client–server communication and gradient computation. TOFA is entirely training-free and completes adaptation in a single round, achieving comparable or superior performance on most datasets.
- **vs. CLIP-GDA**: Both methods employ GDA, but CLIP-GDA is a purely local approach without federated aggregation. TOFA incorporates global information via hierarchical Bayes, yielding greater robustness under heterogeneous settings.
- **vs. FedLPA**: FedLPA is a one-shot method but requires client-side training and relies solely on the visual modality. TOFA is training-free and leverages both modalities, achieving substantially superior performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical Bayes, global text alignment, and adaptive fusion represents a pioneering contribution in the FL + VLM domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 9 datasets with diverse heterogeneity settings, 4 categories of baselines, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the high density of formulas somewhat reduces readability.
- Value: ⭐⭐⭐⭐ Provides a practical solution for resource-constrained federated VLM adaptation; the performance under training-free and one-shot constraints is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)

</div>

<!-- RELATED:END -->
