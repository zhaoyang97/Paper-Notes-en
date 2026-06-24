---
title: >-
  [Paper Note] Towards Multi-modal Transformers in Federated Learning
description: >-
  [ECCV2024][AI Safety][federated learning] The FedCola framework is proposed, which utilizes complementary local training and collaborative aggregation to realize cross-modal knowledge transfer for multi-modal Transformers in federated learning, effectively bridging the gap between uni-modal and multi-modal clients without requiring public data.
tags:
  - "ECCV2024"
  - "AI Safety"
  - "federated learning"
  - "Multi-modal Learning"
  - "Transformer"
  - "Vision-Language"
  - "Mixture of Experts"
date: 2026-05-08
content_hash: 04fdaf40a0aa1ad8
---

# Towards Multi-modal Transformers in Federated Learning

**Conference**: ECCV2024  
**arXiv**: [2404.12467](https://arxiv.org/abs/2404.12467)  
**Code**: [imguangyu/FedCola](https://github.com/imguangyu/FedCola)  
**Area**: AI Safety  
**Keywords**: federated learning, Multi-modal Learning, Transformer, Vision-Language, Mixture of Experts

## TL;DR

The FedCola framework is proposed, which utilizes complementary local training and collaborative aggregation to realize cross-modal knowledge transfer for multi-modal Transformers in federated learning, effectively bridging the gap between uni-modal and multi-modal clients without requiring public data.

## Background & Motivation

- Multi-modal Transformers have achieved remarkable progress across various downstream tasks, but their training demands a vast amount of high-quality paired data, which is often scattered across different institutions due to privacy regulations, forming data silos.
- Federated Learning (FL), as a privacy-preserving paradigm, enables collaborative model training without direct access to raw data. However, existing multi-modal FL studies primarily focus on horizontal scenarios (all clients share the same modalities) or vertical scenarios (different modalities of the same user are distributed across different clients).
- Both settings exclude clients that only possess unpaired uni-modal data from participating in the training. In centralized training, however, unpaired uni-modal data has been proven to significantly enhance multi-modal model performance.
- Therefore, how to transfer knowledge from uni-modal clients in FL (the transfer MFL scenario) is an important yet under-explored research direction.

## Core Problem

Two critical gaps exist in the transfer MFL scenario:

1. **Cross-modality Gap**: Uni-modal clients have never accessed data of other modalities, which causes their local models to bias heavily toward their local modality, failing to acquire cross-modal knowledge.
2. **In-modality Gap**: Even within the same modality, multi-modal clients (performing retrieval tasks) and uni-modal clients (performing classification tasks) employ different training objectives, resulting in discrepancies between their learned representations.

## Method

### Overall Architecture

FedCola leverages the unified architecture design of Transformers across multiple modalities and splits the model into three parts: (i) Embedding layers, (ii) Transformer blocks, and (iii) task heads. Different clients share Transformer blocks of a unified architecture, while the Embedding layers and task-specific heads vary according to their input modalities and training objectives.

### Complementary Local Training

To bridge the cross-modality gap, the core mechanism is designed to let uni-modal clients exploit knowledge from other modalities during local execution:

- Uni-modal clients download Transformer blocks of other modalities during local training.
- Inspired by the Mixture of Experts (MoE) paradigm, a learnable gating parameter $g$ (initialized to 0) is introduced to control the contributions of the complementary modalities.
- The output of each layer is formulated as $W_{\text{local}} x + g \cdot W_{\text{out}} x$, where $W_{\text{local}}$ denotes the local weights and $W_{\text{out}}$ denotes the weights of the other modalities.
- Initializing the gate to 0 ensures that the local modality dominates the early stages of training, while cross-modal complementary information is subsequently learned in an adaptive manner.
- **Weight Compression Trick**: After training is completed, the equivalent weight $W_{\text{local}} + g \cdot W_{\text{out}}$ is computed (exploiting the linear property of the operations). The uploaded size remains identical to the original model, thereby avoiding doubled communication overhead.

### Collaborative Aggregation

To tackle the in-modality gap, a "collaborative aggregation and decomposition" strategy is proposed:

- **Key Observation**: Self-attention layers encode common relationships between tokens and carry more general knowledge, whereas MLP layers adapt general knowledge to local tasks, carrying task-specific knowledge.
- **Selective Aggregation**: Self-attention layers undergo cross-type aggregation across different client types of the same modality, while MLP layers remain independent.
- For example, the Self-attention layers of image uni-modal clients and image-text multi-modal clients are aggregated weighted by data size, but their MLP layers are not aggregated across different client types.

### Compensation Scheme

Selective aggregation inevitably introduces layer-wise alignment issues. Since local updates are naturally coherent across all layers within a single client, aggregating only a portion of the layers breaks this coherence. The paper introduces a compensation matrix $\Omega_{\text{comp}}$ to scale the updates of non-collaborative layers by a matching coefficient, ensuring that updates across different layers stay aligned.

## Key Experimental Results

### Dataset Setup

| Client Type | Dataset | Task |
|:---:|:---:|:---:|
| Image Client | CIFAR-100 / OrganCMNIST | Classification |
| Text Client | AG NEWS / Medical Abstracts | Classification |
| Multi-modal Client | Flickr10k / COCO Captions | Cross-modal Retrieval |

### Main Results (Sum of Top-1 Recall)

| Method | Flickr (Default) | COCO (Default) |
|:---:|:---:|:---:|
| FedAvg | 81.08 | 95.42 |
| FedProx | 78.55 | 95.16 |
| CreamFL | 74.83 | 95.26 |
| FedIoT | 85.51 | 98.40 |
| **FedCola** | **91.96** | **105.10** |

Under default settings, FedCola outperforms the strongest baseline, FedIoT, by approximately 6.5 and 6.7 points, respectively.

### Ablation Study

| Collaborative Aggregation | Compensation | Complementary Training | R@1_sum |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 81.08 |
| ✓ | ✗ | ✗ | 88.70 |
| ✓ | ✓ | ✗ | 90.09 |
| ✓ | ✓ | ✓ | **91.96** |

- Collaborative aggregation contributes the most to performance gain (+7.62).
- The compensation scheme further improves results (+1.39).
- Complementary local training provides additional refinement (+1.87).

### Scales and Extensibility

Performance continues to scale up as more uni-modal datasets are incorporated (0 extra dataset: 81.08 → 1 dataset: 91.96 → 2 datasets: 93.25), proving the extensibility of the framework.

### Fairness Analysis

Shapley values are employed to quantify the exact contribution of each class of clients. The text clients show a contribution value of 6.14, whereas image clients show 4.74, indicating that the text modality contributes more significantly to multi-modal retrieval performance improvements under default settings.

## Highlights & Insights

1. **No Public Data Required**: Unlike CreamFL, which requires sharing public datasets for knowledge distillation, FedCola achieves cross-modal collaboration purely through model parameters, aligning better with the core privacy-preserving purpose of FL.
2. **Clever Use of Unified Transformer Architecture**: Transformer blocks of different modalities are treated as interchangeable/complementary experts, and cross-modal knowledge sharing is achieved through gated mixtures.
3. **Equivalent Weight Compression**: By leveraging the linearity of gating combinations, the upload communication overhead is compressed to the same size as the original model.
4. **Insightful Layer-wise Analysis**: Aggregation strategies differ between Self-attention (general knowledge) and MLP (task-specific knowledge), which is theoretically grounded and experimentally verified.
5. **Comprehensive Robustness Evaluations**: Uniform advantages are maintained across different levels of heterogeneity, participation rates, domain gaps, and client ratios.

## Limitations & Future Work

- Performance still drops when large domain gaps exist, and the paper acknowledges that large domain gaps are not fully resolved.
- Verified only on ViT-Small; larger-scale models (such as ViT-Large or multi-modal pre-trained models) have not been explored.
- Complementary training increases download communication overhead (although upload is compressed via equivalent weights), which might become a bottleneck under bandwidth-constrained scenarios.
- Only vision-language modalities are considered; other modalities like audio and video are not covered.
- Cross-modal collaboration on global aggregation is left for future work.

## Related Work & Insights

| Method | Public Data Required | Support Unpaired Uni-modal Clients | Cross-modal Knowledge Transfer | Architecture Requirements |
|:---:|:---:|:---:|:---:|:---:|
| CreamFL | Yes | Yes | Knowledge Distillation | Heterogeneous |
| FedIoT | No | Yes | Aggregation Strategy | Homogeneous |
| **FedCola** | **No**| **Yes** | **Parameter-level Mixture + Aggregation** | **Unified Transformer** |

- CreamFL relies on public data for knowledge distillation, which offers weaker privacy protection and is sensitive to public data quality.
- FedIoT only achieves cross-modal collaboration at the aggregation level; local training cannot exploit cross-modal information.
- FedCola achieves cross-modal collaboration in both local training and global aggregation stages without any additional public datasets.

## Inspirations & Connections

- **MoE Concepts in FL**: Treating models of different modalities as "experts" and mixing them with gates provides a new paradigm for heterogeneous model collaboration in FL.
- **Layer Functional Characterization**: The observation that Self-attention carries general knowledge while MLP carries task-specific knowledge can be generalized to other FL scenarios where selective aggregation is demanded.
- **Communication Efficiency**: The equivalent weight compression trick (utilizing the linear property of linear layer combinations) has general applicability and can be applied to other scenarios that require mixing multiple model weights.
- Complementary to personalized FL: Although FedCola focuses on global model performance, its strategy of merging Self-attention while keeping MLP independent is fundamentally a balance between generalization and personalization.

## Rating

- Novelty: ⭐⭐⭐⭐ — Explores Transformers in transfer MFL for the first time; complementary training and selective aggregation designs are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive across multi-setup, multi-domain gaps, ablation, and fairness analyses.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, and the method is systematically explained.
- Value: ⭐⭐⭐⭐ — Delivers a practical and extensible framework for multi-modal FL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/ai_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)

</div>

<!-- RELATED:END -->
