---
title: >-
  [Paper Note] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels
description: >-
  [NeurIPS 2025][Model Compression][Semi-supervised learning] ZeroMatch proposes a two-stage framework that combines the zero-shot pseudolabels of foundation models with semi-supervised learning: first initializing the student model with knowledge distillation, and then performing SSL training with an auxiliary KD loss to prevent catastrophic forgetting. It consistently outperforms standard SSL and zero-shot enhancement methods across six vision/NLP benchmarks.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Semi-supervised learning"
  - "Knowledge distillation"
  - "Zero-shot pseudolabels"
  - "Foundation models"
  - "Small model training"
date: 2026-05-08
content_hash: b966bde1c1328113
---

# Enhancing Semi-supervised Learning with Zero-shot Pseudolabels

**Conference**: NeurIPS 2025  
**arXiv**: [2502.12584](https://arxiv.org/abs/2502.12584)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Semi-supervised learning, Knowledge distillation, Zero-shot pseudolabels, Foundation models, Small model training

## TL;DR
ZeroMatch proposes a two-stage framework that combines the zero-shot pseudolabels of foundation models with semi-supervised learning: first initializing the student model with knowledge distillation, and then performing SSL training with an auxiliary KD loss to prevent catastrophic forgetting. It consistently outperforms standard SSL and zero-shot enhancement methods across six vision/NLP benchmarks.

## Background & Motivation

**Background**: Semi-supervised learning (SSL) reduces annotation costs by leveraging unlabeled data, and methods like FixMatch/AdaMatch have achieved excellent performance through consistency regularization and pseudolabeling. Meanwhile, the zero-shot capabilities of foundation models (FMs) provide an additional source of supervision signals.

**Limitations of Prior Work**: (a) Directly fine-tuning large FMs is impractical for resource-constrained users (e.g., personal devices with only a single GPU); (b) Naively using FM zero-shot predictions as pseudolabels to train small models may degrade performance due to pseudolabel noise or domain mismatch; (c) Existing KD methods either only use teacher outputs or only use labeled data, missing out on complementarity.

**Key Challenge**: Labeled data provides accurate but scarce supervision, whereas FM pseudolabels provide extensive but noisy supervision. How to leverage both simultaneously in resource-constrained scenarios is a key challenge.

**Goal**: To design a unified framework that jointly utilizes labeled data, unlabeled data, and FM pseudolabels to train compact student models.

**Key Insight**: KD and SSL improve predictions on unlabeled data from different sources: KD leverages teacher predictions, whereas SSL leverages labeled data. The two are complementary and can be combined.

**Core Idea**: Two-stage training: first perform KD to establish initial high-confidence predictions, and then jointly train with SSL and auxiliary KD to prevent forgetting.

## Method

### Overall Architecture
Inputs: Labeled set $\mathcal{D}_L$, unlabeled set $\mathcal{D}_U$, and FM-generated pseudolabels $\hat{y}^L, \hat{y}^U$. Output: The trained compact student model $f$. The training consists of two stages: Stage 1 for knowledge distillation warmup, and Stage 2 for semi-supervised learning with auxiliary KD.

### Key Designs

1. **Stage 1: Knowledge Distillation Warmup**:

    - **Function**: Training the student model using FM pseudolabels as teacher outputs.
    - **Mechanism**: Performing standard KD on pseudolabels of all data (including labeled inputs): $\mathcal{L}_{KD} = \frac{1}{N}(\sum_{i=1}^{N_L}\mathcal{H}(\hat{y}_i^L, \mathbf{p}(y|x_i)) + \sum_{i=1}^{N_U}\mathcal{H}(\hat{y}_i^U, \mathbf{p}(y|u_i)))$
    - **Design Motivation**: Providing high-quality initial predictions for the SSL stage. Standard SSL suffers from low data utilization early in training due to weak student models; KD warmup allows more unlabeled samples to surpass the confidence threshold and be utilized from the very beginning.

2. **Stage 2: SSL Training with Auxiliary KD**:

    - **Function**: Retaining knowledge learned from the teacher while performing standard SSL training.
    - **Mechanism**: The student model consists of an encoder $g(\cdot)$ + main head $h(\cdot)$ + auxiliary head $h_p(\cdot)$. The main head optimizes the SSL objective ($\mathcal{L}_s + \mathcal{L}_u$), while the auxiliary head optimizes the KD loss: $\mathcal{L}_{KD_2} = \frac{1}{B}(\sum \mathcal{H}(\hat{y}_i^L, \mathbf{q}(y|x_i)) + \sum \mathcal{H}(\hat{y}_i^U, \mathbf{q}(y|u_i)))$
    - **Total Loss**: $\mathcal{L}_{KD\text{-}SSL} = \mathcal{L}_s + \mathcal{L}_u + \alpha_t \cdot \lambda_p \mathcal{L}_{KD_2}$
    - **Design Motivation**: In low-label scenarios, SSL is prone to generating inaccurate pseudolabels that overwrite knowledge learned in Stage 1 (catastrophic forgetting). The auxiliary KD head shares the encoder but remains independent of the main head, ensuring continuous flow of teacher knowledge without directly interfering with SSL decisions.

3. **Annealing**:

    - **Function**: $\alpha_t$ linearly increases from 0 to 1, controlling the weight of the auxiliary KD loss.
    - **Mechanism**: Letting the SSL objective dominate the early training phase to fully utilize labeled data, and gradually introducing KD in the late phase to stabilize training.
    - **Design Motivation**: Avoiding low-quality pseudolabels from dominating the training at the start, thereby achieving a "soft start" integration of knowledge.

### Loss & Training
- Employs identical hyperparameters as AdaMatch to ensure a fair comparison.
- $\alpha_p = 1$ (annealing enabled), $\lambda_p = 1$ (unified across all experiments).
- Uses ViT-Small for vision and BERT-Base for NLP; trained on a single A5000 24GB GPU.
- The auxiliary head has the same architecture as the main head (an MLP classification head).

## Key Experimental Results

### Main Results
NLP datasets (using GPT-4o pseudolabels):

| Dataset | Labels | AdaMatch | Zero-shot | Pseudo-sup | ZeroMatch |
|--------|--------|----------|-----------|------------|-----------|
| Yahoo | 250 | 64.81 | 68.81 | 67.68 | **70.90** |
| Yahoo | 2000 | 69.42 | 68.81 | 67.56 | **72.09** |
| AG News | 40 | 85.21 | 86.25 | 86.33 | **88.70** |
| Amazon | 250 | 52.39 | 59.14 | 56.65 | **59.82** |

Vision datasets (using GPT-4.1 pseudolabels):

| Dataset | Labels | AdaMatch | Zero-shot | Pseudo-sup | ZeroMatch |
|--------|--------|----------|-----------|------------|-----------|
| CIFAR100 | 100 | 71.43 | 83.25 | 84.84 | **88.01** |
| Flowers102 | 204 | 86.71 | 88.37 | 85.40 | **95.17** |
| Resisc45 | 90 | 78.87 | 79.28 | 79.59 | **87.83** |

### Ablation Study
Effects of various ZeroMatch components (CIFAR100, 100 labels):

| Configuration | Accuracy | Description |
|------|---------|------|
| AdaMatch (no pseudolabels) | 71.43 | SSL baseline |
| Direct Zero-shot | 83.25 | FM inference without training |
| Pseudo-supervision | 84.84 | Treating pseudolabels as ground-truth labels |
| PL feature input | 72.81 | Using pseudolabels as additional features |
| ZeroMatch (Full) | **88.01** | Two-stage KD+SSL |

### Key Findings
- ZeroMatch consistently outperforms all baselines across all six datasets, with a larger advantage in extremely low-labeled scenarios (e.g., CIFAR100 with only 1 sample per class).
- Robust to low-quality pseudolabels: Performance does not degrade even when using weaker teachers like FLAN-T5.
- Pseudo-supervision performs even worse than AdaMatch on NLP, indicating that directly training on noisy pseudolabels can be harmful.
- The performance gain is larger in vision (CIFAR100: +16.6, Flowers102: +8.5) due to the higher quality of GPT-4.1 visual pseudolabels.

## Highlights & Insights
- **Elegant Two-Stage Design**: The approach of establishing initial knowledge in Stage 1 and preventing forgetting with an auxiliary head in Stage 2 is simple and effective. It does not require modifying the underlying SSL algorithm, ensuring strong generalizability.
- **Auxiliary Head Design**: Cleverly decouples the two objectives of KD and SSL. The main head focuses on the downstream task, while the auxiliary head retains teacher knowledge, passing it through the shared encoder without conflict.
- **Practical Value**: Only requires the FM inference API (a one-time call) and bypasses the need to fine-tune the FM itself, making it suitable for closed-source models while avoiding data leakage risks.

## Limitations & Future Work
- The format of pseudolabels is limited to one-hot for classification tasks; generative or regression tasks are not explored.
- The auxiliary KD head is discarded after training, resulting in parameter waste.
- Does not account for confidence variations across different samples in FM (all pseudolabels are treated with equal weight).
- The annealing strategy is a simple linear increase, which may not be the optimal schedule.
- It is only integrated with AdaMatch; compatibility with more recent SSL methods like FreeMatch has not been validated.

## Related Work & Insights
- **vs FixMatch/AdaMatch**: Standard SSL does not utilize external knowledge, whereas ZeroMatch expands the sources of supervision via FM pseudolabels.
- **vs GRIP/CPL/FineSSL**: These methods require fine-tuning the FM itself, which incurs high computational overhead, whereas ZeroMatch only requires FM inference.
- **vs Pseudo-supervision**: Simply treating pseudolabels as ground-truth cannot handle noise, whereas the progressive integration in ZeroMatch is more robust.

## Rating
- Novelty: ⭐⭐⭐ The approach is a natural combination of KD and SSL. Although preventing forgetting with an auxiliary head is effective, it is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively compared across six datasets, various teacher qualities, and different scales of labeled data.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth methodological description, and well-organized experimental setup.
- Value: ⭐⭐⭐⭐ Possesses clear practical value for leveraging FM knowledge in low-resource scenarios. The method is simple and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ACL 2025\] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](../../ACL2025/model_compression/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

</div>

<!-- RELATED:END -->
