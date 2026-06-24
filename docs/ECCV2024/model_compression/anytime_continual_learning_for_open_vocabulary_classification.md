---
title: >-
  [Paper Note] Anytime Continual Learning for Open Vocabulary Classification
description: >-
  [ECCV2024][Model Compression][continual learning] The AnytimeCL framework is proposed to achieve open-vocabulary continual learning, allowing the model to receive samples at any time and perform inference on arbitrary label sets. This is realized by partially fine-tuning the final transformer block of CLIP and dynamically fusing predictions from both the fine-tuned and original models.
tags:
  - "ECCV2024"
  - "Model Compression"
  - "continual learning"
  - "open vocabulary"
  - "CLIP"
  - "online learning"
  - "feature compression"
date: 2026-05-08
content_hash: a5f2666807a2a8ac
---

# Anytime Continual Learning for Open Vocabulary Classification

**Conference**: ECCV2024  
**arXiv**: [2409.08518](https://arxiv.org/abs/2409.08518)  
**Code**: [GitHub](https://github.com/jessemelpolio/AnytimeCL)  
**Area**: Model Compression  
**Keywords**: continual learning, open vocabulary, CLIP, online learning, feature compression  
**Authors**: Zhen Zhu, Yiming Gong, Derek Hoiem (UIUC)

## TL;DR

The AnytimeCL framework is proposed to achieve open-vocabulary continual learning, allowing the model to receive samples at any time and perform inference on arbitrary label sets. This is realized by partially fine-tuning the final transformer block of CLIP and dynamically fusing predictions from both the fine-tuned and original models.

## Background & Motivation

- Traditional continual learning operates within discrete label spaces, where newly added labels or tasks alter the problem definition, thereby increasing learning difficulty.
- Under the open-vocabulary setting, classification is formulated as a comparison between continuous features and label embeddings. Consequently, learning becomes an incremental refinement over existing capabilities, making it highly suitable for continuous improvement.
- Although CLIP is trained on web-scale data, its performance remains suboptimal on many specific downstream tasks.
- The prior method by Zhu et al. (using a linear classifier combined with hierarchical clustering) requires saving all training samples, and its AIM weighting strategy only considers whether a sample belongs to seen classes, leading to poor performance in early stages.
- **Key Challenge**: The system needs to efficiently integrate new samples at any time and perform inference on arbitrary candidate label sets.

## Core Problem

1. How to maintain accurate predictions across the complete label set when training data is only received from a subset of classes?
2. How to efficiently integrate each newly incoming training sample within milliseconds?
3. How to prevent the loss of open-vocabulary capabilities and avoid catastrophic forgetting during the continual learning process?
4. How to compress intermediate layer features to reduce storage and computational overhead?

## Method

### Overall Architecture

The system consists of two models: a **frozen original CLIP model** and a **fine-tunable model** (where only the last transformer block is fine-tuned). Both models yield prediction probabilities, denoted as $P_o(y|x)$ and $P_t(y|x)$ respectively, for the same image. The final prediction is generated via online class-wise weighted fusion:

$$P(y|x) = \alpha_o(y) P_t(y|x) + \alpha_t(y) P_o(y|x)$$

### Partial Fine-Tuning Strategy

- Only the last transformer block (decoder) of CLIP ViT is fine-tuned, while keeping the text label embeddings fixed.
- The fine-tuned block is initialized with the original pre-trained weights.
- Keeping the label embeddings fixed helps preserve feature alignment with the text modality, alleviating overfitting to seen labels.
- When a new sample arrives, it is paired with stored instances to construct a class-balanced mini-batch (batch size $B=32$) for a single-step gradient update.

### Class-Balanced Sampling

For a given batch size $B$ and seen label set $\mathcal{Y}_t$, $\min(B-1, |\mathcal{Y}_t|)$ classes are selected uniformly, followed by uniform sampling of an equal number of instances from each class. Experiments show that class-balanced sampling performs similarly to uniform sampling, and both outperform complex strategies such as FIFO.

### "Other" Regularization Loss

To handle cases where the ground-truth label is absent from the candidate set, an "none of the above" option is introduced:

$$\mathcal{L}(x,y,\mathcal{Y}) = \mathcal{L}_{ce}(x, y, \mathcal{Y} \cup \text{other}) + \beta \mathcal{L}_{ce}(x, \text{other}, (\mathcal{Y} \cup \text{other}) \setminus y)$$

Here, "other" is modeled simply using a learnable bias term, and $\beta=0.1$. This regularization stabilizes the open-vocabulary training process.

### Online Class-wise Weighting (OCW)

As one of the core innovations, Exponential Moving Average (EMA, decay $\eta=0.99$) is employed to online estimate the accuracy of both models on each label:

$$c_t(y) = \eta \hat{c}_t(y) + (1-\eta) \mathbb{1}[y_t(x)=y]$$

Key Design: The accuracy estimates are updated with the current sample *prior* to updating the model, which prevents biased accuracy estimation of the fine-tuned model on training samples. The weights of the two models are then allocated proportionally to their estimated accuracy values:

$$\alpha_t(y) = \frac{c_t(y)}{c_t(y) + c_o(y) + \epsilon}$$

For classes unseen by the fine-tuned model, $\alpha_t(y)=0$, and the system relies entirely on the original model's predictions.

### Attention-Weighted PCA Feature Compression

- Partial fine-tuning requires caching intermediate layer features (50 tokens of 768 dimensions), which costs about 153KB per sample.
- Global PCA or VQ-VAE compression yields poor results as the network has already learned highly efficient representations.
- **Per-image PCA**: Computes PCA vectors on the tokens of each individual image, storing only 5 PCA vectors and their corresponding coefficients.
- **CLS Attention Weighting**: Uses attention values between the CLS token and each patch token as the weights for PCA.
- **Quantization**: Further quantizes float values into 8/16-bit unsigned integers.
- Ultimately, this achieves a **30-fold compression** (153KB $\rightarrow$ 5KB) with less than 1% loss in accuracy.

## Key Experimental Results

**Settings**: 8 target tasks (CIFAR100, SUN397, FGVCAircraft, etc.) + 3 novel tasks (ImageNet, UCF101, DTD), containing a total of 226,080 training samples across 1,034 classes.

| Metric | Scenario | AnytimeCL vs. Prev. SOTA |
|------|------|----------------------|
| All stages | Data Incremental | Outperforms CLIP+LinProbe (AIM) at every stage |
| All stages | Class Incremental | Outperforms prior methods at every stage; advantages in early stages are particularly significant |
| All stages | Task Incremental | Consistently outperforms prior methods, showing both online and offline improvements |
| Transfer | MTIL Task Inc. | 69.4 (zero forgetting, on par with CLIP) |
| Avg. | MTIL Task Inc. | 77.0 (+11.7 vs. CLIP) |

**Feature Compression Comparison (CIFAR100)**:

| Method | Storage/Sample | Time/Batch | Accuracy |
|------|----------|-----------|--------|
| Full image | 150.5 KB | 43.9 ms | 77.8% |
| Full features | 153.6 KB | 25.6 ms | 77.8% |
| Per-instance PCA + CLS Weighting + Quantization | 5.3 KB | 13.9 ms | 77.5% |

**Scalability**: Replacing the fine-tuned model with DINOv2 leads to a steeper performance improvement curve in later stages while retaining the zero-shot capability.

## Highlights & Insights

1. **True "anytime" capability**: Integrates new samples within milliseconds, successfully supporting data, class, and task benchmarking scenarios.
2. **Exquisite OCW weighting strategy**: Uses pre-update samples to obtain unbiased estimates of model accuracy, demonstrating a significant advantage over AIM in early phases.
3. **Fixed label embeddings + partial fine-tuning**: A simple yet effective approach to maintain open-vocabulary capabilities with zero forgetting.
4. **30x feature compression**: Per-image attention-weighted PCA provides a novel and practical approach that balances storage, speed, and privacy.
5. **Complete system design**: Full-pipeline consideration from training and inference to storage, supporting extended applications such as federated learning.

## Limitations & Future Work

- Only evaluated on classification tasks, without extending to more complex tasks like object detection or segmentation.
- The scalability experiments for tree-clustering are limited in scale, lacking validation on millions of samples.
- The degree of privacy preservation of feature compression has not been quantitatively analyzed (e.g., whether the original image can be reconstructed from compressed features).
- Only the ViT-B/32 backbone was evaluated, leaving the performance of larger models (e.g., ViT-L) unverified.
- The learning rate for single-step updates requires scaling based on offline training hyperparameters, and its generalization to new task types requires further validation.

## Related Work & Insights

| Method | Open Vocabulary | Online Update | Feature Compression | Dynamic Weighting |
|------|---------|---------|---------|---------|
| WiSE-FT / ZSCL | \checkmark (but generalization gradually degrades) | \times | \times | Fixed weight blending |
| Zhu et al. (AIM) | \checkmark | \checkmark | \times | AIM (only considers whether class is seen) |
| CLS-ER | \times | \checkmark | \times | EMA weight averaging |
| **AnytimeCL** | **\checkmark (zero forgetting)** | **\checkmark (millisecond-level)** | **\checkmark (30x)** | **OCW (class-wise dynamic)** |

The core advantage lies in the robustness of OCW during the early training stages: when only a few classes have samples, AIM mistakenly routes samples to the fine-tuned model, whereas OCW allocates weights according to actual accuracy.

## Related Work & Insights

- The dynamic weighting concept of OCW can be extended to fuse any multi-expert models, and is not limited to only two models.
- The per-image PCA compression concept has direct application value for reducing feature communication overhead in federated learning.
- The "other" regularization loss is highly relevant and beneficial for open-set recognition tasks.
- The paradigm of partial fine-tuning combined with fixed label embeddings could be applied to open-vocabulary detection and segmentation.
- The mapping between this system and complementary learning systems (CLS) theory is worth exploring on a larger scale.

## Rating

- Novelty: ⭐⭐⭐⭐ — OCW and attention-weighted PCA compression are sound innovations, with the overall method assembled coherently.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three continual learning scenarios, flexible inference, and rich ablation studies, though lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete method descriptions, and informative charts and tables.
- Value: ⭐⭐⭐⭐ — Proposes a practical framework for open-vocabulary continual learning; while still a distance away from actual deployment, the direction is correct.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Back to Source: Open-Set Continual Test-Time Adaptation via Domain Compensation](../../CVPR2026/model_compression/back_to_source_open-set_continual_test-time_adaptation_via_domain_compensation.md)
- [\[ECCV 2024\] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](../../ICML2026/model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](../../ICLR2026/model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)

</div>

<!-- RELATED:END -->
