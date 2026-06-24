---
title: >-
  [Paper Note] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning
description: >-
  [CVPR2025][Medical Imaging][Continual learning] To address domain-incremental learning (DIL) without task IDs and data replay, this paper proposes the Residual SODAP framework. It concurrently solves representation adaptation and classifier forgetting through $\alpha$-entmax sparse prompt selection with residual aggregation, pseudo-replay distillation based on feature statistics, prompt usage pattern drift detection, and uncertainty weighting. It achieves state-of-the-art (SO…
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Continual learning"
  - "domain-incremental learning"
  - "prompt-based learning"
  - "catastrophic forgetting"
  - "knowledge distillation"
  - "sparse selection"
date: 2026-05-08
content_hash: 09cb9259b4206fd4
---

# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**Conference**: CVPR2025  
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)  
**Code**: To be confirmed  
**Area**: Medical Images  
**Keywords**: Continual learning, domain-incremental learning, prompt-based learning, catastrophic forgetting, knowledge distillation, sparse selection

## TL;DR

To address domain-incremental learning (DIL) without task IDs and data replay, this paper proposes the Residual SODAP framework. It concurrently solves representation adaptation and classifier forgetting through $\alpha$-entmax sparse prompt selection with residual aggregation, pseudo-replay distillation based on feature statistics, prompt usage pattern drift detection, and uncertainty weighting. It achieves state-of-the-art (SOTA) performance on diabetic retinopathy (DR), skin cancer, and CORe50 datasets.

## Background & Motivation

- **Core Challenge of Continual Learning**: Catastrophic forgetting is particularly severe in domain-incremental learning (DIL), where task IDs are unavailable and historical data cannot be stored.
- **Two Limitations of Prior Prompt-based CL**:
  1. **Inadequate Prompt Selection Schemes**: Top-$k$ hard selection limits expressiveness and is non-differentiable; Softmax soft selection allows irrelevant prompts to exert influence, leading to noise accumulation.
  2. **Ignoring Classifier Structure**: Existing PCL methods primarily focus on prompt pool design to adapt representations, but the classifier layer also exhibits instability under domain shifts (as demonstrated by the cross-composition diagnostic experiments).
- **Key Finding**: Feature extractor (backbone) $\times$ classifier cross-composition analysis (referencing the diagnostic method of Liu et al.) reveals that even if backbone representations are well-maintained via prompt adaptation, the decision boundaries of the classifier layer still degrade significantly during domain-incremental training.
- **Goal**: A unified framework is required to simultaneously address representation adaptation at the prompt layer and knowledge preservation at the classifier layer.

## Method

### 1. $\alpha$-Entmax Residual Prompt Selection
- **Query Enhancement**: Fuses the current-layer CLS token, the initial CLS token (global context), and retrieval signals from a learnable memory bank, generating an enhanced query via Multi-Head Attention (MHA) and a bottleneck adapter.
- **Sparse Selection**: Replaces Softmax with $\alpha$-entmax ($\alpha=1.5$), which assigns exact zero weights to low-scoring prompts, balancing full prompt pool utilization and noise suppression.
- **Residual Structure**: Starting from Stage 2, the prompt pool is split into a frozen set $F$ and an active set $A$. These sets undergo independent sparse routing and are combined in a residual manner: $p_{out} = p_F + 0.1 \cdot p_A$. The frozen set preserves prior knowledge, while the active set performs only residual adaptation.
- **Auxiliary Loss**: Diversity loss (penalizes similarity among highly co-activated prompts) + norm regularization (constrains active prompt values to act purely as residuals).

### 2. Statistical Knowledge Preservation (Pseudo-Replay Distillation)
- At the end of each stage, class-wise feature statistics (mean and variance) are collected using Welford's online algorithm, and the current classifier head is frozen as a teacher.
- During multi-stage training, statistics are cumulatively merged using Welford's formula, which can be completed in a single pass and is memory-efficient.
- During the training of the next stage, two types of distillation are performed:
  1. **Real Feature Distillation**: Aligns current batch features via the KL divergence between the teacher and student heads (temperature $T=2.0$).
  2. **Pseudo-Feature Replay**: Samples pseudo-features from stored class statistics (using the reparameterization trick) and aligns them via the KL divergence between the frozen teacher and trainable student heads.
  - Classes are uniformly sampled to mitigate under-representation of minority classes, with $K=B$ pseudo-features sampled per batch.

### 3. Prompt Usage Pattern Drift Detection (PUDD)
- Concurrently monitors two signals: (a) entropy changes in the prompt selection distribution (short-term fluctuations reflecting domain changes), and (b) structural shifts of the active prompt set (using Intersection over Union, IoU).
- The two signals are weighted and combined into a drift score $D_t$ ($\alpha=1.0, \beta=0.5$), which is averaged across layers and batches to determine the scale of prompt pool expansion.
- The expansion size is directly proportional to the drift intensity: weak drifts trigger minimal prompt addition ($E_{min}=10$), while strong drifts lead to large-scale expansion ($E_{max}=80$).
- After expansion, the existing active prompts are moved to the frozen set, and the newly added prompts become the new active set.

### 4. Uncertainty Weighting
- Employs homoscedastic uncertainty weighting following Kendall et al. to learn a log-variance $s_i$ for each loss term.
- Total Loss: $L_{total} = \sum (e^{-s_i} \cdot L_i + s_i)$, where noisy losses are automatically downweighted.

## Key Experimental Results

### Benchmark Settings
- Three DIL scenarios: Diabetic Retinopathy (DR, 3 domains: APTOS $\rightarrow$ DDR $\rightarrow$ DRD), Skin Cancer (3 domains: ISIC $\rightarrow$ HAM $\rightarrow$ DERM7), and CORe50 (general benchmark).
- No data replay, no task IDs, with all results averaged over 3 independent trials.
- Evaluation metrics: AvgACC (Average Accuracy) and AvgF (Average Forgetting).

### Main Results

| Method | DR AvgACC↑ | DR AvgF↓ | Skin AvgACC↑ | Skin AvgF↓ | CORe50 AvgACC↑ | CORe50 AvgF↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| OS-Prompt++ | 0.769 | 0.113 | 0.725 | 0.063 | 0.983 | 0.014 |
| Coda-Prompt | 0.688 | 0.140 | 0.713 | 0.041 | 0.974 | 0.056 |
| DER++ | 0.607 | 0.288 | 0.722 | 0.099 | 0.994 | 0.061 |
| **Residual SODAP** | **0.850** | **0.047** | **0.760** | **0.031** | **0.995** | **0.003** |

- DR scenario: AvgACC improves by 8.1 percentage points (pp) compared to the second-best method (OS-Prompt++: 0.769), while AvgF drops from 0.113 to 0.047.
- Skin cancer scenario: Achieves the best AvgACC of 0.760 with an AvgF of 0.031 (Dual-Prompt has a lower AvgF of 0.012, but its AvgACC is only 0.637, representing a poor accuracy-forgetting trade-off).
- CORe50: Performance is near-perfect (AvgACC: 0.995, AvgF: 0.003), demonstrating the effectiveness of the proposed method in general domains.

### Ablation Study (DR)
- Removing Query Enhancer: AvgACC drops by 4.2 pp (making it the most critical individual component).
- Removing Residual (degrading to SODAP): AvgACC drops by 1.9 pp, and AvgF degrades by 2.0 pp.
- Removing Pseudo-Replay: AvgACC drops by 1.5 pp.
- Removing Distillation: Also leads to performance degradation.
- PUDD dynamically regulates prompt pool expansion from 60 $\rightarrow$ 84 $\rightarrow$ 94, ensuring no redundant prompts after expansion.

## Highlights & Insights

1. **Comprehensive Design**: Simultaneously addresses three main dimensions—prompt selection noise, classifier forgetting, and domain drift detection—rather than focusing on a single issue.
2. **$\alpha$-Entmax Sparse Selection**: Accurately suppresses irrelevant prompts while preserving differentiable optimization across the entire prompt pool, outperforming both Top-$k$ (non-differentiable) and Softmax (prone to noise accumulation).
3. **Zero-Data Pseudo-Replay**: Stores only class-wise means and variances (only 2D vectors per class, extremely low storage overhead), enabling knowledge preservation via reparameterization sampling.
4. **PUDD Automatic Expansion**: Detects domain drift based on actual usage patterns rather than hard-coded rules, allowing adaptive expansion and avoiding capacity waste or insufficiency.
5. **Generality**: Achieves SOTA results on both medical imaging (DR, skin cancer) and general computer vision (CORe50), demonstrating that the method is not application-restricted.

## Limitations & Future Work

1. The experimental scenarios are relatively short (only 3 domains); scalability under longer sequences (10+ domains) remains unverified, where continuous expansion of the prompt pool might impose memory and computational burdens.
2. Hyperparameters such as $\alpha$ and $\lambda_r$ are fixed ($\alpha=1.5, \lambda_r=0.1$); sensitivity analyses across different scenarios have not been fully explored.
3. Relying on a frozen ViT backbone (pretrained on ImageNet-21K), the method's dependency on the specific backbone architecture and pretraining data remains uninvestigated.
4. Pseudo-replay assumes diagonal Gaussian feature distributions, which may be inaccurate for highly complex multimodal distributions, particularly under domain feature overlap.
5. The clamp range $[-3, 6]$ for uncertainty weighting is empirically chosen, lacking theoretical guidance.
6. Hyperparameters for PUDD (sliding window $W=100$, $D_{max}=5.0$, threshold $\tau_s=0.01$) have not undergone sufficient sensitivity analysis.
7. The fundamental difference from multi-head classifier approaches warrants deeper discussion—although the proposed scheme does not require Task-IDs, it relies on drift detection.

## Rating
- Novelty: 4/5 — $\alpha$-entmax residual prompt selection and PUDD drift detection represent significant innovations, and classifier knowledge preservation addresses a blind spot in PCL.
- Experimental Thoroughness: 4/5 — Covers three benchmarks (2 medical, 1 general), comprehensive ablations, and visualization analyses, though the sequence length is limited (only 3 domains).
- Writing Quality: 4/5 — Structurally sound with accurate formulations. The cross-composition diagnostic analysis is highly convincing, though some notations are dense.
- Value: 4/5 — Offers a systematic and practical solution for DIL without data replay or Task-IDs, holding direct application value for continual learning in medical imaging.

<!-- See Supplementary Material for experimental hardware and hyperparameters -->
<!-- Backbone: ViT-B/16 pretrained on ImageNet-21K; the backbone is frozen, with training restricted to prompt-related parameters -->
<!-- Distillation temperature $T=2.0$, batch size $B=64$, initial prompt pool size of 60 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2026\] Keep It Frozen: Domain-Routed Conditional Residual Modulation for Multi-Domain Vision Transformers](../../CVPR2026/medical_imaging/keep_it_frozen_domain-routed_conditional_residual_modulation_for_multi-domain_vi.md)
- [\[CVPR 2026\] DK-DDIL: Adaptive Knowledge Retention for Dynamic Domain-Incremental Learning in Medical Imaging](../../CVPR2026/medical_imaging/dk-ddil_adaptive_knowledge_retention_for_dynamic_domain-incremental_learning_in_.md)
- [\[CVPR 2026\] Forging a Dynamic Memory: Retrieval-Guided Continual Learning for Generalist Medical Foundation Models](../../CVPR2026/medical_imaging/forging_a_dynamic_memory_retrieval-guided_continual_learning_for_generalist_medi.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)

</div>

<!-- RELATED:END -->
