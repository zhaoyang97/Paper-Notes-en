---
title: >-
  [Paper Note] Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning
description: >-
  [CVPR 2026][Audio & Speech][Active Learning] This paper proposes Refine, an ensemble active learning method that employs a two-stage strategy—progressive filtering (iteratively refining the unlabeled pool via multiple st…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "Active Learning"
  - "Ensemble Strategy"
  - "Progressive Filtering"
  - "Foundation Models"
  - "Coverage-Based Selection"
date: 2026-05-08
content_hash: b0fe24adcd481e55
---

# Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning

**Conference**: CVPR 2026
**arXiv**: [2511.22344](https://arxiv.org/abs/2511.22344)  
**Code**: [GitHub](https://github.com/dhuseljic/dal-toolbox)  
**Area**: Audio/Speech (Active Learning)
**Keywords**: Active Learning, Ensemble Strategy, Progressive Filtering, Foundation Models, Coverage-Based Selection

## TL;DR

This paper proposes Refine, an ensemble active learning method that employs a two-stage strategy—progressive filtering (iteratively refining the unlabeled pool via multiple strategies) and coverage-based selection (selecting high-value, diverse samples from the refined pool)—to consistently outperform individual AL strategies and existing ensemble methods without requiring prior knowledge of the optimal strategy.

## Background & Motivation

**Background**: Pre-trained foundation models (DINOv2, CLIP) still require labeled data for downstream task adaptation. Active learning reduces annotation costs through intelligent sample selection, yet recent benchmarks indicate that no single strategy is consistently optimal.

**Limitations of Prior Work**: (a) Different AL strategies capture distinct notions of "data value"—uncertainty vs. representativeness—with no strategy universally dominant; (b) selecting the wrong strategy can perform worse than random sampling; (c) existing ensemble methods (TCM/TAILOR/SelectAL) rely on heuristic switching or learned scheduling, leading to unstable performance.

**Key Challenge**: AL is a one-shot problem (no opportunity for trial and error), requiring decisions without knowing the optimal strategy in advance.

**Goal**: Design a learning-free ensemble method that automatically integrates the complementary advantages of multiple strategies.

**Key Insight**: Shift the focus from "which samples to select" to "cleaning the pool by removing uninformative samples first."

**Core Idea**: Allow multiple strategies to iteratively vote for sample retention—samples surviving multiple rounds must have been deemed valuable by at least one strategy, while samples never selected by any strategy are necessarily uninformative.

## Method

### Overall Architecture

Two-stage selection: (1) progressive filtering to iteratively refine the unlabeled pool; (2) coverage-based selection to choose the final batch from the refined pool.

### Key Designs

1. **Progressive Filtering**: $R=5$ iterative rounds, where in each round every strategy selects $J=10$ batches from a random subsample of fraction $\alpha=0.4$, and the union of all batches forms the candidate pool for the next round:
    $$\mathcal{C}_r = \bigcup_{m=1}^M \bigcup_{j=1}^J s_m(\text{SubSample}(\mathcal{C}_{r-1}, \alpha \cdot |\mathcal{C}_{r-1}|), b)$$

    Three key design decisions and their motivations:
    - **Union rather than intersection**: Retains all samples deemed valuable by any strategy, avoiding the loss of uniquely discovered samples.
    - **Multi-round iteration**: A single round merely concatenates outputs; across multiple rounds, the survival probability of uninformative samples decays exponentially.
    - **Subsampling with $\alpha < 1$**: Encourages deterministic strategies to produce diverse batches and reduces memory requirements.

2. **Coverage-Based Selection**: UHerding is applied to select a batch from the refined pool $\mathcal{C}_R$ that maximizes coverage:
    $$\mathcal{B}^* = \arg\max_{\mathcal{B} \subset \mathcal{C}_R} \mathbb{E}_{\mathbf{x}}[\max_{\mathbf{x}' \in (\mathcal{L}_t \cup \mathcal{B})} k(\mathbf{x}, \mathbf{x}')]$$
    **Design Motivation**: After filtering, the pool already consists of high-value candidates; coverage-based selection then ensures diversity within this refined set.

3. **Theoretical Guarantees**:
    - Theorem 1 (Value Preservation): $P_r(\mathbf{x}) \geq 1 - (1 - \alpha \cdot \max_m p_{m,r}(\mathbf{x}))^J$
    - Theorem 2 (Exponential Decay): The survival probability of an $\epsilon$-uninformative sample after $R$ rounds is $\leq (MJ\alpha\epsilon)^R$
    - Theorem 3 (Value Monotonicity): $\mathbb{E}[V|\mathcal{C}_R] \geq \ldots \geq \mathbb{E}[V|\mathcal{C}_0]$

### Loss & Training

- Three backbones: DINOv2-ViT-S/14, DINOv2-ViT-S/16, CLIP-ViT-B/16
- Frozen backbone with trained classification head; SGD with LR 0.01, 200 epochs per cycle
- 20 AL cycles, 10 independent runs

## Key Experimental Results

### Main Results — Overall Win Rate

| Refine vs. | Win Rate (3 backbones × 5 datasets × 10 trials) |
|-----------|--------------------------------------------------|
| BAIT | 85% |
| UHerding | 80% |
| SelectAL | 100% |
| TAILOR | 100% |
| TCM | 98% |
| AutoAL | 97% |

### Ablation Study — Progressive Filtering as Preprocessing

| Strategy | AULC from Original Pool | AULC from Refined Pool | Gain |
|----------|------------------------|------------------------|------|
| Random | Baseline | +3.7% | Filtering alone + random is already effective |
| BAIT | Worse than Random | **Better than Random** | Filtering rescues a failing strategy |
| AlfaMix | Baseline | +2.6% | Universally beneficial |
| UHerding | High | +0.7% | Strong strategies also benefit |

### Effect of Number of Filtering Rounds

| R (rounds) | CIFAR-10 AULC Gain | Snacks AULC Gain |
|------------|-------------------|-----------------|
| 1 | +3.02% | +6.91% |
| 3 | +3.72% | +7.22% |
| 5 | +3.71% | +7.79% |
| 9 | +3.78% | +8.43% |

### Key Findings

1. Refine achieves the highest overall win rate against all individual strategies and all ensemble methods.
2. Progressive filtering is a general-purpose preprocessing step—any AL strategy applied to the refined pool consistently outperforms the same strategy applied to the original pool.
3. BAIT underperforms random sampling on the original pool but surpasses it after filtering, demonstrating that filtering removes misleading samples.
4. Using only two strategies—Margin and TypiClust—for filtering automatically integrates uncertainty and representativeness.
5. Performance is stable for $\alpha \in [0.3, 0.9]$.

## Highlights & Insights

- **Progressive filtering as preprocessing** is the most practically impactful contribution—it can be grafted onto any AL strategy at negligible cost.
- The theoretical analysis is elegant: three theorems respectively guarantee value preservation, noise removal, and monotonic quality improvement.
- The insight that "samples never selected by any strategy are necessarily uninformative" is concise yet profound.
- The framework is easily extensible: new strategies can be incorporated directly without retraining.

## Limitations & Future Work

- Invoking multiple strategies multiple times incurs additional computational overhead, though parallelization is feasible.
- The quality of the refined pool may degrade when the majority of ensemble strategies fail (e.g., the Dopanim+CLIP case).
- Strategy weighting is unexplored; the current approach assigns equal weight, whereas dynamic weighting based on historical performance could be beneficial.
- Validation is primarily conducted on image classification; further evaluation on detection and segmentation tasks is needed.

## Related Work & Insights

- Unlike the hard switching of TCM, Refine achieves a natural soft transition through progressive filtering.
- The union-plus-iteration design is generalizable to other scenarios that require the integration of multiple heuristics.
- This work provides a practical, theoretically grounded solution for AL in the era of foundation models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The progressive filtering concept is concise and novel, though it constitutes a strategy combination rather than a fundamentally new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 datasets × 3 backbones × 8 individual strategies + 4 ensemble methods + extensive ablations + theoretical analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theory and experiments complement each other perfectly; structure is clear.
- **Value**: ⭐⭐⭐⭐ Serves as a general AL preprocessing step with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] PACE: Pretrained Audio Continual Learning](../../ICLR2026/audio_speech/pace_pretrained_audio_continual_learning.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/audio_speech/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)

</div>

<!-- RELATED:END -->
