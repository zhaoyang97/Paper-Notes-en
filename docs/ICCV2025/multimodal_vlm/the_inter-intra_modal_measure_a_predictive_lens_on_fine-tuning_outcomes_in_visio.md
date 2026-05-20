---
title: >-
  [Paper Note] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Transferability estimation] This paper proposes the Inter-Intra Modal Measure (IIMM)—a metric that requires only a single forward pass to predict both the performance gain and the degree of ca…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Transferability estimation"
  - "fine-tuning prediction"
  - "modality gap"
  - "contrastive learning"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: f95dcfa70c18194d
---

# The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2407.15731](https://arxiv.org/abs/2407.15731)  
**Code**: [GitHub](https://github.com/mit-ll/IIMM)  
**Area**: Multimodal VLM
**Keywords**: Transferability estimation, fine-tuning prediction, modality gap, contrastive learning, catastrophic forgetting

## TL;DR

This paper proposes the Inter-Intra Modal Measure (IIMM)—a metric that requires only a single forward pass to predict both the performance gain and the degree of catastrophic forgetting following fine-tuning of vision-language dual-encoder models. By quantifying intra-modal image embedding similarity and inter-modal misaligned label alignment, IIMM demonstrates strong linear predictive power ($R^2 > 0.85$) across 4 foundation models and 5 fine-tuning strategies.

## Background & Motivation

Fine-tuning vision-language foundation models (e.g., CLIP) presents two practical challenges:

**Whether fine-tuning is worthwhile**: Computational costs are high, yet performance gains on specialized tasks (e.g., satellite or medical image classification) can be marginal or even negative—can the outcome be predicted before fine-tuning?

**The learning–forgetting trade-off**: Fine-tuning improves target-task performance while disrupting image-text embedding space alignment, leading to catastrophic forgetting (degraded performance on off-target tasks)—can the degree of forgetting be predicted?

Limitations of existing transferability metrics:
- Classical methods such as **H-Score, LEEP, and NCE** rely on source data and task labels, making them unsuitable for contrastively self-supervised pretrained foundation models.
- Recent methods such as **TransRate and SFDA** consider only single-modal encoders and ignore cross-modal interactions.
- Most methods evaluate only rank correlation and cannot directly **predict the magnitude of performance change**.

Core insight: The geometry of the contrastive learning embedding space encodes rich information about fine-tuning potential—**more clustered intra-modal embeddings combined with higher inter-modal misalignment imply greater fine-tuning improvement potential but also higher forgetting risk**.

## Method

### Overall Architecture

IIMM derives from the InfoNCE loss of contrastive learning and decomposes the embedding space geometry into two computable components:
1. Inter-modal misalignment ($S_{inter}$)
2. Intra-modal uniformity ($S_{intra}$)

### Key Designs

1. **Deriving IIMM from InfoNCE Loss**:

   The InfoNCE loss optimizes two objectives: positive-pair alignment and negative-pair separation. IIMM captures the current state of both effects:

   - **Inter-modal misalignment**:
     $S_{inter} = \frac{1}{|X|} \sum_{x \in X} \frac{1}{|Y|-1} \sum_{y' \in Y \setminus \{y(x)\}} x^\top y'$
     High $S_{inter}$ indicates that image embeddings exhibit high similarity to incorrect text labels, implying the model has not sufficiently separated negative pairs—large fine-tuning potential but high forgetting risk.

   - **Intra-modal uniformity**:
     $S_{intra} = \frac{2}{|X|(|X|-1)} \sum_{1 \leq i < j \leq |X|} x_i^\top x_j$
     High $S_{intra}$ indicates overly clustered image embeddings, suggesting room for further separation or reorganization.

   - **IIMM definition**:
     $\text{IIMM} = \frac{1}{2}(S_{inter} + S_{intra})$
     The range is $[-1, 1]$; higher values predict greater learning gain but also greater forgetting risk.

2. **Theoretical Guarantee—Wasserstein Distance Bound**:

   It is proven that changes in IIMM are bounded by the 1-Wasserstein distance between the embedding distributions before and after fine-tuning:
   $|\text{IIMM}(Q) - \text{IIMM}(P)| \leq \frac{\delta_A}{2} + \delta_B$

   Interpretation: IIMM changes significantly only when the embedding space undergoes substantial reorganization, guaranteeing **stability and robustness** of the metric.

3. **Usage Protocol**:

    - Perform a single forward pass on target-task data to compute IIMM.
    - Pre-fit a linear model mapping IIMM to performance gain using a small set of standard visual benchmarks.
    - Apply the linear model to predict fine-tuning gain on new tasks without actually fine-tuning.

### Loss & Training

Experimental fine-tuning configuration:
- 30 epochs, SGD, batch size 128
- Grid search: lr ∈ {1e-2, 1e-3, 1e-4, 1e-5}, weight decay ∈ {1e-3, 1e-4, 1e-5}
- LoRA rank ∈ {2, 4, 8, 16}, CLIP-Adapter reduction ∈ {4, 8, 16, 32}
- Fine-tuned and evaluated on 9 visual classification datasets

## Key Experimental Results

### Main Results (Kendall's τ Correlation: IIMM vs. Baseline Metrics)

Correlation between each metric and fine-tuning gain normalized by zero-shot error:

| Metric | CoCa τ | EVA-02 τ | CLIP τ | SigLIP τ |
|--------|--------|----------|--------|---------|
| **IIMM** | **0.89** | **0.83** | **0.78** | **0.78** |
| GBC | 0.78 | 0.72 | 0.78 | 0.83 |
| TransRate | 0.67 | 0.61 | 0.67 | 0.72 |
| EMMS | 0.61 | 0.28 | 0.61 | 0.44 |
| NCTI | -0.06 | -0.59 | 0.07 | -0.06 |
| H-Score | 0.11 | -0.78 | 0.50 | -0.67 |

### Linear Fit Strength (IIMM → Gain over Zero-Shot Error)

| Model | $R^2$ | p-value |
|-------|-------|---------|
| CoCa | 0.92 | <10⁻³ |
| EVA-02 | 0.89 | <10⁻³ |
| CLIP | 0.92 | <10⁻³ |
| SigLIP | 0.92 | <10⁻³ |

For comparison, GBC (the second strongest metric) achieves $R^2$ of only 0.59–0.72.

### Embedding Space Component Analysis

| Embedding Feature | CoCa $R^2$ | CLIP $R^2$ | SigLIP $R^2$ | Description |
|-------------------|-----------|-----------|-------------|-------------|
| Intra-modal image distance | 0.88 | 0.81 | 0.88 | Most predictive single feature |
| Misaligned label alignment | 0.82 | 0.62 | 0.80 | Second most important |
| Intra-modal text distance | 0.37 | 0.57 | 0.17 | Essentially uninformative |

### Key Findings

- **IIMM achieves $R^2 > 0.89$ across all foundation models**, substantially outperforming all unimodal transferability metrics.
- **IIMM is negatively linearly correlated with forgetting**: higher IIMM corresponds to greater average accuracy degradation on off-target tasks after fine-tuning (with relatively high $R^2$ across models except CLIP).
- **Differentiated behavior of PEFT methods**:
    - CLIP-Adapter's localized adjustment (frozen backbone) weakens the IIMM–gain relationship but substantially mitigates forgetting.
    - LoRA and attention-weight fine-tuning exhibit severe forgetting when IIMM is high.
- **Intra-modal image embedding similarity is the most predictive component** ($R^2$ = 0.81–0.88), but combining it with inter-modal misalignment yields stronger results ($R^2$ = 0.89–0.92), validating the necessity of multimodal information.
- **Correct-label alignment and intra-modal text distance are essentially uninformative**—only the negative-sample direction is useful.

## Highlights & Insights

- **Balance between theory and practice**: The derivation from InfoNCE loss is both elegant and physically meaningful; the Wasserstein distance bound provides theoretical guarantees.
- **Extremely low computational overhead**: Only a single forward pass over target data is required (no fine-tuning needed), making this a genuinely practical tool.
- **Bounded and interpretable extremes**: IIMM ∈ [-1,1], unlike unbounded metrics such as GBC—a high IIMM directly signals "fine-tuning is worthwhile but forgetting risk is high."
- **Reveals forgetting characteristics of PEFT methods**: LoRA can be particularly risky at high IIMM, whereas the conservative design of CLIP-Adapter proves advantageous in such settings.

## Limitations & Future Work

- The predictive linear model requires prior fine-tuning on several benchmarks for calibration—future work could explore approaches that require no reference fine-tuning results whatsoever.
- Experiments are limited to classification tasks; effectiveness on other downstream tasks (detection, segmentation, VQA, etc.) remains unvalidated.
- Only ViT-B-scale models are evaluated; behavior on larger models (ViT-L/H) is unknown.
- Whether the equal-weight average of the two sub-components is optimal remains an open question; convex combination experiments are reported but not thoroughly analyzed.

## Related Work & Insights

- The IIMM framework is generalizable to other contrastive learning models (e.g., CLAP for audio-text, ImageBind for multimodal).
- This work extends and operationalizes modality gap research—moving from observing phenomena to building predictive tools.
- IIMM provides a quantitative basis for PEFT method selection: tasks with high IIMM should favor conservative fine-tuning strategies (e.g., Adapter).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First metric specifically designed to predict fine-tuning outcomes for vision-language dual-encoder models)
- Technical Depth: ⭐⭐⭐⭐⭐ (Derivation from contrastive loss + Wasserstein theoretical bound + comprehensive empirical validation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 models × 5 fine-tuning methods × 9 datasets, but limited to classification tasks)
- Value: ⭐⭐⭐⭐⭐ (A single forward pass suffices to predict fine-tuning outcomes—highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](../../NeurIPS2025/multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)

</div>

<!-- RELATED:END -->
