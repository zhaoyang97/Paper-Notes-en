---
title: >-
  [Paper Note] Aligning Effective Tokens with Video Anomaly in Large Language Models
description: >-
  [ICCV 2025][Video Understanding][Video anomaly understanding] This paper proposes VA-GPT, which efficiently aligns anomaly-relevant tokens within MLLMs via two modules — Spatial Effective Token Selection (SETS) and Tempo…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video anomaly understanding"
  - "multimodal large language models"
  - "effective token selection"
  - "spatiotemporal alignment"
  - "anomaly detection"
date: 2026-05-08
content_hash: 7d8d3061a19a3546
---

# Aligning Effective Tokens with Video Anomaly in Large Language Models

**Conference**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**Code**: N/A  
**Area**: Video Understanding  
**Keywords**: Video anomaly understanding, multimodal large language models, effective token selection, spatiotemporal alignment, anomaly detection

## TL;DR

This paper proposes VA-GPT, which efficiently aligns anomaly-relevant tokens within MLLMs via two modules — Spatial Effective Token Selection (SETS) and Temporal Effective Token Generation (TETG) — enabling precise detection, description, and temporal localization of anomalous events.

## Background & Motivation

### Limitations of Prior Work

Traditional video anomaly detection methods suffer from two fundamental issues: (1) they are inherently closed-set detection and classification problems, precluding comprehensive understanding and explanation of anomalies; and (2) their limited vocabularies hinder handling of unseen or novel scenarios. Although recent video understanding MLLMs (e.g., Video-Chat, Video-ChatGPT) have achieved notable progress in general video analysis, they perform poorly in anomaly detection.

### Core Problem

**Why do existing MLLMs struggle with video anomalies?** The root cause lies in the **spatial and temporal sparsity** of anomalous events: in most cases, only small regions within a few frames contain critical anomaly information. However, existing methods process all latent tokens at equal priority across both spatial and temporal dimensions, causing abundant anomaly-irrelevant redundant tokens to dilute key information and degrade performance.

### Key Findings

The authors observe that anomalous events tend to induce distinctive visual changes and motion in localized regions. Accordingly, the central research question becomes **how to enable multimodal architectures to develop selective token generation and processing mechanisms** that dynamically prioritize anomaly-salient information while maintaining comprehensive scene understanding.

## Method

### Overall Architecture

VA-GPT is built upon a classical video understanding MLLM framework. Given a video of $T$ frames, a frozen ViT-based visual encoder (CLIP) extracts visual tokens $X_t = \{x_t^i\}_{i=1,...,N}$ from each frame. The core innovations are two modules:

1. **Spatial Effective Token Selection (SETS)**: Selects spatially effective tokens $X_t^*$ from $X_t$ to replace the full token set during fine-tuning and inference.
2. **Temporal Effective Token Generation (TETG)**: Generates anomaly-aware temporal prior tokens $S_t^*$, providing temporal information to the LLM directly in the language space.

### Key Design 1: Spatial Effective Token Selection (SETS)

**Why is spatial token selection necessary?** In the MLLM setting, the most critical challenge is vision–language modality alignment. Since textual descriptions primarily describe anomalous events — which occupy only a small portion of the video — aligning all visual patterns with text tokens is both suboptimal and computationally expensive.

**Inter-frame difference computation**: For each frame $V_t$, the preceding frame $V_{t-1}$ serves as a reference. DINOv2 is used as a feature extractor to obtain patch embeddings:

$$F_t, F_{t-1} = FE(V_t), FE(V_{t-1})$$

The patch-wise Manhattan distance is then computed as the inter-frame difference map:

$$D_t = dis(F_t, F_{t-1})$$

**Token filtering strategy**: Based on the difference map $D_t$, the top-$K$ proportion of elements with the largest distances are assigned a value of 1 and the rest 0, forming a mask $M_t$:

$$X_t^* = \{x_t^i | m_t^i = 1, m_t^i \in M_t\}$$

**Why inter-frame difference rather than other approaches?** The core assumption is that regions exhibiting large changes between adjacent frames deserve greater attention, as anomalous events typically manifest as salient visual changes in localized areas. Features extracted by DINOv2 offer strong discriminability and stability, reliably capturing such changes.

### Key Design 2: Temporal Effective Token Generation (TETG)

**Anomaly-aware classifier**: A simple yet effective MLP $F_A$ is designed to determine whether each frame is related to an anomalous event. Using the class embedding $z$ extracted by the feature encoder, embeddings are divided into normal embeddings $z_n$ and anomalous embeddings $z_a$ according to training video annotations, and optimized with a binary classification loss:

$$\mathcal{L} = E_{z \sim z_n}[-\log\frac{1}{1+\exp(-F_A(z))}] + E_{z \sim z_a}[-\log\frac{\exp(-F_A(z))}{1+\exp(-F_A(z))}]$$

**Token generation mechanism**: Since the classifier provides explicit information, it can be projected directly into the LLM's text token space via natural language templates. Based on classification results, the timestamps of high-confidence frames containing anomalous events — `<a-start>` and `<a-end>` — are selected and assembled into the following template:

> "Known common crime types are: 'Shooting', 'Arson', 'Arrest', ... There is one of the crime types occurring from <a-start> to <a-end>"

**Design rationale**: This approach supplies the LLM with prior knowledge about the temporal extent of anomalous events at minimal cost, without additional complex modules, and directly leverages the LLM's inherent text comprehension capability.

### Loss & Training

A two-stage progressive training strategy is adopted:

1. **Stage One**: Fine-tuning on anomalous video data. Instruction-following QA pairs are constructed from the UCF-Crime dataset, mixing multiple instruction types (text dialogues, single/multi-turn visual QA, and video QA). All modules except the frozen visual encoder are optimized.
2. **Stage Two**: Aligning the LLM with spatially effective tokens. Additional short-term fine-tuning is performed using spatial effective tokens extracted per frame from UCF-Crime, requiring fewer than 150 iterations to yield substantial performance improvements.

## Key Experimental Results

### Main Results

| Method | LLM | In-domain Total Acc. (%) | In-domain Temporal Acc. (%) | Cross-domain Total Acc. (%) | Cross-domain Temporal Acc. (%) |
|------|-----|-------------------|---------------------|-------------------|---------------------|
| Video-ChatGPT | Vicuna-7B | 24.13 | 28.51 | 24.00 | 29.10 |
| Otter | LLaMa-7B | 22.41 | 22.17 | 25.20 | 23.80 |
| Valley | Vicuna-7B | 20.34 | 14.48 | 21.00 | 20.20 |
| Video-LLaMA2 | Vicuna-7B | 21.38 | 26.62 | 24.20 | 23.00 |
| Hawkeye | LLaVA-7B | 28.60 | 30.00 | 25.30 | 28.50 |
| LLaMA-VID (Baseline) | Vicuna-7B | 14.83 | 26.70 | 18.80 | 23.60 |
| **VA-GPT (Ours)** | Vicuna-7B | **30.69** | **35.00** | **26.20** | **31.02** |

VA-GPT achieves the best results on all four metrics. In-domain Total Acc. more than doubles the baseline, and cross-domain generalization is also substantially superior.

### Ablation Study

| Configuration | Baseline | Stage One Fine-tuning | Stage Two Fine-tuning |
|------|----------|----------------------|----------------------|
| w/o Both | 14.83 / 26.70 | - | - |
| w. SETS | 24.83 / 27.20 | 25.86 / 29.68 | 29.31 / 31.60 |
| w. TETG | 23.79 / 27.76 | 26.10 / 30.02 | 28.96 / 33.58 |
| w. Both | 25.12 / 28.81 | 27.50 / 30.77 | **30.69 / 35.00** |

(Format: Total Acc. / Temporal Acc.)

**Ablation on sampling ratio K**:

| K | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|-----|-----|-----|-----|-----|
| Total Acc. (%) | 23.61 | 24.83 | **30.69** | 28.67 | 27.27 |
| Temporal Acc. (%) | 29.03 | 29.93 | **35.00** | 31.23 | 31.03 |

$K=0.5$ is optimal; too small a value discards important information, while too large a value introduces excessive noise.

### Key Findings

1. **SETS and TETG are complementary**: The two modules compress anomaly information from spatial and temporal dimensions respectively, and their combination yields the best performance.
2. **Data quality is critical**: Strong performance is achieved with only approximately 4,000 videos — far fewer than the 90k+ videos used by the baseline — attributable to high-quality instruction-following data.
3. **SETS also improves data quality**: By filtering visual regions irrelevant to QA during Stage Two fine-tuning, significant gains are achieved in fewer than 150 iterations.

## Highlights & Insights

1. **Token-level alignment innovation**: VA-GPT is the first to explore assigning different learnable knowledge to distinct tokens in MLLMs for better visual content alignment, rather than treating all tokens uniformly.
2. **Natural language template design for temporal tokens**: Projecting classifier-based temporal predictions into the LLM via natural language templates elegantly leverages the LLM's inherent text comprehension, resulting in an exceptionally concise and efficient design.
3. **Cross-domain evaluation benchmark**: A cross-domain evaluation protocol based on XD-Violence is established, systematically assessing model robustness under domain shift.
4. **Data efficiency**: The work demonstrates that high-quality data combined with effective token selection can substantially reduce training data requirements.

## Limitations & Future Work

1. Detection and description of anomalous events in complex scenarios — such as multiple simultaneous anomalies — remain challenging.
2. SETS relies on inter-frame differences and may lack sensitivity to slowly evolving anomalies (e.g., gradually intensifying fires).
3. The anomaly type templates in TETG are predefined, potentially limiting generalization to entirely novel anomaly categories.
4. Experiments are conducted only at the Vicuna-7B scale; the effectiveness of larger-scale LLMs remains unexplored.

## Related Work & Insights

- **LLaMA-VID**: The baseline model upon which VA-GPT introduces the token selection mechanism.
- **Hawkeye**: Another anomaly-aware video MLLM, but one that does not differentiate token importance.
- **DINOv2**: Employed as the feature extractor in SETS; the stability of its self-supervised features underpins reliable inter-frame difference computation.
- Broader implication: The paradigm of selective token-level processing is generalizable to other MLLM tasks requiring focus on specific information.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](../../NeurIPS2025/video_understanding/monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)

</div>

<!-- RELATED:END -->
