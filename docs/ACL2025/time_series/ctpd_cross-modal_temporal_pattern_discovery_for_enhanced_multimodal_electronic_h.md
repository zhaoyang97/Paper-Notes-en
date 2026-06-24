---
title: >-
  [Paper Note] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis
description: >-
  [ACL 2025][Time Series][Electronic Health Records] A CTPD framework is proposed, which utilizes Slot Attention to discover cross-modality shared temporal prototype patterns from multimodal EHR data (irregular time series and clinical notes). Temporal semantics of both modalities are aligned via a TP-NCE contrastive loss, achieving SOTA performance on mortality prediction and phenotype classification tasks on MIMIC-III.
tags:
  - "ACL 2025"
  - "Time Series"
  - "Electronic Health Records"
  - "Cross-Modal Temporal Patterns"
  - "Slot Attention"
  - "Contrastive Learning"
  - "Clinical Prediction"
date: 2026-05-08
content_hash: 7def295dbed27bc6
---

# CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis

**Conference**: ACL 2025  
**arXiv**: [2411.00696](https://arxiv.org/abs/2411.00696)  
**Code**: [https://github.com/HKU-MedAI/CTPD](https://github.com/HKU-MedAI/CTPD)  
**Area**: Medical Informatics / Time Series  
**Keywords**: Electronic Health Records, Cross-Modal Temporal Patterns, Slot Attention, Contrastive Learning, Clinical Prediction

## TL;DR
A CTPD framework is proposed, which utilizes Slot Attention to discover cross-modality shared temporal prototype patterns from multimodal EHR data (irregular time series and clinical notes). Temporal semantics of both modalities are aligned via a TP-NCE contrastive loss, achieving SOTA performance on mortality prediction and phenotype classification tasks on MIMIC-III.

## Background & Motivation

**Background**: EHR data is inherently multimodal temporal data, containing structured vital signs/laboratory tests (irregular time series) and unstructured clinical notes. Multimodal fusion has been proven beneficial for clinical outcome prediction.

**Limitations of Prior Work**: (1) Existing methods primarily focus on intra-sample temporal interactions and multimodal fusion, ignoring high-level temporal patterns across patients (e.g., abnormal heart rate trends, gradual improvement in respiratory function); (2) These patterns have semantic correspondences between time series and clinical notes, but effective cross-modal pattern discovery methods are lacking; (3) Key temporal patterns unfold across different time scales (acute changes vs. long-term trends), which existing methods struggle to capture simultaneously.

**Key Challenge**: Timestamp-level embeddings lose high-level temporal semantics, requiring a mechanism to "discover" clinically meaningful temporal patterns from the data.

**Goal**: How to discover and align cross-modal temporal patterns from multimodal EHRs, and how to leverage these patterns to improve clinical predictions?

**Key Insight**: Drawing inspiration from object-centric learning (Slot Attention) in computer vision, "temporal prototypes" are analogized to "object slots" to iteratively extract meaningful temporal patterns from the data.

**Core Idea**: Shared initialized learnable temporal prototypes + iterative refinement via Slot Attention + cross-modal alignment via TP-NCE + multi-scale temporal encoding.

## Method

### Overall Architecture
Input: Irregular multivariate time series (MITS) + clinical note sequences → Encoded into regular temporal embeddings respectively → Multi-scale temporal series extraction → Shared prototype vectors aggregate both modalities separately via Slot Attention → Alignment with TP-NCE + reconstruction loss → Transformer-based fusion of prototypes and timestamp embeddings → Classification and prediction.

### Key Designs

1. **Multimodal EHR Encoding**:

    - **Function**: To encode irregular time series and clinical notes into unified, temporally aligned embeddings.
    - **Mechanism**: For time series, a gating mechanism is used to fuse interpolation embeddings and mTAND embeddings ($z^{TS} = g \odot e_{imp}^{TS} + (1-g) \odot e_{mTAND}^{TS}$); for clinical notes, BERT-Tiny encodes the [CLS] token, which is then temporally aligned via mTAND.
    - **Design Motivation**: The gating mechanism allows the model to adaptively select the advantages of both interpolation-based and attention-based temporal representations.

2. **Cross-Modal Temporal Pattern Discovery (Core)**:

    - **Function**: To extract $K$ cross-modal temporal prototypes from multi-scale temporal embeddings using Slot Attention.
    - **Mechanism**:
        - **Multi-scale**: 3 convolutional blocks + mean pooling (stride=2) are applied to time series embeddings to generate 3 scales ($T, T/2, T/4$), concatenated to obtain $z_{MS}^{TS} \in \mathbb{R}^{T' \times D}$.
        - **Shared initial prototypes**: $P^{Shared} \in \mathbb{R}^{K \times D}$, initialized from $\mathcal{N}(\mu, \text{diag}(\sigma))$ and optimized during training.
        - **Slot Attention iteration**: Prototypes act as queries, and modality embeddings act as keys/values, computing attention weights → weighted aggregation update → GRU to refine prototypes. Iterated 3 times.
    - **Design Motivation**: Slot Attention automatically groups data into different "slots," where each slot corresponds to a temporal pattern. The multi-scale design ensures patterns at different temporal granularities are captured.

3. **TP-NCE Contrastive Loss**:

    - **Function**: To align the discovered temporal prototypes of both modalities.
    - **Mechanism**: An InfoNCE variant, where for samples in a batch $B$, $P^{TS}$ and $P^{Text}$ from the same ICU stay are positive pairs, and those from different stays are negative pairs. The similarity is the weighted sum of cosine similarities of the $K$ prototypes: $\text{sim}(\cdot) = \sum_{k=1}^K \beta_k \langle P^{TS}(k), P^{Text}(k) \rangle$, where weights $\beta$ are generated by a global embedding MLP.
    - **Design Motivation**: To ensure that the temporal prototypes of both modalities have consistent semantics, while the weighting mechanism allows different prototypes to have varying importances.

4. **Auxiliary Reconstruction Loss**:

    - **Function**: To reconstruct time series from $P^{TS}$ and text embeddings from $P^{Text}$.
    - **Mechanism**: Two Transformer decoders perform the reconstruction separately using MSE loss.
    - **Design Motivation**: To ensure that the prototypes retain core modality information, preventing information loss caused by contrastive learning.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{TPNCE} + \lambda_2 \mathcal{L}_{Recon}$
- Classification loss: Cross-entropy; batch size=128, lr=4e-5, Adam, cosine annealing, early stopping.
- Single RTX-3090, approximately 1 hour/run.

## Key Experimental Results

### Main Results (MIMIC-III)

| Method | 48-IHM AUROC↑ | 48-IHM F1↑ | 24-PHE AUROC↑ | 24-PHE AUPR↑ |
|------|-------------|-----------|-------------|-------------|
| UTDE (Time-series SOTA) | 86.14 | 49.29 | 73.62 | 36.80 |
| mTAND (Note SOTA) | 85.40 | 35.76 | 82.14 | 54.57 |
| MMTM (Multimodal SOTA) | 87.88 | 51.54 | 81.46 | 51.88 |
| DAFT | 87.53 | 51.95 | 81.18 | 50.91 |
| **CTPD (Ours)** | **88.15** | **53.85** | **83.34** | **56.39** |

### Ablation Study

| Configuration | 48-IHM AUROC | 48-IHM F1 | 24-PHE AUROC |
|------|-------------|-----------|-------------|
| CTPD Full | 88.15 | 53.85 | 83.34 |
| w/o Prototype | 86.89 | 48.47 | 82.24 |
| w/o timestamp embedding | 87.18 | 45.85 | 82.41 |
| w/o Multi-scale | 87.59 | 49.74 | 83.11 |

### Key Findings
- **Cross-modal temporal prototypes contribute the most to F1**: Removing the prototypes drops the 48-IHM F1 from 53.85 to 48.47 (-5.38 absolute loss), demonstrating that the discovered temporal patterns are extremely crucial for prediction.
- **Multimodal fusion outperforms unimodal baseline**: The best unimodal time-series AUROC is 86.14, and note-only is 82.14, while the multimodal CTPD achieves 88.15, indicating complementary strengths between the two data modalities.
- **Strong statistical significance**: On 5 out of 6 metrics, $p < 0.05$ (including 4 with $p < 0.001$), with particularly significant improvements in 24-PHE.
- **Contribution of multi-scale representation**: Removing the multi-scale mechanism drops the F1 from 53.85 to 49.74, indicating that patterns of different temporal granularities are indeed complementary.

## Highlights & Insights
- **Slot Attention for temporal pattern discovery**: Slot Attention, typically used for object segmentation in computer vision, is cleverly migrated to temporal data, allowing each "slot" to automatically learn a temporal pattern. This idea can be generalized to any task that requires extracting high-level concepts from sequential data.
- **Shared initialized prototypes ensure cross-modal correspondence**: Both modalities refine separately starting from the same initial prototype and are then aligned using contrastive learning, ensuring semantic consistency of the patterns.
- **Multi-scale temporal encoding**: Multi-scale representations are generated using simple convolution + pooling, which effectively captures acute changes and long-term trends without adding substantial computational overhead.

## Limitations & Future Work
- Only validated on MIMIC-III/IV; ICU data may not represent outpatient or community scenarios.
- Requires paired time series and clinical notes; scenarios with missing modalities are not handled.
- BERT-Tiny as a text encoder has limited capacity; stronger LLMs (e.g., Clinical LLaMA) could potentially enhance text representation.
- The number of prototypes $K$ is a hyperparameter, and the optimal value may vary by task.
- Interpretability of the learned temporal prototypes has not been analyzed (e.g., matching which prototypes correspond to which clinical events).

## Related Work & Insights
- **vs UTDE**: UTDE also offers irregular time-series encoding (gated mTAND). CTPD adds cross-modal prototype discovery on top of it, improving the AUROC from 86.14 to 88.15.
- **vs DrFuse**: DrFuse uses Transformers for cross-modal token interactions but lacks high-level temporal pattern discovery, resulting in lower performance across all metrics compared to CTPD.
- **vs Slot Attention (Locatello 2020)**: While the original Slot Attention discovers objects in images, CTPD generalizes it to cross-modal scenarios involving time-series and text.

## Rating
- Novelty: ⭐⭐⭐⭐ The transfer of Slot Attention to temporal prototype discovery is highly innovative, and the TP-NCE design is reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough ablations and rigorous statistical testing, but on limited datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear formulation and very intuitive clinical motivation in Fig.1.
- Value: ⭐⭐⭐⭐ Provides a new pattern discovery paradigm for clinical EHR analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Foundation Models for Clinical Records at Health System Scale](../../ICML2025/time_series/foundation_models_for_clinical_records_at_health_system_scale.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ACL 2025\] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)
- [\[ICML 2025\] Causal Discovery from Conditionally Stationary Time Series](../../ICML2025/time_series/causal_discovery_from_conditionally_stationary_time_series.md)

</div>

<!-- RELATED:END -->
