---
title: >-
  [Paper Note] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI
description: >-
  [ICLR 2026][Social Computing][Brain-Computer Interface] This paper proposes FunctionalMap, a framework that uses contrastive learning to learn subject-agnostic functional embeddings from intracranial local field potentia…
tags:
  - "ICLR 2026"
  - "Social Computing"
  - "Brain-Computer Interface"
  - "SEEG"
  - "Functional Embedding"
  - "Contrastive Learning"
  - "Transformer"
  - "Cross-Subject Modeling"
  - "Neural Signals"
date: 2026-05-08
content_hash: bffe6b28d40b0c3f
---

# Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI

**Conference**: ICLR 2026
**arXiv**: [2510.27090](https://arxiv.org/abs/2510.27090)
**Code**: [GitHub](https://github.com/ICLR-Functional-Embedding/ICLR2026_Functional_Map)
**Area**: Social Computing
**Keywords**: Brain-Computer Interface, SEEG, Functional Embedding, Contrastive Learning, Transformer, Cross-Subject Modeling, Neural Signals

## TL;DR

This paper proposes FunctionalMap, a framework that uses contrastive learning to learn subject-agnostic functional embeddings from intracranial local field potentials (LFPs) as a "functional coordinate system," replacing unreliable MNI anatomical coordinates. Combined with a Transformer, it enables cross-subject and cross-electrode aggregation of neural data and signal reconstruction, validated on a multi-area SEEG dataset from 20 subjects.

## Background & Motivation

Cross-subject modeling of intracranial neural recordings (e.g., SEEG/DBS) faces two core challenges:

**Anatomical variability and inconsistent electrode coverage**: The number, placement, and coverage of electrodes vary according to clinical needs. Standard MNI atlas alignment assumes spatial correspondence implies functional similarity, but **recordings at matched anatomical coordinates often capture different functional roles**, and in extreme cases, entirely different brain regions.

**Heterogeneity of multi-area recordings**: Modern DBS surgery simultaneously samples from multiple basal ganglia and thalamic nuclei (GPi, STN, VO, VA, VIM, etc.), offering a unique opportunity to study inter-area communication, but the heterogeneity amplifies alignment challenges.

Limitations of existing approaches:
- EEG foundation models (e.g., LaBraM) assume fixed high-density electrode grids
- MNI coordinate-based methods (Mentzelopoulos et al., 2024) rely on unreliable anatomical localization
- PopT (Chau et al., 2025) aggregates frozen single-channel embeddings but uses positional encodings

**Core assumption**: Neural signals can be more reliably aligned across subjects via their functional characteristics rather than anatomical coordinates.

## Method

### Overall Architecture

FunctionalMap operates in two stages:

1. **Functional embedding learning**: A Siamese encoder with contrastive learning maps LFP signals to 32-dimensional subject-agnostic functional identity embeddings.
2. **Functional Transformer**: Uses functional embeddings as token coordinates to model inter-area relationships across a variable number of channels and performs masked-region reconstruction.

### Key Designs

**Functional embedding network**: A lightweight CNN encoder $f_\theta: \mathbb{R}^T \to \mathbb{R}^d$ ($d=32$) maps 10-second LFP segments to an embedding space. Two contrastive learning variants are proposed:

**Method 1: Paired Siamese Contrastive (PSC)**:

$$\mathcal{L}_{\text{pair}} = \frac{1}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \left[(1-y_{ij}) d_{ij}^2 + y_{ij} (\max(0, m-d_{ij}))^2\right]$$

Pairs from the same region ($y_{ij}=0$) are pulled together; pairs from different regions ($y_{ij}=1$) are pushed apart beyond margin $m=0.5$.

**Method 2: Modified Supervised Contrastive (MSC)**: Multi-positive InfoNCE plus an intra-class variance penalty:

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}$$

where $\mathcal{L}_{\text{sup}}$ uses cosine similarity (temperature $\tau=0.2$) and $\mathcal{L}_{\text{var}}$ penalizes the variance of same-region embeddings ($\lambda_{\text{var}}=0.05$). MSC operates on the hypersphere, emphasizing angular separation.

**Training sampling**: Input pairs may come from the same session or across subjects/sessions with asynchronous timing. Region-consistent random sampling ensures the encoder learns region-specific neural signatures robust to subject and session variability.

**Functional Transformer (masked-region reconstruction)**:
- **Task**: Mask all channels from a target brain region and require the model to predict them from the remaining regions.
- **Tokenization**: A 1D convolutional tokenizer converts source channels into temporal patch features, fused with functional embeddings; target channels are represented by learned query bases fused with functional embeddings.
- **Architecture**: Standard pre-LN encoder-decoder Transformer with no subject IDs or subject-specific heads.
- **Objective**: MSE plus a correlation term:

$$\mathcal{L} = \text{MSE}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda(1 - \rho(\hat{\mathbf{Y}}, \mathbf{Y})), \quad \lambda=0.05$$

The correlation term prevents the pure MSE objective from collapsing to amplitude-reduced or flat predictions.

### Loss & Training

- Functional embeddings: Contrastive loss (PSC or MSC) on 10-second LFP segments.
- Transformer: MSE + Pearson correlation loss, trained jointly across 11 subjects.
- A single shared model with no subject-specific fine-tuning.

## Key Experimental Results

### Main Results

**Dataset**: Intracranial LFP recordings from 20 patients with dystonia, covering GPi/STN/VO/VA/VIM/PPN/SNr, totaling 442.86 electrode-hours.

**Single-subject functional embeddings**:

| Evaluation Setting | Accuracy (Mean±SD) |
|---|---|
| Held-out time segments (seen channels) | 75.78% ± 17.90% |
| Held-out channels (>3 channels/region) | 45.79% ± 18.44% (above chance) |

**Multi-subject joint training vs. single-subject**:

| Setting | Held-out Time Segments | Held-out Channels |
|---|---|---|
| Single-subject | 75.78% ± 17.90% | 45.79% ± 18.44% |
| Multi-subject joint | **80.71% ± 11.41%** | **49.18% ± 12.11%** |

The joint model improves both metrics by approximately 5% without subject-specific fine-tuning.

### Ablation Study

**Coordinate system ablation (masked-region reconstruction, predicting VO channels)**:

| Coordinate System | Pearson Correlation r |
|---|---|
| MNI coordinates | Baseline |
| Functional-1 (PSC) | Positive trend, not significant |
| **Functional-2 (MSC)** | **Significantly outperforms MNI** ($p \approx 0.002$) |

**Comparison between PSC and MSC**:

| Method | Held-out Time Seg. Accuracy | Held-out Channel Accuracy | Characteristics |
|---|---|---|---|
| PSC | Slightly higher | Lower | Euclidean space, compact centroid clustering |
| MSC | Slightly lower | **Higher** | Hypersphere, angular separation, stronger channel generalization |

**Comparison with subject-specific baselines**:

Transformer + Functional-2 significantly outperforms all subject-specific baselines (linear FIR, TCN, 2-layer GRU, CopyBest), with all corrected $p < 0.001$.

### Key Findings

1. **Functional embeddings successfully cluster brain regions**: Clear region-consistent clusters form across subjects.
2. **Zero-shot transfer to unseen channels**: The joint model handles new electrodes without fine-tuning.
3. **Functional coordinates significantly outperform anatomical coordinates**: MSC embeddings substantially improve reconstruction performance.
4. **Failure case of MNI**: When four VO electrodes share nearly identical MNI coordinates, the MNI model produces similar reconstructions for all; functional embeddings yield channel-specific predictions.
5. **Simulation validation**: On simulated data with known parameters, embeddings correctly capture frequency-domain features, and perturbation analysis confirms sensitivity to physiologically relevant components.

## Highlights & Insights

1. **A compelling core hypothesis**: Using "function as a coordinate system" in place of anatomical coordinates provides robust alignment under inconsistent localization and heterogeneous electrode configurations.
2. **Progressive validation hierarchy**: Simulation → single-subject → multi-subject → Transformer reconstruction → coordinate system ablation, each layer validating a different aspect of the hypothesis.
3. **Meaningful geometric distinction in contrastive learning**: The compact centroid structure of PSC versus the angular separation of MSC leads to meaningfully different generalization behaviors.
4. **Elegant self-supervised pretraining objective**: Masked-region reconstruction requires no behavioral labels and purely exploits inter-area neural circuit information.
5. **Clinical significance**: Provides a foundation for cross-patient data sharing in clinical neurotechnologies such as DBS.

## Limitations & Future Work

1. **Dependence on region labels**: Contrastive training requires knowledge of electrode region labels, limiting fully unsupervised extension.
2. **Restricted to basal ganglia–thalamic circuits**: Generalization to cortical ECoG and spike data has not been validated.
3. **Transformer trained on only 11/20 subjects**: Constrained by MNI data availability.
4. **Limited task scope**: Only signal reconstruction is validated; downstream tasks such as behavioral decoding remain untested.
5. A comprehensive comparison with population-level pretraining frameworks such as PopT is still needed.
6. Weakly supervised or self-supervised objectives replacing region labels are worth exploring.

## Related Work & Insights

- **Mentzelopoulos et al. (2024)**: MNI coordinates with subject-specific heads; found no significant improvement from positional encodings.
- **PopT (Chau et al., 2025)**: Population-level Transformer aggregating frozen single-channel embeddings.
- **NDT/STNDT**: Transformers for neural population modeling assuming stable channel identity.
- Insight: Combining functional coordinates with population-level pretraining may yield the best results; the concept of "function as a coordinate system" may generalize to other sensor alignment problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using functional coordinates in place of anatomical coordinates is a compelling new paradigm.
- **Technical Depth**: ⭐⭐⭐⭐ — The two-stage design combining contrastive learning and Transformer is thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Simulation validation + real data + multi-level ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Directly relevant to clinical neuroscience and BCI.
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)

</div>

<!-- RELATED:END -->
