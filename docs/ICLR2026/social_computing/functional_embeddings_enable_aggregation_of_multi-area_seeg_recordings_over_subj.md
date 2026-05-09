---
title: >-
  [Paper Note] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI
description: >-
  [ICLR 2026][Social Computing][Brain-Computer Interface] This paper proposes FunctionalMap, a framework that learns subject-agnostic functional embeddings from intracranial local field potentials (LFPs) via contrastive learning, serving as a "functional coordinate system" to replace unreliable MNI anatomical coordinates. Combined with a Transformer, the framework enables cross-subject and cross-electrode neural data aggregation and signal reconstruction, validated on a multi-area SEEG dataset from 20 subjects.
tags:
  - ICLR 2026
  - Social Computing
  - Brain-Computer Interface
  - SEEG
  - Functional Embedding
  - Contrastive Learning
  - Transformer
  - Cross-Subject Modeling
  - Neural Signals
date: 2026-05-08
content_hash: 2d52a38669a2175d
---

# Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI

**Conference**: ICLR 2026
**arXiv**: [2510.27090](https://arxiv.org/abs/2510.27090)
**Code**: [GitHub](https://github.com/ICLR-Functional-Embedding/ICLR2026_Functional_Map)
**Area**: Social Computing
**Keywords**: Brain-Computer Interface, SEEG, Functional Embedding, Contrastive Learning, Transformer, Cross-Subject Modeling, Neural Signals

## TL;DR

This paper proposes FunctionalMap, a framework that learns subject-agnostic functional embeddings from intracranial local field potentials (LFPs) via contrastive learning, serving as a "functional coordinate system" to replace unreliable MNI anatomical coordinates. Combined with a Transformer, the framework enables cross-subject and cross-electrode neural data aggregation and signal reconstruction, validated on a multi-area SEEG dataset from 20 subjects.

## Background & Motivation

### Limitations of Prior Work

**State of the Field**: Cross-subject modeling of intracranial neural recordings (e.g., SEEG/DBS) faces two core challenges:

**Anatomical variability and inconsistent electrode coverage**: The number, location, and coverage of electrodes vary according to clinical requirements. Standard MNI atlas alignment assumes spatial correspondence equals functional similarity, yet **recordings at matched anatomical coordinates often capture different functional roles**, and in extreme cases, entirely different brain regions.

**Heterogeneity of multi-area recordings**: Modern DBS surgery simultaneously samples from multiple basal ganglia and thalamic nuclei (GPi, STN, VO, VA, VIM, etc.), offering a unique opportunity to study inter-area communication, but this heterogeneity amplifies the alignment problem.

**Core Assumption**: Neural signals can be more reliably aligned across subjects through their functional characteristics rather than anatomical coordinates.

## Method

### Overall Architecture

FunctionalMap consists of two stages:
1. **Functional Embedding Learning**: A Siamese encoder with contrastive learning maps LFP signals to 32-dimensional subject-agnostic functional identity embeddings.
2. **Functional Transformer**: Uses functional embeddings as token coordinates to model inter-area relationships across variable numbers of channels, performing masked-area reconstruction.

### Key Designs

**Functional Embedding Network**: A lightweight CNN encoder $f_\theta: \mathbb{R}^T \to \mathbb{R}^d$ ($d=32$) maps 10-second LFP segments to the embedding space.

**Paired Siamese Contrastive (PSC)**: Same-region pairs are pulled together; different-region pairs are pushed apart beyond a margin of $m=0.5$.

**Modified Supervised Contrastive (MSC)**: Multi-positive InfoNCE with an intra-class variance penalty $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}$, operating on the hypersphere and emphasizing angular separation. MSC generalizes better across channels.

**Functional Transformer**:
- Task: mask all channels from a target brain region and predict them from remaining regions
- A 1D convolutional tokenizer converts source channels into temporal patch features, fused with functional embeddings
- Standard pre-LN encoder-decoder architecture, with no subject ID input
- Loss: $\mathcal{L} = \text{MSE}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda(1 - \rho(\hat{\mathbf{Y}}, \mathbf{Y}))$

### Loss & Training
- Functional embeddings: contrastive loss (PSC or MSC) on 10-second LFP segments
- Transformer: MSE + Pearson correlation loss, jointly trained across 11 subjects
- Single shared model, no subject-specific fine-tuning

## Key Experimental Results

### Main Results

| Setting | Held-out Time Segment Accuracy | Held-out Channel Accuracy |
|---------|-------------------------------|--------------------------|
| Single-subject | 75.78% ± 17.90% | 45.79% ± 18.44% |
| **Multi-subject Joint** | **80.71% ± 11.41%** | **49.18% ± 12.11%** |

### Ablation Study

| Coordinate System | Pearson Correlation r | Significance |
|------------------|-----------------------|--------------|
| MNI Coordinates | Baseline | — |
| Functional-1 (PSC) | Positive trend | Not significant |
| **Functional-2 (MSC)** | **Significantly better than MNI** | $p \approx 0.002$ |

### Key Findings
- Functional embeddings successfully cluster brain regions across subjects with zero-shot transfer to unseen channels
- Functional coordinates significantly outperform anatomical coordinates (MSC embedding)
- MNI failure cases: channels sharing identical MNI coordinates are correctly distinguished by functional embeddings

## Highlights & Insights
- Using "function as a coordinate system" in place of anatomical coordinates represents a compelling new paradigm
- Geometric differences in contrastive learning are meaningful: PSC yields compact centroids, while MSC achieves angular separation
- Masked-area reconstruction requires no behavioral labels, relying purely on inter-area neural circuit information

## Limitations & Future Work
- Relies on region labels for contrastive training
- Limited to the basal ganglia–thalamic circuit; not validated on cortical ECoG
- Only signal reconstruction is evaluated; downstream tasks such as behavioral decoding remain untested

## Related Work & Insights
- **vs. MNI coordinate methods**: MNI assumes anatomy equals function; this work demonstrates that functional coordinates are more reliable
- **vs. PopT**: PopT aggregates frozen single-channel embeddings, whereas this work learns transferable functional coordinates

## Rating
- Novelty: ⭐⭐⭐⭐ Using a functional coordinate system to replace anatomical coordinates is a compelling new paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation validation + real data + multi-level ablation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables
- Value: ⭐⭐⭐⭐ Direct value for clinical neuroscience and BCI

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)

<!-- RELATED:END -->
