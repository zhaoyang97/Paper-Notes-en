---
title: >-
  [Paper Note] BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG
description: >-
  [NeurIPS 2025][Model Compression][Intracranial EEG] BaRISTA systematically investigates spatial encoding scales (electrode/parcel/lobe) for iEEG Transformers…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Intracranial EEG"
  - "Spatiotemporal Transformer"
  - "Spatial Encoding Scale"
  - "Masked Reconstruction"
  - "Pretraining"
date: 2026-05-08
content_hash: 08ef5159ec89ca57
---

# BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG

**Conference**: NeurIPS 2025
**arXiv**: [2512.12135](https://arxiv.org/abs/2512.12135)
**Code**: [https://github.com/ShanechiLab/BaRISTA](https://github.com/ShanechiLab/BaRISTA)
**Area**: Neuroscience / Foundation Models
**Keywords**: Intracranial EEG, Spatiotemporal Transformer, Spatial Encoding Scale, Masked Reconstruction, Pretraining

## TL;DR
BaRISTA systematically investigates spatial encoding scales (electrode/parcel/lobe) for iEEG Transformers, finding that atlas parcel-level encoding combined with spatial masked reconstruction achieves 86.2% AUC on language task decoding (vs. PopT 79.5%). The choice of encoding scale has greater impact than masking strategy, and the model generalizes well across subjects.

## Background & Motivation

**Background**: iEEG provides high spatiotemporal resolution brain activity recordings. Transformer-based pretrained models (PopT, Brant) have been applied to iEEG, but the choice of spatial encoding has not been systematically studied.

**Limitations of Prior Work**: Electrode positions vary entirely across patients, making channel-level encoding difficult to generalize across subjects. Whether parcel-level or lobe-level encoding is superior, and how spatial encoding interacts with masking strategy, remain open questions.

**Key Challenge**: Fine-grained spatial resolution (channel) carries the most information but lacks cross-patient consistency; coarse-grained resolution (lobe) is consistent across patients but may sacrifice local information.

**Goal**: Systematically compare three spatial encoding scales to identify the optimal encoding–masking combination.

**Key Insight**: Channel, atlas parcel, and lobe are treated as experimental variables and systematically ablated within a masked reconstruction pretraining framework.

**Core Idea**: By systematically comparing three spatial encoding scales in an iEEG Transformer, the paper identifies atlas parcel-level encoding as the optimal spatial granularity—balancing cross-patient consistency with local information retention.

## Method

### Overall Architecture
iEEG data (2048 Hz) → **Temporal tokenization** (Dilated CNN extracting 250 ms patches) → **Spatial encoding** (learnable embeddings $E_j$ at channel/atlas parcel/lobe granularity) → token $S_{ij} = B_{ij} + E_j$ → **Spatial masking** (randomly selected spatial categories) → **Transformer** (12 layers / 4 heads / $d=64$ + RoPE) → **EMA target reconstruction** (MSE between online encoder and EMA target encoder)

### Key Designs

1. **Three Spatial Encoding Scales**:

    - Channel: MNI coordinates $(x, y, z)$ → learnable embeddings (finest granularity but inconsistent across patients)
    - Atlas parcels: Destrieux atlas parcellation (intermediate granularity, consistent across patients)
    - Lobes: Cerebral lobes + subcortical regions (coarsest but most stable)

2. **Spatial Masked Reconstruction Pretraining**:

    - Randomly selected spatial categories are masked (e.g., all electrode patches within a given brain region)
    - Online tokenizer $\mathcal{F}$ and EMA target tokenizer $\tilde{\mathcal{F}}$ (momentum warmup from 0 to 0.996)
    - $\mathcal{L} = \frac{1}{|B_{target}|}\sum \|\tilde{B}_{ij} - \hat{B}_{ij}\|_2^2$

3. **Interleaved Spatiotemporal Sequences**: Spatial and temporal tokens are interleaved so that attention jointly captures spatiotemporal dependencies.

### Loss & Training
- Brain Treebank dataset: 10 epilepsy patients, 26 sessions, 2048 Hz
- Evaluation via pretraining followed by linear probing

## Key Experimental Results

### Main Results (Downstream Classification AUC %)

| Encoding / Masking | Sentence Onset | Speech / Non-Speech |
|---|---|---|
| Channel | 77.8% | 76.4% |
| **Parcel** | **86.2%** | **86.9%** |
| Lobe | 84.2% | 84.1% |
| PopT baseline | 79.5% | 77.5% |
| Brant baseline | 76.7% | 69.1% |

### Ablation Study (ANOVA)

| Factor | p-value | Effect Size |
|---|---|---|
| Encoding scale | **p<1e-3** | Large |
| Masking strategy | p=0.01–0.04 | Medium |
| Interaction | Channel encoding + Channel masking yields best match | — |

### Key Findings
- Parcel encoding significantly outperforms channel encoding (+8.4% sentence onset, +10.5% speech)—anatomical priors are more informative than precise coordinates.
- The effect of encoding scale exceeds that of masking strategy—selecting the right encoding is more critical than selecting the right masking approach.
- Cross-subject generalization: held-out subjects achieve 84.1% AUC (vs. 86.9% with target subjects), confirming that parcel-level encoding facilitates cross-patient generalization.
- Performance scales positively with data volume: continuous improvement is observed as pretraining data increases from 5% to 75%.

## Highlights & Insights
- **Spatial encoding scale is a critical yet underexplored design choice**: prior works default to channel-level encoding, while BaRISTA demonstrates that atlas parcel-level encoding is superior.
- **"Intermediate granularity outperforms finest granularity"**: although channel-level encoding carries the most information, its cross-patient inconsistency leads to poor generalization, whereas atlas parcels balance precision and generalizability.

## Limitations & Future Work
- Relies solely on anatomical parcellation; functional brain region encoding is not explored.
- Only spatial masking is evaluated; joint spatiotemporal masking is not tested.
- Experiments are conducted at a single sampling rate (2048 Hz).
- The dilated CNN temporal tokenizer may not be optimal.

## Related Work & Insights
- **vs. PopT**: PopT employs channel-level encoding; BaRISTA demonstrates that parcel-level encoding is superior (+6.7%).
- **vs. Brant**: Brant uses region-level encoding but provides no systematic ablation; BaRISTA offers a comprehensive analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ — Systematic ablation of spatial encoding scales is conducted for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three scales × multiple masking strategies × ANOVA + cross-subject + data scaling.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous experimental design.
- Value: ⭐⭐⭐⭐ — Provides critical guidance for iEEG foundation model design.
- The hierarchical spatial structure of neural signals calls for multi-scale modeling—coarse and fine granularities are complementary.
- Larger spatial scales (lobe-level) improve decoding performance; pretraining substantially benefits low-data regimes.
- The core contribution lies in the simplicity and effectiveness of the design rationale.
- Experimental results thoroughly validate the central hypothesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](../../AAAI2026/model_compression/eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[NeurIPS 2025\] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)

</div>

<!-- RELATED:END -->
