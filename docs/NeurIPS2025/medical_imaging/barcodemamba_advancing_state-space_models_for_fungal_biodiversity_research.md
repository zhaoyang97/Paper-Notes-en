---
title: >-
  [Paper Note] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research
description: >-
  [NeurIPS 2025][Medical Imaging][DNA barcoding] BarcodeMamba+ is a foundation model for fungal DNA barcode classification—built on a state-space model architecture, adopting a pretrain-then-finetune paradigm to leverage partially labeled data, and incorporating hierarchical label smoothing, weighted loss, and multi-head outputs to address fungal classification challenges (93% of specimens lack species-level labels). It surpasses existing methods across all taxonomic ranks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - DNA barcoding
  - fungal taxonomy
  - state-space models
  - foundation models
  - hierarchical classification
date: 2026-05-08
content_hash: 4bb5ec0302111ed1
---

# BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research

**Conference**: NeurIPS 2025
**arXiv**: [2512.15931](https://arxiv.org/abs/2512.15931)
**Code**: [GitHub](https://github.com/bioscan-ml/BarcodeMamba)
**Area**: Bioinformatics / Genomics
**Keywords**: DNA barcoding, fungal taxonomy, state-space models, foundation models, hierarchical classification

## TL;DR

BarcodeMamba+ is an SSM-based foundation model for fungal ITS DNA barcode classification. By adopting a pretrain-then-finetune paradigm to leverage large-scale unlabeled sequences, and incorporating three enhancements—hierarchical label smoothing, inverse square-root weighted loss, and multi-head outputs—it substantially outperforms BLAST, CNN, and Transformer baselines across all taxonomic ranks on three test sets, achieving a top species-level accuracy of 88.9%.

## Background & Motivation

DNA barcoding is a cornerstone of large-scale automated biodiversity monitoring, yet fungal taxonomy poses extreme challenges:

- **Severe label sparsity**: Up to 93% of collected fungal specimens lack species-level annotations.
- **Heavy long-tail distribution**: 5.23M training sequences span 14.7K species with highly imbalanced class frequencies.
- **Bottlenecks of traditional methods**: BLAST is slow (208.6 ms/sample) and generalizes poorly; fully supervised CNN/Transformer training is limited under sparse annotation conditions.
- **Foundation model opportunity**: Vast quantities of unlabeled DNA sequences can be leveraged via pretraining to learn generalizable representations, followed by fine-tuning with limited labeled data.

**Mechanism**: Mamba (an efficient SSM architecture) is introduced for DNA barcode classification, combined with a pretrain-then-finetune paradigm and hierarchical classification enhancements to address data sparsity and long-tail challenges in fungal taxonomy.

## Method

### Overall Architecture

A two-stage training paradigm is adopted:

1. **Pretraining stage**: Self-supervised next-token prediction pretraining on 5.23M ITS sequences from the UNITE+INSD dataset, without using taxonomic labels.
2. **Fine-tuning stage**: Classification heads are added and fine-tuned on labeled data, incorporating three hierarchical classification enhancements.

A BPE tokenizer is used for DNA sequences (rather than character-level or k-mer tokenization), as BPE is empirically validated as the optimal choice for fungal ITS data.

### Key Designs

1. **Mamba SSM Architecture**

   - Based on state-space models with linear time complexity, well-suited for large-scale biological sequences.
   - Base variant: 12.1M parameters (comparable to the CNN baseline); large variant: 49.2M parameters.
   - Compared to Transformer-based BarcodeBERT (44.6M parameters), Mamba achieves a better balance of parameter efficiency and inference speed.

2. **Hierarchical Label Smoothing**

   - Exploits the taxonomic hierarchy (kingdom / phylum / class / order / family / genus / species).
   - Assigns smoothing probabilities in the softmax targets according to taxonomic distance.
   - Allows taxonomically similar classes to receive partial probability mass, enhancing generalization.

3. **Inverse Square-Root Weighted Loss**

   - Assigns higher weights to rare classes to address long-tail distribution.
   - Prevents the model from being dominated by high-frequency classes.

### Loss & Training

- Pretraining uses next-token prediction (language-model style).
- Fine-tuning uses weighted cross-entropy combined with hierarchical label smoothing.
- Multi-head outputs: independent classification heads for each taxonomic rank (phylum / class / order / family / genus / species).
- BPE tokenization outperforms both character-level and k-mer tokenization.

## Key Experimental Results

### Main Results

Species-level accuracy (%) on three test sets:

| Model | Yeast | Filamentous | MycoAI | Params | Inference Time |
|-------|-------|-------------|--------|--------|----------------|
| BLAST | 75.4 | 33.4 | 55.0 | N/A | 208.6ms |
| MycoAI-CNN | 60.0 | 28.2 | 57.1 | 11.6M | 11.8ms |
| MycoAI-BERT | 33.5 | 16.6 | 39.3 | 18.4M | 4.5ms |
| CNN Encoder | 67.6 | 31.4 | 72.6 | 12.1M | 5.8ms |
| BarcodeBERT | 59.1 | 27.7 | 58.9 | 44.6M | 8.8ms |
| **BarcodeMamba+** | **80.6** | **46.5** | **81.7** | 12.1M | 8.0ms |
| **BarcodeMamba+ (large)** | **83.6** | **50.4** | **88.9** | 49.2M | 14.7ms |

### Ablation Study

**Pretraining vs. fully supervised** (BPE tokenization, species-level accuracy on MycoAI test set):

| Training Strategy | Accuracy |
|-------------------|----------|
| Fully supervised (no pretraining) | 78.6% |
| Pretrain + fine-tune | **81.7%** |

Pretraining yields more pronounced gains under k-mer tokenization (77.0% → 81.1%), confirming the advantage of pretraining in annotation-scarce scenarios.

**Tokenization comparison** (pretrain + fine-tune, MycoAI species-level):

| Tokenization | Accuracy |
|--------------|----------|
| Char | 79.0% |
| k-mer | 81.1% |
| **BPE** | **81.7%** |

### Key Findings

- BarcodeMamba+ achieves comprehensive superiority across all taxonomic ranks and all test sets, reaching 81.7% species-level accuracy on MycoAI—9.1 percentage points above the second-best CNN Encoder (72.6%).
- The advantage is most pronounced on the Filamentous test set, which exhibits the largest distribution shift (46.5% vs. 31.4%, a margin of 15 percentage points).
- Scaling the model to 49.2M parameters improves MycoAI species-level accuracy from 81.7% to 88.9%, confirming architectural scalability.
- Inference speed is 8 ms/sample, more than 25× faster than BLAST (208.6 ms).

## Highlights & Insights

- The **pretrain-then-finetune paradigm** demonstrates a substantial advantage in the genomics domain where annotations are extremely sparse (93% lack species-level labels)—an advantage unattainable by conventional fully supervised approaches.
- The linear complexity of SSM architectures over DNA sequences makes them particularly well-suited for large-scale biodiversity monitoring.
- The three hierarchical classification enhancements (label smoothing + weighted loss + multi-head outputs) each contribute meaningful performance gains and are mutually complementary.
- BPE tokenization outperforms k-mer and character-level tokenization on DNA sequences, consistent with empirical findings in NLP.

## Limitations & Future Work

- Validation is limited to the fungal ITS region; transferability to other taxonomic groups (e.g., insect COI barcodes) remains to be investigated.
- No comparison is made against protein language models (e.g., ESM) or recent DNA foundation models.
- Absolute accuracy on the Filamentous test set remains below 50%, indicating room for improvement under extreme distribution shift.
- The large model variant nearly doubles inference time (14.7 ms), which may be a concern for real-time deployment scenarios.

## Related Work & Insights

- Compared to BarcodeBERT (a Transformer-based foundation model), Mamba achieves superior performance with fewer parameters.
- The multi-head output and hierarchical enhancement strategies introduced by MycoAI are systematically integrated into the SSM architecture in this work.
- This work validates the practical utility of the foundation model paradigm for biodiversity monitoring.

## Rating

- Novelty: ⭐⭐⭐ Applying Mamba to DNA classification is a well-motivated architectural choice but not a breakthrough innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three test sets with comprehensive ablations over tokenization, training paradigm, and enhancement strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation with rigorous experimental design.
- Value: ⭐⭐⭐⭐ Offers practical utility for biodiversity research with open-source code.

---

# BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research

**Conference**: NeurIPS 2025
**arXiv**: [2512.15931](https://arxiv.org/abs/2512.15931)
**Code**: [GitHub](https://github.com/bioscan-ml/BarcodeMamba)
**Area**: Bioinformatics / Genomics
**Keywords**: DNA barcoding, fungal taxonomy, state-space models, foundation models, hierarchical classification

## TL;DR

BarcodeMamba+ is a foundation model for fungal DNA barcode classification—built on a state-space model architecture, adopting a pretrain-then-finetune paradigm to leverage partially labeled data, and incorporating hierarchical label smoothing, weighted loss, and multi-head outputs to address fungal classification challenges (93% of specimens lack species-level labels). It surpasses existing methods across all taxonomic ranks.

## Background & Motivation

**Background**: DNA barcoding is foundational to automated biodiversity monitoring, but fungal classification is highly challenging (93% of specimens lack species-level annotations; severe long-tail distribution).

**Limitations of Prior Work**: Traditional methods such as BLAST are slow and generalize poorly; supervised learning struggles with extremely sparse annotations.

**Key Insight**: Pretraining a Mamba-based (efficient SSM) foundation model to exploit unlabeled data.

**Core Idea**: SSM pretraining + hierarchical classification enhancements = a powerful tool for fungal classification under data-sparse conditions.

## Method

### Key Designs

1. **Mamba Architecture Pretraining**: Self-supervised pretraining on large-scale unlabeled/partially labeled DNA sequences.
2. **Hierarchical Label Smoothing**: Exploits the structural information of the taxonomic hierarchy (phylum / class / order / family / genus / species).
3. **Weighted Loss**: Addresses long-tail class distribution.
4. **Multi-Head Outputs**: One classification head per taxonomic rank.

## Key Experimental Results

Outperforms BLAST, RDP, and conventional supervised methods across all taxonomic ranks on the fungal classification benchmark.

## Highlights & Insights

- The **pretrain-then-finetune paradigm** is particularly effective in data-sparse genomics settings.
- The approach is extensible to DNA barcode classification of other taxonomic groups.

## Limitations & Future Work

- Validated only on the fungal ITS region.
- No comparison against protein language models (e.g., ESM).

## Rating

- Novelty: ⭐⭐⭐ Applying Mamba to DNA classification is well-motivated but not a breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive cross-rank taxonomic comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Offers practical utility for biodiversity research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction](bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)

</div>

<!-- RELATED:END -->
