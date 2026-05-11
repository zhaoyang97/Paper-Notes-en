---
title: >-
  [Paper Note] AntigenLM: Structure-Aware DNA Language Modeling for Influenza
description: >-
  [ICLR 2026][Medical Imaging][DNA language model] AntigenLM is a GPT-2-style DNA language model that preserves the integrity of genomic functional units. Pretrained on complete influenza virus whole genomes and subsequent…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "DNA language model"
  - "influenza virus prediction"
  - "functional unit encoding"
  - "whole-genome modeling"
  - "vaccine design"
date: 2026-05-08
content_hash: c56ea60a7f74e610
---

# AntigenLM: Structure-Aware DNA Language Modeling for Influenza

**Conference**: ICLR 2026
**arXiv**: [2602.09067](https://arxiv.org/abs/2602.09067)
**Code**: [https://github.com/peilab-cnic/AntigenLM](https://github.com/peilab-cnic/AntigenLM)
**Area**: Biological Sequence Generation / DNA Language Models
**Keywords**: DNA language model, influenza virus prediction, functional unit encoding, whole-genome modeling, vaccine design

## TL;DR
AntigenLM is a GPT-2-style DNA language model that preserves the integrity of genomic functional units. Pretrained on complete influenza virus whole genomes and subsequently fine-tuned, it autoregressively predicts antigenic sequences of future circulating strains, achieving significantly lower amino acid mismatch rates than the evolutionary model beth-1 and general-purpose genomic models.

## Background & Motivation
**Background**: Influenza viruses evolve rapidly to escape host immunity, necessitating frequent vaccine strain updates. Current WHO vaccine recommendations rely on phylodynamic indicators (e.g., LBI) and site-level evolutionary prediction models (e.g., beth-1).

**Limitations of Prior Work**:
   - Site-level models (beth-1) treat mutations as independent events and cannot capture co-evolution across genomic segments.
   - General-purpose genomic foundation models (DNABERT, HyenaDNA) are trained on multi-species heterogeneous corpora, losing species-specific structural information.
   - Protein-level models (ESM, ProtGPT2) entirely ignore nucleotide-level evolutionary mechanisms such as synonymous mutations, non-coding regulatory elements, and codon adaptation.

**Key Challenge**: Viral evolution is driven by coordinated multi-segment genome-wide interactions (RNA–RNA interactions, segment reassortment constraints, polymerase–antigen co-adaptation); fragmented modeling discards critical signals.

**Goal**: To construct a DNA language model that preserves functional unit integrity and captures whole-genome dependencies at the nucleotide level for accurate influenza antigen sequence prediction.

**Key Insight**: The influenza genome is compact (~13k nucleotides), making it suitable for single-Transformer whole-genome modeling. By maintaining a fixed ordering and complete boundaries of the 8 gene segments, the model can learn cross-segment co-evolutionary patterns.

**Core Idea**: Maintaining the integrity and correct arrangement of genomic functional units during pretraining enables the DNA language model to capture high-order evolutionary constraints across segments.

## Method

### Overall Architecture
AntigenLM adopts a GPT-2-style decoder-only Transformer architecture. The input is the full-genome nucleotide sequence of influenza A virus (up to 13k tokens), and the output is the autoregressively generated next nucleotide. The pipeline consists of two stages: (1) unsupervised pretraining on 54,512 complete influenza genomes; and (2) fine-tuning separately for two downstream tasks—antigen sequence prediction and subtype classification.

### Key Designs

1. **Functional-Unit–Aware Pretraining**:

    - Function: The 8 gene segments (PB2, PB1, PA, HA, NP, NA, MP, NS) are concatenated in a fixed descending-length order into a single whole-genome sequence.
    - Mechanism: Each training sample retains the complete genome; positional encodings span the full 13k positions without truncation or segmentation. A standard causal language modeling loss $\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T-1} \log p(x_{t+1} \mid x_{\leq t})$ is used for training.
    - Design Motivation: Preserving segment ordering and boundary integrity allows the Transformer's attention mechanism to model cross-segment co-evolutionary dependencies (e.g., compensatory mutations between HA and NA), which is unachievable through fragmented training.
    - Novelty: General DNA models trained on multi-species heterogeneous corpora lose species-specific structure; this work targets a single species (influenza) while preserving biological structure.

2. **Two-Stage Functional Unit Encoding Strategy**:

    - Function: Implicit positional alignment during pretraining; explicit sentinel tokens during fine-tuning.
    - Mechanism: During pretraining, fixed segment ordering combined with positional encoding implicitly encodes segment boundaries. During fine-tuning, special tokens such as `<HA>`, `<NA>`, and `<sep>` are introduced to explicitly delimit functional regions, directing attention and constraining decoding to avoid cross-segment continuation.
    - Design Motivation: Omitting explicit markers during pretraining allows the model to freely learn structural patterns, while adding markers during fine-tuning enables precise generation control—balancing generality with controllability.

3. **Temporal Prediction Fine-Tuning Scheme**:

    - Function: HA/NA sequences from three consecutive months are used to predict the antigen sequence of the following month.
    - Mechanism: The input format is $\text{block}^{(1)}\text{block}^{(2)}\text{block}^{(3)}\text{block}^{(\star)}$, where each block = `<subtype><HA>HA<NA>NA<sep>`. Training optimizes the causal LM loss over the full sequence; at inference, three historical blocks are fed as context and the future block is generated autoregressively.
    - Design Motivation: Concatenating antigen sequences from multiple time points implicitly encodes evolutionary trajectories, enabling the model to infer the next evolutionary direction from patterns of sequence change.

4. **Dual-Head Multi-Task Design**:

    - Function: Shared Transformer backbone + LM head (next-nucleotide prediction) + Classification head (subtype classification).
    - Mechanism: The LM head shares weights with the embedding matrix; the Classification head extracts hidden states at sentinel token positions and projects them to subtype logits, trained with cross-entropy loss.
    - Design Motivation: The generative task captures global evolutionary dynamics, while the classification task provides supervisory signal to improve representation quality; both tasks share the backbone and mutually reinforce each other.

### Loss & Training
- Pretraining: Standard causal LM loss; AdamW optimizer (learning rate $1 \times 10^{-4}$, linear warmup for 5% of steps + cosine decay, dropout 0.1, gradient clipping 1.0).
- Effective batch size = 32 genomes/step (8 GPUs × 1 sample × 4 gradient accumulation steps).
- Compact model scale: 6-layer Transformer, 384 hidden dimensions, 6 attention heads, FFN inner dimension 1536.

## Key Experimental Results

### Main Results — Next-Season Antigen Sequence Prediction (Japan, post-2022)

| Method | H1N1-HA AA Mismatch | H3N2-HA AA Mismatch | H1N1-NA AA Mismatch | H3N2-NA AA Mismatch |
|--------|-------------------|-------------------|-------------------|-------------------|
| WHO Current System | ~10+ | ~10+ | ~2 | ~5+ |
| beth-1 | ~6–8 | ~6–8 | ~1–2 | ~3–4 |
| LBI | High | High | — | — |
| **AntigenLM** | **~3–4** | **~3–4** | **<1** | **~1–2** |

- AntigenLM reduces mismatches by >70% relative to WHO recommendations and by ~50% relative to beth-1 on H1N1-HA and H3N2-NA.
- Next-month prediction: average of 3–4 amino acid mismatches on HA (<1% of 566 AAs) and 1–2 on NA (<0.5% of 469 AAs).

### Ablation Study — Pretraining Strategy Comparison

| Pretraining Configuration | Next-Month Token Perplexity | Sequence Generation Validity | Subtype Classification F1 |
|--------------------------|---------------------------|-----------------------------|--------------------------:|
| Full-genome (full model) | **1.26** | High | **99.81%** |
| Incomplete-genome (random cropping) | 3.55 | Low (frequently invalid sequences) | Lower; frequent subtype confusion |
| Segment-wise (single segment) | 4.42 | Moderate | Lower |
| Antigen-only (nuc) | 4.56 | Moderate | 100% (subtype determined by antigen) |
| Antigen-only (protein) | — | Moderate | 100% |

### Key Findings
- **Whole-genome context is critical**: Removing non-HA/NA internal segments raises perplexity from 1.26 to 4.56, demonstrating that segments such as PB1/PB2/PA provide meaningful predictive signal.
- **Functional unit integrity matters more than data volume**: Incomplete-genome uses the same amount of data but disrupts segment boundaries, yielding the worst performance.
- **Cross-subtype generalization**: H7N9 constitutes only 4.68% of pretraining data and 0.3% (48 sequences) of fine-tuning data, yet AntigenLM still predicts accurately.
- **Geographic generalization**: Trained exclusively on European and Asian data, AntigenLM still significantly outperforms beth-1 on completely unseen U.S. data.

## Highlights & Insights
- **The functional unit preservation principle is broadly applicable**: This strategy is not limited to influenza; any genome with well-defined functional unit structures (e.g., segmented RNA viruses) can benefit from analogous structure-aware pretraining.
- **Compact, domain-specialized models can outperform large generalist ones**: A 6-layer, 384-dimensional model specifically designed for influenza whole genomes surpasses general-purpose genomic models with far more parameters.
- **Elegance of the two-stage encoding scheme**: Withholding explicit markers during pretraining lets the model freely learn structural patterns, while introducing them during fine-tuning enables precise generation control—successfully balancing generality with controllability.

## Limitations & Future Work
- Predictions are inherently probabilistic and should serve as a complement to expert decision-making rather than a replacement.
- Validation is limited to influenza A; generalizability to other pathogens (e.g., SARS-CoV-2, HIV) remains unexplored.
- The small model scale (6 layers) may need to be expanded for more complex genomes.
- Training data depend on GISAID, introducing geographic sampling bias.

## Related Work & Insights
- **vs. beth-1**: beth-1 models site-level independent mutations, whereas AntigenLM models cross-segment co-evolution; AntigenLM outperforms beth-1 on all tasks.
- **vs. HyenaDNA/DNABERT**: General-purpose DNA models trained on multi-species corpora lose species-specific structure and frequently generate invalid sequences (large length deviations, missing markers).
- **vs. ProtGPT2**: Protein-level models cannot capture nucleotide-level evolutionary signals such as synonymous mutations.

## Rating
- Novelty: ⭐⭐⭐⭐ Functional-unit–aware pretraining is a novel design principle, though the overall architecture follows standard GPT-2.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five pretraining ablations, comparisons with three categories of methods, and cross-subtype/cross-geographic generalization evaluations—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure with well-motivated reasoning.
- Value: ⭐⭐⭐⭐ Directly applicable to vaccine design; the functional unit preservation principle has broad transferable significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](../../AAAI2026/medical_imaging/trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)
- [\[NeurIPS 2025\] Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins](../../NeurIPS2025/medical_imaging/securing_the_language_of_life_inheritable_watermarks_from_dna_language_models_to.md)
- [\[ICLR 2026\] Protein Structure Tokenization via Geometric Byte Pair Encoding](protein_structure_tokenization_via_geometric_byte_pair_encoding.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](../../AAAI2026/medical_imaging/mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)

</div>

<!-- RELATED:END -->
