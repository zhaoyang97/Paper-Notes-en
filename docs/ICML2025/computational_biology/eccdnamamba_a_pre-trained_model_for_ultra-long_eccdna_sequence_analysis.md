---
title: >-
  [Paper Note] eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis
description: >-
  [ICML 2025][Computational Biology][Extrachromosomal circular DNA (eccDNA)] eccDNAMamba is the first bidirectional state space encoder tailored for circular DNA. Combining BPE tokenization, circular data augmentation, and SpanBERT-style pre-training, it coordinates linear time complexity with ultra-long eccDNA sequence modeling up to 200Kbp. It significantly outperforms DNABERT-2, HyenaDNA, and Caduceus in cancer classification and genuine eccDNA identification tasks.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Extrachromosomal circular DNA (eccDNA)"
  - "State Space Models"
  - "Mamba-2"
  - "Bidirectional Encoder"
  - "Genomic Pre-training"
  - "Ultra-long Sequence Modeling"
  - "BPE Tokenization"
date: 2026-05-08
content_hash: 6af4069dbf27d5b1
---

# eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis

**Conference**: ICML 2025  
**arXiv**: [2506.18940](https://arxiv.org/abs/2506.18940)  
**Code**: [https://github.com/zzq1zh/GenAI-Lab](https://github.com/zzq1zh/GenAI-Lab)  
**Area**: Genomics / Computational Biology  
**Keywords**: Extrachromosomal circular DNA (eccDNA), State Space Models, Mamba-2, Bidirectional Encoder, Genomic Pre-training, Ultra-long Sequence Modeling, BPE Tokenization

## TL;DR

eccDNAMamba is the first bidirectional state space encoder tailored for circular DNA. Combining BPE tokenization, circular data augmentation, and SpanBERT-style pre-training, it coordinates linear time complexity with ultra-long eccDNA sequence modeling up to 200Kbp. It significantly outperforms DNABERT-2, HyenaDNA, and Caduceus in cancer classification and genuine eccDNA identification tasks.

## Background & Motivation

Extrachromosomal circular DNA (eccDNA) represents biochemically and functionally important elements widely distributed across diverse organisms, particularly prominent in cancer genomes. eccDNAs frequently carry oncogenes or distal regulatory elements, substantially impacting tumor evolution, therapeutic resistance, and intratumoral heterogeneity.

However, analyzing full-length eccDNA sequences presents two core challenges:

**Challenges in modeling circular topology**: Linearizing circular sequences at arbitrary breakpoints introduces artificial boundaries, potentially disrupting biologically meaningful head-to-tail interactions.

**Computational bottlenecks of ultra-long sequences**: Since many eccDNAs exceed 10,000 bp, the quadratic complexity of standard Transformers hinders their application to sequences of this scale.

Existing genomic foundation models exhibit notable limitations:
- **HyenaDNA**: Utilizes implicit convolutions for long sequences but is a unidirectional model and lacks circular topology awareness.
- **DNABERT-2**: Employs BPE tokenization to capture sequence motifs but retains the linear input assumption and standard Transformer layers.
- **Caduceus**: Offers linear scalability but relies on single-nucleotide resolution, which limits its scalability in downstream, ultra-long eccDNA applications.

Core Motivation: Design an efficient pre-trained model tailored for circular DNA that integrates bidirectional contextual modeling, circular structural preservation, and linear time complexity.

## Method

### Overall Architecture

The architecture of eccDNAMamba consists of the following core components:

1. **BPE Tokenizer**: Performs byte-pair encoding on DNA sequences, mapping high-frequency nucleotide patterns to tokens.
2. **Circular Data Augmentation**: Appends the first 64 tokens of a sequence to its end to preserve head-and-tail dependencies.
3. **Bidirectional Mamba-2 Encoder**: Utilizes twin parallel Mamba-2 instances (forward and backward directions), which are fused via a shared MLP.
4. **SpanMLM Pre-training Objective**: Reconstructs masked continuous spans.

### Key Designs

#### 1. Byte-Pair Encoding (BPE) Tokenization

DNA sequences are unstructured data. BPE is adopted to adaptively merge high-frequency adjacent substrings:

$(a^*, b^*) = \arg\max_{(a,b)} \text{freq}_{C_t}(a,b)$

By identifying high-frequency nucleotide patterns and encoding them as tokens, BPE enables the model to operate directly on motif-level structures, significantly reducing sequence lengths. After BPE encoding on the pre-training corpus, each token corresponds to 5.16 base pairs on average.

#### 2. Circular Data Augmentation

While eccDNAs are circular molecules, standard linear representations ignore head-to-tail dependencies. The first $s=64$ tokens are appended to the end of the sequence:

$\tilde{x} = (\texttt{[CLS]}, x_1, x_2, \ldots, x_L, x_1, x_2, \ldots, s)$

This allows the model to learn long-range dependencies between the prefix and the tail.

#### 3. Bidirectional Mamba-2 Encoding

Originally a decoder-only architecture, Mamba-2 only supports unidirectional information flow. eccDNAMamba introduces bidirectional awareness:

- **Forward encoding**: $\vec{\mathbf{h}} = \overrightarrow{\text{Mamba}}(\overrightarrow{\tilde{\mathbf{x}}})$
- **Backward encoding**: $\overleftarrow{\mathbf{h}} = \overleftarrow{\text{Mamba}}(\overleftarrow{\tilde{\mathbf{x}}})$

After aligning the outputs from both directions to the forward sequence order, they are aggregated via a shared MLP to obtain final embeddings. The embedding of the `[CLS]` token is used for downstream classification tasks.

#### 4. Span Masking Pre-training

Adopt a SpanBERT strategy, randomly selecting continuous spans (averaging about 3 consecutive tokens) to mask 15% of the sequence in total. Among the masked tokens, 80% are replaced with `[MASK]`, 10% are randomly replaced, and 10% remain unchanged. The pre-training loss is defined via span-masked cross-entropy:

$\mathcal{L}_{\text{SpanMLM}} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid x_{\backslash \mathcal{M}})$

#### 5. Padding Handling

Mamba-2 does not natively support padding. Two strategies are adopted: (1) zeroing out the embeddings of padding tokens and preventing their gradient updates; (2) applying a Transformer-style attention mask to suppress padding influences, while resetting both the hidden states and residuals at padding positions to zero in each layer.

### Pre-training Settings

- Data: 120K eccDNA sequences (approximately 100 million tokens), sourced from CircleBase (601K human eccDNAs) and eccDNA Atlas (630K animal eccDNAs). A total of 1,087,886 sequences remain after filtering out sequences >10Kbp.
- Model: Initialized based on the mamba2-130m configuration.
- Training: 3 epochs, learning rate $5 \times 10^{-4}$, linear warmup over 6% of steps, AdamW optimizer, BF16 mixed-precision, 3×NVIDIA L40S GPUs.

## Key Experimental Results

### Main Results: Cancer vs. Healthy eccDNA Classification

| Task | Model | F1 | Accuracy | Precision | Recall |
|------|------|-----|----------|-----------|--------|
| <10Kbp | **eccDNAMamba** | **0.8242** | **0.8242** | 0.8242 | **0.8242** |
| <10Kbp | DNABERT-2 | 0.8187 | 0.8187 | 0.8187 | 0.8187 |
| <10Kbp | HyenaDNA | 0.8105 | 0.8104 | 0.8105 | 0.8105 |
| <10Kbp | Caduceus | 0.8216 | 0.8220 | **0.8248** | 0.8220 |
| 10-200Kbp | **eccDNAMamba** | **0.8147** | **0.8175** | **0.8377** | **0.8174** |
| 10-200Kbp | DNABERT-2 | 0.5702 | 0.5725 | 0.5740 | 0.5725 |
| 10-200Kbp | HyenaDNA | 0.7261 | 0.7350 | 0.7699 | 0.7350 |
| 10-200Kbp | Caduceus | 0.7102 | 0.7125 | 0.7192 | 0.7125 |

**Key Findings**: On ultra-long sequences (10-200Kbp), the advantages of eccDNAMamba are even more pronounced. DNABERT-2 deteriorates page (F1 only 0.57) due to truncating sequences to 10Kbp, whereas eccDNAMamba operates with no truncation, maintaining a stable performance of 0.81+.

### Genuine eccDNA vs. Pseudo-Circular DNA

| Model | F1 | Accuracy | Precision | Recall |
|------|-----|----------|-----------|--------|
| **eccDNAMamba** | **0.7401** | **0.7407** | **0.7428** | **0.7407** |
| DeepCircle (fine-tuned) | 0.6712 | 0.6742 | 0.6808 | 0.6742 |
| DeepCircle (zero-shot) | 0.6363 | 0.6532 | 0.6883 | 0.6532 |

eccDNAMamba outperforms DeepCircle by 6.9 percentage points in F1 score, demonstrating that genuine eccDNAs indeed encode learnable, non-random sequence patterns.

### Biological Analysis

- Analysis via the MEME-Suite STREME workflow indicates that the cancer prediction decisions of the model are driven by **CG-rich sequence motifs** (characteristic of zinc-finger transcription factor binding sites).
- Tomtom alignment against the CIS-BP 2.0 database produced 568 transcription factor matches, 218 of which belong to the C2H2 zinc-finger family.
- **ZNF24** and **ZNF263** are ranked at the top; these genes are associated with cell proliferation control and oncogene regulation.
- False negative (FN) samples exhibit AT-rich patterns, suggesting the potential existence of alternative regulatory logic.

## Highlights & Insights

1. **First Circular DNA-Specific Pre-trained Model**: eccDNAMamba is the first bidirectional state space encoder designed for circular genomes, filling a major gap in the field.
2. **Extreme Data Efficiency**: Pre-trained with only 100M tokens, it outperforms foundation models trained on 35B tokens (Caduceus) and 262B tokens (DNABERT-2).
3. **Degradation-Free Long-Sequence Extension**: It is the only model that remains robust across all performance metrics when processing ultra-long sequences.
4. **Interpretable Biological Discoveries**: The dominance of CG-rich C2H2 zinc-finger motifs as sequence features yields new clues for understanding the functions of cancerous eccDNAs.

## Limitations & Future Work

1. Relies heavily on CG-rich motifs, potentially leading to blind spots (such as AT-rich false negatives).
2. Pre-training is solely sequence-based, lacking integration with external regulatory annotations or data.
3. Downstream tasks are currently restricted to classification; tasks like eccDNA origin prediction have not yet been explored.
4. The number of appended tokens (64) in the circular augmentation strategy may require fine-tuning for different sequence length intervals.

## Related Work

- **Genomic Foundation Models**: DNABERT-2, HyenaDNA, Caduceus, Nucleotide Transformer
- **eccDNA Modeling Tools**: eccDNA-Pipe (detection), DeepCircle (CNN classifier)
- **State Space Models**: Mamba, Mamba-2

## Rating

⭐⭐⭐⭐ — Highly novel; proposes a tailored solution in the under-explored field of circular DNA modeling. The experiments are solid, particularly showing impressive performance advantages on ultra-long sequences. Biological analysis adds depth to the work. However, the variety of downstream tasks is limited, and the bias toward CG-rich motifs might restrict generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)
- [\[ICLR 2026\] A Joint Diffusion Model with Pre-Trained Priors for RNA Sequence-Structure Co-Design](../../ICLR2026/computational_biology/a_joint_diffusion_model_with_pre-trained_priors_for_rna_sequence-structure_co-de.md)
- [\[AAAI 2026\] TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](../../AAAI2026/computational_biology/trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[ICML 2026\] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](../../ICML2026/computational_biology/routing_by_reaching_composition_of_pre-trained_gflownets_for_multi-objective_gen.md)

</div>

<!-- RELATED:END -->
