---
title: >-
  [Paper Note] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging
description: >-
  [AAAI2026][Medical Imaging][DNA foundation model] MergeDNA is proposed to achieve context-aware dynamic DNA tokenization via differentiable Token Merging…
tags:
  - "AAAI2026"
  - "Medical Imaging"
  - "DNA foundation model"
  - "token merging"
  - "dynamic tokenization"
  - "genome modeling"
  - "masked language modeling"
date: 2026-05-08
content_hash: aabc88f0443b191a
---

# MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging

**Conference**: AAAI2026
**arXiv**: [2511.14806](https://arxiv.org/abs/2511.14806)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: DNA foundation model, token merging, dynamic tokenization, genome modeling, masked language modeling

## TL;DR
MergeDNA is proposed to achieve context-aware dynamic DNA tokenization via differentiable Token Merging, combined with a hierarchical autoencoder and adaptive masked token modeling for pretraining. With 380M parameters, it surpasses GENERator at 1.3B.

## Background & Motivation

### State of the Field

The DNA foundation model landscape is rapidly evolving: DNABERT-2 employs a BPE tokenizer; HyenaDNA/Caduceus uses SSMs for long-sequence modeling; VQDNA introduces a learnable VQ tokenizer; and GENERator scales to 1.3B parameters. These approaches each optimize along one of three axes—tokenization, long-sequence modeling, and pretraining objectives—yet lack a unified framework.

### Limitations of Prior Work

(1) The information density of DNA sequences is highly non-uniform (only ~2% are coding sequences, CDS, while the majority are non-coding regions), yet existing tokenizers (fixed k-mer / BPE) treat all regions uniformly. (2) DNA has no natural "word" boundaries—meaningful units may span 3 bases (codons), 6–10 bases (transcription factor binding sites), or longer, making fixed-granularity tokenization inherently lossy. (3) DNA sequences are extremely long (tens of thousands to millions of bases), requiring simultaneous capture of short-range motifs and long-range dependencies.

### Root Cause

Information-dense regions require fine-grained tokens to preserve detail, while repetitive or low-information-density regions should be merged to reduce computation and expand the receptive field. Fixed-granularity tokenization cannot satisfy both requirements simultaneously: globally fine-grained tokens are computationally expensive, while globally coarse-grained tokens lose detail in coding regions.

### Paper Goals

**Goal**: Design an end-to-end learnable genome modeling framework that jointly addresses dynamic tokenization and information-density-adaptive pretraining. **Key Insight**: Transfer the Token Merging (ToMe) paradigm from the ViT domain to DNA sequences, using differentiable merge operations to automatically learn context-aware token granularity. **Core Idea**: Apply local-window token merging for dynamic compression, coupled with adaptive masked token modeling (weighting mask probability by information density), to address tokenization, modeling, and pretraining objectives within a single unified framework.

## Method

### Overall Architecture
MergeDNA adopts a hierarchical autoencoder architecture comprising four modules: (1) a **Local Encoder** serving as a learnable tokenizer, which merges bases into variable-length tokens via multi-layer local-window self-attention and differentiable token merging; (2) a **Latent Encoder** employing full-attention Transformers to capture global dependencies; (3) a **Latent Decoder** that symmetrically maps back to the token space; and (4) a **Local Decoder** that recovers the original sequence length via token unmerging and performs reconstruction.

### Key Designs

1. **Local-window Token Merging**:

    - **Function**: Achieves context-aware dynamic tokenization within the Local Encoder.
    - **Mechanism**: Each layer first applies local-window self-attention (window size 16), then uses a lightweight grouping embedding to compute similarity between adjacent tokens, selecting the top-$r_l$ pairs for soft merging (weighted averaging to ensure differentiability). Multi-layer stacking progressively compresses the sequence length to $L \approx N/2$. A source matrix $\mathcal{S} \in \{0,1\}^{L \times N}$ records merge relationships for unmerging.
    - **Design Motivation**: Soft merging guarantees end-to-end differentiable training; the local window constrains merging to adjacent bases (consistent with DNA's local semantic continuity); gradual multi-layer compression is more stable than single-step compression.

2. **Adaptive Masked Token Modeling (AMTM)**:

    - **Function**: A pretraining objective weighted by information density.
    - **Mechanism**: The global token merging output of the Latent Encoder is used to identify important tokens (merge group size reflects information density). Mask sampling selects $K$ tokens proportional to their importance. Masking probability is inversely proportional to merge group size—tokens that are more important (harder to merge) are more likely to be masked.
    - **Design Motivation**: Information-dense regions (e.g., CDS, transcription factor binding sites) are less likely to be merged, resulting in smaller merge groups. AMTM ensures that pretraining focuses on these high-information-density regions.

3. **Merged Token Reconstruction (MTR)**:

    - **Function**: End-to-end reconstruction loss that drives the tokenizer to learn meaningful merge strategies.
    - **Mechanism**: The reconstruction loss is $\mathcal{L}_{MTR} = -\frac{1}{N}\sum_{i=1}^{N}\log P(\hat{X}_i | X_i; \theta)$. During training, the compression rate is sampled from a Gaussian distribution ($L \in [0.4N, 0.6N]$) to make the model robust to varying compression ratios.
    - **Design Motivation**: Randomizing the compression rate serves as a data augmentation strategy that prevents the model from overfitting to a single compression ratio.

### Loss & Training
The total loss is: $\mathcal{L}_{total} = \mathcal{L}_{MTR}(\theta) + \lambda \mathcal{L}_{MTR}(\theta \setminus \{\phi\}) + \mathcal{L}_{AMTM}(\theta)$, where $\lambda = 0.25$ and the second term freezes the tokenizer parameters to update only the decoder.

## Key Experimental Results

### Main Results

Evaluated on the GUE Benchmark (8 tasks) and NT Benchmark (18 tasks).

| Method | Params | Enhancers (3 tasks) | Species (2 tasks) | Regulatory (3 tasks) | Avg (8 tasks) |
|------|--------|:---:|:---:|:---:|:---:|
| NT-500M | 500M | 84.56% | 96.64% | 89.05% | 89.26% |
| GENERator | 1.3B | 84.87% | 96.95% | 90.30% | 90.71% |
| **MergeDNA** | **380M** | **85.11%** | 96.84% | **90.66%** | **90.87%** |

NT Benchmark (18 tasks): MergeDNA achieves an average MCC of 78.39%, surpassing MxDNA (78.14%) and all other baselines.

### Ablation Study

| Configuration | Avg MCC (8 tasks) | Note |
|------|:---:|------|
| Full MergeDNA | 90.87% | Complete model |
| w/o AMTM | 89.91% | Remove adaptive masking, −0.96% |
| w/o Token Merging | 89.12% | Fixed tokenization, −1.75% |
| Fixed compression rate (50%) | 90.23% | Remove randomization, −0.64% |

### Key Findings
- MergeDNA at 380M parameters outperforms GENERator at 1.3B, demonstrating the parameter efficiency of dynamic tokenization.
- Token merging contributes the largest gain (−1.75% when removed), confirming that context-aware dynamic tokenization is superior to fixed strategies.
- Performance is especially strong on splice site tasks (Donor: 98.93%, Acceptor: 98.67%), suggesting that the merge strategy can adaptively identify boundary information at splice sites.
- Cross-modal transfer: the model generalizes well to downstream RNA and protein tasks.

## Highlights & Insights
- **Unification across three dimensions**: This work is the first to integrate dynamic tokenization, long-sequence modeling, and adaptive pretraining objectives within a single end-to-end learnable framework.
- **Information-density adaptation**: The tokenizer automatically allocates finer-grained tokens to coding regions and merges repetitive regions, aligning well with the biological properties of DNA.
- **Parameter efficiency**: A 380M model surpasses a 1.3B model, demonstrating that "intelligent tokenization" is more effective than brute-force parameter scaling.
- The successful transfer of token merging from ViT to DNA suggests that merge strategies are generalizable to any long-sequence modality (e.g., audio, time-series signals).

## Limitations & Future Work
- Pretraining sequence length is limited to 4,096, which remains insufficient for real genome-scale sequences (millions of bases).
- The local window for token merging is fixed at 16, potentially limiting the discovery of longer motifs.
- Direct comparisons with ultra-large-scale models such as Evo2 are absent.
- Downstream tasks are predominantly classification; validation on generative tasks (e.g., sequence design) is lacking.

## Related Work & Insights
- **vs DNABERT-2 (BPE tokenizer)**: Fixed BPE ignores context and information density; MergeDNA's dynamic tokenizer achieves an average improvement of 3.5%+.
- **vs VQDNA (VQ tokenizer)**: Both employ learnable tokenizers, but VQDNA uses discrete VQ while MergeDNA uses continuous soft merging, enabling smoother gradient propagation.
- **vs HyenaDNA/Caduceus (SSM)**: MergeDNA replaces SSMs with a hierarchical Transformer, achieving a better balance between efficiency and performance.
- The adaptive masking strategy (adjusting mask probability by information density) can be generalized as a universal pretraining technique.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic application of token merging to DNA, with a complete framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks + ablations + cross-modal transfer, but lacks comparisons with ultra-large-scale models.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for tokenization in DNA foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](../../ICLR2026/medical_imaging/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)

</div>

<!-- RELATED:END -->
