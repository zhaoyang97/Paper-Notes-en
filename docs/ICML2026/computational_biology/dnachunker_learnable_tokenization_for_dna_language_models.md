---
title: >-
  [Paper Note] DNAChunker: Learnable Tokenization for DNA Language Models
description: >-
  [ICML2026][Computational Biology][DNA language models] DNAChunker embeds an end-to-end learnable "dynamic chunker" into masked DNA language models. By utilizing bidirectional Mamba encoding combined with cosine-similarit…
tags:
  - "ICML2026"
  - "Computational Biology"
  - "DNA language models"
  - "learnable tokenization"
  - "adaptive chunking"
  - "Masked Language Modeling"
  - "BiMamba"
date: 2026-05-08
content_hash: 7c3a726bd254526a
---

# DNAChunker: Learnable Tokenization for DNA Language Models

**Conference**: ICML2026  
**arXiv**: [2601.03019](https://arxiv.org/abs/2601.03019)  
**Code**: Not yet public  
**Area**: Scientific Computing / Genomic Language Models  
**Keywords**: DNA language models, learnable tokenization, adaptive chunking, Masked Language Modeling, BiMamba

## TL;DR
DNAChunker embeds an end-to-end learnable "dynamic chunker" into masked DNA language models. By utilizing bidirectional Mamba encoding combined with cosine-similarity boundary prediction, it compresses base-pair sequences into variable-length chunks. Enhanced with mask protection and residual gating to prevent information leakage, the 172M-parameter model—trained solely on the human reference genome—outperforms 2.5B-scale multi-species pre-trained baselines across five genomic benchmarks.

## Background & Motivation

**Background**: DNA language models (NT, DNABERT-2, HyenaDNA, Caduceus, etc.) are migrating the "tokenize-then-encode" paradigm from NLP to genomics. Current tokenization schemes primarily fall into three categories: single-nucleotide, fixed-length k-mer, or BPE trained on large corpora.

**Limitations of Prior Work**: DNA sequences lack natural "word" boundaries, yet the aforementioned solutions rely on context-independent fixed splitting. As illustrated in Figure 1, these methods suffer from two failure modes: (1) k-mers are extremely sensitive to small-scale perturbations, where a single indel can shift the entire token sequence; (2) BPE prioritizes substring frequency, often focusing on non-functional repetitive elements and fragmenting truly meaningful functional segments like TF-binding or cis-regulatory motifs.

**Key Challenge**: There is a structural conflict between "context-independent fixed tokenization" and the "context-dependent nature of genomic functions."

**Goal**: To upgrade tokenization from a "preprocessing hyperparameter" to an "end-to-end learnable module," ensuring chunking results satisfy three criteria: (i) adaptive length to compress redundant regions; (ii) fine-grained resolution in function-rich areas; and (iii) robustness against SNVs, indels, and structural variations.

**Key Insight**: The authors observe that while dynamic chunking exists for autoregressive models (e.g., H-Net), DNA constitutes a bidirectional signal where the semantics of promoters or enhancers depend on both upstream and downstream sequences. Furthermore, the `[MASK]` in MLM training is an artificial token; if it participates in chunking or leaks through encoder residuals to the decoder, the model learns "mask-shape shortcuts" that fail to generalize to downstream mask-free data.

**Core Idea**: Encode base-pair features using bidirectional Mamba $\rightarrow$ predict hard boundaries between adjacent positions using a cosine-similarity routing network $\rightarrow$ merge similar adjacent positions into variable-length chunks for the Transformer backbone, utilizing mask protection and residual gating to block mask information leakage.

## Method

### Overall Architecture
DNAChunker is a bidirectional MLM featuring an encoder–main–decoder structure. The input is a nucleotide sequence of length $T$ (up to 8192 bp), and the output is the prediction for each masked position. The architecture performs hierarchical compression via two stages of "encoding $\rightarrow$ boundary prediction $\rightarrow$ downsampling": base-pair $T \to$ first-order chunk $T' \to$ second-order chunk $T''$. A 30-layer Transformer main network performs long-range modeling at the most compressed $T''$ length, followed by two stages of dechunking layers to upsample back to base-pair resolution for MLM prediction.

The primary design principle is to reserve the most expensive computational resources for long-range contextual reasoning in the main network: the encoder focuses on sequence compression and the decoder on representation expansion using lightweight Mamba, while only the main network employs Transformers.

### Key Designs

1.  **Bidirectional Adaptive Chunking (cosine-similarity routing + hard threshold + mask protection)**:
    - **Function**: Automatically learns "which two adjacent positions should merge into one token" in a bidirectional MLM setting.
    - **Core Idea**: Input features $\widehat{x}^{(s)}$ at stage $s$ are linearly projected to query $q^{(s)}_t$ and key $k^{(s)}_t$. The boundary probability is calculated using the cosine "dissimilarity" between adjacent positions: $p^{(s)}_t = \tfrac{1}{2}\bigl(1 - \tfrac{(q^{(s)}_t)^\top k^{(s)}_{t-1}}{\|q^{(s)}_t\|\,\|k^{(s)}_{t-1}\|}\bigr)$. This is thresholded to obtain hard boundaries $b^{(s)}_t = \mathbf{1}(p^{(s)}_t \ge 0.5)$. Base-pair representations within the same segment are aggregated into a chunk embedding, reducing length from $T$ to $T' = \sum_t b^{(0)}_t$. The mask protection mechanism forces boundaries before and after every `[MASK]`, ensuring masked nucleotides remain single-token chunks.
    - **Design Motivation**: Unlike H-Net or Byte Latent Transformer which make unidirectional decisions, cosine routing with BiMamba leverages both upstream and downstream evidence. Mask protection prevents the model from learning shortcuts based on mask shapes, ensuring chunking decisions are driven by genomic context.

2.  **30-layer Transformer Main Network + Block-level RoPE**:
    - **Function**: Performs long-range dependency modeling on the most compressed sequences, accounting for the majority of parameters and computation.
    - **Core Idea**: Uses standard Pre-LN Transformer blocks (Multi-head Self-attention + GELU FFN) with RoPE. The "center base-pair index" of each chunk serves as its position ID, allowing relative position information to be scaled by base-pairs rather than tokens, thus preserving physical coordinate semantics.
    - **Design Motivation**: By positioning the main network as the "axis of contextual reasoning," the authors use lightweight BiMamba for the encoder/decoder and expensive Transformers for the backbone. This unbalanced allocation allows the 172M model to rival the 1.2B GENERator.

3.  **Hierarchical Dechunking + Bidirectional Probability-gated Smoothing + Mask Residual Gating**:
    - **Function**: Expands $T''$ representations back to base-pair resolution, provides a differentiable path for discrete boundaries, and prevents encoder residuals from leaking ground truth at mask positions.
    - **Core Idea**: Compressed representations $z^{(s)}$ are first copied via piecewise-constant replication based on cumulative boundaries: $\tilde z^{(s+1)}_t = z^{(s)}_{\sum_{k\le t} b^{(S-s)}_k}$. Then, a pair of forward/backward linear recursions $\textsc{Scan}_\rightarrow, \textsc{Scan}_\leftarrow$ perform bidirectional smoothing using $p$ as a gate: $z^{(s+1)}_t = \tfrac{1}{2}(\textsc{Scan}_\rightarrow + \textsc{Scan}_\leftarrow)$. Residual gating only allows encoder residuals for positions where the chunk contains no masks; chunks containing masks receive zero residuals, forcing reconstruction through the main network.
    - **Design Motivation**: As $b^{(s)}_t$ is non-differentiable, $p$ provides the gradient path. Bidirectional scanning aligns with the bidirectional assumptions of MLM. Residual gating ensures the MLM loss trains the main network rather than just the encoder.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \lambda\mathcal{L}^{(0)}_{\text{ratio}} + \lambda\mathcal{L}^{(1)}_{\text{ratio}}$. The MLM term follows the BERT protocol (15% selection: 80% `[MASK]`, 10% random, 10% original), with weights for repetitive regions reduced to 0.1. A compression ratio regularization is added for each stage: $\mathcal{L}^{(s)}_{\text{ratio}} = \tfrac{\bar b^{(s)}\bar p^{(s)}}{\alpha^{(s)}} + \tfrac{(1-\bar b^{(s)})(1-\bar p^{(s)})}{1-\alpha^{(s)}}$, where $\bar b^{(s)}$ and $\bar p^{(s)}$ are average hard boundary ratios and probabilities, and $\alpha^{(s)}\in(0,1)$ is the target ratio. The pre-training corpus is the GRCh38/hg38 human reference genome. Downstream tasks use a linear classification head on mean-pooled valid tokens.

## Key Experimental Results

### Main Results
Superior performance across five benchmarks. The table below compares DNAChunker with the strongest baselines:

| Benchmark | Metric | DNAChunker (172M) | Strongest Baseline | Note |
|---|---|---|---|---|
| NT benchmark | Avg MCC ↑ / Avg Rank ↓ | **0.772** / **1.67** | GENERator (1.2B) 0.728 / 2.06 | Wins 13/18 datasets; Histone MCC 0.701 vs 0.625 |
| Revised NT benchmark | Avg MCC ↑ | **0.660** | PatchDNA 0.626; MxDNA 0.637 | Splice site +0.068 vs MxDNA |
| Genomic Benchmarks | Top-1 Acc ↑ / Avg Rank ↓ | 0.885 / 3.29 | GENERator 0.892 / 2.89 | Matches GENERator with $7\times$ fewer params |
| DNALongBench | 5 Tasks (Up to 1 Mb context) | All > Caduceus-PH | Caduceus-PH (LP) | Enhancer-target +0.061; Txn init +0.047 |
| BEND | Avg Rank ↓ | **1.9** | PatchDNA 2.1 | Variant effect AUROC 0.59 leading |

### Ablation Study
Tested on the revised NT benchmark with a 2B token budget (Linear Probing, higher is better):

| Configuration | Histone | Enhancers | Promoters | Splice | Overall MCC |
|---|---|---|---|---|---|
| 6-mer | 0.338 | 0.319 | 0.593 | 0.147 | 0.347 |
| BPE | 0.339 | 0.349 | 0.667 | 0.223 | 0.375 |
| w/o Mask Protection | 0.316 | 0.293 | 0.614 | 0.128 | 0.332 |
| w/o Residual Gating | 0.338 | 0.298 | 0.607 | 0.185 | 0.353 |
| w/o Ratio Loss | 0.341 | 0.290 | 0.635 | 0.123 | 0.348 |
| **DNAChunker (Ours)** | **0.344** | 0.346 | **0.673** | **0.290** | **0.390** |

### Key Findings
- **Security Mechanisms are Essential**: Removing mask protection causes the largest drop (Overall 0.390 $\rightarrow$ 0.332), confirming the existence of mask-shape shortcuts. Without ratio loss, splice accuracy collapses (to 0.123) as the model over-compresses and loses fine-grained signals.
- **Dynamic vs. Fixed Tokenization**: Ours (0.390 MCC) significantly outperforms BPE (0.375) and 6-mer (0.347). The jump in splice performance (0.147/0.223 to 0.290) validates that learnable chunking preserves functional motifs.
- **Scale Efficiency**: Training on GRCh38 with 172M parameters consistently beats 2.5B multi-species models, indicating gains originate from tokenization and architecture rather than just scaling data/parameters.
- **Long-range Efficiency**: Adaptive compression effectively shortens sequences. On 1 Mb DNALongBench, Ours with frozen backbone exceeds fine-tuned experts, suggesting chunk boundaries align with functional regions.

## Highlights & Insights
- Transforming "tokenization" from a static hyperparameter to a learnable module is the ideal implementation for DNA, which lacks a natural vocabulary. BPE's frequency-count approach is a mismatched prior for genomics.
- The mask protection mechanism is a critical insight for MLM: the authors identified that chunkers might use mask shapes as cues and countered this with dual barriers (forced single-token chunks + residual gating).
- The "thin-ends, thick-middle" computational allocation (BiMamba ends + Transformer core + adaptive compression) provides a scalable template for multi-modal long sequences like proteins or code.

## Limitations & Future Work
- Pre-trained only on GRCh38; cross-species generalization (bacteria, viruses) remains untested compared to autoregressive models like Evo2.
- While significantly better at splice sites than most, it still lags slightly behind GENERator (by 0.014), suggesting variable-length chunks may still have structural disadvantages for tasks requiring absolute single-nucleotide precision; task-adaptive $\alpha$ could be investigated.
- The 0.5 threshold and cosine metric are relatively simple compared to entropy gating; robustness across various $\alpha^{(s)}$ settings is only briefly discussed.

## Related Work & Insights
- **vs. Caduceus / NT-v2 (Fixed MLM)**: Under the same bidirectional MLM framework, DNAChunker outperforms them across all benchmarks, proving the bottleneck is the fixed tokenization, not MLM itself.
- **vs. DNABERT-2 / GROVER (BPE)**: Motif visualization demonstrates that BPE frequently targets non-functional repetitive sequences rather than true functional motifs.
- **vs. MxDNA / PatchDNA (Genomic Learnable Tokenization)**: These either use unidirectional patching or fail significantly on fine-grained tasks (splice site); Ours improves splice MCC from 0.740 to 0.936.
- **vs. H-Net / Byte Latent Transformer (Autoregressive Learnable Tokenization)**: DNAChunker can be viewed as the bidirectional MLM adaptation of the H-Net paradigm, specialized with mask protection for genomics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Learnable tokenization is known in autoregressive contexts; the contribution is the bidirectional adaptation + MLM-specific protection suite.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage of five benchmarks, controlled ablations, and motif visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic from motivation to architecture; Figure 1 effectively illustrates motif fragmentation and mutation robustness.
- **Value**: ⭐⭐⭐⭐⭐ Beating 2.5B multi-species models with 172M parameters redefines the efficiency-accuracy frontier for DNA language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](../../ICLR2026/computational_biology/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](../../ICLR2026/computational_biology/controlling_repetition_in_protein_language_models.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](../../AAAI2026/computational_biology/mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)

</div>

<!-- RELATED:END -->
