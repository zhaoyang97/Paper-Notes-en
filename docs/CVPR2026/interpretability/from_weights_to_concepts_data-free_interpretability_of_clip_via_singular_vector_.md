---
title: >-
  [Paper Note] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition
description: >-
  [CVPR 2026][Interpretability][CLIP interpretability] This paper proposes SITH (Semantic Inspection of Transformer Heads), a fully data-free and training-free interpretability framework for CLIP. SITH applies SVD directly…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "CLIP interpretability"
  - "singular value decomposition"
  - "attention head analysis"
  - "weight-space editing"
  - "data-free"
date: 2026-05-08
content_hash: c4ed1d40e09c49e0
---

# From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition

**Conference**: CVPR 2026
**arXiv**: [2603.24653](https://arxiv.org/abs/2603.24653)  
**Code**: [https://frangente.github.io/SITH](https://frangente.github.io/SITH)  
**Area**: Multimodal VLM / Model Interpretability
**Keywords**: CLIP interpretability, singular value decomposition, attention head analysis, weight-space editing, data-free

## TL;DR
This paper proposes SITH (Semantic Inspection of Transformer Heads), a fully data-free and training-free interpretability framework for CLIP. SITH applies SVD directly to the Value-Output weight matrices of attention heads, then leverages a novel COMP algorithm to interpret each singular vector as a sparse combination of semantically coherent concepts. This achieves finer-grained intra-head interpretability than existing methods and enables precise weight editing to improve downstream performance.

## Background & Motivation

1. **Background**: Vision-language models (VLMs) such as CLIP have been widely adopted across downstream tasks. Mechanistic interpretability seeks to understand how such models internally represent and process concepts. Existing approaches fall into two categories: (1) activation-based methods (e.g., Sparse Autoencoders) rely on dataset-computed activations; (2) TextSpan aligns attention head output activations with textual concepts but only yields coarse-grained head-level explanations.

2. **Limitations of Prior Work**: (1) Activation-based methods depend on large-scale datasets and are susceptible to data bias; (2) SAEs suffer from significant instability—dictionaries learned from different data differ substantially; (3) TextSpan only explains at the level of "this head attends to color," without distinguishing which sub-structures encode red versus green; (4) No existing method can directly interpret CLIP's internal mechanisms from weights alone, without observing any data.

3. **Key Challenge**: Existing interpretability methods either require data (and thus inherit data bias), or only provide coarse head-level explanations—there is no framework that is simultaneously data-free and fine-grained.

4. **Goal**: (1) Can CLIP attention heads be understood directly from weights, without observing any data? (2) Can such understanding reach the level of individual features within a head? (3) Can the resulting understanding enable precise model editing?

5. **Key Insight**: Building on the insight from Elhage et al.—that attention head computation can be expressed as a weighted combination of input patch transformations through the Value-Output (VO) matrix—analyzing the VO matrix reveals what features a head "reads and writes," entirely independent of input data.

6. **Core Idea**: Apply SVD to the VO matrices of CLIP attention heads, then use a semantically coherent sparse coding algorithm (COMP) to map each singular vector to a human-interpretable combination of concepts, achieving data-free, fine-grained weight-space interpretability.

## Method

### Overall Architecture
SITH operates in three steps: (1) isolate the VO matrix of each attention head, $\mathbf{W}_{VO}^{l,h} = \mathbf{W}_V^h \mathbf{W}_O^h$; (2) apply SVD, $\mathbf{W}_{VO} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, yielding right singular vectors (output directions) and corresponding singular values (importance); (3) project each singular vector into CLIP's multimodal space and apply the COMP algorithm to represent it as a sparse non-negative combination of $K$ semantically coherent concepts from a concept pool. This produces human-interpretable explanations for each principal computational direction within every attention head.

### Key Designs

1. **SVD Decomposition of the VO Matrix**

    - **Function**: Identifies the most important information flow directions within each attention head.
    - **Mechanism**: The VO matrix $\mathbf{W}_{VO}$ is a linear transformation; SVD decomposes it into reading directions ($\mathbf{u}_i$, where the head reads from), writing directions ($\mathbf{v}_i$, where the head writes to), and amplification factors ($\sigma_i$, the importance of each direction). Sorting by descending singular values reveals the head's most critical computational directions. The entire process is weight-based and requires no input data.
    - **Design Motivation**: The attention matrix $\mathbf{A}^h$ governs "which patch routes information to which patch" (routing), while the VO matrix governs "what information flows" (content). Analyzing the VO matrix thus yields an input-independent understanding.

2. **COMP (Coherent Orthogonal Matching Pursuit)**

    - **Function**: Interprets each singular vector as a sparse, semantically coherent combination of concepts.
    - **Mechanism**: Given a singular vector $\hat{\mathbf{v}}$ and a concept embedding matrix $\hat{\mathbf{\Gamma}}$, COMP seeks sparse non-negative coefficients $\mathbf{c}$ such that $\hat{\mathbf{v}} \approx \hat{\mathbf{\Gamma}}^T \mathbf{c}$. Standard NNOMP greedily selects the concept with the highest correlation but may yield semantically incoherent sets. COMP modifies the scoring function to $\text{score}(\hat{\gamma}_i) = \langle \mathbf{r}_{k-1}, \hat{\gamma}_i \rangle + \frac{\lambda}{|S_{k-1}|}\sum_{j \in S_{k-1}} \langle \hat{\gamma}_i, \hat{\gamma}_j \rangle$, where the second term encourages new concepts to be semantically similar to already-selected ones; hyperparameter $\lambda$ controls the trade-off between reconstruction fidelity and semantic coherence.
    - **Design Motivation**: Simple top-$k$ similarity selection captures only local semantics of a singular vector, while NNOMP yields good reconstruction but incoherent concepts. COMP strikes an optimal balance between the two.

3. **Weight Editing via SITH**

    - **Function**: Enables precise concept-level model intervention by adjusting singular values.
    - **Mechanism**: Using SITH's concept interpretations, an LLM evaluates the relevance of each singular vector's concept set to a downstream task; singular values of relevant directions are amplified while irrelevant ones are suppressed. No training data or gradient updates are required.
    - **Design Motivation**: Compared to TextSpan's approach of ablating entire heads, SITH operates at the granularity of individual singular vectors within a head, enabling more precise "surgical" editing.

### Loss & Training
- SITH requires no training. The COMP algorithm is a deterministic iterative procedure with two hyperparameters: number of concepts $K$ (default 5) and coherence coefficient $\lambda$ (default 0.3).
- The concept pool uses ConceptNet 5.5.
- Analysis focuses on the last 4 layers of OpenCLIP ViT-L/14 ($L=24, H=16, r=64$).

## Key Experimental Results

### Main Results — Interpretability vs. Fidelity

COMP achieves the best balance at $\lambda=0.3, K=5$:
- Interpretability (LLM score, 5-point scale): COMP ≈ 3.8, NNOMP ≈ 3.0, top-$k$ ≈ 4.2
- Reconstruction fidelity (cosine similarity): COMP ≈ 0.6, NNOMP ≈ 0.65, top-$k$ ≈ 0.35
- Replacing original singular vectors with SITH-reconstructed ones causes virtually no drop in zero-shot classification accuracy.

### Weight Editing Applications

| Task | Original OpenCLIP | TextSpan Editing | SITH Editing |
|------|-------------------|------------------|--------------|
| Waterbirds (Overall Acc) | 73.5 | 81.8 | **82.7** |
| Waterbirds (Worst-group Acc) | 47.9 | 68.0 | **70.6** |
| Flowers 102 (Zero-shot) | 76.5 | - | **77.5** |
| FGVC-Aircraft (Zero-shot) | 36.6 | - | **36.9** |
| DTD (Zero-shot) | 50.1 | - | **50.9** |

### Ablation Study — NSFW Content Suppression

| Method | Safe query → Retrieval | Unsafe query → Retrieval |
|--------|------------------------|--------------------------|
| Safe-CLIP (trained) | T→V: 69.2 | T*→V: 46.3 |
| OpenCLIP (original) | T→V: 75.1 | T*→V: 29.3 |
| **SITH (training-free)** | T→V: **74.5** | T*→V: 29.5 |

SITH suppresses NSFW concepts via weight editing without sacrificing performance on safe queries.

### Key Findings
- Individual singular vectors correspond to human-interpretable semantic concepts (e.g., "pink," "winter clothing," "ocean beach," "two objects"), validating the effectiveness of weight-space analysis.
- Fine-tuning (both full FT and LoRA) primarily re-weights existing semantic bases rather than learning entirely new features—the singular vector space remains highly stable.
- The weight change $\Delta \mathbf{W}$ induced by fine-tuning exhibits singular vectors strongly aligned with the fine-tuning task (e.g., "alpine flowers" emerges after Flowers 102 fine-tuning).
- SITH's surgical editing outperforms TextSpan's head-level ablation, as the latter may inadvertently suppress useful features co-located within the same head.

## Highlights & Insights
- **Fully data-free interpretability**: Understanding CLIP's internals without observing any images fundamentally eliminates data bias. This direction is highly valuable given the growing importance of large model transparency.
- **Elegant design of the COMP algorithm**: Incorporating a semantic coherence regularizer into the greedy selection of sparse coding is a simple yet effective idea, transforming incoherent explanations such as "apple + red" into coherent ones such as "pink red + scarlet reds + red background."
- **Closed loop from interpretability to intervention**: SITH not only explains the model but also enables precise editing based on that understanding (suppressing spurious correlations, removing NSFW content, improving classification performance), forming a complete "understand → intervene" pipeline.
- **New understanding of fine-tuning mechanisms**: Fine-tuning does not learn new features but redistributes weight over existing semantic bases—a finding with implications for understanding methods such as LoRA.

## Limitations & Future Work
- Only the right singular vectors (writing directions) of the VO matrix are analyzed; QK matrices and attention routing patterns are not examined.
- FFN layers, which also store substantial knowledge, are excluded from the analysis.
- The quality of explanations depends on concept pool coverage; ConceptNet may be insufficient for certain specialized domains.
- Weight editing improvements are consistent but modest (typically 1–2 percentage points), and may need to be combined with other methods for practical deployment.
- Validation is currently limited to CLIP ViT; applicability to decoder-only VLMs or other architectures remains to be verified.

## Related Work & Insights
- **vs. TextSpan**: TextSpan requires ImageNet-scale data to compute activations and explains at head-level granularity; SITH is fully data-free and operates at the intra-head singular vector level, offering finer granularity and immunity to data bias.
- **vs. Sparse Autoencoders (SAE)**: SAEs require training, suffer from instability across datasets, and provide sample-level explanations; SITH is a deterministic analysis that yields global, model-level explanations.
- **vs. unimodal weight analysis**: Prior SVD-based weight analyses are limited to language models and use simple nearest-neighbor search for interpretation; SITH's COMP algorithm provides more comprehensive semantic coverage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First data-free intra-head interpretability framework for CLIP.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers interpretability validation, weight editing applications, and fine-tuning analysis comprehensively.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured figures, clear concepts, and a logical progression from method to applications.
- Value: ⭐⭐⭐⭐ Significant contribution to understanding VLM internals; weight editing applications are practically motivated but yield modest improvements in magnitude.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)
- [\[ICLR 2026\] STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings](../../ICLR2026/interpretability/stride_subset-free_functional_decomposition_for_xai_in_tabular_settings.md)
- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)

</div>

<!-- RELATED:END -->
