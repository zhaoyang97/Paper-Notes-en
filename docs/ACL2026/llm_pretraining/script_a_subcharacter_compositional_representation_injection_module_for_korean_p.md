---
title: >-
  [Paper Note] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models
description: >-
  [ACL 2026][LLM Pretraining][Subcharacter compositional representation] This paper proposes SCRIPT, a model-agnostic plug-and-play module that injects subcharacter (Jamo) compositional knowledge from the Korean Hangul wri…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Subcharacter compositional representation"
  - "Korean pre-trained language models"
  - "Hangul structure modeling"
  - "Morphophonological alternation"
  - "Plug-and-play module"
date: 2026-05-08
content_hash: 7fb39bb5d61a476d
---

# SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models

**Conference**: ACL 2026
**arXiv**: [2604.12377](https://arxiv.org/abs/2604.12377)  
**Code**: [GitHub](https://github.com/SungHo3268/SCRIPT)  
**Area**: LLM Pre-training / Korean NLP
**Keywords**: Subcharacter compositional representation, Korean pre-trained language models, Hangul structure modeling, Morphophonological alternation, Plug-and-play module

## TL;DR

This paper proposes SCRIPT, a model-agnostic plug-and-play module that injects subcharacter (Jamo) compositional knowledge from the Korean Hangul writing system into the embedding layer of existing subword-level PLMs via a dual-channel strategy. Without requiring re-pretraining, SCRIPT yields consistent improvements on Korean NLU/NLG tasks and enables the embedding space to better capture morphosyntactic regularities and semantic variations.

## Background & Motivation

**Background**: Mainstream Korean PLMs—including advanced LLMs such as HyperCLOVA X and EXAONE—almost universally rely on subword tokenization. While subword modeling effectively captures lexical semantics from large corpora, its tokenization granularity is insufficient to reflect the internal compositional structure of Hangul.

**Limitations of Prior Work**: (1) Korean is a morphologically rich agglutinative language in which the vast majority of morphophonological alternations occur at the subcharacter (Jamo) level—corpus analysis reveals that 92.75% of morphological modifications are subcharacter-level; (2) existing subword PLMs are insensitive to fine-grained morphosyntactic variations such as tense inflection and phonological assimilation; (3) the few Jamo-based language models (e.g., KOMBO), while robust to morphological alternations, perform poorly on downstream tasks due to weak semantic representations and high computational cost.

**Key Challenge**: Subword-level modeling excels at semantics but overlooks structure, whereas subcharacter-level modeling captures structure at the expense of semantics—the two are complementary yet difficult to combine.

**Goal**: Design a lightweight module that injects subcharacter compositional knowledge into existing subword-level PLMs, achieving the advantages of both paradigms without modifying the model architecture or requiring additional pretraining.

**Key Insight**: The three foundational principles governing Hangul character construction (combinatorial rules, spatial arrangement, and sequential ordering) are used to guide hierarchical compression from subcharacters to subwords, rather than relying on generic attention or linear pooling.

**Core Idea**: Attach a dual-channel module to the embedding layer of a PLM—one channel compresses the subcharacter sequence into a structure-aware subword representation, while the other preserves the original pretrained embeddings; the two are fused via cross-attention.

## Method

### Overall Architecture

SCRIPT is attached to the embedding layer of a PLM. Given a Korean input, two parallel tokenization paths are employed: (1) a subword tokenizer produces the original subword sequence for the PLM; (2) a subcharacter tokenizer generates a fine-grained Jamo sequence for processing by SCRIPT. SCRIPT compresses the Jamo sequence into subword-level representations, which are then fused with the PLM's original embeddings and fed into the Transformer layers.

### Key Designs

1. **Hierarchical Subcharacter-to-Character Compression (Stage 1)**:

    - **Function**: Compress Jamo triplets (onset, nucleus, coda) into character-level representations.
    - **Mechanism**: The three Hangul construction principles are followed—a GRU first encodes sequential ordering information (Principle 3: onset → nucleus → coda); the fused onset+nucleus representation is then vertically concatenated with the coda representation according to spatial arrangement (Principle 2): $\mathbf{h}_R = [\mathbf{h}_{I+V}; \mathbf{h}_F] \in \mathbb{R}^{2 \times N/3 \times D}$; a convolutional layer captures relative positional information; and average pooling yields the final character representation $\mathbf{h}_C$.
    - **Design Motivation**: Generic compression methods (attention pooling, linear pooling) discard the compositional structural priors of Hangul. Ablation experiments confirm that principle-guided compression outperforms generic alternatives by 1.8–4.6 percentage points.

2. **Character-to-Subword Compression (Stage 2)**:

    - **Function**: Aggregate character representations into subword-level representations aligned with PLM tokenization granularity.
    - **Mechanism**: A GRU is applied again to encode the compositional ordering of characters within each subword, after which the representation of the last character at each subword boundary is selected as the subword representation: $\mathbf{h}_S = \text{Pooling}(\text{GRU}(\mathbf{h}_C)) \in \mathbb{R}^{N' \times D}$.
    - **Design Motivation**: Directly averaging or summing character representations leads to training instability; maintaining compositional order via GRU before selecting boundary representations yields more stable optimization.

3. **Dual-Channel Fusion**:

    - **Function**: Fuse SCRIPT's structure-aware representations with the PLM's semantically rich embeddings.
    - **Mechanism**: A cross-attention layer is employed, using the PLM's original embeddings $\mathbf{e}_S$ as Query and SCRIPT's output $\mathbf{h}_S$ as Key/Value: $\mathbf{e}_F = \text{CrossAttn}(Q=\mathbf{e}_S, KV=\mathbf{h}_S)$.
    - **Design Motivation**: Ablation studies show that cross-attention outperforms simple summation or concatenation, as it allows PLM embeddings to dynamically and selectively absorb structural information.

### Loss & Training

SCRIPT is attached and trained only during fine-tuning, using standard task-specific loss functions. No additional pretraining is required; the module is fully plug-and-play. Base models include BERT_base, KoGPT2_base, KoGPT3-1.2B, and EXAONE-2.4B.

## Key Experimental Results

### Main Results

**Korean NLU Tasks (9 Benchmarks)**

| Model | KorNLI | KorSTS | NSMC | PAWS-X | KoBEST Avg. |
|-------|--------|--------|------|--------|-------------|
| BERT_base | 75.85 | 76.72 | 88.96 | 72.38 | 64.22 |
| BERT_base + SCRIPT | 76.49 | 77.68 | 88.96 | 73.68 | 65.14 |
| KoGPT3-1.2B | 80.11 | 76.14 | 90.51 | 77.40 | 81.62 |
| KoGPT3-1.2B + SCRIPT | 80.39 | 79.60 | 90.53 | 79.95 | 82.17 |
| EXAONE-2.4B | 83.99 | 85.08 | 90.04 | 85.24 | 89.57 |
| EXAONE-2.4B + SCRIPT | **85.77** | **85.27** | **90.89** | **85.90** | **90.40** |

**Korean NLG Task (KoCommonGen)**

| Model | BLEU-4 | ROUGE-2 | ROUGE-L | METEOR |
|-------|--------|---------|---------|--------|
| KoGPT2_base | 10.33 | 44.24 | 54.50 | 40.05 |
| KoGPT2_base + SCRIPT | 15.57 | 47.42 | 60.00 | 42.53 |
| EXAONE-2.4B | 28.41 | 62.25 | 64.84 | 54.84 |
| EXAONE-2.4B + SCRIPT | **31.80** | **71.03** | **72.16** | **61.27** |

### Ablation Study

**SCRIPT Architecture Ablation (KoBEST Average, based on KoGPT2_base)**

| Configuration | Average Accuracy |
|---------------|-----------------|
| SCRIPT (Jamo + Principles + CrossAttn) | **73.85** |
| Stroke substituted for Jamo | 72.36 |
| Linear compression substituted for Principles | 69.30 |
| Attention compression substituted for Principles | 72.01 |
| Summation substituted for CrossAttn | 72.28 |
| Subword granularity (non-Jamo) | 67.92 |

### Key Findings

- SCRIPT consistently improves performance across all base models and all task types, demonstrating model-agnosticism.
- The Korean grammatical error correction task (Kor-Learner) shows the largest gains (>3.2 percentage points), as it involves extensive subcharacter-level particle and ending alternations.
- Improvements on NLG tasks are more pronounced than on NLU tasks—likely because Hangul's sequential compositional structure is naturally aligned with autoregressive decoding.
- Embedding space analysis shows that SCRIPT increases the cosine similarity of morphologically related word pairs from 0.71 to 0.80 (+11%).

## Highlights & Insights

- The corpus-level finding that 92.75% of morphological changes occur at the subcharacter level provides compelling linguistic motivation for the proposed design.
- The plug-and-play design offers substantial practical value—it requires no re-pretraining and can instantly augment any Korean PLM.
- The architecture driven by Hangul construction principles, rather than generic attention mechanisms, underscores the importance of incorporating domain-specific linguistic knowledge.

## Limitations & Future Work

- Effectiveness has only been validated at model scales up to 2.4B parameters; performance on larger models (7B+) remains unknown.
- The design is tightly coupled to the structural properties of Korean Hangul, limiting its applicability to other languages.
- The cross-attention mechanism increases inference cost at the embedding layer, particularly for long sequences.
- Comparisons with the latest 7B+ Korean LLMs have not been conducted.

## Related Work & Insights

- **vs. KOMBO**: KOMBO also models Jamo compositional structure but requires pretraining from scratch and supports only encoder architectures; SCRIPT is plug-and-play.
- **vs. Multi-granularity Representations**: Prior work switches between Jamo and subword granularities rather than performing structured fusion; SCRIPT achieves true integration via a dual-channel strategy.
- **vs. char2subword**: Similar modules perform only simple character-to-subword mapping without leveraging Hangul construction principles.

## Rating

- Novelty: ⭐⭐⭐⭐ — Encoding Hangul construction principles into neural network architecture represents an elegant integration of linguistics and deep learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four base models × 9 NLU benchmarks + multiple NLG tasks + detailed ablations + linguistic analysis.
- Writing Quality: ⭐⭐⭐⭐ — The motivation chain is clear and the corpus statistical analysis is convincing.
- Value: ⭐⭐⭐⭐ — Provides direct practical value to the Korean NLP community; the plug-and-play design lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
