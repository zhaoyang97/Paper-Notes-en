---
title: >-
  [Paper Note] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models
description: >-
  [ACL 2026][LLM Pretraining][Subcharacter compositional representation] This paper proposes SCRIPT, a model-agnostic plug-and-play module that injects subcharacter (Jamo) compositional knowledge of Hangul into the embeddi…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Subcharacter compositional representation"
  - "Korean Pre-trained Language Models"
  - "Hangul structural modeling"
  - "Morpho-phonological variation"
  - "Plug-and-play module"
date: 2026-05-08
content_hash: 347a9c3959b18779
---

# SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.12377](https://arxiv.org/abs/2604.12377)  
**Code**: [GitHub](https://github.com/SungHo3268/SCRIPT)  
**Area**: LLM Pre-training / Korean NLP  
**Keywords**: Subcharacter compositional representation, Korean Pre-trained Language Models, Hangul structural modeling, Morpho-phonological variation, Plug-and-play module

## TL;DR

This paper proposes SCRIPT, a model-agnostic plug-and-play module that injects subcharacter (Jamo) compositional knowledge of Hangul into the embedding layer of existing subword-level PLMs via a dual-channel strategy. It achieves consistent improvements in Korean NLU/NLG tasks without retraining and enables the embedding space to better capture grammatical rules and semantic variations.

## Background & Motivation

**Background**: Current mainstream Korean PLMs (including advanced LLMs such as HyperCLOVA X and EXAONE) almost entirely rely on subword tokenization. Subword modeling excels at capturing lexical semantics from large-scale corpora, but its tokenization granularity is insufficient to reflect the internal compositional structure of Hangul.

**Limitations of Prior Work**: (1) Korean is a morphologically rich agglutinative language where numerous morpho-phonological changes occur at the subcharacter (Jamo) level—corpus analysis indicates that 92.75% of morphological modifications happen at the subcharacter level; (2) existing subword PLMs are insensitive to fine-grained morphosyntactic changes (e.g., tense inflection, phonological assimilation); (3) a few Jamo-based language models (e.g., KOMBO), while robust to morphological changes, perform poorly on downstream tasks due to weak semantic representation and high computational costs.

**Key Challenge**: Subword-level modeling excels in semantics but ignores structure, while subcharacter-level modeling captures structure but sacrifices semantics—the two are complementary but difficult to reconcile.

**Goal**: Design a lightweight module to inject subcharacter compositional knowledge into existing subword-level PLMs, obtaining the advantages of both without architectural modifications or additional pre-training.

**Key Insight**: Utilize the three fundamental principles of Hangul creation (compositional rules, spatial arrangement, and sequencing rules) to guide the hierarchical compression from subcharacters to subwords, rather than using generic attention or linear pooling.

**Core Idea**: Attach a dual-channel module to the PLM's embedding layer—one channel compresses Jamo sequences into structure-aware subword representations, while the other maintains the original pre-trained embeddings, fusing both via cross-attention.

## Method

### Overall Architecture

SCRIPT is attached to the PLM's embedding layer. Given a Korean input, two parallel tokenization paths are used: (1) a subword tokenizer generates the original subword sequence for the PLM; (2) a subcharacter tokenizer generates a fine-grained Jamo sequence for SCRIPT processing. SCRIPT compresses the Jamo sequence into subword-level representations, which are fused with the original PLM embeddings before being fed into the Transformer layers.

### Key Designs

1.  **Subcharacter-to-Character Hierarchical Compression (Stage 1)**:

    - **Function**: Compress Jamo triplets (initial, medial, final) into character-level representations.
    - **Mechanism**: Follows the three principles of Hangul—first, use a GRU to encode sequential information (Principle 3: Initial → Medial → Final), then arrange spatially (Principle 2) by vertically concatenating the fused representation of Initial + Medial with the Final representation $\mathbf{h}_R = [\mathbf{h}_{I+V}; \mathbf{h}_F] \in \mathbb{R}^{2 \times N/3 \times D}$. Then, use a convolutional layer to capture relative positional information, and finally apply average pooling to obtain the character representation $\mathbf{h}_C$.
    - **Design Motivation**: Generic compression methods (attention pooling, linear pooling) lose the inductive bias of Hangul's compositional structure. Ablation experiments demonstrate that compression based on creation principles outperforms generic methods by 1.8-4.6%p.

2.  **Character-to-Subword Compression (Stage 2)**:

    - **Function**: Aggregate character representations into representations aligned with the PLM's subword granularity.
    - **Mechanism**: Re-apply a GRU to encode the compositional sequence of characters within a subword, then select the final character representation at each subword boundary as the representation for that subword: $\mathbf{h}_S = \text{Pooling}(\text{GRU}(\mathbf{h}_C)) \in \mathbb{R}^{N' \times D}$.
    - **Design Motivation**: Direct averaging or summation of character representations leads to training instability; utilizing a GRU to maintain compositional order followed by selecting boundary representations is significantly more stable.

3.  **Dual-Channel Fusion (Fusion)**:

    - **Function**: Fuse SCRIPT’s structure-aware representations with the PLM’s semantic-rich embeddings.
    - **Mechanism**: Use a cross-attention layer, where the PLM's original embedding $\mathbf{e}_S$ serves as the Query and the SCRIPT output $\mathbf{h}_S$ serves as Key/Value: $\mathbf{e}_F = \text{CrossAttn}(Q=\mathbf{e}_S, KV=\mathbf{h}_S)$.
    - **Design Motivation**: Ablations show cross-attention is superior to simple summation or concatenation because it allows the PLM embedding to dynamically and selectively absorb structural information.

### Loss & Training

SCRIPT is only trained during the fine-tuning stage using standard task-related loss functions. It requires no additional pre-training and is plug-and-play. The base models include BERT_base, KoGPT2_base, KoGPT3-1.2B, and EXAONE-2.4B.

## Key Experimental Results

### Main Results

**Korean NLU Tasks (9 benchmarks)**

| Model | KorNLI | KorSTS | NSMC | PAWS-X | KoBEST Avg. |
|------|--------|--------|------|--------|-------------|
| BERT_base | 75.85 | 76.72 | 88.96 | 72.38 | 64.22 |
| BERT_base + SCRIPT | 76.49 | 77.68 | 88.96 | 73.68 | 65.14 |
| KoGPT3-1.2B | 80.11 | 76.14 | 90.51 | 77.40 | 81.62 |
| KoGPT3-1.2B + SCRIPT | 80.39 | 79.60 | 90.53 | 79.95 | 82.17 |
| EXAONE-2.4B | 83.99 | 85.08 | 90.04 | 85.24 | 89.57 |
| EXAONE-2.4B + SCRIPT | **85.77** | **85.27** | **90.89** | **85.90** | **90.40** |

**Korean NLG Tasks (KoCommonGen)**

| Model | BLEU-4 | ROUGE-2 | ROUGE-L | METEOR |
|------|--------|---------|---------|--------|
| KoGPT2_base | 10.33 | 44.24 | 54.50 | 40.05 |
| KoGPT2_base + SCRIPT | 15.57 | 47.42 | 60.00 | 42.53 |
| EXAONE-2.4B | 28.41 | 62.25 | 64.84 | 54.84 |
| EXAONE-2.4B + SCRIPT | **31.80** | **71.03** | **72.16** | **61.27** |

### Ablation Study

**SCRIPT Architecture Ablation (KoBEST Avg., based on KoGPT2_base)**

| Configuration | Avg. Acc |
|------|----------|
| SCRIPT (Jamo + Principles + CrossAttn) | **73.85** |
| Stroke instead of Jamo | 72.36 |
| Linear compression instead of Principles | 69.30 |
| Attention compression instead of Principles | 72.01 |
| Summation instead of CrossAttn | 72.28 |
| Subword granularity (non-Jamo) | 67.92 |

### Key Findings

- SCRIPT is consistently effective across all base models and task types, proving its model-agnostic nature.
- The highest gain (>3.2%p) was observed in the Korean Grammatical Error Correction task (Kor-Learner), as this task involves numerous subcharacter-level particle and ending variations.
- The improvement in NLG tasks is more significant than in NLU tasks—likely because Hangul's sequential compositional structure is naturally compatible with autoregressive decoding.
- Embedding space analysis shows that SCRIPT increases the cosine similarity of morphologically related word pairs from 0.71 to 0.80 (+11%).

## Highlights & Insights

- The statistical analysis of the corpus, showing 92.75% of morphological changes occur at the subcharacter level, provides a strong linguistic motivation for the method design.
- The plug-and-play design is highly practical—no re-pre-training is required, allowing for immediate enhancement of any Korean PLM.
- The architecture design driven by Hangul creation principles (rather than generic attention) underscores the importance of domain knowledge.

## Limitations & Future Work

- It has currently only been validated up to a 2.4B parameter scale; its effectiveness on larger models (7B+) remains unknown.
- The design is tightly coupled with the specific features of Hangul, limiting its applicability to other languages.
- Cross-attention increases the inference cost of the embedding layer, particularly on long sequences.
- No comparison was made with the latest 7B+ Korean LLMs.

## Related Work & Insights

- **vs KOMBO**: KOMBO also models Jamo compositional structure but requires pre-training from scratch and only supports encoder architectures; SCRIPT is plug-and-play.
- **vs Multi-granularity Representations**: Prior works switch between Jamo and subword levels rather than performing structured fusion; SCRIPT achieves true fusion through a dual-channel strategy.
- **vs char2subword**: Similar modules only perform simple character-to-subword mapping without utilizing Hangul creation principles.

## Rating

- Novelty: ⭐⭐⭐⭐ Sophisticated integration of linguistics and deep learning by encoding Hangul creation principles into the neural architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 base models × 9 NLU tasks + multiple NLG tasks + detailed ablations + linguistic analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation and persuasive statistical analysis of the corpus.
- Value: ⭐⭐⭐⭐ Directly applicable value to the Korean NLP community; plug-and-play design lowers the barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
