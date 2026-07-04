---
title: >-
  [Paper Note] Unsupervised Morphological Tree Tokenizer
description: >-
  [ACL2025][LLM Pretraining][tokenization] TreeTok is proposed, an unsupervised neural morphological structure induction tokenizer. It learns character-level tree structures through a MorphOverriding mechanism and self-supervised objectives, performing tokenization via top-down vocabulary matching. It outperforms BPE/WordPiece on both morphological segmentation and language modeling tasks.
tags:
  - "ACL2025"
  - "LLM Pretraining"
  - "tokenization"
  - "morphology"
  - "unsupervised parsing"
  - "composition model"
  - "BPE"
date: 2026-05-08
content_hash: deb4c141013f6e4b
---

# Unsupervised Morphological Tree Tokenizer

**Conference**: ACL2025  
**arXiv**: [2406.15245](https://arxiv.org/abs/2406.15245)  
**Code**: [GitHub](https://github.com/martianmartina/TreeTokenizer)  
**Area**: LLM Pre-training  
**Keywords**: tokenization, morphology, unsupervised parsing, composition model, BPE

## TL;DR

TreeTok is proposed, an unsupervised neural morphological structure induction tokenizer. It learns character-level tree structures through a MorphOverriding mechanism and self-supervised objectives, performing tokenization via top-down vocabulary matching. It outperforms BPE/WordPiece on both morphological segmentation and language modeling tasks.

## Background & Motivation

**Background**: BPE and WordPiece are the de facto standard tokenizers in current language models, widely adopted by GPT, BERT, etc.

**Limitations of Prior Work**: These statistical tokenizers often violate word-internal morpheme boundaries, producing unnatural and fragmented tokens. For example, BPE splits "windsurfing" into "wind/sur/fing", destroying the morphological integrity of "surf" and "ing".

**Key Challenge**: Linguistic theory suggests that words, like sentences, possess internal structures. However, existing tokenizers must retain all intermediate "junk" tokens (e.g., "lo", "fing") during the bottom-up merging process, leading to vocabulary space wastage and loss of semantic information.

**Goal**: How to induce grammatically and morphologically sound character-level tree structures under unsupervised conditions, and leverage them for better tokenization?

**Key Insight**: Scaling syntactic composition models down from the word level to the character level, introducing a MorphOverriding mechanism to handle morphological indecomposability, and training with self-supervised objectives.

**Core Idea**: Jointly encoding word-internal structures and representations using a deep composition model, ensuring morpheme indecomposability via MorphOverriding, and performing tokenization through top-down vocabulary matching.

## Method

### Overall Architecture

TreeTok consists of three phases: (1) training a composition model to induce character-level tree structures; (2) building a compact vocabulary based on these tree structures; (3) performing tokenization via top-down vocabulary matching.

### Key Design 1: MorphOverriding Composition Model

- **Function**: Dynamically computing compositional representations and compatibility scores of all substrings for a given word's character sequence bottom-up to induce the optimal binary tree structure.
- **Design Motivation**: Traditional composition models always decompose subword representations into the composition of their components. However, morphemes (e.g., "wind") are minimal units of meaning and should not be more deeply decomposed. If the representation of "wind" is composed of "w" + "ind", it leads to representation entanglement with unrelated morphemes like "win".
- **Mechanism**: A heuristic morpheme vocabulary $\mathbb{V}$ is constructed (generated via BPE), with each entry associated with a learnable embedding. When a substring hits the vocabulary, its morpheme embedding $\mathbf{s}_{i,j}$ is injected into the composition function, allowing the model to learn to mix or override the compositional vector with the morpheme embedding. If it misses, an empty embedding is used. The composition function is implemented based on a multi-layer Transformer.

### Key Design 2: Dual Self-Supervised Training Objectives

- **Function**: Training the composition model by simultaneously leveraging both word-internal information and word-external contextual information.
- **Design Motivation**: Merely predicting characters (selecting from dozens of characters) is too simple to learn good structures; contextual information helps in disambiguation (e.g., "asking" in a progressive tense context tends to be "ask+ing" rather than "as+king").
- **Mechanism**:
    - **Autoencoding Loss $\mathcal{L}_{ae}$ (Word-Internal)**: Predicting all vocabulary-hitting substrings within the word (rather than just single characters), leveraging outside representations to predict masked morphemes from the context. Each substring is also assigned a constituency weight to distinguish genuine constituents from accidental overlapping substrings.
    - **Autoregressive Loss $\mathcal{L}_{ar}$ (Word-External)**: Feeding the word embeddings generated by the composition model into a causal language model (GPT-2), and performing next-word prediction using a candidate vocabulary constructed through in-batch sampling.

### Key Design 3: TreeTok Tokenization Algorithm

- **Tokenization Process**: The parse tree is traversed top-down. If a subword hits the vocabulary, it is kept and the traversal backtracks. For combinable fragments caused by parsing errors, a post-processing step uses dynamic programming to search for a merging scheme that maximizes the total probability.
- **Vocabulary Construction**: First, a heuristic vocabulary is built using a tree-constrained BPE variant (merging only adjacent token pairs with the same parent in the tree). Then, a tree-constrained Unigram variant prunes unimportant subwords based on delta entropy until the target size is reached.
- **Advantages**: Top-down matching eliminates the need to retain intermediate tokens from bottom-up merging, enabling the construction of a more compact vocabulary.

## Key Experimental Results

### Main Results

| Method | Morpho (Acc.) ↑ | Compound (Acc.) ↑ | Vocab Size |
|------|:---:|:---:|:---:|
| BPE | 19.50 | 62.98 | 30,000 |
| WordPiece | 26.20 | 62.19 | 30,000 |
| Unigram | 27.10 | 53.10 | 30,000 |
| **TreeTok** | **37.90** | **68.07** | 30,000 |

### Language Modeling and Statistical Metrics

| Method | Rényi ↑ | PPL ↓ | BLEU ↑ | Avg. Token Count |
|------|:---:|:---:|:---:|:---:|
| BPE | 44.66 | 107.76 | 26.55 | 26.58 |
| WordPiece | 44.54 | 110.97 | - | 26.60 |
| Unigram | 45.07 | 106.91 | - | 31.68 |
| **TreeTok** | **44.82** | **107.26** | **26.68** | **25.99** |

### Ablation Study

| Variant | Morpho ↑ | Compound ↑ |
|------|:---:|:---:|
| Fast R2D2 | 67.69 | 48.96 |
| Neural PCFG | 39.87 | 58.33 |
| **TreeTok (Full)** | **90.10** | **86.20** |
| w/o context | 70.00 | 63.02 |
| w/o MorphOverriding | 75.99 | 46.35 |
| w/o span loss | 89.42 | 78.39 |

### Key Findings

- TreeTok significantly outperforms BPE in morphological segmentation (+18.4 Morpho, +5.1 Compound) while maintaining the lowest average token count (25.99), implying more efficient encoding.
- Removing MorphOverriding causes a performance drop of nearly 50% on the Compound task (86.20 $\rightarrow$ 46.35), confirming that this mechanism is crucial for handling compounds.
- Removing context (autoregressive loss) reduces Morpho performance from 90.10 to 70.00, suggesting that word-external contextual information is extremely critical for disambiguating structure induction.
- In Chinese experiments, a word-boundary recall of 99.24% can be achieved without MorphOverriding, because Chinese characters correspond directly to morphemes, eliminating the issue of indecomposability.

## Highlights & Insights

1. **MorphOverriding Mechanism**: This elegantly resolves the conflict between composition models and morphological indecomposability, serving as the most critical innovation of this work. When a substring is a morpheme, its representation can decouple from its components; this linguistic insight is elegantly encoded as a learnable injection mechanism.
2. **Top-Down Tokenization + Vocabulary Pruning**: This breaks the bottom-up merging paradigm of BPE, avoiding the "junk token" problem and enabling the construction of a more compact and effective vocabulary.
3. **Cross-Lingual Validation**: Chinese experiments (which succeed without MorphOverriding) conversely validate the necessity of MorphOverriding in languages like English, strengthening the theoretical explanatory power.

## Limitations & Future Work

1. **Inference Speed**: The per-token processing time of TreeTok is approximately 80 times that of BPE (1.98e-3 vs. 2.49e-5 seconds). Although this can be mitigated by GPU batching and caching high-frequency words (hit rate 98.11%), it remains an obstacle for practical deployment.
2. **Training Cost**: The composition model requires separate training (under one day on 8×A100 GPUs for WikiText-103), which is not lightweight enough for rapid iteration scenarios.
3. **Mainly Validated on English**: Apart from Chinese structure experiments and German translation experiments, there is a lack of systematic evaluation on morphologically richer languages (e.g., Turkish, Finnish).
4. **Heuristic Vocabulary Dependency**: The morpheme vocabulary for MorphOverriding is constructed heuristically by BPE. The choice of vocabulary size significantly impacts performance (showing an inverted U-shape in Figure 3), and the optimal value requires search.
5. **Limited Downstream Improvement**: TreeTok achieves only marginal improvements over BPE in terms of PPL and BLEU. The benefits of morphological alignment have not yet been validated in large-scale language model pre-training.

## Related Work & Insights

### vs BPE / WordPiece / Unigram
Traditional statistical tokenizers build vocabularies through frequency-driven bottom-up merging (BPE/WordPiece) or top-down pruning (Unigram), completely ignoring word-internal structures. The core difference of TreeTok lies in introducing a neural-induced character-level tree structure, aligning tokenization with morphology. While BPE produces semantically irrelevant intermediate tokens, TreeTok avoids this issue through top-down matching.

### vs Morfessor
Morfessor is a classic unsupervised morphological segmentation method using MDL cost functions and hierarchical segmentation strategies. However, it cannot control the vocabulary size, making it unsuitable as a tokenizer. TreeTok combines the morphological motivation of Morfessor with the compression capability of BPE, while leveraging modern neural methods to better capture context and word-internal semantic information.

### vs Neural Syntactic Induction (R2D2 / Neural PCFG)
R2D2 and Neural PCFG are sentence-level structure induction models. TreeTok scales them down to the character level and finds that direct application yields poor results (R2D2 achieves only 48.96% on Compound), with the primary culprit being the absence of MorphOverriding. This finding itself is a significant academic contribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The MorphOverriding mechanism elegantly resolves the fundamental challenges of composition models at the character level, gracefully integrating morphological constraints into neural structure induction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Conducts comprehensive evaluations including morphological segmentation, language modeling, machine translation, Chinese validation, and detailed ablation studies, but lacks validation on more morphologically rich languages and large-scale language model pre-training.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is well-structured with a logically progressive motivation, flowing smoothly from linguistic insights to technical solutions and experimental validation.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for tokenization, although inference efficiency and compatibility with large-scale models still require follow-up validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](../../ICML2025/llm_pretraining/evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[ACL 2025\] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)

</div>

<!-- RELATED:END -->
