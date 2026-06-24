---
title: >-
  [Paper Note] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries
description: >-
  [ACL 2025][Multilingual & Machine Translation][Vocabulary Transfer] This paper proposes a cross-lingual vocabulary transfer method based on bilingual dictionaries. By exploiting the BPE tokenizer property where "removing subwords causes a fallback to shorter subwords," it maximizes the mapping coverage of target language subwords through an iterative removal-retokenization-alignment process. It significantly outperforms existing methods that rely on monolingual or parallel co…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Vocabulary Transfer"
  - "Bilingual Dictionaries"
  - "BPE"
  - "Embedding Initialization"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: d6b2e0867be4cbf9
---

# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries

**Conference**: ACL 2025  
**arXiv**: [2506.01535](https://arxiv.org/abs/2506.01535)  
**Code**: [GitHub](https://github.com/sj-h4/dict_trans_tokenizer)  
**Area**: Low-Resource Languages / Cross-Lingual Transfer  
**Keywords**: Vocabulary Transfer, Bilingual Dictionaries, BPE, Embedding Initialization, Low-Resource Languages

## TL;DR

This paper proposes a cross-lingual vocabulary transfer method based on bilingual dictionaries. By exploiting the BPE tokenizer property where "removing subwords causes a fallback to shorter subwords," it maximizes the mapping coverage of target language subwords through an iterative removal-retokenization-alignment process. It significantly outperforms existing methods that rely on monolingual or parallel corpora in low-resource language scenarios.

## Background & Motivation

Cross-lingual vocabulary transfer aims to adapt pretrained language models to new languages and is a crucial technique for addressing token over-fragmentation. Existing methods face the following limitations:

**Wechsel**: Requires a large amount of monolingual corpora and a bilingual dictionary.

**Focus**: Relies on subword overlap between the source and target languages, making it ineffective for languages with different writing systems.

**UniBridge**: Also relies on subword overlap, exhibiting limitations for cross-script languages.

**Trans-Tokenizer**: Requires parallel corpora, which are typically unavailable for low-resource languages.

**Key Insight**: For many low-resource languages, while monolingual and parallel corpora are scarce, bilingual dictionaries are often available—thanks to the outputs of descriptive linguists in language documentation efforts. Although dictionaries are small in size, they contain precise vocabulary correspondences, which can effectively establish cross-lingual subword mappings.

A key property of the BPE tokenizer is that when a subword is deleted from the vocabulary, words containing that subword roll back to a combination of shorter subwords. This work ingeniously exploits this property to progressively cover shorter subwords by iteratively removing already mapped subwords.

## Method

### Overall Architecture

The method consists of three steps: (1) training a target language BPE tokenizer using dictionary data; (2) iteratively aligning target and source language subwords; and (3) initializing the target language subword embeddings using the alignment results. Optionally, language-adaptive pre-training (LAPT) can be performed afterwards.

### Key Designs

1. **Tokenizer Training**:  
   Train a byte-level BPE tokenizer using dictionary entries (target language vocabulary). Utilizing byte-level BPE ensures no Out-of-Vocabulary (OOV) issues occur even under limited resources.

2. **Iterative Subword Mapping Algorithm (Core Innovation)**:  
   Repeat the following four steps until no new subwords are mapped:
   
    - **(1) Tokenization**: Tokenize the dictionary entries and definitions using the target language tokenizer and source model tokenizer, respectively.
    - **(2) Alignment**: Treat entry-definition pairs as parallel corpora and perform subword-level alignment using fast_align (based on IBM Model 2).
    - **(3) Mapping**: Create a type-level one-to-many mapping from target subwords to source subwords based on the alignment results, recording mapping frequencies.
    - **(4) Removal**: Delete mapped subwords and their corresponding merge rules from the target tokenizer vocabulary.
   
   **Crucial Role of the Removal Step**: By default, the BPE tokenizer only maps the longest subwords, leaving their constituent shorter subwords unmapped. Deleting the long subwords causes BPE to fallback to shorter fragments, allowing the next iteration to map these shorter subwords.

3. **Embedding Initialization**:  
   For a target subword $t$, its embedding is initialized as the weighted average of the corresponding source subword embeddings:
   
    $\boldsymbol{e}_t^T = \sum_{s \in \mathcal{M}_t} c(s|t) \cdot \boldsymbol{e}_s^S$
   
   where $c(s|t)$ is the relative frequency of subword $s$ in the mapping. Special tokens, numbers, and punctuation are copied directly from the source model. Unmapped subwords are initialized with the UNK token embedding (or randomly initialized in the case of Llama 3).

4. **Differences from Existing Methods**:  
    - Difference from Trans-Tokenizer: Uses subword-level alignment (due to short dictionary entries) instead of word-level alignment, and estimates mappings for shorter subwords via iterative removal.
    - Difference from Focus: Uses mappings trained on dictionaries, rather than subword overlap + FastText static embeddings.
    - Difference from Wechsel: Does not require large-scale monolingual corpora.

### Loss & Training

- MLM (Masked Language Modeling): Fine-tune all layers for downstream NER tasks.
- CLM (Causal Language Modeling): Train the top and bottom two layers using LoRA.
- Utilize multi-token prediction to improve training efficiency.
- LAPT uses a maximum of 3,000 samples.

## Key Experimental Results

### NER Performance — Micro F1 (Table 3)

| Model | German | Japanese | Old English | Uyghur | Sanskrit | Khmer | Manchu |
|------|------|------|--------|---------|------|--------|------|
| RoBERTa | 89.61 | 75.33 | 62.39 | 38.73 | 51.48 | 27.58 | 73.52 |
| XLM-R | 90.27 | 81.28 | 37.59 | 28.30 | 48.85 | 34.78 | 65.32 |
| Focus(XLM-R)+LAPT | 90.00 | 77.46 | 37.57 | 37.16 | 12.33 | 12.33 | 28.03 |
| Ours(RoBERTa)+LAPT | 76.43 | 73.60 | **52.71** | **64.52** | 42.08 | **62.96** | **92.87** |
| Ours(XLM-R)+LAPT | 75.98 | 74.73 | 40.94 | **59.41** | **56.99** | **58.37** | 91.39 |

### Perplexity — Llama 3.1 (Table 4, Excerpt)

| Model | German | Uyghur | Khmer | Manchu |
|------|------|---------|--------|------|
| Llama 3.1 | 655 | $2.07 \times 10^{24}$ | $\infty$ | $2.30 \times 10^{19}$ |
| Focus+LAPT | 2376 | $7.53 \times 10^{20}$ | $\infty$ | $1.29 \times 10^6$ |
| Ours | 444877 | 18053 | 64508 | 144818 |
| **Ours+LAPT** | **88.61** | **168.43** | **4.32** | **502.02** |

### Data Efficiency Comparison (Table 5)

| Language | Words Required by Focus | Words Required by Ours |
|------|-------------|--------------|
| German | 21,582,818 | 101,997 |
| Uyghur | 2,771,058 | **1,131** |
| Sanskrit | 2,812,121 | **5,282** |
| Khmer | 1,937,229 | **5,656** |
| Manchu | 397,659 | **21,620** |

### Key Findings

1. **Significant Lead on Low-Resource Languages**: On Uyghur (+27.36 F1), Khmer (+50.63 F1), and Manchu (+64.84 F1), this method vastly outperforms Focus.
2. **Extreme Data Efficiency**: Achieves better performance using less than 10% of the data volume required by Focus. Uyghur requires only 1,131 dictionary entries.
3. **No Advantage on High-Resource Languages**: Focus performs better on German and Japanese because these languages have abundant monolingual corpora and subword overlap.
4. **High Mapping Coverage**: The subword mapping rate exceeds 85% for most languages (97.27% for Sanskrit), with only Manchu being lower (77.35%).
5. **Stunning Perplexity Improvement**: Khmer drops from $\infty$ to 4.32, and Manchu drops from $10^{19}$ to 502—indicating high-quality embedding initialization.
6. **Aptitude for Specific Language Types**: Works best on languages phylogenetically distant from English that exhibit isolating or agglutinative typological features (e.g., Uyghur, Khmer, Manchu).

## Highlights & Insights

- **Clever Exploitation of BPE Properties**: The concept of "deleting subwords $\rightarrow$ falling back to shorter subwords $\rightarrow$ iterative mapping" is highly ingenious, turning a BPE "flaw" into an advantage.
- **NLP Value of Descriptive Linguistics**: Dictionary data is often overlooked by the NLP community; this work demonstrates that even very small dictionaries (e.g., only 1,131 entries for Uyghur) can yield immense value.
- **Extreme Demonstration of Data Efficiency**: Replacing millions of words of corpora with just a few thousand dictionary entries is of paramount significance for real-world, low-resource scenarios.
- **Methodological Generality**: Applicable to both MLM and CLM architectures (RoBERTa, XLM-R, Llama 3.1, Gemma 2), demonstrating the architecture-agnostic nature of the method.

## Limitations & Future Work

- Performs worse than Focus on high-resource languages, suggesting that the information capacity of dictionaries is ultimately limited and cannot completely replace large-scale corpora.
- Unmapped subwords use UNK embeddings (or are randomly initialized), which may affect languages with extremely low coverage.
- Performance is directly influenced by dictionary quality and coverage; the impact of different dictionary sources (e.g., Wiktionary vs. professional dictionaries) remains unexplored.
- Only evaluated on NER and perplexity; validation on more downstream tasks (such as MT, QA) would be more convincing.
- The data size for LAPT was capped at 3,000 samples; the impact of larger-scale LAPT is unknown.
- The handling of polysemy might not be sufficiently fine-grained (using weighted averages for one-to-many mappings).

## Related Work & Insights

- **Trans-Tokenizer (Remy et al., 2024)**: Employs word-level alignment of parallel corpora for vocabulary transfer; the dictionary-based method in this work serves as an alternative in low-resource scenarios.
- **Focus (Dobler & de Melo, 2023)**: A classic method relying on subword overlap; this work substantially outperforms it on cross-script languages.
- **Wechsel (Minixhofer et al., 2022)**: Requires a large volume of monolingual corpora and bilingual dictionaries; this work requires only a dictionary.
- **ZeTT (Minixhofer et al., 2024)**: Trains a hypernetwork for zero-shot embedding prediction, which is complementary to the dictionary-based method.
- **Insight**: NLP should not be confined solely to large-scale data paradigms; small yet precise linguistic resources can be more effective in specific scenarios.

## Rating

- **Novelty**: 9/10 — The iterative mapping strategy featuring BPE deletion-fallback is a highly original technical contribution, and utilizing dictionaries for vocabulary transfer is also novel.
- **Experimental Thoroughness**: 8/10 — Broad coverage across 7 languages, 4 model architectures, and dual metrics of NER and perplexity. However, the variety of downstream tasks could be expanded.
- **Writing Quality**: 8/10 — Clear algorithmic description, intuitive illustrations, and thorough ablation analysis.
- **Value**: 9/10 — Highly practical for the low-resource language community; the method is robust, simple, elegant, and easy to implement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)

</div>

<!-- RELATED:END -->
