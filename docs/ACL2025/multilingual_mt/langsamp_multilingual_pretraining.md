---
title: >-
  [Paper Note] LangSAMP: Language-Script Aware Multilingual Pretraining
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual pretraining] The LangSAMP method is proposed, which adds language and script embeddings to the output end (rather than the input end) of the Transformer during multilingual pretraining. This enables the model backbone to learn more language-neutral representations, consistently outperforming the baseline in zero-shot cross-lingual transfer across over 500 languages.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual pretraining"
  - "language embedding"
  - "script embedding"
  - "crosslingual transfer"
  - "language neutrality"
date: 2026-05-08
content_hash: 5306158c1b7d909a
---

# LangSAMP: Language-Script Aware Multilingual Pretraining

**Conference**: ACL 2025  
**arXiv**: [2409.18199](https://arxiv.org/abs/2409.18199)  
**Code**: [GitHub](https://github.com/cisnlp/LangSAMP)  
**Area**: Multilingual Translation  
**Keywords**: multilingual pretraining, language embedding, script embedding, crosslingual transfer, language neutrality  

## TL;DR
The LangSAMP method is proposed, which adds language and script embeddings to the output end (rather than the input end) of the Transformer during multilingual pretraining. This enables the model backbone to learn more language-neutral representations, consistently outperforming the baseline in zero-shot cross-lingual transfer across over 500 languages.

## Background & Motivation
**Background**: Multilingual pretrained models (such as XLM-R, mBERT) serve as universal text encoders, supporting zero-shot cross-lingual transfer. Early models (XLM) used language embeddings, but recent models have abandoned them.

**Limitations of Prior Work**: Without language embeddings, token representations must encode both language/script-specific information and semantic information simultaneously. This makes representations less language-neutral, hindering cross-lingual transfer performance.

**Key Challenge**: Utilizing language embeddings makes the model dependent on language ID inputs (which is inconvenient during inference); omitting them compromises the language neutrality of the representations.

**Goal**: How to improve the quality of multilingual representations using language and script embeddings without sacrificing inference flexibility?

**Key Insight**: Place language and script embeddings at the output end rather than the input end of the Transformer. They are used to assist MLM decoding during pretraining, but are completely unnecessary during fine-tuning, allowing the model backbone to function as a universal encoder as usual.

**Core Idea**: Adding language/script embeddings at the output end offloads the decoding burden, enabling the Transformer backbone to learn more language-neutral representations.

## Method

### Overall Architecture
Continuous pretraining is conducted on top of XLM-R, using the Glot500-c corpus (500+ languages, 30 script systems). The input passes through Transformer blocks to obtain the hidden representation $\boldsymbol{h}_i$, which is then added to the corresponding language embedding $\boldsymbol{E}^{Lang}_l$ and script embedding $\boldsymbol{E}^{Script}_s$, forming $\boldsymbol{o}_i = \boldsymbol{h}_i + \boldsymbol{E}^{Lang}_l + \boldsymbol{E}^{Script}_s$. This is then fed into the MLM head to predict masked tokens. During fine-tuning, $\boldsymbol{h}_i$ is used directly for downstream tasks without the need for language or script IDs.

### Key Designs
1. **Output-side Embedding (Core Innovation)**:

    - **Function**: Adds language and script embeddings to the Transformer's output end rather than the input end.
    - **Mechanism**: $\boldsymbol{o}_i = \boldsymbol{h}_i + \boldsymbol{E}^{Lang}_l + \boldsymbol{E}^{Script}_s$, providing language/script "hints" to assist MLM decoding.
    - **Design Motivation**: Conventional methods (like XLM) inject language embeddings at the input layer, requiring language IDs during downstream tasks. Placing them at the output layer confines this requirement to pretraining, enabling the backbone to function independently during fine-tuning.

2. **Dual Language-Script Embeddings**:

    - **Function**: Maintains both language embeddings ($\mathbb{R}^{610 \times 768}$) and script embeddings ($\mathbb{R}^{30 \times 768}$).
    - **Mechanism**: Languages capture lexical and grammatical differences, whereas scripts capture typographic and encoding differences, complementing each other.
    - **Design Motivation**: Ablation studies show that key combinations yield the best results; using language embeddings alone helps retrieval tasks more, while using script embeddings alone is more beneficial for sequence labeling.

3. **By-product: Language/Script Embeddings for Source Language Selection**:

    - **Function**: Reflects typological features of languages via similarity between learned language embeddings.
    - **Mechanism**: Uses similarity in language embeddings to select source languages for cross-lingual transfer.
    - **Design Motivation**: Traditional methods rely on linguistic prior knowledge to select source languages. Embedding similarity offers a data-driven alternative.

## Key Experimental Results

### Main Results (Zero-Shot Cross-lingual Transfer, Table 1)

| Task | LangSAMP (all) | Baseline (all) | Gain |
|------|----------------|-----------------|------|
| SR-B (Sentence Retrieval) | 45.1 | 42.9 | +2.2 |
| SR-T (Tatoeba) | 71.1 | 69.7 | +1.4 |
| Taxi1500 (Classification) | 53.4 | 50.3 | +3.1 |
| SIB200 (Classification) | 75.9 | 75.0 | +0.9 |
| NER | 62.6 | 62.2 | +0.4 |
| POS | 71.6 | 71.5 | +0.1 |

### Ablation Study

| Configuration | SR-B (all) | Taxi1500 (all) |
|------|------------|----------------|
| Vanilla | 23.2 | 28.4 |
| + Lang only | 24.5 | 28.5 |
| + Script only | 23.9 | 28.2 |
| + Lang + Script | 24.9 | 30.3 |

### Key Findings
- Low-resource tail languages benefit more: on SR-B, tail improvement is 2.6% vs. 0.7% for head languages.
- Non-Latin script languages show larger improvements: on SR-B, non-Latin +2.3% vs. Latin +2.1%.
- Language embeddings capture language family relations—languages from the same family naturally cluster in UMAP visualizations.
- Script embedding distances align with Unicode blocks.
- Language neutrality is improved: cross-lingual pairwise cosine similarity increases.

## Highlights & Insights
- The output-side embedding design is highly elegant—used during pretraining but omitted during fine-tuning, incurring zero additional inference cost while preserving the language-ID-free nature of universal encoders.
- Using the resulting language/script embeddings for source language selection is a practical, "two-birds-one-stone" design.
- Evaluation covering 500+ languages is highly comprehensive, making a direct contribution to low-resource language research.

## Limitations & Future Work
- Evaluated only on an encoder-only model (XLM-R) and not yet extended to decoder-only or encoder-decoder architectures.
- Requires reliable language ID detection as pretraining inputs (incorrect language labels might introduce noise).
- Absolute performance gains are relatively small on sequence labeling tasks like NER/POS.
- Computational cost: requires continuous pretraining on the complete Glot500-c corpus (4 weeks on 4x RTX6000 GPUs).

## Related Work & Insights
- **vs XLM**: XLM injects language embeddings at the input layer, requiring language IDs during fine-tuning; LangSAMP only requires them during pretraining.
- **vs XLM-R/mBERT**: Completely omitting language embeddings limits the language neutrality of representations.
- **vs Adapter methods**: Adapters add parameters during fine-tuning, whereas LangSAMP introduces zero additional parameters during inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The output-side embedding idea is simple yet effective, with an elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 500+ languages, 6 tasks, along with thorough ablations and analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and the method is concise.
- **Value**: ⭐⭐⭐⭐ Directly contributes to the multilingual NLP community, especially for low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](lexgen_domain-aware_multilingual_lexicon_generation.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](../../ACL2026/multilingual_mt/multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)

</div>

<!-- RELATED:END -->
