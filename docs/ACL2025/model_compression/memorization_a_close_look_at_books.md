---
title: >-
  [Paper Note] Memorization: A Close Look at Books
description: >-
  [ACL 2025][Model Compression][LLM memorization] This work systematically investigates the memorization of complete books in the Llama 3 model family, demonstrating that book extraction rates strongly correlate with their popularity (a proxy for training data duplication). Furthermore, through LoRA fine-tuning, it reveals that instruction-tuning mitigates memorization via minimal weight updates concentrated in the bottom transformer blocks.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LLM memorization"
  - "training data extraction"
  - "copyright"
  - "fine-tuning extraction"
  - "LoRA weight analysis"
date: 2026-05-08
content_hash: 39190438d2bbd1b5
---

# Memorization: A Close Look at Books

**Conference**: ACL 2025  
**arXiv**: [2504.12549](https://arxiv.org/abs/2504.12549)  
**Code**: None  
**Area**: Others  
**Keywords**: LLM memorization, training data extraction, copyright, fine-tuning extraction, LoRA weight analysis

## TL;DR

This work systematically investigates the memorization of complete books in the Llama 3 model family, demonstrating that book extraction rates strongly correlate with their popularity (a proxy for training data duplication). Furthermore, through LoRA fine-tuning, it reveals that instruction-tuning mitigates memorization via minimal weight updates concentrated in the bottom transformer blocks.

## Background & Motivation

LLM's memorization capabilities raise serious privacy and copyright concerns. Although extensive research demonstrates that sensitive snippets of training data can be extracted from LLMs, several key questions remain under-explored:

**Can complete works be extracted?** Prior studies focus primarily on extracting short text sequences (e.g., phone numbers, emails), but of greater relevance to copyright litigation is whether LLMs memorize entire books.

**What determines memorization?** The quantitative relationship between training data duplication and memorization has not been studied at the level of whole copyrighted works.

**Does alignment truly eliminate memorization?** While Nasr et al. (2025) proved that alignment mitigates memorization and can be "undone" via fine-tuning, the underlying weight changes and patterns remain unclear.

The choice of books as the subject of analysis is two-fold: (a) books are at the center of multiple copyright lawsuits, and (b) they are long and unique, presenting the most technically challenging extraction target.

## Method

### Overall Architecture

The study designs three sets of experiments:
1. Baseline extraction: Evaluating book memorization extraction in Llama 3 / 3.1 pre-trained and instruct models.
2. SFT attacks: Attempting to recover instruct model memorization using the fine-tuning technique proposed by Nasr et al.
3. Scaling study + weight analysis: Large-scale analysis on 32 books and in-depth dissection of LoRA weight updates.

### Key Designs

1. **Dataset Construction**: 32 English books are collected from Project Gutenberg, controlled along two dimensions:

    - **Release date**: Distinguishing books added before and after the training cutoff date (December 2023).
    - **Popularity**: Using the Goodreads rating count as a proxy (ranging from 0 to 1M+ ratings).
    - Preprocessing: Removing the first 2,000 and last 5,000 tokens of each book (preface, legal license, etc.).

2. **Prefix Prompting Extraction Method**:

    - Prompting the model with a 500-token prefix.
    - Letting the model generate the next 30 tokens.
    - Calculating the similarity between the generation and the ground truth 30-token suffix.
    - Greedy decoding is used to ensure deterministic outputs.
    - Two modes: segment-by-segment extraction (via a sliding window) and autoregressive generation (feeding model outputs back as inputs recursively).

3. **SFT Fine-Tuning Attack**:

    - Using 43 Gutenberg books excluded from the evaluation set as fine-tuning data.
    - Employing QLoRA (LoRA rank=16) for efficient fine-tuning.
    - Fine-tuning on 500 and 1,000 samples, respectively.
    - Fine-tuning format: Placing the prefix in user content and the suffix in assistant content, training the model to "continue writing books".

4. **Weight Update Analysis**: This is the most distinct contribution. For the SFT-1000 LoRA model, the full weight updates are reconstructed:

    - Calculate $W_{update} = \alpha r^{-1} \cdot BA$
    - Calculate relative update $W_{rel} = |W_{update} \oslash W_{original}|$ (Hadamard division)
    - Analyze update distribution: Which layers and modules receive the largest relative updates.

### Evaluation Metrics

Multiple similarity metrics are utilized: Jaccard similarity, cosine similarity, Levenshtein distance, BLEU, ROUGE-L, and Sequence Matcher Similarity. Main results are reported based on median Jaccard similarity.

## Key Experimental Results

### Main Results: Baseline Autoregressive Extraction (Jaccard Similarity)

| Book | GoodReads Rating Count | Llama 3 | Llama 3.1 | Llama 3.1 Instruct |
|------|-----------------|---------|-----------|-------------------|
| Alice in Wonderland | 413,400 | **~0.95** | ~0.7 | ~0.05 |
| The Time Machine | 546,286 | ~0.6 | ~0.4 | ~0.05 |
| Peter Pan | 362,694 | ~0.3 | ~0.2 | ~0.05 |
| Ethics | 19,734 | ~0.1 | ~0.05 | ~0.05 |
| Rosin the Beau | 2 | ~0.02 | ~0.02 | ~0.02 |
| A girl and her ways* | 0 | ~0.02 | ~0.02 | ~0.05 |

*: Books added after the cutoff date

### Ablation Study: Impact of SFT on Instruct Models

| Book | Instruct Baseline | +SFT 500 | +SFT 1000 | Pretrained Baseline |
|------|--------------|----------|-----------|-----------|
| Alice | ~0.05 | ~0.7 | **~0.91** | ~1.0 |
| Time Machine | ~0.05 | ~0.3 | ~0.4 | ~0.5 |
| Peter Pan | ~0.05 | ~0.2 | ~0.3 | ~0.4 |
| Ethics | ~0.05 | ~0.1 | ~0.15 | ~0.2 |
| Rosin the Beau | ~0.02 | ~0.02 | ~0.02 | ~0.02 |

Correlation analysis extended to 32 books (Instruct SFT-1000):
- Correlation coefficient between Jaccard similarity and log(GoodReads rating count) = **0.5** (moderate-to-strong positive correlation)
- Highest extraction rates: The Communist Manifesto (0.95), Alice (0.91), Romeo and Juliet (0.76)
- Books added after the cutoff date (red dots): Low extraction rates regardless of popularity

### Key Findings in Weight Update Analysis

| Update Threshold | Ratio of Affected Weights |
|---------|--------------|
| > 1% Relative Change | ~14% |
| > 100% Relative Change | ~0.15% |

Distribution of updates in the network:
- **Concentrated in bottom transformer blocks** (the first 10-20 layers)
- Self-attention layers receive approximately **7 times** more updates compared to MLP layers
- Top layers require almost no changes to restore memorized text retrieval

### Key Findings

1. **Llama 3 can autoregressively generate the entirety of *Alice in Wonderland***: Starting with only a 500-token prefix, the generated book matches the original text with a Jaccard similarity of ~0.95.
2. **Popularity is a strong predictor of memorization**: The 0.5 correlation between Goodreads rating count and extraction rate suggests that popular books are repeated more frequently in the training corpus.
3. **Instruction tuning significantly reduces, but does not eliminate, memorization**: The extraction rate drop for instruct models is nearly 100% (to zero), but SFT on ~1000 samples easily restores most of the memorization.
4. **Memorization suppression involves very few weights**: Only ~0.15% of the weights exhibit relative updates of over 100% of their original values.
5. **The bottom layers are key**: Restoring memorization relies heavily on modifying bottom transformer blocks, particularly self-attention layers. This suggests alignment training acts as a "gating mechanism" in early layers rather than actually deleting the internalized memorization.

## Highlights & Insights

1. **Keep from snippet extraction to complete works**: Proves that LLMs do not just memorize fragments, but actually retain full books, adding substantial weight to copyright debates.
2. **Causal chain of Popularity -> Duplication -> Memorization**: Though direct training data observation is impossible, leveraging Goodreads rating count as a proxy provides compelling indirect evidence.
3. **Unique and in-depth weight analysis**: Goes beyond "what was changed" to reveal "where it was changed." The concentration of updates in bottom layers points to a new direction in LLM safety research—can memorization recovery be prevented by locking bottom layers?
4. **Clever control using cutoff date**: Books added after the cutoff date act as a natural negative control, validating that extraction success indeed reflects training data exposure.

## Limitations & Future Work

1. **Limited to Llama 3.x 70B**: Other model families (GPT, Gemini, etc.) and different parameter scales have not been evaluated.
2. **Public domain books only**: Highly controversial copyrighted works (such as *Harry Potter*) were not tested.
3. **Limitations of popularity proxy**: Goodreads ratings are effective for books but missing for other domains such as news or scientific papers.
4. **Fixed 500-token prefix length**: Shorter prefixes would represent more realistic attack scenarios but were not explored here.
5. **Lack of defense mitigation discussion**: Given bottom layers are the bottleneck, can freezing bottom layers or applying bottom-layer regularization prevent memorization recovery?

## Related Work & Insights

- **Carlini et al. (2021/2023)**: Pioneering work on LLM memorization and training data extraction, demonstrating that memorization scales with model size, data duplication, and prompt length.
- **Nasr et al. (2025)**: Introduced divergence and fine-tuning attacks. This paper extends their findings to complete books and includes detailed weight analysis.
- **Karamolegkou et al. (2023)**: Explored content popularity and memorization, but focused mainly on longest common subsequences whereas this work focuses on complete reconstruction.
- **LoRA (Hu et al., 2022) / QLoRA**: Parameter-efficient fine-tuning methods, used here to analyze "attacks" rather than task adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining complete book extraction with weight update analysis provides a fresh perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic experiments across multiple models, books, and sample counts, featuring scaled analysis of 32 books.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear experimental design, intuitive visualizations, and convincing conclusions.
- **Value**: ⭐⭐⭐⭐⭐ — Directly and significantly impacts AI safety and copyright discussions; weight analysis unlocks new paths for defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](../../ICLR2026/model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

</div>

<!-- RELATED:END -->
