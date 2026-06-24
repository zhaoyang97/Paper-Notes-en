---
title: >-
  [Paper Note] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens
description: >-
  [ACL 2025][Information Retrieval & RAG][Text Embedding] This paper reveals an intriguing phenomenon in LLM text embeddings: when mapping embedding vectors back to the vocabulary space via the decoding layer, the tokens with the highest decoding probability align highly with the keywords of the input text. Furthermore, spectral analysis reveals that this phenomenon is primarily controlled by the first principal component. Based on this, a simple training-free sparse retrieval…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Text Embedding"
  - "LLM"
  - "Token Alignment"
  - "Sparse Retrieval"
  - "Spectral Analysis"
date: 2026-05-08
content_hash: ac5af7eb30231140
---

# A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens

**Conference**: ACL 2025  
**arXiv**: [2406.17378](https://arxiv.org/abs/2406.17378)  
**Code**: [https://github.com/Arthurizijar/Text_aligns_tokens](https://github.com/Arthurizijar/Text_aligns_tokens)  
**Area**: NLP / Text Embedding  
**Keywords**: Text Embedding, LLM, Token Alignment, Sparse Retrieval, Spectral Analysis

## TL;DR

This paper reveals an intriguing phenomenon in LLM text embeddings: when mapping embedding vectors back to the vocabulary space via the decoding layer, the tokens with the highest decoding probability align highly with the keywords of the input text. Furthermore, spectral analysis reveals that this phenomenon is primarily controlled by the first principal component. Based on this, a simple training-free sparse retrieval method is proposed, preserving over 80% of the original dense retrieval performance.

## Background & Motivation

LLM-based text embedding methods have achieved excellent performance in information retrieval and semantic similarity tasks. Existing methods typically divide the LLM into two parts: a feature extraction module $f$ and a decoding layer $g$. They transform $f$ into an embedder $\hat{f}$ (via prompt engineering or contrastive learning), discard $g$, and apply a pooling strategy to obtain the text embeddings.

The authors discover an **intriguing phenomenon**: when re-feeding the text embedding obtained from $\hat{f}$ into $g$ (the decoding layer of the original LLM), **the tokens decoded with the highest probability are highly semantically correlated with the keywords of the input text**. For example, when the input is "What diseases are parrots prone to?", semantically related tokens such as "disease", "birds", and "suscept" are decoded.

This phenomenon is **universal** and is not affected by model architectures (GPT-Neo, OPT, LLaMA, Mistral), training strategies (prompt engineering, contrastive learning), or pooling methods (last pooling, mean pooling).

## Method

### Overall Architecture

**Analysis Framework**: Given a text $s_i$, the literal token set $T_{s_i}$ is obtained via a tokenizer, and the text embedding $\mathbf{h}_i$ is obtained via the embedder $\hat{f}$. By taking the dot product of $\mathbf{h}_i$ with the decoding layer's token embedding matrix $\mathbf{E}_g$, the aligned token set $\hat{T}_{s_i}^K$ is obtained by sorting the scores in descending order.

### Key Designs

**1. Qualitative and Quantitative Analysis of Alignment**

- Comprehensive analysis of 8 LLM embedders, including SGPT, OPT-EOL, LLaMA-EOL, GritLM, and LLM2Vec.
- Three quantitative metrics:
    - **Hit@K**: Whether the top-K aligned tokens contain any token from the input text (hit rate).
    - **LAR (Local Alignment Rate)**: The overlap ratio between the top-$|T_{s_i}|$ aligned tokens and the input tokens.
    - **GAR (Global Alignment Rate)**: The global overlap ratio at the dataset level.
- Key findings:
    - The original $f$ can also align with certain tokens, but mostly meaningless words ("and", "the").
    - The aligned tokens from $\hat{f}$ are more **meaningful** ("game", "November") and more **diverse** (higher GAR).
    - Contrastive learning further enhances the diversity and significance of the alignment.

**2. Explanation via Spectral Analysis**

Analyzing the difference between the embedding spaces of $f$ and $\hat{f}$ by obtaining principal components through SVD of the text embedding matrix:

$$v_j = \mathbb{E}_{s_i \in D}\left[(\hat{\mathbf{h}}_i - \mathbf{h}_i)^\top \mathbf{u}_j\right]$$

Two core observations:
- **Observation 1**: The change of $\hat{f}$ relative to $f$ is **concentrated in the first principal component** $v_1$.
- **Observation 2**: The first principal component mainly contributes to **meaningless tokens** (e.g., "the", "and") rather than key tokens.

From this, the hypothesis is proposed: **the embeddings of the original LLM are actually already aligned with key tokens, but this is obscured by the influence of the first principal component**.

Validation method: Manually adjusting the embedding $\mathbf{h}_i^{adj} = \mathbf{h}_i + \lambda \mathbf{u}_1$. When $\lambda \approx v_1$, the embeddings from $f$ can also align with meaningful key tokens $\rightarrow$ the hypothesis holds.

**3. Sparse Retrieval Application**

Based on the alignment discovery, a training-free sparse retrieval approach is proposed:
- Document side: Retain the top-K aligned tokens and their scores to construct a sparse vector (with only K non-zero dimensions).
- Query side: Literal token set + top-M aligned tokens as expansion.
- Similarity: The sum of intersection weights between the expanded query token set and the document sparse vector.

**4. Interpretability Applications**

- **Instruction-Following interpretation**: The same text aligns with different tokens under different instructions (sentiment classification instruction $\rightarrow$ aligns with sentiment words; no instruction $\rightarrow$ aligns with topic words).
- **Semantic similarity vs. semantic relatedness interpretation**: Models trained on NLI data (SGPTnli) tend to align with "dislike"-like tokens $\rightarrow$ low similarity; models trained on MSMARCO (SGPTmsmarco) balanced-align with "dislike"+"apple" $\rightarrow$ high relatedness.

### Loss & Training

This paper focuses primarily on analysis and does not involve new training. The sparse retrieval method is entirely training-free.

Training methods utilized by each embedder include:
- Prompt Engineering (PromptEOL)
- Contrastive Learning (InfoNCE loss) + LoRA fine-tuning
- Hybrid Training (Contrastive Learning + Next Token Prediction, e.g., GritLM)

## Key Experimental Results

### Main Results

**Sparse Retrieval vs. Dense Retrieval (nDCG@10)**:

| Method | FiQA | NFCorpus | SciFact | ArguAna |
|------|------|----------|---------|---------|
| BM25 | 0.236 | 0.325 | 0.665 | 0.315 |
| SPLADEv2 | 0.336 | 0.334 | 0.693 | 0.479 |
| LLM2Vec (Dense) | 0.531 | 0.393 | 0.789 | 0.575 |
| LLM2Vec → Sparse | 0.404 | 0.326 | 0.669 | 0.481 |
| GritLM (Dense) | 0.600 | 0.409 | 0.792 | 0.632 |
| GritLM → Sparse | 0.457 | 0.336 | 0.703 | 0.526 |

The sparse method retains **~80%** of the dense retrieval performance, while significantly outperforming BM25 and SPLADEv2, with inference FLOPs being only about **13%** of the dense method.

### Ablation Study

**Comparison of alignment metrics (8 embedders vs. corresponding original LLMs)**:

- All $\hat{f}$'s Hit@10 scores are close to 100% (at least one input token is in the top-10 aligned results).
- The GAR of $\hat{f}$ trained with contrastive learning is significantly higher than that of prompt-engineering-trained ones.
- Instruction-Following experiments confirm that aligned tokens change dynamically with instructions.

### Key Findings

1. Alignment between text embeddings and key tokens is a **universal phenomenon** of LLM embedders across various architectures and training strategies.
2. The main change in the embedding space is concentrated in the first principal component — **reducing the first principal component allows the original LLM to also demonstrate alignment behavior**.
3. Contrastive learning methods produce more diverse and meaningful aligned tokens than prompt engineering methods.
4. GritLM (trained simultaneously with contrastive learning + NTP) is the only model where the first principal component increases, behaving differently from other methods.

## Highlights & Insights

1. **Discovery-driven** study: Initiated by finding an interesting phenomenon, followed by root-cause analysis, and finally showcasing practical applications, presenting a complete logical chain.
2. The **first principal component hypothesis** is highly insightful — different embedding improvement methods (PE, CL, hybrid training) are essentially doing the same thing: adjusting the first principal component.
3. **Intuitive explanation for Instruction-Following**: Different instructions guide LLMs to align embeddings with different key tokens, providing a visualized understanding for the first time.
4. Although the sparse retrieval method is simple, it demonstrates the practical value of the discovered rule.

## Limitations & Future Work

1. The sparse retrieval method currently retains only ~80% performance, which could probably be further improved by learning optimal $K$ and $M$.
2. The analysis is based on Wikipedia data; alignment behaviors in other domains (code, mathematics) still need to be validated.
3. The adjustment of the first principal component is global, whereas different texts might require different levels of adjustment.
4. There is currently a lack of quantitative definitions for "meaningful" alignment (tokens marked in red were judged irrelevant, but might have deep semantic connections).

## Related Work & Insights

- **Geva et al. (2022)**: Explains the update behavior of FFNs by multiplying FFN value vectors by the token embedding matrix $\rightarrow$ This paper applies this idea to overall text embeddings.
- **PromptEOL / CSE**: LLM embedding methods based on prompt engineering and contrastive learning $\rightarrow$ This paper provides a unified perspective of understanding for them.
- **SPLADE**: BERT-based sparse retrieval method $\rightarrow$ The LLM sparse retrieval in this paper can be viewed as an LLM version of SPLADE.
- **Dar et al. (2022)**: Analyzing the token space projection of parameters in pre-trained Transformers $\rightarrow$ Similar analysis methodology.

## Rating

- **Novelty**: ★★★★★ — The discovery itself is highly inspiring, providing a brand-new perspective for the LLM embedding field.
- **Value**: ★★★★☆ — The sparse retrieval application has practical value, though the main contribution lies in the analytical findings.
- **Experimental Thoroughness**: ★★★★☆ — 8 embedders, multiple sets of qualitative and quantitative analyses, and 4 IR datasets.
- **Writing Quality**: ★★★★★ — Flowing narrative, progressing step-by-step from discovery to explanation to application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)

</div>

<!-- RELATED:END -->
