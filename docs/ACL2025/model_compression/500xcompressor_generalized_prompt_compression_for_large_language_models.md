---
title: >-
  [Paper Note] 500xCompressor: Generalized Prompt Compression for Large Language Models
description: >-
  [ACL 2025][Model Compression][Prompt Compression] The paper proposes 500xCompressor, which compresses up to around 500 natural language tokens into the KV values of as few as 1 special token, achieving compression ratios from 6x to 480x. It introduces only about 0.25% of additional parameters, while the LLM retains 62.26%–72.89% of its original capabilities after compression, significantly outperforming the ICAE baseline.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Prompt Compression"
  - "Soft Prompt"
  - "KV Cache"
  - "High Compression Ratio"
  - "Autoencoder"
date: 2026-05-08
content_hash: 4baf09c43446c804
---

# 500xCompressor: Generalized Prompt Compression for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2408.03094](https://arxiv.org/abs/2408.03094)  
**Code**: [https://github.com/ZongqianLi/500xCompressor](https://github.com/ZongqianLi/500xCompressor)  
**Area**: Model Compression / Prompt Compression  
**Keywords**: Prompt Compression, Soft Prompt, KV Cache, High Compression Ratio, Autoencoder

## TL;DR

The paper proposes 500xCompressor, which compresses up to around 500 natural language tokens into the KV values of as few as 1 special token, achieving compression ratios from 6x to 480x. It introduces only about 0.25% of additional parameters, while the LLM retains 62.26%–72.89% of its original capabilities after compression, significantly outperforming the ICAE baseline.

## Background & Motivation

Long prompts introduce multiple challenges in NLP applications: reduced inference speed, increased computational costs, degraded user experience, and context length limits that constrain model application scenarios.

Existing prompt compression methods can be categorized into two classes:
- **Hard Prompt**: e.g., SelectiveSentence, LLMLingua, which prune low-information sentences/words/tokens, but only perform selective preservation.
- **Soft Prompt**: e.g., GIST, AutoCompressor, ICAE, which compress natural language tokens into a small number of special tokens.

Main limitations of prior work:

**Low compression ratio**: ICAE only achieves up to approximately 15x compression.

**Unclear information loss**: Metrics like win rate cannot quantitatively depict the information loss from compression.

**Risk of data leakage**: The evaluation text from the Pile dataset potentially overlaps with the LLaMA pre-training data, making it unclear whether the LLM's output stems from the compressed tokens or the model's memory.

## Method

### Overall Architecture

500xCompressor adopts an autoencoder-like architecture containing an encoder and a decoder:
- **Encoder**: A frozen LLM with trainable LoRA parameters (accounting for ~0.25% additional parameters).
- **Decoder**: The original frozen LLM (without any additional parameters).

Core workflow: Original text is input to the encoder $\rightarrow$ Information is encoded into the KV values of the compressed tokens via the attention mechanism $\rightarrow$ The KV values are passed to the decoder $\rightarrow$ The decoder reconstructs the original text or answers questions.

### Key Designs

**1. Utilizing KV Values instead of Embeddings**

Different from ICAE which utilizes the embeddings of compressed tokens, 500xCompressor employs the KV values of compressed tokens at each layer of the LLM. KV values can encapsulate richer information, do not increase inference time, and have a minimal impact on GPU memory.

This design is analogous to the relationship between Prefix Tuning and Prompt Tuning: Prompt Tuning only trains the embeddings of prefix tokens, whereas Prefix Tuning trains the KV values across layers, enabling the encoding of richer information.

**2. BOS Token Triggering Mechanism**

A [BOS] token is used to trigger the LLM to reconstruct the compressed text, whereas ICAE requires creating new trainable tokens. This design is more elegant and compatible with the inherent generation mechanism of LLMs.

**3. Training Design and Mitigating Data Leakage**

The original LLM parameters in both the encoder and decoder are frozen, and no additional parameters are introduced into the decoder. Consequently, no information is retained inside the decoder, ensuring that the inference output comes exclusively from the KV values of the compressed tokens.

### Loss & Training

**Pre-training Phase**:
- Input: KV values of compressed tokens + [BOS] + original text
- Loss: Cross-entropy between decoder output and original text
- Objective: Learn to encode textual information into the compressed tokens

$$\mathcal{L}_P = -\sum_{i=1}^{l} \log P(t_i | H_C, [BOS], t_{1:i-1}; \Theta_{LLM}, \Theta_{Lora})$$

**Fine-tuning Phase**:
- Input: KV values of compressed tokens + question + answer
- Loss: Cross-entropy on the answer segment
- Objective: Learn to retrieve information from the KV values of compressed tokens and generate answers

$$\mathcal{L}_F = -\sum_{j=1}^{n} \log P(a_j | H_C, q_{1:m}, a_{1:j-1}; \Theta_{LLM}, \Theta_{Lora})$$

**Training Data**:
- Pre-training: Arxiv Corpus (abstracts published before July 2023 as the training set)
- Fine-tuning: ArxivQA dataset (extractive QA pairs generated from abstracts by LLaMA-3-70b-chat)
- Evaluation Set: Abstracts from papers published after January 2024 (strictly unseen data)

**Training Configuration**:
- Learning rate: 1e-4 for pre-training, 5e-5 for fine-tuning
- Batch size: 4
- Optimizer: AdamW
- Number of compressed tokens: 1 / 4 / 16

## Key Experimental Results

### Main Results

**Text Reconstruction (Arxiv Corpus Test Set)**:
- 500xCompressor outperforms ICAE across all compression ratios.
- ROUGE-L-F difference range: 12.18%–18.96%.
- BLEU difference range: 12.41%–26.50%.
- The quality drop is minor when reducing from 16 to 4 tokens, but is pronounced when dropping from 4 to 1 token.

**QA on ArxivQA Dataset**:
- F1 gain: 2.06%–9.23%
- EM gain: 0.56%–7.20%
- The higher the compression ratio, the larger the advantage of 500xCompressor over ICAE.

**Cross-Task Generalization (5 Benchmarks)**:

| Task | 500→16 F1 | 500→4 F1 | 500→1 F1 |
|---|---|---|---|
| ArxivQA (Information Extraction) | 40.77 vs 38.70 | 38.30 vs 33.31 | 29.73 vs 20.50 |
| SQuAD (Information Extraction) | 50.01 vs 51.94 | 49.66 vs 47.48 | 42.86 vs 27.20 |
| RelationExtraction | 68.73 vs 65.94 | 63.72 vs 67.28 | 63.09 vs 44.46 |
| HotpotQA (Multi-hop) | 41.68 vs 42.11 | 36.86 vs 40.39 | 37.47 vs 22.44 |
| RACE (Reading Comprehension) | 35.42 vs 23.69 | 21.49 vs 20.06 | 21.75 vs 13.71 |

For 500→1 compression, 500xCompressor comprehensively outperforms ICAE across all benchmarks, with a maximum improvement of 18.62% (RelationExtraction F1).

**Comparison with Gold Standard**:
- Instruct full context (Full Text + Instruction): Average F1 of 61.36 on the 5 benchmarks.
- 500xCompressor 500→16: Average F1 of 45.32, retaining ~73.8% performance.
- 500xCompressor 500→1: Average F1 of 38.98, retaining ~63.5% performance.

### Ablation Study

**Impact of the Number of Compressed Tokens**:
- 500xCompressor does not utilize all tokens uniformly: reconstruction quality changes minimally from 16 to 4 tokens but drops significantly from 4 to 1.
- This indicates that 500xCompressor is more efficient in utilizing a small number of compressed tokens.
- ICAE does not display this two-phase degradation behavior.

**Degradation Rate under Different Compression Ratios**:
- ICAE degrades faster at higher compression ratios.
- 500xCompressor exhibits better scalability.

### Key Findings

1. **KV Values Outperform Embeddings**: The advantage of preserving information under extreme compression ratios is significant.
2. **Compressed Tokens Are Not Uniformly Utilized**: There is an uneven distribution of information across tokens.
3. **Reconstruction Errors Do Not Necessarily Impede QA**: Even if reconstruction fails, the model can still answer questions correctly (and vice versa).
4. **Advantages Are More Pronounced at High Compression Ratios**: The margin of improvement over ICAE is largest at the extreme 500→1 compression setting.
5. **Multi-Hop Reasoning and Reading Comprehension Are Heavily Affected**: Performance drops significantly on HotpotQA and RACE.
6. **Strictly Unseen Data Validates Generalization**: Utilizing post-2024-01 data mitigates potential interference from pre-training model memory.

## Highlights & Insights

- Achieves an extreme compression ratio of 480x, far exceeding the prior limit of <50x and pushing the boundary of prompt compression exploration.
- The design of leveraging KV values instead of embeddings elegantly exploits the characteristics of the Transformer attention mechanism.
- The construction methodology of the ArxivQA dataset is worth mimicking: utilizing paper timestamps to strictly control data leakage.
- The perspective of interpreting compressed tokens as a 'new language for LLMs' is inspiring (comprising three key factors: encoding information, transmitting information, and evaluating adaptability).
- High compression performance is achieved by training only LoRA parameters (~0.25%), showcasing extreme parameter efficiency.

## Limitations & Future Work

- Pre-training and fine-tuning were only conducted on the relatively small Arxiv corpus; scaling up training data is expected to yield further improvements.
- The system currently supports only ~500 input tokens; scalability to much longer text warrants verification.
- Performance drops substantially on multi-hop reasoning (HotpotQA) and reading comprehension (RACE) tasks.
- The uneven information distribution across compressed tokens needs further investigation.
- Performance in real-world application scenarios such as RAG, in-context learning, and role-playing remains unexplored.
- The evaluation is restricted to LLaMA-3-8b-chat; applicability to other LLM architectures is yet to be validated.

## Related Work & Insights

- The core difference from ICAE lies in embeddings vs. KV values, resembling the relationship between Prompt Tuning and Prefix Tuning.
- Complementary to hard prompt methods like LLMLingua: hard prompt selections preserve selectively, while soft prompts compress holistically.
- Applications of soft prompts to RAG, such as xRAG and COCOM, represent a natural direction for downstream expansion.
- Research on Function Vectors shows that special tokens can encode high-level semantic functions, which echoes the 'new language' perspective of compressed tokens.
- Offers valuable insights for KV cache compression and long-context inference acceleration.

## Rating

- **Novelty**: 7/10 — Though architectural modification (KV values replacing embeddings) is not revolutionary, it yields significant efficacy.
- **Technical Depth**: 7/10 — Comprehensive experiments and thorough ablation, but relatively limited theoretical analysis.
- **Practicality**: 8/10 — High compression ratio directly benefits inference acceleration and cost reduction; the codebase is open-sourced.
- **Writing Quality**: 7/10 — Well-structured, although some comparisons with ICAE are slightly repetitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
