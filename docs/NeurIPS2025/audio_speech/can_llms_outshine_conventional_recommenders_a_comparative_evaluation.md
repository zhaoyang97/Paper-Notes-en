---
title: >-
  [Paper Note] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation
description: >-
  [NeurIPS 2025][Audio & Speech][LLM-as-RS] This paper proposes RecBench, a comprehensive evaluation framework that systematically compares 17 LLMs against 10 conventional DLRMs across 5 domain-specific datasets. Results s…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "LLM-as-RS"
  - "RecBench"
  - "CTR prediction"
  - "sequential recommendation"
  - "item representation"
  - "inference efficiency"
date: 2026-05-08
content_hash: e73133a820fbf053
---

# Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation

**Conference**: NeurIPS 2025
**arXiv**: [2503.05493](https://arxiv.org/abs/2503.05493)  
**Code**: [RecBench](https://recbench.github.io)  
**Area**: Audio & Speech
**Keywords**: LLM-as-RS, RecBench, CTR prediction, sequential recommendation, item representation, inference efficiency

## TL;DR

This paper proposes RecBench, a comprehensive evaluation framework that systematically compares 17 LLMs against 10 conventional DLRMs across 5 domain-specific datasets. Results show that LLM-based recommenders achieve up to 5% AUC improvement on CTR tasks and up to 170% NDCG@10 improvement on sequential recommendation, yet incur 10–1000× slower inference. Conventional DLRMs augmented with LLM semantic embeddings (LLM-for-RS) attain approximately 95% of LLM performance at 20× higher throughput, making this paradigm the most industrially viable solution at present.

## Background & Motivation

**Background**: The integration of LLMs with recommender systems (LLM+RS) has attracted considerable research attention, giving rise to two paradigms: LLM-for-RS (LLMs as feature-enhancement plugins) and LLM-as-RS (LLMs serving directly as recommenders). The latter has demonstrated promise in cold-start and explainable recommendation scenarios, yet systematic evaluation remains lacking.

**Limitations of Prior Work**: Existing benchmarks (LLMRec, PromptRec, OpenP5, etc.) suffer from three major deficiencies: (a) they evaluate only a single recommendation formulation (pair-wise or list-wise); (b) item representation forms are insufficiently covered, typically relying solely on text or unique IDs; and (c) the number of evaluated models is limited, with inference efficiency entirely overlooked.

**Core Problem**: Does the accuracy advantage of LLMs on recommendation tasks sufficiently compensate for their substantial inference efficiency drawbacks? How do different item representation strategies affect LLM recommendation capability?

**Key Insight**: The paper constructs RecBench, the most comprehensive LLM recommendation evaluation benchmark to date, simultaneously assessing both accuracy and efficiency across 4 item representations, 2 recommendation scenarios, 27 models, and 5 datasets.

## Method

### Overall Architecture

The RecBench evaluation matrix spans: **5 datasets** (H&M Fashion, MIND News, MicroLens Video, Goodreads Books, Amazon CDs Music) × **4 item representations** (unique ID, text, semantic embedding, semantic identifier) × **2 recommendation tasks** (CTR prediction, sequential recommendation) × **27 models** (17 LLMs + 10 DLRMs), with both accuracy metrics and inference latency measured.

### Four Item Representation Strategies

1. **Unique Identifier**: The conventional approach, assigning each item a randomly initialized embedding vector whose semantics are learned through collaborative filtering signals.
2. **Text**: Item titles or other textual features are used, with item representations obtained by averaging word embeddings—naturally compatible with LLMs' text comprehension capabilities.
3. **Semantic Embedding**: A pretrained LLM (e.g., Llama-1 7B) encodes item text into dense vectors used to initialize DLRM inputs, introducing rich general-purpose semantics.
4. **Semantic Identifier**: SentenceBERT first extracts item embeddings, which are then discretized via RQ-VAE into a 4-layer × 256-codebook encoding sequence. Semantically similar items share longer common subsequences, simultaneously compressing the vocabulary while preserving semantic relationships.

### Two Recommendation Scenarios

**Pair-wise Recommendation (CTR Prediction)**: Given a user–item pair, the model predicts click probability. Models are organized into six groups (A–F):

- **Group A**: Conventional DLRM + Unique ID (DNN, DeepFM, DCN, DCNv2, AutoInt, GDCN, etc.; 9 models)
- **Group B**: Conventional DLRM + Text (DNN_text, DCNv2_text, etc.; 4 models)
- **Group C**: Conventional DLRM + Semantic Embedding (DNN_emb, GDCN_emb, etc.; 4 models; LLM-for-RS paradigm)
- **Group D**: LLM + Unique ID (P5 series, with item IDs as special tokens)
- **Group E**: LLM + Text (GPT-3.5, Llama series, Qwen series, etc.; supporting both zero-shot and fine-tuned settings)
- **Group F**: LLM + Semantic Identifier (SID-BERT, SID-OPT)

**List-wise Recommendation (Sequential Recommendation)**: Given a user's interaction history, the model predicts the next item. Models are organized into four groups (G–J), incorporating **Constrained Beam Search (CBS)**—a technique that leverages the semantic identifier tree to constrain decoding paths and ensure generated token sequences correspond to valid items.

### Loss & Training

- LLM fine-tuning employs LoRA: rank=32/alpha=128 for CTR tasks; rank=128/alpha=128 for sequential recommendation
- Learning rates: 1e-4 for LLMs; 1e-3 for DLRMs
- All experiments are conducted on a single A100 GPU; results are averaged over 5 runs

## Key Experimental Results

### CTR Prediction (Pair-wise, AUC)

| Item Representation | Representative Model | Overall AUC | CPU Latency (ms) | GPU Latency (ms) |
|---|---|---|---|---|
| Unique ID (best DLRM) | GDCN | 0.6825 | 1.20 | 2.02 |
| Text (best DLRM) | GDCN_text | 0.6923 | 5.09 | 3.77 |
| **Semantic Embedding (best DLRM)** | **DNN_emb** | **0.7171** | **1.42** | **2.09** |
| Text (best fine-tuned LLM) | Mistral-2 7B | 0.7578 | 7680 | 76.14 |
| Best zero-shot LLM | GLM-4 9B | 0.6231 | 9690 | 83.38 |

**Key Findings**: Fine-tuned Mistral-2 achieves AUC of 0.7578, approximately 5.7% above the best DLRM (DNN_emb, 0.7171), yet is **5,400× slower** on CPU inference (7,680 ms vs. 1.42 ms).

### Sequential Recommendation (List-wise, NDCG@10)

| Item Representation | Representative Model | Overall NDCG@10 | CPU Latency (ms) |
|---|---|---|---|
| Unique ID (best DLRM) | SASRec_24L | 0.0698 | 103.41 |
| Unique ID (best LLM) | P5-BERT_base | 0.1025 | 41.54 |
| **Semantic ID (best LLM+CBS)** | **SID-BERT_base-CBS** | **0.1877** | **1900** |
| Semantic ID (large model+CBS) | SID-Llama-3 7B-CBS | 0.1607 | 177540 |

**Key Findings**: SID-BERT_base-CBS achieves NDCG@10 of 0.1877, a **169% improvement** over SASRec_24L (0.0698), at the cost of 18× longer inference time. SID-Llama-3 7B-CBS requires approximately 177 seconds per sample, rendering it completely infeasible for practical deployment.

### Zero-Shot LLM Performance

Most LLMs achieve AUC near 0.50 on zero-shot CTR tasks (approaching random), with only Mistral (0.6199) and GLM-4 (0.6231) performing acceptably. Dedicated recommendation models RecGPT (0.4952) and P5_Beauty (0.5049) exhibit extremely poor zero-shot generalization. The Qwen-2 series demonstrates a positive correlation between model scale and zero-shot recommendation capability (0.5B→1.5B→7B: 0.5413→0.5707→0.6075).

### Gains from Fine-Tuning

Instruction fine-tuning improves LLM CTR AUC by 22%–43% in relative terms. For example, Llama-3 8B improves from 0.5252 (zero-shot) to 0.7508 (fine-tuned).

## Highlights & Insights

1. **"LLM-for-RS" offers the best current trade-off**: DLRM augmented with LLM semantic embeddings (Group C) achieves DNN_emb AUC=0.7171 at very low latency (~2 ms GPU)—approximately 94.6% of the best LLM's performance (0.7578)—while being 36× faster. This is the most practically deployable solution for industrial settings.

2. **Semantic identifiers confer substantial advantages in sequential recommendation**: SID representations enable even shallow networks to capture user interest patterns; SID-SASRec_3L-CBS (0.0306) substantially outperforms SASRec_3L (0.0096). However, the advantage diminishes as depth increases, suggesting that deep ID-based models can learn analogous information.

3. **Constrained Beam Search (CBS) is a critical technique**: CBS constrains decoding via the semantic identifier tree to ensure generation of valid items. SID-BERT_base improves from 0.0941 to 0.1877 with CBS, nearly doubling performance.

4. **An abstract similarity exists between pretrained language patterns and user interest patterns**: BERT_base with unique IDs (no text) achieves 0.1025 on sequential recommendation, surpassing same-architecture SASRec_12L (0.0672), suggesting that sequential patterns learned through language modeling transfer to user behavior modeling.

5. **Model scale is not universally beneficial**: In sequential recommendation, SID-BERT_base-CBS (0.1877) substantially outperforms SID-Llama-3 7B-CBS (0.1607), indicating that smaller models can outperform larger ones under specific configurations.

## Limitations & Future Work

- Only single-sample inference latency is measured; the impact of batch inference and acceleration techniques such as KV-cache is not considered.
- The paper does not evaluate LLM advantages in specialized scenarios such as cold-start or cross-domain recommendation—arguably the most genuinely promising direction for LLM-as-RS.
- Semantic embeddings are produced solely by Llama-1 7B; the effect of stronger encoders (e.g., more recent Llama-3 or domain-specific models) remains unexplored.
- All five datasets are preprocessed and truncated to similar scales, which may not reflect real-world industrial data distributions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First benchmark to simultaneously cover accuracy and efficiency, 4 item representations, and 2 task formulations for LLM-based recommendation
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 27 models × 5 datasets × 4 item representations, results averaged over 5 runs; an uncommonly large-scale evaluation
- **Writing Quality**: ⭐⭐⭐⭐ — Experimental analysis is clear and well-organized, with persuasive conclusions
- **Value**: ⭐⭐⭐⭐⭐ — Provides important strategic guidance for the recommender systems community by establishing a clear finding that LLM-for-RS outperforms LLM-as-RS

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)
- [\[NeurIPS 2025\] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers](sand-math_using_llms_to_generate_novel_difficult_and_useful_mathematics_question.md)
- [\[ACL 2026\] StressTest: Can YOUR Speech LM Handle the Stress?](../../ACL2026/audio_speech/stresstest_can_your_speech_lm_handle_the_stress.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](../../AAAI2026/audio_speech/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)

</div>

<!-- RELATED:END -->
