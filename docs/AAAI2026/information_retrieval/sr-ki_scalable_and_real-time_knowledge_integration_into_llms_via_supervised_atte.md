---
title: >-
  [Paper Note] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention
description: >-
  [AAAI 2026][Information Retrieval & RAG][Knowledge injection] This paper proposes SR-KI, a framework that injects structured knowledge bases into LLM KV caches via a two-stage training procedure (retrieval layer localiza…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "Knowledge injection"
  - "large language models"
  - "supervised attention"
  - "KV cache"
  - "retrieval layer"
  - "knowledge base compression"
  - "RAG alternative"
date: 2026-05-08
content_hash: 34d47faafdcc956a
---

# SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention

**Conference**: AAAI 2026
**arXiv**: [2511.06446](https://arxiv.org/abs/2511.06446)
**Authors**: Bohan Yu, Wei Huang, Kang Liu (Baidu; Institute of Automation, Chinese Academy of Sciences)
**Code**: To be released
**Area**: Information Retrieval
**Keywords**: Knowledge injection, large language models, supervised attention, KV cache, retrieval layer, knowledge base compression, RAG alternative

## TL;DR

This paper proposes SR-KI, a framework that injects structured knowledge bases into LLM KV caches via a two-stage training procedure (retrieval layer localization + attention supervision loss). On a single A100 40GB GPU, SR-KI supports injection of up to 40K knowledge base entries, achieves a compression ratio of up to 99.75% via top-100 selection, and maintains an average Recall@10 above 88%.

## Background & Motivation

### State of the Field
Although LLMs possess strong comprehension and reasoning capabilities, efficient knowledge injection remains a critical challenge in scenarios requiring external knowledge (e.g., real-time factual updates, domain-specific data). Existing approaches each have limitations: parameter fine-tuning risks catastrophic forgetting and overfitting and does not support frequent knowledge updates; RAG depends on the quality of an external retriever, is constrained by the LLM context window, and incurs quadratic computational overhead with input length; long-context LLMs can process all inputs directly but demand enormous memory and compute.

### Limitations of Prior Work
KBLaM, the most recent KV-projection method, injects structured knowledge into the LLM KV cache as key-value pairs to avoid memorizing specific facts. However, **as the injected knowledge base grows, KBLaM fails to focus on the most relevant entries**, resulting in severe performance degradation: at a KB size of 10K, Reference ID accuracy drops to 0.87% and BERTScore becomes negative. Furthermore, KBLaM exceeds 30 GB of GPU memory at 30K entries and encounters OOM at 40K. All existing methods also struggle to attribute model outputs to specific knowledge sources under large-scale injection, leaving controllability and interpretability unsatisfactory.

### Root Cause
A scalable, real-time, end-to-end knowledge injection method is needed that can maintain precise retrieval over large knowledge bases while supporting knowledge provenance. The key insight is that LLMs contain specific *retrieval layers* at which knowledge injection is most effective—a property of model architecture rather than a task-specific factor. This motivates applying an attention supervision loss at the retrieval layer to guide the model to attend precisely to relevant knowledge entries.

## Core Problem

How to efficiently inject a large-scale structured knowledge base into an LLM without relying on an external retriever, such that at inference time the model can: (1) precisely retrieve relevant knowledge entries; (2) generate accurate answers based on retrieved knowledge; (3) simultaneously output traceable source IDs; and (4) maintain robust performance as KB size scales from 100 to 40K entries.

## Method

### Knowledge Base Representation and Injection

Each knowledge triple $(s_m, r_m, o_m)$ is converted into a key-value pair: the key is formed from $(s_m, r_m)$ in natural language ("the $r_m$ of $s_m$") and the value is the entity $o_m$. After encoding with a pretrained sentence encoder (bge-large-zh-v1.5), the representations are projected into the LLM embedding dimension via learnable single-layer linear adapters:

$$\{(s_m, r_m, o_m)\}_{m=1}^{M} \xrightarrow{\text{Encode}} \{(k_m, v_m)\}_{m=1}^{M}, \quad \{(\tilde{k}_m, \tilde{v}_m)\} = \{(k_m \tilde{W}_K, v_m \tilde{W}_V)\}$$

The augmented KV cache contains $M$ knowledge entries and $N$ original tokens. Attention computation uses a Rectangle Attention mechanism:

$$\text{RectangleAtt}(Q^l, \tilde{Q}^l, \tilde{K}^l, \tilde{V}^l) = \text{Softmax}\left(\left[A_{\text{KB}}^l \middle| A^l\right]\right) \tilde{V}^l$$

where $A_{\text{KB}}^l \in \mathbb{R}^{N \times M}$ denotes KB attention weights and $A^l \in \mathbb{R}^{N \times N}$ denotes the original self-attention weights.

### Two-Stage Training

**Stage 1: Retrieval Layer Localization.** The pretrained parameters are frozen; only the projection adapters $\tilde{W}_Q^l, \tilde{W}_K^l, \tilde{W}_V^l$ are trained. By injecting the correct KB at each layer in turn (with random negative samples at all other layers), the layer with the best retrieval performance is identified. Experiments show that layer 25 of Qwen2.5-7B-Instruct serves as the retrieval layer, a finding that is consistent across model sizes (3B/14B), model families (Llama-3-8B), and encoders (bge-m3/Qwen3-Embedding-8B).

**Stage 2: Supervised Attention Training.** An attention supervision loss is introduced at the retrieval layer. KB attention weights are aggregated as:

$$\overline{A_{\text{KB}}^l} = \frac{1}{N} \sum_{n=1}^{N} A_{\text{KB}}^l[n,:] \in \mathbb{R}^M$$

The top-$k$ KB entries with the highest attention weights are retained; non-ground-truth entries among them serve as hard negatives $\text{KB}_{\text{neg}}^l$. A candidate set is constructed for each ground-truth KB entry, and cross-entropy loss is computed:

$$\mathcal{L}_a = -\frac{1}{J}\sum_{j=1}^{J} \log\frac{\exp(\overline{A_{\text{KB}}^{\tilde{l}}}[i_j] / \mathcal{T})}{\sum_{i \in \mathcal{N}_j} \exp(\overline{A_{\text{KB}}^{\tilde{l}}}[i] / \mathcal{T})}$$

where the temperature $\mathcal{T} = 0.05$ amplifies the contrast between correct KB entries and negatives. The total training loss is: $\mathcal{L} = \mathcal{L}_{\text{lm}} + \mathcal{L}_a$.

### Inference Optimization

- **KB Compression**: At the retrieval layer, the aggregated attention weights are used to select the top-$k$ ($k=100$) most relevant KB entries.
- **Cross-Layer Reuse**: The KB indices selected at the retrieval layer are reused across all subsequent layers, avoiding redundant compression and reducing inference overhead.
- **Reference ID KB**: Each knowledge entry is assigned a random uppercase letter ID; the same triple may receive different IDs at different training steps, forcing the model to learn robust KV mapping patterns.

## Key Experimental Results

### Experimental Setup
- **Model**: Qwen2.5-7B-Instruct
- **Encoder**: bge-large-zh-v1.5
- **KB sizes**: 100 / 1K / 10K / 40K
- **Tasks**: Single-entity QA, multi-entity QA, unanswerable QA
- **Baselines**: In-context Learning (ICL), KBLaM

### Main Results: Task Performance (ID-Acc / K-BERT)

| KB Size | Method | ID-Acc (avg) | K-BERT (avg) |
|---------|--------|-------------|-------------|
| 100 | ICL | 0.6730 | 0.9851 |
| 100 | KBLaM | 0.9730 | 0.8725 |
| 100 | **SR-KI** | **0.9837** | 0.8547 |
| 1K | KBLaM | 0.7817 | 0.6852 |
| 1K | **SR-KI** | **0.9467** | **0.7817** |
| 10K | KBLaM | 0.0087 | -1.2708 |
| 10K | **SR-KI** | **0.7800** | **0.6677** |
| 40K | KBLaM | OOM | OOM |
| 40K | **SR-KI** | **0.6940** | **0.6039** |

### Retrieval Performance (Recall)

| KB Size | Method | R@100 | R@10 | R@Top |
|---------|--------|-------|------|-------|
| 1K | KBLaM | 0.4375 | 0.0952 | 0.0465 |
| 1K | **SR-KI** | **0.9975** | **0.9808** | **0.9415** |
| 10K | KBLaM | 0.0737 | 0.0118 | 0.0053 |
| 10K | **SR-KI** | **0.9808** | **0.9318** | **0.8702** |
| 40K | KBLaM | OOM | OOM | OOM |
| 40K | **SR-KI** | **0.9593** | **0.8887** | **0.8027** |

At the 40K scale, SR-KI maintains 95.93% R@100 and 88.87% R@10, demonstrating strong retrieval scalability.

### Memory Comparison

SR-KI's memory usage remains **well below the 40 GB limit** at 40K KB entries, whereas KBLaM exceeds 30 GB at 30K and encounters OOM at 40K. ICL hits the memory ceiling at only a few hundred entries.

### Ablation Study: Effect of Cross-Layer Reuse

| KB Size | Metric | w/o Reuse | w/ Reuse |
|---------|--------|-----------|----------|
| 1K | ID-Acc | 0.8167 | **0.9467** |
| 10K | ID-Acc | 0.5000 | **0.7800** |
| 40K | ID-Acc | 0.3600 | **0.6940** |
| 40K | K-BERT | 0.3636 | **0.6039** |

Cross-layer reuse yields **nearly double** the performance at large scales, confirming the importance of the indices selected at the retrieval layer for subsequent inference.

### Comparison with BM25 and Dense Retrieval (Recall@Top)

| KB Size | BM25 | Dense Retrieval | SR-KI |
|---------|------|-----------------|-------|
| 1K | 0.6817 | 0.9300 | **0.9415** |
| 10K | 0.4992 | 0.8417 | **0.8702** |
| 40K | 0.3633 | 0.7108 | **0.8027** |

At the 40K scale, SR-KI outperforms Dense Retrieval by 9.19 points and BM25 by 43.94 points.

## Highlights & Insights

- **End-to-end knowledge retrieval and reasoning**: No external retriever or multi-stage pipeline is required; all retrieval is performed within the model's latent space, simplifying system architecture.
- **Architectural universality of the retrieval layer**: A specific retrieval layer exists across different model sizes (3B/7B/14B), model families (Qwen/Llama), and encoders—a property of model architecture rather than a task-specific artifact.
- **Extreme compression ratio of 99.75%**: Compressing from 40K entries to the top-100 still enables effective inference with minimal and stable memory usage.
- **Knowledge provenance**: The Reference ID KB mechanism enables simultaneous generation of answers and source IDs, supporting transparency and verifiability of outputs.
- **Vastly superior scalability over KBLaM**: At the 10K scale, KBLaM's ID accuracy drops to 0.87%, while SR-KI retains 78%.

## Limitations & Future Work

- **Degraded refusal capability**: Supervised attention training causes a decline in rejection accuracy on unanswerable QA, particularly at large KB scales.
- **Chinese-only knowledge base**: Experiments are based on a Chinese subset of Wikidata and Chinese QA templates; multilingual generalizability has not been validated.
- **Single-hop reasoning only**: Current QA tasks are limited to single-hop and simple multi-entity queries; multi-hop and complex reasoning chains are not addressed.
- **BERTScore degrades with scale**: Although ID accuracy remains relatively high, BERTScore for knowledge-grounded answers drops to 0.60 at 40K, indicating room for improvement in fine-grained knowledge alignment.
- **Semantic-free Reference IDs**: Using random uppercase letters as IDs lacks semantic information, potentially limiting knowledge management in practical applications.

## Related Work & Insights

- **KBLaM** (Wang et al. 2025): Also employs KV projection for knowledge injection but lacks a supervised attention mechanism, causing performance collapse at the 10K scale. SR-KI achieves large-scale scalability through retrieval-layer supervision.
- **RAG** (Lewis et al. 2021): Relies on an external retriever and a multi-stage pipeline; constrained by context window size, and the decoupled retrieve-then-generate approach is prone to hallucination. SR-KI performs end-to-end retrieval within the model.
- **Knowledge editing methods** (ROME, MEMIT): Directly modify model parameters for knowledge updates but support only a limited number of edits and do not support dynamic updates. SR-KI supports runtime knowledge replacement.
- **In-context Learning**: Places knowledge directly in the prompt; memory grows quadratically with KB size, making it infeasible beyond ~100 entries.
- **BM25 / Dense Retrieval**: Traditional retrieval methods struggle with complex instructions and multi-dimensional semantic queries. SR-KI exceeds these methods by 9–44 points at the 40K scale.

The discovery of architecturally universal retrieval layers may advance understanding of knowledge processing mechanisms within Transformers. The supervised attention loss design is generalizable to other settings requiring precise attention guidance (e.g., multimodal alignment, long-document understanding). The Reference ID mechanism for knowledge provenance offers a lightweight approach to trustworthy and interpretable AI.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two-stage design combining retrieval layer localization and attention supervision is novel; the 99.75% compression ratio is impressive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-scale evaluation from 100 to 40K, cross-model validation, multiple baselines, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear and experimental design is sound, though the theoretical explanation for the retrieval layer finding is relatively shallow.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient and practical solution for large-scale knowledge injection, though coverage is limited to Chinese and simple QA scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](../../ACL2026/information_retrieval/curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](../../ACL2026/information_retrieval/taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](towards_inference-time_scaling_for_continuous_space_reasoning.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)

</div>

<!-- RELATED:END -->
