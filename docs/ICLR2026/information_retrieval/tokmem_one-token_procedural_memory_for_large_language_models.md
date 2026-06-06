---
title: >-
  [Paper Note] TokMem: One-Token Procedural Memory for Large Language Models
description: >-
  [ICLR 2026][Information Retrieval & RAG][procedural memory] This paper proposes TokMem, which compiles reusable task procedures into single trainable memory tokens that serve simultaneously as procedure indices and gener…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "procedural memory"
  - "memory token"
  - "continual learning"
  - "context compression"
  - "tool calling"
date: 2026-05-08
content_hash: 8021e0d4d3505072
---

# TokMem: One-Token Procedural Memory for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.00444](https://arxiv.org/abs/2510.00444)  
**Code**: [https://github.com/MANGA-UOFA/TokMem](https://github.com/MANGA-UOFA/TokMem)  
**Area**: Information Retrieval
**Keywords**: procedural memory, memory token, continual learning, context compression, tool calling

## TL;DR

This paper proposes TokMem, which compiles reusable task procedures into single trainable memory tokens that serve simultaneously as procedure indices and generation control signals, enabling efficient invocation of 1,000+ task procedures without long prompts and supporting catastrophic-forgetting-free continual expansion.

## Background & Motivation

- **Efficiency bottleneck of long prompts**: Modern LLMs rely on prompts to control behavior, but constructing long prompts is costly, self-attention incurs quadratic computational overhead, and context window occupation leads to truncation.
- **Limitations of retrieval augmentation**: Methods such as RAG externalize prompts but retrieved content still occupies the context window as text, and frequently used procedures must be re-interpreted at every call.
- **Cognitive science inspiration**: Human procedural memory (e.g., riding a bicycle) is compiled through practice into efficient skills that no longer require re-reading declarative knowledge each time.
- **Core Idea**: Frequently used task procedures are "compressed" into dedicated memory tokens, enabling procedure invocation at constant overhead.

## Method

### Overall Architecture

TokMem augments the LLM vocabulary with $l$ special tokens as a memory bank:

$$M = \begin{bmatrix} \bm{m}_1^\top \\ \vdots \\ \bm{m}_l^\top \end{bmatrix} \in \mathbb{R}^{l \times d}$$

Each $\bm{m}_i$ is a trainable vector representing a unique procedure with no direct textual form.

### Key Design 1: Memory Token Training

Training sequences contain procedure–response pairs, with memory tokens and text tokens interleaved:

$$\bm{a} = (q_1, \ldots, q_k, \underbrace{a_{m_i}, a_{r_{i1}}, a_{r_{i2}}, \ldots}_{\text{procedure–response pair}}, \underbrace{a_{m_j}, a_{r_{j1}}, \ldots}_{\text{procedure–response pair}}, \ldots)$$

Training uses the standard next-token prediction loss, updating only the memory embeddings while the backbone LLM and original token embeddings are fully frozen:

$$\mathcal{L}(\bm{a}; M) = -\sum_{i>k} \log \Pr(a_i \mid \bm{a}_{<i}; M)$$

### Key Design 2: Inference-Time Memory Routing

Given a query $q$, the model predicts a distribution over memory tokens from the final hidden state $h_k$:

$$P(a_{m_i} \mid q) \propto \exp(\text{logit}(m_i \mid h_k))$$

- The highest-probability memory token is appended to the query, and the response is generated autoregressively.
- Multi-step procedure chaining is supported: after generating a partial response, the model predicts the next memory token.
- When no matching procedure exists, all memory logits remain low and the model automatically falls back to ordinary text generation.

### Key Design 3: Renormalization

In continual learning, new embeddings tend to suffer from norm inflation, suppressing existing memories. The proposed solution is:

$$\bm{m}_i \leftarrow \bm{m}_i \cdot \frac{\bar{n}_I}{\|\bm{m}_i\|_2 + \varepsilon}, \quad i \in A$$

where $\bar{n}_I = \text{mean}_{j \in I} \|\bm{m}_j\|_2$ is the typical norm of existing memories. This preserves the direction of new embeddings while aligning only their magnitudes; the computational cost of $O(|A| d)$ is negligible.

### Parameter Isolation Property

- The knowledge of each procedure is stored entirely within an independent token embedding.
- New procedures can be added continuously without interfering with existing ones.
- This naturally supports continual learning without catastrophic forgetting.

## Key Experimental Results

### Atomic Memory Recall: Super-Natural Instructions (ROUGE-L)

| Model | Method | 10 Tasks | 200 Tasks | 1000 Tasks | Avg |
|------|------|--------|---------|---------|------|
| Qwen 0.5B | RAG | 50.4 | 38.8 | 34.7 | 40.7 |
| Qwen 0.5B | Fine-Tuning | 52.4 | 40.6 | 43.2 | 45.2 |
| Qwen 0.5B | Replay Memory | 52.4 | 47.2 | 46.7 | 48.7 |
| Qwen 0.5B | **TokMem** | **52.8** | **49.3** | **50.0** | **50.7** |
| Llama 3.2 3B | RAG | 60.0 | 45.8 | 39.9 | 47.3 |
| Llama 3.2 3B | **TokMem** | **68.0** | **61.2** | **61.5** | **62.9** |
| Llama 3.1 8B | **TokMem** | **75.4** | **65.1** | **64.8** | **67.0** |

### Memory Routing Accuracy

| Method | 10 Tasks | 200 Tasks | 1000 Tasks |
|------|--------|---------|---------|
| Sentence-BERT (RAG) | 99.6 | 88.7 | 79.7 |
| TokMem (Qwen 0.5B) | 99.4 | 97.4 | 94.7 |
| TokMem (Llama 8B) | 99.8 | 98.9 | **97.5** |

### Compositional Memory Recall: Tool Calling (APIGen)

| Model | Method | Parameters | Tool Selection Avg | Argument F1 Avg |
|------|------|--------|-------------|-----------|
| Llama 1B | ICL | - | 16.4 | 0.4 |
| Llama 1B | RAG | - | 16.9 | 2.7 |
| Llama 1B | Fine-Tuning | 0.85M | 9.0 | 68.6 |
| Llama 1B | **TokMem** | **0.10M** | **86.2** | 68.9 |

### Key Findings

- TokMem maintains 94.7% routing accuracy at 1,000 tasks (smallest model), far exceeding RAG's 79.7%.
- Training is highly efficient: as few as 10 samples suffice to surpass RAG trained on 500 samples.
- Trainable parameter count is far smaller than LoRA fine-tuning (0.10M vs. 0.85M), yet achieves comparable or superior performance.
- No catastrophic forgetting is observed in continual learning; performance degrades only slowly as the number of tasks increases.
- Multi-step procedure chaining is supported; in tool-calling scenarios, the model can sequentially invoke parse→search→format procedures.

## Highlights & Insights

- **Conceptual elegance**: Compressing procedural knowledge into a single token achieves a seamless integration of cognitive science and engineering.
- **Parameter isolation**: Each procedure is stored independently, yielding natural forgetting-free continual learning.
- **Extreme efficiency**: Constant memory overhead eliminates the quadratic computational cost of long prompts.
- **Remarkable routing accuracy**: Even on a 0.5B model managing 1,000 tasks, routing accuracy remains at 94.7%.

## Limitations & Future Work

- Procedures must be predefined and trained in advance; zero-shot procedure creation is not supported.
- The information capacity of a single token is limited, and complex procedures may not be fully encoded.
- The capacity ceiling of the embedding space is unknown—routing may degrade when the number of procedures becomes very large.
- Renormalization is a post-hoc operation and may not fully resolve distribution shift in continual learning.
- Evaluation is limited to QA and tool-calling scenarios; applicability to creative generation, long-form text, and other tasks remains unexplored.

## Related Work & Insights

- **Context engineering**: CoT, RAG, MemGPT—all extend prompts with text, occupying the context window.
- **Parameter-efficient fine-tuning**: LoRA, Adapter—update backbone parameters and may cause forgetting.
- **Soft prompting**: Prompt tuning, Prefix tuning—train continuous prompt vectors but typically do not model them as independent memory units.
- **Cognitive science**: Procedural memory in ACT-R theory—skills are compiled through practice into efficient modules.
- **TokMem**: The first work to apply the tokenization paradigm in NLP to procedural memory management.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Practical Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](../../ICML2026/information_retrieval/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](../../ACL2026/information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)

</div>

<!-- RELATED:END -->
