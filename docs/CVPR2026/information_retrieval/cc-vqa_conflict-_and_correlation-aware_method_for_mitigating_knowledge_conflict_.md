---
title: >-
  [Paper Note] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering
description: >-
  [CVPR 2026][Information Retrieval & RAG][Knowledge Conflict] CC-VQA is proposed as a training-free method for mitigating knowledge conflicts in KB-VQA. Through a two-stage strategy combining visual-centric contextual con…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "Knowledge Conflict"
  - "Retrieval-Augmented Generation"
  - "KB-VQA"
  - "Visual Reasoning"
  - "Contrastive Decoding"
  - "Positional Encoding Compression"
date: 2026-05-08
content_hash: 866d81b093a1eb1c
---

# CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering

**Conference**: CVPR 2026
**arXiv**: [2602.23952](https://arxiv.org/abs/2602.23952)
**Code**: [github.com/cqu-student/CC-VQA](https://github.com/cqu-student/CC-VQA)
**Area**: Information Retrieval
**Keywords**: Knowledge Conflict, Retrieval-Augmented Generation, KB-VQA, Visual Reasoning, Contrastive Decoding, Positional Encoding Compression

## TL;DR
CC-VQA is proposed as a training-free method for mitigating knowledge conflicts in KB-VQA. Through a two-stage strategy combining visual-centric contextual conflict reasoning and correlation-guided encoding/decoding, it achieves absolute accuracy improvements of 3.3%–6.4% on three benchmarks: E-VQA, InfoSeek, and OK-VQA.

## Background & Motivation
1. Knowledge-based VQA (KB-VQA) leverages RAG to incorporate external knowledge, but conflicts arise between external knowledge and the model's parametric knowledge.
2. Analysis shows that while RAG yields a 16.82% accuracy gain, it also introduces 10.53% errors — cases where originally correct answers are misled by incorrect context.
3. Existing knowledge conflict mitigation methods (prompt- or decoding-based) are primarily transferred from purely textual settings, overlooking the critical role of visual information in conflict identification.
4. Retrieved contexts contain substantial redundancy: each context averages 107 sentences, yet 90% of correct answers appear only in the top 25% most similar sentences.
5. Conflicts in multimodal RAG systems are more complex than in text-only settings, involving cross-modal retrieval limitations, intricate visual understanding, and amplified model hallucinations.
6. Effectively mitigating knowledge conflicts requires the joint exploitation of visual-semantic features and fine-grained contextual relevance.

## Method

### Overall Architecture (Two-Stage, Training-Free)
Stage 1: Visual-Centric Contextual Conflict Reasoning → Stage 2: Correlation-Guided Encoding and Decoding

### Key Designs

**Stage 1: Visual-Centric Contextual Conflict Reasoning**

1. **Parametric Context Generation**: The VLM generates answers and background knowledge conditioned on the query $(I, Q)$, which are externalized as parametric context $C_M$.
2. **Visual Reasoning Extraction**: For each retrieved context $C_i$, the visual-logical relationship with the query image is extracted: $R_i = \text{VLM}(I, Q, C_i)$.
3. **Visual-Centric Conflict Analysis**: All visual reasoning results $\{R_i\}$ are synthesized to abstract conflict-critical visual features: $R_{vis} = \text{VLM}(I, Q, \{R_i\})$.

**Stage 2: Correlation-Guided Encoding and Decoding**

1. **Fine-Grained Relevance**: EVA-CLIP is used to compute the relevance of each sentence to the rewritten question $Q^*$ and image $I$:
    $r_{ij} = \frac{1}{2}(\text{EVA-CLIP}(Q^*, s_{ij}) + \text{EVA-CLIP}(I, s_{ij}))$

2. **Positional Encoding Compression**: For low-relevance sentences (bottom $\tau$ percentile), the RoPE positional encoding step size is compressed to $\alpha = 0.5$:
    $\text{pos}(t_j) = \begin{cases} \text{pos}(t_{j-1}) + \alpha & \text{if } \text{sent}(t_j) \in \mathcal{L}_\tau \\ \text{pos}(t_{j-1}) + 1 & \text{otherwise} \end{cases}$

3. **Adaptive Decoding**: A relevance-enhanced conflict score is introduced on top of contrastive decoding:
    $s'_t = \sigma(D_t + \Delta H_t + K + \delta)$
   where $K = 1 - (\frac{1}{N}\sum r_i)(1 - \frac{H(\mathbf{r})}{\log M})$ combines mean relevance and concentration.

## Key Experimental Results

### Main Results: Three KB-VQA Benchmarks

| Method | E-VQA Single-Hop | E-VQA All | InfoSeek All |
|--------|-------------------|-----------|--------------|
| Qwen2.5-VL-7B (zero-shot) | 21.7 | 20.3 | 23.7 |
| EchoSight | 26.4 | 24.9 | 30.4 |
| Wiki-LLaVA | 17.7 | 20.3 | 28.9 |
| ReflectiVA | 28.0 | 29.2 | 40.1 |
| MMKB-RAG | 39.7 | 35.9 | 36.4 |
| **CC-VQA** | **Best** | **Best** | **Best** |

### Ablation Study

| Component | E-VQA Impact | InfoSeek Impact |
|-----------|-------------|-----------------|
| w/o Visual Conflict Reasoning | −2–3% | −2–3% |
| w/o Positional Encoding Compression | −1–2% | −1–2% |
| w/o Adaptive Decoding | −2–3% | −2–3% |
| Full CC-VQA | Best | Best |

### Key Findings
- CC-VQA achieves absolute improvements of 3.3%–6.4% across all benchmarks, entirely training-free.
- The contribution of visual-semantic features to contextual conflict identification is empirically validated.
- Positional encoding compression effectively reduces attention weights on low-relevance content without compromising information integrity.

## Highlights & Insights
- CC-VQA is the first to systematically leverage visual information to assist in knowledge conflict detection, rather than relying solely on textual reasoning.
- Positional encoding compression is an elegant design — it avoids context truncation and instead attenuates the attention weights of low-information sentences.
- Being entirely training-free, CC-VQA can be directly deployed on arbitrary VLMs.

## Limitations & Future Work
- The approach relies on the VLM's intrinsic capabilities for conflict reasoning and visual analysis, which may limit effectiveness on weaker models.
- Multi-step VLM invocations (parametric context generation, visual reasoning, and conflict analysis) increase inference latency.
- EVA-CLIP-based relevance estimation may be inaccurate in certain specialized domains.

## Related Work & Insights
- Compared to purely decoding-based methods such as AdaCAD and CoCoA, CC-VQA additionally incorporates encoding-side positional compression and visual conflict reasoning.
- Compared to FaithfulRAG, CC-VQA employs visual information for conflict detection rather than relying solely on text-level self-reflection.
- Insight: Positional encoding manipulation as a means of controlling attention allocation is broadly applicable to long-context scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ (novel combination of visual-centric conflict reasoning and positional encoding compression)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 mainstream benchmarks with comprehensive ablation)
- Writing Quality: ⭐⭐⭐⭐ (observation-driven method design with thorough data analysis)
- Value: ⭐⭐⭐⭐ (strong practical utility due to training-free nature; multimodal RAG conflict is an important problem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](../../ACL2026/information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](../../AAAI2026/information_retrieval/mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)
- [\[CVPR 2026\] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](../../ACL2026/information_retrieval/dqa_diagnostic_question_answering_for_it_support.md)

</div>

<!-- RELATED:END -->
