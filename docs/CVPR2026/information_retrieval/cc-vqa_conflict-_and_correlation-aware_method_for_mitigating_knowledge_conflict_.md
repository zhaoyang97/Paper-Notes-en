---
title: >-
  [Paper Note] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering
description: >-
  [CVPR 2026][Information Retrieval & RAG][KB-VQA] Ours proposes CC-VQA, a training-free method for mitigating knowledge conflict. Through a two-stage strategy involving visual-centric context conflict reasoning and correlation-guided encoding/decoding, it achieves an absolute accuracy gain of 3.3%-6.4% across E-VQA, InfoSeek, and OK-VQA benchmarks.
tags:
  - CVPR 2026
  - Information Retrieval & RAG
  - KB-VQA
date: 2026-05-08
content_hash: 762f75370586e9e4
---
# CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering

**Conference**: CVPR2026  
**arXiv**: [2602.23952](https://arxiv.org/abs/2602.23952)  
**Code**: [github.com/cqu-student/CC-VQA](https://github.com/cqu-student/CC-VQA)  
**Area**: Information Retrieval  
**Keywords**: Knowledge Conflict, Retrieval-Augmented Generation, KB-VQA, Visual Reasoning, Contrastive Decoding, Positional Encoding Compression

## TL;DR
Ours proposes CC-VQA, a training-free method for mitigating knowledge conflict. Through a two-stage strategy involving visual-centric context conflict reasoning and correlation-guided encoding/decoding, it achieves an absolute accuracy gain of 3.3%-6.4% across E-VQA, InfoSeek, and OK-VQA benchmarks.

## Background & Motivation
1. Knowledge-based VQA (KB-VQA) introduces external knowledge via RAG, but conflicts frequently arise between external knowledge and the model's parametric knowledge.
2. Analysis shows that while RAG brings a 16.82% accuracy gain, it also introduces 10.53% new errors—where originally correct answers are misled by incorrect context.
3. Existing knowledge conflict mitigation methods (prompting/decoding) are primarily migrated from text-only scenarios, ignoring the critical role of visual information in conflict identification.
4. High redundancy exists in retrieved contexts: each context averages 107 sentences, yet 90% of correct answers are contained within only the top 25% of the most similar sentences.
5. Conflicts in multimodal RAG systems are more complex than in text-only systems, involving cross-modal retrieval limitations, complex visual understanding, and amplified model hallucinations.
6. There is a need to simultaneously utilize visual semantic features and fine-grained context correlation to mitigate knowledge conflicts.

## Method

### Overall Architecture

CC-VQA targets "knowledge conflict" in KB-VQA: while RAG-introduced external context brings an average 16.82% gain, it also misguides the model in originally correct cases (introducing 10.53% new errors) due to incorrect passages. It is a training-free approach consisting of two sequential phases—Phase 1: visual-centric context conflict reasoning, which explicitly extracts "which claims the image actually supports"; Phase 2: correlation-guided encoding and decoding, using fine-grained relevance to naturally suppress low-information or image-irrelevant sentences within the attention mechanism.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: Query Image I + Question Q<br/>+ Retrieved Context {C_i}"] --> S1
    subgraph S1["Phase 1: Visual-Centric Context Conflict Reasoning"]
        direction TB
        B["VLM externalizes parametric context C_M"] --> C["Extract visual logical relations R_i per context"]
        C --> D["Synthesize conflict visual features R_vis<br/>Generate augmented context with conflict labels"]
    end
    subgraph S2["Phase 2: Correlation-Guided Encoding and Decoding"]
        direction TB
        E["Fine-grained relevance:<br/>EVA-CLIP scores Q* and I per sentence"] --> F["Positional encoding compression:<br/>Step size α=0.5 for low-relevance sentences"]
        F --> G["Adaptive decoding:<br/>Contrastive decoding scores enhanced by relevance"]
    end
    S1 --> S2
    S2 --> H["Output: Answer"]
```

### Key Designs

**1. Visual-Centric Context Conflict Reasoning: Using images rather than pure text logic as the arbiter of conflict**

Prior conflict mitigation often migrated from text-only scenarios, overlooking that vision acts as the judge in VQA. CC-VQA first uses a VLM to generate an answer and background knowledge based on the query $(I,Q)$, externalizing it as parametric context $C_M$. It then extracts the visual logical relationship $R_i = \text{VLM}(I, Q, C_i)$ for each retrieved context $C_i$. Finally, it synthesizes all $\{R_i\}$ to abstract key visual features of the conflict $R_{vis} = \text{VLM}(I, Q, \{R_i\})$. This chain transforms "which context the image supports/refutes" into explicit signals for subsequent encoding and decoding.

**2. Fine-grained Relevance: Locating the few truly useful sentences in retrieved context**

Statistical findings show an average of 107 sentences per context, but 90% of correct answers reside in the top 25% of sentences by similarity, indicating extreme redundancy. CC-VQA uses EVA-CLIP to calculate the relevance of each sentence to the rewritten question $Q^*$ and the image $I$ simultaneously, taking the average: $r_{ij} = \frac{1}{2}(\text{EVA-CLIP}(Q^*, s_{ij}) + \text{EVA-CLIP}(I, s_{ij}))$. This provides a sentence-level basis for de-redundancy.

**3. Positional Encoding Compression: Weakening attention on low-relevance sentences without truncation**

Directly deleting sentences may lose information, while keeping them introduces noise interference. For sentences falling within the low-relevance interval (bottom $\tau$ percentile) $\mathcal{L}_\tau$, CC-VQA compresses their RoPE positional encoding step size to $\alpha=0.5$:

$$\text{pos}(t_j) = \begin{cases} \text{pos}(t_{j-1}) + \alpha & \text{if } \text{sent}(t_j) \in \mathcal{L}_\tau \\ \text{pos}(t_{j-1}) + 1 & \text{otherwise} \end{cases}$$

Low-information sentences are "squeezed" into closer positional intervals, causing their attention weights to decrease, while their content remains fully preserved in the context.

**4. Adaptive Decoding: Incorporating relevance concentration into contrastive decoding**

Building upon contrastive decoding, CC-VQA adds a relevance-enhanced conflict score $s'_t = \sigma(D_t + \Delta H_t + K + \delta)$, where $K = 1 - (\frac{1}{N}\sum r_i)(1 - \frac{H(\mathbf{r})}{\log M})$ combines both mean relevance and the concentration of relevance. When useful information is both high and concentrated, the decoder trusts the context more; otherwise, it relies more on parametric knowledge.

## Key Experimental Results

### Main Results: Three KB-VQA Benchmarks

| Method | E-VQA Single-Hop | E-VQA All | InfoSeek All |
|------|-------------------|-----------|-------------|
| Qwen2.5-VL-7B (zero-shot) | 21.7 | 20.3 | 23.7 |
| EchoSight | 26.4 | 24.9 | 30.4 |
| Wiki-LLaVA | 17.7 | 20.3 | 28.9 |
| ReflectiVA | 28.0 | 29.2 | 40.1 |
| MMKB-RAG | 39.7 | 35.9 | 36.4 |
| **CC-VQA (Ours)** | **Best** | **Best** | **Best** |

### Ablation Study

| Component | E-VQA Impact | InfoSeek Impact |
|------|-----------|--------------|
| w/o Visual Conflict Reasoning | -2-3% | -2-3% |
| w/o Positional Encoding Compression | -1-2% | -1-2% |
| w/o Adaptive Decoding | -2-3% | -2-3% |
| Full CC-VQA | **Best** | **Best** |

### Key Findings
- CC-VQA achieves a 3.3%-6.4% absolute gain across all benchmarks, completely training-free.
- The contribution of visual semantic features to identifying context conflicts is empirically validated.
- Positional encoding compression effectively reduces the attention weight of low-relevance content without destroying information integrity.

## Highlights & Insights
- First to systematically use visual information to assist in knowledge conflict detection, rather than relying solely on textual logic.
- Positional encoding compression is an elegant design—it does not require context truncation, merely weakening the attention weight of low-information sentences.
- Completely training-free, allowing for direct deployment on any VLM.

## Limitations & Future Work
- Dependency on the VLM's inherent ability for conflict reasoning and visual analysis; performance may be limited on weaker models.
- Multi-step VLM calls increase inference latency (parametric context generation + visual reasoning + conflict analysis).
- EVA-CLIP relevance estimation may be inaccurate for certain specialized domains.

## Related Work & Insights
- Difference from pure decoding methods like AdaCAD / CoCoA: CC-VQA adds encoding-side manipulation (positional compression) and visual conflict reasoning.
- Difference from FaithfulRAG: CC-VQA utilizes visual information for conflict detection rather than just text-level self-reflection.
- Insight: Manipulating positional encoding as a means to control attention allocation can be widely applied to long-context scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel combination of visual-centric conflict reasoning and positional encoding compression)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablation across 3 major benchmarks)
- Writing Quality: ⭐⭐⭐⭐ (Observation-driven design with detailed data analysis)
- Value: ⭐⭐⭐⭐ (Strong training-free utility; multimodal RAG conflict is a significant problem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](../../ACL2025/information_retrieval/micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](../../ACL2026/information_retrieval/if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](../../ACL2026/information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](../../ACL2026/information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)

</div>

<!-- RELATED:END -->
