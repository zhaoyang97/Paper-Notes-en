---
title: >-
  [Paper Note] MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG
description: >-
  [ICLR 2026][Information Retrieval & RAG][Parametric RAG] MergePRAG utilizes a hypernetwork to translate retrieved passages from each hop into "passage expert" parameters. These are incrementally superimposed into critical layers of the LLM via a continual merging mechanism based on Gram–Schmidt orthogonalization, extending parametric RAG (PRAG) from single-hop to multi-hop r
tags:
  - ICLR 2026
  - Information Retrieval & RAG
  - Parametric RAG
  - Multi-hop QA
  - Gram–Schmidt
date: 2026-05-08
content_hash: c7b46c2dca6bbe6f
---
# MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FSL1J2gmJV](https://openreview.net/forum?id=FSL1J2gmJV)  
**Code**: [https://github.com/Liu-Xuebing/MhQA_hypernetwork](https://github.com/Liu-Xuebing/MhQA_hypernetwork)  
**Area**: Information Retrieval / Retrieval-Augmented Generation (Parametric RAG)  
**Keywords**: Parametric RAG, Multi-hop QA, Hypernetwork, Orthogonal Merging, Gram–Schmidt, Knowledge Injection  

## TL;DR
MergePRAG utilizes a hypernetwork to translate retrieved passages from each hop into "passage expert" parameters. These are incrementally superimposed into critical layers of the LLM via a continual merging mechanism based on Gram–Schmidt orthogonalization, extending parametric RAG (PRAG) from single-hop to multi-hop reasoning for the first time.

## Background & Motivation
**Background**: There are two main paradigms for injecting external knowledge into LLMs: RAG, which feeds retrieved passages into the context, and Parameterized Knowledge Alignment (PKA), which directly modifies model weights. The emerging Parametric RAG (PRAG) takes a middle ground: using a "hypernetwork" to translate retrieved passages into LoRA-like parameter increments for injection. This "internalizes" knowledge to bypass RAG's inefficiencies with long contexts and noise sensitivity, consistently outperforming standard RAG.

**Limitations of Prior Work**: PRAG has only been validated in **single-hop** settings—one retrieval followed by one answer. Real-world complex QA is **multi-hop**: questions are decomposed into sub-questions, each requiring independent retrieval and sub-answer generation, where passages arrive **incrementally hop-by-hop**. Existing PRAG hypernetworks lack the capability for "continual knowledge accumulation across hops."

**Key Challenge**: How to **continually inject** new passage knowledge into the model in multi-hop scenarios without **retraining or rebuilding** a specialized hypernetwork? Simply applying arithmetic averaging to merge new passage parameters across hops causes **knowledge conflicts** (redundancy, mutual overwriting), disrupting accumulated knowledge.

**Goal**: Extend single-hop PRAG to multi-hop RAG by reusing the same passage-level hypernetwork, allowing passage experts to accumulate without conflict, serving as a bridge to "reasoning-enhanced RAG" (e.g., IRCoT, Self-RAG, DeepRAG).

**Core Idea**: **Orthogonal Continual Merging** — Treat each passage as an "expert" parameter. New experts are added only through the **component orthogonal to the existing expert subspace**, eliminating redundancy and preserving complementary information. Simultaneously, only a pre-selected **critical layer** is updated to efficiently encode passages and stabilize reasoning.

## Method

### Overall Architecture
MergePRAG organizes multi-hop QA into a loop: "Sub-question generation → Retrieval → Parameterization → Continual Merging → Sub-answer generation." At each hop $t$, the sub-question generator produces $sq_t$ from the existing reasoning chain; the retriever returns top passages $SP_t$. Each passage is translated by hypernetwork $H_\phi$ into KV memories. These are first combined via **inner-merging** into the current hop's expert $H_\phi(SP_t)$, then integrated into the cumulative parameters $F(SP_{1:t-1})$ via **orthogonal continual merging** $\text{Merge}_{seq}$ to obtain $F(SP_{1:t})$, which is injected into critical layer $l^*$ of the base LLM to generate $sa_t$. The cycle repeats until no further sub-questions are generated.

```mermaid
flowchart LR
    A[Complex question q] --> B[Sub-question gen Msq<br/>sq_t = Msq(C_1:t-1)]
    B --> C[Retriever R<br/>Returns SP_t]
    C --> D[Hypernetwork Hϕ<br/>Passage→KV Memory Kp,Vp]
    D --> E[Inner-merging<br/>Merge hop expert Hϕ(SP_t)]
    E --> F[Orthogonal Merging Merge_seq<br/>F(SP_1:t-1) ⊕ Hϕ(SP_t)]
    F --> G[Inject to layer l*<br/>Gen sub-answer sa_t]
    G --> H{More sub-questions?}
    H -->|Yes| B
    H -->|No| I[Gen final answer a]
```

### Key Designs

**1. Mechanism (Continual Merging): Reusing single-hop hypernetworks for multi-hop capability.** The core difficulty of multi-hop is expressing the mapping $F$ from "passage sequences → parameters." Directly training $F$ for varying sequence lengths is expensive. MergePRAG bypasses this by **deriving $F$ from a single-hop hypernetwork $H_\phi$ using a recursive merging operator**: at each hop, cumulative parameters $F(SP_{1:t-1})$ are merged with the new passage parameters $H_\phi(SP_t)$, i.e., $F(SP_{1:t}) = \text{Merge}_{seq}\big(F(SP_{1:t-1}), H_\phi(SP_t)\big)$. Sub-answers are generated using the merged parameters $sa_t = M_{\theta_0 \oplus F(SP_{1:t})}(sq_t)$ without placing passages in the context. Variant MergePRAG+ further combines parameter injection with context passages for synergistic gains.

**2. Gram–Schmidt Orthogonal Merging: Preventing passage expert overwriting.** If new experts are added directly to cumulative ones, directional overlap leads to conflicts. MergePRAG adopts Gram–Schmidt orthogonalization: given cumulative parameters $W_F^{t-1}$, its projection matrix $P_{t-1} = W_F^{t-1}\big((W_F^{t-1})^\top W_F^{t-1}\big)^{-1}(W_F^{t-1})^\top$ is calculated. Only the **component of the new parameter $W_t$ orthogonal to this subspace is added**: $W_F^{t} = W_F^{t-1} + (I - P_{t-1})W_t$. This ensures new passages only contribute "novel knowledge," reducing redundancy while accumulating across hops.

**3. Critical-layer KV-Memory: Efficient injection in the most impactful layer.** Instead of injecting into all layers, MergePRAG locates and updates a **single critical layer $l^*$** (similar to ROME/PMET). The hypernetwork generates $k$ pairs of KV vectors $\{K_p, V_p\}$ as "compressed passage memories," inserted into the FFN via **Memory Attention**: using the base FFN output $\text{MLP}_{\theta_0}(x)$ as the query, standard attention is performed on $(K_p, V_p)$ to get $E_{H_\phi(p)}(x)=\text{Attention}(\text{MLP}_{\theta_0}(x), K_p, V_p)$, added as a residual: $\text{MLP}_{\theta_0 \oplus H_\phi(p)}(x) = \text{MLP}_{\theta_0}(x) + E_{H_\phi(p)}(x)$.

**4. Sequence-to-Memory Hypernetwork Architecture and Training.** $H_\phi$ transforms passage token sequences into KV memories using attention pooling for passage embeddings $\text{Emd}(p)$, followed by a 2-layer MLP to obtain latent representation $h_b$. Two linear projections then map $h_b$ to $K_p$ and $V_p$. Training uses cross-entropy $L_{CE}(\phi) = -\sum_{(q,p,a)} \log P_{M_{\theta_0 \oplus H_\phi(p)}}(a \mid q)$. The sub-question generator $M_{sq}$ is trained autoregressively on the sequence $[sq_1, sa_1, \dots, \langle\text{EOS}\rangle]$.

## Key Experimental Results

### Main Results
Performance of LLaMA3.1-8B / Qwen2.5-7B across three multi-hop QA datasets (EM/F1, excerpt at |SP|=4):

| Model | Method | HotpotQA EM/F1 | 2WikiMhQA EM/F1 | MuSiQue EM/F1 |
|------|------|----------------|-----------------|---------------|
| LLaMA3.1-8B | RAG-CoT (E5) | 43.7 / 50.4 | 36.2 / 40.0 | 5.9 / 12.5 |
| LLaMA3.1-8B | R3-RAG† (E5) | 45.6 / 58.8 | 52.9 / 60.9 | **21.2** / **32.8** |
| LLaMA3.1-8B | **MergePRAG+** (E5) | **52.4** / **60.7** | **73.2** / **79.3** | 16.7 / 27.7 |
| Qwen2.5-7B | R3-RAG† (E5) | 46.4 / 59.7 | 54.2 / 62.7 | **21.4** / **34.0** |
| Qwen2.5-7B | **MergePRAG+** (E5) | **50.8** / **58.4** | **77.4** / **81.5** | 12.3 / 21.6 |

Ours leads significantly on HotpotQA and 2WikiMhQA (a 20+ point EM Gain over R3-RAG on 2WikiMhQA), though it lags behind R3-RAG on MuSiQue, which involves longer reasoning chains and RL-based optimization.

### Ablation Study
Key ablations (LLaMA3.1-8B, HotpotQA / 2WikiMhQA, EM/F1):

| Setting | HotpotQA | 2WikiMhQA |
|------|----------|-----------|
| MultihopRAG (No Hypernet, No FT) | 37.8 / 47.6 | 23.3 / 35.6 |
| MultihopRAG FT on [sq→sa] | 43.7 / 50.2 | 58.1 / 62.6 |
| MultihopRAG FT on [(P,sq)→sa] | 40.1 / 46.8 | 60.3 / 62.0 |
| MergePRAG (|SP|=0) | 28.4 / 35.5 | 45.6 / 50.1 |
| **MergePRAG+ (|SP|=1)** | **47.4 / 55.3** | **65.6 / 70.5** |

Comparison of merging methods: Orthogonal merging is approximately **2.4%** higher than TIES-merging and **1% EM** higher than arithmetic averaging.

### Key Findings
- **Hypernetwork injection outperforms direct fine-tuning**: Fine-tuning on $[(P_{gold},sq)\to sa]$ performs worse than $[sq\to sa]$, suggesting vanilla FT hurts generalization; MergePRAG preserves RAG capability while injecting knowledge.
- **Orthogonal merging is the stablest**: It outperforms arithmetic/TIES merging across various |SP| values.
- **Scaling benefits**: Increasing the number of passages |SP| and KV memory size $k$ leads to monotonic performance improvements.

## Highlights & Insights
- **Engineering Philosophy**: Instead of training a complex mapping from passage sequences to parameters, MergePRAG derives multi-hop capability through a recursive operator from a single-hop hypernetwork.
- **Cross-domain transfer**: Migrating de-confliction techniques (orthogonalization) from model merging to RAG knowledge injection provides a clean mathematical solution to "passage expert conflict."
- **Parametric + Context Complementarity**: MergePRAG+ significantly outperforms pure parametric versions, indicating that internal and external knowledge sources provide additive gains.

## Limitations & Future Work
- **Performance on MuSiQue**: On datasets with high hop counts and complex reasoning, RL-based trajectory training (R3-RAG) remains superior, suggesting MergePRAG's reasoning control needs strengthening.
- **Orthogonal Merging Overhead**: Calculating projection matrices (involving matrix inversion) may become a bottleneck as hop count or KV dimensions scale.
- **Single-layer injection**: Focusing on one critical $l^*$ located via perplexity may not be optimal for all tasks; multi-layer injection remains unexplored.

## Related Work & Insights
- **Parametric Knowledge Enhancement**: MergePRAG sits at the intersection of hypernetwork injection, critical layer editing, and external experts.
- **Retrieval-Augmented Generation**: Transitions from classic RAG to PRAG/DyPRAG, and finally to reasoning-centric RAG (FLARE, IRCoT, R3-RAG).
- **Inspiration**: Conflict resolution in model merging (TIES, orthogonalization) can enhance RAG knowledge injection; the "continual parameter accumulation" paradigm could apply to tool-calling or long-term agent memory.

## Rating
- Novelty: ⭐⭐⭐⭐ First to extend PRAG to multi-hop via Gram–Schmidt de-confliction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across models and datasets, though weaker on MuSiQue.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and architecture descriptions.
- Value: ⭐⭐⭐⭐ Provides a reusable de-confliction merging tool and moves Parametric RAG toward reasoning-intensive tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] FrugalRAG: Less is More in RL Finetuning for Multi-hop Question Answering](frugalrag_less_is_more_in_rl_finetuning_for_multi-hop_question_answering.md)
- [\[ICLR 2026\] Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics](demystifying_deep_search_a_holistic_evaluation_with_hint-free_multi-hop_question.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](../../AAAI2026/information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](../../ACL2025/information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[NeurIPS 2025\] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](../../NeurIPS2025/information_retrieval/think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)

</div>

<!-- RELATED:END -->
