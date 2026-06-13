---
title: >-
  [Paper Note] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG
description: >-
  [ICML 2026][LLM Safety][on-device RAG] EPIC shifts the core bottleneck of on-device RAG from "how to use preferences during retrieval" to "what to store during indexing." By employing a three-stage pipeline consisting of…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "on-device RAG"
  - "user preference"
  - "memory constraints"
  - "instruction-based indexing"
  - "query steering"
date: 2026-05-08
content_hash: be29a321a09e98fe
---

# From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG

**Conference**: ICML 2026  
**arXiv**: [2605.18271](https://arxiv.org/abs/2605.18271)  
**Code**: https://github.com/UbiquitousAILab/EPIC (Available)  
**Area**: Information Retrieval / On-device RAG / Personalization  
**Keywords**: on-device RAG, user preference, memory constraints, instruction-based indexing, query steering

## TL;DR
EPIC shifts the core bottleneck of on-device RAG from "how to use preferences during retrieval" to "what to store during indexing." By employing a three-stage pipeline consisting of "coarse filtering + fine verification + query steering," it retains only data aligned with user preferences and generates "instruction-item" pairs as indexing units. This approach reduces storage by 2404× across four preference benchmarks while achieving an absolute improvement of 20.17 percentage points in preference alignment accuracy.

## Background & Motivation

**Background**: Personal AI agents centered on LLMs are increasingly deployed on phones and edge devices to ensure privacy and response speed. These agents need to ground generation on local personal corpora (browsing history, conversations, notifications, Wiki, etc.). RAG is the mainstream solution for integrating external knowledge. Significant work already exists in personalized RAG: organizing user content/profiles into graphs on the memory side (EMG-RAG, PEARL) or using preference rewriting and expansion on the query side (Cognitive Personalized Search, PBR).

**Limitations of Prior Work**: When moving to on-device scenarios, storage and power consumption become hard constraints. Existing works generally assume that the corpus has been "carefully curated," leaving only the problem of how to use it during querying. However, in on-device scenarios, raw data is heterogeneous and continuously growing (Wikipedia, Common Crawl, dialogue streams, notifications...). Indexing all data indiscriminately is impossible under memory budgets in the 1 MB range; for instance, HippoRAG 2's index can reach 2.9 GB. Furthermore, standard retrievers only align query-text similarity and are "preference-agnostic," leading to preference mismatches where "retrieved content is factually correct but violates user preferences" (e.g., recommending sashimi to a user allergic to seafood).

**Key Challenge**: Limited on-device memory vs. infinitely growing personal data; retriever targets centered on query similarity vs. user needs for preference alignment—these two goals are coupled in "what to index and what to match in queries," yet existing methods optimize them separately.

**Goal**: Under tight on-device budgets, this paper addresses the neglected upstream question: "what should be stored?" It also aims to reuse the same preference signal at both the indexing and retrieval ends. Sub-problems include: (i) high-throughput discarding of preference-irrelevant content; (ii) fine-grained semantic verification of potentially relevant content and storing "how to use this data"; (iii) infusing query embeddings with preference semantics with near-zero latency.

**Key Insight**: Among all forms of personal context, **user preferences** ($P$) are the most compact and stable abstractions—tastes, dietary restrictions, and style preferences remain constant across sessions and can be implicitly extracted from dialogue history. By treating the preference set $P = \{p_1, \dots, p_N\}$ as an "indexing-time known" prior, the problem transforms into "selecting a preference-related subset from the raw stream and indexing it in a preference-aligned manner given $P$."

**Core Idea**: Reframing "personalized RAG" from a query-side task into a **memory-construction problem**. Preferences are integrated throughout the pipeline: (1) high-recall coarse filtering using embedding similarity → (2) fine-grained verification using LMs and synthesizing "anchor instructions" as indexing units → (3) steering the query towards the most relevant preferences during retrieval. This ensures the pipeline stores less while retrieving more accurately.

## Method

### Overall Architecture
The input consists of a growing set of candidate items $\mathcal{D}$ (Wiki passages, dialogue fragments, web pages, etc.) and a set of user preferences $P = \{p_1, \dots, p_N\}$. The output is a compact preference-aligned memory $\mathcal{M}$, where each record is formatted as $(x, i_x(p), p, E(i_x(p)))$, representing "original item + preference-conditional instruction + corresponding preference + instruction embedding." At runtime, given a query $q$, it is first steered toward the most relevant preference to obtain $\tilde q$. Then, a FAISS kNN search is performed on the instruction embeddings in $\mathcal{M}$. The retrieved instructions point to their mounted original items, which are provided as context to the generator for preference-aligned output. The pipeline restricts LM calls to a small subset (triggering fine verification only 0.22 times per incoming data entry on average).

### Key Designs

1.  **Semantic-Based Coarse Filtering (Function, Mechanism, Design Motivation)**:
    *   **Function**: Directly discards the vast majority of candidate items unrelated to any preference without invoking any LM, passing only a small potentially relevant subset $\mathcal{D}_{coarse}$ to the next stage.
    *   **Mechanism**: A shared sentence encoder (e.g., Contriever) encodes item $x$ and each preference $p$ into $\ell_2$-normalized vectors $E(\cdot) \in \mathbb{R}^d$. For each $x$, the cosine similarity $\mathrm{Sim}(E(x), E(p))$ is calculated. Preferences exceeding a threshold $\tau$ are collected as $P_{rel}(x) = \{p \in P \mid \mathrm{Sim}(E(x), E(p)) \ge \tau\}$. If $P_{rel}(x)$ is non-empty, $x$ is retained ($\mathcal{D}_{coarse} = \{x \mid P_{rel}(x) \neq \emptyset\}$).
    *   **Design Motivation**: Over 99% of raw stream content is unrelated to current user preferences; LM verification is too expensive. Using pure embedding geometry for high-recall filtering compresses storage (3.95×–77.68× in ablations) and ensures low indexing latency (102.67 ms per entry on-device).

2.  **Preference-Aligned Fine Verification (Function, Mechanism, Design Motivation)**:
    *   **Function**: Performs second-stage linguistic confirmation for candidates passing coarse filtering, discarding "vector-near but semantically irrelevant" noise, and synthesizing a preference-conditional instruction as the final indexing unit.
    *   **Mechanism**: Consists of two LM modules. The Decision Module (DM) outputs a structured $(\mathrm{Decision}, \mathrm{Rationale}, P'_{rel}(x))$ for $(x, P_{rel}(x))$, where Decision $\in \{\langle \text{Keep}\rangle, \langle \text{Discard}\rangle\}$. Only "Keep" items enter $\mathcal{D}_{fine}$. The Instruction Generator (IG) takes $(x, p, \mathrm{Rationale})$ as input to generate a preference-conditional instruction $i_x(p) = \mathrm{IG}(x, p, \mathrm{Rationale})$ for each $p \in P'_{rel}(x)$. The final memory is $\mathcal{M} = \{(x, i_x(p), p, E(i_x(p))) \mid x \in \mathcal{D}_{fine}, p \in P'_{rel}(x)\}$. **FAISS indexing is built on $E(i_x(p))$ instead of $E(x)$**.
    *   **Design Motivation**: Embedding similarity is only an approximation; nuanced trade-offs (e.g., "dislike seafood" vs. "sushi menu") require explicit LM judgment. Reusing the Rationale as input for instruction generation solidifies the "why this data is relevant" into the index layer. Indexing instructions rather than raw text upgrades retrieval from "finding semantically similar facts" to "finding facts matching preference usage."

3.  **Preference-Guided Query Steering (Function, Mechanism, Design Motivation)**:
    *   **Function**: Offsets the raw query embedding towards the most relevant preference, placing it in the same semantic subspace as the preference-conditioned instruction embeddings, thereby improving top-k retrieval alignment with negligible latency.
    *   **Mechanism**: For an incoming query $q$, the most relevant preference is selected: $p^* = \arg\max_{p \in P} \mathrm{Sim}(E(q), E(p))$. The steered query is calculated as $\tilde q = (E(q) + E(p^*)) / \|E(q) + E(p^*)\|$. $\tilde q$ is used for kNN search on the FAISS instruction index.
    *   **Design Motivation**: Even if the index side encodes preferences, the query side remains preference-agnostic, creating an alignment gap. Per-query LM rewriting (as in Pref-QR / PBR) is slow. Using preference vector translation is a zero-parameter, zero-LM-call solution that adds only 0.18 ms per query on Jetson Orin Nano.

### Loss & Training
EPIC is a **pure inference-time pipeline** that introduces no new training objectives or parameters. The shared encoder uses off-the-shelf models like Contriever, while DM/IG utilize existing LLMs (experimentally Qwen3-4B / Llama-3.1-8B / gpt-oss-20b, with LLaMA-3.3-70B-Instruct as the judge). It is plug-and-play for any retriever or LLM backend.

## Key Experimental Results

### Main Results
Preference-following Accuracy (%) across four benchmarks, with LLaMA-3.3-70B-Instruct as the judge. Results for Llama-3.1-8B-Instruct and gpt-oss-20b backends:

| Backend | Method | PrefWiki | PrefRQ | PrefELI5 | PrefEval |
|:---|:---|:---|:---|:---|:---|
| Llama-3.1-8B | BM25 | 38.56 | 64.22 | 69.04 | 27.97 |
| Llama-3.1-8B | NV-Embed-v2 (Prev. SOTA) | 44.53 | 70.22 | 69.86 | 30.88 |
| Llama-3.1-8B | HippoRAG 2 | 42.91 | 66.00 | 69.73 | 32.98 |
| Llama-3.1-8B | Pref-QR | 39.72 | 71.33 | 69.59 | 34.21 |
| Llama-3.1-8B | **EPIC** | **54.07** | **83.00** | **87.95** | **65.61** |
| gpt-oss-20b | NV-Embed-v2 | 42.29 | 86.56 | 77.01 | 29.28 |
| gpt-oss-20b | **EPIC** | **73.26** | **93.89** | **87.61** | **77.96** |

EPIC achieves an average absolute **Gain of +20.17%p** relative to NV-Embed-v2, with dramatic improvements on PrefEval (29.28→77.96).

Efficiency comparison (Figure 3):

| Dimension | NV-Embed-v2 | HippoRAG 2 | PBR | **EPIC** | Advantage vs. Baseline |
|:---|:---|:---|:---|:---|:---|
| Storage (MB) | 648 | 2896 | 286 | **0.27** | **~2404× smaller** |
| Retrieval Latency (ms) | 100 | 812 | 10073 | **3** | **~33× faster** |
| Indexing Latency (s) | 1918 | 4726 | 654 | 246 | Comparable to lightweight |

On-device performance (Jetson Orin Nano 8GB): Total retrieval latency is 29.35 ms/query (steering: 0.18 ms, FAISS: 0.14 ms, bottleneck: query embedding at 29.03 ms). Memory residency is < 1 MB.

### Ablation Study

| Configuration | Accuracy Gain (vs. baseline) | Storage Compression | Note |
|:---|:---|:---|:---|
| C only | Minimal | **3.95×–77.68×** | Main source of memory saving; accuracy is unstable. |
| C + F | **+13.22 to +33.69 %p** | Further ×1.95–8.93 | Main source of accuracy; instruction-centric memory. |
| C + F + S | Additional **+0.78 to +4.03 %p** | Unchanged | Zero extra storage; utilizes encoded preference signals. |

### Key Findings
*   **Clear module contribution profiles**: C handles "store less," F handles "retrieve accurately," and S provides a "free extra boost."
*   **Strongest performance on PrefEval**: While preference-agnostic methods hover around 30%, EPIC reaches 65–78%, showing maximum benefit in high-noise, high-preference-density scenarios.
*   **Robustness under streaming preference drift**: As 5K→40K documents arrive and preferences switch, EPIC's memory remains nearly flat and accuracy remains stable relative to the linear growth of Contriever.
*   **Portability**: Integrating EPIC's components into HippoRAG 2 replicated the same patterns of improvement, proving it is a plug-in memory construction layer rather than an end-to-end solution tied to a specific retriever.

## Highlights & Insights
*   **Paradigm shift to "what to store"**: While most personalized RAG literature focuses on the query side, this paper identifies memory construction as the true upstream bottleneck for on-device budgets.
*   **Indexing instructions instead of raw text**: Traditional indexing places "item semantics" in vector space; however, users often need "usage matching preferences" rather than "similar facts." Indexing anchor instructions upgrades the retrieval target and pre-calculates the "how to use" rationale.
*   **Reusing preference signals at both ends**: Coarse filtering, fine verification, and query steering all consume the same $P$ without training any alignment networks.
*   **Transferable Tricks**: (i) Using $\tilde q = \text{normalize}(E(q) + E(p^*))$ for conditional retrieval can be applied to any attribute-aware retrieval (domain, style, safety). (ii) Reusing DM's Rationale as IG's input is a multi-purpose LLM call pattern applicable to any two-stage "verify then explain" pipeline.

## Limitations & Future Work
*   Assumes preference set $P$ is known during indexing—preference extraction itself is treated as an upstream task.
*   Fixed thresholds $\tau$ and additive query steering do not explicitly model preference conflicts (e.g., "likes spicy" vs. "stomach sensitivity").
*   Accuracy depends heavily on the LM's judgment and instruction generation quality; the trade-off between smaller on-device model quality and indexing cost is not systematically explored.
*   Evaluation relies on the LLM-as-a-judge framework, which may introduce bias.

## Related Work & Insights
*   **vs. HippoRAG 2 / RAPTOR**: These methods focus on better retrieval over a fixed, "pre-selected" index. EPIC priorities "what enters the index," reducing storage by ~10,000× and ~1,000× respectively.
*   **vs. Pref-QR / PBR**: These query-side methods are slow (PBR: 10s/query). EPIC amortizes alignment costs to indexing time, resulting in 3 ms retrieval latency, fitting the "frequent query, sparse inflow" profile of devices.
*   **vs. EMG-RAG / PEARL**: These focus on organizing existing memory. EPIC handles the upstream filtering of raw heterogeneous data and is complementary to these frameworks.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)
- [\[ICML 2026\] Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences](federated_variational_preference_alignment_with_gumbel-softmax_prior_for_persona.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](../../CVPR2026/llm_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](../../ACL2026/llm_safety/leakdojo_decoding_the_leakage_threats_of_rag_systems.md)

</div>

<!-- RELATED:END -->
