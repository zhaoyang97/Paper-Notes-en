---
title: >-
  [Paper Note] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
description: >-
  [ACL 2026][Information Retrieval & RAG][RAG] The authors propose Disco-RAG, which explicitly injects Rhetorical Structure Theory (RST) into the RAG pipeline. It achieves training-free SOTA performance on three long-docum…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "RST"
  - "Discourse Structure"
  - "Rhetorical Graph"
  - "Long-document Reasoning"
date: 2026-05-08
content_hash: 7124856e4edfd6e7
---

# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2601.04377](https://arxiv.org/abs/2601.04377)  
**Code**: https://dongqi.me/projects/Disco-RAG (Available)  
**Area**: Information Retrieval / RAG / Discourse Structure  
**Keywords**: RAG, RST, Discourse Structure, Rhetorical Graph, Long-document Reasoning

## TL;DR
The authors propose Disco-RAG, which explicitly injects Rhetorical Structure Theory (RST) into the RAG pipeline. It achieves training-free SOTA performance on three long-document benchmarks (Loong, ASQA, SciNews) by parsing intra-chunk RST trees (local hierarchy), constructing inter-chunk rhetorical graphs (global coherence), and generating discourse-aware blueprints to guide responses (Loong overall +12.74 LLM Score).

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) is the mainstream solution for connecting external knowledge to LLMs. The standard workflow involves segmenting documents into chunks $\rightarrow$ vectorization for indexing $\rightarrow$ retrieving Top-$k$ chunks at query time $\rightarrow$ concatenating them into the prompt for LLM generation. Structured variants like GraphRAG, RQ-RAG, and StructRAG have emerged, adding structural signals via knowledge graphs, subgraphs, or hierarchical trees.

**Limitations of Prior Work**: Existing RAG systems (including structured variants like GraphRAG) suffer from two neglected discourse-level flaws: (1) **intra-chunk structural blindness**—the rhetorical hierarchy within a chunk (e.g., which sentence is core, which is supplementary, and causal/contrast relations) is not modeled; (2) **inter-chunk coherence gaps**—rhetorical connections between multiple chunks (e.g., a contrast between a conclusion in chunk A and a counter-example in chunk B) cannot be identified.

**Key Challenge**: Consider a counter-example: Chunk A states "study found a 12% lower incidence," and Chunk B states "overall effect is not significant." Standard RAG may not recognize that A is a conditional finding (e.g., only applicable to adults lacking Vitamin D in winter) and could crudely summarize it as "Vitamin D reduces flu risk." The fundamental issue is that **RAG retrieves chunk-level evidence, but generation requires discourse-level reasoning**—there is a gap between "isolated evidence" and a "coherent argumentative chain."

**Goal**: To allow LLMs to perceive both chunks and their intra/inter-chunk rhetorical structures during inference-time (without fine-tuning), and to perform planning before generation based on this structure.

**Key Insight**: Rhetorical Structure Theory (RST, Mann & Thompson 1987/1988) naturally provides nucleus/satellite roles and relation labels like Elaboration, Contrast, and Cause. While previously used for summarization and neural generation models, this work systematically migrates RST to RAG for the first time.

**Core Idea**: Use the LLM to simultaneously act as an RST parser (parsing intra-chunk EDUs and relations), a rhetorical graph constructor (predicting inter-chunk relations), a planner (generating structure-based blueprints), and a generator. These four roles are chained into a pipeline without requiring additional training.

## Method

Disco-RAG is an inference-time strategy that does not modify retriever or generator parameters. The pipeline inserts "discourse modeling + planning" between the standard retrieve and generate stages.

### Overall Architecture

Standard RAG is formalized as $y = \arg\max_{y'} P(y' \mid q, \mathcal{C}(q; \mathcal{D}))$, where $\mathcal{C} = \{c_1, \dots, c_k\}$ represents the Top-$k$ retrieved chunks. Disco-RAG adds three stages:

1.  **Intra-chunk RST tree** $t_i$: For each chunk $c_i$, an LLM-based parser $\mathcal{A}$ performs EDU segmentation, nucleus/satellite role assignment, and relation labeling to obtain the tree $t_i = (V_i, E_i)$. This is done **offline**.
2.  **Inter-chunk rhetorical graph** $\mathcal{G}$: All retrieved chunks are fed to $\mathcal{A}$ in a **listwise** manner to predict rhetorical relations or UNRELATED labels for each pair, forming a directed graph $\mathcal{G} = (\mathcal{C}, \mathcal{F})$.
3.  **Discourse-driven planning blueprint** $\mathcal{B}$: The tuple $(q, \mathcal{C}, \mathcal{T}, \mathcal{G})$ is fed to $\mathcal{A}$ to generate a plan, listing salient content, organization of argumentative flow, and evidence prioritization.

The final generation stage is conditioned on the four-part tuple: $y = \arg\max_{y'} P(y' \mid q, \mathcal{C}, \mathcal{T}, \mathcal{G}, \mathcal{B})$.

### Key Designs

1.  **Intra-chunk RST tree (Local Hierarchy)**:
    *   **Function**: Decomposes the internal structure of each chunk into Elementary Discourse Units (EDUs) and establishes nucleus/satellite roles and relations (e.g., Elaboration/Contrast/Cause) to form an RST tree.
    *   **Mechanism**: The LLM parser $\mathcal{A}$ jointly performs EDU segmentation and relation prediction, formalized as $P(t_i \mid c_i; \theta_\mathcal{A}) = \prod_j P(e_{i_j} \mid c_i; \theta_\mathcal{A}) \cdot \prod_{(u,v)} P(r_{u,v} \mid e_{i_u}, e_{i_v}; \theta_\mathcal{A})$. To save inference costs, trees are pre-parsed **offline**.
    *   **Design Motivation**: Standard RAG treats chunks as bags-of-tokens, losing the distinction between "core conclusions vs. supporting evidence." RST trees explicitly tell the generator which sentence is the nucleus and which is a satellite (e.g., a condition), preventing the model from being misled by satellite information.

2.  **Inter-chunk rhetorical graph (Global Coherence)**:
    *   **Function**: Establishes directed rhetorical connections between retrieved chunks, labeling relations or marking them as UNRELATED.
    *   **Mechanism**: Uses **listwise** reasoning—all $k$ chunks are fed to $\mathcal{A}$ together to jointly predict relations for $k(k-1)$ ordered pairs: $P(\mathcal{G} \mid \mathcal{C}) = \prod_{i=1}^k \prod_{j \ne i} P(r_{i,j} \mid \mathcal{C})$. Allowing UNRELATED enables the model to prune irrelevant connections.
    *   **Design Motivation**: Compared to pairwise reasoning, listwise reasoning allows the parser to see the global context, making it easier to identify relations like "A is a counter-example to B" which require multi-way comparison. This provides the "argumentative" structure lacking in entity-edge graphs like GraphRAG.

3.  **Discourse-aware planning blueprint**:
    *   **Function**: Outputs a plan before generation, defining the sequence of points, supporting evidence, and handling of conflicting evidence.
    *   **Mechanism**: Passes $(q, \mathcal{C}, \mathcal{T}, \mathcal{G})$ to $\mathcal{A}$ to produce a dynamic blueprint $\mathcal{B}$. This plan is neither purely extractive nor free-form but consists of "discourse-aware reasoning steps" organized by rhetorical structure.
    *   **Design Motivation**: Decouples high-level decisions (selection, ordering) from low-level decisions (wording, connectors). Unlike generic planning, discourse-aware planning utilizes RST relations to decide whether to present a nucleus before a satellite or how to frame a contrast.

### Loss & Training
**Completely training-free**. All four LLM roles (parser / graph constructor / planner / generator) share the same base model (Llama-3.1-8B, Llama-3.3-70B, or Qwen2.5-72B). The retriever uses Qwen3-Embedding-8B, chunk size = 256 tokens (no sliding window), Top-10 retrieval, and beam search width = 3. Modules are driven by prompts (full templates provided in the appendix).

## Key Experimental Results

### Main Results: 3 Long-Document Benchmarks (Selection)

**Loong** (4 length categories, 10K $\rightarrow$ 250K tokens; 4 task types) overall performance:

| Length Set | Method | Backbone | LLM Score↑ | EM↑ |
| :--- | :--- | :--- | :--- | :--- |
| Set 1 (10K-50K) | Standard RAG | Llama-3.3-70B | 62.78 | 0.34 |
| Set 1 | StructRAG (prev SOTA) | – | 69.43 | 0.35 |
| Set 1 | **Ours** | Llama-3.3-70B | **71.00** | **0.38** |
| Set 2 (50K-100K) | Standard RAG | Llama-3.3-70B | 53.77 | 0.18 |
| Set 2 | **Ours** | Llama-3.3-70B | **63.61** | **0.28** |
| Set 4 (200K-250K) | Standard RAG | Llama-3.3-70B | 35.61 | 0.07 |
| Set 4 | StructRAG | – | 51.42 | 0.10 |
| Set 4 | **Ours** | Llama-3.3-70B | **54.62** | **0.11** |

**ASQA**: Disco-RAG (Llama-3.3-70B) achieved EM=42.0 / RL=42.3 / DR=32.8, outperforming MAIN-RAG-Llama3-8B (39.2 / 42.0 / —) and Tree of Clarifications (— / 39.7 / 36.6) across all dimensions.

**SciNews**: Disco-RAG (Llama-3.3-70B) achieved RL=21.11 / BERTScore=65.67 / SARI=44.37 / SummaC=69.48, exceeding RSTformer (20.12 / 62.80 / 41.56 / —) and Plan-Input (— / 65.32 / — / 72.40) in most metrics.

### Ablation Study (Loong benchmark, Llama-3.3-70B)

| Method | Overall LLM Score | Overall EM | Description |
| :--- | :--- | :--- | :--- |
| **Disco-RAG (full)** | **62.07** | **0.24** | All three modules included |
| w/o RST tree | 56.22 | 0.20 | Remove intra-chunk tree $\rightarrow$ -5.85 Gain |
| w/o rhetorical graph | 57.10 | 0.21 | Remove inter-chunk graph $\rightarrow$ -4.97 Gain |
| w/o planning | 59.75 | 0.22 | Remove planner $\rightarrow$ -2.32 Gain |
| Standard RAG | 49.33 | 0.17 | Baseline |
| w/ retrieve-and-plan | 50.64 | 0.18 | Standard RAG + free-form plan (no structure) |
| w/ plan-and-retrieve | 51.38 | 0.18 | Plan before retrieve (no structure) |

Generic planning only improves Standard RAG by 1.3–2.0 points, while discourse-aware planning gains 12+ points, proving structural priors are irreplaceable.

### Key Findings
- **Structural Modules > Planner Contribution**: RST trees and rhetorical graphs contribute ~5 points each, while the planner contributes ~2 points. Structure is the foundation; planning is the amplifier.
- **Larger Gains for Longer Documents**: In Set 1 (shortest), Disco-RAG outperforms Standard RAG by 8.22 points; in Set 4 (200K+ tokens), the gap widens to 19 points. Discourse-awareness is crucial for long documents where rhetorical scaffolds connect scattered evidence.
- **Robustness to Retrieval Noise**: Replacing 20–40% of Top-10 chunks with irrelevant ones caused Standard RAG to drop from 49.33 to 45.23, while Disco-RAG maintained 56.17. Rhetorical structure helps the generator identify UNRELATED segments.
- **Structural Perturbation Experiments**: Shuffling RST relation labels dropped scores from 62.07 to 55.48; flipping edge directions dropped it to 55.82; shuffling plan steps dropped it to 57.50. All still outperformed Standard RAG (49.33), indicating performance stems from the structural signal itself rather than just extra tokens.
- **Hybrid Model Deployment**: An 8B parser with a 70B generator achieved 60.52, close to the all-70B performance (62.07) and far exceeding Standard RAG (49.33). This suggests structural modules can be offloaded to smaller models.
- **Orthogonality with SFT**: After fine-tuning the generator on SciNews, adding discourse input increased RL from 22.8 to 23.3 and SummaC from 72.3 to 74.0, showing discourse signals complement parametric learning.
- **Human Evaluation**: Based on a 3-point Likert scale by PhD students, Disco-RAG scored 2.53 on Faithfulness vs. 1.67 for Standard RAG, nearly reaching the human reference score of 2.88.

## Highlights & Insights
- **Discourse is the missing piece of the RAG puzzle**: While variants like GraphRAG focus on entity-level KGs, this work zooms out to argument-level discourse. Capturing causal, contrastive, and elaboration relations addresses the level where LLMs most frequently fail during multi-document synthesis.
- **Listwise inter-chunk relation prediction**: Allowing the LLM to see all chunks before deciding relations enables global context usage for precise rhetorical inference. This listwise trick is transferable to tasks like conflict detection in multi-document summarization.
- **Decoupled modules with model reuse**: Using the same LLM for all roles by only changing prompts is engineering-simple and flexible. Hybrid experiments show potential for cost savings.
- **Structural ablation superior to no structure**: Even perturbed structural signals yielded better results than Standard RAG, suggesting the discourse-aware framework's robustness stems from "directing the generator to focus on structure" itself.
- **Doubled gains in long-document scenarios**: The jump from +8 in Set 1 to +19 in Set 4 proves discourse modeling is a critical bottleneck for RAG in long-context environments.

## Limitations & Future Work
- **Extra LLM calls increase latency/tokens**: RST parsing, graph construction, and planning each require an LLM call, making it 3-4x slower than standard RAG. Latency-sensitive scenarios need structure caching or distilled parsers.
- **Dependence on backbone discourse capability**: On smaller models (e.g., Llama-3.1-8B), parser quality is lower. The performance of an all-8B version (58.94) is lower than the all-70B version (62.07).
- **Narrow dataset scope**: Evaluation is limited to English academic/encyclopedic documents. Applicability to other languages or domains (code, math, dialogue) is unknown.
- **Fixed rhetorical relation set**: Only classic RST relations were used. Specialized domains (legal/medical) might require extended label sets.
- **Lacks joint entity-level + discourse-level comparison**: The authors did not attempt to fuse GraphRAG's entity graphs with Disco-RAG's rhetorical graphs, which remains a high-potential area.

## Related Work & Insights
- **vs. GraphRAG (Edge et al. 2024) / KG-RAG**: GraphRAG uses entity-level KGs to organize evidence by entity co-occurrence; Disco-RAG uses argument-level RST to organize by rhetorical relations. Entity-level solves "what is mentioned," whereas discourse-level solves "what is being argued." The 30+ point lead over GraphRAG on Loong highlights discourse as the more urgent bottleneck.
- **vs. StructRAG (Li et al. 2025b)**: StructRAG uses dynamic structural formats (table/tree/graph) and is a training-time method; Disco-RAG is training-free and slightly outperforms it (71.00 vs 69.43 in Set 1). This suggests the *type* of structural signal (rhetorical vs. general) is more important than the *form*.
- **vs. RST-LoRA (Liu & Demberg 2024) / RSTformer (Liu et al. 2024)**: These inject RST into model parameters; Disco-RAG injects it into the prompt (inference-time), making it easier to migrate to frozen LLMs.
- **vs. Tree of Clarifications / RQ-RAG / MAIN-RAG**: These focus on query refinement; Disco-RAG focuses on the structure of retrieved evidence. These approaches are complementary.
- **vs. FLARE (Jiang et al. 2023)**: FLARE handles active retrieval; Disco-RAG handles post-retrieval structural enhancement. They could be combined to decide when to retrieve and then how to parse the structure.

## Rating
- Novelty: ⭐⭐⭐⭐ (RST for RAG is a novel combination, though both individual components have prior work)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive benchmarks, multiple backbones, ablation, perturbation, human eval)
- Writing Quality: ⭐⭐⭐⭐ (Clear pipeline formulation, intuitive motivation, full prompts provided)
- Value: ⭐⭐⭐⭐⭐ (Significant training-free gains for long-document RAG, modular and engineering-friendly)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
