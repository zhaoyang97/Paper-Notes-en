---
title: >-
  [Paper Note] Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation
description: >-
  [ACL 2026][Recommender Systems][Cross-cultural recipe adaptation] Authors find that standard RAG produces non-diverse outputs in creative tasks even with diverse contexts. They design CARRIAGE…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Cross-cultural recipe adaptation"
  - "RAG"
  - "Diversity"
  - "MMR"
  - "Contrastive context"
  - "CultureScore"
date: 2026-05-08
content_hash: 789261615a2854c9
---

# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation

**Conference**: ACL 2026  
**arXiv**: [2507.21934](https://arxiv.org/abs/2507.21934)  
**Code**: <https://github.com/TenneyHu/CARRIAGE>  
**Area**: Recommendation / RAG / Cross-cultural Generation  
**Keywords**: Cross-cultural recipe adaptation, RAG, Diversity, MMR, Contrastive context, CultureScore

## TL;DR
Authors find that standard RAG produces non-diverse outputs in creative tasks even with diverse contexts. They design CARRIAGE, a plug-and-play framework incorporating query rewriting, diversity-aware MMR re-ranking, sliding-window dynamic context, and contrastive context injection. This successfully propagates "contextual diversity" to "output diversity," improving lexical/semantic/ingredient diversity and CultureScore in Spanish cross-national recipe adaptation, achieving Pareto efficiency over closed-book LLMs.

## Background & Motivation
**Background**: Cross-cultural recipe adaptation involves rewriting a source culture recipe into a target culture version, preserving the essence while adhering to target dietary habits. Existing works (Cao et al. 2024; Hu et al. 2024; Pandey et al. 2025) treat this as a cross-cultural translation task using prompt-based LLMs or Information Retrieval (CARROT) to fetch authentic target recipes.

**Limitations of Prior Work**: Previous work focused solely on "high quality," with almost no attention to "diversity." In reality, adapting a Mexican Nopal dish to a Spanish kitchen allows for multiple valid substitutions (spinach, asparagus, green beans, etc.). Users have diverse dietary preferences, so a system should provide multiple reasonable outputs for a single input. However, applying RAG to this "multi-valid-answer" creative task revealed an unexpected finding: RAG produces less diverse outputs than closed-book LLMs even when fed diverse contexts.

**Key Challenge**: Classical RAG design assumes a 1-to-1 factual mapping between "context → output," but creative tasks require 1-to-many mappings. LLMs often fixate on a single context segment (lost-in-the-middle), and retrievers repeatedly return the same results for the same query, leading to a double squeeze on output diversity.

**Goal**: (1) Verify whether standard RAG can produce diverse outputs under diverse contexts; (2) Design automated metrics balancing quality and diversity; (3) Design a plug-and-play RAG framework that truly propagates contextual diversity to output diversity.

**Key Insight**: Decompose the diversity squeeze into four independent failure points: C1 Retrieval omission due to cultural differences; C2 Diversity-unaware ranking; C3 Limited context utilization by the LLM; C4 Generation phase unawareness of diversity. Each C corresponds to a lightweight component.

**Core Idea**: Construct CARRIAGE (Cultural-Aware Recipe Retrieval Augmented GEneration): query rewriting + historical-aware MMR re-ranking + sliding-window dynamic context organization + contrastive context injection. Each step addresses one C, is training-free, and plug-and-play for any LLM.

## Method

### Overall Architecture
Given a source recipe $q$, CARRIAGE executes four stages sequentially:

1.  **Query Rewriting**: Uses an LLM to rewrite the source recipe title (one based on content regeneration, one adapted for the target culture). Three queries are retrieved in parallel.
2.  **Diversity-aware Re-ranking (Historical-MMR)**: Uses BGE-M3 for relevance scoring and an extended MMR to select top-$k$.
3.  **Dynamic Context Organization**: For each generation round $t$ of the same query, a sliding window selects $w$ documents from the $k$ retrieved contexts. This ensures the "input" actually changes between rounds.
4.  **Contrastive Context Injection**: Injects previously generated outputs for the same source into the prompt, explicitly instructing the LLM "not to be similar to these."

Output: For each source recipe, $K=5$ adaptation candidates are produced, which can be used for per-input diversity evaluation or user selection. The entire pipeline is inference-time only.

### Key Designs

1.  **Query Rewriting + Historical-MMR Re-ranking (Addressing C1+C2)**:
    *   **Function**: Simultaneously addresses "retrieval omission due to cultural differences" and "relevance-only ranking," ensuring the retrieval pool is both broad and non-redundant.
    *   **Mechanism**: Query rewriting generates two title variants (content-based and target-culture-adapted). The classical MMR is extended to consider both the "currently selected set $S$" and the "historical RAG output set $H$." The scoring function is:
        $$\text{Score}(D_i) = \max_{D_i \in R \setminus S} [\lambda \cdot \text{Rel}(D_i) - (1-\lambda) \cdot \max_{D_j \in S \cup H} \text{Sim}(D_i, D_j)]$$
        where $\lambda=0.6$. Including historical outputs in the MMR similarity term is a key innovation, exerting "pressure to move away from the past" across multiple generations.
    *   **Design Motivation**: Pure MMR only ensures top-k within one retrieval are non-redundant. Across multiple generation rounds, it still tends to promote the same set of documents. Integrating the history term penalizes recipes used in previous rounds, forcing the retrieval of new cultural variations.

2.  **Dynamic Context Organization (Addressing C3, propagating contextual diversity to input-level diversity)**:
    *   **Function**: Solves the failure where LLMs receive diverse contexts but only copy one segment, by ensuring the contexts seen in each round are truly different.
    *   **Mechanism**: From $k$ contexts $\mathcal{C} = \{D_1, \ldots, D_k\}$, the $t$-th generation round uses a sliding window to view $w$ items: $\mathcal{C}_{\text{reference}}^{(t)} = \{D_{tw+1}, \ldots, D_{(t+1)w}\}$. Default $k=5, w=1$.
    *   **Design Motivation**: Probing experiments (Table 2) showed Vanilla RAG depends on the same 1-2 contexts in ~76% of 5 generations. CARRIAGE's sliding window increases the average context switching rate from 1.78 to 2.67 (a >40% improvement), proving that "forced partitioning at the input level" is more effective than letting the LLM decide.

3.  **Contrastive Context Injection (Addressing C4, injecting diversity preference into generation)**:
    *   **Function**: Provides an explicit signal to the LLM about what has already been generated, seeking differentiation in the generation layer.
    *   **Mechanism**: Previous outputs for the same source recipe are extracted and placed in the prompt with an explicit requirement to "avoid generating similar results." This achieves diversity-promoting decoding at the prompt level without modifying the model or temperature.
    *   **Design Motivation**: Post-trained LLM output distributions are naturally sharp; contrastive context provides an explicit push away from existing outputs.

### Loss & Training
The framework is completely training-free. Key hyperparameters: temperature=0.7, $k=5$ (retrieved documents), $w=1$ (window size), $\lambda=0.6$ (MMR weight), $K=5$ (generation rounds per input). Retrieval uses JINA-ES dense vectors; re-ranking uses BGE-M3. Base LLMs include LLaMA-3.1-8B and Qwen-2.5-7B.

## Key Experimental Results

### Main Results
Dataset: RecetasDeLaAbuel@ (Spanish recipe set), 500 source recipes (7 Latin American countries) → adapted to Spanish style. Retrieval pool: 9381 Spanish recipes.

| Category | Method | Lexical↑ | Ingredient↑ | Semantic↑ | CultureScore↑ | BERTScore↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Closed-book | Llama3.1-8B | 0.557 | 0.667 | 0.232 | 0.451 | 0.404 |
| Closed-book | Qwen2.5-7B | 0.551 | 0.531 | 0.247 | 0.404 | 0.439 |
| IR | CARROT-MMR | 0.741 | 0.941 | 0.527 | 0.503 | 0.298 |
| RAG | Vanilla-LLaMA RAG | 0.518 | 0.748 | 0.155 | 0.383 | 0.551 |
| **Ours** | **CARRIAGE–LLaMA** | **0.577** | 0.739 | **0.269** | **0.463** | 0.442 |
| **Ours** | **CARRIAGE–Qwen** | **0.628** | 0.676 | **0.303** | **0.590** | 0.342 |

**Key Findings**: (1) Vanilla RAG has the lowest semantic diversity (0.155), confirming diversity collapse; (2) CARRIAGE-LLaMA Pareto dominates LLaMA closed-book across lexical, semantic, and CultureScore metrics; (3) CARRIAGE-Qwen achieves the highest CultureScore (0.590) and high semantic diversity.

### Ablation Study
Contextual diversity utilization (distribution of dominant context switches in 5 generations):

| Method | #1 (Always same) | #3 | #5 (New every round) | Avg Switches |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla RAG | 204 | 78 | 0 | 1.78 |
| CARROT-MMR RAG | 180 | 108 | 0 | 1.90 |
| **CARRIAGE RAG** | **40** | **202** | **13** | **2.67** |

CARRIAGE increases average switches by >40%, proving input-side windowing is the critical component.

## Highlights & Insights
*   **Diagnostic Problem Decomposition**: Breaking down "lack of diversity" into C1-C4 allows for modular solutions and clear methodology.
*   **Historical-MMR**: A simple but effective extension from single-query to session-level diversity, applicable to any "multiple query per user" scenario.
*   **Dynamic Context Organization**: Reveals a "creative version" of lost-in-the-middle where LLMs selectively consume context. Prompt-level windowing is a cheaper, more stable engineering solution than modified decoding.
*   **CultureScore**: Using the probability from a country classifier as a metric bypasses linguistic clues and targets cultural features (ingredients, flavors). It serves as a reliable proxy ($κ=0.59$ with humans).

## Limitations & Future Work
*   **Limitations**: Focused on Spanish-speaking countries; limited to 7-9B open-source models; per-input diversity improved but across-input (global) diversity actually decreased (heavy-tail ingredients still ignored).
*   **Future Work**: Improving global diversity by penalizing high-frequency ingredients in prompts; testing higher cultural-gap language pairs (e.g., Arabic-English); exploring adaptive window sizes.

## Related Work & Insights
*   **vs. CARROT (Hu et al. 2024)**: CARROT provides retrieval results; CARRIAGE builds on this with RAG and 4 diversity components to propagate retrieval diversity to output diversity.
*   **vs. Lost-in-the-middle (Liu et al. 2023)**: While LiM focuses on position-based performance, CARRIAGE focuses on the failure of uniform context utilization in creative tasks.
*   **Insight**: Propagating contextual diversity to output diversity is a universal challenge for creative RAG. CARRIAGE's "input windowing + contrastive prompt" is a cost-effective general solution.

## Rating
*   Novelty: ⭐⭐⭐⭐ First work to explicitly target "diversity" in RAG; clear C1-C4 methodology.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive baselines and probing; lacks human quality evaluation for recipes.
*   Writing Quality: ⭐⭐⭐⭐ Clear narrative progression and intuitive visualizations.
*   Value: ⭐⭐⭐⭐ Highly practical plug-and-play tool for creative RAG tasks (dialogue, marketing, recipes).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](../../AAAI2026/recommender/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
