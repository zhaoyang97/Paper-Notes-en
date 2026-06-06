---
title: >-
  [Paper Note] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment
description: >-
  [ACL 2026][Recommender Systems][Sequential Recommendation] HSUGA addresses two core components of LLM-enhanced sequential recommendation by introducing a "staged + four atomic edits (Add/Delete/Update/Retain)" HSU module…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "LLM Semantic Embeddings"
  - "Staged Preference Evolution"
  - "Editing Operations"
  - "Group-aware Self-distillation"
date: 2026-05-08
content_hash: 4c57f5f1d263a279
---

# HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment

**Conference**: ACL 2026  
**arXiv**: [2605.11662](https://arxiv.org/abs/2605.11662)  
**Code**: HSUGA (GitHub, name provided but link not given in the original PDF)  
**Area**: Recommender Systems / LLM-Enhanced Sequential Recommendation / Long-tail Users  
**Keywords**: Sequential Recommendation, LLM Semantic Embeddings, Staged Preference Evolution, Editing Operations, Group-aware Self-distillation

## TL;DR
HSUGA addresses two core components of LLM-enhanced sequential recommendation by introducing a "staged + four atomic edits (Add/Delete/Update/Retain)" HSU module for robust semantic extraction from long interaction sequences. It further employs a GAA self-distillation alignment strategy grouped by activity (20% Head / 80% Long-tail) to resolve under-supervision for long-tail users and over-alignment for active users. As a plug-and-play solution, it achieves performance gains across Steam/Fashion/Beauty datasets using GRU4Rec/BERT4Rec/SASRec backbones.

## Background & Motivation

**Background**: The mainstream paradigm for LLM-enhanced sequential recommendation (LLM4SR) involves two steps: (1) using an LLM to summarize user interaction histories into semantic vectors (embedding extraction); (2) integrating these vectors back into traditional sequential encoders (GRU4Rec / BERT4Rec / SASRec) for semantic utilization (embedding utilization). Representative methods include LLMEmb, LLM2Rec, LLM-ESR, RLMRec, and LLMInit.

**Limitations of Prior Work**: Structural defects exist at both ends. Extraction: Existing methods feed the entire long sequence into the LLM for a one-time summary, but excessive context triggers "lost-in-the-middle" issues, resulting in unstable user vectors, particularly for long-tail users. Utilization: All users are subjected to a uniform alignment/retrieval strategy, ignoring differences in activity levels—active users already have dense representations, and forced alignment with neighbors may introduce noise, while long-tail users have sparse representations where weak alignment is insufficient.

**Key Challenge**: The reliability of long-sequence reasoning vs. the simplicity of one-time summarization; uniform alignment strategies vs. user activity heterogeneity. A one-size-fits-all approach leads to either noise or underfitting.

**Goal**: To decouple these issues into two plug-and-play plugins that treat both stages respectively while maintaining backbone independence, allowing them to be applied to any LLM4SR method.

**Key Insight**: The authors draw inspiration from CoT and editable LLM memory. By segmenting long sequences into fixed-length stages and mandating a two-step process—"Operation Selection (Add/Delete/Update/Retain) followed by Execution"—within each stage, semantic updates are constrained to a discrete action space, avoiding open-ended semantic drift and error accumulation.

**Core Idea**: Use "atomic editing operations + staged updates" instead of "one-time summarization" for semantic extraction (HSU), and "activity-based grouping + adaptive neighbor count + Pearson percentile filtering for active users" instead of "uniform neighbor alignment" for semantic utilization (GAA). Both are implemented as plugins.

## Method

### Overall Architecture
The input is a complete historical interaction sequence $\mathcal{S}_u = [i_1, \dots, i_T]$ for user $u$, and the output is a Top-K recommendation list. The framework includes:

1.  **HSU Extraction Phase**: $\mathcal{S}_u$ is divided into several stages of fixed length $L$ (e.g., 13). Starting from stage 1, the LLM summarizes initial interest descriptions. From stage 2 onwards, each stage follows two steps: selecting an editing action from {Add, Delete, Update, Retain}, and then executing it to obtain the next-stage interest text. Finally, the converged interest text is vectorized via LLM hidden layers or an external embedding model to obtain $\mathbf{s}_u$.
2.  **GAA Utilization Phase**: Users are sorted by interaction count $n_u$, with the top 20% labeled as active and the remaining 80% as long-tail. Candidate neighbor sets $N_u^{(g)}$ are retrieved based on the cosine similarity of $\mathbf{s}_u$. Long-tail users retain all neighbors, while active users undergo Pearson percentile filtering for denoising.
3.  **Training**: The sequential encoder produces $\mathbf{h}_u$, which is fused with $\mathbf{s}_u$ via $\phi(\cdot)$ (addition/gating/concat-projection) to obtain $\tilde{\mathbf{h}}_u$. The score $\hat{y}_{u,j} = \mathbf{e}_j^\top \tilde{\mathbf{h}}_u$ is jointly optimized with the GAA self-distillation loss $\mathcal{L}_{SD}$.

### Key Designs

1.  **Hierarchical Semantic Understanding (HSU) — Staged Two-Step Editing**:
    *   **Function**: Replaces "one-shot summarization" with "incremental updates of interest descriptions by stage," mitigating lost-in-the-middle effects and supporting incremental inference.
    *   **Mechanism**: Within each stage, the LLM is forced through "Operation Selection → Operation Execution." In Selection, the LLM must choose one of {Add, Delete, Update, Retain}. In Execution, the interest text is rewritten based on the chosen action—Add introduces new concepts, Delete removes outdated interests, Update refines existing preferences, and Retain maintains the status quo. This compresses open-ended semantic evolution into state transitions within a discrete action space.
    *   **Design Motivation**: Allowing the LLM to freely rewrite descriptions often leads to semantic drift and error accumulation. Atomic operations provide explainable change trajectories and limit the LLM's freedom to four operations for stability. The case study (Table 6) shows a user evolving from "plus-size long-sleeve tops → accessories → accessories (retain) → accessories + baby-doll dresses," corresponding to Modify/Delete/Retain/Add operations, proving the readability of the edit sequence.

2.  **Group-Aware Alignment (GAA) — Differentiated Retrieval & Filtering**:
    *   **Function**: Provides more neighbors for long-tail users to supplement sparse signals and stricter filtering for active users to remove noise.
    *   **Mechanism**: Users are segmented by $n_u$ into top 20% active and 80% long-tail. Top-$K$ neighbor sets $N_u^{(g)}$ are retrieved from $\mathcal{U} \setminus \{u\}$ using cosine similarity, where $K$ varies by group (larger for long-tail, smaller for active). Long-tail neighbors are fully retained $N_u^{\text{long-tail}} = N_u^{(g)}$. Active users are filtered via Pearson similarity percentiles: $N_u^{\text{active}} = \{v \in N_u^{(g)} \mid \text{Pearson}(u,v) \ge Q_\tau(\mathcal{S}_u)\}$, where $Q_\tau$ is the $\tau$-th percentile of the user’s own similarity distribution.
    *   **Design Motivation**: Fixed absolute thresholds are unfair across different users; only percentiles of a user's own distribution can be adaptive. Experiments in Figure 3(a) confirm that the optimal number of neighbors for long-tail users is significantly larger than for active users, whereas too many neighbors for active users lead to performance drops due to noise amplification.

3.  **Self-Distillation Alignment Loss**:
    *   **Function**: Pulls each target user toward the "average neighbor representation" to provide implicit collaborative signal enhancement.
    *   **Mechanism**: The mean of neighbors $\frac{1}{|N_u|}\sum_{v \in N_u} f(v)$ acts as the teacher mediator, while the target user $f(u)$ acts as the student mediator. The loss is formulated as $\mathcal{L}_{SD} = \frac{1}{|\mathcal{U}|}\sum_u \|f(u) - \frac{1}{|N_u|}\sum_{v \in N_u} f(v)\|^2$.
    *   **Design Motivation**: Self-distillation is gentler than hard contrastive learning. Combined with group-aware retrieval, long-tail users receive a teacher representing a majority vote of sparse neighbors, while active users receive a teacher curated from high-Pearson neighbors.

### Loss & Training
*   **Ranking Loss (pairwise)**: $\mathcal{L}_{\text{Rank}} = -\sum_u \sum_k \log \sigma(\hat{y}_{u,k}^+ - \hat{y}_{u,k}^-)$.
*   **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{Rank}} + \alpha \cdot \mathcal{L}_{SD}$, with $\alpha \in \{1, 0.5, 0.1, 0.05, 0.01\}$ selected via grid search.
*   **Neighbor Count** $N_u^{(g)} \in \{2, 6, 10, 14, 18\}$; percentile $\tau$ searched by octiles; default LLM is Qwen2.5-7B-Instruct with a stage length of 13.

## Key Experimental Results

### Main Results

| Dataset | Backbone | Metric | LLM-ESR (Prev. SOTA) | HSUGA | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fashion | SASRec | HR@10 | 0.5619 | **0.5880** | +4.6% |
| Fashion | SASRec | Tail Item HR@10 | 0.1095 | **0.1602** | +46.3% |
| Beauty | BERT4Rec | HR@10 | 0.5393 | **0.5711** | +5.9% |
| Beauty | BERT4Rec | Tail Item HR@10 | 0.1379 | **0.1829** | +32.6% |
| Beauty | SASRec | NDCG@10 | 0.3713 | **0.3911** | +5.3% |
| Beauty | SASRec | Tail Item HR@10 | 0.2257 | **0.2415** | +7.0% |

Overall, SOTA is achieved across all three datasets and three backbones, with the **largest improvements observed on tail items** (up to +46%), validating the effectiveness of the group-aware design in supplementing sparse signals.

### Ablation Study (Fashion Dataset, SASRec backbone)

| Configuration | HR@10 | NDCG@10 | Tail Item HR@10 | Description |
| :--- | :--- | :--- | :--- | :--- |
| HSUGA (Full) | **0.5946** | **0.4979** | **0.1930** | Full model |
| w/o Add | 0.5881 | 0.4921 | 0.1746 | Remove Add operation |
| w/o Delete | 0.5892 | 0.4929 | 0.1773 | Remove Delete operation |
| w/o Update | 0.5869 | 0.4929 | 0.1720 | Remove Update operation (-1.3% HR) |
| w/o Retain | 0.5845 | 0.4918 | 0.1657 | Remove Retain operation (-1.7% HR, most severe) |
| w/o Interest Updater | 0.5872 | 0.4930 | 0.1784 | HSU degrades to standard summarization |
| w/o Group-Aware SD | 0.5841 | 0.4904 | **0.1573** | Disable grouping (uniform alignment), tail items drop 18.5% |
| w/o Active User Filter | 0.5903 | 0.4928 | 0.1745 | Remove active user filtering only |

### Key Findings
*   **Group-Aware design is critical**: Removing Group-Aware SD resulted in the largest drop for tail items (0.1930 → 0.1573, -18.5%), indicating that differentiated alignment strategies are the core source of gain in long-tail scenarios.
*   **Retain operation contributes most**: Among the four atomic edits, removing Retain caused the largest performance drop (0.5946 → 0.5845), suggesting that "explicitly choosing to remain unchanged" is more stable than having the LLM default to no action, aligning with the philosophy of explainable memory operations.
*   **HSUGA-7B > CoT-14B**: Table 5 shows HSUGA with Qwen2.5-7B (H@10=0.6445) outperforms CoT with Qwen2.5-14B (0.6391), proving gains stem from the structured reasoning paradigm rather than pure model capacity.
*   **High Stability**: Fluctuations in the grouping threshold (15%/20%/25%) and Pearson percentile (25/50/75) were only ±0.005 H@10, showing the method is insensitive to hyperparameters.
*   **Efficiency Trade-off**: HSU's offline time is double that of standard CoT (0.381 vs 0.189 s/user), but it supports incremental updates (~272ms per stage), avoiding full sequence re-inference, making it acceptable for industrial deployment.

## Highlights & Insights
*   **Converting "semantic editing" into a discrete action space** is the most ingenious design of this paper—unconstrained LLM summarization is too flexible. Discrete actions + execution steps compress evolution into a state machine, which is both interpretable and resistant to error accumulation. This approach could be transferred to any "long-term memory / incremental update" LLM agent scenario.
*   **"Heterogeneous alignment by user activity" is a long-overlooked perspective in long-tail recommendation**: Traditional contrastive learning treats all users equally, but the optimal number of neighbors for long-tail vs. active users is inversely related (Figure 3a shows optimal active neighbors are roughly 1/3 of those for long-tail users). This observation warrants reconsideration by the Recommender System community.
*   **Percentile thresholds instead of absolute thresholds** is a practical engineering trick: Similarity scales vary greatly across users. Fixed absolute thresholds favor high-density users, while percentile-based filtering naturally resists scale drift.
*   **The Plugin paradigm** ensures HSUGA does not compete directly with specific LLM4SR methods—Panel A adds GAA to "extraction-based" methods, and Panel B adds HSU to "utilization-based" methods. Almost all base + plugin combinations showed improvement (all entries in Table 1 marked with `*`), proving the two plugins are orthogonal.

## Limitations & Future Work
*   **LLM inference costs remain high** for industrial-scale user bases despite support for intra-stage parallelism and incremental updates. Lighter preference extraction strategies are needed.
*   **Stage length is hardcoded** (default 13). Interest evolution speed varies by user; theoretically, segmentation should be adaptive, such as using session boundaries or topic shifts. Fixed lengths may sever short-term interest cycles.
*   **Are four atomic operations sufficient?** Finer operations like Merge (combining similar interests) or Split (dividing a generalized interest) were not considered, which might be necessary for more complex evolution scenarios.
*   **GAA grouping only uses interaction counts $n_u$**, ignoring dietary dimensions of activity like interest diversity, consumption amount, or temporal decay, which might fail for users with sudden interest bursts.

## Related Work & Insights
*   **vs LLM-ESR (Liu et al., NeurIPS 2024)**: LLM-ESR uses LLMs for semantic embeddings + dual-tower retrieval, making it the strongest current baseline. HSUGA builds on it with HSU + GAA to achieve comprehensive gains by patching "how to extract semantics stably" and "how to use semantics differentially."
*   **vs CoT Reasoning (Wei et al., NeurIPS 2022)**: HSU is a structured variant of CoT for recommendation—restricting reasoning to "operation selection → execution," similar to a hybrid of ReAct and memory editing.
*   **vs MemoryBank (Zhong et al., AAAI 2024) / LongMem (Wang et al., NeurIPS 2023)**: HSU adopts atomic ops from editable memory but applies them to interest evolution in recommendation, whereas MemoryBank targets memory updates in dialogue.
*   **vs MELT (SIGIR 2023) / CITIES (ICDM 2020)**: Traditional long-tail recommendation uses meta-learning or counterfactuals; HSUGA uses LLM semantics and group-aware alignment, and the two can be orthogonally combined.

## Rating
*   Novelty: ⭐⭐⭐⭐ Applying "editable memory atomic ops" to model interest evolution in recommendation is novel, and group-aware alignment is a rare explicit modeling approach.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets × three backbones × multiple baselines + Panel A/B compatibility verification + detailed ablation + robustness + efficiency analysis + case study.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, well-explained formulas and tables, largely unambiguous; minor point: details on coupling between plugins (how HSU output feeds GAA retrieval) are described somewhat briefly.
*   Value: ⭐⭐⭐⭐ The plugin paradigm lowers adoption barriers, and the 30-46% improvement on tail items provides real practical value for industrial systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](../../AAAI2026/recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)

</div>

<!-- RELATED:END -->
