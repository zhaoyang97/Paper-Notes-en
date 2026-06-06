---
title: >-
  [Paper Note] Optimized Algorithms for Text Clustering with LLM-Generated Constraints
description: >-
  [AAAI 2026][AIGC Detection][Text Clustering] This paper proposes the LSCK-HC framework, which leverages LLMs to generate set-form must-link/cannot-link constraints (as opposed to traditional pairwise constraints)…
tags:
  - "AAAI 2026"
  - "AIGC Detection"
  - "Text Clustering"
  - "LLM Constraint Generation"
  - "k-means"
  - "Semi-supervised Clustering"
  - "Local Search"
date: 2026-05-08
content_hash: fa1aabd8929ac86d
---

# Optimized Algorithms for Text Clustering with LLM-Generated Constraints

**Conference**: AAAI 2026
**arXiv**: [2601.11118](https://arxiv.org/abs/2601.11118)  
**Code**: [https://github.com/weihong-wu/LSCK-HC](https://github.com/weihong-wu/LSCK-HC)  
**Area**: AIGC Detection / Text Clustering
**Keywords**: Text Clustering, LLM Constraint Generation, k-means, Semi-supervised Clustering, Local Search

## TL;DR
This paper proposes the LSCK-HC framework, which leverages LLMs to generate set-form must-link/cannot-link constraints (as opposed to traditional pairwise constraints), coupled with a penalty-based local search clustering algorithm. The approach achieves clustering accuracy comparable to SOTA on five short-text datasets while reducing the number of LLM queries by more than 20×.

## Background & Motivation

**Background**: Short text clustering (STC) is a foundational NLP task in which k-means and its variants are widely employed. To improve clustering quality, semi-supervised methods introduce must-link (ML) and cannot-link (CL) pairwise constraints as background knowledge. Recent work has explored using LLMs to automatically generate such constraints via in-context learning.

**Limitations of Prior Work**: (1) Traditional constraint generation relies on costly expert annotation or existing labels; (2) Generating pairwise constraints with LLMs (e.g., the FSC method) demands a large number of queries (tens of thousands), incurring high cost with limited improvement; (3) LLM-generated constraints may contain errors, yet existing clustering algorithms are not designed to account for the specific characteristics of LLM outputs.

**Key Challenge**: Pairwise constraints are highly query-inefficient—each LLM call assesses the relationship between only two points, and covering a sufficient portion of the data requires an exponentially large number of queries. Furthermore, the balance between hard constraint satisfaction and soft constraint penalization has not been optimized for the error characteristics of LLM outputs.

**Goal**: (1) Reduce the query cost of LLM-based constraint generation; (2) Improve the accuracy of generated constraints; (3) Design clustering algorithms tolerant of erroneous constraints.

**Key Insight**: Extend the constraint format from pairwise to set-form—a single LLM query can generate multiple relational constraints—and distinguish hard from soft constraints via a confidence threshold.

**Core Idea**: Replace pairwise constraints with set-form constraints to improve query efficiency, and employ penalty-based local search to tolerate errors in LLM-generated constraints.

## Method

### Overall Architecture
Input short texts are encoded into vectors using an embedding model (Instructor-large/E5). The framework operates in two stages: (1) **Constraint Generation**: candidate point sets are selected and submitted to an LLM to produce ML/CL set-form constraints; (2) **Constrained Clustering**: hard ML constraints initialize cluster centers, soft ML constraints are handled via penalty-based merging, and CL constraints are processed through maximum-weight matching combined with local search.

### Key Designs

1. **Set-Form Must-Link Constraint Generation**:

    - **Function**: Generate multiple ML relations per LLM query, rather than evaluating each pair individually.
    - **Mechanism**: Coreset techniques partition the data into representative subsets whose internal points are mutually similar. The corresponding texts of each subset are submitted to the LLM, which returns groupings (e.g., $\{t_1, t_2, t_3\}, \{t_4\}$); each group containing 2+ texts constitutes one ML set constraint.
    - **Design Motivation**: Coresets ensure that candidate points do not differ excessively along any single dimension, improving LLM judgment accuracy; the set format allows a single query to cover multiple constraint relations.

2. **Confidence-Based Hard/Soft ML Constraint Distinction**:

    - **Function**: Partition ML constraints into high-confidence (hard) and low-confidence (soft) based on LLM response consistency.
    - **Mechanism**: Pairwise distances and diameters within constraint sets are computed at multiple grid-width levels $r_j = (1+\varepsilon)^j \cdot \sqrt{cost_{kc}/10n\delta}$, and a distance threshold $\psi$ is identified via binary search. For each candidate threshold, the LLM is queried multiple times (5 times for pairwise, 10 for set-form); if all responses are consistent, ML constraints whose diameter falls below the threshold are treated as hard.
    - **Design Motivation**: Hard constraints are used during initialization (influencing k-means++ seed selection), where erroneous constraints are particularly harmful and thus require high-confidence filtering.

3. **Cannot-Link Constraint Generation**:

    - **Function**: Generate set-form CL constraints of size at most $k$.
    - **Mechanism**: Candidate points $q$ are sampled uniformly at random from uncovered points at distance greater than $r = cost_{kc}$ from the current CL set. The LLM determines whether $q$ should be added to the current CL set; if the LLM returns "None," the point is added; otherwise, the next candidate is sought until the CL set reaches size $k$ or no new point can be found.
    - **Design Motivation**: The distance threshold ensures that candidate points genuinely belong to different clusters, simplifying the LLM's classification task.

4. **Penalty-Based ML Clustering (Alg. 1)**:

    - **Function**: Handle the assignment of soft ML constraints.
    - **Mechanism**: For each soft ML set $X$, points are first assigned to their nearest centers, forming sub-partitions. The algorithm iteratively selects the two sub-partitions $P_i, P_j$ with the largest diameter and compares the cost of merging them against retaining the original assignment with penalty: if $(w_m + d(\bar{P_j}, c_j)) \cdot |P_j| + (w_m + d(\bar{P_i}, c_i)) \cdot |P_i| > \sum_{p \in P_i \cup P_j} d(p, c_{ij})$, the partitions are merged. Runtime is $O(nk^2)$.

5. **Penalty-Based CL Clustering (Alg. 2)**:

    - **Function**: Handle CL constraints via maximum-weight matching combined with local search.
    - **Mechanism**: For each CL set $Y$, an auxiliary bipartite graph $G(C, Y; E)$ is constructed to compute the maximum-weight matching between cluster centers and CL points. For each matched point, the marginal change in matching value upon its removal $g_y$ is computed; if the maximum change falls below a threshold, the current matching is accepted; otherwise, the point is removed and assigned to its nearest center.

### Loss & Training
This work proposes a clustering algorithm rather than a deep learning model. The core objective is the k-means cost function augmented with constraint violation penalties:
$$\sum_{i=1}^{k} \sum_{x \in A_i} \|x - c(A_i)\|^2 + w_m \cdot (\text{ML violations}) + w_{cl} \cdot (\text{CL violations})$$

## Key Experimental Results

### Main Results
Experiments are conducted on 5 text datasets (Tweet, Banking77, CLINC-I, CLINC-D, GoEmo) using Instructor-large embeddings and GPT-4o-generated constraints.

| Dataset | Constraint Ratio | LSCK-HC ACC | PCK ACC | k-means++ ACC | Gain (vs k-means++) |
|---------|-----------------|-------------|---------|---------------|---------------------|
| Tweet | 40% | ~68% | ~65% | ~62% | +6% |
| CLINC-D | 40% | ~78% | ~75% | ~74% | +4% |
| Banking77 | 20% | Best | - | - | Optimal at 20% |

### Query Efficiency Comparison (Constraint Generation)

| Dataset | Constraint Ratio | FSC #Query | Ours #Query | Reduction | FSC ML-RI | Ours ML-RI |
|---------|-----------------|-----------|------------|-----------|-----------|-----------|
| Banking77-ML | 2% | 5260 | 91 | **57×** | 9.92% | **96.42%** |
| Banking77-ML | 20% | 26580 | 480 | **55×** | 11.40% | **85.83%** |
| CLINC-ML | 2% | 25095 | 91 | **275×** | 42.05% | **96.25%** |
| Tweet-ML | 2% | 13070 | 99 | **132×** | 50.00% | **100.00%** |
| Tweet-CL | 20% | 31485 | 821 | **38×** | 99.17% | **99.56%** |

### Key Findings
- **20–275× reduction in query count**: The set-form constraint format is the key driver—a single query covers multiple relational constraints.
- **Substantial improvement in ML constraint quality**: FSC's ML accuracy is extremely low at small constraint ratios (only 9.92% on Banking77), since its ML constraints are derived indirectly from CL inference; the proposed method selects candidates based on distance, achieving ML accuracy of 85–100%.
- **20% constraint ratio is the optimal sweet spot**: Higher ratios lead to more erroneous constraints and diminishing returns.
- Weaker embedding models (E5) benefit more from constrained clustering (ARI improvement of nearly 10%), as a lower baseline leaves more room for correction.

## Highlights & Insights
- **Replacing pairwise constraints with set-form constraints is the central contribution**: This not only reduces query count but also improves constraint accuracy through mutual validation among points within a set. The idea is transferable to any semi-supervised task requiring LLM-based relational judgments.
- **Hard/soft constraint separation strategy**: High-confidence constraints guide initialization (k-means++ is sensitive to seed selection), while low-confidence constraints are handled flexibly via penalties. This tiered trust strategy is applicable to downstream tasks involving any LLM-generated annotations.
- **Local search is more robust than PCK**: PCK degrades in performance once the constraint ratio exceeds 10%, whereas local search maintains robustness through penalty-merge iterations.

## Limitations & Future Work
- Validation is limited to short text clustering; effectiveness on long documents or multimodal data remains unknown.
- Constraint generation relies on the distance structure of the embedding space; poor embedding quality would degrade candidate selection.
- The penalty weights $w_m$ and $w_{cl}$ lack an automatic tuning mechanism.
- Only two LLMs (GPT-4o and DeepSeek) are evaluated; the impact of constraint quality across different models is not fully explored.
- The fixed-$k$ assumption is a limitation—in practice, the number of clusters is often unknown.

## Related Work & Insights
- **vs. FSC (Viswanathan et al. 2024)**: FSC uses min-max selection for pairwise queries combined with PCK clustering, resulting in high query volume and low ML constraint accuracy. LSCK-HC adopts set-form constraints, reducing query count by 20×+ and improving ML accuracy from below 50% to above 85%.
- **vs. COP-KMeans**: COP enforces hard constraints greedily and fails entirely when erroneous constraints are encountered. LSCK-HC's penalty mechanism tolerates such errors.
- **vs. BH-KM (Baumann & Hochbaum 2022)**: BH-KM handles soft constraints via mixed-integer programming but scales poorly and cannot complete under large constraint volumes. LSCK-HC's $O(nk^2)$ time complexity is more practical.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of replacing pairwise constraints with set-form constraints is concise and effective; the hard/soft separation aligns well with LLM output characteristics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 5 datasets, multiple embeddings and LLMs, with analysis of query efficiency and constraint quality.
- **Writing Quality**: ⭐⭐⭐⭐ Formal presentation is clear and algorithmic pseudocode is well-structured, though notation is occasionally dense.
- **Value**: ⭐⭐⭐⭐ Broadly applicable to query efficiency optimization in LLM-assisted semi-supervised learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](../../ICML2026/aigc_detection/black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](../../NeurIPS2025/aigc_detection/synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](../../ACL2026/aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](../../ICLR2026/aigc_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)

</div>

<!-- RELATED:END -->
