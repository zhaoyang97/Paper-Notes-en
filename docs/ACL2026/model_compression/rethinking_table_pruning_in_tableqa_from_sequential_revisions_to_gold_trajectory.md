---
title: >-
  [Paper Note] Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search
description: >-
  [ACL2026][Model Compression][TableQA] This paper proposes TabTrim, which transforms table pruning from error-prone single-path sequential revisions into a framework comprising a "SQL trajectory-supervised pruner + loss-a…
tags:
  - "ACL2026"
  - "Model Compression"
  - "TableQA"
  - "table pruning"
  - "gold trajectory"
  - "verifier"
  - "beam search"
date: 2026-05-08
content_hash: f47bed53ff827d94
---

# Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search

**Conference**: ACL2026 Oral  
**arXiv**: [2601.03851](https://arxiv.org/abs/2601.03851)  
**Code**: None  
**Area**: TableQA / Table Pruning / LLM Reasoning  
**Keywords**: TableQA, table pruning, gold trajectory, verifier, beam search

## TL;DR
This paper proposes TabTrim, which transforms table pruning from error-prone single-path sequential revisions into a framework comprising a "SQL trajectory-supervised pruner + loss-aware verifier + parallel trajectory search." It achieves an average accuracy of 73.5% on WikiTQ, TabFact, and TableBench, outperforming the strongest baseline by 3.2 points.

## Background & Motivation
**Background**: TableQA and complex table reasoning often require identifying a small number of relevant rows and columns within large tables. Directly serializing the original table for an LLM introduces significant noise and high long-context costs. Consequently, table pruning is used to remove redundant cells, retaining only a task-relevant sub-table before passing it to a downstream reasoner.

**Limitations of Prior Work**: Existing table pruning methods are generally categorized into program-based and LLM-based approaches. The former relies on the execution of programs like SQL or Python, while the latter depends on CoT or multi-agent planning. Both are prone to pruning critical information. Furthermore, subsequent critique signals are often unreliable: successful program execution does not guarantee semantic correctness, and LLM-as-a-Judge may rationalize incorrect reasoning or excessively reject correct steps.

**Key Challenge**: The primary risk in table pruning is over-pruning, as the downstream reasoner can rarely recover once answer-critical cells are deleted. Conversely, overly conservative pruning retains too much noise. Existing methods typically follow a single trajectory for sequential revision, where early errors become irreversible due to the lack of backtrackable or comparable candidate branches.

**Goal**: The authors aim to provide verifiable intermediate supervision for table pruning, allowing the model to recognize which critical cells should be retained at each step. Additionally, they seek to explore multiple pruning trajectories during inference instead of relying on a single sequential revision.

**Key Insight**: Text-to-SQL datasets contain gold SQL queries. The authors observe that the clause-level execution of a SQL query naturally generates a sequence of intermediate sub-tables. These sub-tables, constrained by the final correct answer, can serve as a gold pruning trajectory without requiring manual annotation.

**Core Idea**: Use gold SQL execution trajectories to train a pruner and a verifier, then employ a beam-search-style parallel trajectory search during inference to maintain multiple candidate sub-tables, thereby avoiding the local optima inherent in single-path pruning.

## Method

### Overall Architecture
The input to TabTrim consists of a question $Q$, the original table $T_0$, and the current sub-table $T_{t-1}$, with the output being a more compact sub-table. During the training phase, a gold sub-table trajectory is constructed from Text-to-SQL data by decomposing gold SQL into clause-level operations (e.g., row filtering and column projection) ordered by execution to obtain $T_0, T_1^+, ..., T_n^+$. Two components are then trained: the pruner learns to generate the next gold sub-table from the current state, and the verifier learns to predict a quality score for any sub-table relative to the final gold sub-table. During inference, starting from the original table, the pruner generates multiple candidates at each step, which are then scored by the verifier to keep the top-$k$. Finally, the sub-table with the highest score across all beams is selected for downstream reasoning.

### Key Designs
1.  **Gold trajectory construction**:
    - **Function**: Automatically generate step-by-step pruning supervision using existing Text-to-SQL annotations, avoiding the need for additional manual annotation of intermediate sub-tables.
    - **Mechanism**: For each sample $(Q, SQL_{gold}, T_{raw})$, the SQL is decomposed into clause-level operations based on execution logic; each step yields a gold sub-table $T_t^+$. The authors also construct off-trajectory negative sub-tables $T_t^-$ by modifying gold operations to create progression and correction datasets.
    - **Design Motivation**: Final answers only provide supervision on terminal correctness, making it difficult to identify where a pruner failed. SQL intermediate states provide alignable procedural supervision, while negative trajectories train the model to recover from erroneous states.

2.  **Trajectory-supervised Pruner and DPO**:
    - **Function**: Generate the next sub-table to align with the gold trajectory and learn to rectify errors from incorrect intermediate states.
    - **Mechanism**: The first phase uses SFT to optimize for both progression and correction samples, summarized by the loss: $L_{SFT}=-\log P_\theta(T_t^+|Q,T_0,T_{t-1}^+)-\lambda\log P_\theta(T_t^+|Q,T_0,T_{t-1}^-)$. The second phase uses DPO to favor the gold next sub-table $T_t^+$ over incorrect sub-tables $T_t^-$, reducing fine-grained semantic pruning errors.
    - **Design Motivation**: Learning only the gold path prevents the model from recovering when it deviates during inference. Correction samples and DPO ensure the pruner can both advance correctly and return to the proper path from erroneous candidates.

3.  **Loss-aware Verifier and Parallel Trajectory Search**:
    - **Function**: Score candidate sub-tables and select paths during inference that are most likely to retain answer-critical cells.
    - **Mechanism**: Sub-tables are represented as canonical cell sets to calculate precision and recall against the final gold sub-table $T_n^+$. An $F$-score with a recall bias is used as the quality score $S(T_t)$. A default $\alpha=1.5$ is used to emphasize the retention of answer-critical cells. Inference utilizes beam search with width $k$, branch factor $b$, and maximum depth $D_{max}$. At each step, a generate-score-select cycle is performed, ultimately choosing the sub-table with the highest verifier score across all beams.
    - **Design Motivation**: Likelihood does not necessarily reflect sub-table quality, particularly in penalizing the omission of critical cells. The loss-aware score explicitly biases toward recall, while parallel search allows the system to abandon early erroneous branches.

### Loss & Training
TabTrim utilizes WikiSQL and SQUALL to construct over 80K training samples. The pruner is trained using Qwen3-4B and Qwen3-8B, while the verifier uses Qwen3-0.6B with a default $\alpha=1.5$. Default inference parameters are $k=b=2$ and $D_{max}=4$, resulting in an upper bound of $O(k\cdot b\cdot D_{max})$ pruner/verifier calls per sample. GPT-4o-mini is employed for final answer generation to ensure consistency with baseline downstream reasoner setups.

## Key Experimental Results

### Main Results
| Method | WikiTQ | TabFact | TB-NR | TB-FC | TB-DA | Average |
|------|--------|---------|-------|-------|-------|---------|
| Direct QA: GPT-4o-mini | 54.3 | 77.4 | 65.5 | 76.0 | 25.1 | 59.8 |
| Binder | 54.8 | 83.3 | 66.8 | 67.7 | 26.8 | 59.9 |
| Chain-of-Table | 67.5 | 88.9 | 68.5 | 78.1 | 30.3 | 66.7 |
| TALON | 70.7 | 87.6 | 67.3 | 77.1 | 28.9 | 66.3 |
| Table-Critic | 72.6 | 90.6 | 73.0 | 81.3 | 33.8 | 70.3 |
| TabTrim-4B | 76.8 | 89.4 | 76.3 | 79.2 | 32.1 | 70.8 |
| TabTrim-8B | 79.4 | 91.2 | 78.8 | 83.3 | 34.7 | 73.5 |

### Ablation Study
| Configuration | WikiTQ | TableBench | Description |
|------|--------|------------|------|
| TabTrim | 79.4 | 61.2 | Full model |
| w/o DPO | 78.1 | 58.6 | Without preference optimization, fine-grained errors are harder to correct |
| w/o Correction Samples | 74.8 | 55.4 | Without off-trajectory recovery, robustness significantly decreases |
| w/o Training | 54.7 | 49.6 | Without trajectory-supervised training, performance reverts to base model capability |
| Balanced score | 77.8 | 58.3 | Changing $\alpha$ to 1 reduces recall bias and decreases performance |
| Rank by Likelihood | 74.2 | 56.7 | Ranking by generation probability is inferior to the verifier score |
| Sequential Revisions | 72.9 | 55.1 | Performance drops significantly without parallel search |

### Key Findings
- TabTrim-8B achieves an average accuracy of 73.5%, which is 3.2 points higher than the strongest non-TabTrim baseline, Table-Critic (70.3%); on WikiTQ, it reaches 79.4%, a 6.8-point improvement over Table-Critic.
- Correction samples are one of the most critical elements in pruner training; removing them results in a 4.6-point drop on WikiTQ and a 5.8-point drop on TableBench, demonstrating the importance of recovery capabilities after deviating from the gold trajectory.
- Both parallel search and verifier ranking are irreplaceable: sequential revision causes a 6.5-point drop on WikiTQ, while likelihood ranking causes a 5.2-point drop.
- As a plug-and-play front-end, TabTrim provides a +25.9 Gain for Qwen3 on WikiTQ and +10.8 on TableBench; it provides +7.4 and +8.8 Gain for Table-R1 respectively.
- Regarding token costs, TabTrim's total token usage is approximately 0.56x to 0.66x that of Table-Critic, indicating it does not rely on brute-force large contexts for its performance.

## Highlights & Insights
- The most ingenious aspect is converting the gold SQL execution process into pruning trajectory supervision. Since many tabular tasks have programmatic annotations or executable queries, these "intermediate execution states" can be leveraged as procedural supervision rather than relying solely on final answer supervision.
- The loss-aware verifier explicitly prioritizes recall, aligning with the risk structure of table pruning: while redundancy adds noise, the loss of answer-critical cells is often irreversible.
- Parallel trajectory search shifts table pruning from "revising a single path" to "simultaneously evaluating multiple paths." This is consistent with search/rerank strategies in complex reasoning and can be effectively transferred to document compression, evidence selection, and multi-hop retrieval tasks.

## Limitations & Future Work
- The authors acknowledge that experiments were limited by compute resources, covering only 4B and 8B pruners, without exploring scaling laws or performance ceilings for larger models.
- The gold trajectories are derived from Text-to-SQL processes. For TableQA or open-table reasoning tasks without program annotations, constructing equally reliable procedural supervision remains an open question.
- The current verifier's quality score depends on cell-set overlap with the gold final sub-table. While clear during training, real-world tasks where multiple reasonable pruning paths exist might result in the verifier unfairly penalizing valid alternatives.
- Future work could investigate lighter verifiers, dynamic search budgets, and integrating TabTrim with weak-supervision trajectory generation methods that do not rely on SQL annotations.

## Related Work & Insights
- **vs Binder / TabSQLify**: These program-based methods rely on executable programs for pruning, which may mistake successful execution for semantic correctness; TabTrim uses gold trajectories and a verifier for direct sub-table quality supervision.
- **vs Chain-of-Table / Dater**: LLM-based methods prune through multi-step reasoning, but their critiques are often subjective; TabTrim’s critique is more objective, derived from SQL trajectories and cell overlap.
- **vs Table-Critic / TALON**: These also attempt critique or revision but remain largely sequential; the key differentiator for TabTrim is the maintenance of multiple candidate trajectories combined with verifier-led search.
- **Insight**: For any system that "compresses context before reasoning," procedural supervision and search are more reliable than one-shot compression. Particularly in tasks where evidence is sparse and the cost of missing data is high, priority should be given to optimizing recall-aware compression objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of SQL gold trajectory + loss-aware verifier + parallel search is a complete solution that addresses the fundamental issues of table pruning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Main results, ablations, difficulty stratification, scaling, plug-and-play, and cost analyses are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐☆ Despite the heavy mathematical notation, the main narrative is clear and the tables directly support the conclusions.
- Value: ⭐⭐⭐⭐⭐ This work has high reuse value for TableQA, RAG compression, and evidence selection, making it a strong baseline for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](../../CVPR2026/model_compression/beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)
- [\[ACL 2026\] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)
- [\[ACL 2026\] DeepPrune: Parallel Scaling without Inter-Trace Redundancy](deepprune_parallel_scaling_without_inter-trace_redundancy.md)
- [\[ACL 2026\] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)
- [\[ICLR 2026\] Parallel Token Prediction for Language Models](../../ICLR2026/model_compression/parallel_token_prediction_for_language_models.md)

</div>

<!-- RELATED:END -->
