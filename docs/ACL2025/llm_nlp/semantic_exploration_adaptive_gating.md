---
title: >-
  [Paper Note] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models
description: >-
  [ACL 2025 (Long Paper)][LLM (Other)][tree search] To address two major sources of waste in LLM tree-search reasoning—"performing complex searches for simple problems" and "repeatedly expanding semantically identical paths"—this paper proposes the SEAG framework. It first uses an entropy-based gating mechanism to decide whether to activate tree search, and then merges equivalent reasoning steps via semantic clustering. Ultimately, SEAG achieves an average accuracy improvement…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM (Other)"
  - "tree search"
  - "MCTS"
  - "semantic clustering"
  - "adaptive gating"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: 191437cdac5d9f65
---

# Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2501.05752](https://arxiv.org/abs/2501.05752)  
**Code**: [https://github.com/ml-postech/SEAG-semantic-exploration-with-adaptive-gating](https://github.com/ml-postech/SEAG-semantic-exploration-with-adaptive-gating)  
**Area**: LLM/NLP  
**Keywords**: tree search, MCTS, semantic clustering, adaptive gating, reasoning efficiency

## TL;DR
To address two major sources of waste in LLM tree-search reasoning—"performing complex searches for simple problems" and "repeatedly expanding semantically identical paths"—this paper proposes the SEAG framework. It first uses an entropy-based gating mechanism to decide whether to activate tree search, and then merges equivalent reasoning steps via semantic clustering. Ultimately, SEAG achieves an average accuracy improvement of 4.3% while requiring only 31% of the inference overhead of RAP.

## Background & Motivation
LLMs have made significant progress in multi-step reasoning tasks such as mathematical reasoning (GSM8K) and common-sense reasoning (ARC). Chain-of-Thought (CoT) and its Self-Consistency variants improve accuracy by sampling multiple reasoning paths and voting on them, but they can only perform "linear" searches. Methods like Tree-of-Thought (ToT) and RAP introduce structured exploration of the reasoning space via tree-search algorithms such as BFS/DFS/MCTS, achieving better performance but at a significantly higher computational cost.

Existing tree search methods suffer from two key limitations:
1. **Difficulty-Agnostic Execution**: They perform full-tree searches even for simple problems (which the model itself can answer with high confidence), leading to a high amount of unnecessary computational waste;
2. **Semantic Redundancy in Expansion**: When expanding child nodes, LLMs often generate reasoning steps that are phrased differently but are semantically identical (e.g., "How many pages did Julie read yesterday?" and "What is the number of pages Julie finished reading yesterday?"). This results in many semantically redundant subtrees in the search tree, wasting the search budget.

## Core Problem
How to allow tree search reasoning methods to "intelligently" allocate computational resources—searching less or not at all for simple problems, and only performing deep exploration for complex ones—while avoiding redundant searches on semantically equivalent reasoning paths? The key challenges lie in: (1) reliably estimating confidence to decide when to initiate or terminate the tree search; and (2) determining semantic equivalence in natural language, which is inherently difficult, especially in open-domain reasoning scenarios.

## Method

### Overall Architecture
SEAG consists of three stages forming a complete reasoning pipeline:

**Input** → **Adaptive Gating (AG)** → If confidence is high, output directly (CoT-SC route); if confidence is low, enter → **Semantic Exploration (SE)** (MCTS tree search based on semantic clustering) → Dynamically check **Early Stopping** conditions during search → **Output final answer**

### Key Designs

1. **Adaptive Gating (AG)**: First, sample $k=10$ reasoning paths with CoT, calculate the frequency of each answer $q(y)$ to compute the entropy of the answer distribution $H(y) = -\sum_{y \in \mathcal{Y}} q(y) \log q(y)$. If $H(y) \leq \tau$ (indicating high confidence), output the answer directly via majority voting, bypassing the tree search. Only high-entropy (uncertain) problems proceed to the SE stage. Experiments demonstrate that approximately 67% of problems are filtered out during the AG stage, significantly reducing the average computational cost. The intuition behind this design comes from data analysis: on low-entropy problems, CoT-SC itself can achieve very high accuracy, leaving almost no room for additional gains from tree search.

2. **Semantic Exploration (SE) — Semantic Clustering + Semantic PUCT**: During each node expansion in the MCTS tree, the LLM generates $d=4$ candidate actions (sub-questions). Then, a DeBERTa-large model is employed as a bidirectional textual entailment detector to determine if any two actions are semantically equivalent, grouping equivalent actions into a semantic equivalence class $\mathcal{C} = \{C_1, C_2, \ldots, C_{d'}\}$ ($d' \leq d$). Consequently, the search tree only needs to maintain one subtree for each semantic equivalence class rather than duplicating subtrees for superficially different actions.

   For action selection, a **Semantic PUCT** algorithm is proposed: it scales the standard PUCT from the action level to the semantic cluster level. The prior probability of a cluster is the sum of the probabilities of all actions within that cluster: $\pi(C_i|s) = \sum_{a \in C_i} p_\theta(a|s,m)$. Once a cluster is selected, the action with the highest probability within the cluster is used to construct the prompt. This design prioritizes the exploration of semantic paths supported by more LLM-generated samples, embodying the self-consistency principle.

3. **Early Stopping — Weighted Aggregate Reward**: During the MCTS iterations, whenever a terminal node is reached, the weighted reward along its path is calculated as $R(n_j) = \sum_{n \in P(n_j)} |C(n)| \cdot r(n)$, where $|C(n)|$ is the size of the semantic cluster to which the node belongs, highlighting reasoning paths supported by more samples. The rewards of terminal nodes yielding the same answer are aggregated as $R_{\text{agg}}(y) = \sum_{n_j: Y(n_j)=y} R(n_j)$. If the maximum aggregate reward exceeds a threshold $\alpha$, the search is terminated early. This allows the system to output an answer with high confidence without executing all MCTS iterations.

## Key Experimental Results

| Dataset | Method | Accuracy ↑ | # Inferences ↓ | Remarks (Llama3-8B-Instruct) |
|--------|------|-----------|-----------------|--------------------------|
| GSM8K | CoT | 0.762 | 1 | Single-path baseline |
| GSM8K | CoT-SC | 0.845 | 10 | Multi-path voting |
| GSM8K | ToT | 0.785 | 104.80 | BFS tree search |
| GSM8K | RAP | 0.825 | 128.40 | MCTS tree search |
| GSM8K | **SE** | **0.850** | **82.63** | Semantic exploration (no gating) |
| GSM8K | **SEAG** | **0.860** | **41.69** | Complete framework |
| ARC | CoT | 0.818 | 1 | - |
| ARC | CoT-SC | 0.823 | 10 | - |
| ARC | RAP | 0.812 | 196.96 | - |
| ARC | **SEAG** | **0.845** | - | Overall best |

- Across three LLMs (Llama3-8B, Llama2-13B, Mistral-7B) and two datasets, SEAG consistently achieves the highest accuracy.
- Compared to RAP, SEAG improves the average accuracy by 4.3% while requiring only 31% of the inference steps of RAP.
- End-to-end latency: SEAG 38.89s vs RAP 151.19s vs ToT 120.63s (ARC, Llama3-8B, A5000 GPU).

### Ablation Study Points
- **Semantic PUCT vs UCT vs PUCT** (Table 4): Semantic PUCT achieves 0.850 on GSM8K vs 0.828 for UCT and 0.840 for PUCT with comparable token count, indicating that balancing exploration and exploitation at the semantic cluster level is more effective than at the action level.
- **Weighted Aggregation vs Equal Aggregation** (Table 5): Aggregation weighted by $|C|$ outperforms equal aggregation in both accuracy and efficiency (GSM8K: 0.850/82.63 vs 0.845/95.89).
- **Semantic Clustering Reduction Rate** (Table 2): Semantic redundancy becomes more severe as search depth increases—reducing by 25-42% at depth 1, and 55-61% at depth 4, demonstrating that semantic clustering is more valuable in deeper searches.
- **AG Gating Analysis** (Figure 4): CoT-SC accuracy is already extremely high in low-entropy zones, whereas the gains from tree search are remarkable in high-entropy zones, validating the rationale behind the adaptive gating mechanism.

## Highlights

- **The "Filter-then-Search" Two-Stage Strategy is Elegant**: Using a lightweight entropy estimation to rapidly filter out simple problems, and only deploying heavy-duty search for difficult ones, makes resource allocation highly efficient and reasonable. This concept can be extended to any reasoning scenario that mixes simple and hard instances.
- **Semantic Clustering Using DeBERTa for Entailment Detection**: Compared to coarse-grained methods like embedding similarity, bidirectional entailment detection captures semantic equivalence more accurately. Moreover, DeBERTa is extremely lightweight compared to the LLM, making the additional overhead negligible (only 2.54s).
- **Semantic PUCT Naturally Conforms to Self-Consistency**: Semantic directions supported by more LLM samples receive higher exploration priority, unifying the information-theoretic meaning of sampling frequency (self-consistency) with the exploration-exploitation trade-off of tree search.

## Limitations

- **Solely Dependent on Internal Knowledge of LLM**: It does not leverage external tools (calculators, retrievers, etc.), which may limit performance in scenarios requiring precise calculations or factual queries.
- **Evaluated Only on Discrete Answer Tasks**: Tested on GSM8K (numeric) and ARC (multiple-choice), without verifying the efficacy on open-ended text generation tasks.
- **Semantic Clustering Relies on the NLI Model**: The quality of entailment detection directly affects clustering accuracy; for highly specialized or domain-specific reasoning steps, general-purpose NLI models may perform poorly.
- **Hyperparameters $\tau$ and $\alpha$ Require Dataset-Specific Tuning**: The generalizability of the gating threshold and early stopping threshold remains to be validated.

## Related Work

- **vs RAP (Hao et al., 2023)**: RAP is the most direct baseline, also modeling with MCTS + MDP. SEAG improves upon RAP in three dimensions: (1) incorporating AG to avoid search waste on simple problems; (2) using semantic clustering to reduce redundant expansion; (3) employing weighted reward aggregation and early stopping. Ultimately, SEAG achieves a 4.3% accuracy improvement over RAP with only 31% of the inference steps.
- **vs ToT (Yao et al., 2023)**: ToT utilizes BFS/DFS, which is less efficient than MCTS, and lacks semantic deduplication and adaptive gating. SEAG outperforms ToT across both accuracy and efficiency.
- **vs CoT-SC (Wang et al., 2023)**: CoT-SC is simple and efficient, but has limited potential (performing only linear search and voting). SEAG, through AG, effectively only conducts more searches than CoT-SC on hard problems, while degrading to CoT-SC on simple ones, thereby blending the advantages of both.

## Related Work & Insights

- The idea of "semantic clustering deduplication" can be applied to any scenario requiring multi-path sampling (e.g., Best-of-N sampling, reward model training data deduplication, etc.).
- The design of weighted aggregate reward (larger cluster = more believable) is inherently a form of soft self-consistency, which is more nuanced than hard voting.

## Rating

- Novelty: ⭐⭐⭐⭐ Individual components (entropy gating, semantic clustering, PUCT variants) are not completely new on their own, but they are integrated into a complete and highly synergistic framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three models and two datasets with thorough ablation and latency analysis; however, the dataset formats are relatively uniform (both having discrete answers).
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, mathematical derivations are complete, and illustrations are highly illustrative.
- Value: ⭐⭐⭐⭐ Provides a practical efficiency optimization paradigm for LLM tree-search reasoning, with highly transferable concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)

</div>

<!-- RELATED:END -->
