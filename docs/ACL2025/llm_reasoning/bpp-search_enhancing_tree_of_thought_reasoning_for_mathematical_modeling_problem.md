---
title: >-
  [Paper Note] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving
description: >-
  [ACL 2025][Reasoning][Tree-of-Thought] The BPP-Search algorithm is proposed, which integrates Beam Search, Process Reward Models (PRMs), and a Pairwise Preference mechanism into the Tree-of-Thought framework for automatic mathematical modeling in operations research, significantly outperforming CoT/SC/ToT baselines on datasets like StructuredOR with fewer reasoning steps.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Tree-of-Thought"
  - "Mathematical Modeling"
  - "Process Reward Model"
  - "Beam Search"
  - "Operations Research"
date: 2026-05-08
content_hash: f4a3b3b93275f00c
---

# BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving

**Conference**: ACL 2025  
**arXiv**: [2411.17404](https://arxiv.org/abs/2411.17404)  
**Code**: [https://github.com/LLM4OR/StructuredOR](https://github.com/LLM4OR/StructuredOR)  
**Area**: LLM Reasoning  
**Keywords**: Tree-of-Thought, Mathematical Modeling, Process Reward Model, Beam Search, Operations Research

## TL;DR
The BPP-Search algorithm is proposed, which integrates Beam Search, Process Reward Models (PRMs), and a Pairwise Preference mechanism into the Tree-of-Thought framework for automatic mathematical modeling in operations research, significantly outperforming CoT/SC/ToT baselines on datasets like StructuredOR with fewer reasoning steps.

## Background & Motivation
**Background**: While LLMs show increasing capabilities in mathematical reasoning, automatically formulating natural language problems into Linear Programming (LP)/Mixed Integer Programming (MIP) models is an emerging research direction. Existing methods primarily employ CoT, SC, or ToT frameworks to guide models through multi-step reasoning.

**Limitations of Prior Work**: Existing open-source operations research datasets (e.g., NL4OPT, MAMO-ComplexLP) only provide annotations for final objective function values, lacking detailed annotations of the modeling process (e.g., variable definitions, constraint setups). This limits the application of reinforcement learning methods based on process supervision. Moreover, although the ToT framework generates multiple reasoning paths, it lacks an effective strategy for leaf node selection.

**Key Challenge**: CoT generates only a single path and is error-prone; SC cannot verify the execution or correctness of intermediate steps; ToT generates a large number of leaf nodes but struggles to determine which one to select as the final answer. While PRMs can guide search by scoring intermediate steps, their scoring in regression tasks is not precise enough—candidates with highly similar structures but differing correctness often receive nearly identical scores.

**Goal**: (a) Map out and construct an operations research dataset with complete modeling process annotations; (b) efficiently search and reliably select the optimal modeling solution within the ToT framework.

**Key Insight**: Observing that PRMs struggle to distinguish between highly similar correct and incorrect candidates in the last layer of Beam Search, a Pairwise Preference model is introduced to perform pairwise comparison ranking.

**Core Idea**: Use Beam Search and PRM for tree search pruning and acceleration, and then utilize a Pairwise Preference model to perform pairwise ranking among the final candidates to select the optimal answer.

## Method

### Overall Architecture
The input is a natural language description of an operations research problem, and the output is a complete mathematical model (sets $\to$ parameters $\to$ variables $\to$ objective function $\to$ constraints). This reasoning process is structured as a 4-layer Tree-of-Thought: Layer 1 is the problem, Layer 2 defines sets and parameters, Layer 3 defines variables, and Layer 4 contains the objective function and constraints. Each node has a maximum of 3 child nodes. The search process is divided into two phases: the intermediate layers utilize Beam Search and PRM pruning to retain the top-$k$ candidates, while the final layer uses a Pairwise Preference model to perform pairwise comparisons and determine the optimal solution.

### Key Designs

1. **StructuredOR Dataset**:

    - **Function**: Provides an operations research dataset with complete modeling process annotations (124 problems covering logistics, scheduling, networks, etc.).
    - **Mechanism**: GPT-4o is first used to generate the parameter distribution of abstract models. Then, instance data are generated via simulation and validated for solvability using the Gurobi solver. Finally, GPT-4o generates natural language descriptions and rephrases them into three semantically equivalent versions.
    - **Design Motivation**: Existing datasets only have annotated objective values, making them unusable for training process-supervised models like PRMs. StructuredOR provides end-to-end annotations spanning sets, parameters, variables, and constraints.

2. **Process Reward Model (PRM)**:

    - **Function**: Scores intermediate reasoning results at each layer of ToT to guide the search direction.
    - **Mechanism**: Fully fine-tuned based on Qwen2.5-Math-1.5B for a binary classification task. The logit corresponding to the correct label is extracted, and the sigmoid function is applied to obtain the score $S_{\text{PRM}} = \frac{1}{1+e^{-l_{prm}}}$.
    - **Design Motivation**: PRM evaluates the correctness of intermediate steps (rather than just final results), enabling smaller models paired with a verifier to outperform direct reasoning of larger LLMs.

3. **Pairwise Preference Model**:

    - **Function**: Performs pairwise comparisons among the final candidates to select the optimal answer.
    - **Mechanism**: Also fine-tuned based on Qwen2.5-Math-1.5B. For any two candidates $(A, B)$, the preference score $S_{PM}(A \succ B) = \frac{1}{1+e^{-l_{pm}}}$ is calculated. For each candidate $A$, its comprehensive score is the average of pairwise preference scores with all other candidates: $S_{PM}(A) = \frac{1}{n-1}\sum_{j \neq i} S(A \succ X_j)$.
    - **Design Motivation**: A PRM yields very small scoring differences for candidates with highly similar structures, failing to distinguish them reliably. Pairwise comparisons better capture subtle differences because the model directly contrasts two solutions rather than scoring them independently.

4. **Random Greedy Search**:

    - **Function**: Serves as an alternative search strategy when PRMs scoring is imprecise.
    - **Mechanism**: Retains candidates whose difference from the peak score is within a threshold: $P(a_{\max}) - P(a_i) \leq \text{threshold}$, and randomly selects from them to continue the search.
    - **Design Motivation**: Since PRM provides coarse preference rankings rather than precise scores, introducing randomness can mitigate the impact of scoring errors.

### Loss & Training
- PRM training: Binary cross-entropy loss, where labels are obtained by manually annotating correct/incorrect reasoning paths generated by CoT/ToT.
- Preference Model training: Binary cross-entropy loss on pairwise combinations of (correct path, incorrect path), with labels indicating which path ranks higher.
- Both models are fully fine-tuned starting from Qwen2.5-Math-1.5B.

## Key Experimental Results

### Main Results

| Dataset | Method | Accuracy | Reasoning Steps |
|--------|------|--------|---------|
| StructuredOR | CoT | 0.633 | 1 |
| StructuredOR | SC | 0.700 | 4 |
| StructuredOR | ToT-Rethink | 0.766 | 40 |
| StructuredOR | **BPP-Search** | **0.933** | **15** |
| Mamo-ComplexLP | CoT | 0.486 | 1 |
| Mamo-ComplexLP | **BPP-Search** | **0.722** | **21** |
| NL4OPT | CoT | 0.566 | 1 |
| NL4OPT | ToT-Rethink | 0.622 | 40 |
| NL4OPT | **BPP-Search** | **0.804** | **15** |

### Ablation Study

| Configuration | StructuredOR | Mamo-ComplexLP | NL4OPT | Steps |
|------|-------------|---------------|--------|------|
| Greedy + PRM | 0.733 | 0.555 | 0.699 | 9 |
| Random Greedy + PRM | 0.833 | 0.513 | 0.692 | 9 |
| Beam Search (W=2) + PRM | 0.800 | 0.652 | 0.783 | 15 |
| Beam Search (W=3) + PRM | 0.766 | 0.666 | 0.755 | 21 |
| BPP-Search (W=2) | **0.933** | 0.652 | **0.804** | 15 |
| BPP-Search (W=3) | 0.866 | **0.722** | 0.797 | 21 |

### Key Findings
- **Increasing the Beam Search width can sometimes degrade accuracy** (e.g., W=3 is 7% lower than W=2 on StructuredOR). This indicates that imprecise PRM scoring can introduce more noise when handling more candidates.
- **The Pairwise Preference model effectively mitigates this issue**: BPP-Search (W=2) achieves 93.3% accuracy on StructuredOR, which is 13.3% higher than pure Beam Search + PRM.
- **BPP-Search achieves higher accuracy in 15 steps than ToT-Fully-Traverse in 40 steps**, significantly improving search efficiency.
- GPT-4o performs best under the ToT framework, whereas smaller models (e.g., Llama-3.2-11B) are virtually unable to accomplish such mathematical modeling tasks.

## Highlights & Insights
- The two-stage evaluation of **PRM + Pairwise Preference** is highly clever: PRM acts as a coarse filter to preserve search efficiency, while the Preference model functions as a fine ranker to resolve subtle differences among the final candidates. This "coarse filter + fine ranking" paradigm can be transferred to other reasoning tasks that require selecting the best option from multiple candidates.
- **Using manual annotation instead of MCTS to generate PRM training data**: MCTS requires massive rollouts and risks reward hacking. Although manual annotation incurs higher costs, it provides more reliable data quality.
- **Random Greedy as a simple baseline when PRM is imprecise** is highly practical—only retaining candidates close to the maximum score and randomly choosing among them performs better than standard Greedy in certain cases.

## Limitations & Future Work
- **Small dataset scale**: StructuredOR comprises only 124 problems, which limits the generalizability of the findings.
- **Tree width and depth are constrained by computational cost**: The current setup of a maximum of 3 child nodes and a depth of 4 layers restricts performance; larger trees might yield better results but are too expensive.
- **High requirements on Policy Model capabilities**: Small models (e.g., Llama-3.2-11B) cannot generate valid modeling plans under the ToT framework, limiting the applicability of the method.
- **The PRM and Preference Model are quite small (1.5B)**: Larger verifier models might further boost performance.
- Extending Pairwise Preference to the intermediate layers of the tree, rather than just the final layer, could be explored.

## Related Work & Insights
- **vs CoT/SC**: CoT is a single path and is error-prone, while SC offers multiple paths but lacks intermediate verification. BPP-Search enables multi-path exploration and intermediate-step evaluation via a tree structure and process rewards.
- **vs MCTS-based PRM**: MCTS methods (e.g., Math-Shepherd) require massive rollouts to generate training data. BPP-Search uses deterministic data from manual annotation, avoiding reward hacking.
- **vs ToT-Rethink**: ToT-Rethink hands all leaf nodes back to the LLM for re-evaluation. BPP-Search relies on a specially trained small model for evaluation, which is more reliable and computationally cheaper.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of PRM + Pairwise Preference within the ToT framework is relatively novel, though the individual components themselves are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets and ablations over various search variants are provided, but the datasets are on the smaller side.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed methodological descriptions.
- Value: ⭐⭐⭐ Primarily targets the niche field of operations research mathematical modeling, but the "coarse filter + fine ranking" search paradigm holds some general value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ACL 2025\] Dynamic and Generalizable Process Reward Modeling (DG-PRM)](dgprm_dynamic_process_reward.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [\[ICLR 2026\] Generalization in LLM Problem Solving: The Case of the Shortest Path](../../ICLR2026/llm_reasoning/generalization_in_llm_problem_solving_the_case_of_the_shortest_path.md)

</div>

<!-- RELATED:END -->
