---
title: >-
  [Paper Note] Dynamic and Generalizable Process Reward Modeling (DG-PRM)
description: >-
  [ACL 2025][Reasoning][process reward model] The DG-PRM framework is proposed, which dynamically stores and selects multi-dimensional evaluation criteria by building a hierarchical reward tree, and identifies positive and negative sample pairs under multiple objectives in combination with Pareto dominance estimation, achieving dynamic and generalizable process reward modeling.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "process reward model"
  - "reward tree"
  - "Pareto dominance"
  - "LLM-as-judge"
  - "dynamic evaluation"
date: 2026-05-08
content_hash: b9f00fbb24aad118
---

# Dynamic and Generalizable Process Reward Modeling (DG-PRM)

**Conference**: ACL 2025  
**arXiv**: [2507.17849](https://arxiv.org/abs/2507.17849)  
**Code**: Not publicly available  
**Area**: LLM Reasoning/Reward Modeling  
**Keywords**: process reward model, reward tree, Pareto dominance, LLM-as-judge, dynamic evaluation  

## TL;DR

The DG-PRM framework is proposed, which dynamically stores and selects multi-dimensional evaluation criteria by building a hierarchical reward tree, and identifies positive and negative sample pairs under multiple objectives in combination with Pareto dominance estimation, achieving dynamic and generalizable process reward modeling.

## Background & Motivation

- **Problem Definition**: Process Reward Models (PRMs) provide dense reward signals for each intermediate step of LLMs during complex reasoning, which is crucial for improving reasoning quality.
- **Limitations of Heuristic PRMs**: Rely on manually crafted, fixed evaluation criteria (such as answer correctness), require objective reference answers, exhibit poor cross-domain generalization, and are susceptible to reward hacking.
- **Limitations of Generative PRMs**: Although they leverage LLM-as-Judge to provide feedback, existing methods only utilize final judgments (correct/incorrect), ignoring the rich detailed information contained within the judgment texts (such as error severity and error types).
- **Key Observation**: LLM judgment feedback contains rich multi-dimensional instructional information (e.g., logical consistency, computational accuracy), yet current methods assign negative rewards uniformly to all erroneous steps, failing to distinguish the severity of different errors.

## Method

### Overall Architecture

DG-PRM comprises three core modules: (1) Automatic process reward design, which extracts multi-dimensional evaluation criteria from LLM judgments and organizes them into a hierarchical reward tree; (2) Dynamic process reward allocation, which dynamically selects relevant criteria from the reward tree based on the content of each step for scoring; (3) Multi-objective reward optimization, which employs Pareto dominance to identify positive and negative sample pairs for step-wise DPO training.

### Key Designs

- **Reward Tree Construction**: Uses an LLM Judge to analyze differences and extract evaluation criteria $R_{raw}$ for positive and negative output pairs $(y_+, y_-)$ $\rightarrow$ filters low-quality criteria $\rightarrow$ maps criteria to vector space with a text encoder $\rightarrow$ builds a tree structure $\mathcal{T}$ (coarse-grained parent nodes + fine-grained child nodes) via incremental hierarchical clustering, where criteria with a cosine distance below the threshold $\xi$ are merged for deduplication.
- **Dynamic Reward Allocation**: When evaluating step $y^{(t)}$, relevant parent criteria are first selected from the top level of the reward tree $\rightarrow$ an analysis function $\Phi$ determines whether fine-grained evaluation is necessary $\rightarrow$ child node criteria are matched using cosine distance (distance $< \zeta$) $\rightarrow$ a sliding window $\mu$ is introduced to utilize reward contextual information from preceding steps.
- **Pareto Dominance Optimization**: For multiple candidate outputs at the same step, the Pareto frontier is computed under multi-dimensional reward scores $\rightarrow$ Pareto-optimal solutions serve as positive samples while dominated solutions serve as negative samples $\rightarrow$ preference pairs are constructed for step-wise DPO training.

### Loss & Training

Step-wise optimization objective based on DPO:

$$\mathcal{L}_{\text{DG-PRM}}(\theta) = -\mathbb{E}_{(\hat{y}_+^{(t)}, \hat{y}_-^{(t)}) \in \mathbf{V}} \left[\log \sigma\left(\beta \Delta^{(t)}\right)\right]$$

where $\Delta^{(t)} = r_\theta^{(t)}(\hat{y}_+^{(t)}) - r_\theta^{(t)}(\hat{y}_-^{(t)})$, and $r_\theta^{(t)}$ is the log-ratio between the policy and reference policy.

## Key Experimental Results

### Main Results (PRMBench)

| Model | Overall | Simplicity | Soundness Avg. | Sensitivity Avg. |
|------|---------|------------|----------------|------------------|
| Llemma-PRM800k-7B | 52.0 | 51.4 | 50.9 | 66.0 |
| RLHFlow-PRM-Mistral-8B | 54.4 | 46.7 | 57.5 | 68.5 |
| GPT-4o (Critic) | 66.8 | 59.7 | 70.9 | 75.8 |
| o1-mini (Critic) | 68.8 | 64.6 | 72.1 | 75.5 |
| DeepSeek-R1 (Critic) | 69.5 | 65.6 | 72.5 | 76.5 |
| **DG-PRM (o1-mini)** | **73.5** | **70.2** | **76.1** | - |

### Ablation Study

| Component | Effect |
|------|------|
| Removing Reward Tree (Fixed Criteria) | Performance decreases significantly; cross-domain generalization deteriorates |
| Removing Pareto Dominance (Random Positive/Negative Pairs) | Unclear training target, leading to performance degradation |
| Removing Dynamic Selection (Using All Criteria) | Noisy criteria interfere with scoring, leading to performance degradation |
| Removing Context Window | Loss of cross-step consistency signals |

### Key Findings

- DG-PRM significantly outperforms all open-source discriminative PRMs and LLM-as-Critic methods on PRMBench.
- Compared to directly employing LLMs as Critics, DG-PRM exhibits higher training efficiency and stronger capability to generalize to out-of-distribution (OOD) scenarios.
- The hierarchical organization of the reward tree allows fine-grained criteria to be reused across different domains.
- Pareto dominance estimation provides a clearer optimization direction compared to simple positive/negative binary classification.

## Highlights & Insights

- For the first time, multi-dimensional detailed information within LLM Judge feedback is systematically utilized to construct process rewards.
- The reward tree structure elegantly resolves the issues of storage, deduplication, and dynamic retrieval of evaluation criteria.
- Pareto dominance estimation is a natural and effective solution for handling multi-objective reward signals.

## Limitations & Future Work

- The construction of the reward tree depends on the judgment quality of high-performance LLMs (e.g., GPT-4o/o1-mini), resulting in high API call costs.
- The threshold parameters for hierarchical clustering ($\xi, \zeta$) and the sliding window size $\mu$ require manual tuning.
- The experiments are primarily validated on mathematical reasoning and evaluation tasks; performance in other reasoning scenarios, such as code generation and creative writing, remains to be explored.
- As task domains expand, the reward tree may grow excessively large, which could affect retrieval efficiency.
- The discriminative capability of Pareto dominance may decrease in high-dimensional reward spaces (due to a large number of mutually non-dominated solutions).

## Related Work & Insights

- Outcome Reward Model (ORM): Stiennon et al. 2020; Ouyang et al. 2022
- Process Reward Model (PRM): Lightman et al. 2024; Wang et al. 2024a (Math-Shepherd)
- LLM-as-Judge: Zheng et al. 2023 (MT-Bench); Kwon et al. 2023
- Multi-Objective Optimization and Pareto: Miettinen 1999
- DPO: Rafailov et al. 2023

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OR-PRM: A Process Reward Model for Algorithmic Problem in Operations Research](../../ICLR2026/llm_reasoning/or-prm_a_process_reward_model_for_algorithmic_problem_in_operations_research.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ICLR 2026\] Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning](../../ICLR2026/llm_reasoning/linking_process_to_outcome_conditional_reward_modeling_for_llm_reasoning.md)
- [\[ACL 2025\] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [\[ACL 2025\] An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning](an_efficient_and_precise_training_data_construction_framework_for_process-superv.md)

</div>

<!-- RELATED:END -->
