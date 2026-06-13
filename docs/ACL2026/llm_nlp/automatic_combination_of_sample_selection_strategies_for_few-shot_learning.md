---
title: >-
  [Paper Note] Automatic Combination of Sample Selection Strategies for Few-Shot Learning
description: >-
  [ACL 2026][LLM/NLP][Few-shot Learning] This paper proposes ACSESS, a method that automatically identifies and weight-combines complementary sample selection strategies through forward selection, backward selection…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Few-shot Learning"
  - "Sample Selection"
  - "Strategy Combination"
  - "In-context Learning"
  - "Meta-learning"
date: 2026-05-08
content_hash: 2333a49433903b0b
---

# Automatic Combination of Sample Selection Strategies for Few-Shot Learning

**Conference**: ACL 2026 Findings  
**arXiv**: [2402.03038](https://arxiv.org/abs/2402.03038)  
**Code**: [https://github.com/kinit-sk/ACSESS](https://github.com/kinit-sk/ACSESS)  
**Area**: LLM/NLP  
**Keywords**: Few-shot Learning, Sample Selection, Strategy Combination, In-context Learning, Meta-learning

## TL;DR

This paper proposes ACSESS, a method that automatically identifies and weight-combines complementary sample selection strategies through forward selection, backward selection, and Datamodels. Evaluated across 23 strategies, 5 ICL models, 3 gradient-based few-shot learning methods, 6 text datasets, and 8 image datasets, the combined strategies consistently outperform single strategies and ICL-specific baselines.

## Background & Motivation

**Background**: Few-shot learning faces a critical challenge in sample selection, as performance can fluctuate drastically depending on the chosen samples. Existing selection strategies often focus on a single attribute (e.g., similarity, diversity, informativeness), and many new strategies for In-context Learning (ICL) are effective but designed for specific scenarios with poor transferability.

**Limitations of Prior Work**: (1) Single-attribute strategies have inherent limitations—informative samples might be difficult to learn, while similar samples might lack diversity. (2) ICL-specific strategies (e.g., LENS, Active Prompt, EXPLORA, CASE) are optimized for specific scenarios and have limited generalization. (3) Classical supervised learning selection strategies (e.g., active learning, coreset selection) have been systematically overlooked in LLM contexts.

**Key Challenge**: A single sample attribute cannot comprehensively measure a sample's contribution to few-shot learning, yet the computational cost of exhaustively testing all strategy combinations is prohibitive.

**Goal**: Automatically identify complementary sample selection strategies and optimize their combination so that ensembles of classical strategies can match or exceed ICL-specific methods.

**Key Insight**: Borrow feature selection methods (Forward/Backward selection) and Datamodels concepts from traditional machine learning and elevate them from the sample level to the strategy level.

**Core Idea**: The quality of a sample cannot be measured by a single attribute; informativeness, representativeness, and learnability are complementary dimensions. Automatically combining strategies from these dimensions can select high-quality samples with complementary properties.

## Method

### Overall Architecture

ACSESS consists of three stages: (1) Defining a set of single-attribute strategies (covering informativeness, representativeness, and learnability); (2) Identifying high-contribution strategy subsets using three independent mechanisms—forward selection, backward selection, and Datamodels—and taking their intersection; (3) Calculating a composite score for each sample using a weighted combination to select the top $N$ samples.

### Key Designs

1.  **Tri-dimensional Strategy Definition (23 Single-attribute Strategies)**:
    - **Function**: Covers three complementary attributes of sample selection.
    - **Mechanism**: **Informativeness**—Similarity, Diversity, Active Learning (Entropy, Margin, Least Confidence, Loss), Coreset selection (CAL, DeepFool, GraNd, Graph-Cut). **Representativeness**—Herding, KCenter, CRAIG, Glister. **Learnability**—Forgetting frequency and Cartography (Hard/Easy/Ambiguous samples). Each strategy assigns a normalized score $[0,1]$ to each sample.
    - **Design Motivation**: Different few-shot learning methods have different preferences—ICL tends to favor hard-to-learn samples, while gradient-based learning favors easy-to-learn samples.

2.  **Three-way Independent Strategy Identification**:
    - **Function**: Efficiently finds the most valuable strategy subsets.
    - **Mechanism**: (a) **Forward Selection**: Starts from an empty set, iteratively adding strategies that yield the largest performance gain until no improvement is observed. (b) **Backward Selection**: Starts from the full set, iteratively removing strategies that do not decrease performance. (c) **Datamodels Selection**: Creates 150 random strategy combinations, evaluates them, trains a LASSO regression to predict combination performance, and retains strategies with positive weights. The final set is the intersection $S_{final} = S_F \cap S_B \cap S_D$.
    - **Design Motivation**: Any single selection method may be biased; the intersection ensures the most robust strategies are retained while minimizing the number of strategies for efficiency.

3.  **Weighted Combination and Scoring**:
    - **Function**: Merges multi-strategy scores into a single sample score.
    - **Mechanism**: $score(x) = \sum_{s \in S} w_s \cdot objective_s(x)$. Three weighting schemes: Uniform ( $w_s = 1/|S_{final}|$, low cost, high transferability), Datamodels Weights (using LASSO weights, dataset/model-specific, optimal performance), and Weighted with Randomness (usually performs worse).
    - **Design Motivation**: Uniform weighting provides a robust default (only $0.10$-$0.25\%p$ worse than weighted), while weighted combinations provide optimal performance when resources permit.

### Loss & Training

ACSESS itself does not involve model training but acts as a preprocessing step for sample selection. For ICL, selected samples are used as few-shot examples. For gradient-based few-shot learning (Prototypical Networks, MAML, Few-Shot Fine-Tuning), they form the support set. Evaluation uses a 5-way 5-shot setting, repeated over 5 data splits × 10 random seeds × 300/600 tasks.

## Key Experimental Results

### Main Results

**ACSESS vs. ICL-specific Baselines (Avg. Accuracy Gain on Text Datasets relative to Classic selection)**

| Method | Avg. ICL Gain (pp) | Type |
| :--- | :--- | :--- |
| ACSESS (Weighted) | +2.5 | Ours |
| CASE (Purohit et al., 2025) | +2.34 | ICL-specific |
| EXPLORA (Purohit et al., 2024) | +1.8 | ICL-specific |
| Active Prompt (Diao et al., 2024) | +1.6 | ICL-specific |
| LENS (Li & Qiu, 2023) | +1.55 | ICL-specific |
| Best Single (Cartography-Hard) | +2.0 | Single Strategy |
| Random selection | 0.0 | Baseline |

ACSESS achieves statistical significance across all comparisons via the Wilcoxon test.

### Ablation Study

**Impact of Sample Quantity on Selection Strategy Effectiveness**

| Number of Shots | ACSESS vs Random (ICL, pp) | ACSESS vs Random (Gradient, pp) |
| :--- | :--- | :--- |
| 1-shot | +4 ~ +7 | +7 |
| 5-shot | +2.5 | +1.8 |
| 20-shot | +10-12 (Old models) / +2-3 (New) | Peak Performance |
| 30-40-shot | Starts to regress | Regresses to Random |
| 50-shot | ICL performance drops | — |

**Impact of Dataset Size**
- ICL: Using only 25% (50 samples/class) matches full dataset selection performance.
- Gradient Learning: Using only 10% (20 samples/class) matches performance.
- At 10 samples/class, selection benefits decrease by 20-40%.

### Key Findings

- **Learnability is the most important attribute for few-shot learning**: ICL prefers hard-to-learn samples (Cartography-Hard), while gradient-based learning prefers easy/ambiguous samples and low forgetting frequency. Representativeness strategies were completely excluded from the final ACSESS selections.
- The optimal strategy combination identified by ACSESS varies by learning paradigm—ICL favors Cartography-Hard + Forgetting + Margin + Entropy; Gradient-based favors Cartography-Easy&Ambiguous + Forgetting + Margin + Graph-Cut.
- A uniform combination of Cartography + Margin (+ optional Forgetting) serves as a zero-cost default recommendation, performing slightly below full ACSESS.
- As the number of samples increases to 30-40, all strategies regress toward random selection, indicating that sample selection is primarily valuable in ultra-low data scenarios.
- More samples are not always better—ICL performance drops at 50+ shots, likely due to context length constraints.

## Highlights & Insights

- SYSTEMATICALLY compared 23 selection strategies across ICL and gradient-based learning in a unified framework for the first time.
- Elevating Datamodels from the sample level to the strategy level is an elegant abstraction—achieving effective search of the combination space at a lower computational cost.
- The importance ranking of "Learnability > Informativeness > Representativeness" subverts intuition; much prior work focused on similarity and diversity.
- The practical suggestion of uniformly combining Cartography + Margin lowers the barrier to entry for the method.
- The discovery that sample selection is crucial at low shot counts but loses effectiveness at larger scales provides direct practical guidance.

## Limitations & Future Work

- Assumes a sufficiently large annotated dataset is available for selection (up to 200 samples/class); true ultra-low resource scenarios may require different solutions.
- Only uses 5-way classification; ICL performance might degrade under higher class counts due to context limits.
- Did not perform extensive prompt engineering, which might underestimate certain strategies.
- High computational cost (approx. 2500 GPU hours on A100, 270 kgCO2).
- Future work could explore strategy selection in unlabeled scenarios and performance with larger-scale LLMs.

## Related Work & Insights

- **vs LENS (Li & Qiu, 2023)**: LENS uses a two-step search (informativeness + diversity); ACSESS automatically discovers optimal strategy combinations and performs better in most scenarios.
- **vs CASE (Purohit et al., 2025)**: The strongest ICL-specific baseline; ACSESS matches it with uniform combination and exceeds it by +0.16pp with weighted combination.
- **vs Datamodels (Ilyas et al., 2022)**: Original Datamodels operate at the sample level; ACSESS abstracts this to the strategy level, reducing computational complexity.

## Rating

- Novelty: ⭐⭐⭐⭐ Strategy-level automatic combination is a valuable methodological innovation, though the components (forward/backward selection, Datamodels) are existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 strategies × 5 ICL models × 3 gradient methods × 14 datasets × multiple repetitions; massive scale and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and explicit practical advice, though long.
- Value: ⭐⭐⭐⭐ Directly guides the practice of sample selection in few-shot learning; unified comparison fills an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)
- [\[ACL 2026\] Model-Agnostic Meta Learning for Class Imbalance Adaptation](model-agnostic_meta_learning_for_class_imbalance_adaptation.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2026\] OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning](toward_robust_in-context_learning_leveraging_out-of-distribution_proxies_for_tar.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](../../ICLR2026/llm_nlp/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)

</div>

<!-- RELATED:END -->
