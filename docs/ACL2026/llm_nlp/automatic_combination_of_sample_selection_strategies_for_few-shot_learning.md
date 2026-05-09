---
title: >-
  [Paper Note] Automatic Combination of Sample Selection Strategies for Few-Shot Learning
description: >-
  [ACL 2026][LLM/NLP][few-shot learning] This paper proposes ACSESS, a method that automatically identifies complementary sample selection strategies and combines them via weighted aggregation, using three mechanisms: forward selection, backward selection, and Datamodels. Experiments across 23 strategies, 5 ICL models, 3 gradient-based few-shot learning methods, 6 text datasets, and 8 image datasets demonstrate that combined strategies consistently outperform individual strategies and ICL-specific baselines.
tags:
  - ACL 2026
  - LLM/NLP
  - few-shot learning
  - sample selection
  - strategy combination
  - in-context learning
  - meta-learning
date: 2026-05-08
content_hash: 82b5c89bc354d49d
---

# Automatic Combination of Sample Selection Strategies for Few-Shot Learning

**Conference**: ACL 2026
**arXiv**: [2402.03038](https://arxiv.org/abs/2402.03038)
**Code**: [https://github.com/kinit-sk/ACSESS](https://github.com/kinit-sk/ACSESS)
**Area**: LLM/NLP
**Keywords**: few-shot learning, sample selection, strategy combination, in-context learning, meta-learning

## TL;DR

This paper proposes ACSESS, a method that automatically identifies complementary sample selection strategies and combines them via weighted aggregation, using three mechanisms: forward selection, backward selection, and Datamodels. Experiments across 23 strategies, 5 ICL models, 3 gradient-based few-shot learning methods, 6 text datasets, and 8 image datasets demonstrate that combined strategies consistently outperform individual strategies and ICL-specific baselines.

## Background & Motivation

**Background**: Few-shot learning faces a critical challenge in sample selection—performance can vary dramatically depending on which samples are chosen. Existing selection strategies typically focus on a single attribute (e.g., similarity, diversity, or informativeness), while many newly proposed ICL-specific strategies, though effective, are often designed for particular scenarios and exhibit limited transferability.

**Limitations of Prior Work**: (1) Single-attribute strategies each have inherent limitations—the most informative samples may be difficult to learn from, while the most similar samples may lack diversity. (2) ICL-specific strategies (e.g., LENS, Active Prompt, EXPLORA, CASE) are optimized for specific settings and generalize poorly. (3) Classical supervised selection strategies (e.g., active learning, coreset selection) have been systematically overlooked in LLM contexts.

**Key Challenge**: No single sample attribute can comprehensively capture a sample's contribution to few-shot learning, yet exhaustively evaluating all strategy combinations is computationally prohibitive.

**Goal**: To automatically identify complementary sample selection strategies and optimize their combination, such that ensembles of classical selection strategies can match or surpass ICL-specific strategies.

**Key Insight**: Drawing inspiration from feature selection methods in traditional machine learning (forward/backward selection) and the Datamodels framework, the paper lifts these techniques from the sample level to the strategy level.

**Core Idea**: The quality of a sample cannot be measured by a single attribute—informativeness, representativeness, and learnability are complementary dimensions. Automatically combining strategies along these dimensions yields samples with diverse and mutually reinforcing properties.

## Method

### Overall Architecture

ACSESS operates in three stages: (1) defining a set of single-attribute strategies covering three property families—informativeness, representativeness, and learnability; (2) independently identifying high-contribution strategy subsets via forward selection, backward selection, and Datamodels, then taking their intersection; and (3) computing a composite score for each sample via weighted combination and selecting the top-$N$ samples.

### Key Designs

1. **Three-Dimensional Strategy Definition (23 Single-Attribute Strategies)**:

    - **Function**: Covers three complementary attributes of sample selection.
    - **Mechanism**: **Informativeness**—similarity, diversity, active learning strategies (Entropy, Margin, Least Confidence, Loss), and coreset selection methods (CAL, DeepFool, GraNd, Graph-Cut). **Representativeness**—Herding, KCenter, CRAIG, Glister. **Learnability**—Forgetting (forgetting frequency) and Cartography (hard-to-learn / easy-to-learn / ambiguous samples). Each strategy assigns a normalized score in $[0,1]$ to each sample.
    - **Design Motivation**: Different few-shot learning paradigms favor different sample properties—ICL tends to prefer hard-to-learn samples, while gradient-based methods prefer easy-to-learn samples.

2. **Three-Path Strategy Identification Mechanism**:

    - **Function**: Efficiently identifies the most valuable subset of strategies.
    - **Mechanism**: (a) **Forward Selection**: Starting from an empty set, iteratively adding the strategy that yields the largest performance gain, until no positive gain remains. (b) **Backward Selection**: Starting from the full set, iteratively removing strategies that do not degrade performance. (c) **Datamodels Selection**: Generating 150 random strategy combinations and evaluating them, training a LASSO regression to predict combination performance, and retaining strategies with positive weights. The final strategy set is the intersection of all three: $S_{final} = S_F \cap S_B \cap S_D$.
    - **Design Motivation**: Any single selection method may introduce bias; the three-path intersection ensures only the most robust strategies are retained while minimizing the number of selected strategies for efficiency.

3. **Weighted Combination and Scoring**:

    - **Function**: Aggregates scores from multiple strategies into a single per-sample score.
    - **Mechanism**: $score(x) = \sum_{s \in S} w_s \cdot objective_s(x)$. Three weighting schemes are considered—uniform weighting ($w_s = 1/|S_{final}|$, low computational cost, high transferability), Datamodels weighting (using LASSO regression weights, dataset/model-specific, best performance), and weighted combination with randomness (additional random scoring, generally inferior).
    - **Design Motivation**: Uniform weighting provides a robust default (performance gap of only 0.10–0.25 percentage points vs. weighted), while weighted combination offers optimal performance when resources permit.

### Loss & Training

ACSESS itself does not involve model training; it serves as a preprocessing step for sample selection. For ICL, selected samples are directly used as few-shot demonstrations. For gradient-based few-shot learning (Prototypical Networks, MAML, Few-Shot Fine-Tuning), selected samples are used as the support set. Evaluation follows a 5-way 5-shot setup, with each experiment repeated over 5 data splits × 10 random seeds × 300/600 tasks.

## Key Experimental Results

### Main Results

**ACSESS vs. ICL-Specific Baselines (Average Accuracy Gain over Classic Selection on Text Datasets, in pp)**

| Method | Avg. ICL Gain (pp) | Type |
|---|---|---|
| ACSESS (weighted) | +2.5 | Ours |
| CASE (Purohit et al., 2025) | +2.34 | ICL-specific |
| EXPLORA (Purohit et al., 2024) | +1.8 | ICL-specific |
| Active Prompt (Diao et al., 2024) | +1.6 | ICL-specific |
| LENS (Li & Qiu, 2023) | +1.55 | ICL-specific |
| Best single strategy (Cartography-Hard) | +2.0 | Single strategy |
| Random selection | 0.0 | Baseline |

ACSESS achieves statistical significance over all comparisons via the Wilcoxon test.

### Ablation Study

**Effect of Shot Count on Selection Strategy Performance**

| Shots | ACSESS vs. Random (ICL, pp) | ACSESS vs. Random (Gradient, pp) |
|---|---|---|
| 1-shot | +4 ~ +7 | +7 |
| 5-shot | +2.5 | +1.8 |
| 20-shot | +10–12 (older models) / +2–3 (newer models) | Peak performance |
| 30–40-shot | Begins to regress | Regresses to random |
| 50-shot | ICL performance declines | — |

**Effect of Dataset Size**
- ICL: Using only 25% of data (50 samples/class) matches full-dataset selection performance.
- Gradient-based learning: Using only 10% (20 samples/class) suffices.
- Reducing to 10 samples/class decreases selection benefit by 20–40%.

### Key Findings

- **Learnability is the most important sample attribute for few-shot learning**: ICL favors hard-to-learn samples (Cartography-Hard), while gradient-based methods favor easy-to-learn and ambiguous samples with low forgetting frequency. Representativeness strategies are entirely excluded from ACSESS's final selection.
- The optimal strategy combination identified by ACSESS varies by learning paradigm—ICL tends toward Cartography-Hard + Forgetting + Margin + Entropy; gradient-based learning tends toward Cartography-Easy&Ambiguous + Forgetting + Margin + Graph-Cut.
- Uniformly combining Cartography + Margin (optionally with Forgetting) serves as a recommended default with zero additional computational cost, achieving only marginally lower performance than the full ACSESS pipeline.
- Once the shot count exceeds 30–40, all strategies regress to random-selection levels, indicating that sample selection is primarily valuable in extremely low-data regimes.
- More samples are not always better—ICL performance degrades at 50+ shots, likely due to context length constraints.

## Highlights & Insights

- This work is the first to systematically compare 23 sample selection strategies across both ICL and gradient-based few-shot learning within a unified framework, filling an important gap in the literature.
- Lifting the Datamodels framework from the sample level to the strategy level represents an elegant abstraction, enabling effective search over the combinatorial space at manageable computational cost.
- The importance ordering "learnability > informativeness > representativeness" is counterintuitive—prior work has predominantly focused on similarity and diversity.
- The practical recommendation of uniformly combining Cartography + Margin lowers the barrier to adoption.
- The finding that sample selection matters in low-shot regimes but becomes ineffective at larger shot counts provides direct guidance for practitioners.

## Limitations & Future Work

- The method assumes a sufficiently large labeled pool for selection (up to 200 samples/class); truly low-resource scenarios require different approaches.
- Experiments are limited to 5-way classification; ICL performance may degrade under higher class counts due to context length constraints.
- Extensive prompt engineering was not performed, potentially underestimating the effectiveness of certain strategies.
- Computational cost is substantial (approximately 2,500 A100 GPU hours, 270 kgCO₂).
- Future work may explore strategy selection in unlabeled settings and behavior under larger-scale LLMs.

## Related Work & Insights

- **vs. LENS (Li & Qiu, 2023)**: LENS employs a two-step search (informativeness + diversity); ACSESS automatically discovers the optimal strategy combination and outperforms LENS in most settings.
- **vs. CASE (Purohit et al., 2025)**: The strongest ICL-specific baseline; ACSESS with uniform weighting matches it, and weighted ACSESS surpasses it by +0.16 pp.
- **vs. Datamodels (Ilyas et al., 2022)**: The original Datamodels operates at the sample level; ACSESS abstracts it to the strategy level, reducing computational complexity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Automatic strategy-level combination is a valuable methodological contribution, though the individual components (forward/backward selection, Datamodels) are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 23 strategies × 5 ICL models × 3 gradient-based methods × 14 datasets × multiple repetitions; the scale is substantial and ablations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and practical recommendations are well-articulated, though the paper is lengthy.
- **Value**: ⭐⭐⭐⭐ Provides direct practical guidance for sample selection in few-shot learning; the unified comparison fills an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](../../ICLR2026/llm_nlp/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)

</div>

<!-- RELATED:END -->
