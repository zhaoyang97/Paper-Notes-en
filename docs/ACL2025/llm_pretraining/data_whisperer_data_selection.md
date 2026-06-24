---
title: >-
  [Paper Note] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning
description: >-
  [ACL 2025][LLM Pretraining][data selection] Data Whisperer proposes a training-free few-shot ICL data selection method using attention weighting. It leverages the pre-trained model's own ICL capabilities and attention scores to evaluate training samples, outperforming full-data fine-tuning with only 10% of the data while operating 7-20 times faster than existing methods.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "data selection"
  - "in-context learning"
  - "attention weighting"
  - "fine-tuning efficiency"
  - "coreset selection"
date: 2026-05-08
content_hash: 44ca7529826d803a
---

# Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning

**Conference**: ACL 2025  
**arXiv**: [2505.12212](https://arxiv.org/abs/2505.12212)  
**Code**: [https://github.com/gszfwsb/Data-Whisperer](https://github.com/gszfwsb/Data-Whisperer)  
**Area**: LLM Pre-training  
**Keywords**: data selection, in-context learning, attention weighting, fine-tuning efficiency, coreset selection

## TL;DR
Data Whisperer proposes a training-free few-shot ICL data selection method using attention weighting. It leverages the pre-trained model's own ICL capabilities and attention scores to evaluate training samples, outperforming full-data fine-tuning with only 10% of the data while operating 7-20 times faster than existing methods.

## Background & Motivation
**Background**: LLM fine-tuning requires training on task-specific data. As datasets grow increasingly larger, data selection (coreset selection) has become a critical challenge to balance performance and computational costs.

**Limitations of Prior Work**:
   - Existing methods (such as GraNd, EL2N, Nuggets) require first fine-tuning a scoring model on the target dataset, taking even more time than direct full-scale fine-tuning ($\text{STR} > 1$).
   - Heuristic methods (such as CCS) fail to fully exploit the predictive capabilities of the model.
   - Nuggets uses one-shot ICL for scoring, which has low computational efficiency (evaluating only one demonstration at a time).

**Key Challenge**: The overhead of data selection itself should be far smaller than fine-tuning overhead. However, the Selection-to-Tuning Ratio (STR) of existing methods generally exceeds 1, meaning that data selection takes even longer than fine-tuning.

**Goal**: To design a training-free data selection method that significantly reduces selection time while preserving election quality.

**Key Insight**: ICL is theoretically equivalent to implicit fine-tuning ($\text{ICL} \approx \text{implicit fine-tuning}$), meaning that ICL performance can be utilized to predict fine-tuning outcomes.

**Core Idea**: To evaluate the contribution value of each training sample to the task using the attention-weighted scores of demonstrations in few-shot ICL.

## Method

### Overall Architecture
The pipeline of Data Whisperer consists of two steps: (1) Few-shot ICL evaluation: randomly sample $n_d$ demonstrations and $n_q$ queries from the training set, perform ICL inference via the pre-trained model, and calculate the average performance score; (2) Context-Aware Weighting: weight the scores of demonstrations using attention scores to eliminate order sensitivity. This process is iteratively executed until all samples are scored, and the top-k samples are selected as the training subset.

### Key Designs

1. **Selection-to-Tuning Ratio (STR) Metric**:

    - **Function**: Defined as $\text{STR} = t_p(\tau, \rho) / t_{ft}$ to quantify the efficiency of data selection methods.
    - **Mechanism**: An $\text{STR} < 1$ implies that selection time is lower than full fine-tuning time, which indicates actual practical value.
    - **Design Motivation**: Exposes a critical flaw of existing methods—most of them have an $\text{STR} > 1$, making them slower than direct training on the full dataset.

2. **Few-shot ICL Scoring Mechanism**:

    - **Function**: Periodically samples $n_d$ demonstrations and $n_q$ queries from the dataset to perform ICL inference using the pre-trained model.
    - **Mechanism**: Combines demonstrations and queries into context $C$. After the model generates answers for the queries, the average performance score is calculated using task-specific metrics (e.g., Accuracy/ROUGE-L): $s = \frac{1}{n_q}\sum_{j=1}^{n_q} f(\hat{y}_q^{(j)}, y_q^{(j)})$.
    - **Design Motivation**: Based on the theory that $\text{ICL} \approx \text{implicit fine-tuning}$, the attention influence of a demonstration on a query during ICL is equivalent to the parameter update $\Delta W_{icl}$ in fine-tuning.

3. **Context-Aware Attention Weighting**:

    - **Function**: Weights the scores of demonstrations using self-attention scores to eliminate positional bias in ICL.
    - **Mechanism**: Extracts the sub-attention matrix between each demonstration and the queries from the $l$-th attention layer, sums across all attention heads, and normalizes by demonstration length: $w_{(x_d^{(i)}, y_d^{(i)})} = \sum_h \mathbf{1}^\top A_{(x_d^{(i)}, y_d^{(i)})}^{(h)} \mathbf{1}$.
    - **Design Motivation**: ICL is highly sensitive to demonstration ordering; directly assigning identical scores leads to systematic overestimation or underestimation of samples placed later in the sequence.

4. **Weak-to-Strong Strategy**:

    - **Function**: Employs a smaller model within the same family to perform ICL scoring (e.g., using Qwen-2.5-3B to score data for Qwen-2.5-7B).
    - **Mechanism**: Models within the same family share similar knowledge representations, allowing scoring results from smaller models to transfer well to larger models.
    - **Design Motivation**: To further reduce selection costs, lowering the STR to 0.03–0.17.

### Theoretical Analysis
Under linear attention approximation, the influence of demonstrations on predictions in ICL can be decomposed as: $\mathcal{M}_p(q) = (W_{zsl} + \Delta W_{icl})q$, where $\Delta W_{icl} = \sum_i (W_V x_d^{(i)}) \otimes (W_K x_d^{(i)})$. In comparison, fine-tuning is expressed as $\mathcal{M}_f(q) = (W_{zsl} + \Delta W_{ft})q$. The structural similarity between the two indicates that using ICL scoring as a proxy for fine-tuning performance is theoretically well-founded.

## Key Experimental Results

### Main Results
Evaluated on three datasets (GSM8K, DialogSum, BioInstruct) using three models (Llama-3-8B, Qwen-2.5-7B, Mistral-Nemo):

| Dataset | Model | Data Whisperer (10%) | Full Fine-Tuning | vs Random |
|--------|------|---------------------|----------|-----------|
| GSM8K | Llama-3-8B | 72.46 | 71.39 | +2.80 |
| GSM8K | Qwen-2.5-7B | 85.03 | 85.43 | +4.95 |
| DialogSum | Llama-3-8B | 42.18 | 43.33 | +0.73 |
| BioInstruct | Llama-3-8B | 39.20 | 40.21 | +0.50 |

Key Findings: On GSM8K, using only 10% of the data selected by Data Whisperer outperforms full fine-tuning (72.46 > 71.39).

### Efficiency Comparison

| Method | STR (GSM8K 10%) | STR (DialogSum 10%) | Speedup vs Nuggets |
|------|-----------------|--------------------|--------------------|
| GraNd | 1.08 | 1.11 | - |
| Nuggets | 1.26 | 2.53 | 1× |
| **Data Whisperer** | **0.17** | **0.25** | **7.4-10×** |

### Ablation Study

| Configuration | GSM8K 10% (Llama-3) | Description |
|------|---------------------|------|
| Full (nd=5, nq=5) | 72.46 | Full model |
| w/o attention weighting | ~70.0 | Stripping attention weighting drops performance by ~2% |
| nd=1 (similar to Nuggets) | ~69.5 | Degrades to one-shot with poor performance |
| Weak-to-strong (3B→7B) | Close to direct 7B scoring | Scores from the smaller model are transferable |

### Key Findings
- **STR metric exposes critical issues**: While the STR of prior works is systematically $>1$, Data Whisperer significantly reduces it to 0.03–0.17.
- **Outperforming full dataset with minor subsets**: On GSM8K, using a selected 10% subset outperforms fine-tuning on 100% of the data.
- **Crucial role of attention weighting**: Removing context-aware weighting leads to a substantial performance drop.
- **Few-shot > One-shot**: $n_d=5$ is significantly better than $n_d=1$, since multiple demonstrations provide much richer task-specific information.

## Highlights & Insights
- The **STR metric** serves as an excellent evaluation dimension. It highlights a neglected issue in the data selection community—that many selection methods take longer than fine-tuning itself, which limits their practical utility. This metric holds potential for analyzing other AutoML methods.
- The **theoretical bridge of $\text{ICL} \approx \text{Fine-tuning}$** elegantly links ICL scoring to fine-tuning outcomes. The clean mathematical derivation ($\Delta W_{icl}$ vs $\Delta W_{ft}$ under linear attention approximation) provides a solid theoretical foundation for ICL-based data selection.
- The **Weak-to-strong strategy** is a highly practical acceleration technique. Using a 3B model to score data for a 7B model further curtails costs with negligible performance degradation.

## Limitations & Future Work
- The theoretical analysis is based on a linear attention approximation, which still differs from practical softmax attention.
- Experiments were primarily verified on 7-8B models; the effectiveness on larger scales (e.g., 70B+) remains to be investigated.
- Selecting the attention layer $l$ relies on hyperparameters (targeting a fixed layer); adaptive layer selection could be a better alternative.
- The evaluation focused solely on LoRA; applicability to full parameter fine-tuning or other PEFT methods is yet to be validated.

## Related Work & Insights
- **vs Nuggets**: Nuggets utilizes one-shot ICL combined with a fine-tuned model for scoring, which incurs high computational overhead ($\text{STR} > 1$). Data Whisperer instead leverages few-shot ICL with a pre-trained model and attention weighting, resulting in a 7-20× speedup.
- **vs STAFF**: STAFF estimates gradients using small models for scoring, which still requires training. Data Whisperer is entirely training-free.
- **vs CCS**: CCS balances coverage and importance based on heuristics, whereas Data Whisperer is strictly data-driven.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the STR metric and attention-weighted ICL scoring is highly novel, though the theoretical base of ICL being equivalent to fine-tuning is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively verified across 3 datasets, 3 models, and multiple selection ratios, with detailed ablation and synthetic data studies.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical formulations, highly precise definition of the STR metric, and informative visualizations.
- Value: ⭐⭐⭐⭐ Highly practical. It brings a 7-20× speedup without dropping performance, offering high utility for realistic LLM fine-tuning pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](scar_style_consistency_data_selection.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ICLR 2026\] Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Fine-Tuning](../../ICLR2026/llm_pretraining/task-aware_data_selection_via_proxy-label_enhanced_distribution_matching_for_llm.md)
- [\[ACL 2025\] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)
- [\[ACL 2025\] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)

</div>

<!-- RELATED:END -->
