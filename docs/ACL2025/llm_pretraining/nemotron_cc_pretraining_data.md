---
title: >-
  [Paper Note] Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset
description: >-
  [ACL 2025][LLM Pretraining][Common Crawl] Nemotron-CC constructs a 6.3T token long-horizon pretraining dataset (consisting of 4.4T unique real tokens + 1.9T synthetic tokens) from Common Crawl. It implements three core strategies: classifier ensembling to increase high-quality token recall, synthetic data rewriting to expand the count of unique tokens, and removing heuristic filters for high-quality data. In a 15T token training scenario, it enables an 8B model to achieve an…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Common Crawl"
  - "Data Quality"
  - "Classifier Ensembles"
  - "Synthetic Data"
  - "Long-Horizon Pretraining"
date: 2026-05-08
content_hash: 77fbc9672badfe50
---

# Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

**Conference**: ACL 2025  
**arXiv**: [2412.02595](https://arxiv.org/abs/2412.02595)  
**Code**: [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)  
**Institution**: NVIDIA  
**Area**: LLM Pretraining / Pretraining Data  
**Keywords**: Common Crawl, Data Quality, Classifier Ensembles, Synthetic Data, Long-Horizon Pretraining  

## TL;DR

Nemotron-CC constructs a 6.3T token long-horizon pretraining dataset (consisting of 4.4T unique real tokens + 1.9T synthetic tokens) from Common Crawl. It implements three core strategies: classifier ensembling to increase high-quality token recall, synthetic data rewriting to expand the count of unique tokens, and removing heuristic filters for high-quality data. In a 15T token training scenario, it enables an 8B model to achieve an MMLU of 70.3, surpassing Llama 3.1 8B (65.3) trained on the same scale.

## Background & Motivation

**Background**: English Common Crawl pretraining datasets (e.g., FineWeb-Edu, DCLM) have significantly improved benchmark scores under short training runs through aggressive model-based filtering. For instance, the DCLM-7B model, trained on 2.6T tokens, achieved performance close to closed-source models.

**Limitations of Prior Work**: Aggressive filtering discards approximately 90% of the data—DCLM contains only about 1T unique tokens, and FineWeb-Edu only 0.2T. In long training runs (like Llama 3.1's 15T tokens), this means the same sample must be seen over 15 times. However, Muennighoff et al. have shown that the returns from repeated data diminish sharply after 4 epochs.

**Key Challenge**: The trade-off between data quality and data quantity—high-quality filtering improves short-term benchmark scores but sacrifices the data diversity and total unique tokens required for long-horizon training.

**Key Insight**: Instead of pursuing "more aggressive filtering," this work simultaneously improves both quality and quantity through three complementary strategies: (1) multi-classifier ensembling to expand the recall of high-quality tokens, (2) synthetic rewriting to generate new unique tokens, and (3) skipping heuristic filtering for high-quality data to avoid false positives.

**Core Idea**: Use classifier ensembling + stratified synthetic data + selective filtering to break the bottleneck of the quality-quantity trade-off.

## Method

### Overall Architecture

HTML of 99 Common Crawl snapshots $\rightarrow$ **JusText extraction** (yields 28.6% more HQ tokens than Trafilatura) $\rightarrow$ English language filtering (pycld2 + FastText) $\rightarrow$ global fuzzy deduplication + exact substring deduplication $\rightarrow$ **three-classifier ensemble scoring** (each classifier outputs 0-19 points, taking the maximum) $\rightarrow$ **annealing experiment grading** (20 buckets $\rightarrow$ High/MH/M/ML/Low five-level grades) $\rightarrow$ skipping heuristic filtering for high-quality data while keeping it for low-quality data $\rightarrow$ **stratified synthetic data generation** (low-quality is rewritten in a Wikipedia style, high-quality is diversified using 4 types of prompts) $\rightarrow$ final yield of 6.3T token dataset (4.4T unique real + 1.9T synthetic).

### Key Designs

**1. Classifier Ensembling + Quality Bucketing**

- Built three heterogeneous quality classifiers: ① an educational quality classifier annotated by Nemotron-340B, ② an educational quality classifier annotated by Mixtral-8x22B, and ③ DCLM's fastText informativeness classifier. These three capture different dimensions (educational value vs. informativeness) with only a 10.1% intersection, showing strong complementarity.
- Each classifier ranks all documents and maps them into 0-19 integer buckets (about 5% of documents per bucket), taking the maximum of the three as the final quality score.
- **Design Motivation**: The HQ recall of a single classifier is only 8-14%, whereas the ensemble reaches 25%—this is key to breaking the long-horizon training data bottleneck. Concurrently, 20 buckets are restructured into 5 quality levels via annealing experiments (evaluating the downstream performance of each bucket using 50B tokens on a 70% pre-trained 8B model), thereby aligning quality labels directly with downstream performance rather than raw classifier scores.

**2. Stratified Synthetic Data Generation**

- **Low-quality data** (Low level, 402B tokens): Rewritten using a Wikipedia-style prompt to reduce noise/errors while preserving valuable information, producing 336B synthetic tokens.
- **High-quality data** (High level, 451B tokens): Generates diversified variants using four types of prompts—① Diverse QA Pairs (multi-format Q&A, 500B tokens), ② Extract Knowledge (knowledge extraction, 304B tokens), ③ Distill (distillation/compression, 158B tokens), ④ Knowledge List (structured knowledge lists, 203B tokens), totaling 1.5T tokens.
- Generation Model: Mistral NeMo 12B-instruct (FP8), accelerated with TensorRT-LLM. Long documents are segmented based on token limits before being generated chunk-by-chunk, with post-processing to remove incomplete outputs and formatting.
- **Design Motivation**: To create entirely new unique tokens for HQ data to avoid multi-epoch diminishing returns, rather than generating knowledge out of thin air using LLMs (which reduces hallucination risks).

**3. Selective Heuristic Filtering**

- Traditional approaches uniformly apply C4/Gopher/KenLM PPL filtering to all data. This work finds that these filters remove 18.1% of HQ tokens (under the FineWeb-Edu standard).
- This work proposes applying heuristic filters only to low-quality buckets and completely skipping them for high-quality buckets (which score high on model classifiers), achieving a +57.4% increase in HQ token yield (80B $\rightarrow$ 127B per 13 snapshots).
- Ablation studies confirm that this not only preserves quality but also improves MMLU by +2 (55.5 $\rightarrow$ 57.5).

## Key Experimental Results

### Main Results: 8B Model Trained on 1T Tokens Comparison

| Dataset | Unique Real Tokens | MMLU | ARC-C | Hellaswag | CSQA | 10-Task Avg. |
|---|---|---|---|---|---|---|
| FineWeb-Edu | 0.2T | 42.9 | 48.0 | 70.7 | 30.0 | 53.2 |
| FineWeb-Edu-2 | 1.1T | 42.4 | 44.7 | 75.4 | 25.5 | 53.2 |
| DCLM | 1.0T | 53.4 | 47.0 | 76.3 | 44.1 | 57.0 |
| Nemotron-CC (6.3T) | 4.4T | 53.0 | 50.7 | 75.9 | 47.7 | 57.8 |
| **Nemotron-CC-HQ (1.1T)** | **0.6T** | **59.0** | **52.9** | **76.6** | **55.8** | **60.1** |

Nemotron-CC-HQ outperforms DCLM by **+5.6 MMLU** and **+3.1 on average** in 1T short training runs. The complete Nemotron-CC matches DCLM in quality but possesses **4x** the number of unique tokens.

### Long-Horizon Training: 8B Model with 15T Tokens

| Model | MMLU | ARC-C | Hellaswag | Winogrande | CSQA | 10-Task Avg. |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | 65.3 | 55.0 | 79.3 | 74.7 | 70.6 | 64.2 |
| **Nemotron-CC 8B** | **70.3** | **58.1** | **80.8** | 73.8 | 69.9 | **64.7** |

In 15T token training, Nemotron-CC leads Llama 3.1 by **+5.0** in MMLU, validating the decisive advantage of more unique tokens in long-horizon training.

### Ablation Study: Contribution of Each Module

| Ablation Comparison | Key Findings |
|---|---|
| Extractor: JusText vs. Trafilatura | JusText yields 38.8% more total tokens and 28.6% more HQ tokens, with no loss in downstream performance |
| Filtering: No Filtering for HQ vs. Full Filtering | Leaving HQ unfiltered yields +2.0 MMLU (55.5 $\rightarrow$ 57.5) and a +57.4% increase in HQ token yield |
| Classifier: Single vs. Ensemble | Ensembling increases the HQ ratio from 8–14% to 25%, achieving the highest average score (59.4) |
| Synthetic Data: LQ Rewriting | Low-quality rewriting increases the average score by +1.5 (52.5 $\rightarrow$ 54.0) |
| Synthetic Data: HQ Diversification | Replacing 4/8 repeated epochs with synthetic data yields +0.9 average score (55.8 $\rightarrow$ 56.7) |

### Classifier Complementarity Analysis

| Classifier Combination | Document Count | % of Cohort High Quality Union |
|---|---|---|
| Overlap of Two Classifiers | 1.15M | 10.1% |
| Exclusive to FineWeb-Edu | 4.02M | 35.4% |
| Exclusive to DCLM | 6.18M | 54.4% |
| Union Set (Total) | 11.36M | 100% |

## Rating

| Dimension | Score (1-10) | Description |
|---|---|---|
| Practicality | 9 | The dataset is fully open-sourced, released split by quality grades, and reproducible via the open-source NeMo-Curator package; it is directly usable for large-scale pretraining. |
| Novelty | 7 | While individual sub-techniques (ensembling, rewriting, filtering) are not entirely new, their systematic combination and the "selective filtering" concept are highly inspiring. |
| Experimental Thoroughness | 9 | Broad comparisons across 1T/15T training horizons and detailed four-dimensional ablations on extractors, filters, classifiers, and synthetic data provide a comprehensive picture across 10 benchmarks. |
| Reproducibility | 9 | Datasets, classifiers, and the codebase have been fully released with detailed training hyper-parameters (Appendix D). The only entry barrier is compute resources. |

## Highlights & Insights

**Highlights**:
- The concept of "shifting from static heuristic pipelines to learning-based flywheels" is forward-looking: stronger models $\rightarrow$ better data quality $\rightarrow$ even stronger models.
- Annealing experiments directly define quality grading using downstream performance instead of relying on raw classifier scores.
- Synthetic data does not generate new knowledge but instead rewrites, distills, or diversifies existing content, which reduces hallucination risks.
- Skipping heuristic filtering for HQ data is a "counter-intuitive" strategy that is simple yet effective.
- Releasing the dataset split by quality level + synthesis type facilitates curriculum learning experiments for the community.

**Limitations**:
- Covers only English Common Crawl; multilingual expansion is not addressed.
- Synthetic data is generated by a 12B model without factual accuracy verification; introducing stronger models and fact-checkers could yield further improvements.
- 15T training is validated only on the 8B model; transferability to 70B+ models remains unverified.
- Dataset decontamination is not performed; although baselines did not do this either, it remains a potential confounding variable.
- Medium-level data did not have synthetic data generated (due to resource constraints), leaving room for further improvement.
- Deduplication strategies may still not be aggressive enough, leaving semantic-level near-duplicates.

## Related Work & Insights
- **vs DCLM**: DCLM uses a single fastText classifier and retains only about 10% of the data; Nemotron-CC uses a three-classifier ensemble to retain more high-quality data.
- **vs FineWeb-Edu**: FineWeb-Edu extracts aggressively down to 0.2T unique tokens, while Nemotron-CC retains 4.4T.
- **vs DSIR/QuRating**: These focus on data selection but do not perform synthetic expansion, whereas Nemotron-CC performs both selection and synthesis.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination strategy of classifier ensembling + synthetic data + reduced filtering is innovative in the pretraining data domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1T and 15T training + detailed ablation + outperforming Llama 3.1.
- Writing Quality: ⭐⭐⭐⭐ Clear, organized, and rich in tables.
- Value: ⭐⭐⭐⭐⭐ A 6.3T open-source dataset + experimental results surpassing Llama 3.1, offering extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset](../../ICLR2026/llm_pretraining/nemotron-cc-math_a_133_billion-token-scale_high_quality_math_pretraining_dataset.md)
- [\[ICLR 2026\] Pretraining with Hierarchical Memories: Separating Long-Tail and Common Knowledge](../../ICLR2026/llm_pretraining/pretraining_with_hierarchical_memories_separating_long-tail_and_common_knowledge.md)
- [\[ACL 2025\] Data Caricatures: On the Representation of African American Language in Pretraining Corpora](data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](../../ICML2026/llm_pretraining/on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ACL 2025\] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)

</div>

<!-- RELATED:END -->
