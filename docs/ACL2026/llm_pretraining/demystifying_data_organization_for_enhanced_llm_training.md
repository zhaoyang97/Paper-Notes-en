---
title: >-
  [Paper Note] Demystifying Data Organization for Enhanced LLM Training
description: >-
  [ACL2026][LLM Pretraining][Data Ordering] This paper systematically investigates the influence of "sample appearance order" in LLM training. By reusing existing sample-level quality/difficulty scores…
tags:
  - "ACL2026"
  - "LLM Pretraining"
  - "Data Ordering"
  - "Curriculum Learning"
  - "Pre-training Efficiency"
  - "STR"
  - "SAW"
date: 2026-05-08
content_hash: 76233688b0096504
---

# Demystifying Data Organization for Enhanced LLM Training

**Conference**: ACL2026  
**arXiv**: [2605.30334](https://arxiv.org/abs/2605.30334)  
**Code**: None  
**Area**: LLM Pre-training / Data Organization  
**Keywords**: Data Ordering, Curriculum Learning, Pre-training Efficiency, STR, SAW

## TL;DR
This paper systematically investigates the influence of "sample appearance order" in LLM training. By reusing existing sample-level quality/difficulty scores, the authors propose four data organization principles: boundary sharpening, cyclic scheduling, curriculum continuity, and local diversity. These principles are implemented via STR and SAW to achieve stable performance gains in both pre-training and SFT.

## Background & Motivation
**Background**: LLM data engineering typically focuses on data collection, deduplication, filtering, mixing, synthesis, and selection. Many pipelines already compute quality, difficulty, educational value, or learnability scores for each sample to determine "which samples enter the training set."

**Limitations of Prior Work**: These scores are often used only as one-time filtering tools, while the training order itself is treated simply with random shuffling or naive curriculum strategies. For the common single-epoch or few-epoch training paradigms of modern LLMs, sample order directly affects the optimization trajectory: early samples determine how the model enters the training state, late samples determine which capability region the final model settles in, and abrupt distribution shifts in the middle can trigger forgetting or optimization oscillations.

**Key Challenge**: Data selection answers "what to train," whereas data organization answers "in what order to train." While the former has been extensively studied, the latter is frequently neglected. Under a fixed token budget, incorrect ordering can lead to significantly different learning outcomes for the identical dataset.

**Goal**: The authors aim to extend sample-level scores from "filtering tools" to "ordering signals," summarize generalizable data organization principles, and propose ordering strategies with negligible additional computational costs, covering general pre-training, mathematical SFT, and code SFT.

**Key Insight**: Instead of redesigning data scorers, the paper reuses scores derived from existing data efficiency methods. This allows the research to focus on the ordering function $f_o$: given the data and their scores, how to construct a training sequence that enables the model to start stably, conclude with high-value samples, and avoid catastrophic forgetting or local homogenization.

**Core Idea**: The approach does not change the data scale but only the sample arrangement. The training sequence is designed to satisfy terminal high-value, periodic review, attribute continuity, and local diversity simultaneously.

## Method
The paper decomposes data engineering into three stages: scoring, selection, and organization. The scoring function $g$ produces a score vector $\gamma$ for each sample; the selection function $f_s$ chooses a training subset via proportions or top-$K$ strategies; the data organization function $f_o$ maintains the sample count but constructs a permutation $\pi$ based on $\gamma$, resulting in $\mathcal{D}_{ord}=[x_{\pi(1)},x_{\pi(2)},\dots,x_{\pi(K)}]$. While standard Curriculum Learning (CL) simply sorts samples by scores in ascending order, this work investigates more granular sequential structures.

### Overall Architecture
The overall pipeline involves reusing existing data selection scores, designing multiple ordering operators around the training sequence, and validating them on FineWeb-Edu, QuRatedPajama, DeepMath-103K, and OpenCodeInstruct. The authors summarize their empirical findings into four guidelines, validating individual principles via SEG, FO, ZIG, and JIT, and finally combining them into two primary methods: STR and SAW.

STR and SAW are the final recommended strategies. STR combines G1, G2, and G4: it maintains global score trends, applies folding review in local transition regions, and incorporates local diversity. SAW builds upon STR by adding G3, replacing folding in transition regions with Zig-zag patterns to ensure a more continuous score curve.

### Key Designs
1.  **Boundary Sharpening and SEG**:
    -   **Function**: Control the data attributes at the start and end of training to ensure the model starts stably and concludes with high-quality or high-difficulty samples.
    -   **Mechanism**: SEG discretizes sorted data into several segments and allocates samples to different training stages based on score rank. In pre-training, "starting low and ending high" performs better; in SFT, using high-score data at both the beginning and end is optimal.
    -   **Design Motivation**: Samples in the final stage of training directly influence the final achievable capability. If the final stage contains only low-quality or low-difficulty samples, model progress stagnates at the critical tail end. Conversely, using high-score samples only at the start yields diminishing returns because, under a fixed data volume, it pushes low-score samples to the end.

2.  **Cyclic Scheduling / Curriculum Continuity with FO, ZIG**:
    -   **Function**: Prevent the loss of foundational capabilities after transitioning from easy to hard samples in a naive curriculum, while reducing optimization shocks caused by distribution shifts.
    -   **Mechanism**: FO divides sorted data into multiple folding layers using a stride, ensuring each cycle covers the full score spectrum so the model periodically reviews foundational samples. ZIG reverses the order of odd cycles in FO to create a continuous triangular-wave-like score trajectory, reducing attribute cliffs at cycle boundaries.
    -   **Design Motivation**: In CL, the PPL on low-score samples often rebounds after the model enters high-score regions in the latter half, indicating the forgetting of basic knowledge. FO enables review, but cycle switches can cause spikes in gradient norms; ZIG stabilizes training dynamics through continuous transitions.

3.  **Local Diversity with JIT, and Combined Strategies STR/SAW**:
    -   **Function**: Maintain global curriculum trends while avoiding excessive sample similarity within a single mini-batch or local window.
    -   **Mechanism**: JIT partitions sorted data into windows or buckets and shuffles samples within these local windows, preserving the relative order between buckets while restoring local heterogeneity. STR maintains monotonic trends in stable regions and injects FO in transition regions; SAW replaces FO with ZIG for smoother transitions between regions.
    -   **Design Motivation**: Strict ordering results in highly similar scores for adjacent samples, which reduces gradient diversity. Perturbation analysis of JIT shows it helps the model find flatter minima that are less sensitive to weight noise.

### Loss & Training
This paper does not propose a new model loss but rather strategies for training data sequencing. The training objectives follow the standard pre-training language modeling or SFT objectives of the respective models. The core variable is the data sequence. In experiments, general pre-training utilizes the Mistral architecture, while SFT uses official Qwen3 weights. Data sources include FineWeb-Edu, QuRatedPajama, DeepMath-103K, and OpenCodeInstruct. For each strategy, the authors compare against random ordering, CL, DELT, single-principle strategies, and cross-principle strategies, performing scaling-up experiments in a 50B-token setting.

## Key Experimental Results

### Main Results
| Strategy | FineWeb-Edu Avg. | DeepMath Avg. | OpenCode Avg. | Description |
| :--- | :--- | :--- | :--- | :--- |
| Random | 37.09 | 1.30 | 55.37 | Random ordering baseline |
| CL | 37.61 | 1.78 | 58.30 | Naive ascending curriculum; gains but unstable |
| DELT | 37.35 | 2.42 | 59.70 | Review-based baseline; strong on SFT |
| STR | 38.65 | 2.48 | 60.83 | Combines boundary, review, and local diversity; best on Code SFT |
| SAW | 38.78 | 2.53 | 60.48 | Adds continuity; best on Pre-training and Math SFT |

### Ablation Study
| Configuration | FineWeb-Edu | QuRatedPajama | DeepMath | OpenCodeInstruct | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CL | 37.61 | 36.12 | 1.78 | 58.30 | Naive sorting |
| CL (JIT) | 38.20 | 36.46 | 1.78 | 59.50 | Local perturbations improve PT and Code SFT |
| FO | 38.12 | 36.62 | 2.42 | 60.90 | Cyclic review significantly outperforms CL |
| FO (JIT) | 38.25 | 36.85 | 2.74 | 60.96 | JIT further improves Math SFT |
| ZIG | 38.29 | 36.74 | 2.69 | 60.11 | Continuous transitions mitigate FO mutations |
| ZIG (JIT) | 38.32 | 36.88 | 2.76 | 61.34 | Most stable single-principle combo; highest OpenCode |

### Key Findings
- Data order is a first-order factor in single-epoch or few-epoch training. By changing only the order without modifying the dataset, the FineWeb-Edu average score improves from 37.09 (Random) to 38.78 (SAW).
- The end of training is more critical than the beginning. SEG experiments demonstrate that ending with high-score data provides consistent gains; using high-score data only at the start shows minimal benefit because low-quality data is postponed to the end.
- Cyclic review mitigates forgetting. The PPL curve for FO-3 drops again when the second cycle re-introduces simple data, whereas CL shows a PPL rebound for low-score samples in the latter half.
- Continuity affects optimization stability. FO exhibits gradient norm spikes at cycle boundaries, which ZIG reduces by reversing odd cycles to minimize attribute shocks.
- Scaling-up results support extensibility. In 50B-token pre-training, the gain from ordering does not disappear with scale; SAW and STR consistently outperform Random across model sizes (160M to 1.7B).

## Highlights & Insights
- The most significant takeaway is that data scores should not only serve filtering. Since scoring is expensive, reusing those same scores to organize training order adds very little marginal cost.
- The philosophy of STR/SAW is more transferable than the specific algorithms. Any existing data selection pipeline that outputs sample-level scores can explore strategies like terminal high-scoring, cyclic review, and local perturbation.
- "Local diversity" is a point often overlooked by curriculum learning. A perfectly ordered curriculum may appear logical but leads to homogenized gradients within local batches; JIT recovers the benefits of randomness without disrupting the global trend.
- This paper compares pre-training and SFT under the same data organization framework, which is more representative than validation on small curriculum benchmarks.

## Limitations & Future Work
- The methodology depends on existing sample-level scores. If score quality is low or decoupled from the target task, STR/SAW may organize incorrect signals more "exquisitely" without yielding real gains.
- Experiments primarily cover linguistic data. The authors acknowledge the need for unbiased evaluations in other modalities, such as multimodal pre-training, audio data, or code-text mixed corpora.
- Results for large models include scaling law extrapolations. Table 7 provides test loss extrapolations for GPT-3 and Llama-scale models, but these are not empirical results from full training of those models.
- Ordering strategies may be tightly coupled with optimizers, batching, mixing ratios, and deduplication strategies. Future work could investigate online adaptive ordering instead of one-time offline sequence generation.

## Related Work & Insights
- **vs Curriculum Learning**: CL typically sorts by difficulty from easy to hard. This paper argues that monotonic ordering leads to the forgetting of foundational samples and that terminal low-quality data harms final performance.
- **vs DELT**: DELT introduced the concept of review via folding learning. This paper systematizes it into G2 and further incorporates continuity and local diversity to form STR/SAW.
- **vs Data Selection**: Data selection alters the sample set; data organization alters the permutation without changing the set. It can be layered onto data pipelines like SemDeDup or FineWeb-Edu.
- **vs Data Mixing**: Data mixing focuses on ratios between different sources or domains; data organization focuses on the temporal order within a chosen set. Combining both is a promising direction for future training recipes.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The perspective is highly practical, and the organization principles are clearly systematized; while individual techniques are not entirely new, their combination into an LLM training recipe is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers general pre-training, mathematical SFT, code SFT, various corpora, and scaling-up, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐☆ Structure is complete, though tables are very dense, and some mathematical notations may be less accessible to readers outside of data engineering.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for actual LLM training pipelines, particularly for low-cost improvements to existing data engineering workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](../../ICLR2026/llm_pretraining/common_corpus_ethical_data_for_llm_pretraining.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
