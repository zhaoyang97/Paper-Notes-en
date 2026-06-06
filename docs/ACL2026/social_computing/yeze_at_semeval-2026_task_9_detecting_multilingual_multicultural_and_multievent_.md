---
title: >-
  [Paper Note] YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling
description: >-
  [ACL2026][Social Computing][Multilingual Polarization Detection] The YEZE system decomposes the 22-language online polarization identification of SemEval-2026 Task 9 into independent sub-tasks. It fine-tunes XLM-RoBERTa-…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Multilingual Polarization Detection"
  - "SemEval"
  - "Class Imbalance"
  - "XLM-R"
  - "Heterogeneous Ensembling"
date: 2026-05-08
content_hash: c81ca4660885bfad
---

# YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling

**Conference**: ACL2026  
**arXiv**: [2605.06231](https://arxiv.org/abs/2605.06231)  
**Code**: https://github.com/FezeGo/SemEval-2026-Task9-Polar  
**Area**: multilingual_mt / Multilingual Content Safety / Social Media Analysis  
**Keywords**: Multilingual Polarization Detection, SemEval, Class Imbalance, XLM-R, Heterogeneous Ensembling

## TL;DR
The YEZE system decomposes the 22-language online polarization identification of SemEval-2026 Task 9 into independent sub-tasks. It fine-tunes XLM-RoBERTa-large and mDeBERTa-v3-base separately and alleviates multi-label sparsity through weighted probability averaging and weighted BCE, achieving a stable official Top-10 ranking in fine-grained polarization type and manifest prediction.

## Background & Motivation
**Background**: Online polarization detection has expanded from binary classification of "is polarized" to more granular multi-label problems: identifying the polarized target and how the polarization is manifested linguistically. SemEval-2026 Task 9 places this problem within the context of 22 languages and multiple cultures/events, requiring systems to handle three sub-tasks: POLARDETECT, POLARTYPE, and POLARMANIFEST.

**Limitations of Prior Work**: The distribution of multilingual social media data is highly imbalanced. The proportion of positive polarization instances varies significantly across languages, and fine-grained labels are extremely sparse. When Macro-F1 is used as the metric, failures in minority classes lead to significant penalties. Furthermore, while Multi-Task Learning (MTL) seemingly shares representations, coarse-grained binary tasks often dominate the gradients, drowning out fine-grained target and manifest labels.

**Key Challenge**: The system must share cross-lingual capabilities without allowing interference between different languages, sub-tasks, and labels. It must improve the recall of low-resource labels without compromising precision through over-prediction.

**Goal**: The authors aim to build a robust shared task submission system that maintains stable performance across all languages and three sub-tasks. This includes selecting robust multilingual encoders, designing optimization for class-imbalanced multi-label tasks, comparing independent modeling vs. MTL, and analyzing which languages and labels remain difficult.

**Key Insight**: Instead of turning to Large Language Model (LLM) prompting, the paper returns to supervised multilingual encoders. The authors argue that under conditions of low resources, class imbalance, and fine-grained multi-labeling, controllable XLM-R/mDeBERTa fine-tuning is more stable than LLM generative approaches.

**Core Idea**: Replace shared MTL with "independent modeling per sub-task + weighted BCE + heterogeneous ensembling of XLM-R/mDeBERTa" to separately handle cross-lingual robustness and fine-grained label stability.

## Method
As a system description paper, the focus is not on new model architectures but on a robust engineering pipeline for the multilingual polarization shared task. The system targets three outputs: Subtask 1 (binary classification), Subtask 2 (multi-label prediction of five polarization targets), and Subtask 3 (multi-label prediction of six polarization manifestation types). Subtasks 1/2 cover 22 languages, while Subtask 3 covers 18 languages.

### Overall Architecture
The pipeline is straightforward: first, train XLM-RoBERTa-large and mDeBERTa-v3-base separately for each sub-task. Binary relevance is used for multi-label sub-tasks, treating each label as an independent binary classification. Weighted BCE is applied to handle category sparsity. Ensemble weights for XLM-R and mDeBERTa are searched on the dev set, ultimately adopting a probability average with $\alpha=0.7$. A global threshold of $\tau=0.5$ is used during inference; per-label threshold tuning was discarded as it decreased Macro-F1 under extreme sparsity.

The authors also implemented two baselines: single-backbone models and an MTL version with a shared XLM-R encoder and task-specific heads. The final submission uses the heterogeneous ensemble of independently trained sub-tasks, as it proved most stable across the average of three tasks, particularly in protecting fine-grained multi-label signals.

### Key Designs
1.  **Independent Modeling per Sub-task**:
    - **Function**: Prevents interference between binary classification and fine-grained multi-label tasks within a shared encoder.
    - **Mechanism**: Models are trained separately for the three tasks instead of optimizing POLARDETECT, POLARTYPE, and POLARMANIFEST using one shared encoder. Subtasks 2/3 are further decomposed into multiple binary label predictions, each outputting an independent sigmoid probability.
    - **Design Motivation**: Training signals for Task 1 are denser and labels are coarser; Task 2/3 labels are sparser. Shared updates would bias representation toward the binary boundary, causing negative transfer for fine-grained labels.

2.  **Multi-label Sparsity Optimization via Weighted BCE**:
    - **Function**: Increases the learning weight of minority labels under the Macro-F1 metric to alleviate extreme class imbalance.
    - **Mechanism**: Positive class weights are calculated for each label based on positive/negative frequencies, formulated as:
      $$w_c = \frac{N_{neg,c}}{\max(N_{pos,c}, 1)}$$
      This weight is integrated into `BCEWithLogitsLoss(pos_weight=...)`. Consequently, the loss for rare positive instances is amplified, preventing them from being overwhelmed by negative instances.
    - **Design Motivation**: The SemEval metric is label-wise Macro-F1, meaning minority labels cannot be masked by overall accuracy. Standard BCE tends to predict "all negative" on sparse labels; Focal Loss helps in high-resource languages but is unstable for extremely sparse languages.

3.  **Heterogeneous Ensemble of XLM-R and mDeBERTa**:
    - **Function**: Leverages complementary representations and tokenization behaviors of two multilingual encoders to improve cross-lingual robustness.
    - **Mechanism**: Probabilities $P_{XLM-R}(c=1|x)$ and $P_{mDeBERTa}(c=1|x)$ are obtained separately, then averaged using:
      $$\bar{P} = \alpha P_{XLM-R} + (1-\alpha) P_{mDeBERTa}$$
      A dev set search yielded $\alpha=0.7$, indicating XLM-R as the primary driver with mDeBERTa providing supplementary support.
    - **Design Motivation**: Different language scripts, morphologies, and social media expressions create language-specific vulnerabilities in single models. Heterogeneous ensembling reduces the variance of a single backbone at a low cost.

### Loss & Training
Training utilized only official task data without external lexicons. The development stage used an 85/15 split of official training data: standard stratification for Subtask 1 and iterative stratification for Subtasks 2/3 to preserve rare label co-occurrences. After dev labels were released, the official dev set was used for hyperparameter tuning, and the model was retrained on train+dev for hidden test prediction.

Implementation details: Emojis were retained as emotional cues, empty texts were deleted, max length 256, dynamic padding. Models were fine-tuned using PyTorch/Hugging Face on A100 with bf16/tf32. XLM-R: LR `1e-5`, 4 epochs, BS 32. mDeBERTa: LR `2e-5`, 5 epochs, BS 64. AdamW, linear scheduler, warmup ratio 0.1, weight decay 0, early stopping patience 2. Both per-label threshold tuning and translation augmentation (4,000 Hausa samples via Gemini) were attempted but showed no significant gain or decreased Macro-F1.

## Key Experimental Results

### Main Results
Official results indicate the ensemble model is the most stable choice for average Macro-F1. It achieved the highest scores for Subtask 1 and 2, and performed almost level with the single XLM-R model for Subtask 3, while still outperforming mDeBERTa and MTL.

| Sub-task | XLM-R | mDeBERTa | Ensemble | MTL | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Subtask 1: POLARDETECT | 0.788 | 0.778 | 0.796 | 0.792 | Ensemble highest, MTL second |
| Subtask 2: POLARTYPE | 0.565 | 0.550 | 0.575 | 0.554 | Ensemble significantly best |
| Subtask 3: POLARMANIFEST | 0.485 | 0.456 | 0.484 | 0.476 | XLM-R slightly higher, Ensemble level |

In official rankings, the system reached the Top 10 for 11/22, 16/22, and 17/18 languages across the three sub-tasks, demonstrating higher competitiveness in fine-grained tasks.

| Metric | Subtask 1 | Subtask 2 | Subtask 3 |
| :--- | :---: | :---: | :---: |
| Top-10 Languages | 11/22 | 16/22 | 17/18 |
| Representative Top-5 | Odia 4th | Amharic 4th, Urdu/Odia/Polish 5th | Arabic 3rd, Urdu 3rd, Spanish 4th, English/Khmer 5th |
| Main Weakness | Lower ranking in highly competitive languages (Eng/Ara) | Fluctuations in sparse target labels | Manifestation labels are the sparsest with high inter-class variance |

### Ablation Study
Optimization target ablation shows that WBCE is particularly important for low-resource and sparse labels. In Subtask 2 dev experiments, WBCE brought significant gains in languages like Telugu and Hausa.

| Language/Average | BCE Base | Focal Loss | WBCE | Gain vs. Base |
| :--- | :---: | :---: | :---: | :---: |
| Chinese | 0.6905 | 0.6893 | 0.7218 | +0.0313 |
| Hindi | 0.7724 | 0.8127 | 0.7996 | +0.0272 |
| Telugu | 0.2253 | 0.2986 | 0.4372 | +0.2119 |
| Amharic | 0.3760 | 0.4568 | 0.4589 | +0.0829 |
| Hausa | 0.1115 | 0.2719 | 0.2513 | +0.1398 |
| Average | 0.4351 | 0.5053 | 0.5338 | +0.0987 |

| Design Choice | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Independent vs. MTL | MTL occasionally helps Sub 1, but Sub 2/3 are weaker | Coarse binary tasks dominate shared encoder; sparse labels suffer negative transfer |
| Global vs. Per-label Threshold | Per-label threshold tuning reduces overall Macro-F1 | Sparse label estimates on the dev set are unstable, leading to overfitting |
| Translation Augmentation | 4k Gemini-translated samples for Hausa gave no gain | Translationese and pragmatic drift weaken region-specific expressions |
| Ensemble vs. Single | More stable on average, though not #1 in every language | Two encoders complement each other at the cost of local optima in specific languages |

### Key Findings
- Binary detection is relatively mature; difficulties stem from fine-grained multi-label sparsity in Subtask 2/3. Labels like Gender/Sexual, Religious, Other, Dehumanization, Lack of Empathy, and Invalidation are particularly fragile.
- WBCE is better suited than Focal Loss for the extreme sparsity of this task. Focal Loss improves some high-resource languages but remains unstable for low-frequency labels.
- Independent modeling improves robustness but introduces cross-task inconsistency: a sample might be judged non-polarized in Subtask 1 but given a positive label in Subtask 2/3. This suggests a need for lightweight gating or hierarchical calibration.
- Multilingual model difficulties extend beyond language coverage to include scripts, tokenization, cultural context, and regional political expressions. Translation augmentation cannot reliably bridge these cultural pragmatic gaps.

## Highlights & Insights
- The highlight of this system paper is its "conservative but steady" approach. It avoids complex modules in favor of independent modeling, class weighting, and heterogeneous ensembling based on clear engineering judgment of shared task bottlenecks.
- The critique of MTL is insightful: related tasks are not always suitable for shared training, especially when one task has dense labels and another has sparse ones, as the shared representation may "squash" fine-grained signals.
- The failure of translation augmentation is noteworthy. Multilingual content safety involves more than translating text; regional political metaphors, sarcasm, group slurs, and pragmatic intensity may be washed out during translation.
- Posterior analysis locates the problem in calibration and label collapse rather than simply stating "low-resource languages are hard." This naturally leads toward hierarchical calibration and cultural context enhancement.

## Limitations & Future Work
- The system relies primarily on supervised encoders and fails to fully exploit the cross-cultural reasoning and long-context capabilities of LLMs; however, the paper also shows LLMs may be unreliable for fine-grained labels.
- The ensemble uses a fixed $\alpha=0.7$ and global threshold $0.5$. While robust, it cannot adapt to calibration differences at the language/label level. Future work could explore language- or label-aware thresholds.
- Independent modeling leads to hierarchical inconsistency across tasks. Gating could be added during inference: if Subtask 1 is non-polarized, Subtask 2/3 probabilities are suppressed.
- The lack of efficacy in translation augmentation suggests a need for "culturally grounded synthesis," such as generative augmentation that preserves region-specific events and styles.

## Related Work & Insights
- **vs. Multi-Task Learning (MTL)**: MTL expects to utilize task correlation via a shared encoder, but this study found negative transfer for sparse fine-grained labels; independent modeling sacrifices parameter sharing for a more stable Macro-F1.
- **vs. LLM Prompting**: While LLMs possess cross-cultural knowledge, their controllability and calibration in multi-label fine-grained classification are inferior to supervised encoders.
- **vs. Focal Loss**: Focal Loss focuses on hard examples and is suitable for object detection-style imbalance; WBCE directly compensates for positive instance scarcity via label frequency, proving more stable in sparse multi-label scenarios.
- **Insight**: Multilingual content safety systems should report precision-recall gaps and script/language family coverage rather than just average scores, as high averages can mask total label collapse in sensitive minority classes.

## Rating
- Novelty: ⭐⭐⭐ The combination is conventional, but the analysis of MTL negative transfer, class imbalance, and translation failure is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid for a shared task paper, covering 22 languages and comparisons across MTL, ensembling, and loss functions.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with actionable conclusions from error analysis.
- Value: ⭐⭐⭐⭐ Directly relevant for multilingual content safety, low-resource multi-label classification, and building shared task systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)

</div>

<!-- RELATED:END -->
