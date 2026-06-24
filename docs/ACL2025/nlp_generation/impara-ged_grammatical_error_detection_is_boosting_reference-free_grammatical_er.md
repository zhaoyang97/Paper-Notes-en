---
title: >-
  [Paper Note] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator
description: >-
  [ACL 2025][Text Generation][Grammatical Error Correction Evaluation] By introducing a Grammatical Error Detection (GED) pre-training step before constructing IMPARA's quality estimator, and removing the ineffective similarity estimator, reference-free GEC evaluation achieves the highest sentence-level correlation on SEEDA.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Grammatical Error Correction Evaluation"
  - "Grammatical Error Detection"
  - "Reference-free Evaluation"
  - "Quality Estimation"
  - "SEEDA"
date: 2026-05-08
content_hash: 428fa916f6b8d484
---

# IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator

**Conference**: ACL 2025  
**arXiv**: [2506.02899](https://arxiv.org/abs/2506.02899)  
**Code**: [HuggingFace](https://huggingface.co/naist-nlp/IMPARA-GED)  
**Area**: Text Generation  
**Keywords**: Grammatical Error Correction Evaluation, Grammatical Error Detection, Reference-free Evaluation, Quality Estimation, SEEDA

## TL;DR

By introducing a Grammatical Error Detection (GED) pre-training step before constructing IMPARA's quality estimator, and removing the ineffective similarity estimator, reference-free GEC evaluation achieves the highest sentence-level correlation on SEEDA.

## Background & Motivation

Grammatical Error Correction (GEC) systems require reliable automatic evaluation methods to replace expensive human judgment. Existing evaluation methods are categorized into two classes:

- **Reference-based methods** (ERRANT, GLEU, etc.): Rely on human reference sentences. However, as a sentence can have multiple correct corrections, reference sets with low coverage decrease evaluation reliability.
- **Reference-free methods** (SOME, IMPARA, Scribendi Score, etc.): Rely solely on the input and system output, showing greater potential.

IMPARA is currently a strong reference-free method, consisting of a Quality Estimator (QE) and a Similarity Estimator (SE). However, the authors identify two critical issues:

**Failure of the Similarity Estimator**: On the outputs of modern GEC systems, PLMs fail to capture semantic changes accurately. For example, the spelling correction "healty $\to$ healthy" paradoxically decreases similarity, leading to incorrect filtering, while a semantic reversal like "like $\to$ dislike" erroneously receives high similarity.

**Insufficiently Strong Quality Estimator**: The vanilla PLM used in the original IMPARA lacks fine-grained sensitivity to grammatical errors.

## Method

### Overall Architecture

IMPARA-GED introduces two improvements: removing the similarity estimator and adding GED task training before constructing the quality estimator.

The entire pipeline consists of three steps:
1. Fine-tune the PLM on GED data (learning token-level error classification).
2. Construct the quality estimator using the fine-tuned PLM following the IMPARA method (learning pairwise quality ranking).
3. During inference, directly output sigmoid(R(O)) as the evaluation score, omitting the similarity filtering.

### Key Designs

#### Key Design 1: Removing the Similarity Estimator

The scoring formula of the original IMPARA is:

$$S(I,O) = \begin{cases} \sigma(R(O)) & \text{sim}(I,O;\text{PLM}) > \theta \\ 0 & \text{otherwise} \end{cases}$$

The authors show through experiments (Table 1) that when different PLMs serve as the SE, some (such as BERT-Base, ELECTRA-Large) yield similarity scores exceeding the threshold of 0.9 for all SEEDA instances, rendering the SE entirely ineffective, while the SE from other PLMs actually degrades performance.

The simplified scoring formula is: $S(I,O) = \sigma(R(O))$

#### Key Design 2: GED Task Pre-training

Inspired by Yuan et al. (2021), the PLM is fine-tuned on the GED task prior to constructing the quality estimator. The GED model performs error classification for each token in the input sentence, where the training loss is:

$$\mathcal{L}^{\text{GED}}(\boldsymbol{x}, \boldsymbol{t}) = -\frac{1}{N}\sum_{i=1}^{N}\log p(t_i | x_i, \boldsymbol{x})$$

Error labels have four levels of granularity:
- **2-class**: Correct/Incorrect (most reliable)
- **4-class**: Correct/Insertion/Deletion/Replacement
- **25-class**: POS categories defined based on ERRANT
- **55-class**: Combinations of the above categories (most informative but lowest label reliability)

### Loss & Training

The quality estimator adopts the pairwise ranking loss from IMPARA:

$$\mathcal{L}^{\text{QE}} = \frac{1}{|\mathcal{T}|}\sum_{(S_+, S_-) \in \mathcal{T}} \sigma(R(S_-) - R(S_+))$$

Key modification: **Mean pooling** is used instead of the first token embedding to obtain sentence representations, in order to better utilize token-level error detection information.

Training pipeline: GED for 5 epochs $\to$ QE for 10 epochs, selecting the best model across 5 random seeds. The datasets used are CoNLL-2013 and FCE.

## Key Experimental Results

### Main Results

Meta-evaluation results on the SEEDA benchmark (Table 2), compared with existing methods:

| Method | SEEDA-S Acc. | SEEDA-S τ | SEEDA-E Acc. | SEEDA-E τ |
|------|-------------|-----------|-------------|-----------|
| SOME | .778 | .555 | .766 | .532 |
| IMPARA | .753 | .506 | .752 | .504 |
| GPT-4-S | .784 | .567 | .798 | .595 |
| GPT-4-S+Fluency | .819 | .637 | .831 | .662 |
| **ModernBERT-Large+2-class** | **.829** | **.658** | .797 | .594 |

IMPARA-GED (ModernBERT-Large + 2-class GED) outperforms all methods including the GPT-4-S series on sentence-level SEEDA-S.

System-level results of different PLMs (selected from Table 2):

| PLM | GED | SEEDA-S r | SEEDA-S ρ | SEEDA-E r | SEEDA-E ρ |
|-----|-----|-----------|-----------|-----------|-----------|
| DeBERTa-v3-Large | None | .960 | .937 | .912 | .944 |
| DeBERTa-v3-Large | 25-class | .945 | .930 | .906 | .930 |
| ModernBERT-Large | None | .949 | .909 | .912 | .937 |
| ModernBERT-Large | 2-class | **.971** | .930 | **.919** | .930 |

### Ablation Study

**Impact of the Number of GED Classes**: Finer label granularity is not always better. 2-class (binary classification) achieves the best performance. This is because prioritization of label reliability over information density is more suitable for GEC evaluation tasks. 55-class is informative but suffers from degraded label reliability, leading to inferior performance.

**Window Analysis** (Figure 1): Sliding window analysis on system-level ranking (window size 4) shows that GED training significantly improves the ability to distinguish among the **top-performing systems**.

**Pairwise Sentence-level Analysis** (Figure 2): GED enhances the capability to distinguish different system ranks, with particularly pronounced improvements for system pairs with large ranking gaps.

### Key Findings

1. The similarity estimator fails on modern GEC system outputs, and omitting it does not degrade performance.
2. Binary GED pre-training is sufficient to yield significant improvements, proving that label reliability is more critical than information density.
3. ModernBERT as a backbone outperforms BERT-Base and DeBERTa-v3-Large.
4. Reference-free methods can outperform GPT-4-level evaluation methods.

## Highlights & Insights

- Identifies and validates the practical failure of the similarity estimator in IMPARA, presenting convincing counterexamples.
- Outperforms expensive GPT-4-based evaluation through a simple two-stage training paradigm (GED $\to$ QE), offering high cost-effectiveness.
- The insight of "label reliability > label information density" is also inspiring for other NLP tasks.

## Limitations & Future Work

- Validated solely on the SEEDA meta-evaluation benchmark, lacking cross-domain generalization validation.
- Only two small datasets, CoNLL-2013 and FCE, were utilized as training data; leveraging larger datasets (e.g., W&I+LOCNESS) might yield further performance gains.
- The training of GED and QE is sequential; multi-task joint learning might be more optimal.
- The impact of different error types within GED on evaluation performance remains unexplored.

## Related Work & Insights

- **IMPARA** (Maeda et al., 2022): The methodology base of this work, which proposed the reference-free QE+SE evaluation framework.
- **GED-enhanced GEC** (Yuan et al., 2021): GED pre-training can improve GEC system performance, which this paper transfers to the evaluation domain.
- **TrueSkill Aggregation** (Goto et al., 2025b): Shares concepts with "Rethinking Evaluation Metrics," employing aggregation aligned with human evaluation.

## Rating

- **Novelty**: 3/5 — Though intuitive, the improvements are effective; the core contributions lie in identifying the failure of SE and validating the utility of GED pre-training.
- **Technical Depth**: 3/5 — Short paper, with a concise method and solid experiments.
- **Experimental Thoroughness**: 4/5 — Comprehensive evaluation with multiple PLMs, granularities, window analysis, and pairwise analysis.
- **Value**: 4/5 — Open-sourced model, directly applicable to GEC evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)
- [\[ACL 2025\] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)
- [\[ACL 2025\] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)

</div>

<!-- RELATED:END -->
