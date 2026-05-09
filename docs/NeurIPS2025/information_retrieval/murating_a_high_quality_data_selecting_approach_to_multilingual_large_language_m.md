---
title: >-
  [Paper Note] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining
description: >-
  [NeurIPS 2025][multilingual data selection] This paper proposes MuRating, a scalable multilingual data selection framework that aggregates multiple English data quality scorers via pairwise comparisons, transfers quality signals to 17 languages through translation, and trains a language-agnostic multilingual quality assessment model, achieving consistent performance gains in LLM pretraining at both 1.2B and 7B scales.
tags:
  - NeurIPS 2025
  - multilingual data selection
  - pretraining data quality
  - Bradley-Terry model
  - pairwise comparison
  - cross-lingual alignment
date: 2026-05-08
content_hash: 1498a59930fd8641
---

# MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining

**Conference**: NeurIPS 2025
**arXiv**: [2507.01785](https://arxiv.org/abs/2507.01785)
**Code**: [https://github.com/aialt/MuRater](https://github.com/aialt/MuRater)
**Area**: Information Retrieval
**Keywords**: multilingual data selection, pretraining data quality, Bradley-Terry model, pairwise comparison, cross-lingual alignment

## TL;DR

This paper proposes MuRating, a scalable multilingual data selection framework that aggregates multiple English data quality scorers via pairwise comparisons, transfers quality signals to 17 languages through translation, and trains a language-agnostic multilingual quality assessment model, achieving consistent performance gains in LLM pretraining at both 1.2B and 7B scales.

## Background & Motivation

High-quality pretraining data is a critical driver of LLM performance. Existing model-based data selection methods (e.g., QuRater, DCLM, AskLLM, FineWeb-Edu) have achieved success in English data selection but are almost **entirely English-centric**, leaving a critical gap in multilingual LLM pretraining.

Limitations of prior work:

**English-centrism**: Mainstream data selection methods are neither designed nor validated for non-English languages.

**Scoring inconsistency**: Different scorers exhibit complementary strengths across tasks (e.g., DCLM is strong on HellaSwag but weak on ARC-Challenge; QuRater shows the opposite pattern), with no unified framework to reconcile them.

**Risks of multilingual attempts**: FineWeb2-HQ trains language-specific classifiers using benchmark datasets as positive examples, introducing the risk of test set contamination.

**Cross-lingual inconsistency of absolute scoring**: Subtle biases introduced by translation affect pointwise scoring, rendering cross-lingual scores unreliable.

## Method

### Overall Architecture

MuRating is a two-stage framework:

**Stage 1: English scorer aggregation** → Unifies the judgments of multiple English scorers into a single quality score.

**Stage 2: Multilingual transfer** → Projects English quality signals to 17 target languages via translation.

### Key Designs

**1. English AutoRater Aggregation (Stage 1)**

Given a text pair $(t_A, t_B)$ and $N$ scorers, each scorer $n$ produces scores $S_A^n$ and $S_B^n$. A majority-vote confidence is computed as:

$$P_{A>B} = \frac{1}{|N|} \sum_{n \in N} \mathbb{I}[S_A^n > S_B^n]$$

A unified scorer is then trained using the Bradley-Terry model with BCE loss:

$$\mathcal{L}_\theta = \mathbb{E}_{(t_A, t_B, p_{B \succ A}) \in \mathcal{J}} \left[ -p_{B \succ A} \log \sigma(s_\theta(t_B) - s_\theta(t_A)) - (1 - p_{B \succ A}) \log \sigma(s_\theta(t_A) - s_\theta(t_B)) \right]$$

Four scorers are aggregated: GPT-4o (averaged over multiple forward/reverse evaluations), AskLLM (Flan-T5-XXL), FineWeb-Edu Classifier, and DCLM (fastText).

**2. Translation-Based Multilingual Preference Transfer (Stage 2)**

The core assumption is that translation preserves semantic content and the relative quality relationship between text pairs, i.e., $P_{A^m > B^m} \approx P_{A^{en} > B^{en}}$.

Three types of training data are constructed:
- **Monolingual pairs**: $(t_A^m, t_B^m)$, intra-language comparisons, 150K pairs.
- **Cross-lingual pairs**: $(t_A^m, t_B^{m'})$, inter-language comparisons that promote cross-lingual consistency, 150K pairs.
- **Parallel pairs**: $(t_A^m, t_A^{m'})$, translations of the same content in different languages, assigned a neutral preference $P \approx 0.5$, 75K pairs.

The regularization loss for parallel pairs is:

$$\mathcal{L}_{\text{parallel}} = \mathbb{E}_{(t_A^m, t_A^{m'}) \in \mathcal{J}'} \left[ -\log\sigma(s_\theta(t_A^m) - s_\theta(t_A^{m'})) - \log\sigma(s_\theta(t_A^{m'}) - s_\theta(t_A^m)) \right]$$

The final objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pairwise}} + \lambda \cdot \mathcal{L}_{\text{parallel}}$$

**3. Pairwise vs. Pointwise Scoring**

The paper empirically demonstrates that pairwise comparison is more suitable than pointwise (absolute) scoring for cross-lingual transfer:
- Pointwise scoring exhibits significant cross-lingual variance in the medium-quality range (scores 3–6).
- Pairwise scoring maintains cross-lingual consistency under translation-induced bias, requiring only that relative rankings be preserved.

**4. Model Architecture**

The model fine-tunes the BGE-M3 encoder with an added linear head for quality score prediction. BGE-M3 is selected for its strong multilingual representation capabilities and lightweight design. Validation accuracy reaches 93%, with training accuracy at 97%.

### Loss & Training

- English scorer aggregation: 300K English text pairs; GPT-4o scores in both directions multiple times to eliminate position bias.
- Multilingual training set: 150K monolingual pairs + 150K cross-lingual pairs + 75K parallel pairs.
- Translation performed using GPT-4o, balanced across 17 target languages.
- Pretraining uses the LLaMA architecture; 1.2B parameter models trained for a single epoch on 200B English tokens + 300B multilingual tokens.

## Key Experimental Results

### Main Results

**Multilingual results (18 languages, 1.2B model):**

| Selection Method | Reading Comprehension (5 tasks) | Commonsense Reasoning (2 tasks) | World Knowledge (4 tasks) | Average (11 tasks) |
|---|---|---|---|---|
| Uniform (+50%) | 53.16 | 54.58 | 38.25 | 48.66 |
| HPLT-2 | 50.38 | 49.77 | 36.96 | 45.70 |
| FineWeb-2 | 50.83 | 52.48 | 35.53 | 46.28 |
| QuRater-M | 54.58 | 54.87 | 38.12 | 49.19 |
| MuRater(M) | 54.91 | 55.48 | 39.68 | 50.02 |
| **MuRater(E)** | **56.05** | **56.42** | **40.40** | **50.96** |

**English results (12 benchmarks, 1.2B model):**

| Selection Method | Reading Comprehension (6 tasks) | Commonsense Reasoning (4 tasks) | World Knowledge (2 tasks) | Average (12 tasks) |
|---|---|---|---|---|
| Uniform (+50%) | 43.93 | 59.06 | 20.36 | 48.70 |
| AskLLM | 42.83 | 58.40 | 20.21 | 47.82 |
| DCLM | 46.00 | 58.99 | 22.37 | 50.23 |
| QuRater | 43.54 | 58.58 | 20.47 | 48.33 |
| **MuRater** | **47.13** | **59.95** | **22.53** | **51.23** |

**7B model results (1T token training):**

| Selection Method | Reading Comprehension | Commonsense Reasoning | World Knowledge | Average |
|---|---|---|---|---|
| QuRater-M | 61.96 | 63.28 | 43.31 | 56.18 |
| **MuRater** | **62.78** | **64.40** | **44.50** | **57.23** |

### Ablation Study

**Effectiveness of cross-lingual and parallel pairs:**
- Adding alignment training (cross-lingual + parallel pairs) yields lower MSE on parallel text scoring and slopes closer to 1.
- Cross-lingual scoring consistency is substantially improved.

**Pairwise vs. pointwise scoring transfer:**
- GPT-4o scores 200 Arabic and Spanish parallel text pairs, each evaluated 20 times.
- Pointwise scoring exhibits large cross-lingual variance in the medium-quality range.
- Pairwise scoring demonstrates substantially stronger cross-lingual consistency, with points closely following the $y=x$ line.

**MuRater(E) vs. MuRater(M):**
- MuRater(E): quality signals originate from English scoring and are transferred to multiple languages via translation (English-anchored training).
- MuRater(M): multilingual pairs are translated into English for scoring, then projected back.
- MuRater(E) consistently outperforms MuRater(M), as English corpora are more diverse and provide a more stable supervision signal.

### Key Findings

1. **Advantage of scorer aggregation**: MuRating combines the strengths of individual scorers, achieving uniformly strong performance across all benchmarks and avoiding the preference bias of any single scorer.
2. **English-anchored training is more effective**: MuRater(E) > MuRater(M); the diversity of English corpora provides richer signals for cross-lingual transfer.
3. **Robustness of pairwise comparison**: Pairwise scoring maintains consistency when translation introduces subtle biases, whereas pointwise scoring becomes notably unstable.
4. **Scale consistency**: MuRater's advantage persists from 1.2B to 7B (×5.8 parameter scale, ×2 data volume).
5. On a 13-language subset, MuRater(E) surpasses FineWeb2-HQ by approximately 3 percentage points.

## Highlights & Insights

1. **Core insight behind pairwise comparison**: The framework exploits the key observation that translation preserves relative quality but not absolute quality, making the pairwise paradigm naturally suited for cross-lingual scenarios.
2. **Parallel pairs as regularization**: Assigning a neutral preference ($P \approx 0.5$) to parallel translations is an elegant design choice—it forces the model to learn language-agnostic quality metrics rather than language-specific features.
3. **Unifying heterogeneous judgments**: The Bradley-Terry model consolidates the heterogeneous outputs of multiple scorers into a consistent quality score, avoiding the need to manually select the "best" scorer.
4. **Practical translation strategy**: Although using GPT-4o for translation incurs cost, this step is performed only once during training set construction; at inference time, MuRater can directly evaluate text in any language.

## Limitations & Future Work

1. **Limited language coverage**: Only 17 target languages are included, excluding a large number of low-resource languages.
2. **Dependence on GPT-4o biases**: Using GPT-4o for both translation and scoring may introduce model-specific biases.
3. **Domain preference**: English scorers favor factual and informative content, with limited ability to evaluate narrative and creative text.
4. **Lack of language-specific tuning**: A unified multilingual scorer may fail to capture the unique cultural and linguistic characteristics of individual languages.
5. The sensitivity of scoring transfer quality to translation quality warrants deeper analysis.
6. Validation is limited to the LLaMA architecture; generalizability to other architectures remains unknown.

## Related Work & Insights

- **QuRater** (Wettig et al., 2024): Uses LLMs to assess the educational value of data; MuRating extends this English-centric framework to the multilingual setting.
- **DCLM** (Li et al., 2024): A fastText classifier approach that is efficient but prone to preference bias; MuRating achieves greater robustness through aggregation.
- **FineWeb2-HQ** (Messmer et al., 2025): Trains language-specific classifiers with benchmark datasets as positives, risking test set contamination; MuRating's transfer from English is safer.
- **AskLLM** (Sachdeva et al., 2024): A prompt-based LLM scoring method, incorporated as one of MuRating's input scorers.
- The application of the Bradley-Terry pairwise comparison model in this context is closely analogous to reward model training in RLHF.
- The parallel pair regularization idea for cross-lingual consistency is transferable to other multilingual evaluation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-stage framework combining pairwise comparison and translation-based transfer is elegantly designed and addresses a genuine pain point in multilingual data quality assessment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two model scales (1.2B and 7B), 18 languages, multiple baselines, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and experiments are well-organized, though the related work and method sections are slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in multilingual pretraining data selection with a scalable and reusable framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface](the_transparent_earth_a_multimodal_foundation_model_for_the_earths_subsurface.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](../../ACL2026/information_retrieval/all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)

</div>

<!-- RELATED:END -->
