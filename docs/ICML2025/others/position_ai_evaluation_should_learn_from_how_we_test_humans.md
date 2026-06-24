---
title: >-
  [Paper Note] Position: AI Evaluation Should Learn from How We Test Humans
description: >-
  [ICML 2025][Adaptive Testing] Proposes systematically introducing the adaptive testing paradigm from human psychometrics into AI evaluation, achieving efficient and reliable model capability assessment by estimating item characteristics (difficulty, discrimination, and guessing factor), reconstructing full benchmark scores with only 3% of the questions.
tags:
  - "ICML 2025"
  - "Adaptive Testing"
  - "IRT"
  - "Psychometrics"
  - "AI Evaluation"
  - "Benchmark Efficiency"
date: 2026-05-08
content_hash: 412b7cdb19e6726d
---

# Position: AI Evaluation Should Learn from How We Test Humans

**Conference**: ICML 2025  
**arXiv**: [2306.10512](https://arxiv.org/abs/2306.10512)  
**Code**: None  
**Area**: AI Evaluation / Psychometrics  
**Keywords**: Adaptive Testing, IRT, Psychometrics, AI Evaluation, Benchmark Efficiency

## TL;DR
Proposes systematically introducing the adaptive testing paradigm from human psychometrics into AI evaluation, achieving efficient and reliable model capability assessment by estimating item characteristics (difficulty, discrimination, and guessing factor), reconstructing full benchmark scores with only 3% of the questions.

## Background & Motivation

**Background**: AI evaluation relies on large-scale fixed benchmarks and average score metrics. Evaluating a single LLM on the complete HELM requires over 4000 GPU hours (around \$10,000 in API costs), and LLM inference latency can be up to 1000 times that of traditional models (e.g., BERT), making large-scale evaluation extremely expensive.  
**Limitations of Prior Work**: Only 56.3% of datasets report quality information; benchmarks contain annotation errors, redundant questions, and data contamination. Accuracy itself lacks explanatory power—does GPT-4o's 85.7% accuracy on MedQA mean it can serve real patients?  
**Key Challenge**: Traditional evaluation treats all questions with equal weight, neglecting the massive differences in difficulty and informativeness among items, which makes evaluation both expensive and unreliable.  
**Goal**: Build an adaptive evaluation paradigm for AI by drawing on mature theories from human psychometrics.  
**Key Insight**: Psychometrics has a 70-year history (Lord, 1952), driving adaptive testing in high-stakes exams like GRE, TOEFL, and Duolingo. Its core lies in estimating latent ability/traits rather than simply summing up scores.  
**Core Idea**: Redefine evaluation as a parameter estimation problem—the model has a latent ability $\theta$, while each question has a difficulty $\beta$, discrimination $\alpha$, and a guessing factor $c$. The interaction between the two is modeled using Item Response Theory (IRT) for precise evaluation.

## Method

### Overall Architecture
A two-stage workflow: (1) Item parameter calibration—estimating the IRT parameters of each benchmark question based on historical model response data; (2) Interactive dynamic evaluation—adaptive question selection coupled with real-time capability updates.

### Key Designs

1. **IRT-based Ability-Oriented Evaluation**:
    - **Function**: Estimate the latent ability $\theta$ of AI models instead of simply calculating raw accuracy.
    - **Mechanism**: Use a 3-parameter IRT model $P(y_i=1|\theta) = c_i + (1-c_i)\sigma[\alpha_i(\theta - \beta_i)]$, where $\beta_i$ is difficulty, $\alpha_i$ is discrimination, and $c_i$ is the guessing factor. The capability estimation provides a confidence interval via the Bayesian posterior distribution.
    - **Design Motivation**: Traditional accuracy is unstable under subset sampling, whereas IRT leverages item characteristics to provide reliable estimations even with a small number of questions.

2. **Adaptive Question Selection Algorithm**:
    - **Function**: Dynamically select the most informative questions for each model.
    - **Mechanism**: Based on Fisher Information $I_i(\theta) = \alpha_i^2 \cdot P(y_i=1|\theta) \cdot P(y_i=0|\theta)$, select the next question that maximizes this information. Models with higher capabilities are assigned harder questions.
    - **Design Motivation**: Proven in exams like the GRE (where higher-ability test-takers experience increasingly difficult questions). Achieving a 90% abandonment correlation requires only 50 questions, and 3% of questions are sufficient to reconstruct the full benchmark score.

3. **Item Quality Diagnosis and Contamination Detection**:
    - **Function**: Automatically identify annotation errors, low-quality questions, and data contamination.
    - **Mechanism**: Negative discrimination ($\alpha<0$) means models with higher ability are more likely to answer incorrectly, which typically indicates annotation errors. Models answering highly difficult questions correctly while failing easy ones suggests data leakage.
    - **Design Motivation**: In SQuAD, negative discrimination questions were found to correspond to annotation errors (e.g., "Why did rent demand drop?" with the incorrect ground truth answer "Demand for high-quality housing increased").

### Loss & Training
- Item parameters are fitted from historical model response data using Maximum Likelihood Estimation (MLE) or Bayesian estimation.
- Capability estimation is iteratively refined via Bayesian updates during the adaptive testing process.

## Key Experimental Results

### Main Results

| Application | Method | Results | Source |
|---------|------|------|------|
| MMLU Pruning | IRT Selection | Accurate score reconstruction with 100/14042 questions | Polo et al., 2024 |
| 6 Major Benchmarks | Fisher Information Selection | Reconstructed original scores with ≤3% of questions | Kipnis et al., 2024 |
| SQuAD Quality Check | IRT Discrimination Analysis | Automatic discovery of annotation errors | Rodriguez et al., 2021 |
| Model Ranking Efficiency | Fisher Selection | 90% Kendall correlation with 50/1000 questions | Rodriguez et al., 2021 |

### Ablation Study

| Evaluation Method | No. of Questions | Accuracy Estimation | Rank Maintenance |
|---------|--------|-----------|-----------|
| Full Benchmark | ~29000 | Baseline | Baseline |
| Random Subset | 100 | High Fluctuation | ~70% |
| IRT Adaptive | 100 | Stable | ~90% |

### Key Findings
- The three key pillars of psychometrics (uncertainty modeling, mitigating the curse of dimensionality, and interpretability/comparability) are fully applicable to AI evaluation.
- AI systems exhibit human-like "universal regularities"—sharing the Transformer architecture and training paradigms leads to predictive, structural correlations across model performance.
- Item importance is personalized—the same question provides different amounts of information depending on the model's capability level.
- Adaptive testing naturally prevents further data contamination, as each model is evaluated on a different subset of questions.

## Highlights & Insights
- Redefining evaluation from "counting correct answers across all questions" to "estimating latent capability parameters" offers strong theoretical guarantees; MLE asymptotic theory ensures capability estimates converge to the true value.
- The insight that "not all questions are created equal" directly leads to high-efficiency evaluation: questions with high discrimination and matching difficulty are much more valuable than a vast pool of redundant ones, explaining why 3% of questions suffice.
- The transfer from psychometrics to AI is feasible because AI systems exhibit "universal regularities"—different models share architectures and training paradigms, similar to how humans share cognitive structures.

## Limitations & Future Work
- IRT assumes local independence (responses are independent given the ability), but LLMs may exhibit strong correlations across different questions.
- The concept of the guessing factor needs to be re-evaluated for LLMs—they do not "randomly guess" but rather suffer from systematic biases.
- Adaptive testing relies on an initial calibration population; novel architectures (e.g., MoE) may require recalibration.
- The applicability to generative tasks and open-domain evaluations is not sufficiently discussed.
- Given its nature as a position paper, it lacks an end-to-end full system implementation and large-scale validation.

## Related Work & Insights
- **vs Traditional Benchmarks (BIG-bench, HELM)**: Traditional methods aggregate performance by averaging across all questions. In contrast, this paper advocates for weighted or selective evaluation utilizing item characteristics.
- **vs Chatbot Arena**: Where Arena uses Elo ratings to compare models, IRT provides more fine-grained multidimensional ability profiles and item-level diagnostics.
- **vs Data Contamination Detection (Oren et al., 2023)**: Conventional methods check training data overlap with test items, whereas IRT indirectly detects contamination via anomalous response patterns without requiring access to the training dataset.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative cross-disciplinary perspective, systematically bringing 70 years of psychometrics accumulation into AI evaluation.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, it primarily cites existing verifications and lacks end-to-end system experiments.
- Writing Quality: ⭐⭐⭐⭐ Logical, clear arguments and intuitive illustrations (such as the ability interval visualization in Fig. 2).
- Value: ⭐⭐⭐⭐ Provides directional guidance for the AI evaluation paradigm; the result of reconstructing scores with only 3% of questions is of high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ICML 2025\] Democratic AI is Possible. The Democracy Levels Framework Shows How It Might Work](democratic_ai_is_possible_the_democracy_levels_framework_shows_how_it_might_work.md)
- [\[ICML 2025\] Practical Principles for AI Cost and Compute Accounting](practical_principles_for_ai_cost_and_compute_accounting.md)
- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)

</div>

<!-- RELATED:END -->
