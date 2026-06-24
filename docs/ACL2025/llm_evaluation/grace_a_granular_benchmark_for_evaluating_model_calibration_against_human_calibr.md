---
title: >-
  [Paper Note] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration
description: >-
  [ACL 2025][LLM Evaluation][Model Calibration] This paper proposes the GRACE benchmark, which collects 1749 data points through progressive incremental question answering and human-vs-model tournaments to evaluate LLM calibration capability with human calibration performance as a reference. By introducing the CalScore metric, the study reveals that while human accuracy may be lower than that of models, humans generally outperform SOTA models in calibration—models are overconfi…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Model Calibration"
  - "Human Calibration"
  - "Incremental QA"
  - "Adversarial Benchmark"
  - "Confidence Assessment"
  - "CalScore"
date: 2026-05-08
content_hash: 6e3fc496272a109f
---

# GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration

**Conference**: ACL 2025  
**arXiv**: [2502.19684](https://arxiv.org/abs/2502.19684)  
**Code**: [GitHub](https://github.com/yysung/advcalibration)  
**Area**: LLM Evaluation  
**Keywords**: Model Calibration, Human Calibration, Incremental QA, Adversarial Benchmark, Confidence Assessment, CalScore

## TL;DR

This paper proposes the GRACE benchmark, which collects 1749 data points through progressive incremental question answering and human-vs-model tournaments to evaluate LLM calibration capability with human calibration performance as a reference. By introducing the CalScore metric, the study reveals that while human accuracy may be lower than that of models, humans generally outperform SOTA models in calibration—models are overconfident when uncertain and underconfident when correct.

## Background & Motivation

LLMs often "make confident mistakes"—there is a severe mismatch between model confidence and actual accuracy, which leads users to overtrust models and even reject their own correct judgments.

Existing calibration research suffers from three core limitations:

**Lack of comparison with human calibration**: Prior work only conducts calibration comparisons among models, without incorporating human calibration performance as a reference. However, users expect models to be at least as well-calibrated as humans—when the model's calibration performance is worse than that of humans, users are often unprepared to handle these errors.

**Insufficient granularity**: Traditional calibration evaluation only computes aggregated metrics (such as ECE) over the entire dataset, failing to identify specific questions where the model's calibration performance is particularly poor.

**Overly simplistic existing benchmarks**: Existing incremental QA datasets are too easy for modern models—GPT-4 achieves 80% accuracy on the TrickMe dataset with only 60% of the clues, failing to test calibration capability effectively.

GRACE addresses these issues through three innovations: (1) adversarial incremental questions authored by experts; (2) live human-vs-model tournaments to collect human calibration data; (3) the CalScore metric that incorporates human performance.

## Method

### Overall Architecture

The construction workflow of GRACE consists of three phases:

1. **Question Creation**: Expert authoring + adversarial editing interface
2. **Data Collection**: Offline model confidence + online human-machine tournaments
3. **Calibration Evaluation**: CalScore metric

### Key Designs

#### 1. Incremental Adversarial Question Design

Each question contains at least 5 clue sentences, arranged in **descending order of difficulty**, with all clues pointing to the same answer. Key characteristics:
- **Adversarial**: Created using a human-AI collaborative interface. Question authors see GPT-3.5's guesses and confidence for each clue in real-time, adjusting clues to make them harder for the model but still solvable for humans.
- **Incremental**: Each clue represents a decision point—participants can choose to buzz in with an answer or wait for more clues.
- **Quality Control**: 6 question authors + 10 editors, with each question undergoing three rounds of review by the author, category editor, and editor-in-chief.

The final dataset contains 243 QA pairs with a total of 1236 clue sentences, covering 6 categories such as Science, History, and Literature.

#### 2. Human-Machine Tournaments—Human Buzzpoint Collection

Simulating quiz bowl format: a host reads questions, and human teams compete with models simultaneously—
- **Model buzzpoints**: Pre-computed. Models calculate confidence for each clue and "buzz in" when it exceeds a threshold.
- **Human buzzpoints**: Recorded in real-time on-site. Players hit physical buzzers, and the host verifies answers.
- **Tournament Scale**: 3 tournaments, 17 human teams vs 3 LLMs (GPT-4o, GPT-4, Mistral-7b), totaling 93 matches and 1749 data points.

#### 3. CalScore—Human-Referenced Calibration Metric

**Model Calibration Error (MCE)**:

$$\text{MCE} = 1 - r(\mathbb{E}_t[g_t c_t])$$

where $g_t$ is the correctness of the guess at clue $t$ (1 or -1), $c_t$ is the model confidence, and $r(\cdot)$ is a normalization to [0,1].

**CalScore (Human-Adjusted Calibration Error)**:

$$\text{CalScore}_q = 1 - r(\mathbb{E}_t[(1-h_t) g_t c_t])$$

where $h_t$ is the cumulative probability of humans correctly answering by clue $t$. Key designs:
- Weighted by $(1-h_t)$: **Model performance is weighted higher when humans do not yet know the answer**.
- Model is confident and correct when humans are uncertain $\to$ High reward.
- Model is confident but incorrect when humans are uncertain $\to$ High penalty (since humans cannot correct the error).
- Supports question-by-question evaluation, not just aggregated metrics.

### Confidence Acquisition Methods

Two approaches are evaluated in parallel:
- **Logit-based**: Geometric mean of token logit probabilities.
- **Verbalized**: Prompting the model to express confidence directly.

## Key Experimental Results

### Calibration Metric Comparison

**Verbalized Confidence**:

| Model | Brier Score | ECE | MCE | CalScore |
|------|:---:|:---:|:---:|:---:|
| GPT-4 | 0.274 | 0.259 | 0.584 | 0.588 |
| GPT-4o | 0.266 | 0.224 | 0.601 | 0.604 |
| Llama-3.1-70B | 0.373 | 0.392 | 0.685 | 0.719 |
| Llama-2-70b | 0.490 | 0.570 | 0.739 | 0.803 |
| Llama-3.1-8B | 0.623 | 0.693 | 0.774 | 0.843 |
| Mistral-7b | 0.716 | 0.784 | 0.790 | 0.881 |

**Logit-based Confidence**:

| Model | Brier Score | ECE | MCE | CalScore |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | 0.341 | 0.353 | 0.654 | 0.661 |
| Llama-3.1-70B | 0.323 | 0.339 | 0.651 | 0.679 |
| GPT-4 | 0.380 | 0.388 | 0.672 | 0.684 |
| Llama-3.1-8B | 0.302 | 0.397 | 0.675 | 0.718 |

### Human vs. Model Buzzing Performance

- The cumulative correct buzzing rate peak of the **top quartile human teams** is more than twice that of the best model.
- GPT-4 and GPT-4o have error buzzing rates significantly higher than all human teams.
- Models are particularly overconfident in the early stages of questions (when clues are fewer and information is insufficient).
- As clues increase, humans tend to buzz in more when they are correct (increasing confidence), whereas models surprisingly tend to buzz in less—contrary to intuition.

### Conditional Probability Analysis

- **P(buzz|correct)**: Humans > 50%, models < 45%. Models lack confidence when correct.
- **P(buzz|incorrect)**: GPT-4 is the highest. Models (especially GPT-4) are overconfident when incorrect.
- Humans increase their buzzing probability when correct after seeing more clues, whereas models decrease theirs.

### Key Findings

1. **CalScore > MCE (All Models)**: Calibration error consistently increases after incorporating human performance, revealing calibration flaws underestimated by traditional metrics.
2. **Weak models exhibit larger CalScore/MCE gaps**: Weak models expose more calibration issues when human performance is factored in.
3. **Models excel at retrieval but struggle with abstract reasoning**: Models perform well on questions involving specific proper nouns (e.g., telomere questions containing "TRF2 protein") but are severely miscalibrated on abstract descriptions and questions requiring multi-step reasoning.
4. **Models give "unreasonable" incorrect answers**: Incorrect answers provided by humans are related within the same domain as the correct answer, whereas models may provide completely irrelevant answers (e.g., answering "Fermat's Little Theorem" to a philosophy question), indicating a more fundamental calibration flaw.
5. **High variance in calibration among human experts**: Even experienced quiz bowl players show significant differences in calibration performance. The strongest humans significantly outperform top models, but this does not apply to all humans.

## Highlights & Insights

1. **Ingenious Quiz Bowl Format Design**: "Gamifying" calibration evaluation—the buzzing mechanism naturally requires players to trade off accuracy and speed, which is the core of calibration capability. Directly comparing models and humans on the same task eliminates the inconsistency of comparative frameworks in prior studies.
2. **Human-Referenced Philosophy of CalScore**: Instead of asking "Is the model well-calibrated?", it asks "Is the model better-calibrated than humans?"—which is what users care about in actual deployment. A model making a confident mistake when humans are still uncertain is far more hazardous than calibration errors when both parties are certain.
3. **Effectiveness of Adversarial Question Creation**: GPT-4 does not reach 50% accuracy on GRACE until seeing 90%+ of the clues, whereas on older benchmarks like TrickMe, it achieves 80% accuracy with 60% of clues. This provides a more discriminative test scenario for calibration evaluation.
4. **Collateral Value of Community Engagement**: Live tournaments not only collected high-quality data but also stimulated interest and engagement with AI among non-researchers.

## Limitations & Future Work

- **Task Format Limitations**: GRACE only covers factual QA scenarios and does not extend to broader NLP tasks like open-ended generation or multi-turn dialogues.
- **CalScore Incompleteness**: As a baseline metric, CalScore does not capture all forms of uncertainty and miscalibration.
- **Limited Data Scale**: The corpus of 243 questions is somewhat small, which might be insufficient for statistically robust cross-category analysis.
- **Expert Bias among Human Participants**: All tournament participants are experienced quiz bowl players, who might not represent the calibration performance of average users.
- **Lack of Calibration Improvement Methods**: GRACE is primarily a diagnostic tool and does not propose methods to improve model calibration.

## Related Work & Insights

- **Incremental Question Answering (Incremental QA)**: Early work by Boyd-Graber et al. GRACE builds on this by adding adversarial creation and confidence assessment.
- **Insights for AI-Assisted Decision Making**: Poorly calibrated models can cause negative impacts in human-AI collaboration—humans are more likely to be misled when models are confident but incorrect, especially when humans do not know the answer.
- **Insights for Calibration Improvement Research**: GRACE can serve as an evaluation platform for calibration methods to assess improvements in verbalized confidence, personalized abstention strategies, etc.
- **Academic Parallel to Watson/Jeopardy!**: GRACE is in some sense a modern academic version of IBM Watson's Jeopardy! challenge, but with the addition of calibration dimensions and multi-model comparisons.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first benchmark to integrate human calibration into model calibration evaluation + quiz bowl format data collection + the CalScore metric, a triple-innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 6 LLMs + 17 human teams + 93 tournaments + comparison across multiple calibration metrics + qualitative analysis + player feedback.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definitions, rigorous experimental design, rich and highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐ High value as a benchmark and diagnostic tool; CalScore offers a new paradigm for calibration evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)
- [\[ACL 2025\] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles](influences_on_llm_calibration_a_study_of_response_agreement_loss_functions_and_p.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_evaluation/on_the_entropy_calibration_of_language_models.md)

</div>

<!-- RELATED:END -->
