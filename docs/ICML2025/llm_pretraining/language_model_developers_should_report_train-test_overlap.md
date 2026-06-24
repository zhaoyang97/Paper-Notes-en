---
title: >-
  [Paper Note] Language Model Developers Should Report Train-Test Overlap
description: >-
  [ICML 2025 Spotlight][LLM Pretraining][Train-test overlap] This paper systematically investigates the reporting practices of 30 language model developers regarding train-test overlap. It finds that only 9 models provide sufficient overlap information and calls on all developers to report train-test overlap statistics or release training data when publishing evaluation results.
tags:
  - "ICML 2025 Spotlight"
  - "LLM Pretraining"
  - "Train-test overlap"
  - "data contamination"
  - "evaluation transparency"
  - "benchmark trustworthiness"
  - "model evaluation"
date: 2026-05-08
content_hash: 1e931d4727a9df22
---

# Language Model Developers Should Report Train-Test Overlap

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2410.08385](https://arxiv.org/abs/2410.08385)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Train-test overlap, data contamination, evaluation transparency, benchmark trustworthiness, model evaluation

## TL;DR

This paper systematically investigates the reporting practices of 30 language model developers regarding train-test overlap. It finds that only 9 models provide sufficient overlap information and calls on all developers to report train-test overlap statistics or release training data when publishing evaluation results.

## Background & Motivation

### Problem Definition

Train-test overlap refers to the degree to which a language model's training data contains the test data on which it is evaluated. In the traditional machine learning paradigm, training and test sets are partitioned conjointly by the evaluation designer, making the overlap issue naturally controllable. However, in the era of large language models, model developers determine the training set (usually closed-source) and evaluation designers determine the test set (usually open-source). This new paradigm of **separate control over training/test data by two distinct parties** makes train-test overlap highly challenging to track.

### Three Main Pathways of Overlap

**Crawled Test Sets**: Evaluation datasets are routinely published on platforms like GitHub and Hugging Face, making them easily retrievable by web crawlers and incorporated into pre-training corpora.

**Underlying Data Overlap**: Evaluation datasets are often constructed based on publicly available source materials (e.g., SQuAD utilizes Wikipedia data), which are highly likely to have been ingested during pre-training.

**API Call Leakage**: Test queries inputted into models during evaluation may be logged and reused for training subsequent iterations of the model.

### Existing Evidence of Severe Overlap Issues

- **GPT-4 Codeforces Incident**: OpenAI claimed GPT-4 achieved SOTA on Codeforces with zero contamination. However, subsequent evaluations revealed that the model achieved 100% accuracy on problems published prior to 2021, but 0% accuracy on newer problems.
- **Claude 3.5 CTF Incident**: Anthropic reported breakthroughs on Capture The Flag (CTF) tasks, but Transluce discovered that Claude 3.5 was actually solving corrupted tasks by memorizing the answers.
- Studies have demonstrated that high train-test overlap causes **significant performance degradation** when comparing seen versus unseen test samples.

## Method

### Overall Architecture

This study adopts an analytical framework of **systematic survey + scoring + policy recommendations**:

1. **Model Selection**: 30 flagship language models from various developers are selected from the HELM MMLU leaderboard and Ecosystem Graphs.
2. **Information Collection**: For each model, papers, technical reports, and official websites are searched for information regarding train-test overlap.
3. **Binary Scoring**: Each developer is graded (0 or 1) based on whether they provide sufficient information to evaluate the degree of overlap.
4. **Developer Communication**: Developers graded with 0 are proactively contacted, granting them an opportunity to provide supplementary information or dispute the score.
5. **Alternative Strategy Analysis**: The pros and cons of four existing mitigation strategies are systematically evaluated.

### Key Designs: Scoring Criteria System

The grading scheme revolves around the following three core dimensions:

| Dimension | Evaluation Content | Requirements |
|------|----------|----------|
| Training Data Openness | Whether the training data is publicly available | Training data is released under an open-source license, allowing third parties to directly compute overlap. |
| Verification Completeness | Whether overlap statistics are reported for public benchmarks where results are claimed | Quantitative overlap metrics are provided for each evaluated benchmark. |
| Methodological Transparency | Whether the method of computing overlap is clearly described | The methodology used for overlap detection is explicitly detailed. |

**Scoring Rules**:
- If training data is open-source $\rightarrow$ Earns 1 point (as third parties can compute overlap independently).
- If training data is closed-source, but the overlap reporting is sufficiently specific with a clear methodological description $\rightarrow$ Earns 1 point.
- If none of the above conditions are met $\rightarrow$ Earns 0 points.

### Systematic Analysis of Four Alternative Strategies

The paper thoroughly evaluates four existing community strategies to address train-test overlap:

**1. Black-box Methods**

These methods estimate overlap via the model's APIs and test sets, including:
- **Prompting** (Golchin & Surdeanu, 2023): Prompting the model with dataset names and initial segments to check if it reproduces full ground-truth instances.
- **Word Probability** (Shi et al., 2023): Estimating overlap via the probability of anomalous tokens, assuming unseen instances have fewer extremely low-probability tokens.
- **Order Detection** (Oren et al., 2023): Leveraging the assumption that models might memorize the sequential order of test instances.

**Limitations**: Vulnerable to adversarial setups (e.g., developers fine-tuning models to avoid revealing pre-training data), reliant on specific assumptions, and facing severe challenges even under white-box conditions, which are exacerbated in black-box constraints.

**2. Private Test Sets**

Hiding test suites, as seen in evaluations like SQuAD and SEAL.
- **Pros**: Reduces the likelihood of test data being used during training.
- **Cons**: Limits data transparency, forcing the community to rely on a single organization's validation of the test set's integrity.

**3. Novel Test Sets**

Deploying data generated after the model's knowledge cutoff date, such as Livebench.
- **Pros**: Circumvents overlap from a temporal dimension.
- **Cons**: High economic cost of maintaining continuous stream updates as new models emerge, and difficult to interpret longitudinal progress over time.

**4. Canary Strings**

Flagging test sets with unique strings, allowing developers to filter out data containing these identifiers during training.
- **Pros**: Establishes a lightweight signaling mechanism.
- **Cons**: Verification is inconsistent, easily filtered out, and susceptible to false positives.

### Loss & Training

As a position paper, this work does not involve model training. Its core "strategy" is to propose a **reporting standard**: when language model developers publish evaluation results on open benchmarks, they must concurrently publish train-test overlap statistics and/or open-source their training data. This guideline mirrors the statistical convention of requiring confidence intervals alongside reported findings—delegating the choice of method to the researchers, but mandating that the information be disclosed.

## Key Experimental Results

### Main Results: Current Status of Train-Test Overlap Reporting across 30 Models

| Model | Developer | Score | Notes |
|------|--------|------|------|
| OLMo | AI2 | 1 | Open-source training data |
| Pythia | EleutherAI | 1 | Open-source training data |
| RedPajama-INCITE 7B | Together AI | 1 | Open-source training data |
| StarCoder 2 | BigCode | 1 | Open-source training data |
| GPT-4 | OpenAI | 1 | Published overlap methodology and statistics |
| Llama 3.1 | Meta | 1 | Published overlap methodology and statistics |
| Qwen2 | Alibaba | 1 | Published overlap methodology and statistics |
| Palmyra | Writer | 1 | Published overlap methodology and statistics |
| Apple Intelligence | Apple | 1 | Published overlap methodology and statistics |
| Other 21 models | Various | 0 | Failed to sufficiently report train-test overlap |

### Ablation Study: Two Reporting Paths for 1-Point Models

| Reporting Path | Number of Models | Representative Models | Characteristics |
|----------|----------|----------|------|
| Open-source training data | 4 | OLMo, Pythia, StarCoder 2, RedPajama | The community can directly calculate overlap on any test suite. |
| Published overlap statistics | 5 | GPT-4, Llama 3.1, Qwen2, Palmyra, Apple Intelligence | Developers self-report methodologies and calculated outcomes. |
| Post-communication updates | 3 | (New info gathered via author outreach) | Proactive communication led to improved transparency. |

### Key Findings

1. **Severe Deficit in Transparency**: Only 9 out of 30 models (30%) provided sufficient train-test overlap details. 21 models reported bench performance benchmarks without disclosing overlap metrics.
2. **Opacity Breeds Trust Crisis**: The GPT-4 Codeforces incident and Claude 3.5 CTF incident illustrate that the absence of overlap reporting breeds unfounded claims and erodes community trust.
3. **Black-Box Methods Cannot Replace White-Box Reporting**: Existing black-box detection techniques suffer from profound limitations and cannot serve as a substitute for transparent, developer-led disclosure.
4. **Proactive Outreach Sparks Progress**: Researchers extracted new overlap details for 3 additional models through direct communication with developers, demonstrating the efficacy of external auditing.
5. **Alternative Strategies Are No Panacea**: While private test sets, novel test sets, and canary strings are beneficial, none can resolve the issue in isolation; they must be coupled with developer disclosures.

## Highlights & Insights

1. **Precise Demarcation of Paradigm Shifts**: The paper insightfully identifies the transition from "single-party control of train/test data" to "two-party separate control" as the root cause of the evaluation crisis.
2. **Pragmatic Evaluation Standards**: Rather than policing specific overlap detection techniques, the paper establishes a minimum threshold—any meaningful reporting counts—lowering the compliance barrier for developers.
3. **Analogy to Confidence Intervals**: Likening overlap reporting to statistical confidence intervals makes a compelling case: this is not an undue administrative burden, but a foundational requirement for rigorous academic reporting.
4. **Action-Oriented Approach**: Beyond merely diagnosing the predicament, the authors took active steps to contact the 21 developers graded with 0, yielding a direct real-world impact.

## Limitations & Future Work

1. **Coarse-Grained Binary Grading**: The 0/1 binary scoring fails to distinguish between "failing to report entirely" and "reporting, but insufficiently." A more granular rating rubric would provide better developmental incentives.
2. **No Assessment of Reporting Quality**: The paper explicitly bypasses assessing the quality of the detection methodologies used by developers, meaning lightweight or flawed reports can still earn full points.
3. **Temporal Bounds of the Study**: The survey only covers models and publications released prior to September 1, 2024; the rapidly evolving landscape may have shifted since.
4. **Lack of Quantitative Impact Analysis**: The paper does not quantify the precise degree to which varying levels of overlap translate into downstream benchmark inflation, failing to provide a threshold of "how much overlap is too much."
5. **Limited Domain Scope**: The investigation is confined strictly to language models, omitting multi-modal models, code generators, and other foundation models.
6. **Absence of Mechanism Design**: The study primarily appeals to developer ethics, without exploring structural incentives (e.g., leaderboard penalties) or regulatory policies to enforce adherence.

## Related Work & Insights

- **Data Transparency**: The groundwork laid by Longpre et al. (2023) and Bommasani et al. (2024) concerning foundation model transparency serves as a blueprint for this paper.
- **Contamination Detection**: Black-box detection methodologies proposed by Golchin & Surdeanu (2023), Shi et al. (2023), and Oren et al. (2023) are vital companion pieces to this work.
- **Evaluation Infrastructure**: Robust evaluation frameworks like HELM (Liang et al., 2023) and BIG-bench (Srivastava et al., 2023) could serve as central vectors to enforce overlap reporting.
- **Inspirations for Future Work**: Future efforts could build automated overlap detection pipelines, formulate standardized reporting templates for developers, and integrate overlap indices directly into benchmark leaderboards.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 3 | Although the issue is widely recognized, the systematic survey and scoring framework represent novel contributions. |
| Technical Depth | 2 | As a position paper, its technical depth is modest, focusing primarily on policies, governance, and guidelines. |
| Experimental Thoroughness | 3 | The evaluation of 30 models is broad, though it lacks quantitative experimentation on the performance impacts of overlap. |
| Writing Quality | 4 | Extremely coherent, well-structured, logically sound, and backed by relatable examples. |
| Impact | 4 | Addresses a foundational trust issue in LLM evaluation, offering actionable guidelines for industry practices. |
| Overall | 3.5 | An important and timely call to action, though technical contributions are limited. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](../../ICML2026/llm_pretraining/predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[ICLR 2026\] Should We Still Pretrain Encoders with Masked Language Modeling?](../../ICLR2026/llm_pretraining/should_we_still_pretrain_encoders_with_masked_language_modeling.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[ICML 2025\] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning](chameleon_a_flexible_data-mixing_framework_for_language_model_pretraining_and_fi.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)

</div>

<!-- RELATED:END -->
