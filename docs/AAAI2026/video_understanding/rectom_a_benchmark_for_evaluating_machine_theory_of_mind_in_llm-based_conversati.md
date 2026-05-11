---
title: >-
  [Paper Note] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems
description: >-
  [AAAI 2026][Video Understanding][Theory of Mind] This paper introduces RecToM, the first human-annotated benchmark for evaluating Theory of Mind (ToM) reasoning capabilities of LLMs in conversational recommender systems…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Theory of Mind"
  - "Conversational Recommender Systems"
  - "LLM Evaluation"
  - "Cognitive Reasoning"
  - "Behavioral Prediction"
date: 2026-05-08
content_hash: 0dbaaac908d19b65
---

# RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems

**Conference**: AAAI 2026
**arXiv**: [2511.22275](https://arxiv.org/abs/2511.22275)
**Code**: [github.com/CGCL-codes/RecToM](https://github.com/CGCL-codes/RecToM)
**Area**: Video Understanding / Conversational Recommendation
**Keywords**: Theory of Mind, Conversational Recommender Systems, LLM Evaluation, Cognitive Reasoning, Behavioral Prediction

## TL;DR

This paper introduces RecToM, the first human-annotated benchmark for evaluating Theory of Mind (ToM) reasoning capabilities of LLMs in conversational recommender systems (CRS). It covers two dimensions—cognitive inference (desire/intention/belief) and behavioral prediction (strategy prediction/strategy judgment)—comprising 10 question types and 20,524 QA pairs, and exposes systematic deficiencies of current LLMs in fine-grained intention inference and strategy judgment.

## Background & Motivation

**Background**: Large language models (LLMs) are transforming conversational recommender systems (CRS), demonstrating remarkable capabilities in instruction understanding, reasoning, and human-computer interaction. Effective recommendation dialogue depends on the ability to infer and reason about users' mental states (e.g., desires, intentions, beliefs)—a cognitive capacity referred to in cognitive science as **Theory of Mind (ToM)**.

**Limitations of Prior Work**:

**Existing benchmarks are ill-suited for CRS**: Current ToM benchmarks (e.g., Hi-ToM, FANTOM, NegotiationToM) primarily rely on the Sally-Anne test paradigm, using simplified synthetic narratives (e.g., a person entering a room, moving objects). These lack the complexity of real conversational scenarios and are inadequate for evaluating mental state reasoning in recommendation dialogues.

**Behavioral prediction dimension is neglected**: Existing benchmarks focus mainly on **retrospective reasoning** over past dialogue (e.g., inferring beliefs or intentions) while ignoring a core aspect of human ToM—**using inferred mental states to guide strategic decision-making in future interactions**.

**CRS-specific challenges are unmodeled**: Recommendation dialogues exhibit unique characteristics—asymmetric roles (recommender vs. seeker), hierarchical intentions (coarse/fine-grained), multi-dimensional beliefs (multiple aspects jointly determining attitude), and concurrent desires (simultaneously holding different preferences toward multiple recommended items)—none of which are captured by general-purpose ToM benchmarks.

**Key Challenge**: There is no effective evaluation framework to assess whether LLMs in CRS can genuinely understand users' mental states and make strategic decisions accordingly.

**Key Insight**: Drawing on the BDI (Belief-Desire-Intention) cognitive model, the paper constructs a comprehensive ToM evaluation benchmark covering both cognitive inference and behavioral prediction within real recommendation dialogues.

## Method

### Overall Architecture

RecToM is an evaluation benchmark (not a model). Its core design revolves around two reasoning dimensions:

- **Cognitive Inference**: Evaluates the ability of LLMs to infer the mental states of dialogue participants, including desire inference, intention inference, and belief inference.
- **Behavioral Prediction**: Evaluates the ability of LLMs to predict and assess future dialogue strategies using inferred mental states, including strategy prediction and strategy judgment.

### Key Designs

#### 1. **Data Construction**

**Data Source**: Based on the ReDial dataset (movie recommendation dialogues), 253 satisfactory recommendation dialogues (where seekers first rejected then accepted recommendations) and 83 unsatisfactory dialogues (where all recommendations were rejected) are selected, yielding 336 dialogues, 4,583 turns, and 20,524 QA pairs.

**Human Annotation**: Three trained doctoral students participated in annotation—two performed initial labeling and the third resolved conflicts. Fleiss's K = 0.79 (substantial agreement). Annotated content includes:
- **Belief dimension**: Identifying specific utterances in which seekers explicitly express their attitude toward recommended movies.
- **Desire dimension**: Annotating each mentioned movie along three core dimensions—Suggestion (who proposed it), Seen (whether watched), and Liked (whether enjoyed).

#### 2. **Question Type Design (10 Types)**

**Cognitive Inference**:

- **Desire Inference** (1,448 QA): Binary single-choice questions—"Is the seeker likely to watch [movie]?"—tracking dynamically changing motivational states.
- **Intention Inference** (recommender: 2,205 + seeker: 2,205 QA × coarse/fine-grained):
    - **Coarse-grained**: 5 categories for recommender, 4 for seeker (multiple-choice)
    - **Fine-grained**: 10 categories for recommender, 16 for seeker (multiple-choice)
    - Question format: "Given the dialogue history, what intention does [recommender/seeker] express in [utterance]?"
- **Belief Inference** (1,762 QA): 7-option single-choice—"What does the recommender believe is the seeker's attitude toward [movie]?"—requiring integration of multiple dimensions including who proposed the movie, whether it was seen, and whether it was liked.

**Behavioral Prediction**:

- **Strategy Prediction** (recommender: 2,098 + seeker: 2,149 QA): Multiple-choice—"What strategy will [recommender/seeker] adopt next?"
- **Strategy Judgment** (recommender: 2,098 + seeker: 2,149 QA): Binary single-choice—"Is the strategy [strategy] adopted by [recommender/seeker] to advance the conversation effective?"

#### 3. **Four Distinctive Design Features**

- **Multi-choice Strategy**: A single utterance may express multiple distinct intentions, necessitating multi-label selection.
- **Multi-granular Intention**: Intentions are organized hierarchically into coarse-grained categories and fine-grained sub-intentions.
- **Multi-dimensional Belief**: Beliefs about an item require integrated reasoning across multiple associated dimensions.
- **Multi-concurrent Desire**: Seekers simultaneously hold different preferences toward multiple recommended items.

### Evaluation Protocol

- **Zero-shot direct answering**: LLMs are prompted to select answers directly without explanation.
- **Chain-of-Thought prompting (CoT)**: The instruction "Let's think step by step" is added to elicit explicit reasoning.
- Temperature is uniformly set to 0.7.

## Key Experimental Results

### Main Results

| Model | Fine-grained Intent (Rec) | Coarse-grained Intent (Rec) | Belief | Fine-grained Intent (Seek) | Coarse-grained Intent (Seek) | Desire | Strategy Pred. (Rec) | Strategy Judge. (Rec) | Strategy Pred. (Seek) | Strategy Judge. (Seek) |
|------|----------|----------|------|----------|----------|------|---------|---------|---------|---------|
| Random | 0.10 | 3.23 | 14.29 | 0.00 | 6.67 | 50.00 | 3.23 | 50.00 | 6.67 | 50.00 |
| Human | 64.32 | 86.31 | 96.84 | 59.92 | 82.74 | 98.25 | 87.44 | 96.37 | 85.18 | 97.23 |
| GPT-4o | 32.61 | 40.45 | 74.74 | 28.84 | 64.22 | 92.27 | 24.07 | 33.84 | 49.23 | 32.34 |
| DeepSeek-v3 | 29.71 | 44.26 | 69.86 | 33.20 | 59.32 | 86.05 | 26.84 | 39.18 | 48.02 | 35.60 |
| DeepSeek-v3+CoT | 33.02 | 46.21 | 79.46 | 29.61 | 58.59 | 76.10 | 19.54 | 37.94 | 38.11 | 35.55 |
| Model Average | 27.74 | 41.13 | 68.72 | 28.20 | 55.77 | 86.35 | 20.54 | 34.84 | 30.59 | 34.53 |

### Ablation Study / Strategy Judgment Bias Analysis

| Model | Prediction Bias (↓) | FPR (↓) | Recall-No (↑) |
|------|---------------------|--------|-------------|
| GPT-4o | 94.90 | 94.45 | 5.55 |
| GPT-4o+CoT | 94.08 | 94.44 | 5.56 |
| GPT-4o-mini | 99.07 | 98.91 | 1.09 |
| DeepSeek-v3 | 88.42 | 85.84 | 14.16 |
| Claude 3.5 | 97.86 | 97.64 | 2.36 |
| Model Average (Recommender) | ~93.37 | ~93.28 | ~7.22 |

### Key Findings

1. **Multi-choice complexity severely impairs ToM reasoning**: Multiple-choice tasks (fine-grained intention) yield an average of only 27.74% for recommenders, significantly lower than single-choice tasks (desire: 86.35%, belief: 68.72%), indicating that LLMs incur excessive cognitive load when discriminating among multiple plausible options.

2. **Fine-grained intention discrimination is the core bottleneck**: Coarse-grained intention classification is comparatively manageable (GPT-4o seeker: 64.22%), but performance drops sharply on fine-grained tasks (28.84%). Fine2Coarse analysis shows that while fine-grained outputs are imprecise, most fall within the correct coarse-grained category—models get the "direction right but lack sufficient precision."

3. **Severe answer sycophancy**: In the strategy judgment task, LLMs exhibit a Prediction Bias of ~93%, FPR of ~93%, and Recall-No of only ~7%—nearly all responses are "Yes," even when a strategy is clearly ineffective. This is highly problematic and would lead to excessively affirmative recommendation strategies in CRS deployment.

4. **CoT prompting yields marginal and unstable gains**: GPT-4o with CoT actually decreases seeker coarse-grained intention accuracy from 64.22% to 54.10%, suggesting that CoT may introduce noise in complex dialogue reasoning tasks.

5. **Multi-dimensional belief reasoning shows nascent capability**: DeepSeek-v3+CoT achieves 79.46% on belief inference, far exceeding the random baseline of 14.29%, indicating that at sufficient model scale with structured prompting, LLMs possess a preliminary ability to integrate multi-dimensional cues for coherent reasoning.

## Highlights & Insights

1. **First ToM benchmark for CRS**: The paper pioneers systematic evaluation of LLMs' ToM capabilities in real recommendation dialogues by integrating the BDI cognitive model with recommender system evaluation.
2. **Introduction of the behavioral prediction dimension**: Beyond assessing "understanding of mental states," the benchmark further evaluates the ability to "use mental states to guide actions," more closely approximating the complete definition of human ToM.
3. **Quantitative revelation of answer sycophancy**: Through confusion matrix analysis (Prediction Bias / FPR / Recall-No), the paper systematically quantifies LLMs' tendency toward obsequious responses, carrying important implications for CRS deployment.
4. **Multi-granularity intention analysis methodology**: The Fine2Coarse error analysis reveals a failure pattern of "correct direction but insufficient precision," providing clear guidance for future improvements.
5. **High annotation quality**: Annotation by doctoral students, Fleiss's K = 0.79, dual annotation with third-party arbitration.

## Limitations & Future Work

1. **Limited data scale**: Only 336 dialogues, which may not capture the full diversity of recommendation scenarios.
2. **Single domain (movie recommendation)**: Not extended to other recommendation domains (e.g., music, products, travel); generalizability remains to be validated.
3. **English-only evaluation**: Multilingual models (e.g., Chinese) are not covered.
4. **No model improvement proposals**: As a pure benchmark contribution, the paper identifies problems without proposing solutions (e.g., ToM-enhanced fine-tuning methods).
5. **Static evaluation**: The QA format cannot fully assess real-time ToM reasoning in dynamic multi-turn interactions.

## Related Work & Insights

- Compared to NegotiationToM (negotiation scenarios) and PersuasiveToM (persuasion scenarios), RecToM focuses on the unique dynamics of recommendation contexts—asymmetric roles, hierarchical intentions, and concurrent desires—serving as an exemplar of applying ToM evaluation to specific application domains.
- The extreme manifestation of answer sycophancy in strategy judgment (FPR ~93%) suggests that in scenarios requiring LLMs to render "negative judgments," dedicated debiasing training or adversarial evaluation may be necessary.
- The performance gap between fine-grained and coarse-grained intention recognition implies that future intention modeling in CRS should adopt a hierarchical approach, proceeding from coarse to fine granularity.

## Rating

- Novelty: ⭐⭐⭐⭐ (First ToM benchmark for CRS; behavioral prediction dimension is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (5 models, 2 prompting strategies, 10 question types comprehensively evaluated, with error analysis)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous structure, in-depth analysis, insightful findings)
- Value: ⭐⭐⭐⭐ (Provides a systematic benchmark for evaluating and improving ToM capabilities of LLMs in CRS)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[AAAI 2026\] LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding](livibench_an_omnimodal_benchmark_for_interactive_livestream_video_understanding.md)
- [\[AAAI 2026\] LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph](loom_personalized_learning_informed_by_daily_llm_conversations_toward_long-term_.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](../../ICLR2026/video_understanding/log_probability_tracking_of_llm_apis.md)

</div>

<!-- RELATED:END -->
