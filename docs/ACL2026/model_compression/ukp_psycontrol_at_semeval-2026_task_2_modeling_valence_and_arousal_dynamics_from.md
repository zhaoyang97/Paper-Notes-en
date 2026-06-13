---
title: >-
  [Paper Note] UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text
description: >-
  [ACL 2026][Model Compression][Emotion assessment] UKP_Psycontrol achieved first place in both subtasks at SemEval-2026 Task 2. By combining LLM prompting, a MaxEnt model with Ising interactions…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Emotion assessment"
  - "longitudinal analysis"
  - "valence-arousal"
  - "LLM prompting"
  - "MaxEnt model"
date: 2026-05-08
content_hash: ce01f7e9a86eb90e
---

# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text

**Conference**: ACL 2026  
**arXiv**: [2604.21534](https://arxiv.org/abs/2604.21534)  
**Code**: [GitHub](https://github.com/)  
**Area**: Affective Computing / Longitudinal Emotion Modeling  
**Keywords**: Emotion assessment, longitudinal analysis, valence-arousal, LLM prompting, MaxEnt model

## TL;DR

UKP_Psycontrol achieved first place in both subtasks at SemEval-2026 Task 2. By combining LLM prompting, a MaxEnt model with Ising interactions, and a neural regression model, the study found that LLMs excel at capturing static emotional signals, while short-term emotional changes are explained more by recent numerical trajectories than by textual semantics.

## Background & Motivation

**Background**: In computational sentiment analysis, emotions are typically represented by two continuous dimensions: valence (positive/negative) and arousal (level of activation). Most NLP research utilizes social media or review data, which is annotated by external evaluators or approximated via emotional proxies, only indirectly accessing internal emotional states.

**Limitations of Prior Work**: SemEval-2026 Task 2 poses a new challenge—requiring longitudinal emotional assessment and prediction from time-series ordered self-reported texts. The data consists of logs from US service workers over several years, including free prose and lists of emotional words, paired with self-rated valence and arousal. This requires models to not only understand the sentiment of a single text but also capture the dynamic changes of emotion over time.

**Key Challenge**: The contributions of textual semantics versus numerical emotional trajectories to emotion prediction may differ—LLMs are proficient at understanding textual meaning, but short-term emotional fluctuations might reflect an individual's emotional inertia rather than new textual information.

**Goal**: (1) Evaluate static emotion recognition (Subtask 1); (2) Predict future emotional changes (Subtask 2A); (3) Understand the relative importance of textual semantics vs. numerical trajectories across different tasks.

**Key Insight**: A combination of three complementary methods—LLM prompting (leveraging linguistic understanding), MaxEnt+Ising (leveraging structured dependency modeling in probabilistic graphical models), and neural regression (leveraging numerical trajectories and user embeddings).

**Core Idea**: Static emotional assessment relies on LLM textual understanding, while dynamic emotional prediction depends on the short-term inertia of numerical trajectories; both have their respective strengths.

## Method

### Overall Architecture

The system consists of three modules: (1) an LLM prompting module—using GPT-5 to predict valence and arousal under user-aware and user-agnostic settings; (2) a MaxEnt+Ising module—modeling structured dependencies of emotional states using a Maximum Entropy model and Ising interactions; (3) a neural regression module—predicting the next emotional change using a sliding window of recent emotional trajectories + RoBERTa text embeddings + trainable user embeddings.

### Key Designs

1.  **LLM Prompting Strategy**:
    - **Function**: Leverages the language understanding capabilities of LLMs to predict emotions within the text.
    - **Mechanism**: Distinguishes between user-aware (using historical examples from the same user as few-shot) and user-agnostic (using label-balanced random examples) settings. Compares two output formats: textual emotion labels (mapped to numerical values) vs. direct numerical prediction, finding textual labels to be more stable. Introduces a sliding window dynamic update strategy (most recent N=15 entries), but observes that prediction errors propagate cumulatively, making fixed examples preferable. Prose and emotional words are processed separately.
    - **Design Motivation**: Utilizes the strong correlation between LLMs and human valence/arousal scores, with the user-aware setting capturing individual expression patterns.

2.  **MaxEnt+Ising Structural Model**:
    - **Function**: Uses a probabilistic graphical model to capture structured dependencies between emotional states.
    - **Mechanism**: Defines an energy function $E(\mathbf{x}) = -\mathbf{x}^\top \mathbf{h} - \frac{1}{2}\mathbf{x}^\top \mathbf{J}\mathbf{x}$, where $\mathbf{h}$ models linear effects and $\mathbf{J}$ captures pairwise interactions. Emotional variables are one-hot encoded, and semantic information is compressed into binary vectors via an autoencoder. Since the state space is bounded, the partition function $Z$ can be calculated exactly, enabling maximum likelihood training. Inference is performed via conditional expectation decoding, generating continuous predictions aligned with correlation evaluation metrics.
    - **Design Motivation**: Psychological theories suggest that mental states evolve on an underlying energy landscape and follow a Boltzmann distribution.

3.  **Neural Regression Model (Subtask 2A)**:
    - **Function**: Uses recent emotional trajectories to predict the next emotional change.
    - **Mechanism**: Inputs include recent text embeddings (RoBERTa mean-pooling) within a sliding window (1-4 steps), current valence/arousal, previous state changes, and trainable user embeddings. Compares three settings: (a) a text-less baseline (numerical features + user embeddings only), (b) text-enhanced, and (c) semantic cluster representations.
    - **Design Motivation**: Hypothesizes that short-term emotional changes are primarily driven by personal emotional inertia, with user embeddings capturing individual differences.

### Loss & Training

The MaxEnt model is trained using maximum likelihood estimation, while the neural regression model uses standard regression loss. LLM prompting requires no training. The final submission combines the optimal configurations from each module.

## Key Experimental Results

### Main Results

**Subtask 1 (Longitudinal Emotion Assessment) — Test Set**

| Method | Valence r_composite | Arousal r_composite |
| :--- | :--- | :--- |
| Baseline linear(BERT) | 0.557 | 0.299 |
| MaxEnt Ising | 0.589 | 0.327 |
| **LLM-based (Submitted)** | **0.667** | **0.554** |

**Subtask 2A (Emotion Change Prediction) — Test Set**

| Method | Valence r | Arousal r |
| :--- | :--- | :--- |
| Baseline linear(prev) | 0.520 | 0.609 |
| MaxEnt Ising | — | — |
| **Neural Regression (Submitted)** | **0.675** | **0.683** |

### Key Findings

- User-aware prompting was only slightly better than user-agnostic prompting, indicating that label-balanced random examples already approximate most user-specific information.
- Increasing the number of shots improved valence correlation (10→20 shots: 0.617→0.661), but had no similar effect on arousal.
- Textual emotion label prediction outperformed direct numerical prediction, suggesting that natural language descriptions better match the LLM's pre-training distribution.
- **Key Finding**: In Subtask 2A, the performance of the text-less baseline (numerical trajectory + user embeddings only) was comparable to the text-enhanced model, illustrating that short-term emotional changes are explained more by recent numerical states than by textual semantics.
- Dynamic update strategies (sliding window replacement) were inferior to fixed shots due to the cumulative propagation of prediction errors.

## Highlights & Insights

- The finding that "short-term emotional change is driven by numerical trajectories rather than textual semantics" provides significant insights for the field of affective computing—implying that emotion may have its own time-series inertia, where text serves as a snapshot rather than a driver.
- The MaxEnt+Ising model introduces psychological theory (the energy landscape hypothesis) into NLP tasks, providing an interpretable probabilistic framework.
- The complementary combination strategy of the three methods is noteworthy: LLMs handle semantic understanding, probabilistic models handle structured dependencies, and neural networks handle numerical sequences.

## Limitations & Future Work

- Small data volume (2764 entries, 137 users) limits the potential of deep learning methods.
- Significant data quality issues: 92% of users participated in only one time period, and some users' valence/arousal remained constant throughout.
- The binary semantic representation in the MaxEnt model may lose emotional nuances.
- Validation was only conducted on English-speaking service workers; cultural differences may affect emotional expression patterns.

## Related Work & Insights

- **vs. Pure LLM methods**: LLMs are powerful for static assessment but insufficient for dynamic prediction, requiring supplementation by numerical trajectory modeling.
- **vs. Traditional BERT baselines**: The system combining multiple methods (LLM+MaxEnt+Neural Regression) achieved first place in both subtasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination strategy of the three methods and the application of MaxEnt+Ising are innovative, though the individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation and comparisons, though limited by the data scale of the shared task.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed methodological descriptions.
- Value: ⭐⭐⭐⭐ The "text vs. numerical trajectory" finding provides important insights for affective computing research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](../../AAAI2026/model_compression/distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)

</div>

<!-- RELATED:END -->
