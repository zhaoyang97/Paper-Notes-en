---
title: >-
  [Paper Note] UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text
description: >-
  [ACL 2026][Model Compression][Affect Assessment] UKP_Psycontrol achieves first place on both subtasks of SemEval-2026 Task 2 by combining LLM prompting, a MaxEnt model with Ising interactions, and a neural regression model. The system reveals that LLMs excel at capturing static affective signals, whereas short-term affective changes are better explained by recent numerical trajectories than by textual semantics.
tags:
  - ACL 2026
  - Model Compression
  - Affect Assessment
  - Longitudinal Analysis
  - Valence-Arousal
  - LLM Prompting
  - MaxEnt Model
date: 2026-05-08
content_hash: e91ba2b2a46e5404
---

# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text

**Conference**: ACL 2026
**arXiv**: [2604.21534](https://arxiv.org/abs/2604.21534)
**Code**: [GitHub](https://github.com/)
**Area**: Affective Computing / Longitudinal Affect Modeling
**Keywords**: Affect Assessment, Longitudinal Analysis, Valence-Arousal, LLM Prompting, MaxEnt Model

## TL;DR

UKP_Psycontrol achieves first place on both subtasks of SemEval-2026 Task 2 by combining LLM prompting, a MaxEnt model with Ising interactions, and a neural regression model. The system reveals that LLMs excel at capturing static affective signals, whereas short-term affective changes are better explained by recent numerical trajectories than by textual semantics.

## Background & Motivation

**Background**: In computational sentiment analysis, affect is typically represented along two continuous dimensions: valence (positive–negative polarity) and arousal (activation level). Most NLP research relies on social media or review data annotated by external raters or approximated via sentiment proxies, allowing only indirect access to internal affective states.

**Limitations of Prior Work**: SemEval-2026 Task 2 introduces a novel challenge requiring longitudinal affect assessment and prediction over temporally ordered self-report texts. The dataset consists of diary entries from U.S. service-industry workers spanning several years, comprising free-prose narratives and affect word lists paired with self-rated valence and arousal scores. This demands that models not only understand the affect conveyed in individual texts but also capture how affect evolves over time.

**Key Challenge**: Textual semantics and numerical affect trajectories may contribute differently to affect prediction. LLMs are strong at understanding textual meaning, yet short-term affective fluctuations may reflect personal affective inertia rather than new information conveyed in the text.

**Goal**: (1) Evaluate static affect recognition (Subtask 1); (2) predict future affective change (Subtask 2A); (3) understand the relative importance of textual semantics versus numerical trajectories across tasks.

**Key Insight**: Three complementary approaches are combined — LLM prompting (leveraging language understanding), MaxEnt+Ising (leveraging structured dependency modeling via probabilistic graphical models), and neural regression (leveraging numerical trajectories and user embeddings).

**Core Idea**: Static affect assessment benefits from LLM-based textual understanding, while dynamic affect prediction benefits from the short-term inertia of numerical trajectories; the two are complementary.

## Method

### Overall Architecture

The system comprises three modules: (1) an LLM prompting module that uses GPT-5 to predict valence and arousal under user-aware and user-agnostic settings; (2) a MaxEnt+Ising module that employs a maximum entropy model with Ising interactions to capture structured dependencies among affective states; and (3) a neural regression module that predicts the next affective change using recent affect trajectories within a sliding window, RoBERTa text embeddings, and trainable user embeddings.

### Key Designs

1. **LLM Prompting Strategy**:

    - **Function**: Exploits LLMs' language understanding to predict affect from text.
    - **Mechanism**: Two settings are distinguished — user-aware (using historical examples from the same user as few-shot demonstrations) and user-agnostic (using label-balanced randomly sampled examples). Two output formats are compared: textual affect labels (mapped to numerical values) versus direct numerical predictions; textual labels prove more stable. A sliding-window dynamic update strategy (most recent $N=15$ entries) is explored but found to accumulate prediction errors; fixed demonstrations perform better. Prose and affect word lists are processed separately.
    - **Design Motivation**: LLM outputs exhibit strong correlation with human valence/arousal ratings; the user-aware setting captures individual expressive patterns.

2. **MaxEnt+Ising Structured Model**:

    - **Function**: Captures structured dependencies among affective states via a probabilistic graphical model.
    - **Mechanism**: An energy function is defined as $E(\mathbf{x}) = -\mathbf{x}^\top \mathbf{h} - \frac{1}{2}\mathbf{x}^\top \mathbf{J}\mathbf{x}$, where $\mathbf{h}$ models linear effects and $\mathbf{J}$ captures pairwise interactions. Affective variables are one-hot encoded, and semantic information is compressed into binary vectors via an autoencoder. Because the state space is bounded, the partition function $Z$ can be computed exactly, enabling maximum likelihood training. At inference time, continuous predictions are decoded via conditional expectations, aligning with the correlation-based evaluation metric.
    - **Design Motivation**: Psychological theory posits that mental states evolve on a latent energy landscape and follow a Boltzmann distribution.

3. **Neural Regression Model (Subtask 2A)**:

    - **Function**: Predicts the next affective change from recent affect trajectories.
    - **Mechanism**: Inputs include recent text embeddings (RoBERTa mean-pooling) within a sliding window of 1–4 steps, current valence/arousal values, the previous state change, and trainable user embeddings. Three configurations are compared: (a) a text-free baseline (numerical features + user embeddings only), (b) text-augmented, and (c) semantic cluster representations.
    - **Design Motivation**: Short-term affective changes are hypothesized to be primarily driven by individual affective inertia; user embeddings capture individual differences.

### Loss & Training

The MaxEnt model is trained via maximum likelihood estimation; the neural regression model uses a standard regression loss. LLM prompting requires no training. The final submission combines the optimal configurations of each module.

## Key Experimental Results

### Main Results

**Subtask 1 (Longitudinal Affect Assessment) — Test Set**

| Method | Valence $r_\text{composite}$ | Arousal $r_\text{composite}$ |
|--------|------------------------------|------------------------------|
| Baseline linear (BERT) | 0.557 | 0.299 |
| MaxEnt Ising | 0.589 | 0.327 |
| **LLM-based (submission)** | **0.667** | **0.554** |

**Subtask 2A (Affect Change Prediction) — Test Set**

| Method | Valence $r$ | Arousal $r$ |
|--------|-------------|-------------|
| Baseline linear (prev) | 0.520 | 0.609 |
| MaxEnt Ising | — | — |
| **Neural Regression (submission)** | **0.675** | **0.683** |

### Key Findings

- The user-aware prompting setting yields only marginally better results than the user-agnostic setting, suggesting that label-balanced random demonstrations already approximate most user-specific information.
- Increasing the number of shots improves valence correlation (10→20 shots: 0.617→0.661) but produces no comparable effect for arousal.
- Predicting textual affect labels outperforms direct numerical prediction, indicating that natural language descriptions better align with LLM pretraining distributions.
- **Key finding**: In Subtask 2A, the text-free baseline (numerical trajectories + user embeddings only) performs comparably to text-augmented models, demonstrating that short-term affective changes are better explained by recent numerical states than by textual semantics.
- The dynamic update strategy (sliding-window replacement of demonstrations) underperforms fixed demonstrations due to accumulated propagation of prediction errors.

## Highlights & Insights

- The finding that "short-term affective changes are driven by numerical trajectories rather than textual semantics" carries important implications for affective computing — suggesting that affect has its own temporal inertia and that text serves as a snapshot rather than a driving force.
- The MaxEnt+Ising model introduces a psychological theory (the energy landscape hypothesis) into NLP tasks, providing an interpretable probabilistic framework.
- The complementary combination strategy across three methods is noteworthy: LLMs handle semantic understanding, probabilistic models handle structured dependencies, and neural networks handle numerical sequences.

## Limitations & Future Work

- The dataset is relatively small (2,764 entries, 137 users), limiting the potential of deep learning approaches.
- Data quality issues are prominent: 92% of users participated in only one time period, and some users exhibit no variation in valence/arousal ratings.
- The binarized semantic representations in the MaxEnt model may lose affective nuance.
- Validation is limited to English-speaking service-industry workers; cultural differences may influence affective expression patterns.

## Related Work & Insights

- **vs. pure LLM methods**: LLMs are powerful for static assessment but insufficient for dynamic prediction, necessitating complementary numerical trajectory modeling.
- **vs. traditional BERT baseline**: The combined system (LLM + MaxEnt + neural regression) achieves first place on both subtasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination strategy and the application of MaxEnt+Ising are novel, though individual components are not new in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations and comparisons, albeit constrained by the shared task's data scale.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed method descriptions.
- Value: ⭐⭐⭐⭐ The "text vs. numerical trajectory" finding offers important insights for affective computing research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)

<!-- RELATED:END -->
