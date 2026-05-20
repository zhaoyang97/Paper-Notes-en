---
title: >-
  [Paper Note] Emergent World Beliefs: Exploring Transformers in Stochastic Games
description: >-
  [NeurIPS 2025][Reinforcement Learning][Emergent World Models] This work extends the study of emergent world models in LLMs from perfect-information games (Othello…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Emergent World Models"
  - "Transformer"
  - "Poker"
  - "POMDP"
  - "Belief State"
  - "Activation Probing"
  - "GPT-2"
date: 2026-05-08
content_hash: 5f508e3767165b4e
---

# Emergent World Beliefs: Exploring Transformers in Stochastic Games

**Conference**: NeurIPS 2025
**arXiv**: [2512.23722](https://arxiv.org/abs/2512.23722)  
**Code**: [GitHub](https://anonymous.4open.science/r/poker-interp-4653/)  
**Area**: Reinforcement Learning
**Keywords**: Emergent World Models, Transformer, Poker, POMDP, Belief State, Activation Probing, GPT-2

## TL;DR
This work extends the study of emergent world models in LLMs from perfect-information games (Othello, Chess) to the partial-information setting (Texas Hold'em Poker). By pre-training GPT-2 on PHH-format poker data and probing its internal activations, the paper demonstrates that the model not only learns deterministic features (hand rank recognition at ~98% accuracy) but also spontaneously develops internal representations of stochastic features (win rate/equity, correlation coefficient 0.59).

## Background & Motivation
Research on the internal world representations of Transformer LLMs has made significant progress in perfect-information games:

- **OthelloGPT** (Li et al.): A model trained solely on legal move sequences spontaneously develops internal representations of board states; nonlinear probes can decode cell occupancy, and causal interventions can flip cells and influence downstream move predictions.
- **ChessGPT** (multiple works): Linear decoders can recover piece/square features, and editing these features predictably alters move probabilities.
- **Nanda et al.**: Demonstrates that the world model in OthelloGPT can be extracted even with linear probes.

However, all of the above work is restricted to **perfect-information** games, where all players can observe the complete game state. In the real world, many decision-making scenarios involve **incomplete information** (e.g., poker, business negotiation, military games). In these POMDP environments, a model must maintain a **belief distribution** over hidden states rather than a deterministic state.

Core question: Can an LLM, trained on sequential data from an incomplete-information game, spontaneously develop internal representations corresponding to POMDP belief states?

## Core Problem
1. In partially observable stochastic games (e.g., poker), can Transformers develop a deterministic world model (e.g., hand rank recognition)?
2. More importantly, can Transformers develop a **stochastic world model**—i.e., probabilistic estimates of hidden information such as the opponent's hole cards or win rate/equity?

## Method

### Dataset Generation
Due to the lack of large-scale complete poker hand history datasets, the authors generate data via simulation:
- 6-player No-Limit Texas Hold'em games
- AI agents driven by simulation-based equity estimation
- Each agent randomly initialized with distinct playing style parameters (aggression, tightness, bluff frequency, call willingness, bet sizing, etc.)
- Over 2 million synthetic hands in PHH (Poker Hand History) format

### Model Training
- **Architecture**: GPT-2 base configuration (12 attention heads, hidden dimension 768, ~87M parameters)
- **Training**: AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$), learning rate $5 \times 10^{-5}$
- **Special design**: A `<GAP>` special token is inserted in the input sequence to replace certain tokens, which are moved to appear after an `<ANS>` marker; loss is computed only on tokens following `<ANS>`
- **Training scale**: 13 epochs (stopped upon validation loss plateau), effective batch size 128, 95–5 train/test split
- **Hardware**: NVIDIA H200 GPU, ~7 hours

### Probe Design

#### Linear Classification Probe
$$p_\theta(x_t^l) = \text{argmax}(W x_t^l), \quad \theta = \{W \in \mathbb{R}^{C \times F}\}$$
where $F$ is the activation vector dimension and $C$ is the number of classes.

#### Two-Layer MLP Probe
$$p_\theta(x_t^l) = \text{argmax}(W_1 \text{ReLU}(W_2 x_t^l)), \quad \theta = \{W_1 \in \mathbb{R}^{C \times H}, W_2 \in \mathbb{R}^{H \times F}\}$$
where $H$ is the hidden dimension. For stochastic features (continuous values), the argmax is removed and the probe directly outputs a regression value.

### Probing for Deterministic World Models
The probe targets **hand rank** categories such as high card, one pair, two pair, three of a kind, etc. To avoid over-representation of high-frequency hand ranks (e.g., high card, one pair):
- Sample counts for each category are capped at the 40th percentile
- Independent probes are trained at each layer

### Probing for Stochastic World Models
The probe targets **hand equity** (win rate):
- Simulation-based equity estimates serve as labels
- All hole cards except Player 1's are deliberately masked
- A two-layer MLP regression probe is trained to predict equity values

### Theoretical Grounding: Linear Decodability of Belief States

A formal theoretical analysis is provided in the appendix. In a POMDP, the belief state $b_t(s) = \Pr(S_t = s | H_t)$ is a sufficient statistic for decision-making. For any finite-horizon event $F$:

$$\Pr_\pi(F | H_t) = \sum_{s \in \mathcal{S}} b_t(s) v_F(s) = \langle b_t, v_F \rangle$$

That is, predictions of future events are **linear functions** of the belief state. If the Transformer stores an affine transformation of the belief state in its residual stream, a linear probe exists that can recover the relevant predictions—providing a theoretical foundation for linear probe detectability.

## Key Experimental Results

### Deterministic Features: Hand Rank Recognition

| Probe Type | Accuracy | Notes |
|-----------|----------|-------|
| Linear probe | ~80% | Hand rank classification |
| MLP probe | **~98%** | Hand rank classification, 40th-percentile balancing |

- A strongly diagonal confusion matrix confirms that the model's internal activations reliably encode hand rank information.
- Low cross-seed variance indicates high representational consistency.
- Early layers (Layer 0–3) exhibit the strongest encoding.

### Stochastic Features: Hand Equity (Win Rate)

| Metric | Value | Notes |
|--------|-------|-------|
| Correlation coefficient $r$ | **0.59** | Cross-seed average, Layer 0 |
| Strongest $R^2$ layers | Layer 0–4 | Decreasing with depth |
| $R^2$ decay pattern | Information bottleneck-like | Deeper layers retain information more relevant to token prediction |

### Action Recognition (Additional Experiment)
After masking action tokens (fold, check/call, etc.) and probing:
- Both linear and MLP probes achieve ~80% accuracy
- The model has learned to associate actions with contextual information

### Activation Space Visualization
- **t-SNE**: Clear hand rank clusters; pair and three-of-a-kind form compact regions
- **PCA**: Triangular activation structure, resembling the belief state geometry described by Shai et al.
- **UMAP**: Partial clustering behavior
- Individual hand ranks (e.g., pair) exhibit **multiple sub-clusters**, suggesting the model learns finer-grained subcategories (pairs of different face values)

## Highlights & Insights
- **Significance of domain extension**: This is the first work to advance the study of emergent world models in LLMs to the incomplete-information setting, representing a conceptually important step.
- **Evidence for stochastic belief representations**: An equity correlation of 0.59 is not exceptionally high, but it demonstrates that the model does spontaneously develop probabilistic estimation capabilities through unsupervised training.
- **Strong theory–experiment alignment**: The formal proof of linear decodability of POMDP belief states provides a solid foundation for the probing experiments.
- **Information bottleneck pattern**: The decay of $R^2$ with layer depth is consistent with information bottleneck theory, whereby deeper layers compress away input information unrelated to direct token prediction.
- **Multi-cluster phenomenon**: The multiple sub-clusters observed for the pair hand rank suggest that the model has learned a richer internal structure beyond simple classification.

## Limitations & Future Work
- **Limited dataset scale**: Synthetic data generation is computationally expensive, and data volume constrains both model capability and the upper bound of probe results.
- **Simplification of synthetic data**: AI agents use simple equity-plus-heuristic strategies and lack the adaptive learning capacity of human players or sophisticated game-theoretic agents, potentially resulting in insufficient data diversity.
- **Insufficient rare hand samples**: Rare hand ranks such as straights and flushes are underrepresented in the dataset, precluding accurate probing of these categories.
- **Small model scale**: GPT-2 base (87M parameters); belief representations may be stronger or exhibit different encoding patterns in larger models.
- **Equity correlation has room for improvement**: A value of 0.59 is meaningful but falls short of high-fidelity belief representation.
- **Absence of causal intervention experiments**: Only passive probing is conducted; unlike OthelloGPT, no activation editing for causal validation is performed, leaving it unclear whether the identified representations are actually used in model decisions.
- **Generalizability to other POMDP domains**: Whether the findings transfer beyond poker to other incomplete-information domains remains an open question.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension from perfect-information to incomplete-information settings is conceptually significant.
- Experimental Thoroughness: ⭐⭐⭐ Deterministic probing is thorough; the correlation coefficient for stochastic probing offers moderate persuasive force; causal validation is absent.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, the theoretical appendix is rigorous, and the experimental presentation is systematic.
- Value: ⭐⭐⭐⭐ Opens a new window for understanding how LLMs internally model uncertainty.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)
- [\[NeurIPS 2025\] The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis](the_world_is_bigger_a_computationally-embedded_perspective_on_the_big_world_hypo.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)

</div>

<!-- RELATED:END -->
