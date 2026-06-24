---
title: >-
  [Paper Note] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models
description: >-
  [ICML 2025][Self-Supervised Learning][world model] This paper proposes "Inductive Bias Probes", which evaluate whether the extrapolation behavior of foundation models aligns with hypothesized world models by repeatedly fine-tuning them on synthetic datasets. The findings reveal that while foundation models can accurately predict sequences in domains such as orbital mechanics, Othello, and lattice problems, they do not truly learn the underlying world models but rather develop…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "world model"
  - "foundation model"
  - "inductive bias"
  - "sequence prediction"
  - "Newtonian mechanics"
date: 2026-05-08
content_hash: b1c57926cc988ed2
---

# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models

**Conference**: ICML 2025  
**arXiv**: [2507.06952](https://arxiv.org/abs/2507.06952)  
**Code**: [https://github.com/keyonvafa/inductive-bias-probes](https://github.com/keyonvafa/inductive-bias-probes)  
**Area**: Self-Supervised Learning  
**Keywords**: world model, foundation model, inductive bias, sequence prediction, Newtonian mechanics

## TL;DR

This paper proposes "Inductive Bias Probes", which evaluate whether the extrapolation behavior of foundation models aligns with hypothesized world models by repeatedly fine-tuning them on synthetic datasets. The findings reveal that while foundation models can accurately predict sequences in domains such as orbital mechanics, Othello, and lattice problems, they do not truly learn the underlying world models but rather develop task-specific heuristic strategies.

## Background & Motivation

**Background**: A core hypothesis of foundation models is that "learning to predict sequences will lead to the discovery of deeper domain understanding"—just as Kepler's predictions of planetary motion eventually led to the discovery of Newtonian mechanics. Current approaches to evaluate whether foundation models have learned world models primarily follow two paths: (1) mechanistic probing (probing internal representations); (2) static behavioral testing (evaluating outputs on a single task).

**Limitations of Prior Work**: Mechanistic probes are difficult to scale to large models, and internal representations do not necessarily reflect actual behavior. Static behavioral tests observe only a single task, failing to capture the adaptation capability of foundation models to new tasks—which is precisely their core application scenario.

**Key Challenge**: The No Free Lunch theorem indicates that the performance of a learning algorithm depends on its preference for learning certain classes of functions (its inductive bias). If a foundation model has truly learned a world model, its inductive bias should lean toward the functions permitted by that world model, meaning it should automatically follow the structure of the world model when extrapolating from small amounts of data. However, existing methods cannot systematically test this.

**Goal**: Given a foundation model and a hypothesized world model, how can one quantitatively test whether the model possesses the world model as its inductive bias?

**Key Insight**: Scientists use world models to make inferences from small amounts of data—similarly, the world model of a foundation model should be exposed through its extrapolation behavior on small datasets.

**Core Idea**: By repeatedly fine-tuning the foundation model on small synthetic datasets consistent with the world model, one can observe whether its extrapolation behavior "respects" and "distinguishes" the state structures of the world model.

## Method

### Overall Architecture

The Inductive Bias Probe is a general evaluation workflow: (1) Given a hypothesized world model (including state spaces and state mappings); (2) Generate a large number of small synthetic datasets consistent with the world model; (3) For each synthetic dataset, let the foundation model learn on the training set and then extrapolate on test inputs; (4) Compare the extrapolation behavior of the foundation model with that of an oracle that "knows the true state"—if the inductive bias of the foundation model aligns with the world model, the two should be highly consistent.

### Key Designs

1. **R-IB and D-IB Metrics (Special Cases for Finite State Spaces + Binary Outputs)**:

    - **Function**: Quantify whether the inductive bias of a foundation model is consistent with the world model.
    - **Mechanism**: Define two complementary metrics: **R-IB (Respecting State)** = $\mathbb{E}[\mathbf{1}(\hat{m}_D(X_i), \hat{m}_D(X_j)) | \phi(X_i)=\phi(X_j)]$, which measures whether inputs with the same state yield the same prediction; and **D-IB (Distinguishing State)** = $1 - \mathbb{E}[\mathbf{1}(\hat{m}_D(X_i), \hat{m}_D(X_j)) | \phi(X_i) \neq \phi(X_j)]$, which measures whether inputs of different states result in different predictions. Analogous to precision and recall, both are indispensable—always outputting the same constant yields a perfect R-IB but a D-IB of zero.
    - **Design Motivation**: Prediction accuracy alone cannot distinguish whether a model has learned the state structure or has merely guessed the correct answer.

2. **Generalized Inductive Bias Probes (Continuous Outputs + General State Space + General Tasks)**:

    - **Function**: Generalize the probe to continuous domains such as orbital mechanics.
    - **Mechanism**: Define **extrapolative predictability** $\hat{I}(x_i, x_j) = -\min_{h \in \mathcal{H}} \mathbb{E}_D[\ell(h(\hat{m}_D(x_i)), \hat{m}_D(x_j))]$, which measures how predictable the extrapolations for two inputs are across multiple synthetic datasets. An oracle version is defined and compared: if the model behaves like the oracle, the IB curve should fall on the 45-degree diagonal line. This is essentially a calibration curve.
    - **Design Motivation**: In continuous domains, "whether states are identical" needs to be generalized to "how similar the states are", making a more flexible metric necessary.

3. **Validation of the "Next-Token Partition" Hypothesis**:

    - **Function**: Reveal what inductive bias the foundation model actually utilizes when it has not learned the world model.
    - **Mechanism**: Propose and validate the hypothesis that the inductive bias of foundation models points toward state partitions "sharing the same set of valid next tokens," rather than the true world states. By decomposing D-IB into $\text{D-IB}_{q=}$ (different states but same valid next token) and $\text{D-IB}_{q\neq}$ (different states and different valid next tokens), the study finds that $\text{D-IB}_{q=} < \text{D-IB}_{q\neq}$ holds across all models, indicating that models group based on next-token partitions rather than the true state.
    - **Design Motivation**: Answer the critical question: "If not a world model, what did the model actually learn?"

### Loss & Training

This paper does not propose a new training loss function, but uses MSE as the loss function $\ell$ for extrapolative predictability during the evaluation phase.

## Key Experimental Results

### Main Results

Performance of R-IB and D-IB on the lattice problem (5 states) and Othello (1 is perfect, 0 is uninformative model):

| Model | Lattice R-IB↑ | Lattice D-IB↑ | Othello R-IB↑ | Othello D-IB↑ |
|---|---|---|---|---|
| RNN (untrained) | 0.346 | 0.749 | 0.228 | 0.990 |
| RNN (NTP trained) | 0.574 | 0.803 | 0.632 | 0.797 |
| Transformer (untrained) | 0.268 | 0.742 | 0.708 | 0.843 |
| Transformer (NTP trained) | 0.483 | 0.677 | 0.703 | 0.624 |
| Mamba (NTP trained) | 0.571 | 0.866 | 0.682 | 0.728 |
| LSTM (NTP trained) | 0.782 | 0.921 | 0.563 | 0.610 |

### Ablation Study

Force laws recovered by Transformer via symbolic regression in orbital mechanics (compared with the ground truth $F \propto m_1 m_2/r^2$):

| Data Slice | Recovered Force Law |
|---|---|
| Galaxy 1 | $F \propto (\sin(1/\sin(r-0.24))+1.45) \times 1/(1/r+m_2)$ |
| Galaxy 2 | $F \propto \cos(\cos(2.19 \times m_1))$ |
| Galaxy 3 | $F \propto \cos(\sin(0.48/m_1))$ |
| Galaxy 4 | $F \propto \sin(r+8569.2+1/m_1)$ |
| Galaxy 5 | $F \propto \cos(\cos(e^{m_2}))$ |

### Key Findings

- The orbital mechanics model achieves a prediction accuracy of $R^2 > 0.9999$, but the inductive bias probe reveals that it has entirely failed to learn Newtonian mechanics—recovering completely different and meaningless gravitational laws across different data subsets.
- Transformers exhibit the worst inductive bias among all architectures—on the lattice problem, the D-IB actually decreases after training compared to the untrained state, suggesting that pre-training might damage state distinction capabilities.
- LSTM performs the best on the lattice problem (R-IB = 0.782), and recurrent/state-space architectures generally outperform Transformers.
- In Othello, even when the predicted board state is not entirely correct, the matching rate of valid move sets is still significantly higher than that of the board matches, confirming that the model learns a coarse representation "sufficient to recover valid moves" rather than the complete board state.

## Highlights & Insights

- Formulates the philosophical question of "whether foundation models learn world models" into a quantitatively testable statistical framework, making a significant academic contribution.
- The analogy of "from Kepler to Newton" is highly apt—accurate prediction does not equate to understanding the underlying laws.
- The finding that models learn a "next-token partition" rather than the true state is a profound insight for understanding the limitations of LLMs.
- The experimental design spans physics, game theory, and combinatorics, demonstrating strong persuasiveness.

## Limitations & Future Work

- The tested world model must be hypothesized in advance, making it unable to automatically discover the representations actually used inside the model.
- The models in the experiments are small-scale (109M parameters); it remains unclear whether large-scale LLMs exhibit similar behaviors.
- The computational overhead of the inductive bias probe is non-trivial, requiring multiple fine-tuning runs and collection of extrapolation results.
- The current findings incline toward "negative results" (models do not learn world models), lacking constructive suggestions on how to improve them.
- It lacks an in-depth explanation of "why LSTM performs better than Transformer"—is the inductive bias of hidden states naturally more suited to tracking states?
- The choice of sampling distribution $P_D$ and input distribution $P_X$ for synthetic datasets could significantly affect the conclusions, but sensitivity analysis is lacking.

## Related Work & Insights

- It complements the internal probes for Othello by Nanda et al. (2023)—while the latter inspects internal representations, this work examines behavioral performance.
- Lemos et al. (2023) show that modifying model architectures to explicitly encode Newtonian laws can recover laws of gravitation, suggesting that domain-specific inductive biases are critical.
- The study on valid move prediction in Othello by Li et al. (2023) is highly consistent with the "next-token partition" discovery of this work.
- Insight: Evaluating foundation models should not merely rely on prediction accuracy; one should also examine whether their extrapolation follows the expected structure of world models.

## Rating

⭐⭐⭐⭐ (8/10)

This is a research work with a unique perspective and rigorous methodology. The proposed inductive bias probe framework is a significant addition to the evaluation toolbox for foundation models. The cross-domain experimental design is ingenious, and the finding that "models learn next-token partitions instead of world models" is profound and has widespread influence. Limitations include the small scale of the models, the lack of constructive improvements, and the questionable reproducibility of some experiments (e.g., symbolic regression in orbital mechanics).

Furthermore, the findings in this paper have important implications for AI Safety: if models only learn surface heuristics rather than true world models, they might make seemingly plausible yet fundamentally incorrect decisions when facing out-of-distribution scenarios. This also forms an interesting contrast with the "grokking" phenomenon (where a model suddenly generalizes after overfitting)—although the models in this work achieve extremely high prediction accuracy ($R^2 > 0.9999$), generalization tests reveal that they do not truly understand the underlying laws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[CVPR 2025\] OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad](../../CVPR2025/self_supervised/ocrt_boosting_foundation_models_in_the_open_world_with_object-concept-relation_t.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

</div>

<!-- RELATED:END -->
