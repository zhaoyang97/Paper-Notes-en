---
title: >-
  [Paper Note] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games
description: >-
  [AAAI 2026][Behavioral game theory] This paper proposes ElementaryNet, a neural network architecture that is **provably incapable of strategic reasoning**, designed to model "level-0" (non-strategic) human behavior in games. It achieves prediction accuracy statistically indistinguishable from GameNet (current SOTA) while offering substantially better interpretability.
tags:
  - AAAI 2026
  - Behavioral game theory
  - human behavior prediction
  - non-strategic neural network
  - iterated reasoning
  - interpretability
date: 2026-05-08
content_hash: c0968c939b73e5b3
---

# ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games

**Conference**: AAAI 2026
**arXiv**: [2503.05925](https://arxiv.org/abs/2503.05925)
**Code**: [https://github.com/gregdeon/elementarynet](https://github.com/gregdeon/elementarynet)
**Area**: Interpretability
**Keywords**: Behavioral game theory, human behavior prediction, non-strategic neural network, iterated reasoning, interpretability

## TL;DR

This paper proposes ElementaryNet, a neural network architecture that is **provably incapable of strategic reasoning**, designed to model "level-0" (non-strategic) human behavior in games. It achieves prediction accuracy statistically indistinguishable from GameNet (current SOTA) while offering substantially better interpretability.

## Background & Motivation

### Problem Setting
In **normal-form (one-shot simultaneous) games**, human behavior routinely deviates from the rational predictions of classical game theory (e.g., players may choose suboptimal strategies in the Prisoner's Dilemma). Behavioral game theory seeks to construct models that more faithfully predict real human behavior.

### Limitations of Prior Work
The current SOTA model **GameNet** embeds a neural network within the **Quantal Cognitive Hierarchy (QCH)** framework, replacing the conventional uniform distribution with a learnable level-0 network as the non-strategic baseline, achieving performance far superior to traditional models. However, GameNet's best performance occurs when **only level-0 is used (with no strategic reasoning layers at all)**. This raises the core question:

- Is iterated strategic reasoning simply a poor model of human behavior?
- Or is GameNet's level-0 network **too flexible**, covertly simulating strategic reasoning?

### Key Finding and Motivation
The authors **prove that GameNet's level-0 layer is capable of expressing strategic behavior** (specifically, it can precisely simulate a "quantal best response to maxmax"). This means the interpretability GameNet claims is illusory. A new network that is **architecturally incapable of strategic reasoning** is therefore needed to support genuinely credible interpretability conclusions.

## Method

### Overall Architecture

The core idea of ElementaryNet is to introduce an **information bottleneck** into GameNet's feature layers, making it mathematically impossible to express strategic behavior. The overall architecture is:

$$f_i(G) = \sum_{p=1}^{P} w_p \cdot h_i^p(\Phi^p(G))$$

where $\Phi^p$ is a potential function, $h_i^p$ is a response function, and $w_p$ are convex combination weights.

### Key Designs

#### 1. **Proof of GameNet's Strategic Capacity (Theorem 2)**
- **Core Idea**: Constructively specifies a particular parameterization of GameNet that precisely computes a "quantal best response to the maxmax strategy."
- **Mechanism**: A 3-hidden-layer network extracts the opponent's maxmax action via colmax/rowmax operations, then computes the player's optimal response.
- **Key Formulas**: $M_c = \text{colmax}(U^2)$, $M_* = \text{rowmax}(M_c)$, $B = \text{relu}(M_c/C_{gap} - M_*/C_{gap} + 1)$
- **Design Motivation**: Demonstrates that GameNet's ostensibly "non-strategic" level-0 layer can express strategic behavior, exposing a fundamental architectural flaw.

#### 2. **Elementary Model and Information Bottleneck**
- **Core Idea**: Grounded in the theory of elementary behavioral models, potential functions compress both players' utilities into a single scalar value, forming an information bottleneck.
- **Potential Function Definition**: $\varphi^p(x, y) = \theta_x^p x + \theta_y^p y$ (learned-potential variant)
- **Non-Encoding Property**: A linear potential function is either dictatorial (depends on only one input) or non-encoding (maps arbitrarily distant distinct inputs to the same output); both cases preclude strategic reasoning.
- **Design Motivation**: Leverages the mathematical guarantee of the elementary model — a convex combination of elementary models is necessarily weakly non-strategic (Theorem 1, Wright & Leyton-Brown).

#### 3. **Two Instantiations: Learned-Potential and Fixed-Potential**
- **Learned-potential**: $P$ learnable linear potential functions $\varphi^p(x,y) = \theta_x^p x + \theta_y^p y$, with parameters optimized during training.
- **Fixed-potential**: Four fixed potential functions:
    - $\varphi_{\text{own}}(x,y) = x$ (own payoff)
    - $\varphi_{\text{opp}}(x,y) = y$ (opponent's payoff)
    - $\varphi_{\text{sum}}(x,y) = x + y$ (social welfare)
    - $\varphi_{\text{diff}}(x,y) = x - y$ (fairness)
- **Design Motivation**: The fixed potentials correspond to heuristics known from cognitive psychology, enabling investigation of which factors primarily drive human behavior.

#### 4. **Formal Proof of Non-Strategic Property (Theorem 3)**
- **Proof Approach**: The level sets of a linear potential function are either axis-aligned (dictatorial) or extend unboundedly (non-encoding); both cases satisfy the definition of an elementary model.
- **Implication**: Every component of ElementaryNet is an elementary model, and their convex combination is necessarily weakly non-strategic — it cannot express a "quantal best response to any dominance-responsive model."

### Loss & Training

- **Loss Function**: Squared L2 error between predicted and empirical distributions, following prior literature recommendations.
- **Training Protocol**: 60/20/20 train/validation/test split; grid search over 36 hyperparameter combinations (L1 regularization coefficient, dropout probability, initial QCH parameters); model selected by lowest validation loss.
- **Statistical Method**: 50 random splits; paired differences relative to a reference model are reported (to eliminate variance from data partitioning); BCa bootstrap confidence intervals are used.

## Key Experimental Results

### Dataset
An aggregated dataset from 12 experimental studies comprising 26,553 observations across 366 distinct games, drawn from multiple sources including large-scale Amazon Mechanical Turk experiments.

### Main Results

| Model | Configuration | Loss Improvement over Uniform+QCHp | Notes |
|---|---|---|---|
| GameNet + QCHp | 1 layer, 50 units | Best performance | Deeper models overfit |
| ElementaryNet + QCHp | 1 layer, 50 units, 1 learned potential | **No statistical difference** | Equivalent to GameNet's best |
| Uniform + QCHp | — | Baseline | Conventional method |
| ElementaryNet (no QCH) | 1–3 layers | Significantly worse | Demonstrates strategic reasoning is necessary |

**Core Conclusion**: Despite the strict architectural constraints imposed, ElementaryNet + QCHp achieves prediction performance statistically indistinguishable from GameNet.

### Ablation Study

| Configuration | Key Result | Notes |
|---|---|---|
| ElementaryNet + QCHp (full model) | Best | SOTA |
| ElementaryNet without QCH (purely non-strategic) | Very poor (worse than Uniform baseline) | **Demonstrates iterated reasoning is a good model of human behavior** |
| ElementaryNet + QCH1/QCH2/QCH3 | No statistical difference from QCHp | The specific form of the reasoning level distribution matters little |
| Fixed-potential (own only) | Close to Uniform baseline | Own payoff alone is insufficient to model non-strategic behavior |
| Fixed-potential (4 fixed potentials) | Better than own-only, but worse than learned | Cognitive-psychology heuristics are valuable but insufficient |

### Key Findings

1. **Iterated reasoning is a good model of human behavior**: Purely non-strategic models (without QCH) perform very poorly — a reliable conclusion now that the confound of GameNet's "pseudo level-0 that can simulate strategy" has been eliminated.
2. **70%+ of players are level-0**: The QCH model trained jointly with ElementaryNet assigns 70%+ probability mass to level-0, far above the 33% assigned by Uniform + QCHp, indicating that a large proportion of human subjects engage in rich non-strategic reasoning.
3. **Opponent payoff information matters**: Simple level-0 models using only own payoffs perform poorly, confirming that human non-strategic behavior does take opponent payoffs into account.
4. **Richer potential functions add value**: Learned-potential outperforms fixed-potential, suggesting the existence of non-strategic reasoning patterns beyond welfare and fairness considerations.

## Highlights & Insights

1. **Closed theory-practice loop**: The paper first proves GameNet's flaw (theory), then designs a new architecture (design), and finally validates it empirically — a complete and coherent logical chain.
2. **Interpretability with substantive content**: Rather than simply claiming interpretability on grounds of architectural simplicity, the paper mathematically proves that the network cannot express a specific class of behaviors, giving subsequent analyses a causal-inference character.
3. **Negative results are also valuable**: Proving that GameNet's level-0 can simulate strategy explains why adding strategic reasoning layers yields no benefit — a compelling explanation for a previously "counterintuitive" experimental finding.

## Limitations & Future Work

1. **Restricted to 2-player normal-form games**: Sequential interaction, stochastic events, and incomplete information are not addressed, leaving a significant gap from real-world settings.
2. **Response functions remain black boxes**: Although the potential functions are interpretable, the response functions $h_i^p$ that follow are still uninterpretable neural networks.
3. **Potentials limited to linear forms**: Nonlinear potential functions might be more expressive but could undermine the non-strategic guarantees; further theoretical investigation is needed.
4. **Limited data scale**: 366 games and 26K observations are modest relative to model complexity, leading to high variance.

## Related Work & Insights

- **QCH model (Camerer 2004)**: The classical framework for iterated reasoning; this paper replaces the level-0 component within this framework.
- **GameNet**: Replaces hand-crafted level-0 behavior with a permutation-equivariant neural network; the direct target of improvement in this paper.
- **Elementary models (Wright & Leyton-Brown)**: Provide the formal definition of "non-strategic" and the theorem that convex combinations preserve non-strategicness; the theoretical foundation of ElementaryNet.
- **Insight**: In settings that require interpretability, rather than explaining network behavior post hoc, it is preferable to **embed theoretical guarantees at the architectural level** (design for interpretability).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Embeds formal game-theoretic concepts into neural network architecture design in a highly elegant combination of theory and practice
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies, but limited dataset
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear; experimental reasoning is well-structured
- Value: ⭐⭐⭐⭐ — Significant contribution to behavioral game theory, though with a relatively narrow scope of application

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[AAAI 2026\] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)
- [\[AAAI 2026\] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)

<!-- RELATED:END -->
