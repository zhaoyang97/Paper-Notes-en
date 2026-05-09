---
title: >-
  [Paper Note] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning
description: >-
  [AAAI 2026][Reinforcement Learning][LTLfMT] This paper proposes a novel reward specification framework based on Linear Temporal Logic over finite traces modulo theories (LTLfMT), replacing manually coded labeling functions with first-order logic formulas. Combined with CRM and HER to address the inherent sparse reward problem in logic-based specifications, the framework achieves significant improvements on continuous control tasks.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - LTLfMT
  - temporal logic
  - reward specification
  - Hindsight Experience Replay
  - sparse reward
date: 2026-05-08
content_hash: 6e46a55045529a8a
---

# Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning

**Conference**: AAAI 2026
**arXiv**: [2602.06227](https://arxiv.org/abs/2602.06227)
**Code**: [github](https://bit.ly/4i20C4Z)
**Area**: Reinforcement Learning / Reward Design
**Keywords**: LTLfMT, temporal logic, reward specification, Hindsight Experience Replay, sparse reward

## TL;DR

This paper proposes a novel reward specification framework based on Linear Temporal Logic over finite traces modulo theories (LTLfMT), replacing manually coded labeling functions with first-order logic formulas. Combined with CRM and HER to address the inherent sparse reward problem in logic-based specifications, the framework achieves significant improvements on continuous control tasks.

## Background & Motivation

### State of the Field

Reward engineering in reinforcement learning is a challenging task that typically relies on domain expertise. Many real-world tasks are inherently non-Markovian (e.g., "pick up the box first, then deliver it to the target"), rendering standard Markovian reward functions ineffective. Logic-based reward specification enables encoding of complex tasks via temporal logic formulas, making reward definitions closer to natural language.

### Limitations of Prior Work

**Expressivity limitations of LTLf**: Classical Linear Temporal Logic over finite traces (LTLf) supports only Boolean predicates, and evaluating these predicates requires manually defined labeling functions.

**Labeling functions as black boxes**: Labeling functions encode the mapping from continuous states to Boolean values by hand, are non-reusable, error-prone, and undermine the interpretability of logic formulas.

**Difficulty with heterogeneous data types**: When states contain heterogeneous types such as position (real numbers), object IDs (strings), and weight (real numbers), LTLf requires separate encoders for each condition.

### Root Cause — Warehouse Robot Example

A robot must navigate to an object's location, identify a specific ID, and satisfy a weight constraint. The predicate is:

$$A = (x-x_o)^2 + (y-y_o)^2 < r^2 \wedge i = \text{"H123"} \wedge weight < 10$$

Under LTLf, three encoders must be hand-written: Euclidean distance threshold detection, ID matching, and weight lookup. The proposed approach: directly write the formula using Nonlinear Real Arithmetic (NRA) theory and let an SMT solver evaluate it automatically, eliminating manual encoding.

## Method

### Overall Architecture

1. **Select a theory**: Choose a first-order theory appropriate for the domain (e.g., NRA for continuous control).
2. **Write the formula**: The user provides only an LTLfMT formula $\varphi$ and a constant assignment dictionary.
3. **Propositionalize**: Replace first-order atoms with propositional letters to obtain an LTLf formula $\varphi'$.
4. **Convert to DFA**: Transform $\varphi'$ into a Deterministic Finite Automaton $\mathcal{A}_{\varphi'}$.
5. **Construct the product MDP**: State space $S' = S \times Q$; DFA transition evaluation is handled by an SMT solver at runtime.
6. **Combine CRM + HER**: Apply counterfactual and hindsight experience replay on the product MDP.

### Key Designs

#### 1. **LTLfMT Syntax and Semantics**

LTLfMT replaces the Boolean alphabet of LTLf with a first-order signature $\Sigma = \mathcal{S} \cup \mathcal{P} \cup \mathcal{C} \cup \mathcal{F} \cup \mathcal{V} \cup \mathcal{W}$. Formulas are structured in three layers:

- $\alpha$ (atomic layer): first-order predicates, e.g., $(x-x_a)^2 + (y-y_a)^2 < r_a^2$
- $\lambda$ (first-order logic layer): logical combinations of atoms, supporting quantifiers $\exists, \forall$
- $\varphi$ (temporal layer): standard temporal operators $X$ (next) and $\mathcal{U}$ (until)

A theory $\mathcal{T}$ is integrated by providing an interpretation for the symbols in the signature. For example, NRA introduces a real-number sort, arithmetic operations $+, \times$, and comparison predicates $<, =$.

**Design Motivation**: Theories serve as composable building blocks; additional theories increase expressivity. Without any additional theory, the framework reduces to standard LTLf.

#### 2. **Identification of a Decidable Fragment**

Full LTLfMT is generally undecidable. This paper identifies a **lookahead-free fragment**:

$$t := v \mid w \mid c \mid f(t_1, \ldots, t_k)$$

The lookahead operator $\bigcirc$ and weak lookahead $\widetilde{\bigcirc}$ are removed, while $X$ and $\mathcal{U}$ are retained at the temporal layer.

**Rationale**: The $\Sigma$ layer should serve solely as a data collection interface capturing immediate environment information; defining temporal constraints over data collection is not useful for reward specification. When the underlying theory is decidable (e.g., NRA admits quantifier elimination), this fragment is fully decidable.

#### 3. **Propositionalization and DFA Construction**

The practical pipeline:
1. Replace each first-order atom $\alpha$ in $\varphi$ with a propositional letter → obtain LTLf formula $\varphi'$.
2. Convert $\varphi'$ to a DFA using the LTLf2DFA library.
3. At runtime, use an SMT solver (e.g., Z3, cvc5) to determine the truth value of each propositional letter at every time step.
4. The SMT solver serves as a **universal labeling function**.

**Key advantage**: Users need only provide a formula string and a constants dictionary; no labeling function implementation is required.

#### 4. **CRM + HER Integration for Sparse Reward**

Logic-based specifications naturally produce sparse rewards. Two techniques are combined:

**CRM (Counterfactual experience)**: Exploits the DFA structure to generate $|Q|$ synthetic experiences per trajectory $\tau$ by substituting DFA states and recomputing rewards.

**HER (Hindsight Experience Replay)**: Uses product MDP states $\langle s, q \rangle$ as goals, where $q$ is a predecessor of the DFA accepting state. The goal space is $\mathcal{G} = S'$ with mapping $m(s) = s$.

**CRM-HER combination**: Each real trajectory → $|Q|$ counterfactual trajectories → HER applied to each → a total of $2|Q|$ synthetic experiences added to the replay buffer.

**Design Motivation**: CRM exploits automaton structure; HER leverages the generalization capacity of continuous state spaces. The LTLfMT framework makes HER goal definition natural by parameterizing goals using first-order constants in the formula.

### Loss & Training

- Algorithm: DDPG (Deep Deterministic Policy Gradients)
- Product MDP construction: Standard $S' = S \times Q$ extension
- Experience generation: At each step, real experiences, CRM counterfactual experiences, and HER experiences are generated simultaneously.

## Key Experimental Results

### Experimental Setup

- Environments: Parking (HighwayEnv, robot controls a car to park in a parking lot) and Reacher
- Tasks of increasing complexity: from simple goal reaching (parking_0) to multi-step sequential reaching with safety constraints (parking_2)
- Example formula: $\varphi = \neg F\neg(x \geq x_{min} \wedge x \leq x_{max}) \wedge F(\alpha_1 \wedge F(\alpha_2))$
- 20 independent runs, 95% confidence intervals
- Baselines: Baseline (vanilla DDPG), CRM, HER, CRM-HER

### Main Results

| Method | parking_0 | parking_1 | parking_2 | task_1 | task_2 |
|--------|-----------|-----------|-----------|--------|--------|
| Baseline | Low | Fails | Fails | Low | Fails |
| CRM | Moderate | Low | Low | **High** | High |
| HER | High | **High** | Moderate | Moderate | Moderate |
| CRM-HER | **High** | **High** | **Only success** | **High** | **High** |

### Ablation Study

| Component | parking_simple | parking_complex | Notes |
|-----------|---------------|-----------------|-------|
| Baseline only | ✓ | ✗ | Complete failure on complex tasks |
| +CRM | ✓ | Partial | Helps with DFA complexity but not sparsity |
| +HER | ✓ | Partial | Addresses goal discovery but ignores automaton |
| +CRM-HER | ✓ | ✓ | Only method consistently succeeding on all tasks |

### Key Findings

1. **CRM-HER consistently optimal**: Achieves the best or near-best performance across all tasks; notably the only method capable of learning a successful policy on the most complex task (parking_2).
2. **Complementary effects**: CRM excels at handling DFA complexity; HER excels at goal discovery; each has blind spots when used alone.
3. **Zero-encoding advantage**: Users need only supply a formula string and a constants dictionary; no labeling function implementation is required.
4. **Theory–expressivity tradeoff**: The lookahead-free fragment maintains decidability while being sufficiently expressive for most continuous control tasks.

## Highlights & Insights

1. **Elimination of labeling functions**: Using an SMT solver as a universal labeling function constitutes a substantive simplification of existing logic-based reward specification.
2. **Composability of the theoretical framework**: Theories serve as building blocks; incorporating different theories extends applicability to new domains (database queries, string matching, etc.).
3. **Practically decidable fragment**: The identified lookahead-free fragment is both decidable and sufficiently expressive, representing a solid theoretical contribution.
4. **Elegant CRM + HER combination**: The first integration of counterfactual reasoning from reward machines with hindsight experience replay, with goal definition automated by the LTLfMT framework.
5. **Opening new directions**: Reintroducing first-order terms into specification languages paves the way for safety constraints, persistent rewards, and formal verification in continuous-domain RL.

## Limitations & Future Work

1. **SMT solver efficiency for NRA**: SMT solving for nonlinear real arithmetic introduces computational overhead at each step, potentially limiting real-time applicability.
2. **DFA state explosion**: Complex formulas cause exponential growth in DFA state counts, affecting the scale of the product MDP.
3. **Binary rewards only**: The current framework supports only binary (achieved/not-achieved) rewards and cannot express graded reward signals.
4. **Limited experimental environments**: Validation is restricted to Parking and Reacher; more complex robotic manipulation tasks have not been tested.
5. **Restricted quantifier usage**: Although the framework supports quantifiers, practical usage is constrained by theory decidability.

## Related Work & Insights

- **LTLf (De Giacomo & Vardi 2013)**: Classical temporal logic over finite traces; the direct theoretical foundation for this work.
- **Reward Machines (Icarte 2018)**: Programmatic alternative to logic-based specifications; source of the CRM technique.
- **HER (Andrychowicz 2017)**: Hindsight Experience Replay; a standard method for addressing sparse rewards.
- **LTLfMT (Geatti 2022)**: First-order temporal logic modulo theories; the theoretical basis of this paper.
- **Restraining Bolts (De Giacomo 2019)**: Feature functions map world attributes to propositional atoms but require manual implementation.
- Implications for continual learning and safe RL: Safety constraints can be directly encoded via first-order logic.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of LTLfMT to RL reward specification; CRM + HER combination is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task, multi-baseline comparisons with confidence intervals; limited environment diversity.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical framework with clear illustrations (warehouse robot example used throughout).
- Value: ⭐⭐⭐⭐⭐ — Significantly lowers the barrier to using logic-based reward specification and opens new research directions.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[AAAI 2026\] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India](regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)

<!-- RELATED:END -->
