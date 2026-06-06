---
title: >-
  [Paper Note] Self-Supervised Inductive Logic Programming
description: >-
  [AAAI 2026][Self-Supervised Learning][inductive logic programming] This paper proposes a new self-supervised inductive logic programming (SS-ILP) setting and the Poker system…
tags:
  - "AAAI 2026"
  - "Self-Supervised Learning"
  - "inductive logic programming"
  - "meta-interpretive learning"
  - "grammar learning"
  - "predicate invention"
date: 2026-05-08
content_hash: 1c969cd4572b339a
---

# Self-Supervised Inductive Logic Programming

**Conference**: AAAI 2026
**arXiv**: [2507.16405](https://arxiv.org/abs/2507.16405)  
**Code**: [https://github.com/stassa/aaai_26_experiments/](https://github.com/stassa/aaai_26_experiments/)  
**Area**: Artificial Intelligence / Logic Programming
**Keywords**: inductive logic programming, self-supervised learning, meta-interpretive learning, grammar learning, predicate invention

## TL;DR

This paper proposes a new self-supervised inductive logic programming (SS-ILP) setting and the Poker system, which starts from a small number of positive labeled examples and unlabeled examples, automatically generates positive and negative examples, and employs a maximally general second-order normal form (SONF) background theory to learn logic programs with recursion and predicate invention in the absence of negative examples.

## Background & Motivation

Inductive Logic Programming (ILP) is a paradigm for learning logic programs from positive and negative examples along with a background theory. Meta-Interpretive Learning (MIL) can learn programs with recursion and invented predicates from few examples, achieving strong generalization. However, traditional ILP faces two core burdens:

1. **Background theories must be manually crafted for each learning task**: Domain experts must carefully design the theory with respect to the target program, limiting practical applicability.
2. **Negative examples must be manually selected**: The absence of negative examples leads to over-generalization.

Root Cause: ILP's learning capability depends on carefully designed background theories and negative examples, yet these are precisely the bottlenecks that constrain its broad application.

The core idea of this paper is **contradiction detection**: if two atoms $e_1, e_2$ are both entailed by the same hypothesis $H$ (i.e., $H \models \{e_1, e_2\}$), then assuming $e_1$ is a positive example and $e_2$ is a negative example constitutes a contradiction. By iteratively detecting such contradictions, Poker can automatically classify unlabeled examples as positive or negative, thereby avoiding over-generalization.

## Method

### Overall Architecture

Poker operates in three steps: (1) **Generalise**—construct an initial hypothesis set $T$ from positive labeled examples $E^+$; (2) **Generate**—use $T$ as a generator to produce new examples, add them to the unlabeled example set $E^?$, and assume all are negative; (3) **Label**—examine each hypothetically negative example, relabel it as positive or confirm it as negative via contradiction detection, and iteratively specialize $T$ until consistency is achieved.

### Key Designs

**1. Self-Supervised Contradiction Detection Algorithm**

The core logic of the algorithm (Algorithm 1):

- **Generalise**: Construct an initial hypothesis set satisfying constraints: $T = \{H_{|H| \leq l} : \mathcal{M} \cup B \cup V \models H \wedge B \cup H \models e^+ \in E^+\}$
- **Generate**: Merge all hypotheses $T' = \bigcup T$, use $T'$ to generate $k$ new examples, and add all to the hypothetical negative set $E^- = E^? \cup \{e_i : B \cup T' \models e_i\}$
- **Label**: For each hypothetically negative $e \in E^-$, remove hypotheses that entail it: $T' = T \setminus \{H : B \cup H \models e\}$. If $B \cup T' \models E^+$ (i.e., removing $e$ does not affect positive examples), confirm $e$ as negative ($T \leftarrow T'$); otherwise a contradiction exists—$e$ is actually a positive example and is moved back to $E^+$

**Theorem 1**: The probability that Poker returns a correct hypothesis increases monotonically with the number of unlabeled examples.

**2. Second-Order Normal Form (SONF)**

SONF is a set of constrained meta-rules sufficiently general to learn all correct programs under every possible interpretation $I$ of a target predicate $\Theta$. Specifically, two SONFs are proposed:

**Chomsky-Greibach Normal Form (C-GNF)**—for learning context-free language DCGs:
- Identity meta-rule: $P(x,y) \leftarrow Q(x,y)$, with constraint $\text{target}(P) \wedge (\text{background}(Q) \vee \text{empty}(Q))$
- Chain meta-rule: $P(x,y) \leftarrow Q(x,z), R(z,y)$, with constraint $P \neq Q \wedge (\text{target}(P) \vee \text{invented}(P)) \wedge \neg\text{target}(Q)$
- Tri-Chain meta-rule: $P(x,y) \leftarrow Q(x,z), R(z,u), S(u,y)$ (for palindromes, etc.)

**Lindenmayer Normal Form (LNF)**—for learning L-System grammars.

The key advantage of SONF is that task-specific meta-rules are no longer required; a single SONF applies to an entire problem class. For example, C-GNF can learn any context-free grammar over a binary alphabet.

**3. Constrained Meta-Rule System**

The meta-rule definition is extended with rich constraints:
- $\text{target}(P)$: $P$ must be instantiated to the target predicate symbol
- $\text{invented}(P)$: $P$ must be instantiated to an invented predicate symbol
- $\text{background}(P)$: $P$ must be instantiated to a background theory symbol
- $\text{invented}(P,Q) \rightarrow P \neq Q$: prevents redundant recursion among invented predicates
- Composite constraints: $C_M^1 \wedge C_M^2$, $C_M^1 \vee C_M^2$, $\neg C_M$

These constraints primarily govern recursion, eliminate redundancy, and improve search efficiency.

### Loss & Training

ILP does not employ gradient-based optimization. Poker constructs and specializes hypotheses via SLD-Refutation, using SWI-Prolog's tabled execution (SLG-Resolution) to handle recursion and improve performance. The learning process is purely symbolic reasoning and does not involve numerical optimization.

## Key Experimental Results

### Main Results

**Experiment 1: L-System Grammar Learning** (Dragon Curve, Hilbert Curve, Koch Curve, Sierpinski Triangle)

| System | k (auto-generated examples) | Generation accuracy trend | Hypothesis size trend |
|--------|----------------------------|--------------------------|----------------------|
| Poker | 0 | Lower | Larger (over-generalization) |
| Poker | Increasing | Monotonically increases | Monotonically decreases |
| Louise | 0 (always) | Decreases as positive examples increase | Increases as positive examples increase |

Poker's generation accuracy improves and hypothesis complexity decreases as the number of auto-generated examples grows; Louise persistently over-generalizes without negative examples.

**Experiment 2: Binary Context-Free Language Learning** (6 CFLs: even parity, $1^n0^n$, $n_a=n_b$, $a^nb^m$, palindromes, matching parentheses)

| Setting | TPR | TNR | Notes |
|---------|-----|-----|-------|
| Poker, k=0 | ~1.0 | ~0.0 | No auto-generated examples; complete over-generalization |
| Poker, k increasing | ~1.0 | Gradually → 1.0 | Both TPR and TNR maximized |
| Louise (all k) | ~1.0 | ~0.0 | Persistent over-generalization |

Across all 6 CFLs, Poker achieves the highest TPR and TNR as $k$ increases. Experiments are repeated over 100 randomly sampled sets and standard errors are reported.

### Ablation Study

**Effect of auto-generated example count $k$** (using $1^n0^n$ as a case study):

| $k$ | TPR | TNR | Behavior |
|-----|-----|-----|----------|
| 0 | ~1.0 | ~0.0 | Complete over-generalization |
| Low (e.g., 10) | ~1.0 | Partial improvement | Specialization begins |
| High (e.g., 100+) | ~1.0 | ~1.0 | Correct learning |

This validates the prediction of Theorem 1: accuracy increases monotonically with $k$.

### Key Findings

- Automatically generating negative examples is critical for avoiding over-generalization, and accuracy improves monotonically with sample count.
- SONF successfully replaces manually crafted meta-rule sets; a single C-GNF applies to all binary CFLs.
- Louise (a SOTA MIL system) persistently over-generalizes without negative examples, validating the necessity of the proposed approach.
- The more general the background theory, the more important negative examples become—this is the core trade-off of the paper.

## Highlights & Insights

- **Paradigm innovation**: Defines the new SS-ILP setting, achieving self-supervised learning in ILP for the first time (automatically generating and labeling positive and negative examples).
- **Theoretical guarantee**: Proves that the probability of hypothesis correctness increases monotonically with the number of unlabeled examples.
- **SONF concept**: Replaces task-specific meta-rules with a general second-order normal form, representing a significant practical simplification for MIL.
- **Elegant contradiction detection mechanism**: Distinguishes positive from negative examples via a remove-and-verify procedure, requiring no pre-specified task knowledge.

## Limitations & Future Work

- The derivation of SONFs (from Chomsky/Greibach normal forms) still requires manual effort and has not yet been automated.
- Experiments are limited to grammar learning (CFLs and L-Systems); effectiveness on broader ILP tasks has not been demonstrated.
- Scalability is questionable: the hypothesis set $T$ may grow exponentially with problem size.
- Comparison with modern ILP systems such as Popper is only preliminary; the Popper authors confirmed it does not support learning without negative examples.
- The generality of unlabeled examples affects performance, but this paper only experiments with auto-generated examples.

## Related Work & Insights

- **vs Louise (Vanilla-Louise)**: Louise is a SOTA MIL system using Top Program Construction, but cannot handle the no-negative-examples scenario—it persistently over-generalizes.
- **vs Popper**: Popper does not require a second-order background theory but still needs mode declarations and negative examples, and is only applicable to specific recursive data structures.
- **vs DeepLog**: DeepLog learns from a single example but is restricted to dyadic logic (arity 2); Poker's SONF imposes no arity restriction.
- **vs MetaAbd**: A neuro-symbolic hybrid system that still requires manually defined problem-specific abducible predicates.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Defines a fundamentally new SS-ILP setting; the SONF concept represents a theoretical breakthrough.
- Experimental Thoroughness: ⭐⭐⭐ Covers two categories of grammar learning tasks, but broader ILP benchmarks are lacking.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are rigorous, examples are intuitive, and the choice of grammar learning scenarios is appropriate.
- Value: ⭐⭐⭐⭐ Has the potential to advance practical ILP applications in no-negative-example settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](../../ICML2026/self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](../../CVPR2026/self_supervised/a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](../../ICLR2026/self_supervised/snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)

</div>

<!-- RELATED:END -->
