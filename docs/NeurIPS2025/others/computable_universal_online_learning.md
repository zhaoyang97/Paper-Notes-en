---
title: >-
  [Paper Note] Computable Universal Online Learning
description: >-
  [NeurIPS 2025][universal online learning] This paper introduces computability constraints into the universal online learning framework, proving that "mathematically learnable" does not imply "learnable by a computer program," and provides precise characterizations of computable learning under both agnostic and proper variants.
tags:
  - NeurIPS 2025
  - universal online learning
  - computability
  - online binary classification
  - Littlestone dimension
  - agnostic learning
  - proper learning
date: 2026-05-08
content_hash: f96b004375b29bf5
---

# Computable Universal Online Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.18352](https://arxiv.org/abs/2510.18352)
**Code**: None (purely theoretical work)
**Area**: Other
**Keywords**: universal online learning, computability, online binary classification, Littlestone dimension, agnostic learning, proper learning

## TL;DR

This paper introduces computability constraints into the universal online learning framework, proving that "mathematically learnable" does not imply "learnable by a computer program," and provides precise characterizations of computable learning under both agnostic and proper variants.

## Background & Motivation

Universal online learning is an online binary classification framework introduced by Bousquet et al. (STOC'21). Unlike the classical Littlestone setting, this framework does not require a uniform mistake bound—the Learner only needs to guarantee a finite number of mistakes (though the count may vary with the adversary's strategy). Additionally, the Adversary may dynamically change the target hypothesis (as long as local consistency with the hypothesis class is maintained).

Bousquet et al. provide a combinatorial characterization of universal online learnability: a hypothesis class is learnable if and only if it has ordinal Littlestone dimension. However, the learner in this characterization exists only as a mathematical object (an abstract function)—it may be uncomputable, i.e., not realizable by any computer program.

The core motivation of this paper is: **when can universal online learning be implemented by an actual computer program?** This question has not been systematically studied in prior literature.

## Core Problem

1. **Computability gap**: Does there exist a hypothesis class that is mathematically universally online learnable yet admits no computable learner?
2. **Agnostic learning**: Under computability constraints, are realizable learning and agnostic learning still equivalent?
3. **Proper learning**: Under computability constraints, when does a proper learner exist (one whose outputs always belong to the closure of the hypothesis class)?

## Method

### Basic Setup

The model is an infinite-round game between a Learner and an Adversary. Each round, the Adversary presents $x \in \mathbb{N}$, the Learner predicts a label $\hat{y} \in \{0,1\}$, and the Adversary then reveals the true label $y$. The Adversary must maintain local consistency with the hypothesis class $\mathcal{H}$ but may change the target hypothesis across rounds.

A key concept is the **closure** $\overline{\mathcal{H}}$: the set of all functions locally consistent with $\mathcal{H}$. An Adversary that changes targets is essentially equivalent to one that fixes a function in $\overline{\mathcal{H}}$ from the outset (Closure Lemma).

### Main Result 1: Computability Gap (Proposition 2)

**There exists an RE-representable hypothesis class that is universally online learnable but not computably universally online learnable.**

Proof sketch: The argument leverages the classical result that a computable binary tree may contain uncomputable infinite paths. The construction sets $\mathcal{H} = \{\sigma 0^\infty : \sigma \in T\}$, whose closure $\overline{\mathcal{H}}$ contains all infinite paths of tree $T$. By Lemma 4 (all elements of the closure reachable by a computable learner must themselves be computable), the existence of an uncomputable path implies no computable learner exists.

### Main Result 2: Characterization of Agnostic Learning (Theorem 4 series)

Three equivalent conditions are established:

- (1) $\mathcal{H}$ is learnable by a **total** computable learner in the realizable setting
- (2) There exists an RE class $\mathcal{Z}$ such that $\overline{\mathcal{H}} \subseteq \mathcal{Z}$
- (3) $\mathcal{H}$ is computably agnostically learnable (achieving sublinear regret)

Key insight: The standard reduction from realizable to agnostic learning requires the learner to be defined on non-realizable samples (i.e., to be total). A Separation Theorem shows there exist hypothesis classes that are computably universally online learnable (in the realizable sense) but not computably agnostically learnable.

The separation proof constructs an "evil sequence" class: for each total learner $\varphi_n$, a unique sequence is constructed that forces infinitely many mistakes. This class is learnable by a partial learner but not by any total learner.

### Main Result 3: Unification for RE Classes (Theorem 4)

For RE classes, the gap disappears—computable universal online learning and agnostic learning become equivalent again. The proof exploits the property that RE classes admit a computable enumeration of all realizable samples.

### Main Result 4: Characterization of Proper Learning (Theorem 4)

For an RE class $\mathcal{H}$: a computable proper learner exists if and only if $\overline{\mathcal{H}}$ is itself an RE class.

A separation result further shows there exist RE classes that are computably universally online learnable yet admit no proper learner (Separation of proper vs. improper). The construction follows a priority argument style, defeating each candidate proper learner through incremental enumeration of hypotheses.

## Key Experimental Results

This is a purely theoretical paper with no experiments. The main contributions are a series of theorems and counterexample constructions:

| Result | Content |
|--------|---------|
| Proposition 2 | There exists an RE (even DR) class that is universally online learnable but not computably so |
| Theorem 4 (agnostic) | Total computable realizable ⟺ RE cover of closure ⟺ computable agnostic |
| Theorem 4 (separation) | There exists a class that is computably realizable learnable but not computably agnostically learnable |
| Theorem 4 (RE) | For RE classes, computable realizable ⟺ computable agnostic |
| Theorem 4 (proper) | For RE classes, computable proper learnable ⟺ $\overline{\mathcal{H}}$ is RE |
| Theorem 4 (sep-proper) | There exists an RE class that is computably learnable but admits no proper learner |

## Highlights & Insights

- **Filling an important theoretical gap**: This is the first systematic study of computability in universal online learning, revealing the fundamental distinction between existence proofs and implementability
- **Precise characterizations**: Tight necessary and sufficient conditions are provided for both the agnostic and proper variants, yielding clean results
- **Central role of the closure**: The computability properties of $\overline{\mathcal{H}}$ (whether it admits an RE cover, whether it is itself RE) emerge as the key criteria for learnability
- **Connection to inductive inference theory**: The Projection Lemma parallels the locking sequence concept of Blum & Blum (1975), establishing a bridge between online learning and inductive inference
- **Elegant evil sequence construction**: The diagonalization argument underlying the Separation Theorem is conceptually clear, yielding a concise and powerful construction

## Limitations & Future Work

- **Restriction to countable domains**: The domain is assumed to be $\mathbb{N}$; while practically motivated, this limits the generality of the theory
- **Missing characterization for general classes**: For non-RE hypothesis classes, a precise characterization of computable realizable learning remains an open problem
- **Computability characterization under uniform mistake bounds**: The related problem is noted to be potentially very difficult (Delle Rose et al., 2025)
- **No computational complexity analysis**: The paper addresses only the computability question, leaving efficiency considerations (e.g., polynomial-time computability) unexamined
- **No practical application examples**: Despite the depth of the theoretical results, no guidance is provided on how they might inform the design of practical systems

## Related Work & Insights

| Work | Focus | Distinction from Ours |
|------|-------|----------------------|
| Bousquet et al. (STOC'21) | Combinatorial characterization of universal online learning | This paper adds computability constraints and reveals the gap |
| Littlestone (1988) | Uniform mistake bound (Littlestone dimension) | Extended to computability in the non-uniform setting |
| Assos et al. (2023) | RE classes + finite Littlestone dim → computable | Finite-dimensional results do not generalize to infinite dimensions |
| Hasrati & Ben-David (2023) | Computability of uniform online learning | This paper addresses the more general non-uniform setting |
| Delle Rose et al. (2023) | Computable PAC learning | Different learning framework (PAC vs. online) |

The closure $\overline{\mathcal{H}}$ and its computability properties serve as the central criterion, an observation likely transferable to other learning frameworks. Example 3 draws an analogy between LLM token generation and the Adversary's behavior, suggesting that the universal learning framework may be applicable to analyzing the predictability of LLMs. When hypothesis classes are RE-representable (as is the case for most practical hypothesis classes), the computability gap vanishes and stronger theoretical guarantees hold. The distinction between total and partial learners carries practical significance, as partial learners may loop indefinitely on invalid inputs, which is unacceptable in real deployment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: N/A (purely theoretical)
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[NeurIPS 2025\] FlashMD: Long-Stride, Universal Prediction of Molecular Dynamics](flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[NeurIPS 2025\] On Agnostic PAC Learning in the Small Error Regime](on_agnostic_pac_learning_in_the_small_error_regime.md)

</div>

<!-- RELATED:END -->
