---
title: >-
  [Paper Note] On Learning Verifiers and Implications to Chain-of-Thought Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][verifier] This paper proposes a formal PAC learning framework for Chain-of-Thought verifiers, defining three progressively stronger verification objectives (Simple → Trustable → γ-Trustable)…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "verifier"
  - "chain-of-thought"
  - "PAC learning"
  - "sample complexity"
  - "trustable verification"
  - "computational complexity gap"
date: 2026-05-08
content_hash: 2ed7ce70ccc0cd61
---

# On Learning Verifiers and Implications to Chain-of-Thought Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.22650](https://arxiv.org/abs/2505.22650)
**Code**: None
**Area**: LLM Reasoning / Computational Learning Theory
**Keywords**: verifier, chain-of-thought, PAC learning, sample complexity, trustable verification, computational complexity gap

## TL;DR

This paper proposes a formal PAC learning framework for Chain-of-Thought verifiers, defining three progressively stronger verification objectives (Simple → Trustable → γ-Trustable). It proves that when each problem admits only a bounded number of correct proofs, the sample complexity is $O(\log|H|)$; however, when the number of correct proofs is unbounded, the sample complexity inevitably grows to $\Theta(|H|)$, unless the verifier class satisfies additional structural assumptions such as intersection-closure. The paper also exploits the USAT problem to demonstrate a computational complexity gap between verification and generation.

## Background & Motivation

**Background**: CoT reasoning has become the dominant paradigm for LLMs to solve complex mathematical and logical problems, yet reasoning chains frequently contain erroneous or unjustified inference steps. Formal proof systems (e.g., Lean) can verify rigorously, but current LLMs struggle to generate formal proofs directly—formalizing even an informal problem statement is itself challenging.

**Limitations of Prior Work**:
- CoT reasoning suffers from "error accumulation"—subtle mistakes in long reasoning chains are difficult to detect.
- Existing LLM verifiers lack theoretical guarantees; it is unknown how much training data suffices to learn a reliable verifier.
- More critically, once a verifier provides feedback and the reasoning model revises its chain, the new chain is out-of-distribution (OOD), and the original verifier no longer offers any guarantee.

**Key Challenge**: The simplest verification objective (correctness under the same distribution) cannot handle adaptive use scenarios—the verifier's own feedback induces distributional shift. A stronger verification guarantee is needed: for a given problem, reject *all* incorrect reasoning chains, not merely the in-distribution ones.

**Goal**: To establish a rigorous PAC learning framework for CoT verifiers, answer the fundamental question of how much data is required to learn a reliable verifier, and reveal the intrinsic sample complexity differences across verification objectives of varying strength.

**Core Idea**: The learnability of a verifier depends on how "trustable" one requires it to be. Moving from "correct on the training distribution" to "rejecting all incorrect proofs" increases the required sample size from logarithmic to linear; however, an intersection-closed structure can break through this barrier.

## Method

### Overall Architecture

The paper defines a unified verification framework: a problem space $X$, a reasoning-step space $\Sigma$, and a verifier $h: X \times \Sigma^* \to \{\text{YES}, \text{NO}\}$. Given a problem $x_0$ and a reasoning chain $\tau = (x_1, \ldots, x_T)$, the verifier inspects each prefix $h(x_0, (x_1, \ldots, x_j))$ sequentially and identifies the first erroneous step upon the first NO output. Three progressively stronger verification objectives are defined within this framework.

### Key Designs

1. **Simple Verification (SVPAC)**:

    - Function: Learn a verifier from randomly labeled reasoning-chain samples; classify reasoning chains and locate the first erroneous step with low error rate under the same distribution.
    - Mechanism: The training set consists of (problem, reasoning chain, label) triples, where labels indicate either "correct" or the position of the first error. The learner selects a consistent verifier $h \in H$ via ERM.
    - Theoretical Results: For finite $H$, sample complexity is $O\!\left(\frac{1}{\epsilon}\left(\log|H| + \log\frac{1}{\delta}\right)\right)$; for finite VC dimension, $O\!\left(\frac{1}{\epsilon}\left(\operatorname{VCdim}(H)\log T + \log\frac{1}{\delta}\right)\right)$, with only logarithmic dependence on chain length $T$.
    - Limitation: The guarantee applies only to in-distribution reasoning chains. Once the verifier's feedback causes the reasoning model to revise its chain, the new chain is OOD—this motivates Trustable verification.

2. **Trustable Verification (TVPAC)**:

    - Function: For *most* problems, reject *any* incorrect reasoning chain (not limited to in-distribution ones), while accepting all gold-standard correct proofs.
    - Mechanism: Assuming a gold-standard reasoner $g$ that provides $\leq k$ correct reasoning chains per problem $x$, positive examples (correct chains) and negative examples (single-step deviations from correct chains) are constructed from $g(x)$. The training distribution operates at the problem level ($x \sim D$), while the guarantee over reasoning chains is global.
    - Theoretical Results: Sample complexity is of the same order as SVPAC—$O\!\left(\frac{1}{\epsilon}\left(\log|H| + \log\frac{1}{\delta}\right)\right)$ for finite $H$; $O\!\left(\frac{1}{\epsilon}\left(\operatorname{VCdim}(H)\log(kT|\Sigma|) + \log\frac{1}{\delta}\right)\right)$ for finite VC dimension.
    - Additional Finding: When $k=1$, stepwise TVPAC verification is equivalent to CoT generation—a good verifier can be used to generate correct proofs by enumerating accepted steps from $\Sigma$. This equivalence breaks down when $|\Sigma|$ is large.

3. **γ-Trustable Verification (γ-TVPAC)**:

    - Function: When the number of correct proofs is unbounded, learn a verifier from randomly sampled correct proofs (rather than all of them) per problem; require the verifier to accept $\geq \gamma$ of the correct proofs while rejecting all incorrect ones.
    - Mechanism: Algorithm 1 (intersection-consistent verifier)—take the intersection of all verifiers consistent with the training set, accepting only when every consistent verifier outputs YES. This guarantees zero false-positive rate (never accepting an incorrect proof).
    - Theoretical Results: For general finite $H$, sample complexity is $O\!\left(\frac{|H|}{\eta\epsilon}\right)$—a jump from logarithmic to linear. Lower bounds establish this is unavoidable (Theorems 4.10 and 4.11).
    - Breakthrough: If the verifier class $H$ is intersection-closed, the complexity reduces to $O\!\left(\frac{\operatorname{VCdim}(H)}{\eta\epsilon}\right)$.

### Loss & Training

- **SVPAC loss**: $\ell_h(x_0, \tau) = \mathbb{I}[h(x_0, \tau_j) \neq h^*(x_0, \tau_j) \text{ for some } j \leq \mathsf{f}(h^*, (x_0, \tau))]$ — the verifier must agree with the ground-truth verifier on every step up to and including the first error.
- **TVPAC**: ERM operates on positive and negative samples constructed from the gold-standard reasoner.
- **γ-TVPAC**: Takes the intersection of consistent verifiers (Algorithm 1); no gradient-based optimization is required—purely combinatorial/set operations.

## Key Experimental Results

This is a purely theoretical paper with no experiments. Core theoretical results are summarized below.

### Main Results

| Verification Objective | Data Format | Finite $H$ Complexity | Finite VCdim Complexity | Learning Algorithm |
|---|---|---|---|---|
| SVPAC (§3) | Random (problem, reasoning chain, first-error position) | $O\!\left(\frac{\log\|H\|}{\epsilon}\right)$ | $O\!\left(\frac{\operatorname{VCdim}(H)\log T}{\epsilon}\right)$ | ERM |
| TVPAC (§4.1) | Random problem + $\leq k$ gold proofs | $O\!\left(\frac{\log\|H\|}{\epsilon}\right)$ | $O\!\left(\frac{\operatorname{VCdim}(H)\log(kT\|\Sigma\|)}{\epsilon}\right)$ | ERM (with negative example construction) |
| γ-TVPAC (§4.2) | Random (problem, single random correct proof) | $O\!\left(\frac{\|H\|}{\eta\epsilon}\right)$ | $O\!\left(\frac{\operatorname{VCdim}(H)}{\eta\epsilon}\right)$ if intersection-closed | Intersection of consistent verifiers |
| Agnostic SVPAC (§C) | Same as SVPAC | $O\!\left(\frac{\log\|H\|}{\epsilon^2}\right)$ | $O\!\left(\frac{\operatorname{VCdim}(H)\log T}{\epsilon^2}\right)$ | ERM |

### Ablation Study

| Theorem | Condition | Lower Bound | Implication |
|---|---|---|---|
| Theorem 4.10 | Proper learner, $\|\Sigma\| \geq 2$ | $\Omega(\|H\|)$ (soundness only) | Even requiring only that incorrect proofs are rejected, proper learning demands linear samples. |
| Theorem 4.11 | Improper learner, $\frac{1}{2}$-complete | $\Omega(\|H\|)$ (soundness + completeness) | Even with an improper learner that may output verifiers outside $H$, adding completeness preserves the linear lower bound. |

### Key Findings

- **Complexity phase transition**: An exponential sample complexity jump exists between "bounded correct proofs" (TVPAC, $O(\log|H|)$) and "unbounded correct proofs" (γ-TVPAC, $\Theta(|H|)$), arising from the intrinsic difficulty of learning from positive examples only.
- **Intersection-closure as the key structural condition**: If $H$ is intersection-closed (e.g., axis-aligned hyperrectangles, CNF formulas), the complexity of γ-TVPAC reduces to $O(\operatorname{VCdim}(H))$, circumventing the linear lower bound.
- **Computational gap between verification and generation**: Using the USAT (Unique SAT) problem, the paper constructs a concrete example where a verifier is computable in polynomial time (checking whether an assignment satisfies all clauses), but generating a correct proof is equivalent to solving USAT—which admits no polynomial-time algorithm under $\text{RP} \neq \text{NP}$.
- **Verifiers can enable generation**: When $k=1$ and $|\Sigma|$ is finite, a stepwise TVPAC verifier can generate correct proofs by enumerating accepted steps—"a good verifier can train a good reasoner."
- **Agnostic extension**: Extending the framework to the non-realizable setting (where no perfect verifier exists in $H$) increases the sample complexity from $O(1/\epsilon)$ to $O(1/\epsilon^2)$, consistent with standard agnostic learning.

## Highlights & Insights

- **Hierarchical framework design**: The three verification objectives progress naturally (in-distribution → global soundness + full completeness → global soundness + approximate completeness), each corresponding to distinct practical scenarios and theoretical difficulty levels.
- **Proof technique in Theorem 4.9**: A union bound over $2^{|H|}$ possible subsets combined with exponential decay in per-subset sample probability establishes a bridge from the intersection of consistent verifiers to sample complexity bounds.
- **The USAT computational gap example** is particularly elegant: a verifier need only check whether an assignment satisfies all clauses (polynomial time), whereas generating the correct assignment requires solving USAT (NP-hard), perfectly instantiating the intuition that verification is easier than generation.
- **Equivalence with CoT generation** (Remark 4.6) is a surprising finding—it is commonly assumed that verification should be simpler than generation, yet strong verifiers possess a "guiding" capability and can themselves generate proofs. This provides a theoretical foundation for "training a reasoner with a verifier."
- **Intersection-closure** as the structural condition that breaks the linear barrier connects the classical PAC learning closure algorithm with the new verification framework.

## Limitations & Future Work

- **Realizability assumption**: The main results assume a perfect verifier $h^* \in H$ exists; the agnostic extension, while provided, incurs an additional $O(1/\epsilon)$ factor in sample complexity.
- **Problem distribution assumption**: Trustable verifiers are distribution-free over reasoning chains but still depend on the problem distribution $D$—no guarantee is provided for problem types outside the training distribution.
- **Finite $\Sigma$ assumption**: TVPAC requires a finite reasoning-step space, limiting direct applicability to continuous or open-ended natural-language reasoning.
- **Computational efficiency**: Algorithm 1 (intersection of consistent verifiers) is conceptually clean but may be computationally infeasible—enumerating all consistent verifiers in $H_S$ and taking their intersection is impractical for large $|H|$.
- **Interactive verification**: The paper raises but does not resolve whether an interactive protocol (where the verifier may query the prover) could circumvent the linear lower bound.
- **Unbounded chain length**: When reasoning-chain length $T$ is unrestricted, the learnability of multi-class classification with an infinite label space remains an open problem.

## Related Work & Insights

- This work is closely related to the CoT generation theory of **JVB+ (2025)**: the proof techniques for SVPAC are directly adapted from their sample complexity analysis for generative models, but TVPAC verifiers provide stronger OOD guarantees, and the agnostic setting remains an open problem in their generative framework.
- The **closure algorithm** originates from the classical one-sided error learning literature (Natarajan 1987; Kivinen 1995) and is elegantly repurposed as the central tool for verifier learning.
- This work is complementary to **interactive prover** frameworks (GRSY 2021; AGPR 2024), which study how a prover can convince a fixed verifier; the present paper studies how to learn a good verifier from data.
- Empirical work such as **LightmanVerify-Step-by-Step (2024)** shows that stepwise verification outperforms holistic verification; the paper's distinction between stepwise and overall soundness provides a theoretical basis for this observation.
- The finding that "a verifier can train a reasoner" may inspire new training paradigms—first learn a strong verifier, then use it to guide the reasoning model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first systematic PAC learning framework for CoT verifiers; the three-tier verification objectives and the complexity phase transition from $O(\log|H|)$ to $\Theta(|H|)$ constitute wholly original theoretical contributions.
- Experimental Thoroughness: ⭐⭐ — A purely theoretical paper with no experiments, though rich concrete examples (USAT, graph paths, LLM verifier ensembles) compensate for the lack of empirical intuition.
- Writing Quality: ⭐⭐⭐⭐⭐ — Definitions are precise, theorem statements are rigorous, proofs are concise, and the progression from weak to strong objectives makes for a highly readable exposition.
- Value: ⭐⭐⭐⭐ — Provides fundamental theoretical guidance on data requirements for CoT verifiers; the intersection-closure condition and the verification–generation equivalence have potential practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)

</div>

<!-- RELATED:END -->
