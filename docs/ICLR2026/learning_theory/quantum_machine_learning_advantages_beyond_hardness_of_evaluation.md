---
title: >-
  [Paper Note] Quantum Machine Learning Advantages Beyond Hardness of Evaluation
description: >-
  [ICLR 2026][learning_theory][BQP] This paper provides the first proof that for data labeled by quantum functions ($\mathsf{BQP}$-complete), classical algorithms cannot even "identify" the labeling function itself, even without the requirement to "evaluate" new samples, unless $\mathsf{BQP}$ falls into the lower levels of the polynomial hierarchy (a col
tags:
  - ICLR 2026
  - learning_theory
  - BQP
date: 2026-05-08
content_hash: a704fa0d816dea81
---
# Quantum Machine Learning Advantages Beyond Hardness of Evaluation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=on2lie43Kl](https://openreview.net/forum?id=on2lie43Kl)  
**Code**: None (Theoretical paper)  
**Area**: Learning Theory / Quantum Machine Learning / Computational Complexity  
**Keywords**: Quantum Machine Learning, PAC Learning, Learning Separations, BQP, Polynomial Hierarchy

## TL;DR
This paper provides the first proof that for data labeled by quantum functions ($\mathsf{BQP}$-complete), classical algorithms cannot even "identify" the labeling function itself, even without the requirement to "evaluate" new samples, unless $\mathsf{BQP}$ falls into the lower levels of the polynomial hierarchy (a collapse widely believed to be false). This shifts the quantum machine learning advantage from "hardness of evaluation" to "hardness of the learning process itself."

## Background & Motivation

**Background**: In recent years, several rigorous proofs of "quantum advantage" in Quantum Machine Learning (QML) have emerged. The typical paradigm involves data labeled by a cryptographic or quantum function $f$ that classical polynomial circuits cannot efficiently compute. Consequently, even if a classical learner "learns" the model, it cannot label new inputs, whereas a quantum learner can. The source of advantage in such results is **hardness of evaluation**.

**Limitations of Prior Work**: Hardness of evaluation is not intrinsically related to "learning"—it merely indicates that classical circuits cannot compute $f$, without addressing the difficulty of "recognizing $f$ from data." In other words, existing quantum advantages might stem entirely from the final "inference" step of evaluating a new point rather than the learning process. Ideally, one would determine if **learning itself** (recovering the labeling function from labeled samples) possesses a quantum advantage.

**Key Challenge**: To decouple "learning hardness" from "evaluation hardness," a natural approach is to examine the **identification task**. Given a set of $(x, f(x))$, the learner is only required to output a description of the labeling function $f$ (i.e., its index $\alpha$), without being required to evaluate $f$ on new inputs $x$. This excludes evaluation hardness, leaving only the difficulty of learning. However, a fundamental obstacle exists: existing proofs of identification hardness (for cryptographic functions or DNF formulas) rely on **random generatability**—the ability to efficiently generate labeled samples $(x, f(x))$ for random inputs $x$. For quantum functions, the authors conjecture (and prove in this paper) that this property **does not hold**, rendering previous proof frameworks inapplicable and leaving the identification advantage of quantum functions unexplored.

**Goal**: To provide the **first** classical hardness proof for the identification of quantum functions under reasonable complexity assumptions and construct a genuine quantum-classical learning separation based on this.

**Key Insight**: Since quantum functions are not randomly generatable and traditional proofs fail, a new proof path is required. Instead of "generating samples," the existence of a classical identification algorithm is used via an NP oracle to construct a classical machine capable of deciding BQP languages. This forces $\mathsf{BQP}$ into the polynomial hierarchy, leading to a contradiction.

**Core Idea**: By using "**inverting the identification algorithm + climbing the polynomial hierarchy**" instead of "randomly generating samples," it is proved that for a broad class of quantum concept classes ($c$-distinct or average-case-smooth), classical identification is hard unless $\mathsf{BQP} \subseteq \mathsf{BPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$.

## Method

### Overall Architecture

This paper is a purely complexity-theoretic work. The "method" consists of a chain of reductions aimed at transforming the assumption that "classical algorithms can identify quantum functions" into the widely rejected conclusion that "$\mathsf{BQP}$ collapses into the polynomial hierarchy." The chain progresses through three stages:

**Stage 1 (Preliminaries, ruling out old paths)**: It is first proven that quantum functions are **not randomly generatable**. If a classical algorithm could efficiently produce $(x, f(x))$ for a uniform random $x$, then $\mathsf{BQP} \subseteq \mathsf{P}^{\textsf{NP}}$ (exact version) or $(\mathsf{BQP}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}}$ (approximate version). This step demonstrates that the core tools of existing identification hardness proofs fail for quantum functions.

**Stage 2 (Verifiable identification, strong assumption version)**: The "verifiable identification" task is examined, where the algorithm must identify $\alpha$ and reject invalid datasets inconsistent with any concept. Under this stronger assumption, it is proven that classical verifiable identification of quantum functions implies $(\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}}$. The ability to "reject invalid data" is the key property that enables the inversion reduction.

**Stage 3 (Non-verifiable identification, main result)**: The strong assumption of "verifiability" is removed, requiring only a proper PAC learner (approximate-correct identification) satisfying two mild additional conditions. The cost is that the reduction moves two levels higher in the polynomial hierarchy: classical identification $\Rightarrow (\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$. Finally, a physically motivated $c$-distinct concept class (related to learning observables/Hamiltonian learning) is provided that satisfies the conditions of the main theorem, establishing a real-world quantum-classical identification separation.

The input is a training set $T = \{(x_\ell, y_\ell)\}$ labeled by some quantum function $f^\alpha$, and the output is a theorem stating that if classical identification is possible, $\mathsf{BQP}$ collapses.

### Key Designs

**1. Identification Task: Decoupling "Learning" from "Evaluation"**

To study if the "learning process itself has a quantum advantage," a learning task without evaluation is designed: the identification task. The concept class is defined as $F = \{f^\alpha : \{0,1\}^n \to \{0,1\} \mid \alpha \in \{0,1\}^m\}$ where $m = \mathrm{poly}(n)$. Upon receiving $T = \{(x_\ell, f^\alpha(x_\ell))\}$, the learner only needs to output the index $\alpha$ (or an approximate $\tilde\alpha$ such that $\mathbb{E}_{x \sim D}|f^\alpha(x) - f^{\tilde\alpha}(x)| \le \epsilon$). It is **not required** to evaluate new $x$. This corresponds to proper PAC learning where the hypothesis class $H$ is identical to the concept class $F$.

**2. Quantum Functions are Not Randomly Generatable**

Previous proofs of identification hardness (for cryptographic functions) rely on concise representation and random generatability. Random generatability implies the existence of an efficient classical algorithm $A_D(f_n, r) = (x_r, f_n(x_r))$ that produces labeled samples for random inputs. This paper proves this does not hold for quantum functions:

$$\text{If } f \in \mathsf{BQP} \text{ and is exactly randomly generatable, then } \mathsf{BQP} \subseteq \mathsf{P}^{\mathsf{NP}}.$$

The proof uses an NP oracle to **invert** $A$. Given an input $\tilde x$, the NP oracle finds a random string $\tilde r$ such that $A(f_n, \tilde r) = (x_{\tilde r}, f_n(x_{\tilde r}))$ and $x_{\tilde r} = \tilde x$. Since $A$ is classical poly-time, the NP oracle can efficiently verify $\tilde r$, allowing $f_n(x)$ to be computed in $\mathsf{P}^{\textsf{NP}}$.

**3. Inversion + Climbing the Polynomial Hierarchy: A New Proof Engine**

The core technique uses the "assumed identification algorithm $A_B$" as an oracle that can be inverted by an NP oracle. In the verifiable version (Thm 3), $A_B$ outputs $\tilde\alpha \approx \alpha$ when $T$ is consistent and "invalid" otherwise. This allows an NP oracle to **search** for a dataset $T$ such that $A_B(T, \epsilon, \cdot) = \alpha$. Once $T$ is obtained, any new $x$ can be evaluated by checking whether $y=0$ or $y=1$ keeps $T \cup \{(x, y)\}$ consistent, thus $(\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}}$.

The main result (Thm 5) removes the verifiability assumption. For $c$-distinct or average-case-smooth concept classes, an approximate-correct identification algorithm can be transformed into one that rejects datasets with a large fraction of mislabeled samples within the first level of PH. By climbing two more levels, one can invert this to find a dataset where most samples are correctly labeled by $f^\alpha$:

$$\text{Classical approximate-correct identification} \implies (\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}.$$

**4. Concept Class Structural Conditions ($c$-distinct / average-case-smooth)**

These conditions ensure that the inverted dataset uniquely identifies the target concept.
- **$c$-distinct (Def. 5)**: Any two distinct concepts $f^{\alpha_1} \ne f^{\alpha_2}$ differ on at least a $c$ fraction of inputs. The main theorem requires $c \ge 1/3$.
- **Average-case-smooth (Def. 6)**: Proximity in function space implies proximity in parameter space, $\mathbb{E}_{x \sim \mathrm{Unif}}|f^{\alpha_1}(x) - f^{\alpha_2}(x)| \ge C\, d(\alpha_1, \alpha_2)$.

Hamiltonian learning is known to be **classically solvable**. This implies that if these structural assumptions were removed while maintaining identification hardness, it would conflict with the classical decidability of Hamiltonian learning.

## Key Experimental Results

### Main Results Comparison

| Result | Task / Assumption | Classical Feasibility Implies | Source |
| :--- | :--- | :--- | :--- |
| Thm 1 | Exact random generatability of quantum functions | $\mathsf{BQP} \subseteq \mathsf{P}^{\mathsf{NP}}$ | §3 |
| Thm 2 | Approximate random generatability | $(\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}}$ | §3 |
| Thm 3 | Verifiable identification | $(\mathsf{L}, \mathrm{Unif}) \in \textsf{HeurBPP}^{\mathsf{NP}}$ | §4 |
| Thm 5 | Non-verifiable identification ($c \ge 1/3$ or avg-smooth) | $(\mathsf{L}, \mathrm{Unif}) \in \mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$ | §5 |

### Key Findings
- **Relocation of Advantage**: While previous quantum advantages might stem from "evaluation hardness," this work shows that "identification/learning" itself is hard for quantum functions.
- **Necessity of New Proof Engine**: Thm 1/2 prove that quantum functions are not randomly generatable, requiring the "Inversion + PH climbing" strategy.
- **Assumption vs. Hierarchy Trade-off**: Removing the "verifiable" assumption increases the PH level from one ($\mathsf{NP}$) to three ($\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}$).
- **Structural Boundaries**: Hamiltonian learning's classical solvability highlights that $c$-distinctness/smoothness are necessary boundaries for the separation.

## Highlights & Insights
- **Inverting the "Assumed Algorithm"**: The cleverest aspect is not constructing a sample generator but using the identification algorithm $A_B$ as an oracle for reverse searching via an NP oracle.
- **Verifiable to Non-verifiable Bridge**: Transforming an approximate-correct algorithm into an "approximately verifiable" one within the first level of PH is a robust template for handling inconsistent algorithms.
- **Negative Lemma as a Foundation**: The proof that quantum functions are not randomly generatable justifies the need for the new proof strategy.
- **Physical Task Linkage**: The theoretical identification hardness is applied to real quantum tasks like Hamiltonian learning and learning observables.

## Limitations & Future Work
- **Dependency on Complexity Assumptions**: Core conclusions rely on $\mathsf{BQP}$ not being in the low levels of PH (e.g., $\mathsf{BQP} \not\subseteq \mathsf{BPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$).
- **Structural Constraints**: Separations only hold for $c$-distinct ($c \ge 1/3$) or average-case-smooth classes. Relaxing these might conflict with known classical algorithms for Hamiltonian learning.
- **Lack of Empirical Implementation**: The work is purely existence and impossibility proofs without benchmarking against dequantized algorithms.
- **Future Directions**: Reducing the required PH levels for non-verifiable identification and finding more natural concept classes without $c$-distinctness.

## Related Work & Insights
- **vs. Huang et al. (2021)**: They suggest data helps classical learners evaluate functions; this work studies scenarios where quantum functions remain hard to identify despite data.
- **vs. Gyurik & Dunjko (2023)**: They established learning separations in evaluation; this work provides the first **identification-level** separation.
- **vs. Cryptographic Proofs**: Traditional proofs rely on random generatability, which this paper proves is absent in quantum functions.
- **vs. Hamiltonian Learning**: Hamiltonian learning's classical solvability provides the boundary for the structural assumptions (c-distinct/smoothness) used in this paper.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof of classical hardness for quantum identification using a new PH-climbing strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical; proofs are complete and rigorous.
- Writing Quality: ⭐⭐⭐⭐ Clear progression of reductions, though dense in complexity notation.
- Value: ⭐⭐⭐⭐⭐ Shifts the perspective of QML advantage from evaluation to the learning process itself.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Learning to Adapt: In-Context Learning Beyond Stationarity](learning_to_adapt_in-context_learning_beyond_stationarity.md)
- [\[ICLR 2026\] The Lie of the Average: How Class Incremental Learning Evaluation Deceives You?](the_lie_of_the_average_how_class_incremental_learning_evaluation_deceives_you.md)
- [\[ICLR 2026\] Subquadratic Algorithms and Hardness for Attention with Any Temperature](subquadratic_algorithms_and_hardness_for_attention_with_any_temperature.md)
- [\[ICLR 2026\] Parameterized Hardness of Zonotope Containment and Neural Network Verification](parameterized_hardness_of_zonotope_containment_and_neural_network_verification.md)

</div>

<!-- RELATED:END -->
