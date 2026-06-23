---
title: >-
  [Paper Note] Softmax Transformers are Turing-Complete
description: >-
  [ICLR 2026][learning_theory][Chain-of-Thought] This paper provides the first proof that **softmax** attention Transformers with Chain-of-Thought (CoT) are Turing-complete. This construction inherits length generalization guarantees by simulating a Minsky counter machine through the intrinsic "counting" capability of softmax (via C-RASP), rather than hard-coding Tur
tags:
  - ICLR 2026
  - learning_theory
  - Chain-of-Thought
  - C-RASP
date: 2026-05-08
content_hash: 83327b633bf8a6d1
---
# Softmax Transformers are Turing-Complete

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FdkPOHlChS](https://openreview.net/forum?id=FdkPOHlChS)  
**Code**: TBD  
**Area**: Learning Theory / Transformer Expressivity / Formal Languages  
**Keywords**: Turing completeness, softmax attention, chain-of-thought, C-RASP, counter machine

## TL;DR
This paper provides the first proof that **softmax** attention Transformers with Chain-of-Thought (CoT) are Turing-complete. This construction inherits length generalization guarantees by simulating a Minsky counter machine through the intrinsic "counting" capability of softmax (via C-RASP), rather than hard-coding Turing machine tape heads. It utilizes a task-agnostic Relative Position Encoding (RPE) to encode arbitrary inputs into numerical values.

## Background & Motivation

**Background**: Characterizing what Transformers "can and cannot compute" using formal language theory is a recent research hotspot. A fundamental question is whether CoT-equipped Transformers are Turing-complete. Prior works by Pérez et al. (2021), Merrill & Sabharwal (2024), and Li & Wang (2025) have provided affirmative answers.

**Limitations of Prior Work**: All existing Turing-completeness proofs rely on **hardmax (hard attention)**. Hardmax is an unrealistic assumption—it is essentially a discrete operation that "picks a specific position," which lacks **trainability guarantees** as gradients cannot propagate effectively. Real Transformers use softmax. A long-standing open question is whether CoT Transformers remain Turing-complete when hardmax is replaced with softmax.

**Key Challenge**: Hardmax facilitates proofs because it can directly "locate" the position of a Turing machine (TM) tape head (e.g., using average hard attention in the form of $\langle x, y\rangle$, or layer normalization). However, softmax is a soft, normalized weighted average that **cannot precisely extract a single position**. Finding whether softmax can even simulate average hardmax is itself an open problem. Consequently, directly simulating TM tape heads with softmax is non-trivial or even impossible.

**Goal**: (i) Provide the first proof of Turing-completeness for softmax CoT Transformers; (ii) simultaneously provide length generalization guarantees, ensuring that a model trained on finite data can extrapolate correctly to arbitrary lengths.

**Key Insight**: The authors bypass the difficulty of "simulating TM tape heads" and instead simulate **Minsky counter machines**. Counter machines do not require tape head positioning; their states are defined by the values of several counters. Softmax Transformers are naturally adept at "counting" (e.g., counting how many times a symbol has appeared). Since counter machines are Turing-complete, simulating them is sufficient to prove Turing-completeness.

**Core Idea**: Use the counting capability of softmax (characterized via the declarative language **C-RASP**) to simulate a counter machine. While this holds naturally for unary or letter-bounded languages, the authors introduce a task-agnostic Relative Position Encoding (RPE) to encode arbitrary language inputs into numbers.

## Method

### Overall Architecture

The argument is not a "neural network pipeline" but a sequence of **expressivity equivalence and simulation chains**:

$$\text{softmax CoT Transformer (SMAT)} \;\supseteq\; \text{CoT C-RASP} \;\xrightarrow{\text{Simulate}}\; \text{Minsky Counter Machine} \;\equiv\; \text{Turing Machine}$$

Specifically, the authors adopt the formalization from Huang et al. (2025) for length-generalizable softmax Transformers (called SMAT): the attention weights are defined as $\bar w = \mathrm{softmax}(\log n \cdot \{v_j^\top K^\top Q v_i\}_{j=1}^{i})$, where the $\log n$ scaling is necessary to express sparse functions, and the FFN utilizes a single-layer ReLU/Heaviside network. CoT is defined on top: the model autoregressively generates CoT tokens until a "halt/accept" symbol is produced; the language $L(F)$ consists of all inputs that are eventually accepted.

C-RASP is a declarative language containing only "past" operators (equivalent to a fragment of $K_{\text{tr}}[\#]$ / `LTL[Count]`). Its core is the **count term** $\overleftarrow{\#}[\varphi]$, representing "the number of positions up to the current index that satisfy predicate $\varphi$." A key bridge (Prop 2.1) ensures: any language accepted by CoT C-RASP is also accepted by CoT SMAT. Thus, a construction at the clean symbolic level of C-RASP automatically applies to the softmax Transformer.

The argument progresses in three steps: **(1) Unary/letter-bounded/permutation-invariant languages**: directly simulate counter machines using C-RASP (Section 3); **(2) Impossibility for arbitrary languages**: pure C-RASP cannot even recognize palindromes (Section 4 counterexample); **(3) Arbitrary languages with RPE**: Adding an RPE allows C-RASP to encode input words as numbers for the counter machine (Section 4).

### Key Designs

**1. Simulating Counter Machines via C-RASP instead of Simulating Turing Machines via Transformer**

This is the core move to bypass the softmax dilemma. TMs are difficult because tape heads must be "precisely located," which softmax struggles with. In contrast, a counter machine's state consists of a finite control state $q$ and several counter values $x \in \mathbb{Z}^k$. Transitions only depend on tests such as "is a counter zero or greater than zero." Softmax Transformers, via C-RASP, are naturally capable of "counting."

The authors use a CoT token set $\Gamma = \Sigma \cup \Delta$ (input alphabet plus transition relations $\Delta$) to record the computational trajectory. Emitting a transition $\tau$ corresponds to a step in the counter machine, and **the current value of each counter can be reconstructed purely through counting terms**. For permutation-invariant languages, the value of the $i$-th counter is given by:

$$t_i = \overleftarrow{\#}[Q_{a_i}] + \sum_{\rho \in \Delta} u_\rho(i)\cdot \overleftarrow{\#}[Q_\rho], \quad i = 1,\dots,n$$

Intuitively: $\overleftarrow{\#}[Q_{a_i}]$ counts the initial number of alphabet $a_i$ in the input (the initial counter value, corresponding to the Parikh vector $\Psi(w)$), and $\overleftarrow{\#}[Q_\rho]$ counts how many times transition $\rho$ has been traversed. Multiplying by the increment $u_\rho(i)$ restores the current counter value. Rules take the form $O_{\tau'} \leftarrow \varphi_{\tau'}(t_1,\dots,t_{n+3}) \wedge Q_\tau$: if the current last symbol is transition $\tau$ and the counter test $\varphi_{\tau'}$ holds, output the next transition $\tau'$. Since the machine is deterministic, rule order does not matter. Using Lemma 3.5 (where $n+3$ counters are Turing-complete), this yields completeness for unary languages. The paper illustrates this with PARITY (Example 1): using a 2-counter machine, initial rules check if $\overleftarrow{\#}[Q_a]>0$, and subsequent steps decrement the $a$ count by 2 until it reaches 0—a language **inexpressible by C-RASP without CoT**.

**2. The Impossibility for Arbitrary Languages: Communication Complexity Ceiling of C-RASP**

The authors honestly state that the above construction does **not** hold for arbitrary languages. Prop 4.1 proves CoT C-RASP is not Turing-complete on $\Sigma=\{a,b\}$, with PALINDROME as a counterexample. The root cause (Lemma 4.2) is: if a language is recognized by CoT C-RASP, its restriction $L_n$ to length $\le n$ is recognizable by an automaton of size **polynomial in $n$**. This follows from the **logarithmic communication complexity** upper bound of Limit Transformers / C-RASP (Huang et al. 2025, Thm 12), which holds even with learned Absolute Position Encodings (APE). Palindromes require bit-by-bit alignment of the first and second halves; such "precise pairing of arbitrary positions" exceeds pure counting capability. This negative result necessitates "positional information" to handle arbitrary languages.

**3. Task-Agnostic RPE for Encoding Inputs into Numbers**

To break the ceiling, the authors augment C-RASP with **Relative Position Encoding (RPE)**, which is a relation $R \subseteq \mathbb{N}\times\mathbb{N}$. They introduce relative counting terms $\overleftarrow{\#}_R[\varphi]$ that count $\{i\in[1,j]: (i,j)\in R,\ i\models\varphi\}$ at position $j$. In SMAT, this adds a bias to the attention logit: $\bar w = \mathrm{softmax}(\log n\cdot\{v_j^\top K^\top Q v_i + \lambda[\![R]\!](i,j)\}_j)$. Surprisingly, the authors prove that **a single fixed RPE, independent of the Turing machine being simulated, is sufficient**.

This RPE is based on a partial function $\beta:\mathbb{N}\rightharpoonup\{0,1\}^*$: represent $x$ in binary and take all bits after the "first zero from the MSB" as its string representation. The RPE is defined as $(i,j)\in R \iff i\le j,\ i\in[1,|\beta(j)|],$ and the $i$-th bit of $\beta(j)$ is 1. With this, the construction follows two phases:

- **Phase I (Encoding)**: Encode an arbitrary word $w\in\Sigma^*$ into a vector $x\in\mathbb{N}^n$ such that $\sigma(x)=w$. This is done by appending dummy symbols $l_i$ to adjust length $\ell$ until $\beta(\ell)$ equals the target string $w_i$, at which point a marker $\text'_i$ is placed. The rule for $\beta(\ell)=w_i$ cleverly uses the equality $\overleftarrow{\#}_R[Q_{a_i}] = \overleftarrow{\#}[Q_{a_i}]$ and $\overleftarrow{\#}_R[\top]=\overleftarrow{\#}[Q_{a_i}]$ to express "the positions selected by $R$ are exactly those carrying $a_i$." The final encoding is extracted via $X_i = \overleftarrow{\#}[\overleftarrow{\#}[\text'_i]=0]$ (the position of $\text'_i$ minus one).
- **Phase II (Simulation)**: Given encoding $x$, apply the counter machine for set $S=\{x\in\mathbb{N}^n:\sigma(x)\in L\}$ (recursively enumerable as $\sigma$ is computable) via Lemma 3.5. This yields Turing-completeness for arbitrary languages (Thm 4.3). Example 2 ("words ending in b") uses this to check if $x(1)$ is even, reusing the PARITY counter machine.

### Loss & Training
The theoretical part has no loss function. For experiments, a decoder-only LLaMA architecture is trained from scratch using AdamW (weight decay 0.01), batch size 64, up to 30k steps, with EarlyStopping monitoring validation loss (stops at 100% in-distribution accuracy for 3 epochs).

## Key Experimental Results

The experiments aim to **validate theoretical predictions**: unary representations require no position encoding (NoPE) for length generalization, while binary representations require RPE. Models are trained on lengths $[1,100]$ and evaluated on in-distribution (test0, $[1,100]$) and two OOD intervals (test1 $[101,200]$, test2 $[201,300]$). Tasks involve complex number theory: Prime, Exponential, Division, GCD, and Multiplication.

### Main Results

| Task | Unary test0 | Unary test2 | Binary+RPE test0 | Binary+RPE test2 | Binary NoPE test0 | Binary NoPE test2 |
|------|------|------|------|------|------|------|
| Prime | 100 | 100 | 100 | 100 | 95.00 | 0.00 |
| Exponential | 99.95 | 99.96 | 100 | 100 | 82.80 | 0.00 |
| Division | 99.90 | 99.99 | 100 | 100 | 76.40 | 0.00 |
| GCD | 99.99 | 99.70 | 100 | 100 | 70.20 | 0.00 |
| Multiplication | 99.99 | 99.98 | 100 | 100 | 64.40 | 0.00 |

### Ablation Study (Role of RPE)

| Configuration | Representative Performance (test0 → test2) | Description |
|------|------|------|
| Unary + NoPE | ~99.9% → ~99.9% | Natural generalization in unary mode, consistent with theory. |
| Binary + RPE | 100% → 100% | Near-perfect extrapolation, validating RPE enables arbitrary input completeness. |
| Binary No RPE | 64–95% → ~0% | OOD performance collapses, proving RPE is necessary for length generalization. |

### Key Findings
- **RPE is the Switch for Length Generalization**: removing RPE in binary representations drops OOD accuracy to ~0%, while adding it maintains 100%, directly reflecting the theory that arbitrary languages require RPE.
- **Unary Requires No Position Encoding**: Unary tasks generalize stably even with NoPE up to 3x training length, verifying that permutation-invariant languages only require counting.
- **Task Difficulty does not Alter Conclusions**: conclusions remain consistent from Primes to Multiplication, indicating structural expressivity factors rather than task-specific artifacts.

## Highlights & Insights
- **Changing the Proxy for Simulation**: Instead of fighting the difficulty of simulating tape heads with softmax, the authors simulate counter machines. Softmax's counting ability fits this perfectly. This "match your compute engine to its strength" strategy is a valuable perspective for expressivity analysis.
- **A Single Task-Agnostic RPE Suffices**: Traditional assumptions often require specific encodings for specific languages. This paper shows a single fixed structure can compress positional information into a reusable format for all Turing machines.
- **Simultaneous Completeness and Generalization**: Previous proofs used hardmax, sacrificing trainability. This work uses the learnable framework of Huang et al. (2025), unifying "strong expressivity" with "learnability and extrapolation."

## Limitations & Future Work
- **Theory-Practice Precision Gap**: The construction assumes $\log n$ scaling and Heaviside activations. While these are approximated in finite-precision training (e.g., via ReLU), the exact reproduction remains a challenge.
- **Lack of Complexity Linkage**: Unlike Merrill & Sabharwal (2024), this paper does not explicitly link CoT steps to complexity classes like PSPACE. Refining these bounds is future work.
- **Limited Experimental Scale**: Validated on number theory tasks and lengths $\le 300$ with small LLaMA models. Robustness across larger models and sequences is not fully explored.
- **Specialized RPE Form**: The RPE is based on a specific binary encoding $\beta$. Its relationship with mainstream learned RPEs (RoPE, ALiBi) requires further clarification.

## Related Work & Insights
- **vs. Pérez et al. (2021) / Merrill & Sabharwal (2024)**: They used hardmax to simulate TM heads, proving completeness without trainability. This paper uses softmax and counter machines to achieve completeness within a realistic, generalizable framework.
- **vs. Huang et al. (2025)**: This work builds on their Limit Transformer / C-RASP framework but extends it with CoT and RPE, both of which are required for Turing-completeness.
- **vs. Hou et al. (2025)**: They provided RASP-level proof for length-generalized completeness, but it relied on an unrealistic "no repeating n-grams" condition. This paper provides a true softmax construction with full theoretical guarantees.
- **vs. Sälzer et al. (2026)**: They proved softmax Transformers are "almost" Turing-complete (via projection). This paper provides a direct, complete proof for CoT SMAT.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First Turing-completeness proof for softmax attention. Transitioning from TM simulation to counter machine simulation is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments align well with theory, particularly the RPE ablation, though scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Concepts are well-structured, but the formalization of C-RASP and counter machines is dense for non-theorists.
- Value: ⭐⭐⭐⭐⭐ Resolves a long-standing open question, unifying Turing-completeness with learnability and length generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Transformers Are Inherently Succinct](transformers_are_inherently_succinct.md)
- [\[ICLR 2026\] Probability Distributions Computed by Autoregressive Transformers](probability_distributions_computed_by_autoregressive_transformers.md)
- [\[ICLR 2026\] The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens](the_softmax_bottleneck_does_not_limit_the_probabilities_of_the_most_likely_token.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)

</div>

<!-- RELATED:END -->
