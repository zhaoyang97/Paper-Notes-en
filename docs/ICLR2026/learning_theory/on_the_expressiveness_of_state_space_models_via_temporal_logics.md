---
title: >-
  [Paper Note] On the Expressiveness of State Space Models via Temporal Logics
description: >-
  [ICLR 2026][Learning Theory][Linear Temporal Logic] This paper characterizes a hierarchy of expressivity lower bounds for State Space Models (SSMs) with different gating mechanisms (diagonal S6, time-invariant S4, mixed gating) and arithmetic precisions (fixed-width vs. logarithmic) using Pure Past Linear Temporal Logic over finite traces (PLTLf) and its counting/modular extensions. It proves several hard inexpressibility results (e.g., fixed-width diagonal SSMs cannot recogn…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Expressivity"
  - "State Space Models"
  - "Linear Temporal Logic"
  - "Expressivity Lower Bounds"
  - "Arithmetic Precision"
  - "Inexpressibility"
date: 2026-05-08
content_hash: c10afa39c1beeb5a
---

# On the Expressiveness of State Space Models via Temporal Logics

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Vg511oJScS](https://openreview.net/forum?id=Vg511oJScS)  
**Code**: None  
**Area**: Learning Theory / Expressivity / State Space Models  
**Keywords**: State Space Models, Linear Temporal Logic, Expressivity Lower Bounds, Arithmetic Precision, Inexpressibility

## TL;DR
This paper characterizes a hierarchy of expressivity lower bounds for State Space Models (SSMs) with different gating mechanisms (diagonal S6, time-invariant S4, mixed gating) and arithmetic precisions (fixed-width vs. logarithmic) using Pure Past Linear Temporal Logic over finite traces (PLTLf) and its counting/modular extensions. It proves several hard inexpressibility results (e.g., fixed-width diagonal SSMs cannot recognize $(aa)^*$) and aligns these results with the known logic characterizations of Transformers.

## Background & Motivation
**Background**: SSMs (e.g., S4, Mamba/S6, RetNet, Griffin) are rapidly emerging as linear-time alternative architectures to Transformers. Parallel to this, the theory community has been using "logic / circuit complexity" to characterize which languages neural sequence models can recognize in principle. Such analyses are independent of training data and optimization, answering whether an architecture can inherently represent a certain pattern. While Transformers have mature results (UHAT corresponds to First-Order logic FO[<], AHAT corresponds to temporal logic with counting, etc.; see survey by Strobl et al.), the logic characterization for SSMs remains largely blank.

**Limitations of Prior Work**: Existing theoretical results for SSMs are sparse and only provide coarse upper and lower bounds. Merrill et al. (2024) provided upper bounds using circuit complexity: diagonal-gated, time-invariant, and fixed-width/log-precision SSMs all fall within $\mathrm{TC}^0$. Sarrof et al. (2024) provided a lower bound for diagonal SSMs, showing they can recognize all star-free languages and proving this bound is tight under specific output functions. However, these works: (1) did not characterize time-invariant and mixed SSMs; (2) did not systematically analyze the impact of arithmetic precision (fixed-width vs. logarithmic) on expressivity; (3) did not directly compare SSMs within the logic characterization landscape of Transformers.

**Key Challenge**: What exactly determines the expressivity of an SSM? The authors identify two key dimensions: **gating mechanism type** (whether the gate matrix depends on the input and whether it is restricted to be diagonal) and **arithmetic precision** (whether each step uses fixed $b$ bits or grows with sequence length $n$ to $O(\log n)$ bits). How these two dimensions jointly determine the class of languages an SSM can recognize had no unified answer.

**Goal**: To precisely map various SSM variants to different fragments of PLTLf, thereby: (1) providing a finer hierarchy of lower bounds than prior work; (2) clarifying the impact of precision on counting abilities; (3) achieving a one-to-one alignment with Transformer logic characterizations.

**Key Insight**: The authors employ the same logical tools used to characterize Transformers—Pure Past Linear Temporal Logic over finite traces (PLTLf). The choice of "pure past" over full LTLf is because the recursive structure of SSMs can only see the prefix of the input at each step, naturally supporting only "backward-looking" operators. This makes the comparison between SSMs and Transformers "comparable on the same scale."

**Core Idea**: Each PLTLf formula is decomposed bottom-up according to its syntax tree. Each sub-formula is mapped to one dimension of the SSM's hidden state, with each SSM layer computing one level of sub-formulas. This transforms the "logic formula $\to$ SSM construction" into a constructive proof. Lower bounds are then derived by gating and precision categories, accompanied by inexpressibility results proven via a monotonicity lemma.

## Method

### Overall Architecture
This is a purely theoretical paper without experiments; the "method" consists of a formalized expressivity analysis. The work can be viewed as a constructive machine that "translates temporal logic formulas into SSMs" plus a set of associated impossibility proofs. First, SSMs are formally unified (Layer = tuple $(h_0, \text{gate}, \text{inc}, \phi)$, recurrence $h_t = \text{gate}(x_t)\cdot h_{t-1} + \text{inc}(x_t)$, output $z_t = \phi(h_t, x_t)$), distinguishing four gating classes (diagonal, time-invariant, mixed, arbitrary) and two precision classes (fixed-width, logarithmic). Next, PLTLf and its extensions (counting operator $\overleftarrow{\#}$, modular predicate MOD) serve as "benchmarks" to provide lower bounds for the language classes recognized by each (gating, precision) combination. Finally, a monotonicity lemma proves hard upper bounds for fixed-width diagonal SSMs (unable to recognize $(aa)^*$), and all conclusions are embedded into the Transformer UHAT/AHAT landscape.

The core proof skeleton is "layer-by-layer construction per sub-formula": to evaluate a PLTLf formula $\varphi$ at position $i$, sub-formulas of $\varphi$ are sorted by nesting depth $\mathrm{nd}(\varphi)$ (independent sub-formulas can be processed in parallel in the same layer). Each sub-formula $\psi$ uses a one-dimensional hidden state in an SSM layer to track whether "$\sigma, i \models \psi$" holds. The number of layers required for the full formula equals the syntax tree depth. Whether a gating/precision combination can implement a specific temporal operator determines which PLTLf fragment it corresponds to.

### Key Designs

**1. Unified SSM Formalization and Four Gating Categories: Converging architectural differences into the "gate function" variable**

To perform precise expressivity characterization, the first step is abstracting diverse SSM architectures into analyzable objects. The authors follow the generalized setting of Merrill et al. (2024): an SSM layer is a tuple $(h_0, \text{gate}, \text{inc}, \phi)$, transforming an input sequence $x_1\cdots x_k$ via linear recurrence $h_t = \text{gate}(x_t)\cdot h_{t-1} + \text{inc}(x_t)$ into hidden states, then outputting via $\phi$ (with residuals, implemented by ReLU FNNs). An $L$-layer SSM with embedding emb and output out computes a function $\Sigma^* \to \mathbb{R}$, accepting if $S(\sigma)=1$. Architectural differences are categorized by the gate function: **Time-invariant** SSMs require a constant matrix $A$ such that $\text{gate}(x)=A$ (corresponding to S4); **Diagonal-gated** SSMs allow $\text{gate}(x)$ to be a non-negative diagonal matrix that can depend on $x$ (corresponding to S6/Mamba); **Mixed** SSMs allow layers to be either diagonal or time-invariant (unused in practice but theoretically bounded by the first two); **Arbitrary** gating places no restrictions. This classification allows each theorem to be anchored to a specific "gating type + precision type."

**2. Three Levels of Expressivity Benchmarks: PLTLf $\to$ Counting Extension $\overleftarrow{\#}$ $\to$ Modular Predicate MOD**

The authors split "sequence capability" into three progressive conceptual layers, each paired with a logic fragment. Use **pattern matching** as the base—Pure Past LTLf (PLTLf, with operators yesterday $Y$, previously $P$, since $S$), which characterizes the ability to detect events in relative order without counting, equivalent to FO[<] / star-free regular languages. The middle layer adds the **backward counting operator** $\overleftarrow{\#}$: $\overleftarrow{\#}\,\varphi$ counts the number of positions up to current position $i$ that satisfy $\varphi$, with new counting sub-formulas of the form $\sum_i a_i \overleftarrow{\#}\,\varphi_i \sim c$ ($\sim\in\{<,\le,=,\ge,>\}$). Only $\overleftarrow{\#}$ is used (not forward $\overrightarrow{\#}$) because SSMs can only see prefixes. The top layer adds the **modular predicate** MOD: $\sigma, i \models \mathrm{MOD}^m_r \iff i \equiv r \pmod m$, enabling the model to distinguish "periodic positions" (e.g., parity), elevating expressivity to all regular languages in AC$^0$ ($\equiv$ FO[<, MOD]). This corresponds to the effect of positional encodings in Transformers. Once these benchmarks are set, the mapping of each SSM variant becomes clear.

**3. Constructive Lower Bounds for Diagonal SSMs: Why the since operator "must" depend on input-diagonal gates**

Theorem 1 proves that fixed-width diagonal SSMs can recognize all PLTLf-definable languages (star-free languages) constructively. The proof idea: each sub-formula occupies one dimension of the hidden state; the inc function along with position-wise FNNs handles non-temporal sub-formulas (conjunction, negation, comparison), and after each layer, the hidden state is a boolean vector marking which sub-formulas hold. The key insight is that different temporal operators have different gating requirements: $Y$ (yesterday) only needs to reference the previous position, and $P$ (previously) only needs to accumulate. Both can be implemented with diagonal, time-invariant gates without conjunction with the current input. However, the **since operator $S$ is inherently recursive**—$\varphi\,S\,\psi \equiv \psi \lor (\varphi \land Y(\varphi\,S\,\psi))$. Evaluation requires a conjunction of the previous state with "the current input's evaluation of $\varphi$," and this "dependence on current input" specifically necessitates the **diagonal gate must depend on input** $x$. In other words, diagonal SSMs are stronger than time-invariant SSMs because their gates can vary with the current symbol. Theorem 4 further shows that with log-precision, adding a layer to accumulate $\varphi_j$ occurrences via inc and checking linear combinations via FNNs allows diagonal SSMs to recognize PLTLf[$\overleftarrow{\#}$], thus recognizing non-regular and even non-context-free languages like $\{a^n b^n c^n\}$, which can be written as:
$$\varphi = H\big((a \to \neg P(b\lor c)) \land (b \to \neg P\,c)\big) \land (\overleftarrow{\#}a - \overleftarrow{\#}b = 0) \land (\overleftarrow{\#}c - \overleftarrow{\#}b = 0).$$

**4. Monotonicity Lemma and Time-Invariant SSM Cycle-Permutation Gates: Two Incomparable Capability Boundaries**

Beyond lower bounds, the authors provide a hard upper bound for diagonal SSMs via a monotonicity lemma. Lemma 2: When a fixed-width diagonal SSM repeatedly reads the same symbol $\sigma$, the output must eventually stabilize—because diagonal gates mean each dimension evolves independently as $(h_t)_i = \text{gate}(\text{emb}(\sigma))_i\cdot(h_{t-1})_i + \text{inc}(\text{emb}(\sigma))_i$. Non-negative gates make the sequence monotonic, and under fixed-width precision, each dimension can only take finite values, so there exists $N$ such that output is constant for $n \ge N$. Theorem 3 follows immediately: fixed-width diagonal SSMs **cannot recognize** $(aa)^*$ because they would eventually accept or reject all sufficiently long $a^n$, whereas $(aa)^*$ requires distinguishing odd/even lengths. This is a hard architectural bottleneck independent of data/optimizers. Conversely, time-invariant SSMs cannot implement since (Conjecture 1 suggests they cannot even recognize $L(a\,S\,b)$), yet they can do what diagonal SSMs cannot: Lemma 6 uses an $m$-th order **cyclic permutation matrix** $P$ as a gate ($B=0$), letting $h_t = P^t h_0$ cycle through $m$ basis vectors to maintain a modulo $m$ counter in the hidden state. A single layer can compute any MOD predicate and thus recognize $L(H\,a \land \mathrm{MOD}^2_0)=(aa)^*$. This reveals that diagonal and time-invariant SSMs are **incomparable in expressivity**: the former does since but not MOD, the latter vice versa. By mixing layers (Mixed), Corollary 8 achieves both, recognizing PLTLf[MOD], i.e., all regular languages in AC$^0$. Arbitrary gating recognizes all regular languages per Merrill et al.

### An Example: Recognizing Odd Positions with MOD Predicates
Take $\mathrm{MOD}^2_1$ (checking if a position is odd) as an example of the Lemma 6 construction: $m=2$, permutation matrix $P=\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$, initial $h_0=(1,0)^T$. The state sequence becomes $e_1=(0,1)^T$ at odd positions and $e_0=(1,0)^T$ at even positions. The "active dimension" of the hidden state vector always encodes the current position modulo 2. This shows how time-invariant SSMs maintain periodic counts solely through the gate matrix (without inc terms) to recognize $(aa)^*$. Diagonal SSMs cannot do this because their monotonicity would flatten the parity difference.

## Key Experimental Results
This is a purely theoretical work with no training experiments. The core "data" consists of the expressivity hierarchy and the alignment between architectures and Transformers. The following tables summarize the main conclusions.

### Main Results: Expressivity Lower Bounds for SSM Variants

| SSM Variant | Fixed-width Precision (Lower Bound) | Log-Precision (Lower Bound) | Source |
|:---|:---|:---|:---|
| Diagonal Gated | PLTLf $\equiv$ FO[<] (Star-free) | PLTLf[$\overleftarrow{\#}$] (Includes non-regular $a^nb^nc^n$) | Thm. 1 / Thm. 4 |
| Time-Invariant | UN-PLTLf[MOD] | UN-PLTLf[MOD, $\overleftarrow{\#}$] | Thm. 7 |
| Diag. & Time-Inv. | UN-PLTLf | UN-PLTLf[$\overleftarrow{\#}$] | Cor. 5 |
| Mixed | PLTLf[MOD] $\equiv$ AC$^0\cap$REG | PLTLf[$\overleftarrow{\#}$, MOD] | Cor. 8 |
| Arbitrary Gating | All Regular Languages REG | — | Merrill et al. (2024) |

### Inexpressibility and Precision Thresholds

| Conclusion | Content | Explanation |
|:---|:---|:---|
| Lemma 2 (Monotonicity) | Fixed-width diag. SSM output eventually stabilizes on same symbol | Non-neg gates + finite precision |
| Theorem 3 | Fixed-width diag. SSM cannot recognize $(aa)^*$ | $(aa)^*$ is non-monotonic, contradicts Lemma 2 |
| Conjecture 1 | Fixed-width time-inv. SSM cannot recognize $L(a\,S\,b)$ | since requires input-dependent gates |
| Global counting requires log precision | $a^nb^nc^n$ strictly requires $O(\log n)$ bits | Fixed-width only recognizes regular langs (Zubic et al. 2025) |

### Key Findings
- **Gating determines "Quality," Precision determines "Counting"**: Whether a gate is diagonal/time-invariant determines the ability to compute since or MOD predicates (qualitative difference), whereas fixed-width vs. log-precision determines ability for global counting (recognizing non-regular languages like $a^nb^nc^n$). These dimensions are orthogonal.
- **Diagonal and Time-Invariant SSMs are Incomparable**: Diagonal can do since (covering star-free) but misses periodic parity; time-invariant can do MOD (recognizing $(aa)^*$) but likely cannot do since. Mixed layers combine the two.
- **Precise Alignment with Transformers**: Fixed-width diagonal SSMs $\equiv$ UHAT without positional encoding (both FO[<]). Adding time-invariant layers $\approx$ adding positional encoding to Transformers (both gain MOD, reaching FO[<, MOD]). Under log-precision, SSMs strictly contain C-RASP (as SSMs have strong local operators like yesterday) but are strictly weaker than AHAT+PE, as SSMs are causally restricted to backward counting $\overleftarrow{\#}$ while global attention can do forward counting $\overrightarrow{\#}$.
- **Gap between Lower and Upper Bounds**: The lower bounds for fixed-width SSMs fall within AC$^0$, while the upper bound from Merrill et al. is $\mathrm{TC}^0$ (AC$^0 \subsetneq \mathrm{TC}^0$). The authors speculate the upper bound could be tightened to AC$^0$ (as no mechanism for "modulo constant counting / parity" of events was found), which would align SSMs with UHAT in AC$^0$.

## Highlights & Insights
- **The "layer-per-subformula" construction paradigm is elegant**: Mapping the syntax tree depth of a logic formula directly to SSM layers and mapping each sub-formula to a hidden state dimension provides a constructive lower bound and explains mechanistically why since requires input-dependent gates.
- **Monotonicity Lemma is a powerful tool for impossibility**: Simply by observing "non-negative diagonal gates + finite precision $\implies$ repeated symbols lead to stability," the authors prove fixed-width diagonal SSMs cannot recognize $(aa)^*$. This is a hard architectural bottleneck regardless of optimization or data.
- **The SSM ↔ Transformer bridge is highly transferable**: Mapping gate types to attention types, precision to hard/soft attention, and time-invariant layers to positional encodings creates a dictionary. If a Transformer variant can do X, one can now look up the required SSM gating/precision.
- **Using permutation gates as modulo counters** is a reusable trick: Using the periodicity $P^m=I$ of a permutation matrix to maintain a modulo counter for free suggests that time-invariant gates are not redundant for periodic tasks.

## Limitations & Future Work
- **Mostly lower bounds + a key conjecture**: Results are primarily lower bounds, and the gap with $\mathrm{TC}^0$ upper bounds (AC$^0$ gap) remains unclosed. Conjecture 1 ("Time-invariant SSMs cannot recognize $a\,S\,b$") remains a conjecture due to the difficulty of formalizing dimension interactions and saturated fixed-width arithmetic.
- **Gap between theory and engineering**: The analysis assumes non-negative diagonal gates, saturated fixed-width arithmetic, and ReLU FNNs. Mixed SSMs are not used in practice. Real-world Mamba/S4 floating-point details, normalization, and training dynamics are not characterized. "Expressible" is not "Learnable."
- **Only characterizes "Backward Counting"**: Limited by causal structure, the paper only covers $\overleftarrow{\#}$ and cannot cover forward counting $\overrightarrow{\#}$; the gap with AHAT+PE is structural.
- **Future directions**: Tightening the upper bound to AC$^0$ (verifying if SSMs truly cannot compute global parity) and establishing finer complexity hierarchies based on nesting depth (analogous to Transformer nesting-depth hierarchies).

## Related Work & Insights
- **vs Merrill et al. (2024)**: They provided upper bounds via circuit complexity. Ours provides logic lower bounds; the two are complementary. Ours further speculates the fixed-width SSM upper bound can be tightened to AC$^0$.
- **vs Sarrof et al. (2024)**: They proved diagonal SSMs recognize star-free languages. Theorem 1 recovers this as a special case in a larger hierarchy and extends it along three axes: time-invariant/mixed SSMs (adding MOD), precision impact (log-precision for non-regular counting), and Transformer alignment.
- **vs Zubic et al. (2025)**: They proved fixed-width SSMs only recognize regular languages. Ours is consistent with this and refines the regular range into PLTLf / UN-PLTLf[MOD] / PLTLf[MOD] levels.
- **vs Transformer Logic (Yang et al. 2024a, Barcelo et al. 2024, etc.)**: UHAT $\equiv$ FO[<], UHAT+PE $\equiv$ FO[<, MOD], AHAT $\equiv$ PLTLf[$\overleftarrow{\#}, \overrightarrow{\#}$, MOD], SAT $\equiv$ C-RASP. This paper inserts SSMs into this map: fixed-width diagonal SSMs align with UHAT, mixed layers align with UHAT+PE, and log-precision SSMs contain SAT but are weaker than AHAT+PE. This is the first work to link SSM expressivity directly to formal logic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic characterization of SSM expressivity using temporal logic, building a complete lower bound hierarchy across gating and precision.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical, but the chain of theorems/lemmas/counterexamples is self-consistent; one key conclusion (time-invariant cannot do since) remains a conjecture.
- Writing Quality: ⭐⭐⭐⭐⭐ The three-tier benchmark and layer-by-layer construction are very clear; Figures 1–3 illustrate the hierarchy and alignment intuitively.
- Value: ⭐⭐⭐⭐⭐ Provides hard architectural bottlenecks independent of training, offering substantial guidance for SSM architecture selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models](to_infinity_and_beyond_tool-use_unlocks_length_generalization_in_state_space_mod.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Quotient-Space Diffusion Models](quotient-space_diffusion_models.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)

</div>

<!-- RELATED:END -->
