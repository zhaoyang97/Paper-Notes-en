---
title: >-
  [Paper Note] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure
description: >-
  [ICML 2026][Complexity measures] This paper proposes Complexity-as-Advantage (CAA): redefining "complexity" as the **dispersion of regret** across a family of **resource-constrained observers** for the same process. It p…
tags:
  - "ICML 2026"
  - "Complexity measures"
  - "regret dispersion"
  - "information theory"
  - "logical depth"
  - "MDL"
date: 2026-05-08
content_hash: 8dd11e728ca97ca3
---

# Complexity as Advantage: A Regret-Based Perspective on Emergent Structure

**Conference**: ICML 2026  
**arXiv**: [2511.04590](https://arxiv.org/abs/2511.04590)  
**Code**: None  
**Area**: Theory/ML Foundations  
**Keywords**: Complexity measures, regret dispersion, information theory, logical depth, MDL

## TL;DR
This paper proposes Complexity-as-Advantage (CAA): redefining "complexity" as the **dispersion of regret** across a family of **resource-constrained observers** for the same process. It proves that under a log-loss + Markov framework, this is equivalent to the sum of conditional mutual information atoms (recovering excess entropy), and from a coding perspective, it is equivalent to the variance of excess description length (MDL). This unifies Kolmogorov complexity, Bennett's logical depth, and excess entropy into a **computable and empirically estimable** scalar spectrum.

## Background & Motivation

**Background**: There are numerous classical definitions of complexity—Shannon entropy characterizes uncertainty, Kolmogorov complexity $K(x)$ is the shortest program length, Bennett's logical depth describes the "computational effort to unfold structure," and Crutchfield's excess entropy measures predictive information. Each captures an aspect of "structure" from a specific perspective.

**Limitations of Prior Work**: The authors highlight the issue with an intuitive question: why do Shakespearean texts and random noise have similar compression ratios under gzip, yet Large Language Models can easily learn the former while struggling with the latter? Classical measures either **conflate** these two categories (e.g., entropy rate), are **uncomputable** (e.g., $K(x)$), or **do not depend on observer resources** (ignoring who can actually extract the structure).

**Key Challenge**: The "usefulness" of complexity is inherently **relative to observer capability**—a source has "exploitable structure" only if a strong observer can consistently predict it better than a weak one. Most classical definitions are **single scalars** or **absolute quantities**, failing to express "to whom the structure is visible."

**Goal**: To construct a complexity measure that satisfies (i) computability (estimated via regret); (ii) observer relativity (explicit dependence on an observer family); (iii) compatibility with classical theory (recovering entropy, MI, and logical depth in the limit); and (iv) the ability to empirically separate shallow, chaotic, and deep processes.

**Key Insight**: Instead of asking "how complex is this sequence," one should ask "what is the dispersion of regret across a family of observers." If all observers have the same regret (equally poor or equally good), the structure of the source is **indistinguishable** for that family; structure is truly "exploitable" only when significant regret dispersion exists.

**Core Idea**: Define complexity as the **variance** (or maximum gap) of regret over a distribution of observers, thereby characterizing "structure" as the **degree of dispersion** in "which observers can gain an advantage over others."

## Method

### Overall Architecture

CAA is a **general operator** that takes three inputs:
- A process $X = (X_u)_{u \in I}$ (time series, spatial grids, or graph nodes);
- A family of observers $\mathcal{A}$ (any set of predictors/encoders);
- A reference distribution $\pi$ over observers.

It outputs a scalar $\mathrm{CAA}(X; \mathcal{A}, \pi)$ characterizing the regret dispersion. The paper does not introduce new models or train new networks; instead, it translates "complexity" into an **estimable statistic** and proves its equivalence to classical quantities under three specialized scenarios (Markov ladder, budget ladder, and coder ladder).

Overall pipeline:
1. Select $X$ and $\mathcal{A}$;
2. Estimate the average loss $L(A; X)$ and regret $R(A;X) = L(A;X) - L^*(X)$ for each $A \in \mathcal{A}$;
3. Calculate $\mathrm{Var}_{A \sim \pi}[R(A;X)]$ or the maximum gap under $\pi$ to obtain CAA;
4. Scan across different ladders (Markov order, computational budget, encoder sets) to obtain an "advantage profile" and extract scalar metrics.

### Key Designs

1. **General Definition of CAA and Two-Observer Closed-Form Solution**:

    - **Function**: Provides a **scalar measure** of "to whom structure is visible," serving as the foundation for all subsequent specializations.
    - **Mechanism**: Asymptotic average loss is defined as $L(A;X) = \limsup_{|\Lambda|\to\infty}\frac{1}{|\Lambda|}\sum_{u\in\Lambda}\ell(\hat{y}^A_u, X_u)$, optimal loss as $L^*(X) = \inf_A L(A;X)$, and regret as $R(A;X)=L(A;X)-L^*(X)$. Complexity is defined as $\mathrm{CAA}(X;\mathcal{A},\pi) = \mathrm{Var}_{A\sim\pi}[R(A;X)]$, with a max-gap variant $\mathrm{CAA}_{\max}(X) = \sup_{A,B}|R(A;X)-R(B;X)|$. For two observers $\{A_{\text{naive}}, A_{\text{soph}}\}$ under a uniform prior, the closed form is $\mathrm{CAA}(X) = \tfrac14 (\Delta L)^2$ (where $\Delta L = L_{\text{naive}} - L_{\text{soph}}$), directly linking CAA to classical "performance gaps."
    - **Design Motivation**: Redefines "complexity" from an absolute attribute ($K(x)$) into a **statistic relative to a family of observers**. This bypasses uncomputability and explicitly encodes "resource constraints"—the observer family represents the resources.

2. **CAA Decomposition under Markov Ladder: CAA gap = CMI Atoms, Summation = Excess Entropy**:

    - **Function**: Embeds CAA into information theory, proving it is a **step-by-step decomposition** of excess entropy rather than an arbitrary invention.
    - **Mechanism**: Under log-loss, the loss of an order-$m$ Markov predictor is exactly the conditional entropy $L(A^{(m)};X) = H(X_t \mid X_{t-1}, \dots, X_{t-m})$. The difference between adjacent orders $\Delta L_m = L(A^{(m-1)};X) - L(A^{(m)};X) = I(X_t; X_{t-m} \mid X_{t-1}^{t-m+1})$ is precisely a **conditional mutual information atom**. The summation telescopes: $\sum_{m=1}^{M}\Delta L_m = H(X_t) - H(X_t \mid X_{t-1}^{t-M})$, which as $M\to\infty$ converges to excess entropy $E = I(X_{-\infty}^{t-1}; X_t)$. For a $K$-order Markov source, $E = H(X_t) - H(X_t \mid X_{t-1}^{t-K})$, which truncates exactly at $m=K$.
    - **Design Motivation**: This identity equates "the actual advantage gained by extending context by one step" with "the total amount of predictable information in the source." Thus, CAA becomes a **fine-grained version of excess entropy**, identifying exactly at which order the predictive dividend is unlocked.

3. **Budget Ladder + Scalar Depth Metrics: Making Bennett's Logical Depth Measurable**:

    - **Function**: Translates the abstract, uncomputable "logical depth" (computational effort to unfold structure) into shape features of a budget-indexed advantage profile.
    - **Mechanism**: Given an observer family $\{A^{(b)}\}$ where $b$ denotes a computational budget (search depth, rollout length, CA neighborhood radius, etc.), the gap between adjacent budgets $\Delta L_b = L(A^{(b-1)};X) - L(A^{(b)};X) \ge 0$ forms a two-stage CAA gap. Three scalars are extracted from the profile $\{\Delta L_b\}$: tail fraction $\mathrm{TailFrac}_\alpha = \sum_{j>\lfloor\alpha B\rfloor}\Delta L_j / \sum_j \Delta L_j$, half-mass budget $b_{50} = \min\{b:\sum_{j\le b}\Delta L_j \ge M/2\}$, and normalized depth score $D = \frac{1}{B}\cdot \sum_b b\,\Delta L_b / \sum_b \Delta L_b$. Shallow processes (e.g., Rule 90) show early gains $\to$ small $b_{50}$, small $D$; deep processes (e.g., Rule 110) show late gains $\to$ large tail fraction, large $D$; chaotic processes (e.g., Rule 30) show negligible overall gains.
    - **Design Motivation**: Bennett's original definition requires both $K$ and runtime, making it uncomputable. CAA replaces this by identifying "at which budget an observer begins to gain dividends," preserving the intuition that "deep = late extraction" while allowing for **actual computation** in scenarios like cellular automata or cryptography.

### Loss & Training

This is a theoretical and empirical diagnostic framework; **no models were trained**. All "losses" refer to the log-loss of observer predictions $\ell(x, \hat{P}) = -\log_2 \hat{P}(x)$ (or bits-per-symbol for encoders). The estimation protocol follows classical online averaging: calculating the mean online log-loss for a sequence of length $N$, and calculating the mean/std of $\Delta L$ and CAA across $B$ independent resampled sequences. Markov predictors use Laplace smoothing with $\alpha = 1$. In the cryptography ladder, "alignment" details include matching the burn-in with the encryption start and resetting the key phase for the Search observer.

## Key Experimental Results

### Main Results

Experiment I: U-shaped CAA curve on tunable sources. The source is a Bernoulli mixture—outputting a fixed periodic template with probability $p$ and a fair coin with probability $1-p$. Thus, $p=0$ is pure noise and $p=1$ is a pure period. Two observer pairs: Pair A (order-1 vs. order-3 on period-2); Pair B (order-3 vs. order-5 on period-6).

| Observer Pair | $p \approx 0$ (Pure Noise) | $p \approx 1$ (Pure Period) | Intermediate Regime | Behavior |
|---|---|---|---|---|
| Pair A (k=1 vs k=3, period-2) | Small $\Delta L$ | Large $\Delta L$ (order-1 can't lock phase) | Monotonic Increase | Near-monotonic, reflecting persistent strong observer advantage |
| Pair B (k=3 vs k=5, period-6) | Small $\Delta L$ | Small $\Delta L$ (both orders sufficient) | Peak in Middle | Classic U-shape: CAA is significant only in "semi-structured" zones |

$\to$ Validates that CAA captures "whether structure is exploitable"—neither noise nor trivial periods, but **potential patterns of intermediate strength** allow strong observers to truly gain an advantage.

Experiment II: Relativistic Complexity (HMM vs. Crypto Source × Statistical vs. Search Observers). The HMM has sticky transitions + biased emissions; the crypto source uses a periodic key XORed with alternating plaintext.

| Source \ Observer | Stat | Search | CAA$_{\max}$ |
|---|---|---|---|
| HMM | order-$k$ Markov | XOR-seeker (Degenerates to Markov without key) | Stat/Stat=0.135, Stat/Search=0.135 (Nearly identical) |
| Crypto | order-$k$ Markov | XOR-seeker (Immediate decryption with key) | Crypto/Stat=0.536, Crypto/Search=**0.963** |

$\to$ The same cryptographic source is almost "structureless" to statistical observers (appearing i.i.d.), but **fully structured** for search observers. The CAA gap reaches 0.963 bits, demonstrating the core thesis that "complexity is observer-relative."

### Ablation Study

Scalar depth metrics on the CA ladder ($k=20$):

| Process | TailFrac$_{2/3}$ | $b_{50}$ | $D$ | Interpretation |
|---|---|---|---|---|
| Rule 90 (shallow, additive) | 1.00 | 20 | 1.00 | Gain concentrated at small $b$ (front-loaded) |
| Rule 30 (chaotic) | 0.22 | 2 | 0.29 | Almost no gain, overall noise-like |
| Rule 110 (deep, Turing-complete) | 0.40 | 7 | 0.42 | Gain delayed until large $b$ (rear-loaded) |

$\to$ The three scalars cleanly separate shallow, chaotic, and deep processes. Rule 110's deep characteristics (delayed gain) are consistently reflected in higher $b_{50}$ and $D$ values than Rule 30.

Ablation of Observer Set (gzip/bz2 coding perspective):

| Source | $\mathcal{A}_1=\{$gzip, bz2$\}$ | $\mathcal{A}_2=\{$huffman, gzip, bz2$\}$ | Change |
|---|---|---|---|
| Simple order | 0.000 | 1.269 | +1.269 |
| Chaos (i.i.d.) | 0.002 | 0.002 | 0.000 |
| Structured text | 0.194 | 1.203 | +1.009 |

$\to$ Huffman only considers zero-order frequencies, while LZ-based methods (gzip/bz2) exploit long-range dependencies. Adding Huffman drastically increases CAA for "periodic" and "natural text" sources but has no effect on pure noise. This proves CAA's sensitivity to the observer family and its accuracy in diagnosing "which structures are visible to which observers."

### Key Findings

- **U-shaped Curve is CAA's Fingerprint**: Both pure noise and pure structure cause CAA to approach zero; it is significant only in the "semi-learnable" regime—which is precisely where "interesting datasets" emerge in machine learning.
- **CAA is Observer-Relative**: For a cryptographic source, adding/removing key search can shift CAA from 0.536 to 0.963; adding/removing Huffman can shift it from 0 to 1.27. Complexity is not an isolated attribute of the source, but a joint product of the source × observer family.
- **Budget Ladders Correspond to Logical Depth**: The deep nature of Rule 110 (computationally complex, Turing-complete) is consistently characterized by $b_{50}, D, \text{and } \mathrm{TailFrac}$, validating the operational interpretation of "depth = delayed gain."
- **Truncation and Control Experiments Show Robustness**: Block-shuffling disrupts long-range dependencies and collapses the gap between Huffman and LZ coders, causing CAA to drop accordingly. Adding a run-length encoder closes the gap on periodic strings. CAA shifts strictly according to the intuition of "whether exploitable structure remains."

## Highlights & Insights

- **Elevating "Observer-Dependence" to a First-Class Citizen**: Traditional complexity measures either ignore the observer or hide it within the "choice of a universal Turing machine." CAA explicitly integrates the observer family $\mathcal{A}$ and prior $\pi$ into the definition. Consequently, the same source can have entirely different complexity values under different $\mathcal{A}$, aligning with the ML intuition that "dataset difficulty varies by architecture."
- **Strong Unification with Non-Triviality**: CAA equals a decomposition of excess entropy under log-loss + Markov limits, equals the variance of excess description length under MDL, and recovers Bennett's logical depth on budget ladders. These three equivalent paths converging under a single variance formula represent a rare "theoretical trinity."
- **Transferability to ML Practice**: The authors outline three directions—dataset difficulty (the "structural criterion" missing from scaling laws), inductive bias (architecture = observer family; bias is effective $\iff$ regret dispersion in that family is high), and intrinsic motivation (curiosity reward = advantage potential). Each grounds heuristic concepts into an information-theoretic scalar.
- **CA Case Study as a Pedagogical Model**: For Rule 90/30/110—cellular automata that have been studied for decades—there is now a unified scalar framework to simultaneously characterize "front-loaded gain / no gain / rear-loaded gain," providing a more direct "ML-style" description than traditional Lyapunov exponents or entropy rates.

## Limitations & Future Work

- **Observer Family and Prior Selection are Highly Subjective**: CAA values depend heavily on $\mathcal{A}$ and $\pi$. While the paper suggests "budget alignment" and "trimmed variance" for stabilization, it lacks a concept of a "standard observer family," making cross-paper or cross-data comparisons difficult.
- **Small Experimental Scale**: Experiments rely on synthetic sources (Bernoulli mixtures, HMM, XOR crypto, CA, and gzip/bz2/huffman). There are no validations on real datasets or modern neural predictors. The authors acknowledge this as future work, but "diagnostics for ML practice" currently remain at the argumentative level.
- **Lack of Rigorous Analysis of Finite-Sample Bias**: Issues like header/warmup overhead in short sequences, Laplace smoothing parameter $\alpha$, and variance estimation stability under finite $B$ are only briefly mentioned. A rigorous bias/variance analysis is needed for CAA to become a trusted metric.
- **Missing Link to MDL Estimation Algorithms**: While CAA is the "variance of excess description length" from an MDL perspective, the paper does not discuss handling finite-size MDL biases (e.g., NML, Bayesian mixture code) in practice. Using engineering compressors like gzip may introduce systematic biases.
- **Future Directions**: (i) Replacing observer families with transformers of different scales to verify which part of the data contributes the most "advantage dispersion" in scaling laws; (ii) Using CAA as a signal for curriculum learning or data pruning; (iii) Comparing advantage profiles across different modalities of multi-modal data; (iv) Using varying rollout lengths in RL as a budget ladder to ground curiosity rewards.

## Related Work & Insights

- **vs. Kolmogorov Complexity / MDL**: Classical $K(x)$ is a single shortest description length and is uncomputable; MDL approximates it with encoders but outputs a single scalar. CAA extends this to the "variance of excess length across a family," making it computable and explicitly characterizing structure visibility. The advantage of CAA is that it no longer seeks an "absolute compression rate" but diagnoses "which coder can outperform another."
- **vs. Bennett's Logical Depth**: Defined as "minimum steps of the shortest program to output $x$," which is theoretically elegant but uncomputable. CAA operationalizes this by identifying "at which budget $b$ significant $\Delta L_b$ appears," providing scalar metrics for CA rules.
- **vs. Excess Entropy / Computational Mechanics**: Crutchfield's excess entropy is a single scalar. CAA **decomposes it into observer-dependent atoms** under a Markov ladder, allowing one to distinguish how much each order contributes to predictive information.
- **vs. Scaling Laws (Kaplan et al., 2020)**: Scaling laws describe performance trends with data/compute but lack a structural criterion. CAA explains **why** some datasets show larger gaps—they possess high advantage dispersion where structure favors strong models more significantly.
- **vs. Curiosity / RND (Pathak et al., Burda et al.)**: Curiosity-driven RL uses surprise or prediction error as rewards but lacks theoretical grounding. CAA grounds this as "advantage potential," i.e., the extent to which a strong observer outperforms a weak one, providing a clear information-theoretic objective for curiosity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Using regret variance as complexity to unify Kolmogorov, Bennett, and excess entropy under one formula while naturally introducing "observer relativity" is a concept-level original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Conceptual experiments are cleverly designed (U-shape, CA rules, crypto/HMM controls), but the reliance on synthetic sources and small-scale coders means the promises for ML practice remain unverified.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression of ideas with theoretical propositions echoed by empirical charts. However, some formulaic derivations (e.g., the multi-step telescoping for $\Delta L_m$) are somewhat abrupt, and the bibliography is relatively traditional.
- **Value**: ⭐⭐⭐⭐ Provides a computable, comparable, and interpretable new perspective on "complexity." It is insightful for theory, information theory, and ML scaling/inductive bias communities. If advantage profiles can be established for large models, its influence will be significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](../../AAAI2026/others/an_epistemic_perspective_on_agent_awareness.md)
- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](../../ICLR2026/others/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)

</div>

<!-- RELATED:END -->
