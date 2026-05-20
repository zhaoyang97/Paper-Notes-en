---
title: >-
  [Paper Note] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure
description: >-
  [ICML 2026][Complexity Measures] This paper proposes Complexity-as-Advantage (CAA): redefining "complexity" as the **regret dispersion** among a family of **resource-bounded observers** on the same process. It is shown t…
tags:
  - "ICML 2026"
  - "Complexity Measures"
  - "Regret Dispersion"
  - "Information Theory"
  - "Logical Depth"
  - "MDL"
date: 2026-05-08
content_hash: f90490e9db0bf6b1
---

# Complexity as Advantage: A Regret-Based Perspective on Emergent Structure

**Conference**: ICML 2026  
**arXiv**: [2511.04590](https://arxiv.org/abs/2511.04590)  
**Code**: None  
**Area**: Theory/ML Foundations  
**Keywords**: Complexity Measures, Regret Dispersion, Information Theory, Logical Depth, MDL

## TL;DR
This paper proposes Complexity-as-Advantage (CAA): redefining "complexity" as the **regret dispersion** among a family of **resource-bounded observers** on the same process. It is shown that, under the log-loss + Markov framework, this is equivalent to the sum of conditional mutual information atoms (recovering excess entropy), and from a coding perspective, to the variance of excess description length (MDL). Thus, Kolmogorov complexity, Bennett logical depth, and excess entropy are unified into a **computable, empirically estimable** scalar spectrum.

## Background & Motivation

**Background**: There are many classic definitions of complexity—Shannon entropy quantifies uncertainty, Kolmogorov complexity $K(x)$ is the length of the shortest program, Bennett’s logical depth measures the computation needed to "unfold structure," and Crutchfield’s excess entropy quantifies predictive information. Each captures one aspect of "structure."

**Limitations of Prior Work**: The authors highlight an intuitive issue—why do Shakespeare’s texts and random noise have similar compression ratios under gzip, yet large language models can easily learn the former but not the latter? Classic measures either **conflate** these two (e.g., entropy rate), are **incomputable** (e.g., $K(x)$), or **ignore observer resources** (failing to capture "who can extract structure").

**Key Challenge**: The "usefulness" of complexity is inherently **relative to observer capability**—a source is "structured" only if strong observers can consistently outperform weak ones. Classic definitions are mostly **single scalars** or **absolute quantities**, unable to express "for whom structure is visible."

**Goal**: Construct a complexity measure that (i) is computable (via regret estimation); (ii) is observer-relative (explicitly depends on the observer family); (iii) is compatible with classic theory (reduces to entropy, MI, logical depth in the limit); (iv) can empirically separate shallow/chaotic/deep processes.

**Key Insight**: Rather than asking "how complex is this sequence," ask "how dispersed is regret among a family of observers." If all observers have the same regret (all weak or all strong), the structure is **indistinguishable** to that family; only when there is significant regret dispersion does structure become truly "exploitable."

**Core Idea**: Define complexity as the **variance** (or max gap) of regret over the observer distribution, i.e., characterize "structure" as the **dispersion in who can outperform whom**.

## Method

### Overall Architecture

CAA is a **general operator** that takes three inputs:
- A process $X = (X_u)_{u \in I}$ (can be a time series, spatial grid, graph nodes, etc.);
- A family of observers $\mathcal{A}$ (any set of predictors/encoders);
- A reference distribution $\pi$ over observers.

It outputs a scalar $\mathrm{CAA}(X; \mathcal{A}, \pi)$, quantifying regret dispersion for this process among the observer family. The paper introduces no new models or networks, but translates "complexity" into an **estimable statistic**, and proves its equivalence to classic quantities in three specialized scenarios (Markov ladder, budget ladder, coder ladder).

Overall pipeline:
1. Select $X$ and $\mathcal{A}$;
2. For each $A \in \mathcal{A}$, estimate average loss $L(A; X)$ and regret $R(A;X) = L(A;X) - L^*(X)$;
3. Compute $\mathrm{Var}_{A \sim \pi}[R(A;X)]$ or the max gap under $\pi$ to obtain CAA;
4. Scan across different ladders (Markov order, computational budget, encoder sets) to get the "advantage profile," and extract scalar indicators.

### Key Designs

1. **General Definition of CAA and Closed-form for Two Observers**:

    - **Function**: Provides a **scalar measure** for "to whom structure is visible," forming the foundation for all specializations.
    - **Mechanism**: Asymptotic average loss is $L(A;X) = \limsup_{|\Lambda|\to\infty}\frac{1}{|\Lambda|}\sum_{u\in\Lambda}\ell(\hat{y}^A_u, X_u)$, optimal loss $L^*(X) = \inf_A L(A;X)$, regret $R(A;X)=L(A;X)-L^*(X)$. Complexity is defined as $\mathrm{CAA}(X;\mathcal{A},\pi) = \mathrm{Var}_{A\sim\pi}[R(A;X)]$, with a max-gap variant $\mathrm{CAA}_{\max}(X) = \sup_{A,B}|R(A;X)-R(B;X)|$. For two observers $\{A_{\text{naive}}, A_{\text{soph}}\}$ under uniform prior, the closed form is $\mathrm{CAA}(X) = \tfrac14 (\Delta L)^2$ (where $\Delta L = L_{\text{naive}} - L_{\text{soph}}$), directly connecting CAA to the classic "performance gap."
    - **Design Motivation**: Recasts "complexity" from an absolute property ($K(x)$) to a **statistic relative to an observer family**, thus bypassing incomputability and explicitly encoding "resource-boundedness"—the observer family is the resource.

2. **CAA Decomposition under Markov Ladder: CAA gap = Conditional Mutual Information Atom, Sum = Excess Entropy**:

    - **Function**: Embeds CAA into information theory, proving it is not ad hoc but a **stepwise decomposition** of excess entropy.
    - **Mechanism**: Under log-loss, order-$m$ Markov predictor loss is conditional entropy $L(A^{(m)};X) = H(X_t \mid X_{t-1}, \dots, X_{t-m})$. The difference between adjacent orders $\Delta L_m = L(A^{(m-1)};X) - L(A^{(m)};X) = I(X_t; X_{t-m} \mid X_{t-1}^{t-m+1})$ is exactly a **conditional mutual information atom**. Summing telescopes: $\sum_{m=1}^{M}\Delta L_m = H(X_t) - H(X_t \mid X_{t-1}^{t-M})$, which converges to excess entropy $E = I(X_{-\infty}^{t-1}; X_t)$ as $M\to\infty$. For $K$-order Markov sources: $E = H(X_t) - H(X_t \mid X_{t-1}^{t-K})$, with exact truncation at $m=K$.
    - **Design Motivation**: This equality equates "the actual advantage from extending context by one step" with "the total predictable information in the source," making CAA a **fine-grained version of excess entropy**—showing at which order the predictive bonus is unlocked.

3. **Budget Ladder + Scalar Depth Indicators: Making Bennett Logical Depth Measurable**:

    - **Function**: Translates the abstract, incomputable "logical depth" (computation needed to unfold structure) into shape features of a budget-indexed advantage profile.
    - **Mechanism**: Take observer family $\{A^{(b)}\}$, with $b$ as computational budget (search depth, rollout length, CA neighborhood radius, etc.). Each adjacent budget gap $\Delta L_b = L(A^{(b-1)};X) - L(A^{(b)};X) \ge 0$ is a two-step CAA gap. Extract three scalars from the profile $\{\Delta L_b\}$: tail fraction $\mathrm{TailFrac}_\alpha = \sum_{j>\lfloor\alpha B\rfloor}\Delta L_j / \sum_j \Delta L_j$, half-mass budget $b_{50} = \min\{b:\sum_{j\le b}\Delta L_j \ge M/2\}$, normalized depth score $D = \frac{1}{B}\cdot \sum_b b\,\Delta L_b / \sum_b \Delta L_b$. Shallow processes (e.g., Rule 90) have front-loaded gain → small $b_{50}$, small $D$; deep processes (e.g., Rule 110) have delayed gain → large tail fraction, large $D$; chaotic processes (Rule 30) have little overall gain.
    - **Design Motivation**: Bennett’s original definition is "the minimal steps for the shortest program to output $x$," requiring both $K$ and time, making it incomputable; CAA replaces this with "at which budget observers start to gain," preserving the intuition "deep = late to extract," and enabling concrete computation in cellular automata, cryptography, etc.

### Loss & Training

This work is a theoretical + empirical diagnostic framework, **with no model training**. All "losses" are observer-predicted log-loss $\ell(x, \hat{P}) = -\log_2 \hat{P}(x)$ (or encoder bits-per-symbol). Estimation uses classic online averaging: compute mean online log-loss over sequences of length $N$, and mean/std of $\Delta L$ and CAA over $B$ independent resamplings. Markov predictors use Laplace smoothing $\alpha = 1$; in the cryptography ladder, key alignment is ensured by synchronizing burn-in and encryption start, and resetting key phase for Search observers.

## Key Experimental Results

### Main Results

Experiment I: U-shaped CAA curve on tunable sources. The source is a Bernoulli mixture—outputs a fixed periodic template with probability $p$, and a fair coin with probability $1-p$. Thus, $p=0$ is pure noise, $p=1$ is pure periodic, and the middle is noisy periodic. Two observer pairs: Pair A (order-1 vs order-3 on period-2), Pair B (order-3 vs order-5 on period-6).

| Observer Pair | $p \approx 0$ (pure noise) | $p \approx 1$ (pure periodic) | Middle Range | Behavior |
|---|---|---|---|---|
| Pair A (k=1 vs k=3, period-2) | $\Delta L$ small | $\Delta L$ large (order-1 can't lock phase) | Monotonically increasing | Nearly monotonic, strong observer always has advantage |
| Pair B (k=3 vs k=5, period-6) | $\Delta L$ small | $\Delta L$ small (both orders suffice) | Peak in the middle | Classic U-shape: CAA significant only in "semi-structured" region |

→ Validates that CAA captures "exploitable structure"—neither noise nor trivial periodicity, but **moderate latent regularity** gives strong observers a real edge.

Experiment II: Relativistic complexity (HMM vs crypto source × statistical vs search observers). HMM is sticky transition + biased emission, crypto source is periodic key XOR with alternating plaintext.

| Source \ Observer | Stat | Search | CAA$_{\max}$ |
|---|---|---|---|
| HMM | order-$k$ Markov | XOR-seeker (degenerates to Markov without key) | Stat/Stat=0.135, Stat/Search=0.135 (almost identical) |
| Crypto | order-$k$ Markov | XOR-seeker (immediately decrypts with key) | Crypto/Stat=0.536, Crypto/Search=**0.963** |

→ For the same crypto source, statistical observers see "no structure" (appears i.i.d.), but search observers see **full structure**, with CAA gap up to 0.963 bits, directly demonstrating the core point that "complexity is observer-relative."

### Ablation Study

Scalar depth indicators on CA ladder ($k=20$):

| Process | TailFrac$_{2/3}$ | $b_{50}$ | $D$ | Interpretation |
|---|---|---|---|---|
| Rule 90 (shallow, additive) | 1.00 | 20 | 1.00 | Gain all concentrated at small $b$, front-loaded |
| Rule 30 (chaotic) | 0.22 | 2 | 0.29 | Almost no gain, overall noise-like |
| Rule 110 (deep, Turing-complete) | 0.40 | 7 | 0.42 | Gain appears only at large $b$ |

→ The three scalars cleanly separate shallow/chaotic/deep; Rule 110’s deep feature (delayed gain) is higher than Rule 30 in both $b_{50}$ and $D$.

Observer set ablation (gzip/bz2 coding perspective):

| Source | $\mathcal{A}_1=\{$gzip, bz2$\}$ | $\mathcal{A}_2=\{$huffman, gzip, bz2$\}$ | Change |
|---|---|---|---|
| Simple order | 0.000 | 1.269 | +1.269 |
| Chaos (i.i.d.) | 0.002 | 0.002 | 0.000 |
| Structured text | 0.194 | 1.203 | +1.009 |

→ Huffman only sees zero-order frequencies, LZ-based (gzip/bz2) can exploit long-range dependencies. Adding Huffman greatly increases CAA for "periodic" and "natural text," but not for pure noise. This directly shows CAA’s sensitivity to observer family and accurate diagnosis of "which structure is visible to which observer."

### Key Findings

- **U-shaped curve is the fingerprint of CAA**: Both pure noise and pure structure drive CAA to zero; only the "semi-learnable" region is significant—precisely where "interesting datasets" appear in machine learning.
- **CAA is observer-relative**: For the same crypto source, enabling/disabling key search raises CAA from 0.536 to 0.963; adding/removing Huffman raises CAA from 0 to 1.27. Complexity is not an isolated property of the source, but a joint product of source × observer family.
- **Budget ladder almost perfectly matches logical depth**: Rule 110’s deep property (computationally complex, Turing-complete) is consistently captured by $b_{50}, D, \mathrm{TailFrac}$, validating the operational interpretation "depth = late gain."
- **Truncation and control experiments show robustness**: Block-shuffling destroys long-range dependencies and flattens the Huffman–LZ gap, with CAA decreasing accordingly; adding run-length encoder closes the gap on periodic strings. CAA’s changes strictly follow the intuition of "whether structure remains to be exploited."

## Highlights & Insights

- **Elevates "observer dependence" to first-class status**: Traditional complexity either ignores observers or hides them in "universal Turing machine choice." CAA explicitly writes observer family $\mathcal{A}$ and prior $\pi$ into the definition, so the same source can have completely different complexity under different $\mathcal{A}$—perfectly matching the ML intuition that "dataset difficulty varies by architecture."
- **Extremely strong but nontrivial unification**: CAA equals the decomposition of excess entropy under log-loss + Markov, equals the variance of excess description length under MDL, and recovers Bennett logical depth on the budget ladder—three equivalences converging under a single variance formula, a rare "theoretical trifecta."
- **Transferable to ML practice**: The authors outline three directions—dataset difficulty (a "structure criterion" missing from scaling laws), inductive bias (architecture = observer family, bias effective ⇔ regret is significantly lower for that family), intrinsic motivation (curiosity reward = advantage potential). Each grounds existing heuristics in an information-theoretic scalar.
- **CA case studies are textbook examples**: Rule 90/30/110, three well-studied cellular automata, are for the first time characterized by a unified scalar framework that simultaneously captures "front-loaded gain / no gain / delayed gain," more directly and "ML-flavored" than traditional Lyapunov or entropy rate.

## Limitations & Future Work

- **Choice of observer family + prior is highly subjective**: CAA values depend strongly on $\mathcal{A}$ and $\pi$. The paper suggests "aligned budget" and "trimmed variance" for stabilization, but lacks a concept of "standard observer family," making cross-paper/data comparability weak.
- **Small experimental scale**: All experiments use synthetic sources (Bernoulli mixtures, HMM, XOR crypto, CA, gzip/bz2/huffman), with no real datasets or modern neural predictors. The paper acknowledges this as future work; current "ML practice diagnostics" remain at the argumentation stage.
- **Finite-sample bias not rigorously analyzed**: Header/warmup overhead for short sequences, Laplace smoothing parameter $\alpha$, and variance estimation stability for finite $B$ are only briefly mentioned in the "reproducibility" section; rigorous variance/bias analysis is needed for CAA to become a reliable metric.
- **Lack of connection to MDL estimation algorithms**: CAA is "variance of excess description length" under MDL, but practical handling of finite-size MDL bias (e.g., NML, Bayesian mixture code) is not discussed; using engineering compressors like gzip may introduce systematic bias.
- **Possible improvements**: (i) Replace observer family with transformers of different scales to verify which data segments contribute most to advantage dispersion in scaling laws; (ii) Use CAA as a curriculum/data pruning signal; (iii) Compare advantage profiles across modalities in multimodal data; (iv) Use different rollout lengths in RL as budget ladder, grounding CAA as an intrinsic reward.

## Related Work & Insights

- **vs Kolmogorov Complexity / MDL**: Classic $K(x)$ is a single shortest description length, incomputable; MDL uses encoders as approximations but outputs only a scalar. CAA generalizes to "variance of excess length over a family of encoders," making it computable and explicitly characterizing which encoders can see which structure. CAA’s advantage is not chasing "absolute compression rate," but diagnosing "which coder can outperform whom."
- **vs Bennett Logical Depth**: Bennett defines deep = "shortest program needs minimal steps to output," theoretically elegant but incomputable. CAA operationalizes this as "at which budget $b$ does significant $\Delta L_b$ appear," with CA case studies directly yielding $b_{50}, D$ and other scalars.
- **vs Excess Entropy / Computational Mechanics**: Crutchfield’s excess entropy is a single scalar. CAA decomposes it into **observer-dependent atoms** under the Markov ladder, so the predictive information of the same source can be attributed to "which order contributes how much."
- **vs Scaling Laws (Kaplan et al., 2020)**: Scaling laws describe performance curves as data/compute increases, but lack a structure criterion. CAA explains **why** some datasets have large gaps and others small—high advantage dispersion means structure is more accessible to strong models.
- **vs Curiosity / RND (Pathak et al., Burda et al.)**: Curiosity-driven RL uses surprise/prediction error as intrinsic reward, but lacks theoretical grounding. CAA grounds it as "advantage potential = the part strong observers outperform weak ones," giving curiosity a clear information-theoretic target.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Uses regret variance as complexity, unifying Kolmogorov/Bennett/excess entropy in a single formula, and naturally introduces "observer relativity"—a truly conceptual innovation.
- Experimental Thoroughness: ⭐⭐⭐ Clever conceptual experiments (U-shape, CA three rules, crypto/HMM contrast), but all on synthetic sources + small-scale coders, lacking real datasets and neural network validation; practical ML promises not yet fulfilled.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression, theory and empirical charts reinforce each other, but some formula derivations (e.g., multi-step telescoping for $\Delta L_m$) are a bit abrupt, and references are somewhat conservative.
- Value: ⭐⭐⭐⭐ Provides a new, computable, comparable, and interpretable perspective on "complexity," inspiring for theory, information theory, and ML scaling/inductive bias research; if advantage profiles are produced for large models in the future, it will be highly influential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](../../AAAI2026/others/an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](../../AAAI2026/others/a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](../../ACL2025/others/the_harmonic_structure_of_information_contours.md)
- [\[ICML 2026\] Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)

</div>

<!-- RELATED:END -->
