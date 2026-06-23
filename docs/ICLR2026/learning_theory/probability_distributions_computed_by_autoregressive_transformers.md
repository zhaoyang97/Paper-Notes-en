---
title: >-
  [Paper Note] Probability Distributions Computed by Autoregressive Transformers
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper extends research on Transformer expressivity from the traditional "classifier" setting (accept/reject strings) to the "autoregressive probabilistic language model" setting. It demonstrates that the shifts to "autoregression" and "real-valued weights (probabilities)" alter or even break existing equivalence r
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 2c931a4246e3288a
---
# Probability Distributions Computed by Autoregressive Transformers

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=gZIcyx1tQY](https://openreview.net/forum?id=gZIcyx1tQY)  
**Code**: To be confirmed  
**Area**: Learning Theory / Transformer Expressivity  
**Keywords**: Transformer Expressivity, Autoregressive Language Models, Weighted Automata, Linear Temporal Logic, Formal Languages

## TL;DR
This paper extends research on Transformer expressivity from the traditional "classifier" setting (accept/reject strings) to the "autoregressive probabilistic language model" setting. It demonstrates that the shifts to "autoregression" and "real-valued weights (probabilities)" alter or even break existing equivalence results—sometimes making Transformers stronger, and sometimes making classifiers and autoregressors incomparable.

## Background & Motivation
**Background**: Most theoretical work on Transformer expressivity (Yang et al. 2024, Jerad et al. 2025, Li & Cotterell 2025, etc.) analyzes Transformers as **language recognizers**: they take a full string as input and output a Boolean decision. Under this setting, several elegant results have been established, such as the equivalence of unique hard-attention Transformers (UHAT) as classifiers to Linear Temporal Logic (LTL) and counter-free finite automata (cfDFA).

**Limitations of Prior Work**: Transformers are not used this way in practice. Real language models differ in two fundamental ways: first, the input is a **prefix** rather than the whole string, and the output is a **prediction for the next token**; second, the output is a **probability distribution** rather than a Boolean decision. There are two systematic gaps between theoretical objects (Boolean classifiers) and practical ones (real-valued autoregressors).

**Key Challenge**: No one has verified whether the equivalence results from the "recognizer era" still hold in the "language model" setting. If they do, existing theories can be directly reused; if not, assuming recognizer results apply to language models is risky—a common practice in current literature.

**Goal**: Deconstruct expressivity along two independent dimensions: (1) **Unweighted (Boolean) vs. Real-valued weights**, and (2) **Classifier (full string to scalar) vs. Autoregressor (prefix to next-symbol distribution)**. This forms a 2×2 quadrant framework to determine if established equivalences hold in each cell.

**Key Insight**: The authors introduce a unified abstraction: the **state encoder**. It treats Transformers, DFAs, and LTL as the same category of objects that map strings to state sequences where the $i$-th state depends only on the prefix $w_{\le i}$. By pairing this with different output functions, classifiers and autoregressors can be analyzed within the same framework.

**Core Idea**: By using the "state encoder + output function" pair, expressivity problems are unified and parameterized as "classifier output function vs. autoregressive output function" and "Boolean semiring vs. real semiring," accurately characterizing which equivalences are broken by autoregression or probabilistic weighting.

## Method

### Overall Architecture
The paper builds a **formal framework** rather than proposing a new model. The logic follows three steps: first, use the **state encoder** abstraction to unify Transformers, DFAs, and LTL (all are "prefix to state" functions); second, equip the state encoder with two output functions to derive **classifier** and **autoregressor** usages; finally, compare the expressivity of these four combinations across **Boolean** and **real semirings**, deriving a series of "equivalence / strict containment / incomparability" results.

The logical backbone treats "weighted languages" $S:\Sigma^*\to K$ as the research object ($K$ is a semiring). Classifiers and autoregressors are simply two ways to turn the same state encoder into a weighted language, reducing expressivity comparison to "which output function can express which weighted languages."

### Key Designs

**1. State Encoder: Unifying Transformers, DFAs, and LTL**

In traditional research, Transformers, finite automata, and temporal logic are separate formal systems. This paper unifies them via the **state encoder**: a function mapping $w_1\cdots w_n$ to states $q_0,\dots,q_n\in Q$ where $q_i$ depends only on $w_{\le i}$ (causality). The hidden state $h^{(i)}$ of a Transformer, the transition $\delta^*(\iota,w_1\cdots w_i)$ of a DFA, and the truth vector of LTL formulas $\Phi(w)_i$ at position $i$ are all valid state encoders. The authors prove (Thm. 5.1) that UHAT, LTL, and cfDFA define **equivalent state encoders**, forming the foundation for further comparison.

**2. Classifier vs. Autoregressor: Two "Readout" Methods**

A **classifier** reads a scalar weight at the final position: $C(w)=c(T(w)_n)$. An **autoregressor** reads a weight distribution for the next symbol (including EOS) at each position: $\Pr_A(\sigma\mid w_{\le i})=r(T(w)_i)(\sigma)$. These conditional distributions are multiplied to form the weight of the string:

$$ \Pr_A(v\mid u)=\Big(\bigotimes_{i=1}^{|v|}\Pr_A(v_i\mid u v_{<i})\Big)\otimes\Pr_A(\mathrm{EOS}\mid uv) $$

A **normalization condition** is enforced: for any prefix $u$, $\bigoplus_{v\in\Sigma^*}\Pr_A(v\mid u)=1$. This constraint is the root of expressivity divergence—it implies autoregressors must define normalized languages without "dead ends" or "infinite loops."

**3. Separation in Real Semirings: Incomparability and Determinism**

Under real weights, real classifiers express **aperiodic step functions**: $S(w)=\bigoplus_{i=1}^m k_i\otimes\mathbb{I}\{w\in L_i\}$. This leads to the counter-intuitive conclusion (Cor. 5.4) that classifiers and autoregressors are **incomparable**. For example, $(\tfrac{1}{2} a)^*$ is expressible by an LTL autoregressor but not by an LTL classifier. Moreover, under real semirings, UHAT/LTL as autoregressors are only equivalent to **deterministic** cfDFAs, while weighted cfNFAs are strictly stronger (Prop. 5.5).

**4. Separation in Boolean Semirings: Fragmented LTL and Autoregressive Strength**

In Boolean semirings, classifiers and autoregressors of full LTL are **equivalent** (Thm. 5.6). However, the proof relies on $\mathsf{Y}$ (Yesterday) and $\mathsf{H}$ (Historically) operators. If these are removed, the equivalence collapses. The authors confirm (Prop. 5.8) that $(ab)^*$ is expressible by a $\mathrm{TL}[\varnothing]$ autoregressor but not by any $\mathrm{TL}[\mathsf{H}]$ or $\mathrm{TL}[\mathsf{Y}]$ classifier. This implies that **Transformer variants (like SMAT or UHAT) as autoregressors are strictly stronger than as classifiers**.

### Mechanism: Why $(ab)^*$ is Autoregressive-Expressible but not Classifier-Expressible
Using $\Phi=(\mathrm{BOS},a,b)$ as a state encoder, states track "start of string / last was a / last was b." The autoregressive function allows $a$ if $q_{\mathrm{BOS}}$ or $q_b$ is true, $b$ if $q_a$ is true, and EOS if $q_{\mathrm{BOS}}$ or $q_b$ is true. This precisely generates $(ab)^*$. A classifier, reading the state only once at the end, cannot enforce these step-by-step constraints in fragments lacking specific temporal operators.

## Key Experimental Results

### Main Results: Expressivity Across the Four Quadrants

| Setting | Classifier | Autoregressor | Relationship |
|---|---|---|---|
| Boolean · UHAT (=Full LTL) | = LTL/cfDFA | = LTL/cfDFA | Equivalent (Thm. 5.6) |
| Boolean · Left-to-right UHAT / Fixed-point SMAT (=TL[P]≡TL[H]) | TL[P] | **Strictly Stronger** | Autoreg. > Class. (Prop. 5.8) |
| Real · UHAT (=LTL) | Aperiodic step functions = cfDFA = cfNFA | = cfDFA | **Incomparable** (Cor. 5.4) |
| Real · cfNFA vs UHAT Autoreg. | — | UHAT limited to cfDFA | cfNFA **Strictly Stronger** (Prop. 5.5) |

### Key Separation Examples

| Weighted Language | Classifier Expressible? | Autoregressor Expressible? | Explanation |
|---|---|---|---|
| $(\tfrac{1}{2} a)^*$ (Real) | No | Yes | Infinite weight variety vs. finite step function |
| $(1a)^*$ (Real) | Yes | No | Non-normalized; violates autoregressive constraint |
| $(ab)^*$ (Boolean, TL[H]/TL[Y]) | No | Yes | Fragment lacks operators; autoreg. recovers via step constraints |
| $(aab)^*$ (Boolean, TL[H]) | No | No | Differentiating aab/aaab requires nested Y; autoreg. only adds one Y |

### Key Findings
- **Normalization is the switch for divergence**: In real semirings, incomparability stems from autoregressors being forced to be normalized while classifiers are not.
- **Operator absence enables gain**: In full LTL, autoregression adds no power. Gain only appears in fragments lacking $\mathsf{Y}$ or $\mathsf{H}$, where next-symbol prediction compensates for missing operators.
- **Construction cost lower bound**: Desugaring $\mathrm{prefix}(\phi)$ into an equivalent LTL formula is exponential; no poly-time equivalent construction exists unless P=PSPACE (Prop. 5.7).

## Highlights & Insights
- **One abstraction for three systems**: The state encoder provides a reusable interface for analyzing any causal sequence model (RNN, SSM, etc.).
- **Probability as a first-class citizen**: Using semirings treats "Boolean logic" and "real probability" uniformly, revealing how probabilistic constraints break Boolean equivalences.
- **Explaining empirical anomalies**: Provides a theoretical explanation for why SMAT models in Yang et al. (2025) perform better on next-token prediction than predicted by classifier theory.
- **Methodological warning**: Directly porting recognizer results to language models is risky.

## Limitations & Future Work
- **Idealized models**: Analysis focuses on UHAT and fixed-point SMAT; extending this to realistic softmax attention and log-precision is future work.
- **Lack of inexpressibility tools**: There are currently no robust methods to prove a language cannot be expressed by an autoregressor in certain quadrants (e.g., real weights).
- **Chain-of-Thought (CoT)**: This study only covers direct autoregression without intermediate reasoning steps.
- **Lack of empirical verification**: Conclusions are based on constructive counterexamples rather than trained Transformers.

## Related Work & Insights
- **vs. Yang et al. (2024) / Jerad et al. (2025) / Li & Cotterell (2025)**: These works establish equivalences for **Boolean classifiers**. This paper asks if these hold for **autoregressors**, finding they often break in fragmented LTL or real-weighted settings.
- **vs. Svete & Cotterell (2024)**: While they focus on n-gram coverage by average hard-attention, this paper provides a **general** characterization of classifier vs. autoregressor expressivity.
- **vs. Hahn (2020) / Yao et al. (2021)**: Previous works were limited to specific languages (PARITY, Dyck). The state encoder framework provides a unified, generalizable way to compare distributions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically characterizes autoregressive expressivity for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong theoretical proofs but lacks empirical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear framework and well-chosen counterexamples.
- Value: ⭐⭐⭐⭐⭐ Corrects a common misconception in the field.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Transformers Are Inherently Succinct](transformers_are_inherently_succinct.md)
- [\[ICLR 2026\] Neyman-Pearson Classification under Both Null and Alternative Distributions Shift](neyman-pearson_classification_under_both_null_and_alternative_distributions_shif.md)
- [\[ICLR 2026\] Softmax Transformers are Turing-Complete](softmax_transformers_are_turing-complete.md)
- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Quantitative Bounds for Length Generalization in Transformers](quantitative_bounds_for_length_generalization_in_transformers.md)

</div>

<!-- RELATED:END -->
