---
title: >-
  [Paper Note] Characterizing the Expressivity of Local Attention in Transformers
description: >-
  [ACL 2026][LLM/NLP][Local Attention] The authors utilize Linear Temporal Logic (LTL) as a unified characterization tool to rigorously prove the equivalence between global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Local Attention"
  - "Expressivity"
  - "Linear Temporal Logic"
  - "Regular Languages"
  - "Transformer"
date: 2026-05-08
content_hash: 9cd192b411470ff7
---

# Characterizing the Expressivity of Local Attention in Transformers

**Conference**: ACL 2026  
**arXiv**: [2605.00768](https://arxiv.org/abs/2605.00768)  
**Code**: None (Theory + reproduction scripts based on Delétang et al. 2023 / Li & Cotterell 2025)  
**Area**: LLM Theory / Attention  
**Keywords**: Local Attention, Expressivity, Linear Temporal Logic, Regular Languages, Transformer

## TL;DR
The authors utilize Linear Temporal Logic (LTL) as a unified characterization tool to rigorously prove the equivalence between global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$, $k$-local-only ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$, and global+local hybrid ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$. Consequently, they demonstrate that **local and global expressivities are incomparable**, hybrid models are strictly more powerful, and **1-local is the most expressive within the local family**. These theoretical predictions are empirically validated on synthetic regular languages and WikiText-2.

## Background & Motivation

**Background**: Global attention is the core of Transformers—each token can attend to all predecessors, resulting in $O(N^2)$ complexity for sequence length $N$. Models like Longformer, BigBird, and Sparse Transformer employ local attention (where each token only attends to its $k$ nearest neighbors) to reduce complexity to $O(Nk)$.

**Limitations of Prior Work**: Local attention is typically viewed as a compromise that sacrifices expressivity for computational efficiency. However, multiple empirical studies have repeatedly observed that under controlled or even equal compute budgets, incorporating local attention actually improves the quality of tasks like machine translation and language modeling. This "counter-intuitive" phenomenon has lacked a rigorous explanation—is it a coincidental optimization effect, an inductive bias, or a genuine change in expressivity?

**Key Challenge**: Intuitively, "looking further" should be more powerful, yet restricting the field of view performs better empirically. This suggests that the linguistic properties captured by global and local attention are fundamentally different dimensions rather than one being a subset of the other. Proving this requires a language-independent, quantitatively comparable formal semantics—a tool provided by formal language theory and temporal logic.

**Goal**: (1) Find precise LTL fragments corresponding to local-only and hybrid Transformers; (2) prove the inclusion/non-inclusion relationships between these fragments to determine the expressivity hierarchy of different attention patterns; (3) identify the optimal window size; (4) empirically validate theoretical predictions and extend conclusions to natural language scenarios like WikiText-2.

**Key Insight**: Li & Cotterell (2025) proved a bidirectional equivalence between fixed-precision global-attention Transformers and $\mathrm{LTL}[\mathrm{P}]$ (containing a single "past" operator). This paper uses that as a pivot, naturally asking: "Does adding a local mask equate to adding specific temporal operators?"

**Core Idea**: Local attention looks at a bounded suffix → this corresponds exactly to the "yesterday/$k$-steps ago" operator $\mathrm{Y}^{\leq k}$. Hybrid attention corresponds to possessing both $\mathrm{P}$ (unbounded past) and $\mathrm{Y}^{\leq k}$ (bounded past). Since the expressivities of these two operators are mutually incomparable → it naturally follows that global and local expressivities are complementary.

## Method

### Overall Architecture
The paper follows a complete "Logic → Architecture → Experiment" chain: (1) On the LTL side, it characterizes $\mathrm{LTL}[\mathrm{Y}]$ (= definite languages determined by finite suffixes), $\mathrm{LTL}[\mathrm{P}]$ (= left-deterministic polynomials, as proven by Li & Cotterell 2025), and $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}]$ (= locally $\mathcal{R}$-trivial monoids, identifying locally testable languages), and proves strict non-inclusion among them. (2) On the Transformer side, attention masks are formalized: global mask $\mathbf{M}^*_{n,m}=1$ iff $m<n$, $k$-local mask $\mathbf{M}^{\leq k}_{n,m}=1$ iff $\max(1,n-k)\leq m<n$; fixed precision constraints ensure alignment with finite automata. (3) Following the proof framework of Li & Cotterell, it is shown that $k$-local Transformer ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ and hybrid Transformer ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$. (4) Expressivity is validated on 8 synthetic regular languages, and subword prediction is performed on WikiText-2 using GPT-2 small scale models.

### Key Designs

1.  **Proving local/global expressivity complementarity via the incomparability of $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ and $\mathrm{LTL}[\mathrm{P}]$**:
    - **Function**: It fundamentally overturns the intuition that "local is just a weakened global" and provides a formal basis for why "adding local can strictly improve expressivity."
    - **Mechanism**: Two diagnostic witness languages are constructed—$a\Sigma^*$ ("the first character is a") can be defined by $\mathrm{P}$ using $\mathrm{P}(\pi_a \land \neg \mathrm{P}\top)$, but $\mathrm{Y}^{\leq k}$ only looks at bounded suffixes and cannot express "beginning" properties (Theorem 2.2 proves $\mathrm{LTL}[\mathrm{Y}]$ corresponds to definite languages, while languages dependent on the beginning are not definite). Conversely, $L_k = \bigcup_{i=0}^{k-1} \Sigma^* a \Sigma^i$ ("a appears within the last $k$ steps") is directly definable by $\mathrm{Y}^{\leq k} \pi_a$, but it is not a left-deterministic polynomial and thus not in $\mathrm{LTL}[\mathrm{P}]$ (Propositions 2.3, 2.4, Theorem 2.9). The two witnesses prove that $\mathrm{LTL}[\mathrm{P}]$ and $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ are incomparable → hybrid is strictly more powerful than either (Corollary 2.10).
    - **Design Motivation**: To transform the empirical observation that "global+local is strictly stronger than global-only" into a falsifiable formal theorem, providing mathematical grounding for subsequent hybrid architecture design.

2.  **$k=1$ is the most expressive window size in the local family**:
    - **Function**: To provide a precise explanation for why 1-local (sliding window=1) often performs best in practice.
    - **Mechanism**: For $k>1$, witnesses $\mathbf{w}=(\mathtt{ab}^{k-1})^r \mathtt{a}$ and $\mathbf{w}'=(\mathtt{ab}^{k-1})^r$ are constructed; the former belongs to $\Sigma^*\mathtt{a}$ while the latter does not. Lemma C.1 uses dual induction on operator depth $s$ for "$s$-close pairs" to prove that any $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$ formula of depth $\leq r$ cannot distinguish between the end of $\mathbf{w}$ and $\mathbf{w}'$ (Key: $\mathrm{Y}^{\leq k}$ looks at $k$ consecutive steps, but the witness has a period of exactly $k$, making the token at step $i$ identical to $i-k$). This shows that $\mathrm{LTL}[\mathrm{Y}^{\leq k}] \subsetneq \mathrm{LTL}[\mathrm{Y}]$ (Propositions 2.11, 2.12), meaning increasing the window size actually loses expressivity. **1-local is equivalent to the full $\mathrm{Y}$, making it the strongest in the local family**.
    - **Design Motivation**: To explain the phenomenon in works like Longformer/Sparse-Transformer where "smaller windows are better" and point out a potential trade-off: 1-local is most expressive, but implementing $\mathrm{Y}^{\leq k} \psi$ requires $k$ nested $\mathrm{Y}$ operators, increasing depth by $k-1$, which corresponds to a Transformer layer overhead of up to $\times k$. This explains why deeper models can fully exploit the 1-local advantage.

3.  **Position Encoding = Numerical Predicate, incapable of closing the local/global gap**:
    - **Function**: To address the common rebuttal: "Since Transformers have position encodings, can global+RoPE simulate local?"
    - **Mechanism**: Position encodings are viewed as "numerical predicates" $\mathrm{MOD}_m^r$ in LTL (Proposition 2.13). It is proved that if a modulo predicate with $m \geq k$ exists, $\mathrm{Y}^{\leq k}$ can be simulated by $\mathrm{Y}$+$\mathrm{MOD}_m^r$ (using $\mathrm{Y}\psi = \bigvee_{i=1}^m (\mathrm{MOD}_m^i \land \mathrm{Y}^{\leq k}(\mathrm{MOD}_m^{i-1} \land \psi))$). However, SiPE and RoPE correspond to "rational variants" of modulo predicates, which cannot stably simulate this (as shown by Chiang et al. 2023). Theory predicts that neither SiPE nor RoPE can elevate global-only to the level of hybrid.
    - **Design Motivation**: To clarify the myth of "all-powerful position encodings" and provide theoretical support for why RoPE should not simply replace local attention.

## Key Experimental Results

### Main Results: Synthetic Regular Languages (Max perfect generalization length, NoPE, Train len ≤40, Test len 41–500)

| Language (LTL Class) | local-1 | local-2 | local-4 | hybrid-1 | hybrid-2 | hybrid-4 | global |
|---------------------|---------|---------|---------|----------|----------|----------|--------|
| $\Sigma^*\mathtt{a}$ ($\mathrm{LTL}[\mathrm{Y}]$) | **100.0** | 99.7 | **100.0** | **100.0** | 99.7 | 92.7 | 58.4 |
| $\Sigma^*\mathtt{ab}$ ($\mathrm{LTL}[\mathrm{Y}]$) | **100.0** | 99.8 | **100.0** | **100.0** | 99.8 | 80.7 | 53.5 |
| $\mathtt{a}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P}]$) | 50.1 | 49.9 | 50.0 | **100.0** | **100.0** | 99.2 | 99.4 |
| $\Sigma^*\mathtt{a}\Sigma^*\mathtt{b}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P}]$) | 71.6 | 74.3 | 74.9 | **100.0** | 99.9 | **100.0** | 99.0 |
| $(\mathtt{ab})^*$ ($\mathrm{LTL}[\mathrm{P,Y}]$) | 52.0 | 54.1 | 57.3 | **99.8** | 95.1 | 94.6 | 75.2 |
| $\Sigma^*\mathtt{ab}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P,Y}]$) | 71.2 | 85.1 | 96.2 | **99.7** | **100.0** | 71.9 | 56.9 |
| Right-det. poly. (Outside $\mathrm{LTL}[\mathrm{S}]$) | 78.4 | 91.8 | 99.1 | 98.6 | 99.9 | 90.4 | 76.1 |
| Bounded Dyck-2 (Outside $\mathrm{LTL}[\mathrm{S}]$) | 75.2 | 94.7 | 98.4 | 89.7 | 87.4 | 83.5 | 66.7 |

Values represent average accuracy (%) over 5 seeds × 3 learning rates. Bold indicates best in row.

### Ablation Study: WikiText-2 LM perplexity (GPT-2 small, 12 layers / 768 d / 12 heads)

| Position Encoding | global-only | local-only $k=1$ | hybrid $k=1$ | Gain (vs global) |
|---------|-------------|------------------|--------------|------------------|
| Learned absolute | baseline | Better than global | **best**, $-69.7$ ppl | $-69.7$ |
| RoPE | baseline | Weaker than large-$k$ | **best**, $-15.2$ ppl | $-15.2$ |
| SiPE | baseline | Weaker than large-$k$ | **best**, $-11.5$ ppl | $-11.5$ |
| hybrid (increasing $k$) | – | – | Decays to near/below global | Validates $k=1$ optimality |
| local-only (RoPE/SiPE) | – | Prefers large $k$ | – | Matches modulo simulation theory |

### Key Findings
- All three major theoretical predictions are confirmed by synthetic experiments: local-only models only achieve 100% on $\mathrm{LTL}[\mathrm{Y}]$ languages (first two rows), global-only only achieves 100% on $\mathrm{LTL}[\mathrm{P}]$ languages (third and fourth rows), and hybrid (especially $k=1$) achieves ~100% on all first 6 languages belonging to $\mathrm{LTL}[\mathrm{P,Y}]$.
- $k=1$ is consistently strongest within hybrid models: while hybrid-2 achieves 100% on $\Sigma^*\mathtt{ab}\Sigma^*$, hybrid-4 drops to 71.9%, whereas hybrid-1 remains stable at ≥99.7%. This suggests large windows sacrifice the "immediate predecessor" characterization capability unique to 1-local.
- Position encodings do not allow global-only to catch up with hybrid: SiPE often lowers performance; RoPE allows local-2/4 to learn $\Sigma^*\mathtt{a}$ in some tasks (validating the modulo predicate simulation theory), but this comes with degradation in other tasks. Overall, hybrid remains superior.
- On WikiText-2, hybrid-1 achieves the lowest ppl across all 3 position encodings, with a maximum reduction of 69.7 (learned absolute). This is the first end-to-end evidence for the "1-local + global" combination in a natural language setting.
- For languages outside $\mathrm{LTL}[\mathrm{S}]$ (bounded Dyck depth 2, right-det poly), no model generalizes perfectly, yet hybrid still provides the strongest partial generalization—indicating the advantages of hybrid persist beyond the characterized scope.

## Highlights & Insights
- **Advancing "Arch vs. Expressivity" research from semantic intuition to formal bidirectional theorems**: Previous discussions on local vs. global often stalled at the "sparsity is inductive bias" level. This paper provides a strict iff equivalence. This "Architecture ↔ Logic Fragment" mapping is a high-level form of mechanistic interpretability.
- **"1-local is strongest" as a directly reusable design principle**: Many industrial Transformers (e.g., GPT-NeoX, Mistral's SWA) use large sliding windows by default. This paper suggests, theoretically and empirically, paralleling global heads with 1-local heads rather than simply reducing the window count.
- **Explaining "Position encoding is not a panacea"**: Many researchers assume global-only + RoPE/SiPE can simulate local behavior. This paper uses modulo predicates to point out exactly "which encoding can simulate what, and how accurately." This serves as a guide for future position encoding design.
- **"Depth overhead" analysis provides a quantitative formula for expressivity-depth trade-offs**: Replacing $\mathrm{Y}$ with $\mathrm{Y}^{\leq k}$ can require up to $k$ times more layers. This explains why deep models benefit most from the 1-local advantage and provides a theoretical explanation for why large models especially benefit from hybrid attention.
- **Transferable to other sparse patterns**: The "mask → temporal operator" mapping framework can be used to analyze any sparse attention, including the random/global tokens of BigBird/Longformer or various dilated masks.

## Limitations & Future Work
- All theory is built on fixed-precision + softmax assumptions, creating a gap with modern bf16/fp16 training; conclusions under exact arithmetic might be broader.
- WikiText-2 experiments are limited by compute to GPT-2 small scale and a single dataset—the authors acknowledge qualitative conclusions might change in large-scale settings.
- The analysis is limited to a "global vs local" dichotomy; common modern features like sliding-window-with-attention-sink (Mistral), global tokens (BigBird), or dilated patterns are not yet fully characterized in this framework.
- The "1-local is strongest but requires more depth" trade-off does not provide a closed-form depth-window Pareto boundary; choosing $k$ in practice still requires heuristics.
- Position encoding analysis only covers SiPE/RoPE; ALiBi, T5 relative bias, and NoPE are not yet strictly mapped to numerical predicates.

## Related Work & Insights
- **vs. Li & Cotterell (2025)**: They established the global-only ↔ $\mathrm{LTL}[\mathrm{P}]$ baseline; this work is a direct follow-up, adding local and hybrid chains. The proof skeleton follows their Lemma B.12 refinement step.
- **vs. Yang et al. (2024, 2026) "masked hard-attention transformer = star-free languages"**: They focus on a different granularity of hard-attention; this paper’s focus on soft-attention + fixed precision is closer to actual deployment, characterizing different logic fragments (star-free vs. $\mathrm{LTL}[\mathrm{P,Y}^{\leq k}]$).
- **vs. Delétang et al. (2023) Chomsky hierarchy benchmark**: They provided the experimental paradigm (train short, test long + multiple seeds). This paper reuses their codebase but elevates "empirical phenomena" to "formal theorems + validation."
- **vs. Beltagy et al. 2020 (Longformer) / Zaheer et al. 2020 (BigBird)**: Empirically observed hybrid advantages are given their first rigorous theoretical explanation here. This work further suggests replacing default large $k$ in Longformer with a 1-local + global combination.
- **vs. Chiang et al. (2023)**: They studied the precise semantic boundaries of RoPE rational variants; this paper borrows their conclusion to show SiPE/RoPE cannot stably simulate $\mathrm{MOD}$ predicates.
- **Insights**: (1) The "architecture-to-logic-fragment" mapping can be extended to SSMs, Mamba, RWKV, etc. (2) Any sparse-attention paper should include "witness language" tests to pair empirical gains with expressivity validation. (3) Position encoding design should explicitly target "which predicate to simulate"—a new goal-oriented direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work to explain hybrid attention advantages at the LTL bidirectional equivalence level; "1-local is strongest" is a significant counter-intuitive finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematically validates 8 LTL fragment classes via synthetic languages; WikiText-2 covers 3 position encodings × multiple $k$, though lacking large-scale model experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous structure with definitions, propositions, lemmas, and theorems; 60+ pages of appendix make proofs self-contained and traceable.
- **Value**: ⭐⭐⭐⭐⭐ Directly influences sparse attention design and position encoding selection; provides a new baseline for Transformer expressivity theory with practical guidelines for industrial LLM architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](../../NeurIPS2025/llm_nlp/cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](../../NeurIPS2025/llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)

</div>

<!-- RELATED:END -->
