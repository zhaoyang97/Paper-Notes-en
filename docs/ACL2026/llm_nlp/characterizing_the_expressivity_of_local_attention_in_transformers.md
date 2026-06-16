---
title: >-
  [Paper Note] Characterizing the Expressivity of Local Attention in Transformers
description: >-
  [ACL 2026][LLM (Other)][Transformer] The authors utilize Linear Temporal Logic (LTL) as a unified characterization tool to strictly prove the following equivalences: global-only Transformer $\leftrightarrow \mathrm{LTL}[\mathrm{P}]$, $k$-local-only $\leftrightarrow \mathrm{LTL}[\mathrm{Y}^{\leq k}]$, and hybrid global+local $\leftrightarrow \mathrm{LTL}[\
tags:
  - ACL 2026
  - LLM (Other)
  - Transformer
date: 2026-05-08
content_hash: 46affbacc8a5cba3
---
# Characterizing the Expressivity of Local Attention in Transformers

**Conference**: ACL 2026  
**arXiv**: [2605.00768](https://arxiv.org/abs/2605.00768)  
**Code**: None (Theoretical + replication scripts based on Delétang et al. 2023 / Li & Cotterell 2025)  
**Area**: LLM Theory / Attention  
**Keywords**: Local Attention, Expressivity, Linear Temporal Logic (LTL), Regular Languages, Transformer

## TL;DR
The authors utilize Linear Temporal Logic (LTL) as a unified characterization tool to strictly prove the following equivalences: global-only Transformer $\leftrightarrow \mathrm{LTL}[\mathrm{P}]$, $k$-local-only $\leftrightarrow \mathrm{LTL}[\mathrm{Y}^{\leq k}]$, and hybrid global+local $\leftrightarrow \mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$. Consequently, they demonstrate that **local and global expressivities are incomparable**, hybrid models are strictly more powerful, and **1-local is the most expressive** within the local family. Theoretical predictions are empirically validated on synthetic regular languages and WikiText-2.

## Background & Motivation

**Background**: Global attention is the core of Transformers—each token attends to all predecessors, resulting in $O(N^2)$ complexity for sequence length $N$. Models like Longformer, BigBird, and Sparse Transformer use local attention (each token only attends to the previous $k$ neighbors) to reduce complexity to $O(Nk)$.

**Limitations of Prior Work**: Local attention is typically viewed as a compromise that "sacrifices expressivity to save compute." However, multiple empirical studies repeatedly observe that under controlled or even equal compute conditions, the addition of local attention improves the quality of tasks like machine translation and language modeling. This "counter-intuitive" phenomenon lacks a rigorous explanation: is it a coincidental optimization effect, an inductive bias, or a fundamental change in expressivity?

**Key Challenge**: Intuitively, "seeing further" should be more powerful, yet restricting the field of view with local attention is empirically superior. This suggests that the linguistic properties captured by global and local attention are fundamentally different dimensions, rather than one being a subset of the other. Proving this requires a language-independent, quantitatively comparable formal semantics—a tool provided by formal language theory and temporal logic.

**Goal**: (1) Identify precise LTL fragments corresponding to local-only and hybrid Transformers; (2) Prove containment/non-containment relations between these fragments to determine the expressivity hierarchy; (3) Identify the optimal window size; (4) Empirically validate theoretical predictions and extend findings to WikiText-2.

**Key Insight**: Li & Cotterell (2025) proved that fixed-precision global-attention Transformers are bi-directionally equivalent to $\mathrm{LTL}[\mathrm{P}]$ (containing a single "past" operator). This study asks: "What temporal operator does adding a local mask correspond to?"

**Core Idea**: Local attention observes a bounded suffix, which corresponds to the "yesterday/$k$-steps ago" operator $\mathrm{Y}^{\leq k}$. Hybrid attention corresponds to possessing both $\mathrm{P}$ (unbounded past) and $\mathrm{Y}^{\leq k}$ (bounded past). Since the expressivities of these two operators are incomparable, it naturally follows that global and local expressivities are complementary.

## Method

### Overall Architecture
The paper presents a complete "Logic $\rightarrow$ Architecture $\rightarrow$ Experiment" proof chain. The **Mechanism** involves translating attention masks into LTL operators, converting architectural differences in "lookback" into comparable logical expressivity. Specifically: on the logic side, $\mathrm{LTL}[\mathrm{Y}]$ (= definite languages), $\mathrm{LTL}[\mathrm{P}]$ (= left-deterministic polynomials), and $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}]$ (= locally $\mathcal{R}$-trivial monoids/locally testable languages) are characterized, and their mutual non-containment is proven. On the Transformer side, masks are formalized as global masks ($\mathbf{M}^*_{n,m}=1$ iff $m<n$) and $k$-local masks ($\mathbf{M}^{\leq k}_{n,m}=1$ iff $\max(1,n-k)\leq m<n$). Fixed precision constraints ensure alignment with finite automata. Bi-directional equivalences are established: $k$-local Transformer $\leftrightarrow \mathrm{LTL}[\mathrm{Y}^{\leq k}]$ and hybrid Transformer $\leftrightarrow \mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$. Predictions are verified on 8 synthetic regular languages and WikiText-2.

### Key Designs

**1. Proving Complementary Local/Global Expressivity through Incomparability of $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ and $\mathrm{LTL}[\mathrm{P}]$**
To overturn the intuition that local is a "compute-saving weaker version" of global, the authors construct witness languages that are mutually unreachable. One is $a\Sigma^*$ ("starts with a"), expressible in $\mathrm{LTL}[\mathrm{P}]$ as $\mathrm{P}(\pi_a \land \neg \mathrm{P}\top)$, but outside $\mathrm{Y}^{\leq k}$ because $\mathrm{Y}^{\leq k}$ only considers bounded suffixes. Theorem 2.2 proves $\mathrm{LTL}[\mathrm{Y}]$ corresponds to definite languages, and "start-dependency" is not definite. The other is $L_k = \bigcup_{i=0}^{k-1} \Sigma^* a \Sigma^i$ ("'a' appears within the last $k$ steps"), expressible via $\mathrm{Y}^{\leq k} \pi_a$ but not a left-deterministic polynomial, thus outside $\mathrm{LTL}[\mathrm{P}]$ (Theorem 2.9). Their incomparability proves that hybrid models are strictly stronger than either alone (Corollary 2.10), formalizing the "global+local > global-only" observation.

**2. 1-local is the Most Expressive Window Size in the Local Family**
While industrial practice often defaults to large sliding windows, this paper proves that expanding the window actually loses expressivity. For any $k>1$, witness sequences $\mathbf{w}=(\mathtt{ab}^{k-1})^r \mathtt{a}$ and $\mathbf{w}'=(\mathtt{ab}^{k-1})^r$ are used. Lemma C.1 uses dual induction on operator depth $s$ for "$s$-close pairs" to prove that any $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$ formula of depth $\leq r$ cannot distinguish between the ends of $\mathbf{w}$ and $\mathbf{w}'$. The periodicity of the witness matches $k$, making the tokens at step $i$ and $i-k$ identical, effectively "blinding" $\mathrm{Y}^{\leq k}$. This establishes $\mathrm{LTL}[\mathrm{Y}^{\leq k}] \subsetneq \mathrm{LTL}[\mathrm{Y}]$ (Proposition 2.11), meaning 1-local (equivalent to full $\mathrm{Y}$) is the strongest. The trade-off is depth: implementing $\mathrm{Y}^{\leq k} \psi$ with 1-local requires $k$ nested $\mathrm{Y}$ operators, increasing depth by $k-1$, explaining why deep models benefit most from 1-local.

**3. Positional Encodings do not Bridge the Local/Global Gap**
The authors treat positional encodings as LTL numerical predicates $\mathrm{MOD}_m^r$. They prove that if a $\mathrm{MOD}$ predicate with $m \geq k$ exists, $\mathrm{Y}^{\leq k}$ can be simulated by $\mathrm{Y}$ plus $\mathrm{MOD}$. However, SiPE and RoPE correspond to "rational variants" of modulo predicates. As Chiang et al. (2023) proved these cannot stably provide precise $\mathrm{MOD}$ semantics, the theory predicts that neither SiPE nor RoPE allows global-only models to match hybrid models.

## Key Experimental Results

### Main Results: Synthetic Regular Languages (Length Generalization Accuracy, NoPE, Train $\leq 40$, Test $41$–$500$)

| Language (LTL Class) | local-1 | local-2 | local-4 | hybrid-1 | hybrid-2 | hybrid-4 | global |
|----------------------|---------|---------|---------|----------|----------|----------|--------|
| $\Sigma^*\mathtt{a}$ ($\mathrm{LTL}[\mathrm{Y}]$) | **100.0** | 99.7 | **100.0** | **100.0** | 99.7 | 92.7 | 58.4 |
| $\Sigma^*\mathtt{ab}$ ($\mathrm{LTL}[\mathrm{Y}]$) | **100.0** | 99.8 | **100.0** | **100.0** | 99.8 | 80.7 | 53.5 |
| $\mathtt{a}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P}]$) | 50.1 | 49.9 | 50.0 | **100.0** | **100.0** | 99.2 | 99.4 |
| $\Sigma^*\mathtt{a}\Sigma^*\mathtt{b}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P}]$) | 71.6 | 74.3 | 74.9 | **100.0** | 99.9 | **100.0** | 99.0 |
| $(\mathtt{ab})^*$ ($\mathrm{LTL}[\mathrm{P,Y}]$) | 52.0 | 54.1 | 57.3 | **99.8** | 95.1 | 94.6 | 75.2 |
| $\Sigma^*\mathtt{ab}\Sigma^*$ ($\mathrm{LTL}[\mathrm{P,Y}]$) | 71.2 | 85.1 | 96.2 | **99.7** | **100.0** | 71.9 | 56.9 |

Values represent mean accuracy (%) across 5 seeds. Bold indicates the best row performance.

### Ablation Study: WikiText-2 LM Perplexity (GPT-2 small, 12 layers / 768 d / 12 heads)

| Positional Encoding | global-only | local-only $k=1$ | hybrid $k=1$ | Gain (vs global) |
|---------------------|-------------|------------------|--------------|------------------|
| Learned absolute | baseline | > global | **best**, $-69.7$ ppl | $-69.7$ |
| RoPE | baseline | weaker than large-$k$ | **best**, $-15.2$ ppl | $-15.2$ |
| SiPE | baseline | weaker than large-$k$ | **best**, $-11.5$ ppl | $-11.5$ |

### Key Findings
- **Verification of Predictions**: Local-only models achieve 100% only on $\mathrm{LTL}[\mathrm{Y}]$ languages; global-only achieve 100% on $\mathrm{LTL}[\mathrm{P}]$; hybrid-1 passes all six $\mathrm{LTL}[\mathrm{P,Y}]$ languages.
- **Superiority of $k=1$**: Within hybrid models, increasing $k$ degrades performance (e.g., hybrid-4 drops to 71.9% on $\Sigma^*\mathtt{ab}\Sigma^*$), confirming that large windows sacrifice the "immediate predecessor" characterization of 1-local.
- **Positional Encoding Limits**: Positional encodings do not enable global-only models to catch up with hybrids. RoPE helps local-2/4 on some tasks (verifying modulo simulation theory) but degrades on others.
- **WikiText-2**: Hybrid-1 achieves the lowest perplexity across all encoding types, with the largest gain being $69.7$ (learned absolute), providing end-to-end evidence in natural language.

## Highlights & Insights
- **Formalized expressivity vs. intuition**: Moves the discussion from "sparsity is inductive bias" to rigorous bi-directional equivalence theorems.
- **Design Principle**: Suggests that "1-local is strongest." Industrial models using large sliding windows (e.g., Mistral) might benefit from parallelizing global heads with 1-local heads rather than just adjusting window size.
- **Myth of Positional Encodings**: Clarifies that RoPE/SiPE cannot reliably simulate local behavior, providing theoretical support for why positional encodings cannot replace local attention.
- **Depth-Expressivity Trade-off**: Quantifies that using $\mathrm{Y}^{\leq k}$ instead of $\mathrm{Y}$ requires up to $k$ times the depth, theoretically explaining why large models benefit most from hybrid attention.

## Limitations & Future Work
- The theory assumes fixed-precision + soft-max; conclusions might shift under exact arithmetic.
- WikiText-2 experiments were limited to GPT-2 small; larger models/datasets might show different qualitative trends.
- Modern mechanisms like attention sinks (Mistral), global tokens (BigBird), and dilated patterns are not yet fully characterized in this framework.
- Future work aims to extend analysis to encoder-decoder architectures and formalize the expressivity of register tokens.

## Related Work & Insights
- **vs. Li & Cotterell (2025)**: Direct follow-up adding local/hybrid chains to the baseline global $\leftrightarrow \mathrm{LTL}[\mathrm{P}]$ correspondence. 
- **vs. Beltagy et al. (Longformer) / Zaheer et al. (BigBird)**: Provides the first rigorous theoretical explanation for the empirically observed hybrid advantage.
- **vs. Chiang et al. (2023)**: Uses their conclusions on rational variants to explain why SiPE/RoPE cannot stably simulate $\mathrm{MOD}$ predicates.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First study to explain hybrid attention via bi-directional LTL equivalence; the "1-local is strongest" finding is highly counter-intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic verification across 8 LTL classes and WikiText-2, though lacks large-scale LLM validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous structure with 60+ pages of self-contained proofs.
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts sparse attention design and positional encoding selection; provides a new baseline for Transformer expressivity theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](../../NeurIPS2025/llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[CVPR 2025\] Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](../../CVPR2025/llm_nlp/rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)

</div>

<!-- RELATED:END -->
