---
title: >-
  [Paper Note] Unique Hard Attention: A Tale of Two Sides
description: >-
  [ACL 2025][transformer expressivity] This paper proves that in finite-precision transformers, leftmost unique hard attention (UHA) is strictly weaker than rightmost UHA. The former is equivalent to the linear temporal logic fragment LTL[$\Diamond^-$] (i.e., partially ordered finite automata) and has the same expressive power as soft-attention transformers, thereby precisely characterizing the impact of attention directionality on transformer expressivity.
tags:
  - "ACL 2025"
  - "transformer expressivity"
  - "unique hard attention"
  - "linear temporal logic"
  - "formal language"
  - "tiebreaking"
date: 2026-05-08
content_hash: d6d8de6929ed6774
---

# Unique Hard Attention: A Tale of Two Sides

**Conference**: ACL 2025  
**arXiv**: [2503.14615](https://arxiv.org/abs/2503.14615)  
**Code**: None  
**Area**: Other  
**Keywords**: transformer expressivity, unique hard attention, linear temporal logic, formal language, tiebreaking

## TL;DR

This paper proves that in finite-precision transformers, leftmost unique hard attention (UHA) is strictly weaker than rightmost UHA. The former is equivalent to the linear temporal logic fragment LTL[$\Diamond^-$] (i.e., partially ordered finite automata) and has the same expressive power as soft-attention transformers, thereby precisely characterizing the impact of attention directionality on transformer expressivity.

## Background & Motivation
1. **Background**: Understanding the expressivity of transformers is currently a hot topic in theoretical research. Many works establish correspondences between transformers and formal languages/logics by analyzing unique hard attention transformers (UHAT, where attention selects only one position with the maximum score). Yang et al. (2024) proved that UHAT with future masking is equivalent to linear temporal logic (LTL).
2. **Limitations of Prior Work**: When multiple positions share the maximum score, UHA requires tiebreaking, which can select either the leftmost or the rightmost position. The results of Yang et al. utilized both leftmost and rightmost tiebreaking rules but did not carefully distinguish the impact of using only one of them, leading to the potential misconception that the two tiebreaking rules are equivalent.
3. **Key Challenge**: The choice of tiebreaking direction, which seems trivial, actually fundamentally alters the expressivity of transformers, a crucial difference overlooked by existing work.
4. **Goal** (1) Do leftmost UHA and rightmost UHA have the same expressive power? (2) If they differ, what formal systems do they respectively correspond to? (3) What is the relationship between leftmost UHA and soft attention?
5. **Key Insight**: By constructing restricted variants of the B-RASP programming language, this paper proves that leftmost operations cannot simulate rightmost operations (such as being unable to read the value immediately adjacent to the left of the current position), thereby establishing a strict separation of expressivity. Combining this with the soft attention equivalence result from Jiaoda et al., the corollary leftmost UHA = soft attention is derived.
6. **Core Idea**: The tiebreaking direction is not a trivial implementation detail but a key factor determining the expressivity of transformers—leftmost UHA = soft attention < rightmost UHA = full LTL.

## Method

### Overall Architecture
The technical roadmap of this paper consists of three steps: (1) intuitively demonstrating the separation of leftmost and rightmost within the B-RASP programming language framework; (2) rigorously proving that leftmost B-RASP is equivalent to the LTL[$\Diamond^-$] fragment; (3) combining existing results to establish the equivalence of leftmost UHA and soft attention, and providing explicit constructions of LTL formulas and partially ordered finite automata (POFA).

### Key Designs

1. **Intuitive Separation Proof in the B-RASP Framework**:

    - **Function**: Illustrating the difference in capacity between the two attention mechanisms at an easily understandable programming language level.
    - **Mechanism**: The attention operation $P(t) = \blacktriangleleft_{t'}[t'<t, s(t')]\ v(t'):d(t)$ in B-RASP represents selecting the leftmost (◀) position satisfying the condition $s(t')=1$ and reading its value $v(t')$. The key observations are: (a) every leftmost operation can be simulated by two rightmost operations—first finding all satisfying positions and then locating the first one; (b) rightmost can easily read $v(t-1)$ (looking directly at the closest position to the left), whereas leftmost cannot—because leftmost always returns the furthest satisfying position, rendering it unable to locate the position "immediately adjacent to the current position".
    - **Design Motivation**: Providing clear intuition to pave the way for subsequent rigorous proofs.

2. **Proof of Equivalence between Leftmost B-RASP and LTL[$\Diamond^-$]**:

    - **Function**: Precisely characterizing the boundary of leftmost UHA's expressivity.
    - **Mechanism**: LTL[$\Diamond^-$] is a fragment of LTL that only retains the past diamond operator $\Diamond^- \varphi$ ("$\varphi$ holds at some point in the past") and excludes the since (S) and until (U) operators. The proof proceeds in two directions: (a) LTL[$\Diamond^-$] $\rightarrow$ leftmost B-RASP: translating each $\Diamond^-$ formula into a leftmost attention operation; (b) leftmost B-RASP $\rightarrow$ LTL[$\Diamond^-$]: proving that leftmost attention operations can only express "whether there exists a position in the past satisfying the condition," and cannot achieve the "continuously holding since some point" semantics required by the since operator.
    - **Design Motivation**: Clarifying what leftmost UHA can and cannot do through its equivalence with classical formal systems.

3. **Corollary: Leftmost UHA = Soft Attention**:

    - **Function**: Connecting theoretical results with practical transformers.
    - **Mechanism**: Directly utilizing the prior results of Jiaoda et al.—finite-precision future-masked soft attention transformers are equivalent to LTL[$\Diamond^-$]. Since this paper proves that leftmost UHA is also equivalent to LTL[$\Diamond^-$], leftmost UHA and soft attention are equivalent. This implies that leftmost UHA might approximate practical transformers using softmax better than rightmost UHA.
    - **Design Motivation**: Revealing the surprising fact that three seemingly different attention mechanisms (soft attention, average hard attention, and leftmost UHA) have identical expressive power under finite precision.

### Formal Tools
The paper also provides explicit LTL[$\Diamond^-$] formulas and partially ordered finite automata (POFA) to describe the language classes recognizable by leftmost UHA, corresponding algebraically to $\mathcal{R}$-trivial monoids and $\mathcal{R}$-expression regular expressions.

## Key Experimental Results

### Expressivity Hierarchy Comparison

| Transformer Type | Equivalent LTL | Equivalent Logic | Equivalent Automata | Language Class |
|----------------|---------|---------|----------|-------|
| Rightmost+Leftmost UHA | LTL[$\Diamond^-,\Diamond^+$,S,U] | FO[<] | Counter-free | Star-free |
| Leftmost UHA / Soft Attn | LTL[$\Diamond^-$] | PFO²[<] | POFA | $\mathcal{R}$-trivial |
| Rightmost UHA (past mask) | LTL[$\Diamond^+$] | FFO²[<] | RPOFA | $\mathcal{L}$-trivial |

### Key Capability Differences

| Operation | Rightmost | Leftmost |
|------|-----------|----------|
| Reading $v(t-1)$ (immediately adjacent left value) | ✓ | ✗ |
| Detecting "whether $\sigma$ has appeared in the past" | ✓ | ✓ |
| Implementing the since operator "$\varphi$ has continuously held since $\psi$" | ✓ | ✗ |
| Recognizing flip-flop languages | ✓ | ✗ |

### Key Findings
- Leftmost UHA is strictly weaker than rightmost UHA—differing by the expressive power of a full "since" operator.
- Soft attention theoretically cannot recognize flip-flop languages (which require reading the symbol after the most recent "write" instruction), which is consistent with the empirical findings of Liu et al. and provides a theoretical explanation.
- The empirical success of position encodings that bias toward proximal tokens, such as ALiBi, can be partially attributed to their assistance in approximating rightmost tiebreaking.
- The three "weak" attentions (soft, average hard, and leftmost UHA) have completely identical expressive power, with perfectly corresponding language classes, logics, and automata.

## Highlights & Insights
- Elevating the tiebreaking direction from an "implementation detail" to a "key theoretical parameter"—this finding is elegant and profound, reminding theoreticians that tiebreaking rules must be explicitly specified when analyzing transformers.
- The equivalence of Leftmost UHA = soft attention is highly valuable—it suggests that leftmost UHA might be a better theoretical proxy than rightmost UHA, as actual transformers employ soft attention.
- The theoretical explanation of the difficulty with flip-flops directly connects theoretical analysis with empirical phenomena, strengthening the persuasiveness of the findings.

## Limitations & Future Work
- Only finite-precision and position-encoding-free settings are analyzed; how results would change after incorporating position encodings requires further investigation.
- Theoretical results are based on worst-case analysis; actual transformers on finite-length inputs might bypass expressivity limitations through approximation.
- No experimental validation is provided—all results are purely theoretical, lacking comparative experiments on actual transformers.
- Does the simultaneous utilization of leftmost and rightmost (which restores full LTL expressivity) offer practical architectural design insights?
- There is a lack of empirical analysis regarding the trend of attention patterns during actual training (whether they lean toward leftmost or rightmost).
- The impact of the specific number of bits in the finite-precision assumption on the results was not discussed.

## Related Work & Insights
- **vs Yang et al. (2024)**: They proved that leftmost+rightmost UHA = LTL, while this paper precisely separates the individual contributions of the two directions.
- **vs Jiaoda et al.**: They independently proved that soft attention = LTL[$\Diamond^-$]. This paper reaches the same conclusion from the UHA perspective, resulting in cross-validation from two independent paths.
- **vs Merrill & Sabharwal (2023)**: They analyzed log-precision transformers, whereas this work focuses on finite precision, making them complementary.
- The theoretical explanation in this paper regarding the success of ALiBi (approximating rightmost tiebreaking) can guide the design of novel positional encodings.

## Rating
- **Overall Evaluation**: An elegant theoretical work that precisely characterizes the impact of attention directionality on transformer expressivity.
- **Novelty**: ⭐⭐⭐⭐⭐ Proves that the seemingly trivial tiebreaking direction is a crucial distinction, providing deep insights.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work, lacking experimental validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The progression from intuition to formalization is very elegant.
- **Value**: ⭐⭐⭐⭐ Significantly advances the theoretical understanding of transformers.

<!-- Core result: leftmost UHA = soft attn = LTL[Diamond-] ⊊ rightmost UHA = full LTL -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)

</div>

<!-- RELATED:END -->
