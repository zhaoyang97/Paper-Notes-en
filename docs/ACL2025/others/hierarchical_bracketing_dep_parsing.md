---
title: >-
  [Paper Note] Hierarchical Bracketing Encodings for Dependency Parsing as Tagging
description: >-
  [ACL 2025][Dependency Parsing] Proposes a family of hierarchical bracketing encodings for the sequence tagging paradigm of dependency parsing, proves that the existing 4-bit encoding is a non-optimal special case of this family, derives an optimal encoding requiring only 12 tags, and extends it to handle arbitrary non-projectivity.
tags:
  - "ACL 2025"
  - "Dependency Parsing"
  - "Sequence Tagging"
  - "Bracketing Encodings"
  - "Projective Trees"
  - "Non-projectivity"
date: 2026-05-08
content_hash: 940c305ce6dd7d0b
---

# Hierarchical Bracketing Encodings for Dependency Parsing as Tagging

**Conference**: ACL 2025  
**arXiv**: [2505.11693](https://arxiv.org/abs/2505.11693)  
**Code**: None  
**Area**: Others  
**Keywords**: Dependency Parsing, Sequence Tagging, Bracketing Encodings, Projective Trees, Non-projectivity

## TL;DR

Proposes a family of hierarchical bracketing encodings for the sequence tagging paradigm of dependency parsing, proves that the existing 4-bit encoding is a non-optimal special case of this family, derives an optimal encoding requiring only 12 tags, and extends it to handle arbitrary non-projectivity.

## Background & Motivation

### Background

**Background**: **Background**: Dependency parsing in the sequence tagging paradigm utilizes pre-trained encoders to convert dependency trees into tag sequences, offering high computational efficiency and competitive performance.

**Limitations of Prior Work**: Early bracketing encodings had an unbounded tag space ($\Theta(n^2)$). Although the 4-bit encoding limits projective trees to 16 tags, it is still sub-optimal. Existing extensions for non-projectivity (such as multi-planar approaches) do not fully align with the encoding framework.

**Key Challenge**: A smaller tag space facilitates easier learning for sequence tagging models; however, how can one minimize the number of tags while maintaining decoding correctness?

**Goal**: Derive the most compact bracketing encoding that simultaneously supports both projective and non-projective dependency trees.

**Key Insight**: Based on Yli-Jyrä's concept of rope cover, a hierarchical bracketing encoding framework is established to distinguish structural arcs (superbrackets) from auxiliary arcs (semibrackets), minimizing tag usage through an optimal rope cover.

**Core Idea**: Leverage a proper rope cover (the minimal set of structural arcs) to compress the tag space of projective trees from 16 to 12, and extend it to non-projective trees through indexed brackets.

## Method

### Overall Architecture

A two-level bracketing system is defined: superbrackets encode structural arcs (requiring one left and one right bracket), and semibrackets encode auxiliary arcs (requiring only a single bracket). The core optimization is to minimize the number of structural arcs.

### Key Designs

1. **Hierarchical Bracketing Framework**: Dependency arcs are divided into structural arcs (encoded as superbracket pairs) and auxiliary arcs (encoded as semibrackets). Auxiliary arcs "depend on" structural arcs, sharing one endpoint, and only require a single bracket to mark the other endpoint.
2. **Proper Rope Cover**: Follows Yli-Jyrä's algorithm: greedily select the longest unlabeled arc starting from the leftmost endpoint as a structural arc, mark dependent arcs as auxiliary arcs, and repeat until all arcs are processed. This is proven to be a rope cover of minimal cardinality.
3. **Optimal Encoding Derivation**: Proves that the tag format for projective trees is $(\backslash)? (> | \mathbf{>} | < | \mathbf{<}) (/)?,$ but the proper rope cover excludes the $</$ and $\backslash>$ combinations (as they imply structural arcs depending on each other), reducing the number of tags from 16 to 12.
4. **Non-projective Extension**: Supports crossing arcs by adding an index $i$ to semibrackets and superbrackets (skipping $i$ matching brackets). The upper bound on the index is determined by "rope thickness", and the UD treebanks require an index of at most 2 in practice.

### Decoding Algorithm

Linear-time stack-based decoding: push to the stack when reading an open bracket, match but do not pop the stack top when reading a closed semibracket, and pop until the matching open superbracket when reading a closed superbracket. In the non-projective extension, indexed brackets skip a specified number of matches.

## Key Experimental Results

### Main Results (Projective Tree LAS)

| Encoding | Tag Size | English PTB LAS | Multilingual Average |
|------|-------|------------|-----------|
| 4-bit (B4) | 16 | Competitive | Competitive |
| Optimal (OP) | 12 | Competitive | Competitive |
| Hexatagging | 8 | Competitive | Competitive |

### Ablation Study (Comparison of Non-projective Handling)

| Method | Coverage | Tag Size |
|------|---------|-------|
| B4 + Pseudo-Projective | Projective + Post-processing | 16 |
| OP + Pseudo-Projective | Projective + Post-processing | 12 |
| OP + Index Extension | Direct Non-projective | 12 $\times$ (1 + Index) |
| 7-bit (B7) | 2-planar | 128 |

- Both OP and B4 show accuracy improvements after pseudo-projective transformation.
- The indexed non-projective extension is more compact on certain languages.

### Key Findings

- The optimal encoding achieves comparable performance to the 16-tag 4-bit encoding using only 12 tags.
- The proper rope cover indeed produces the minimum number of structural arcs (first proof of Theorem 2).
- In non-projective extensions, most languages only require an index of $\le 2$, leading to a limited increase in the actual number of tags.
- Hierarchical bracketing encodings are particularly effective in low-resource scenarios.

## Highlights & Insights

- Elegant theoretical work: unifies the 4-bit encoding into a single framework and proves its sub-optimality.
- Rigorous derivation of the optimal encoding, involving the first proof of the minimality of rope covers.
- The indexed extension for non-projectivity is more natural and compact than multi-planar methods.

## Limitations & Future Work

- The practical performance difference between 12 and 16 tags is minor, meaning the theoretical significance outweighs the practical utility.
- In non-projective encoding, indices introduce extra tag variants, which, though bounded, still increase complexity.
- Insufficient comparison with recent LLM-based dependency parsing methods.
- Evaluated only on UD treebanks, without covering domain-specific treebanks.

## Related Work & Insights

- Systematically applies Yli-Jyrä's rope cover theory to an actual parsing system for the first time.
- Provides a theoretically optimal solution for dependency parsing within the sequence tagging paradigm.
- The theoretical lower-bound analysis of encoding compactness can be generalized to other structure prediction tasks.

## Supplementary Technical Details

- Form of the compact hierarchical bracketing tags: $(\backslash)? (> | \mathbf{>} | < | \mathbf{<}) (/)?,$ which reduces from 16 to 12 tags after excluding $\mathbf{<}/$ and $\backslash\mathbf{>}.$
- Proper rope cover computation: greedily select the leftmost longest unlabeled arc as a structural arc $\rightarrow$ mark dependent arcs as auxiliary arcs $\rightarrow$ repeat.
- Decoding complexity: $O(n),$ with exactly one push and one pop/peek per arc.
- Index distribution in non-projective extensions: In UD treebanks, most languages achieve full coverage with an index of $\le 2,$ while only Ancient Greek requires higher indices.
- Rope cover strategy of 4-bit encoding: takes the longest leftward arc and the longest rightward arc of each node as structural arcs, which is a sub-optimal special case of the proper rope cover.
- Theorem 2 is the first formal proof that proper rope cover has minimal cardinality (previously unproven by Yli-Jyrä).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Elegant theoretical derivation, first proof of the optimal encoding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multilingual evaluation, but lacks comparison with the latest methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous reasoning, with clear theorems and proofs.
- Value: ⭐⭐⭐⭐ High theoretical value, limited practical gain but the direction is correct.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[NeurIPS 2025\] Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers](../../NeurIPS2025/others/coreset_for_robust_geometric_median_eliminating_size_dependency_on_outliers.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](../../AAAI2026/others/model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](../../AAAI2026/others/structure-aware_encodings_of_argumentation_properties_for_clique-width.md)

</div>

<!-- RELATED:END -->
