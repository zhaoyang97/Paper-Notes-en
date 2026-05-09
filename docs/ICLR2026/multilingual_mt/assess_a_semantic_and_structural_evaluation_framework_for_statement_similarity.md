---
title: >-
  [Paper Note] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity
description: >-
  [ICLR 2026][autoformalization] This paper proposes the ASSESS framework, whose core contribution is the TransTED Similarity metric. By parsing formal mathematical statements into Operator Trees (OPTs) and augmenting standard Tree Edit Distance (TED) with semantic transformations driven by Lean proof tactics, ASSESS achieves state-of-the-art performance of 70.16% accuracy and a Kappa score of 0.35 on the EPLA benchmark, while requiring only CPU resources for reproduction.
tags:
  - ICLR 2026
  - autoformalization
  - evaluation metrics
  - tree edit distance
  - Lean
  - formal mathematics
date: 2026-05-08
content_hash: bd284c9792126b6b
---

# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity

**Conference**: ICLR 2026
**arXiv**: [2509.22246](https://arxiv.org/abs/2509.22246)
**Code**: [https://github.com/XiaoyangLiu-sjtu/ASSESS](https://github.com/XiaoyangLiu-sjtu/ASSESS)
**Area**: Formal Mathematics / Evaluation Metrics
**Keywords**: autoformalization, evaluation metrics, tree edit distance, Lean, formal mathematics

## TL;DR

This paper proposes the ASSESS framework, whose core contribution is the TransTED Similarity metric. By parsing formal mathematical statements into Operator Trees (OPTs) and augmenting standard Tree Edit Distance (TED) with semantic transformations driven by Lean proof tactics, ASSESS achieves state-of-the-art performance of 70.16% accuracy and a Kappa score of 0.35 on the EPLA benchmark, while requiring only CPU resources for reproduction.

## Background & Motivation

**State of the Field**: Autoformalization — the task of translating natural language mathematical statements into formal proof languages such as Lean and Isabelle — has advanced rapidly with the development of LLMs. However, automatically evaluating translation quality remains an open and critical problem, as the lack of reliable evaluation metrics severely constrains progress in this area.

**Limitations of Prior Work**: Existing evaluation methods face a fundamental dilemma between semantic and structural fidelity. Text-based metrics (e.g., BLEU) measure only n-gram overlap, are sensitive to minor lexical variations, and entirely ignore semantics — two semantically equivalent but syntactically different Lean expressions (e.g., $a+b$ and $b+a$) may receive very low BLEU scores. Proof-based metrics (e.g., BEq) attempt to assess similarity by automatically proving the equivalence of two statements, but are overly strict — many highly similar yet non-identical statements are directly classified as dissimilar, and the approach is limited by the capabilities of current theorem provers (high false negative rate). LLM-as-Judge approaches are flexible but suffer from poor reproducibility, high computational cost, and GPU dependency.

**Root Cause**: There is a fundamental trade-off between semantic understanding and structural matching — existing methods are either too lenient (BLEU considers only surface form) or too strict (BEq requires a complete equivalence proof), with no intermediate ground for fine-grained evaluation.

**Starting Point**: Formal mathematical statements can be parsed into Operator Trees (OPTs), and tactic operations in Lean correspond to semantics-preserving transformations over these trees. Incorporating semantic transformation information into the tree edit distance computation introduces semantic awareness on top of structural matching — neither ignoring semantics like BLEU nor demanding full equivalence like proof-based methods.

**Core Idea**: TransTED, a semantics-transformation-augmented tree edit distance, is used to measure the similarity between formal statements, striking an optimal balance between purely structural and purely semantic approaches.

## Method

### Overall Architecture

ASSESS is a two-stage framework. In the first stage, the Lean Language Server parses pairs of formal statements into OPTs, capturing hierarchical structural information. In the second stage, a curated set of Lean tactic transformations is incorporated on top of standard TED to enable semantic awareness, yielding the TransTED Similarity as a real-valued similarity score. The entire pipeline relies only on a CPU and the Lean Language Server/REPL, with no GPU required.

### Key Designs

1. **Operator Tree (OPT) Construction**:

   - Function: Converts the raw text of a formal statement into a tree representation that preserves structural information.
   - Mechanism: The Lean Language Server is used to parse formal statements; operators (e.g., function applications, quantifiers, logical connectives) become internal nodes, and operands become their ordered children. Two normalization steps are applied during construction: (1) non-leaf nodes are assigned a `<SLOT>` placeholder label to distinguish operators from operands; (2) parentheses are omitted, as precedence information is already implicitly encoded in the tree structure. This representation captures the hierarchical structure of statements more precisely than plain text.
   - Design Motivation: The syntactic structure of formal languages inherently encodes rich semantic information; tree representations encode operator precedence and dependency relations more naturally than sequential representations.

2. **TED Similarity (Baseline Metric)**:

   - Function: Quantifies the structural difference between two OPTs.
   - Mechanism: The set of OPTs is treated as a pseudometric space — permitting $d(x,y)=0$ while $x \neq y$ (since semantically equivalent but structurally different expressions should have zero distance). The tree edit distance $d_{\text{TED}}$ is defined as the minimum cost of transforming one tree into another via deletion, insertion, and relabeling operations. TED Similarity is normalized as $\text{sim}_{\text{TED}}(T_1, T_2) = 1 - d_{\text{TED}}(T_1, T_2) / \max(|T_1|, |T_2|)$, where $|T|$ denotes the number of nodes.
   - Design Motivation: While TED is effective for structural matching, it exhibits systematic bias toward semantically equivalent expressions with different syntax (e.g., $a+b$ vs. $b+a$), which require multiple edit steps to align in tree form.

3. **TransTED Similarity (Core Contribution)**:

   - Function: Augments TED with semantic transformations to address its bias toward semantically equivalent but syntactically different expressions.
   - Mechanism: A new pseudometric $d^*$ is defined satisfying two constraints: (a) it is upper-bounded by TED, i.e., $d^*(T_1, T_2) \leq d_{\text{TED}}(T_1, T_2)$; (b) semantic transformation monotonicity — if an expression pair $(e_x, e_y)$ can be transformed into a logically stronger pair $(e_u, e_v)$ (i.e., $e_u=e_v \Rightarrow e_x=e_y$), then $d^*(OPT(e_x), OPT(e_y)) \leq d^*(OPT(e_u), OPT(e_v))$. The paper proves that the unique maximal pseudometric satisfying both constraints exists (Theorem 1), which is TransTED. In practice, a pair of statements is first connected by an equality to form an equation; a curated set of tactics (e.g., `rw?`, `apply congrArg`, `ext`, `norm_cast`) is then applied in the Lean REPL via heuristic search, using TED as a heuristic to prioritize transformations that reduce the OPT difference between both sides. Search terminates upon proof of equivalence, node limit exceeded (NLE), or time limit exceeded (TLE).
   - Design Motivation: Pure structural distance cannot handle the large number of semantically equivalent but syntactically different expressions in formal mathematics. By leveraging Lean tactic-driven transformations, the distance metric gains awareness of commutativity, quantifier decomposition, and other semantic equivalences — achieving semantic sensitivity without sacrificing structural information.

### EPLA Benchmark Dataset

The paper also introduces the EPLA (Evaluating Provability and Likeness for Autoformalization) benchmark. Natural language statements from miniF2F-test and ProofNet-test are formalized using four translation models (Herald Translator, Goedel-Formalizer-V2-8B, Gemini-2.5-Pro, and Qwen3-Max). After filtering with the Lean compiler, 7 expert annotators label the resulting pairs for semantic equivalence and structural similarity, yielding 1,247 annotated pairs (831 from miniF2F, 416 from ProofNet).

## Key Experimental Results

### Main Results

| Metric | Identity Match | BLEU | Majority Voting | BEq | TED Sim | **TransTED Sim** |
|--------|---------------|------|----------------|-----|---------|-----------------|
| miniF2F Accuracy | 32.61% | 68.96% | 46.93% | 59.45% | 69.56% | **70.16%** |
| miniF2F Kappa | 0.05 | 0.26 | 0.14 | 0.29 | 0.31 | **0.35** |
| ProofNet Accuracy | 43.51% | 57.21% | 54.57% | 60.34% | 64.67% | **67.31%** |
| ProofNet Kappa | 0.03 | 0.18 | 0.12 | 0.28 | 0.23 | **0.30** |

TransTED Similarity achieves state-of-the-art accuracy and Kappa scores on both datasets. Compared to BEq (proof-based), Kappa improves from 0.29 to 0.35 on miniF2F, demonstrating that semantics-augmented structural comparison better balances precision and recall than rigid proof-based methods.

### Ablation Study

| Configuration | miniF2F Acc | miniF2F Kappa | ProofNet Acc | ProofNet Kappa |
|--------------|------------|---------------|-------------|----------------|
| TED Similarity (w/o transformation) | 69.56% | 0.31 | 64.67% | 0.23 |
| **TransTED Similarity (w/ transformation)** | **70.16%** | **0.35** | **67.31%** | **0.30** |
| Gain | +0.60pp | +0.04 | +2.64pp | +0.07 |

The semantic transformation component is a key driver of performance gains, particularly on ProofNet where Kappa improves by 0.07, indicating that transformations are especially advantageous for more complex mathematical statements.

### Key Findings

- **High precision but low recall of proof-based methods (BEq)**: BEq achieves 98.60% precision on miniF2F but only 45.77% recall, indicating that automatic theorem provers frequently fail to recognize valid equivalences. TransTED avoids this rigid dependency through incremental transformations.
- **Complementary tactic usage patterns**: `rw?` and `norm_cast` are high-frequency general-purpose tools (for exploring the search space), while `apply forall_congr; intro _` and `rw [propext and_imp]` are low-frequency but high-adoption specialized tools (for precisely resolving specific logical structures). The synergy between general and specialized tactics constitutes a robust search strategy.
- **TransTED exhibits stable performance across thresholds**: In contrast to BLEU's sensitivity to threshold selection, TransTED maintains high performance across a broad range of threshold values.

## Highlights & Insights

- **Mathematical elegance of the pseudometric space**: Permitting $d(x,y)=0$ while $x \neq y$ precisely captures the core requirement of formal mathematics — "semantically equivalent but syntactically different" — making this formulation more suitable than metrics that enforce strict distance axioms.
- **Lean tactics as semantic bridges**: The proof assistant's own tactic system is cleverly repurposed as a semantic transformation engine — no model training or GPU is required; only the Lean Language Server and a curated set of tactics are needed, enabling efficient and reproducible semantic awareness.
- **Unified role of TED as metric and heuristic**: TED simultaneously serves as a component of the final metric and as the heuristic guiding the search process — transformations that reduce TED correspond directly to meaningful semantic equivalence steps, yielding a clean and coherent design.

## Limitations & Future Work

- **Limited transformation set**: Only a small number of manually curated tactic commands are employed, covering a restricted range of semantic transformations. Additional Lean tactics or custom transformation rules could further improve performance.
- **Search efficiency**: Evaluating a single statement pair may require up to 10 minutes of search time, which could become a bottleneck in large-scale applications.
- **TransTED computes an upper bound**: Due to the finite set of available transformations, the computed value is an upper bound on the theoretically optimal TransTED. The distance is exact (i.e., zero) only when the transformation sequence successfully proves equivalence.
- **Limited scale of EPLA**: Although carefully annotated, the 1,247 pairs may be insufficient in diversity and coverage — extending to a broader range of mathematical domains and supporting additional formal languages are natural directions for future work.

## Related Work & Insights

- **vs. BLEU**: BLEU treats formal statements as ordinary text and entirely ignores mathematical semantics. TransTED retains structural matching while introducing semantic awareness, improving Kappa from 0.26 to 0.35 on EPLA.
- **vs. BEq (proof-based)**: BEq requires a successful proof of full equivalence, representing an all-or-nothing evaluation scheme. TransTED provides continuous-valued similarity scores, better suited to evaluating the quality gradient of "nearly correct" translations.
- **vs. GTED (Liu et al., 2025c)**: GTED is the first work to use operator trees for formal evaluation, but suffers from implementation instability and limits transformations to variable renaming. ASSESS formalizes the approach within a rigorous pseudometric space framework and introduces a rich set of proof-based transformations.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating the Lean tactic system as a semantic transformation engine into tree edit distance is an elegant innovation; the pseudometric space theoretical framework is also mathematically appealing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Baselines are comprehensive, ablation analysis is thorough, and the tactic usage pattern analysis offers genuine insight; however, the scale of EPLA is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ The theoretical sections are formally rigorous and clearly presented; the experimental sections are well-organized.
- Value: ⭐⭐⭐⭐ Provides the autoformalization community with a more reliable, efficient, and reproducible evaluation tool.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[AAAI 2026\] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](../../AAAI2026/multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)

<!-- RELATED:END -->
