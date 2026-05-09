---
title: >-
  [Paper Note] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity
description: >-
  [ICLR 2026][Autoformalization] This paper proposes the ASSESS framework and the TransTED Similarity metric, which parses formal statements into operator trees and incorporates semantic transformations into tree edit distance computation, achieving state-of-the-art evaluation of autoformalization statement similarity (70.16% accuracy, 0.35 Kappa). The paper also releases the EPLA benchmark comprising 1,247 expert-annotated statement pairs.
tags:
  - ICLR 2026
  - Autoformalization
  - Formal Statement Similarity
  - Tree Edit Distance
  - Semantic Transformation
  - Lean Theorem Proving
date: 2026-05-08
content_hash: 4aeb82b56bb58f99
---

# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity

**Conference**: ICLR 2026
**arXiv**: [2509.22246](https://arxiv.org/abs/2509.22246)
**Code**: [GitHub](https://github.com/XiaoyangLiu-sjtu/ASSESS)
**Area**: Multilingual Translation
**Keywords**: Autoformalization, Formal Statement Similarity, Tree Edit Distance, Semantic Transformation, Lean Theorem Proving

## TL;DR
This paper proposes the ASSESS framework and the TransTED Similarity metric, which parses formal statements into operator trees and incorporates semantic transformations into tree edit distance computation, achieving state-of-the-art evaluation of autoformalization statement similarity (70.16% accuracy, 0.35 Kappa). The paper also releases the EPLA benchmark comprising 1,247 expert-annotated statement pairs.

## Background & Motivation

1. **Background**: Autoformalization — converting natural language mathematical propositions into formal languages such as Lean — has advanced rapidly, while evaluation metrics have lagged significantly behind.

2. **Limitations of Prior Work**: String-based methods (e.g., BLEU) ignore semantics, treating $a+b$ and $b+a$ as distinct; proof-based methods provide no graded feedback when proofs fail; LLM-as-Judge approaches are costly and non-reproducible.

3. **Key Challenge**: There is a need for an automated evaluation metric that simultaneously captures semantic equivalence and structural similarity.

4. **Goal**: Design a reproducible, CPU-only evaluation metric that balances semantic and structural information.

5. **Key Insight**: Leverage the Lean Language Server to parse formal statements into operator trees (OPTs) and introduce semantic transformations into the tree edit distance computation.

6. **Core Idea**: TransTED = TED + semantic transformation search, treating logically equivalent expressions as having distance zero.

## Method

### Overall Architecture
A two-stage framework: (1) parse formal statements into operator trees using the Lean Language Server; (2) apply semantic transformations (tactic commands) via heuristic search to minimize the tree edit distance after transformation.

### Key Designs

1. **TED Similarity (Baseline Metric)**:
    - Function: Quantifies the structural correspondence between two formal statements.
    - Mechanism: Statements are parsed into OPTs (operators as internal nodes, operands as leaf nodes); standard tree edit distance is computed and normalized: $sim_{TED}(T_1,T_2) = 1 - d_{TED}(T_1,T_2) / \max(|T_1|,|T_2|)$
    - Design Motivation: Tree structure naturally encodes operator precedence and hierarchical relationships.

2. **TransTED Similarity (Core Metric)**:
    - Function: Integrates semantic transformations on top of TED.
    - Mechanism: The two statements are joined by an equality to form an equation; Lean tactics (rw?, congrArg, ext, etc.) are applied via transformation search, using TED as a heuristic to prioritize transformations that reduce distance. Termination conditions: successful proof (distance = 0), node limit, or time limit.
    - Design Motivation: Theorem 1 proves the unique existence of the maximal pseudo-metric satisfying the TED upper-bound constraint and the transformation monotonicity constraint.

3. **EPLA Benchmark Dataset**:
    - Function: A reliable benchmark for evaluating formal statement similarity metrics.
    - Mechanism: Four translators (Herald, Goedel-Formalizer, Gemini-2.5-Pro, Qwen3-Max) are used to autoformalize miniF2F-test and ProofNet-test; after compilation-based filtering, seven experts annotate the pairs for semantic provability and structural similarity.
    - Design Motivation: Existing benchmarks provide only coarse-grained binary labels, which are insufficient for evaluating fine-grained metric performance.

### Loss & Training
- No training is required; the method is purely inference-based.
- Search parameters: top-5 rw? suggestions, maximum tree size of 32, search timeout of 10 minutes.
- Runs on CPU only, with no GPU dependency.

## Key Experimental Results

### Main Results

| Benchmark | Metric | TransTED | BEq (Proof) | BLEU | Majority Voting |
|-----------|--------|----------|-------------|------|-----------------|
| EPLA-miniF2F | Accuracy | **70.16%** | 59.45% | 68.96% | 46.93% |
| EPLA-miniF2F | Kappa | **0.35** | 0.29 | 0.26 | 0.14 |
| EPLA-ProofNet | Accuracy | **67.31%** | 60.34% | 57.21% | 54.57% |
| EPLA-ProofNet | Kappa | **0.30** | 0.28 | 0.18 | 0.12 |

### Ablation Study

| Configuration | EPLA-miniF2F Kappa | Notes |
|---------------|-------------------|-------|
| TransTED | 0.35 | Improvement from semantic transformations |
| TED only | 0.31 | Pure structure cannot distinguish semantic equivalence |

### Key Findings
- Proof-based methods (BEq) achieve high precision but low recall (many false negatives); TransTED achieves a better balance.
- rw? and norm_cast are used most frequently but have low adoption rates; forall_congr and propext have high adoption rates.
- TransTED is far more stable than BLEU with respect to threshold selection.
- TransTED also achieves the best results on EPLA-ProofNet (67.31% accuracy / 0.30 Kappa), demonstrating cross-dataset generalization.
- Using TED alone yields a Kappa of 0.31; adding semantic transformations improves it to 0.35, validating the critical role of the transformation component.

## Highlights & Insights
- Incorporating theorem prover tactics as semantic transformation operations within an evaluation metric elegantly bridges formal proof and evaluation.
- The theoretical framework (pseudo-metric space + maximal constraint optimization) provides a rigorous mathematical foundation for TransTED.
- The EPLA benchmark adopts a three-way annotation scheme (provability × pre/post-transformation similarity), offering finer granularity than binary labels.
- Full reproducibility on CPU alone makes the approach accessible in resource-constrained settings.
- The deterministic nature of TransTED's computation stands in sharp contrast to the stochasticity of LLM-as-Judge approaches.

## Limitations & Future Work
- The search is constrained to a finite set of tactics, potentially missing certain cases of semantic equivalence.
- Search timeouts and node limits mean that the computed TransTED value is an upper bound rather than an exact value.
- The method supports only Lean 4 and has not been extended to other proof assistants such as Isabelle or Coq.
- The scale of EPLA remains limited (1,247 pairs); larger benchmarks remain to be constructed.
- Expanding the tactic library may further improve the semantic coverage of TransTED.
- The applicability of the method to formal statements outside mathematics (e.g., program verification) has yet to be evaluated.

## Related Work & Insights
- **vs. BLEU**: BLEU entirely ignores mathematical semantics, treating $a+b$ and $b+a$ as different.
- **vs. BEq / Definitional Equality**: Relies on the theorem prover and provides no feedback when the proof fails.
- **vs. LLM-as-Judge**: TransTED is fully reproducible and requires no GPU.
- **vs. GTED**: GTED supports only variable renaming transformations, whereas TransTED supports a richer set of proof-based transformations.
- The success of TransTED demonstrates the value of incorporating domain-specific semantic operations into evaluation metrics.
- Complementary use of TransTED and proof-based methods can yield more comprehensive evaluation results.
- Operator tree parsing depends on the availability and stability of the Lean Language Server.
- Future work may explore extending the approach to other formal languages such as Isabelle and Coq.
- Combining TransTED with neural networks to learn better transformation search strategies is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing tactics into tree edit distance for evaluation metrics is an innovative design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-baseline comparisons, ablation studies, and tactic usage analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are tightly integrated, with clear figures and tables.
- Value: ⭐⭐⭐⭐ Provides a much-needed evaluation tool for the autoformalization community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_for_perplexity.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[AAAI 2026\] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](../../AAAI2026/multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)

<!-- RELATED:END -->
