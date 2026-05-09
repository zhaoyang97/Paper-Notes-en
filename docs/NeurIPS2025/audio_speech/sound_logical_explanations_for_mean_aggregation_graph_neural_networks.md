---
title: >-
  [Paper Note] Sound Logical Explanations for Mean Aggregation Graph Neural Networks
description: >-
  [NeurIPS 2025][Audio & Speech][graph neural networks] For GNNs using mean aggregation (MAGNN, i.e., mean-GNNs with non-negative weights), this work precisely characterizes the class of monotone logical rules that can serve as sound explanations, constructs a restricted fragment of first-order logic to explain arbitrary MAGNN predictions, and empirically demonstrates that restricting to non-negative weights does not significantly hurt performance while enabling effective extraction of sound rules.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - graph neural networks
  - explainability
  - logical rules
  - mean aggregation
  - knowledge graph completion
date: 2026-05-08
content_hash: 80cb45e38c4eb69a
---

# Sound Logical Explanations for Mean Aggregation Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2511.11593](https://arxiv.org/abs/2511.11593)
**Code**: None
**Area**: graph_neural_networks / AI_theory
**Keywords**: graph neural networks, explainability, logical rules, mean aggregation, knowledge graph completion

## TL;DR
For GNNs using mean aggregation (MAGNN, i.e., mean-GNNs with non-negative weights), this work precisely characterizes the class of monotone logical rules that can serve as sound explanations, constructs a restricted fragment of first-order logic to explain arbitrary MAGNN predictions, and empirically demonstrates that restricting to non-negative weights does not significantly hurt performance while enabling effective extraction of sound rules.

## Background & Motivation
**State of the Field**: GNNs are widely applied to knowledge graph completion (KGC), drug combination prediction, recommendation systems, and related tasks. To ensure trustworthy outputs, sound logical rules are required to explain GNN predictions—i.e., inferences produced by the rules must be a subset of the model's predictions.

**Limitations of Prior Work**: Existing work (Tena Cucala et al.) provides sound Datalog rules and equivalent programs for non-negative-weight GNNs with max and sum aggregation, but **mean aggregation**—the most commonly used default in practice (e.g., R-GCN)—lacks a corresponding theoretical analysis of explainability and expressive power.

**Root Cause**: Mean-GNNs are widely adopted for their simplicity and training stability, yet their non-monotonic behavior (adding more neighbors can decrease the aggregated value) prevents direct application of the sum/max-GNN theoretical framework and precludes the existence of equivalent first-order logic programs.

**Paper Goals**: Establish a theoretical foundation for sound logical rules for mean-aggregation GNNs and provide a practically applicable rule extraction method.

**Starting Point**: Restricting weights to be non-negative (consistent with prior sum/max-GNN work) isolates the source of non-monotonicity to the aggregation function itself, thereby simplifying the analysis.

**Core Idea**: MAGNNs can express logical functions beyond FOL, yet the class of sound monotone rules they admit is highly restricted—every sound ELUQ rule can be subsumed by a set of minimal rules, and these rules can be enumerated and verified within a finite search space.

## Method

### Overall Architecture
The work comprises two main theoretical contributions:
1. **Theorem 3**: Precisely characterizes which monotone rules (ELUQ rules) can be sound for MAGNNs—any sound ELUQ rule is subsumed by a set of simple rules of the form `∃P₁.⊤ ⊓ ... ⊓ ∃Pⱼ.⊤ ⊓ A₁ ⊓ ... ⊓ Aₖ ⊑ Aₖ₊₁`.
2. **Theorem 6**: Provides a sound explanatory rule for any MAGNN prediction, using a restricted FOL fragment Ω.

### Key Designs
1. **ELUQ Rule Language (Theorem 3)**:

    - ELUQ is defined as EL extended with union (⊔) and number restrictions (≥n), positioned between ALCQ and EL in the description logic hierarchy.
    - Core finding: For MAGNNs, any sound ELUQ rule degenerates to conjunctions containing only `∃P.⊤` (existence of a neighbor via some relation) and atomic concepts $A$.
    - Proof insight: Exploiting the property of mean aggregation—as the number of neighbors tends to infinity, the mean approaches the value of a single neighbor—so `≥n` degenerates to `∃`.
    - This implies the space of sound monotone rules is finite ($\delta \cdot 2^{\delta \cdot |\text{Col}|}$ rules) and can be exhaustively searched.

2. **Rule Soundness Verification (Proposition 4)**:

    - Given a simple rule, soundness can be determined by running the MAGNN on a minimal dataset $D_\text{base}$ and checking the output.
    - $D_\text{base}$ contains only the minimal facts required by the rule body, keeping verification cost negligible.
    - Relies on the monotonicity of MAGNN: any extension of $D_\text{base}$ cannot decrease the output for the target node.

3. **Explanatory Rules (Theorem 6, Ω Language)**:

    - Since MAGNNs cannot be equivalently represented in FOL (Proposition 1: determining "at least half the neighbors satisfy a condition" goes beyond FOL), the focus shifts to providing sound explanations.
    - The Ω language introduces a new operator `∃ₙP.(C₁,...,Cₙ)` (existence of $n$ distinct neighbors each satisfying different conditions) and `≤mP.⊤` (at most $m$ neighbors).
    - For a given prediction $A(a)$, a rule body $C_L^a$ is recursively constructed over the $L$-hop neighborhood of the $L$-layer GNN.
    - Guarantee: the rule both explains the prediction on the original dataset and is sound with respect to all datasets.

4. **Adaptation to Link Prediction**:

    - By combining with the encoding scheme of Tena Cucala et al., node-level rules are unrolled into link prediction rules.
    - In the link prediction setting, sound monotone rules reduce to the form `R₁(x,y) ∧ ... ∧ Rₘ₋₁(x,y) → Rₘ(x,y)`.

### Loss & Training
- Binary Cross Entropy loss with Adam optimizer (lr = 0.001)
- 8000 training epochs with early stopping at 50 epochs
- Non-negative weight constraint: negative weights are clamped to 0 after each optimization step
- 2-layer GNN, ReLU + Sigmoid activations, hidden dimension = 2 × input dimension
- Classification threshold selected via grid search on the validation set ($10^8$ candidates)

## Key Experimental Results

### Main Results

**Model Performance (Mean-GNN, Standard vs. Non-Negative Weights)**:

| Dataset | Weight Type | Acc(%) | Prec(%) | Rec(%) | F1(%) |
|--------|---------|--------|---------|--------|-------|
| LUBM | Standard | 97.1 | 96.9 | 97.2 | 97.1 |
| LUBM | Non-Neg | 91.5 | 87.8 | 96.4 | 91.9 |
| WN18RRv1 | Standard | 93.7 | 98.5 | 88.8 | 93.4 |
| WN18RRv1 | Non-Neg | **95.5** | 98.1 | 92.7 | **95.3** |
| FB237v1 | Standard | 68.7 | 95.4 | 39.3 | 55.7 |
| FB237v1 | Non-Neg | **71.8** | 75.4 | 64.8 | **69.7** |
| NELLv1 | Standard | 75.2 | 93.8 | 53.4 | 65.7 |
| NELLv1 | Non-Neg | **93.4** | 88.8 | 99.4 | **93.8** |

The non-negative weight constraint actually improves performance on several benchmarks (WN18RRv1, FB237v1, NELLv1).

### Ablation Study

**Number of Extracted Sound Monotone Rules**:

| Dataset | Total | Unary Only | Binary Only | Mixed |
|--------|------|-------|-------|------|
| LUBM | 11.6 | 1.4 | 9.8 | 0.4 |
| WN-hier | 22.6 | 15.6 | 0 | 7 |
| FB237v1 | 136 | 136 | 0 | 0 |
| WN18RRv1 | 0 | 0 | 0 | 0 |
| NELLv1 | 1 | 1 | 0 | 0 |

**LUBM Explanatory Rule Statistics**:
- Among 1,990 true positive predictions, only 22 cannot be explained by the form in Theorem 3.
- Unoptimized rules contain an average of 25 individual atoms; after optimization, the average reduces to 11.

### Key Findings
1. Restricting to non-negative weights has minimal impact on mean-GNN performance and even improves it on several benchmarks (WN18RRv1, FB237v1, NELLv1), possibly due to a regularization effect.
2. FB237v1 yields 136 sound rules, including 29 empty-body rules (e.g., `⊤ → R(x,y)`), exposing spurious reasoning learned by the model.
3. WN18RRv1 and NELLv1 yield virtually no sound monotone rules, confirming that the expressive power of mean-GNN with respect to monotone rules is indeed limited.
4. On LUBM, the model is found to have learned absurd rules such as `⊤ ⊑ Publication` (everything is a publication) and `GraduateStudent ⊑ ResearchAssistant` (all graduate students are research assistants), demonstrating the importance of explainability.
5. Compared to sum-GNN: sum-GNN can recover all monotone rules on WN-sym, whereas mean-GNN almost cannot, suggesting that sum/max aggregation may be preferable when provably sound rules are required.

## Highlights & Insights
- **Theoretical Depth**: Theorem 3 reveals that the space of sound monotone rules for MAGNNs is extremely restricted—this is the first characterization result for mean aggregation.
- **Practical Value**: The work provides an actionable rule verification pipeline (Proposition 4, verifiable with a single forward pass) and an explanation generation method.
- **Counterintuitive Finding**: Despite strong test-set performance, the extracted sound rules expose spurious reasoning learned by the model—demonstrating that accuracy does not imply trustworthiness.
- **Theory-Practice Connection**: While MAGNNs are theoretically proven to surpass FOL (Proposition 1), meaningful explanations can still be effectively extracted on real-world benchmarks.

## Limitations & Future Work
1. **Exponential Growth of Search Space**: The rule space $\delta \cdot 2^{\delta \cdot |\text{Col}|}$ grows exponentially with the number of relations; for FB237v1, the search must already be restricted to at most one individual atom.
2. **Is ELUQ the Maximal Monotone Fragment?** This remains an open question; there may exist sound monotone rules outside ELUQ.
3. **Choice of Classification Threshold**: Using $\geq$ versus $>$ yields entirely different theoretical results; only one case is handled in this paper.
4. **Explanatory Rules May Be Long**: Unoptimized rules average 25 individual atoms, limiting human readability.
5. Only 2-layer GNNs are considered; rule extraction for deeper models remains unexplored.

## Related Work & Insights
- **Tena Cucala et al. (2023/2024)**: Provide sound Datalog rules and equivalent programs for sum/max aggregation GNNs; the most direct predecessor of this work.
- **Morris et al. (NeurIPS 2023)**: Find that sum-GNNs often lack sound Datalog rules in practice, though all such rules are recoverable on WN-sym.
- **Barceló et al. (ICLR 2020)**: Prove that rules captured by GNNs can be expressed in ALCQ.
- **Schönherr & Lutz (2025, concurrent)**: Characterize the FOL expressive power of mean-GNNs (uniform/non-uniform settings), but do not provide practical rule extraction or explanation methods.
- Insight: Model explainability serves not only to explain individual predictions but, more importantly, to reveal model deficiencies—trustworthy AI requires sound guarantees rather than approximate explanations.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to establish a sound rule theory for the most widely used mean-aggregation GNNs.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple benchmarks, but dataset scale is limited and only 2-layer models are evaluated.
- Writing Quality: ⭐⭐⭐⭐ Theoretical exposition is rigorous and clear; proof sketches aid comprehension.
- Value: ⭐⭐⭐⭐ Fills an important theoretical gap; sound explanations carry significant implications for trustworthy AI.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism](shap_meets_tensor_networks_provably_tractable_explanations_with_parallelism.md)
- [\[NeurIPS 2025\] Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)
- [\[NeurIPS 2025\] Generating Physically Sound Designs from Text and a Set of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)

<!-- RELATED:END -->
