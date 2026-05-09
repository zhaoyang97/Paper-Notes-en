---
title: >-
  [Paper Note] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees
description: >-
  [ICLR 2026][mechanistic interpretability] This paper introduces neural network (NN) verification into mechanistic interpretability, proposing the first circuit discovery framework with provable guarantees: input robustness guarantees circuit faithfulness over continuous input domains, patching robustness guarantees circuit consistency over continuous patching domains, and a four-level minimality hierarchy (quasi → local → subset → cardinal) is formalized. A monotonicity theory unifies all three types of guarantees.
tags:
  - ICLR 2026
  - mechanistic interpretability
  - circuit discovery
  - neural network verification
  - provable guarantees
  - minimality
date: 2026-05-08
content_hash: 7842156acdba9c18
---

# Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees

**Conference**: ICLR 2026
**arXiv**: [2602.16823](https://arxiv.org/abs/2602.16823)
**Code**: None
**Area**: AI Safety / Interpretability
**Keywords**: mechanistic interpretability, circuit discovery, neural network verification, provable guarantees, minimality

## TL;DR
This paper introduces neural network (NN) verification into mechanistic interpretability, proposing the first circuit discovery framework with provable guarantees: input robustness guarantees circuit faithfulness over continuous input domains, patching robustness guarantees circuit consistency over continuous patching domains, and a four-level minimality hierarchy (quasi → local → subset → cardinal) is formalized. A monotonicity theory unifies all three types of guarantees.

## Background & Motivation
**State of the Field**: Circuit discovery in mechanistic interpretability (MI) aims to identify subgraphs (circuits) that drive specific model behaviors. Methods such as ACDC and edge attribution patching have made progress, but all rely on heuristics or sampling-based approximations.

**Limitations of Prior Work**: Evaluating circuit faithfulness via sampling is fundamentally flawed—even if a circuit is faithful at all sampled points, small input perturbations can break consistency. This is unacceptable in safety-critical settings. Furthermore, the notion of "minimality" remains loosely defined, and the minimality level achieved by different algorithms has not been formally discussed.

**Root Cause**: Circuit discovery simultaneously requires faithfulness guarantees over continuous domains (rather than discrete points), circuits that are as small as possible (for interpretability), and the choice of patching strategy (zero/mean/sampling) itself introduces additional uncertainty.

**Paper Goals**: (1) How to guarantee circuit faithfulness over continuous input domains? (2) How to guarantee circuit consistency over continuous patching domains? (3) How to formalize and pursue different levels of minimality?

**Starting Point**: Leveraging NN verification tools such as α-β-CROWN, the paper places the circuit and the original model side by side via Siamese encoding, reducing circuit faithfulness to verifiable input-output constraints.

**Core Idea**: Replace sampling with NN verifiers to prove circuit faithfulness over continuous domains, and unify input robustness, patching robustness, and minimality guarantees through a monotonicity theory.

## Method

### Overall Architecture
Three types of provable guarantees are defined: (1) **Input-robust circuit**—the circuit agrees with the original model over a continuous input neighborhood $\mathcal{Z}$; (2) **Patching-robust circuit**—the circuit remains faithful over a continuous patching domain; (3) **Minimality**—a four-level hierarchy: quasi < local < subset < cardinal. These guarantees are encoded as NN verification queries via Siamese encoding, and circuits are discovered using greedy, exhaustive, or MHS-based algorithms.

### Key Designs

1. **Siamese Encoding (Input Robustness Verification)**:

    - *Function*: The circuit $C$ and the full model $G$ are combined into a joint network $G \sqcup C'$ sharing a common input layer.
    - *Mechanism*: Activations of non-circuit components are fixed to the patching value $\alpha$; the input constraint $\psi_{in}$ restricts $z \in \mathcal{Z}$; the output constraint $\psi_{out}$ checks $\|f_C(z|\bar{C}=\alpha) - f_G(z)\|_p \leq \delta$. α-β-CROWN is then invoked for verification.
    - *Guarantee*: If verification passes, the circuit is 100% faithful over the entire continuous domain $\mathcal{Z}$.

2. **Patching Robustness Verification**:

    - *Function*: Ensures circuit faithfulness under any reachable patching value, rather than only zero or mean patching.
    - *Siamese Encoding Variant*: $G$ and $C'$ receive different inputs; non-circuit activations in $C'$ are connected to the corresponding activations of $G$ over $\mathcal{Z}'$.
    - Addresses the criticism that zero-patching produces out-of-distribution activations.

3. **Four-Level Minimality Hierarchy**:

    - **Quasi-minimal**: There exists some component whose removal renders the circuit unfaithful.
    - **Locally-minimal**: Every component is necessary (removing any single one yields an unfaithful circuit).
    - **Subset-minimal**: No proper subset forms a faithful circuit.
    - **Cardinally-minimal**: The globally smallest faithful circuit.
    - Key theory: **Monotonicity**—when the faithfulness predicate $\Phi$ is monotone (enlarging the circuit does not break faithfulness), a greedy algorithm is guaranteed to converge to a subset-minimal circuit. Proposition 5 proves that when $\mathcal{Z} \subseteq \mathcal{Z}'$ and the activation space is closed, predicates for input and patching robustness automatically satisfy monotonicity.

4. **MHS Duality for Cardinally-Minimal Circuit Discovery**:

    - *Function*: Approximates the globally minimal circuit via the minimum hitting set (MHS) duality between circuits and blocking sets.
    - *Mechanism* (Proposition 7): The minimum hitting set of all circuit blocking sets equals the cardinally-minimal circuit. Solved efficiently using a MaxSAT solver.

## Key Experimental Results

### Input Robustness Comparison

| Dataset | Method | Circuit Size | Robustness | Time (s) |
|--------|------|---------|--------|---------|
| CIFAR-10 | Sampling | 16.5 | **46.5%** | 0.23 |
| | **Provable** | **19.2** | **100%** | 2970 |
| MNIST | Sampling | 12.6 | **19.2%** | 0.31 |
| | **Provable** | **15.8** | **100%** | 612 |
| GTSRB | Sampling | 28.9 | **27.6%** | 0.11 |
| | **Provable** | **29.6** | **100%** | 991 |

### Patching Robustness Comparison

| Dataset | Zero Patching Rob.% | Mean Patching Rob.% | **Provable Rob.%** |
|--------|--------------------|--------------------|-------------------|
| CIFAR-10 | 46.4 | 33.3 | **100** |
| MNIST | 58.0 | 55.7 | **100** |
| TaxiNet | 57.1 | 63.3 | **100** |

### Key Findings
- Sampling-based methods achieve only 9.5–58% robustness across all datasets, whereas the provable method achieves 100%, at the cost of 100–10,000× longer computation time.
- Provable circuits are only marginally larger than sampled ones (+10–20%), indicating that robustness guarantees do not require substantially larger circuits.
- The MHS lower bound never exceeds the circuit size and matches exactly in some cases, validating the MHS duality.
- The quasi-minimal algorithm is the fastest but yields the largest circuits; the cardinally-minimal algorithm is the slowest but yields the smallest, consistent with theoretical predictions.

## Highlights & Insights
- **First integration of NN verification and MI**: Two rapidly developing but previously independent fields are connected. Verification tools provide formal guarantees for MI, while MI offers new application domains for verification tools.
- **Unifying role of monotonicity theory**: A single, elegant property ($\Phi$ being monotone) simultaneously guarantees convergence to subset-minimality and the composability of input and patching robustness.
- **Practical significance of the four-level minimality hierarchy**: Researchers can select an appropriate level of minimality according to their computational budget—from quasi (fastest) to cardinal (strongest)—rather than making vague claims of being "minimal."

## Limitations & Future Work
- **Scalability of NN verification**: Current experiments are conducted on vision models (MNIST/CIFAR-10 scale) and cannot be directly applied to Transformers or LLMs. The framework will scale automatically as verification tools improve.
- The computational cost of provable methods is 100–10,000× higher, making them impractical for real-time MI analysis.
- Validation is limited to vision models; circuit discovery for NLP/language models (e.g., the IOI circuit) is not addressed.
- Minimality guarantees depend on the choice of faithfulness predicate $\Phi$; different $\delta$ thresholds may yield different results.

## Related Work & Insights
- **vs. ACDC (Conmy et al., 2023)**: ACDC performs greedy search using sampling and KL-divergence thresholding, serving as the sampling baseline in this paper. The provable method improves robustness from ~40% to 100% at comparable circuit sizes.
- **vs. Adolfi et al. (2025)**: They propose quasi-minimality as a minimality criterion. This paper extends it to a four-level hierarchy, proving stronger guarantees and algorithmic convergence.
- **vs. AlphaSteer/ASIDE**: These methods manipulate behavior at the activation/embedding level, whereas this paper analyzes behavior at the circuit level. The null-space constraints in AlphaSteer can be viewed as a soft definition of a "circuit."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First integration of NN verification into MI; the four-level minimality hierarchy and monotonicity theory are original theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets × three types of guarantees × multiple algorithm comparisons, though limited to small-scale vision models.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear definition–proposition–algorithm structure, and intuitive Siamese encoding diagrams.
- Value: ⭐⭐⭐⭐⭐ Establishes a formal foundation for MI and represents a significant theoretical advance in circuit discovery.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/interpretability/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)

<!-- RELATED:END -->
