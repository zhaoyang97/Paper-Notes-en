---
title: >-
  [Paper Note] Data Verification is the Future of Quantum Computing Copilots
description: >-
  [AAAI 2026][Physics & Scientific Computing][quantum circuit optimization] This position paper argues that data verification must be elevated from a post-hoc filtering step to a foundational architectural principle in qua…
tags:
  - "AAAI 2026"
  - "Physics & Scientific Computing"
  - "quantum circuit optimization"
  - "LLM hallucination"
  - "formal verification"
  - "data quality"
  - "verification-first architecture"
date: 2026-05-08
content_hash: aa88e513c5ec27a5
---

# Data Verification is the Future of Quantum Computing Copilots

**Conference**: AAAI 2026
**arXiv**: [2602.04072](https://arxiv.org/abs/2602.04072)  
**Code**: None  
**Area**: Quantum Computing / AI4Science
**Keywords**: quantum circuit optimization, LLM hallucination, formal verification, data quality, verification-first architecture

## TL;DR
This position paper argues that data verification must be elevated from a post-hoc filtering step to a foundational architectural principle in quantum computing AI copilots. Three positions are advanced: (1) verified data is a minimum requirement; (2) prior constraints outperform posterior filtering; (3) scientific domains governed by physical laws require verification-aware architectures. Experiments demonstrate that LLMs trained without verified data achieve at most 79% accuracy on circuit optimization tasks.

## Background & Motivation

**Background**: Quantum program generation and compilation spans multiple abstraction layers—from high-level circuit descriptions to low-level gate arrangements—and demands zero error tolerance. LLMs have begun to be applied to quantum assistance (IBM Qiskit Code Assistant, AlphaTensor-Quantum achieving 37–47% T-gate reduction, AlphaEvolve, etc.), yet face fundamental limitations.

**Limitations of Prior Work**: (1) LLMs perform statistical pattern matching rather than formal reasoning, systematically failing at multi-step logical operations; (2) hallucination is mathematically unavoidable and cannot be resolved through scaling; (3) training data in the quantum domain is extremely scarce—the most comprehensive dataset, QDataSet, required three months of HPC cluster time to generate only 14 TB of data for 1–2 qubit systems.

**Key Challenge**: Correctness of quantum circuits is binary (correct/incorrect) and admits no approximation. The fraction of valid designs among all possible designs decreases exponentially with the number of qubits at $O((\delta/d)^n)$, rendering posterior filtering computationally infeasible.

**Goal**: To propose a verification-first architectural paradigm for quantum computing copilots and, more broadly, for AI4Science systems.

**Key Insight**: The Cuccaro Adder is used as a concrete case study. Through exhaustive search combined with Lean/Z3 formal verification, the ratio of valid designs to the total design space is quantified, and 34 LLMs are evaluated on this basis.

**Core Idea**: In scientific domains where correctness is a binary constraint, verification must be embedded in both the training data and the generation loop of AI systems, rather than applied as a post-hoc filter.

## Method

### Overall Architecture
This is a position paper rather than a methodology paper. The central argument unfolds through three positions (P1–P3), each supported by experiments or analysis.

### Key Designs

1. **P1: Verified Data is a Minimum Requirement**:

    - Argument: Training on unverified data is equivalent to implicit data poisoning. Unlike natural language tasks, quantum circuit constraints do not permit noise to be averaged out by large datasets; a single invalid gate sequence violates unitarity.
    - Experimental Support: 34 publicly available LLMs (270M–120B parameters) are evaluated, each answering 70,000 multiple-choice questions (10,000 per qubit count from 2 to 8). Verification-aware models (gemma3:12b, gpt-oss:120b) achieve 60–79% accuracy, while general-purpose models cluster near the 25% random baseline (21–29%).
    - Calibration Analysis: The GPT-OSS model assigns 60–80% confidence to correct answers, whereas Gemma3 assigns only 20–35%, indicating that verification-aware training improves not only accuracy but also uncertainty quantification.

2. **P2: Prior Constraints Outperform Posterior Filtering**:

    - Argument: The fraction of valid designs decreases exponentially as design components grow. With $d$ gate choices and $O(n)$ positions, the total space is $O(d^n)$; after applying constraints, valid designs number $O(\delta^n)$, yielding a valid fraction of $O((\delta/d)^n)$.
    - Experimental Support: Exhaustive search with Lean/Z3 verification over MAJ/UMA modules yields 540 valid MAJ designs and 529 valid UMA designs, combining to over 285,660 valid adders—out of a total design space exceeding 148 trillion, giving a valid fraction of approximately $10^{-9}$.
    - Design Implication: Verification should be embedded within the token generation loop, requiring each circuit fragment to satisfy SMT constraints or symbolic specifications before being committed.

3. **P3: Verification-Aware Architectures Apply to All Hard-Constraint Scientific Domains**:

    - Argument: The "leaky abstraction" and "non-decomposability" of quantum circuits are cross-disciplinary properties—local optima do not guarantee global optima.
    - Cuccaro Adder Case: A seemingly suboptimal UMA design (with more gates) may in fact be superior when unrolled into the full circuit due to parallelism; this cross-layer non-decomposability renders greedy approaches ineffective.
    - Generalization: Drug discovery (optimizing one property at the expense of others), combinatorial optimization (where greedy algorithms may yield the worst solution), and engineering design (with multidisciplinary coupling) share the same characteristics.

### Loss & Training
Evaluation uses a weighted scoring function: $\text{Score} = 0.5 \times \text{Toffoli}_n + 0.25 \times \text{Depth}_n + 0.25 \times \text{TotalGates}_n$. All four candidate circuits satisfy the adder specification but differ in optimization level; the correct answer is the circuit with the lowest score.

## Key Experimental Results

### Main Results
Performance of 34 LLMs on quantum circuit optimization multiple-choice questions (70,000 questions per model).

| Model Category | Representative Model | Accuracy | Confidence (Correct Answer) |
|---|---|---|---|
| Verification-aware | gpt-oss:120b | **~79%** | 60–80% |
| Verification-aware | gemma3:12b | ~60% | 20–35% |
| General-purpose LLMs | Most models | 21–29% | Near random |
| Format-failure models | mixtral:8x7b, deepseek-r1 | <25% | Malformed output |

### Design Space Analysis

| Metric | Value |
|---|---|
| Valid MAJ designs | 540 |
| Valid UMA designs | 529 |
| Total valid adders | 285,660+ |
| Total design space | >148 trillion |
| Valid fraction | ~$10^{-9}$ |
| Formal verification tools | Lean + Z3 |

### Key Findings
- **Scaling does not resolve the problem**: Models ranging from 270M to 120B parameters remain near the random baseline in the absence of verified training data.
- **The valid design space is extremely sparse**: A valid fraction of $10^{-9}$ renders posterior filtering computationally infeasible.
- **Output format failure is prevalent**: Multiple models (including deepseek-r1 variants) fall below the random baseline by producing verbose explanations rather than single-letter answers.
- Verification-aware training simultaneously improves accuracy and calibration—an indicator that the model genuinely understands rather than memorizes.

## Highlights & Insights
- **Quantification of the exponential scarcity of valid designs**: The $10^{-9}$ valid fraction provides a compelling argument applicable not only to quantum computing but to any combinatorial design problem with hard constraints. This concept transfers to AI system design in domains such as chip design and protein folding.
- **The "implicit data poisoning" analogy is instructive**: In domains where correctness is binary, unverified training data is not noise but poison—an observation that carries a cautionary message for all AI4Science applications.
- **Formalization of leaky abstraction**: The non-decomposability condition $f(g_1, \ldots, g_m) \neq \sum f_k(g_k)$ offers a concise abstraction of a problem common across disciplines.

## Limitations & Future Work
- As a position paper, experiments are limited to the specific case of the quantum adder; verification complexity for other quantum algorithms (QFT, Grover, etc.) may differ.
- Only multiple-choice evaluation is conducted; open-ended circuit generation by LLMs is not assessed.
- The proposed architecture of embedding verification within the generation loop has no concrete implementation; it remains primarily a future direction.
- Generating verified data is itself extremely costly (QDataSet required three months of HPC time), and scaling verified data production constitutes a core bottleneck.
- The engineering challenges of hybrid LLM + formal verifier architectures (e.g., latency, computational cost) are not discussed.

## Related Work & Insights
- **vs. AlphaTensor-Quantum**: AlphaTensor embeds correctness into the search objective, achieving 37–47% T-gate reduction, corroborating P2 (prior constraints outperform posterior filtering).
- **vs. AlphaEvolve**: Filtering variants through machine-checkable specifications represents a successful instance of the verification-first paradigm.
- **vs. QCircuitNet**: Pairing tasks with executable verification oracles embodies P1 (verified data is a minimum requirement).

## Rating
- Novelty: ⭐⭐⭐⭐ Elevates formal verification from software engineering to the AI architectural level; the argument is persuasive in the quantum computing context.
- Experimental Thoroughness: ⭐⭐⭐ The evaluation across 34 models is reasonably broad, but is confined to the adder case and lacks an actual implementation of a verification-first architecture.
- Writing Quality: ⭐⭐⭐⭐⭐ The position paper is clearly written, with three positions developing progressively; the choice of case study is well-suited.
- Value: ⭐⭐⭐⭐ Offers important directional guidance for the AI4Science community; the verification-first paradigm is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](../../NeurIPS2025/physics/quantum_doubly_stochastic_transformers.md)
- [\[ICLR 2026\] Feedback-driven Recurrent Quantum Neural Network Universality](../../ICLR2026/physics/feedback-driven_recurrent_quantum_neural_network_universality.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/physics/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

</div>

<!-- RELATED:END -->
