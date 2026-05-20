---
title: >-
  [Paper Note] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates
description: >-
  [NeurIPS 2025][Image Restoration][circuit discovery] This paper systematically introduces AND, OR, and ADDER gates to decompose language model circuits…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "circuit discovery"
  - "mechanistic interpretability"
  - "logic gates"
  - "completeness"
  - "faithfulness"
date: 2026-05-08
content_hash: 64d02fbcbbb29800
---

# Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates

**Conference**: NeurIPS 2025
**arXiv**: [2505.10039](https://arxiv.org/abs/2505.10039)  
**Code**: None  
**Area**: LLM Interpretability (Note: original category image_restoration is incorrect)
**Keywords**: circuit discovery, mechanistic interpretability, logic gates, completeness, faithfulness

## TL;DR
This paper systematically introduces AND, OR, and ADDER gates to decompose language model circuits, reveals that circuit incompleteness primarily stems from the omission of OR gates, and proposes a framework combining noising and denoising interventions to fully recover all three gate types while guaranteeing both faithfulness and completeness.

## Background & Motivation

**Background**: Circuit discovery is a core methodology in mechanistic interpretability, aiming to understand internal model mechanisms by extracting subgraphs (circuits) from the Transformer computation graph that implement specific tasks. Mainstream approaches include greedy search (ACDC), linear estimation (EAP), and differentiable masking (EdgePruning).

**Limitations of Prior Work**: Existing circuit discovery methods lack completeness—discovered circuits vary significantly across random seeds, and key mechanisms are frequently missed. Theoretical analysis shows that incomplete circuits give rise to two failure modes: non-transitivity and preemption, which impede a genuine understanding of circuit mechanisms.

**Key Challenge**: Incompleteness primarily arises from the presence of OR gates. When two parallel paths can substitute for each other (OR relationship), noising-based methods need only discover one to ensure faithfulness, causing the other to be discarded. This "partial discovery" undermines completeness and makes results contingent on randomness.

**Goal**: (a) How to formally define logical relationships within circuits? (b) How to fully recover all logic gates? (c) How to simultaneously guarantee faithfulness and completeness?

**Key Insight**: The observation that noising methods recover AND and ADDER gates but miss OR gates, while denoising methods recover OR and ADDER gates but miss AND gates—making the two approaches complementary.

**Core Idea**: Decompose circuits into AND/OR/ADDER gates and fully recover all gates via the set relationships $\mathcal{C}_{Ns} \cap \mathcal{C}_{Dn} = \text{ADDER}$, $\mathcal{C}_{Ns} \setminus \mathcal{C}_{Dn} = \text{AND}$, $\mathcal{C}_{Dn} \setminus \mathcal{C}_{Ns} = \text{OR}$.

## Method

### Overall Architecture
**Input**: computation graph $\mathcal{G}$ of a language model and task $\mathcal{T}$. Circuit discovery proceeds in two steps: (1) apply a noising intervention strategy to obtain $\mathcal{C}_{Ns}$; (2) apply a denoising intervention strategy to obtain $\mathcal{C}_{Dn}$. Set operations then separate the three gate types. **Output**: a logically complete circuit containing all AND/OR/ADDER gates.

### Key Designs

1. **Formal Definition of Three Logic Gates**

    - **Function**: Classify the relationship between a receiving node $B$ and its sending nodes $A_1, A_2, \ldots$ into three categories.
    - **Mechanism**: **AND gate** $B = A_1 \wedge A_2 \wedge \ldots$ (B produces significant output only when all sending nodes are retained); **OR gate** $B = A_1 \vee A_2 \vee \ldots$ (B produces significant output when any single sending node is retained); **ADDER gate** $B = A_1 + A_2 + \ldots$ (each sending node contributes independently, with effects summing linearly).
    - **Design Motivation**: Prior work observed AND/OR relationships but did not formalize them. The introduction of the ADDER gate is a key innovation—it is neither "all-or-nothing" nor "sufficient with one," but rather linear superposition, and in practice constitutes the most important gate type in circuits (contributing the largest effect).

2. **Ns+Dn Joint Framework**

    - **Function**: Combine noising and denoising intervention strategies to fully recover all three gate types.
    - **Mechanism**: Run Ns and Dn separately to obtain two circuits, then separate gates via set operations: $\mathcal{C}_{AND} = \mathcal{C}_{Ns} \setminus \mathcal{C}_{Dn}$, $\mathcal{C}_{OR} = \mathcal{C}_{Dn} \setminus \mathcal{C}_{Ns}$, $\mathcal{C}_{ADDER} = \mathcal{C}_{Ns} \cap \mathcal{C}_{Dn}$. The framework integrates seamlessly into all three categories of existing circuit discovery methods (greedy search, linear estimation, differentiable masking), requiring only one additional Dn run with no change in computational complexity.
    - **Design Motivation**: Theoretical derivation shows this is the minimal solution that simultaneously satisfies faithfulness and completeness.

3. **Granularity Alignment (Misalignment Score)**

    - **Function**: Ensure that the granularity of the two intervention strategies (Ns and Dn) is consistent, preventing incorrect gate classification due to scale mismatches.
    - **Mechanism**: A misalignment score is defined to measure whether AND and OR gates behave under joint intervention as theoretically expected. If the perturbation magnitudes of Ns and Dn are mismatched, some edges may be misclassified. Perturbation parameters are adjusted to minimize the misalignment score.
    - **Design Motivation**: Ns and Dn are fundamentally interventions in opposite directions with different reference states; alignment is required before they can be correctly compared.

### Theoretical Corollaries
- **Corollary 1**: The minimal edge set for optimal faithfulness = all AND edges + all ADDER edges + one edge per OR gate; the minimal edge set for optimal completeness = all OR edges + all ADDER edges + one edge per AND gate.
- This explains why existing methods predominantly use the Ns strategy: Ns recovers AND + ADDER + partial OR, which corresponds precisely to the optimal faithfulness–sparsity trade-off.

## Key Experimental Results

### Main Results (IOI Task — GPT2-small)

| Method | Strategy | Faithfulness (KL↓) | Completeness (KL↑) | Sparsity |
|--------|----------|-------------------|-------------------|----------|
| ACDC | Ns | Good | Poor | Good |
| ACDC | Dn | Poor | Good | Good |
| ACDC | **Ns+Dn** | **Good** | **Good** | Medium |
| EAP | Ns | Poor | Poor | Good |
| EAP | **Ns+Dn** | **Medium** | **Good** | Medium |
| EdgePruning | Ns | Good | Poor | Good |
| EdgePruning | **Ns+Dn** | **Good** | **Good** | Medium |

### Ablation Study / Randomness Analysis (Hamming Distance, 30 Runs)

| Method | Sparsity | Ns Hamming | Ns+Dn Hamming | Note |
|--------|----------|-----------|--------------|------|
| ACDC | 500 edges | 14.3±4.3 | 5.7±2.8 | Randomness substantially reduced |
| ACDC | 2000 edges | 49.5±12.9 | 7.9±2.9 | Ns+Dn more stable |
| EdgePruning | 2000 edges | 55.7±14.9 | 8.6±3.5 | Randomness reduced by 85% |

### Key Findings
- **ADDER gates contribute most**: Across all experiments, the gate effect of ADDER gates is significantly larger than that of AND and OR gates, indicating that linear superposition is the dominant computational pattern in circuits.
- **OR gates connect functionally equivalent paths**: In the IOI circuit, OR gates connect almost exclusively components performing the same function (e.g., multiple name mover heads), confirming they serve as mutual backups.
- **AND gates connect functionally distinct paths**: AND gates typically connect components with different functions (e.g., positional attention heads in earlier layers feeding into name mover heads in later layers), indicating that these functions must operate in concert.
- **Linear estimation (EAP) fails entirely to discover OR gates**: Because EAP estimates the independent contribution of each edge via gradients, and OR-gate edges have near-zero gradient contributions due to redundancy.

## Highlights & Insights
- **Formalization of logic gates**: The paper translates the vague notions of "redundant paths" and "necessary paths" in circuits into precise AND/OR/ADDER definitions, and proves that their set relationships correspond to the minimal requirements for faithfulness and completeness. This framework-based approach is highly elegant.
- **Complementarity of Ns and Dn**: A simple yet profound insight—the two intervention strategies each recover different parts of the circuit (AND vs. OR), and combining them yields a complete circuit at only a constant-factor increase in computational cost.
- **Function–logic correspondence**: The finding that OR gates connect functionally identical paths while AND gates connect functionally distinct paths offers a new perspective for understanding redundancy mechanisms within models.

## Limitations & Future Work
- Only AND/OR/ADDER gates are considered; more complex logical relationships such as XOR are not addressed.
- Experiments are conducted primarily on GPT2-small and a limited set of tasks; scalability to larger models has not been thoroughly evaluated.
- Granularity alignment via the misalignment score still requires manual tuning; automated approaches warrant exploration.
- Gate classification relies on hard set-theoretic partitioning, and boundary ambiguities may arise in practice.

## Related Work & Insights
- **vs. ACDC (Conmy et al.)**: ACDC is representative of greedy search and uses only the Ns strategy, yielding faithful but incomplete circuits. This paper supplements it with Dn to simultaneously guarantee both properties.
- **vs. EAP (Syed et al.)**: EAP is based on linear estimation and is theoretically and empirically shown to be entirely incapable of discovering OR gates. This paper explains its failure and provides a remedy.
- **vs. Mueller et al.**: That work identifies the incompleteness problem but proposes brute-force combinatorial enumeration as a solution (an NP-hard problem). This paper resolves it in constant additional time via Ns+Dn set operations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The formalization of logic gates and the Ns/Dn complementarity constitute genuinely novel and profound insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments on the IOI task are thorough, though diversity of models and tasks could be strengthened.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear, with theoretical derivations and intuitive explanations given equal weight.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic solution to the completeness problem in circuit discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/image_restoration/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
