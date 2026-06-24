---
title: >-
  [Paper Note] Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models
description: >-
  [ACL 2025][LLM (Other)][Mechanistic Interpretability] By identifying circuits for 10 compositional string editing operations on the PCFG SET dataset, the modular relationships between functionally related circuits in Transformers are investigated. It is found that functionally similar circuits exhibit significant node overlap and cross-task fidelity, and that circuits can be combined via set operations (union) to represent more complex functions beyond the capability of indiv…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Mechanistic Interpretability"
  - "Circuit Discovery"
  - "Modular Structures"
  - "Continuous Sparsification"
  - "Subnetwork Composition"
date: 2026-05-08
content_hash: e9778c77333f9a35
---

# Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.01434](https://arxiv.org/abs/2410.01434)  
**Code**: [https://github.com/mainlp/circuit-compositions](https://github.com/mainlp/circuit-compositions)  
**Area**: LLM/NLP  
**Keywords**: Mechanistic Interpretability, Circuit Discovery, Modular Structures, Continuous Sparsification, Subnetwork Composition

## TL;DR
By identifying circuits for 10 compositional string editing operations on the PCFG SET dataset, the modular relationships between functionally related circuits in Transformers are investigated. It is found that functionally similar circuits exhibit significant node overlap and cross-task fidelity, and that circuits can be combined via set operations (union) to represent more complex functions beyond the capability of individual circuits.

## Background & Motivation

**Background**: Mechanistic interpretability identifies the minimal computational subgraphs, or "circuits," responsible for specific behaviors in language models using causal intervention analysis (activation patching). Prior studies have successfully identified circuits for tasks such as indirect object identification, factual recall, and arithmetic operations.

**Limitations of Prior Work**:
   - Most studies focus only on circuit identification for a single task, without exploring the relationships between functionally related circuits.
   - The few studies comparing circuits focus on tasks with limited functional similarity.
   - There is a lack of systematic research on the "compositionality" of circuits—whether circuits can be combined like Lego blocks to form new functions.

**Key Challenge**: Do neural networks implement reusable and modular functional subnetworks, and can these subnetworks be composed to perform more complex tasks?

**Key Insight**: The highly compositional PCFG SET string editing task suite (copy, echo, repeat, reverse, swap, etc.) is selected. These operations have clear functional relationships (e.g., repeat $\approx$ copy $\times$ 2), making them highly suitable for studying circuit modularity and compositionality.

**Core Idea**: Apply continuous sparsification to automatically discover circuits for 10 functionally related operations, analyze their node overlap and cross-task fidelity, and demonstrate that circuits can be composed via union operations to yield new functional capabilities.

## Method

### Overall Architecture
Train an encoder-decoder Transformer (~58M parameters) to perform all operations in PCFG SET $\rightarrow$ Identify circuits for each operation using **activation pruning + continuous sparsification** (binary mask $\mathbf{m} \in \{0,1\}^N$) $\rightarrow$ Evaluate circuit **fidelity, cross-task performance, and node overlap** $\rightarrow$ Compose circuits via **set operations** (union) and evaluate new capabilities.

### Key Designs

1. **Activation Pruning via Continuous Sparsification**:

    - Function: Automatically discover faithful and minimal circuits.
    - Mechanism: Formulate circuit discovery as an optimization problem that jointly minimizes fidelity loss and circuit size.
        - Reparameterize the binary mask using sigmoid: $\sigma(\beta \cdot \mathbf{s})$, where $\beta$ is gradually annealed towards the Heaviside step function.
        - Loss: $\mathcal{L}_{CE}(\text{full model}, \text{circuit output}) + \lambda \|\sigma(\beta \cdot \mathbf{s})\|_1$
    - Design Motivation: Avoid the combinatorial explosion issue of activation patching and achieve end-to-end optimization via continuous relaxation.
    - Validated on ground-truth circuits compiled with Tracr, recovering 100% of all relevant neurons.

2. **Cross-Task Fidelity Analysis**:

    - Evaluate the performance (fidelity $F_T$ and accuracy) of each circuit on other tasks.
    - Distinguish between "fidelity on all tokens" and "fidelity on differing tokens only", the latter being more precise.

3. **Circuit Composition (Set Operations)**:

    - Union of two circuits: $\mathbf{m}^{T_1, T_2} = \mathbf{m}^{T_1} \cup \mathbf{m}^{T_2}$
    - Use the ablation values of the first circuit to handle nodes outside the union.
    - Test whether the composed circuit acquires capabilities on new tasks that individual circuits do not possess.

## Key Experimental Results

### Circuit Performance (Fidelity $F_T$)

| Circuit | Target Task $F_T$ | Circuit Size (% Remaining Activations) |
|---------|------------------|----------------------------------------|
| copy    | >0.94            | ~8% (Minimal)                          |
| echo    | >0.94            | ~15%                                   |
| repeat  | >0.94            | ~20%                                   |
| reverse | >0.97            | ~22%                                   |
| prepend | >0.90            | ~39% (Maximal)                         |

All circuits achieve $F_T > 0.90$ on their respective target tasks.

### Circuit Composition Results (Accuracy, Single Circuit -> Composed Circuit)

| Composition | New Task | Individual Acc | Composed Acc |
|-------------|----------|----------------|--------------|
| repeat ∪ reverse | **echo** | 0% + 0% | **78%** |
| swap ∪ reverse | **shift** | 0% + 0% | **33%** |

Two circuits, neither of which can individually solve "echo", retrieve 78% accuracy on "echo" after composition!

### Node Overlap

| Circuit Pair | IoU | IoM |
|--------------|-----|-----|
| reverse-swap | 0.42 | ~0.60 |
| reverse-shift | 0.36 | ~0.60 |
| append-prepend | High | High |
| copy vs. others | Low IoU | High IoM (copy is enclosed within other circuits) |

### Key Findings
- **The copy circuit is the smallest** (~8% activations) and is almost entirely embedded within the circuits for echo, repeat, and swap, indicating that these operations reuse the copy functionality internally.
- **Functionally similar circuits show high node overlap**: The reverse-swap-shift trio yields IoU of 0.36-0.42 and IoM of ~0.60.
- **Circuit composition can generate new capabilities**: repeat ∪ reverse $\rightarrow$ echo (78%), which is a significant finding of "emergent compositionality".
- **The structural sparsity of the copy circuit reveals its mechanism**: Cross-attention in the decoder is almost entirely preserved, while FFNs and self-attention are mostly pruned—consistent with the intuition that copy tasks primarily rely on copying from the encoder.
- More complex operations (prepend, append) utilize larger circuits (39%), while simple operations (copy) use the minimal circuit.

## Highlights & Insights
- **Acquiring new capabilities through circuit composition** is the most exciting finding: two circuits (repeat and reverse) that individually cannot perform echo achieve an echo accuracy of 78% through a union operation. This strongly implies the existence of reusable and composable functional modules within neural networks.
- **The continuous sparsification method for circuit discovery** bypasses the combinatorial explosion of traditional activation patching and scales to larger models.
- The observation that **the copy circuit is nested within other circuits** supports the hypothesis that foundational operations are reused by complex operations, providing strong evidence of neural network modularity.
- Localized sparsity analysis (which modules in which layers are retained) delivers mechanistic insights.

## Limitations & Future Work
- The approach was only validated on a small encoder-decoder model (58M parameters); adaptability to larger models (e.g., GPT-2/LLaMA scale) remains unexplored.
- PCFG SET is a highly controlled synthetic task suite; circuits for natural language tasks are likely more complex.
- Circuit completeness is not guaranteed—some causally relevant nodes might be missed.
- Ablation values (mean vs. zero) significantly affect circuit identification results; different ablation choices yield different circuits.
- The union operation for circuit composition is asymmetric (utilizing the ablation values of the first circuit), which might introduce bias.

## Related Work & Insights
- **vs. Merullo et al. (2024)**: They studied the reuse of circuit components across tasks, but did not systematically compose circuits using set operations.
- **vs. Hanna et al. (2024b)**: They pointed out that circuit overlap alone should not determine functional relationships; this work monitors both fidelity and overlap simultaneously.
- **vs. Edge Pruning (Bhaskar et al.)**: Edge pruning focuses on edges rather than nodes; this work employs node-level continuous sparsification and achieves a 100% recovery rate validated by Tracr.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery that circuit composition generates new functions holds substantial theoretical significance and marks a milestone in understanding neural network modularity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation with Tracr validation, 10 operations, cross-task analysis, node overlap, and composition experiments, though constrained by model scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical formalization, rigorous experimental design, and rich visualizations.
- Value: ⭐⭐⭐⭐ Provides a substantial theoretical contribution to mechanistic interpretability and introduces new tools to explore neural network modularity and compositionality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Circuit Stability Characterizes Language Model Generalization](circuit_stability_characterizes_language_model_generalization.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](a_systematic_study_of_compositional_syntactic_transformer_language_models.md)
- [\[ACL 2025\] A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)

</div>

<!-- RELATED:END -->
