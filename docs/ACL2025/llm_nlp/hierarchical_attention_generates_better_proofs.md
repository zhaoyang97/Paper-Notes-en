---
title: >-
  [Paper Note] Hierarchical Attention Generates Better Proofs
description: >-
  [ACL 2025][LLM (Other)][Formal Theorem Proving] This paper proposes a Hierarchical Attention regularization method. By establishing a five-layer semantic hierarchy to guide the attention mechanism of LLMs, the approach aligns it with the natural information flow of mathematical reasoning. It improves proof success rates on miniF2F and ProofNet by 2.05% and 1.69%, respectively, while reducing proof complexity by 23.81% and 16.50%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Formal Theorem Proving"
  - "Hierarchical Attention"
  - "Lean"
  - "Mathematical Reasoning"
  - "Regularization"
date: 2026-05-08
content_hash: 7bf38aaa15b00dae
---

# Hierarchical Attention Generates Better Proofs

**Conference**: ACL 2025  
**arXiv**: [2504.19188](https://arxiv.org/abs/2504.19188)  
**Code**: [https://github.com/Car-pe/HAGBP](https://github.com/Car-pe/HAGBP)  
**Area**: Others  
**Keywords**: Formal Theorem Proving, Hierarchical Attention, Lean, Mathematical Reasoning, Regularization

## TL;DR

This paper proposes a Hierarchical Attention regularization method. By establishing a five-layer semantic hierarchy to guide the attention mechanism of LLMs, the approach aligns it with the natural information flow of mathematical reasoning. It improves proof success rates on miniF2F and ProofNet by 2.05% and 1.69%, respectively, while reducing proof complexity by 23.81% and 16.50%.

## Background & Motivation

Formal theorem proving is an important research direction at the intersection of AI and mathematics. Proof assistants such as Lean, Coq, and Isabelle have become key platforms for exploring this direction. Although LLMs have shown potential in generating proofs, they still face core challenges:

**Gap between Sequence Processing and Structural Reasoning**: LLMs fundamentally process flat token sequences and lack an explicit understanding of formal semantics. However, mathematical proofs possess an inherent hierarchical structure, with dependencies and compositional relationships among concepts.

**Blind Spots of the Attention Mechanism**: The attention mechanism of the standard Transformer allows free interaction among arbitrary tokens, which fails to reflect the natural partial order relationship of "low-level concepts supporting high-level theorems" in mathematical reasoning.

**Proof Complexity Issues**: LLMs frequently generate unnecessarily verbose proofs or fail completely on difficult problems.

The core thesis is that mathematical reasoning follows a natural hierarchical information flow, and this structural prior should be injected into the model's attention patterns.

## Method

### Overall Architecture

The proposed method consists of two steps: first, extracting the hierarchical flow pattern from the input (Extract Flow Pattern), and then guiding the model's attention to adhere to these hierarchical constraints using a specialized loss function (Hierarchical Attention Loss).

### Key Designs

1. **Five-Layer Semantic Hierarchy**: Exploiting the strong typing characteristics of the Lean language, tokens are classified into five natural levels:

    - **Level 0 (Context Level)**: Contains background information, auxiliary concepts, and general knowledge.
    - **Level 1 (Case Level)**: Pattern matching and case analysis.
    - **Level 2 (Type Level)**: Type declarations and definitions.
    - **Level 3 (Instance Level)**: Instance declarations and concrete examples.
    - **Level 4 (Goal Level)**: Core theorems or propositions (the highest level).
   
   These levels follow a partial order relationship: $context \prec case \prec type \prec instance \prec goal$.

2. **Three Types of Information Flows**: Information flows are defined based on the hierarchical relationship between two tokens $t_i, t_j$:

    - **Unrestricted**: $level(t_i) = level(t_j)$, allowing free interaction between tokens of the same level.
    - **Guided**: $level(t_i) < level(t_j)$, where low-level tokens pass information to high-level ones (encouraged).
    - **Limited**: $level(t_i) > level(t_j)$, where high-level tokens pass information to low-level ones (penalized).
   
   Core Idea: High-level concepts (such as Goal) should retrieve information from low-level concepts (such as Context and Type) to construct reasoning, whereas the reverse direction should be restricted.

3. **Hierarchical Attention Mask**: A binary mask $M_{ij}$ is constructed, where $M_{ij} = 1$ (allowed) when $level(t_i) \leq level(t_j)$, and $M_{ij} = 0$ (restricted) otherwise.

4. **Layer-wise Adaptation Factor**: $\alpha_l = 1 - l/L$, which enforces stricter hierarchical constraints in shallower Transformer layers while allowing more flexible attention patterns in deeper layers. This reflects the intuition that shallower layers are responsible for encoding basic structure, whereas deeper layers require complex cross-level reasoning.

### Loss & Training

Flow Loss penalizes attention patterns that violate hierarchical constraints:

$$\mathcal{L}_{flow} = \frac{1}{|T|} \sum_{l=1}^{L} \alpha_l \cdot \sum_{i,j} \text{ReLU}(att_l(t_i, t_j) \cdot (1 - M_{ij}))$$

The final training objective is the weighted sum of the standard cross-entropy loss and the Flow Loss:

$$\mathcal{L} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{flow}$$

where $\lambda$ controls the intensity of the hierarchical constraints. The base model is Pythia-2.8B, fine-tuned on LeanDojo Benchmark 4 for 3 epochs. The evaluation protocol includes two strategies: best-first search (BFS) and single-pass sampling (SPS).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (BFS, K=64) | Baseline (BFS, K=64) | Gain |
|--------|------|------|----------|------|
| miniF2F test | Pass Rate | **31.56%** | 29.51% | +2.05% |
| miniF2F valid | Pass Rate | **34.02%** | 31.56% | +2.46% |
| ProofNet test | Pass Rate | **15.25%** | 13.56% | +1.69% |
| ProofNet valid | Pass Rate | **11.86%** | 10.17% | +1.69% |
| miniF2F test | Proof Complexity Ratio $R_{avg}$ | **0.76** | 1.00 | Reduced by 23.81% |
| ProofNet test | Proof Complexity Ratio $R_{avg}$ | **0.84** | 1.00 | Reduced by 16.50% |

Single-pass sampling improvements are even more significant: miniF2F test increases from 23.36% to 27.87% (+4.51%), and miniF2F valid increases from 21.72% to 26.64% (+4.92%).

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Attention Distribution: Constrained vs. Unconstrained Layers | Limited flow from 5.5-27.8% to ≈0% | Constraints successfully enforced |
| Limited flow in Unconstrained Layers | Only 0.5-3.2% | Model has internalized the hierarchical structure |
| Guided flow in Goal Level | 68.7% in constrained layers, 84.5% in unconstrained layers | High-level effectively integrates low-level information |
| Guided flow in Type/Instance Levels | 77.7%/71.0% (constrained layers) | Intermediate levels also maintain hierarchical propagation |

### Key Findings

- The BFS strategy consistently outperforms SPS, and the proposed method achieved further improvements upon BFS.
- Performance gains are more pronounced when the computational budget $K \geq 16$, indicating that the method is more advantageous given sufficient search resources.
- Attention pattern analysis reveals an important phenomenon: even in deep layers where $\alpha_l = 0$ (no constraint), the attention patterns of the model still spontaneously maintain a hierarchical structure, indicating that the model has internalized the hierarchical patterns of mathematical reasoning.
- The reduction in proof complexity (23.81%) is more substantial than the increase in the pass rate (2.05%), indicating that the core advantage of the proposed method lies in generating more concise proofs.

## Highlights & Insights

- **Deep Core Insight**: Explicitly aligning the hierarchical nature of mathematical reasoning with the Transformer attention mechanism is an elegant way of injecting structural priors.
- **Lightweight Design**: It merely adds a regularization term without modifying the model architecture, making it easy to integrate with other approaches.
- **Ingenious Design of the Layer-wise Adaptation Factor**: $\alpha_l = 1 - l/L$ constrains structure in shallower layers and releases flexibility in deeper layers, which aligns with the hierarchical representation characteristics of Transformers.
- **Significance of the "Internalization" Phenomenon**: The discovery that unconstrained layers spontaneously maintain hierarchical structures indicates that appropriate training signals can enable models to genuinely learn structured reasoning, rather than merely passively adhering to constraints.

## Limitations & Future Work

1. The hierarchy definition is tailored to the semantics of the Lean language, which requires re-adaptation when transferring to other proof assistant languages like Coq or Isabelle.
2. The fixed five-layer hierarchical structure might limit dynamic reasoning patterns—some complex proofs may require flexible cross-level information flows.
3. Data limitations prevented evaluation on more advanced models (such as DeepSeek-Prover, InternLM-Math).
4. Hierarchical parsing relies on string pattern matching, which might not be sufficiently robust for complex nested structures.
5. Future work can explore adaptive hierarchical structures and cross-domain generalization capabilities.

## Related Work & Insights

- Compared to prior efforts that parse mathematical formulas into trees or graphs (Wang et al. 2017; Paliwal et al. 2020), this work does not rely on hand-crafted rules or programmatically generated data; instead, it allows the model to learn the structure via soft constraints.
- The baseline method LLMSTEP (Welleck & Saha, 2023) provides complete models, data, and hyperparameters, ensuring the reproducibility of the comparative analysis.
- This work provides a generalizable approach for injecting domain-specific structural priors into LLMs. Similar methodologies can be applied to other structured tasks such as code generation and logical reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The application of hierarchical attention regularization to theorem proving is novel, and the core insight is deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two benchmarks (miniF2F and ProofNet), two strategies (BFS/SPS), with sufficient attention visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, beautiful framework diagrams, and concise yet rigorous mathematical expressions.
- Value: ⭐⭐⭐⭐ The method is lightweight and generalizable, but the absolute performance gain is limited, requiring validation on a larger scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](../../NeurIPS2025/llm_nlp/solving_inequality_proofs_with_large_language_models.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)

</div>

<!-- RELATED:END -->
