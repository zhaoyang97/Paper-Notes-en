---
title: >-
  [Paper Note] Skill Path: Unveiling Language Skills from Circuit Graphs
description: >-
  [AAAI2026 Oral][Graph Learning][Mechanistic Interpretability] This paper proposes the concept of Skill Path and a three-step framework (Decomposition–Pruning–Causal Mediation) to extract linear paths corresponding to specific language skills from circuit graphs, and quantitatively validates two core conjectures: Stratification and Inclusiveness of skills.
tags:
  - "AAAI2026 Oral"
  - "Graph Learning"
  - "Mechanistic Interpretability"
  - "Circuit Discovery"
  - "Skill Path"
  - "Causal Mediation"
  - "Language Models"
date: 2026-05-08
content_hash: 993a078c7b6d041a
---

# Skill Path: Unveiling Language Skills from Circuit Graphs

**Conference**: AAAI2026 Oral  
**arXiv**: [2410.01334](https://arxiv.org/abs/2410.01334)  
**Code**: [GitHub](https://github.com/Zodiark-ch/Language-Skill-of-LLMs)  
**Area**: Causal Inference
**Keywords**: Mechanistic Interpretability, Circuit Discovery, Skill Path, Causal Mediation, Language Models

## TL;DR

This paper proposes the concept of Skill Path and a three-step framework (Decomposition–Pruning–Causal Mediation) to extract linear paths corresponding to specific language skills from circuit graphs, and quantitatively validates two core conjectures: Stratification and Inclusiveness of skills.

## Background & Motivation

Circuit Discovery is a central approach to understanding the internal mechanisms of language models, identifying faithful subgraphs by pruning low-contribution edges and nodes from a computation graph. However, existing circuit graphs face two critical limitations:

1. **Skill Entanglement**: Circuit graphs derived from real datasets inevitably contain effects from skills unrelated to the target skill (e.g., induction datasets may include paths associated with arithmetic or multi-choice skills).
2. **Atomic Ablation Problem**: Existing methods perform ablations at the edge or node level, ignoring causal dependencies between collaborating components; furthermore, the nonlinearity of MLPs prevents full linearization of the computation graph, making it difficult to isolate the effects of paths spanning multiple components.

These limitations prevent existing circuit graphs from precisely localizing the mechanism of a single target skill.

## Core Problem

How to extract fine-grained "Skill Paths" from coarse-grained circuit graphs — compact subcircuits composed of chains of linear components that precisely reflect the location of a target skill within the circuit, while excluding interference from other skills and noise.

## Method

### Overall Architecture: Three-Step Framework

#### Step 1: Decomposition

The core contribution is a fully lossless linear decomposition of the MLP layers in Transformers. Since MLP inputs include the residual stream $X$ and attention output $\text{attn}(X)$, the nonlinear activation prevents direct decomposition of $\text{mlp}(\text{attn}(X)+X)$. The authors introduce a **Compensation Component**:

$$\text{mlp}(\text{attn}(X)+X) = \text{mlp}(\text{attn}(X)) + \text{mlp}(X) + \text{Cps}(X)$$

Based on this, each Transformer layer is decomposed into 29 components (using GPT2-small as an example, 12 layers × 12 heads):

- $C^0$: Self (direct residual stream pass-through)
- $C^{1-12}$: Attention (12 attention heads)
- $C^{13}$: MLP (pure MLP path)
- $C^{14-25}$: Attention+MLP (attention passed through MLP, one per head)
- $C^{26-27}$: Compensation (captures nonlinear synergistic effects)
- $C^{28}$: Bias

Among these, $C^{1-25}$ are **Functional Components**, each expressible as $C = f(X) \cdot W$, where $f(X)$ is a nonlinear activation and $W$ is an input-independent parameter matrix (interpretable as trained "memory").

Key property: the decomposition is **lossless**, i.e., $LM_l(X) = \sum_{i=0}^{28} C^i$, exactly recovering the original model output.

#### Step 2: Pruning

Using the decomposed computation graph $\mathcal{G} = \{\mathcal{C}, \mathcal{E}\}$, existing pruning strategies (ACDC by default) are applied via interchange ablation to remove components with negligible effect on output KL divergence. To support subsequent causal analysis, rather than averaging over the entire dataset, a set of circuit graphs is generated per group of 10 samples.

#### Step 3: Post-pruning Causal Mediation

This is the critical step for eliminating confounding effects. Paths in the circuit graph are categorized into three effect types:

- **Skill Effect**: the target skill's contribution to the output (the focus of this paper)
- **Background Effect**: influence from other latent skills present in the input text (e.g., induction effects mixed into ICL samples)
- **Self Effect**: influence arising solely from the memory of the last token (analogous to a bigram model)

For each input sample $s$, two perturbed texts are constructed — $s_{\text{Bkg}}$ (background text) and $s_{\text{Self}}$ (self text) — and three circuit graphs $\mathcal{G}^*_{\text{Ori}}$, $\mathcal{G}^*_{\text{Bkg}}$, $\mathcal{G}^*_{\text{Self}}$ are obtained via pruning. The criterion for identifying skill paths is:

$$\text{Eff}_{\text{Skill}}(\text{Path}^i) = \frac{N_{\text{Path}^i \in \mathcal{G}^*_{\text{Ori}},\; \text{Path}^i \notin \mathcal{G}^*_{\text{Bkg}},\; \text{Path}^i \notin \mathcal{G}^*_{\text{Self}}}}{N_{\text{all}}}$$

That is, only paths appearing in the original circuit graph but absent from both the background and self-perturbed circuit graphs are identified as target skill paths. A threshold $\delta$ (optimally 0.6–0.7) is applied to form the skill graph $\mathcal{G}^S$.

### Three Skills Investigated

In order of increasing complexity:

1. **Previous Token Skill (PVT)**: Receiving information from the immediately preceding token.
2. **Induction Skill (IDT)**: Recognizing prefix-matching patterns and copying repeated token sequences ("A B ... A" → predict B).
3. **ICL Skill**: Recognizing and copying patterns from demonstrations for in-context reasoning.

## Key Experimental Results

### Path Ablation Validation

Upon removing skill paths, accuracy drops to near zero across all task types, confirming that the paths indeed encode the target skills:

| Ablation Type | PVT Acc. | IDT Acc. | ICL1 Acc. |
|--------------|----------|----------|-----------|
| Full circuit | 1.00 | 1.00 | 1.00 |
| Remove 50 random paths | 0.46 | 0.58 | 0.61 |
| Remove 500 random paths | 0.23 | 0.29 | 0.23 |
| Remove PVT skill paths | **0.01** | 0.08 | 0.01 |
| Remove IDT skill paths | 0.00 | **0.00** | 0.00 |
| Remove ICL skill paths | — | — | **0.00** |

### Stratification: Quantitative Results

- PVT key receivers are located in layers 1–2 (simplest skill at shallowest layers).
- IDT key receivers are located in layers 2–6 (intermediate complexity at middle layers).
- ICL key receivers are distributed across nearly all layers (most complex skill spans the full depth).

### Inclusiveness: Quantitative Results

Using path overlap ratio $\text{ovlp}(A,B)$, the inclusive relationships among skill paths are pronounced:

| Comparison | Circuit Graph Overlap | Skill Path Overlap |
|-----------|----------------------|--------------------|
| ovlp(IDT, PVT) | 0.19 | **0.74** |
| ovlp(ICL1, PVT) | 0.06 | **0.81** |
| ovlp(ICL1, IDT) | 0.17 | **0.63** |

Conventional circuit graphs reveal almost no inclusive structure, whereas skill paths clearly show that 74% of IDT edges are present in PVT, and 81% of ICL1 edges are present in PVT.

### Robustness Across Pruning Strategies

The framework is compatible with mainstream pruning methods including ACDC, E-pruning, EAP, DiscoGP, and Scrubbing. Across all combinations, the inclusiveness metric of skill paths consistently and significantly exceeds that of circuit graphs.

## Highlights & Insights

1. **Solid theoretical contribution**: A fully lossless linear decomposition of the Transformer computation graph is proposed, with a compensation component introduced to resolve MLP nonlinear coupling.
2. **Elegant causal reasoning**: By constructing background and self-perturbed texts and applying counterfactual intervention techniques, confounding effects are disentangled — a clean application of standard causal inference methods to mechanistic interpretability.
3. **First quantitative validation**: The hypotheses that "simpler skills reside in shallower layers and more complex skills in deeper layers" and "complex skills subsume simpler ones" are elevated from qualitative conjectures to quantitative evidence.
4. **Framework generality**: The three-step framework is decoupled from specific pruning strategies and can be used as a plug-and-play module with different circuit discovery methods.

## Limitations & Future Work

1. **Limited model scale**: Validation is restricted to GPT2-small (12 layers, 12 heads); applicability to larger models remains unexplored.
2. **Skill paths do not preserve output faithfulness**: Since they capture only the skill effect (excluding background and self effects), they cannot fully recover the original output, limiting the use of conventional circuit evaluation metrics.
3. **Handling of compensation components**: The assumption that all edges of the compensation component are always present is a simplification that may overlook certain interactions.
4. **Manual skill definition**: The three skills and their hierarchical relationships are derived from prior knowledge; the framework itself cannot automatically discover new skills.
5. **Extension to more skills**: Only three progressively complex skills are validated; more complex or parallel skill relationships remain unexplored.

## Related Work & Insights

| Dimension | Conventional Circuit Discovery | Ours: Skill Path |
|-----------|-------------------------------|-----------------|
| Granularity | Edge/node-level circuit graph | Path-level (linear component chains) |
| Objective | Maintain output faithfulness | Isolate specific skill mechanisms |
| Confound handling | None | Causal mediation removes background/self effects |
| MLP decomposition | Not fully linearized | Lossless linear decomposition via compensation component |
| Skill coverage | Local (instantiated for specific inputs) | Global (complete skill paths) |
| Inclusiveness detection | Low overlap (~0.17) | High overlap (~0.74) |

Key distinction from works such as IOI circuit analysis: existing methods discover circuit instances for specific input samples (e.g., a particular induction head), whereas this paper identifies complete skill paths that generalize across large numbers of samples.

**Implications and connections:**

- **Implications for model compression**: If complex skills are built upon simpler ones, pruning should prioritize preserving lower-level skill paths.
- **Complementarity with probing**: Probing investigates "whether a given layer encodes certain information," while Skill Path reveals "how information flows across layers to realize a skill."
- **Precise targets for model editing**: Skill paths provide more precise intervention targets than circuit graphs, with potential applications in knowledge editing and selective skill removal.
- **Interpretable scaling**: Whether Stratification and Inclusiveness hold in larger models is an important open question for understanding emergent abilities.

## Rating
- Novelty: ⭐⭐⭐⭐ (Both the Skill Path concept and the three-step framework are original contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablations, visualizations, and cross-strategy comparisons, though limited to GPT2-small)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, though notation density is high)
- Value: ⭐⭐⭐⭐ (First quantitative validation of skill stratification and inclusiveness; meaningful advancement for the mechanistic interpretability community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[ICLR 2026\] Topology Matters in RTL Circuit Representation Learning](../../ICLR2026/graph_learning/topology_matters_in_rtl_circuit_representation_learning.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[ICLR 2026\] Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models](../../ICLR2026/graph_learning/global-recent_semantic_reasoning_on_dynamic_text-attributed_graphs_with_large_la.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)

</div>

<!-- RELATED:END -->
