---
title: >-
  [Paper Note] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations
description: >-
  [ICCV 2025][Interpretability][Visual Circuit Discovery] This paper proposes Granular Concept Circuit (GCC), a method that automatically discovers fine-grained visual circuits encoding specific concepts in deep visual models by iteratively evaluating inter-neuron functional dependency (Neuron Sensitivity Score) and semantic consistency (Semantic Flow Score). GCC is the first method capable of discovering multiple concept-level circuits within a single query.
tags:
  - "ICCV 2025"
  - "Interpretability"
  - "Visual Circuit Discovery"
  - "Concept Representations"
  - "Neuron Connectivity"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: 5cb4f48c81ea8519
---

# Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations

**Conference**: ICCV 2025
**arXiv**: [2508.01728](https://arxiv.org/abs/2508.01728)  
**Code**: [https://github.com/daheekwon/GCC](https://github.com/daheekwon/GCC)  
**Area**: Interpretability
**Keywords**: Interpretability, Visual Circuit Discovery, Concept Representations, Neuron Connectivity, Mechanistic Interpretability

## TL;DR

This paper proposes Granular Concept Circuit (GCC), a method that automatically discovers fine-grained visual circuits encoding specific concepts in deep visual models by iteratively evaluating inter-neuron functional dependency (Neuron Sensitivity Score) and semantic consistency (Semantic Flow Score). GCC is the first method capable of discovering multiple concept-level circuits within a single query.

## Background & Motivation

Deep visual models form concept representations through hierarchical architectures—from low-level edges and textures to high-level objects and scenes. Understanding how these concepts are encoded within models is a core problem in explainable AI.

Limitations of existing methods:

**Single-neuron analysis** (NetDissect, CLIP-Dissect, etc.): Associates concepts with individual neurons, ignoring the distributed nature of representations—concepts are encoded collaboratively across multiple neurons and layers.

**VCC** (Visual Concept Connectome): Analyzes inter-layer connections using concept activation vectors (CAVs), but is misaligned with network structure and cannot precisely localize where concepts emerge.

**ADVC** (Rajaram et al.): Iteratively discovers circuits via gradient × activation cross-layer attribution, but constructs only a single unified circuit tied to class labels, lacking concept-level granularity.

4. All existing methods **do not support** decomposing model responses into multiple concept-level circuits—distinct concepts (e.g., sky, flag, clock) are conflated within a single circuit.

## Method

### Overall Architecture

GCC aims to discover multiple fine-grained concept circuits for a given query, each corresponding to a specific concept related to that query. The pipeline proceeds as follows: (1) extract root nodes → (2) evaluate cross-layer connections → (3) iteratively trace until no further connections are found → (4) repeat for all root nodes to obtain a complete set of circuits.

### Key Designs

1. **Neuron Sensitivity Score ($S_{NS}$)**: An intervention-based measure of functional dependency.

   Connection strength is quantified by muting a source neuron and observing the resulting change in the target neuron's activation:

   $\tilde{S}_{NS,c} = \max(0, f^{l+1}(a_c^l) - f^{l+1}(\hat{a}_c^l))$
   $S_{NS} = \frac{\tilde{S}_{NS}}{\sum \tilde{S}_{NS}}$

   where $\hat{a}_c^l$ denotes the layer-$l$ activation after zeroing out the $c$-th neuron. A high $S_{NS}$ indicates that the target neuron strongly depends on the source neuron. Positive clipping is applied to focus exclusively on positive correlations.

    - **Design Motivation**: Causal intervention captures true dependency more accurately than gradients.
    - A first-order approximation (per-neuron intervention) is used to avoid the $O(2^{|N|})$ cost of exhaustive combinatorial search.

2. **Semantic Flow Score ($S_{SF}$)**: A semantic consistency constraint.

   $S_{SF} = \frac{|\mathcal{S}_{src} \cap \mathcal{S}_{tgt}|}{|\mathcal{S}_{src}|}$

   where $\mathcal{S}_{src}$ and $\mathcal{S}_{tgt}$ are the top-$k$ highly activated sample sets for the source and target neurons, respectively. High overlap indicates that both neurons encode similar semantic information.

    - **Design Motivation**: A high $S_{NS}$ alone is insufficient—nonlinearities may produce spurious connections (functionally dependent but semantically unrelated); $S_{SF}$ filters out such false connections.

3. **Circuit Construction Algorithm**:

    - **Root node extraction**: Selects neurons whose activations rank in the top 1% across all samples.
    - **Connection criterion**: A connection is accepted when both $S_{NS} > \tau_{NS}$ and $S_{SF} > \tau_{SF}$; $\tau_{NS}$ is determined automatically via the Peak-over-Threshold (POT) method from extreme value theory; $\tau_{SF}$ uses the mean score across all nodes.
    - **Iterative expansion**: Newly added nodes serve as new starting points, extending the search to the next layer until no further qualifying connections are found.
    - **Efficient computation**: Previously computed source-node connections are reused, and recursive techniques avoid redundant computation.

### Loss & Training

GCC is a post-hoc analysis method that does not involve any training. It operates directly on pre-trained models (VGG19, ResNet50/101, MobileNetV3, ViT, etc.) using the ImageNet1K validation set as the reference sample pool.

## Key Experimental Results

### Main Results

**Faithfulness and Completeness Evaluation** (tested on 100 random ImageNet1K queries; logit changes are observed after ablating neurons inside/outside the circuit):

| Ablation Condition | ResNet50 | ResNet101 | VGG19 | MobileNetV3 | Mean Drop |
|---|---|---|---|---|---|
| Original (no ablation) | 17.17 | 17.46 | 20.94 | 17.34 | — |
| Random neuron ablation | 15.66 | 13.80 | 19.03 | 15.01 | ▼2.35 |
| Ablate neurons inside GCC | 6.41 | 6.18 | 12.93 | 12.95 | **▼8.60** |
| Ablate neurons outside GCC | 16.12 | 14.58 | 19.93 | 15.88 | ▼1.74 |

Ablating neurons inside GCC causes a substantial drop in logits (8.60), far exceeding random ablation (2.35); ablating neurons outside GCC has minimal impact (1.74), demonstrating that the discovered circuits are both faithful and complete.

### Ablation Study (User Study)

33 participants rated GCC on five dimensions (5-point scale):

| Evaluation Dimension | Mean Score |
|---|---|
| Query relevance: whether GCC relates to the query | 3.65/5 |
| Diversity: whether GCC captures a variety of concepts | 4.00/5 |
| Prototypicality: whether GCC represents commonalities across multiple queries | 4.45/5 |
| Connection plausibility: whether node–node and query–node connections are reasonable | >90% rated as plausible |
| Comparison with VCC: which captures more meaningful concepts | 70% preferred GCC |

Edge ablation/insertion experiments: edges are removed or added in order of $S_{NS}$ rank; removing high-ranked edges leads to rapid performance degradation, while adding them yields significant gains.

### Key Findings

- **First concept-level decomposition of visual circuits**: A single "scoreboard" image can be decomposed into 17 GCCs, each corresponding to a distinct concept such as sky background, flag, and clock.
- **Generalization across models and datasets**: Validated on both CNNs (VGG19, ResNet50/101, MobileNetV3) and Transformers (ViT).
- **Hierarchical concept evolution**: The "blue texture" concept in a peacock image progressively refines from "broad blue tones" in the first layer to "blue scales" → "structured blue patterns" → "decorative blue patterns".
- **Cross-category shared concept discovery**: The method identifies shared "radial" concepts across different categories (daisy and peacock) and "wheel" concepts across tanks, minibuses, and trucks.
- **Misclassification auditing**: The specific concept circuits responsible for misclassification can be localized and verified through stimulation/suppression.

## Highlights & Insights

- **Complementary dual-score design**: $S_{NS}$ captures functional dependency while $S_{SF}$ ensures semantic consistency—both are indispensable, as relying solely on $S_{NS}$ yields spurious connections.
- **Automatic thresholding via POT**: Eliminates manual hyperparameter tuning, enhancing practical usability.
- **Paradigm shift from "one circuit" to "multiple circuits"**: Prior methods (VCC, ADVC) construct a single unified circuit per query; GCC is the first to decompose it into multiple concept-specific circuits.
- **Balance between interpretability and practicality**: Application scenarios such as misclassification auditing and cross-category concept discovery offer tangible practical value.
- **Sankey diagram visualization**: Connection strength is conveyed via link width, and neuron semantics are illustrated through high-activation sample crops, yielding an intuitive and informative visualization scheme.

## Limitations & Future Work

- Connection evaluation relies on a first-order approximation via per-neuron intervention, which may miss higher-order interactions.
- Under strict thresholds, a single concept may be distributed across multiple circuit paths.
- Some high-$S_{NS}$ connections remain difficult to interpret in human-understandable language.
- Validation is limited to classification models; circuit discovery in generative models (GAN/Diffusion) merits future exploration.
- The reference dataset for semantic coverage (ImageNet1K validation set) may introduce bias, as activation samples for certain concepts may be insufficiently abundant.
- The approach could be extended to video models and larger-scale Transformers (e.g., ViT-L/H).

## Related Work & Insights

- Corresponding to mechanistic interpretability in NLP (Conmy et al.), GCC is the first to achieve fine-grained circuit discovery in the visual domain.
- CRP computes conditional relevance but is limited to pairwise layers; CRAFT back-propagates from the classifier layer and can only extract classification-relevant concepts—GCC propagates forward and discovers a broader range of concepts.
- Inspired by Hebbian learning and synaptic plasticity in neuroscience: functional connectivity combined with information flow preservation.
- GCC's forward circuit discovery can identify abstract features shared across categories, which is beyond the reach of methods that back-propagate from class logits (e.g., ADVC).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to achieve concept-level fine-grained circuit discovery in visual models; the dual-score design is original and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers quantitative faithfulness/completeness validation, user study, edge ablation, and evaluation across multiple models and datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Concepts are articulated clearly with rich visualizations; algorithm pseudocode is well-structured.
- **Value**: ⭐⭐⭐⭐ Provides a new tool for visual model interpretability; application scenarios such as misclassification auditing carry practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[CVPR 2025\] Towards Human-Understandable Multi-Dimensional Concept Discovery](../../CVPR2025/interpretability/towards_human-understandable_multi-dimensional_concept_discovery.md)
- [\[ICCV 2025\] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps](ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)
- [\[ACL 2025\] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](../../ACL2025/interpretability/separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](../../ICML2026/interpretability/all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)

</div>

<!-- RELATED:END -->
