---
title: >-
  [Paper Note] Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing
description: >-
  [ICML 2025][Knowledge Editing][Representation Shattering] Through synthetic experiments training Transformers on cyclic knowledge graphs, this work reveals that knowledge editing (KE) "shatters" the learned geometric representation manifolds inside the model. The degree of shattering is positively correlated with edit distance ($r^2=0.905$). Based on this, "representation shattering" is proposed as a mechanistic hypothesis for how KE impairs model capabilities…
tags:
  - "ICML 2025"
  - "Knowledge Editing"
  - "Representation Shattering"
  - "Transformers"
  - "Knowledge Graphs"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: f4a664dd99a8547b
---

# Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing

**Conference**: ICML 2025  
**arXiv**: [2410.17194](https://arxiv.org/abs/2410.17194)  
**Code**: None  
**Area**: Knowledge Editing  
**Keywords**: knowledge editing, Representation Shattering, Transformers, Knowledge Graphs, mechanistic interpretability

## TL;DR

Through synthetic experiments training Transformers on cyclic knowledge graphs, this work reveals that knowledge editing (KE) "shatters" the learned geometric representation manifolds inside the model. The degree of shattering is positively correlated with edit distance ($r^2=0.905$). Based on this, "representation shattering" is proposed as a mechanistic hypothesis for how KE impairs model capabilities, and the universality of this phenomenon is validated on Llama 3 and Mamba.

## Background & Motivation

**Background**: Knowledge Editing (KE) aims to precisely modify specific factual associations in LLMs while keeping other knowledge intact. Methods like ROME and MEMIT achieve this by locating knowledge inside specific MLP layers and performing closed-form updates. However, recent works (Cohen et al., 2023; Gupta et al., 2024; Gu et al., 2024) have found that KE not only impacts target facts but also impairs the broader factual recall and reasoning abilities of the model.

**Limitations of Prior Work**: Although extensive empirical evidence exists showing the harmfulness of KE, a mechanistic understanding of how editing alters the model's internal representations—and subsequently leads to widespread capability degradation—remains lacking. Analyzing the internal representations of large-scale LLMs directly is too complex to yield precise hypotheses.

**Key Challenge**: The "locate-and-edit" paradigm of KE assumes that knowledge is stored locally. In reality, however, models compress unrelated facts into overlapping subspaces through parameter sharing and superposition. Consequently, local edits can have global repercussions.

**Ours**: To design a controllable synthetic task—training a Transformer on a cyclic knowledge graph so that model representations precisely encode the global topology of the graph. Knowledge editing is then applied to observe how representations are destroyed, thereby establishing the "representation shattering" hypothesis.

**Key Insight**: Following the methodology of "synthetic task $\rightarrow$ precise hypothesis $\rightarrow$ real-world validation" (reminiscent of works like Allen-Zhu & Li), a causal understanding is established through a simplified but interpretable setting, which is then generalized to real LLMs.

## Method

### Overall Architecture

1. **Synthetic Data Construction**: 2048 entities are defined and arranged in 3 random cyclic orders (cyclic order I/II/III). Each order generates 8 relations (1-4 hop clockwise/counter-clockwise neighbors), totaling 24 relations.
2. **Data Generation**: Random walks are performed on the knowledge graph to generate alternating "entity-relation-entity-relation-..." sequences as training data.
3. **Model Training**: A 2-layer nanoGPT Transformer is used for next-token prediction.
4. **Evaluation and Editing**: The effects of corrective editing and counterfactual editing on direct recall, logical inference, and compositional inference are evaluated respectively.

### Key Designs

1. **Motivation and Structure of Cyclic Knowledge Graphs**:

    - Each relation subgraph is a set of disjoint cyclic graphs, corresponding to relations like "clockwise k-hop neighbors" (e.g., "I_C2" denotes the clockwise 2-hop neighbor in cyclic order I).
    - The 3 cyclic orders are used as edit (edit target), retain (retention set), and test (test set) subgraphs, respectively.
    - Motivation for choosing cyclic topology: In real LLMs, concepts such as months and weekdays are arranged cyclically in the representation space (e.g., in Llama-3.1-405B); cyclic structures are a common pattern for entity relations in natural language.
    - **Edit distance** is defined as the shortest distance between the old entity and the new entity in the cyclic order.

2. **Representation Shattering Metric**:

    - Define the metric $R(D_*)$ to quantify the degree of distortion in representations after editing:

$$R(D_*) = \frac{\|D_* - D_\varnothing\|_F}{\|D_\varnothing\|_F}$$

   where $D_\varnothing$ is the pairwise distance matrix between entities of the unedited model, $D_*$ is the distance matrix under the edited model, and $\|\cdot\|_F$ denotes the Frobenius norm.
    - This metric is permutation-sensitive: $R=0$ means every entity token remains in its original position. Even if the geometric structure is isomorphic (e.g., swapping two entities), it will yield a non-zero value.
    - Design Motivation: A quantitative metric is needed to correlate "the level of representation destruction" with "the degree of performance degradation."

3. **Three Evaluation Tasks**:

    - **Direct Recall**: Evaluates whether facts seen during training are still correctly recalled.
    - **Logical Inference**: Evaluates hold-out relations that can be inferred from other relations (e.g., 1-hop counter-clockwise can be inferred from 1-hop clockwise).
    - **Compositional Inference**: Evaluates the composition of two relations (which requires the model to preserve the geometric structure to generalize).
    - Evaluated using the average softmax probability over 5 random context sequences.

### Loss & Training

- Training uses the standard next-token prediction cross-entropy loss.
- Knowledge editing is performed using the ROME method (rank-one model editing), which applies a rank-one update to the weight matrices of MLP layers.
- Other methods like MEMIT, PMET, and AlphaEdit were also tested, yielding consistent conclusions.

## Key Experimental Results

### Main Results

| Evaluation Type | Pre-edit Accuracy (Cyclic I/II/III) | Corrective Edit ΔAcc | Counterfactual Edit (d=1) ΔAcc | Counterfactual Edit (d=4) ΔAcc |
|---------|---------|------|----------|------|
| Direct Recall | 98.3 / 93.7 / 99.4 | -21.95 | -1.49 | -77.94 |
| Logical Inference | 98.2 / 94.0 / 99.4 | -22.24 | -1.44 | -78.02 |
| Compositional Inference | 88.2 / 79.3 / 93.5 | -29.60 | -5.32 | -80.63 |

### Ablation Study

| Counterfactual Edit Distance d | R(D*) Edit Subgraph | R(D*) Retain Subgraph | R(D*) Test Subgraph |
|------|---------|------|------|
| d=1 | 1.80 | 1.80 | 1.84 |
| d=2 | 21.93 | 20.84 | 21.89 |
| d=3 | 26.22 | 25.32 | 26.52 |
| d=4 | 27.90 | 27.28 | 28.68 |

### Key Findings

1. **Strong Correlation between Shattering and Performance**: The correlation coefficient $r^2=0.905$ between $R(D_*)$ and the drop in accuracy indicates that more severe shattering leads to larger performance declines.
2. **Edit Distance is a Critical Factor**: Counterfactual editing with distance d=1 is nearly harmless (ΔAcc ~ -1.5%), whereas d=4 causes catastrophic degradation (ΔAcc ~ -78%). Intuitive analogy: Editing "December" to "November" is far safer than editing it to "July".
3. **Corrective Edits are also Harmful**: Correcting errors that the model learned incorrectly during training paradoxically degrades all performance metrics (ΔAcc ~ -22%), contrary to intuition.
4. **Global Impact**: Editing a single fact not only affects relations in the edit subgraph but also inflects identical levels of damage to the retain and test subgraphs (which contain relations unrelated to the edit).
5. **Models Learn Data Geometry**: Isomap visualization shows that the Transformer's internal representations precisely reflect the cyclic topology (see Fig. 4a).
6. **LLM Validation**: Applying MEMIT to Llama 3 8B Instruct for counterfactual edits of month sequences also reveals that (a) MMLU-Redux reasoning accuracy decreases with edit distance, and (b) the cyclic representation structure of months is progressively disrupted (see Fig. 7).

## Highlights & Insights

1. **Elegant Synthetic Experimental Design**: The cyclic knowledge graph simultaneously satisfies the requirements of being "structured" (having a global topology) and "controllable" (allowing the edit distance to be precisely defined), which facilitates causal analysis.
2. **Explanatory Power of the "Representation Shattering" Hypothesis**: It not only explains "why KE degrades performance" but also predicts "what kinds of edits are more harmful" (greater distance $\rightarrow$ more severe shattering $\rightarrow$ greater degradation), formulating a falsifiable hypothesis.
3. **From Synthetic to Real-World Validation Pathway**: By utilizing the cyclic representation of month/weekday concepts in LLMs (discovered by Engels et al., 2024) as a natural validation scenario, the hypothesis is successfully verified across various model scales.
4. **Fundamental Questioning of the "Locate-and-Edit" Paradigm**: It reveals that the vulnerability of KE stems from the entangled, compressed nature of factual storage, rather than the difficulty of the knowledge preservation task itself.

## Limitations & Future Work

1. **Simplicity of the Synthetic Task**: A 2-layer nanoGPT with a 2048-entity cyclic graph is substantially simpler than real-world LLMs; more complex models may exhibit additional dynamics not captured by this framework.
2. **Restriction to Cyclic and Tree Geometries**: Knowledge structures in natural language are far more complex than simple cycles and trees (e.g., hierarchical structures, multi-relational intersections), requiring validation across more geometric topologies.
3. **Focus on Single Edits**: This work primarily analyzes the impact of single edits. Since real-world applications often require batch editing, the cumulative shattering effect of sequential edits warrants a more systematic study.
4. **Lack of Remedying Solutions**: The work only diagnoses the problem without proposing concrete methods to alleviate shattering. The authors suggest that alternative approaches like RAG, lifelong editing, or synthetic document fine-tuning might be more promising.
5. **Caution around Causal Claims**: Although the correlation between edit distance and the level of shattering provides evidence toward causality, a rigorous causal proof has yet to be established.

## Related Work & Insights

- **Relationship to ROME/MEMIT**: These methods assume that knowledge is localized in MLP layers and perform closed-form updates. This work exposes the fundamental limitation of this assumption—local updates lead to deformation of the global representation manifold.
- **Connection to Engels et al. (2024)**: That work discovered multi-dimensional structured representations in LLMs (e.g., the circular structure of months). The synthetic experiments in this work can be seen as reproducing and explaining this phenomenon in a controlled setting.
- **Insights for Future Directions of KE**:
    - **Representation-Preserving Constraints**: Explicitly constraining the representation manifold from being shattered during editing.
    - **"Edit-Distance-Aware" KE**: Adjusting editing strategies based on how much the edit affects the manifold.
    - **Retrieval-Augmented Alternatives to Editing**: Instead of modifying weights directly, updating information via external knowledge bases could be more robust.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ This work is the first to provide a mechanistic explanation for the destructiveness of KE from the perspective of representation geometry. The "representation shattering" hypothesis is original and compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ The synthetic experiments are finely crafted, and the LLM validation is convincing. However, experiments on more diverse knowledge graph geometries and larger synthetic models are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, the narrative logic from synthetic to real settings flows smoothly, and the visualizations are excellent.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for the KE domain, potentially driving a fundamental rethinking of the "locate-and-edit" paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](../../ACL2026/knowledge_editing/representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](../../ICLR2026/knowledge_editing/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [\[ICML 2025\] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs](wikibigedit_understanding_the_limits_of_lifelong_knowledge_editing_in_llms.md)
- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](../../ACL2025/knowledge_editing/sake_steering_activations_for_knowledge_editing.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](../../ACL2025/knowledge_editing/efficient_knowledge_editing.md)

</div>

<!-- RELATED:END -->
