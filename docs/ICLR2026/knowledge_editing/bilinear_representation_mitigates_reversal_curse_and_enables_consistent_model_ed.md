---
title: >-
  [Paper Note] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing
description: >-
  [ICLR 2026][Knowledge Editing][reversal curse] By training Transformers from scratch on a synthetic relational knowledge graph, this work demonstrates that appropriate regularization induces the emergence of bilinear rel…
tags:
  - "ICLR 2026"
  - "Knowledge Editing"
  - "reversal curse"
  - "bilinear representation"
  - "model editing"
  - "relational structure"
  - "knowledge graph"
date: 2026-05-08
content_hash: eb5f8bb95f9397db
---

# Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing

**Conference**: ICLR 2026
**arXiv**: [2509.21993](https://arxiv.org/abs/2509.21993)  
**Code**: Available (GPT-NeoX framework)  
**Area**: LLM Reasoning / Knowledge Representation / Model Editing
**Keywords**: reversal curse, bilinear representation, model editing, relational structure, knowledge graph

## TL;DR
By training Transformers from scratch on a synthetic relational knowledge graph, this work demonstrates that appropriate regularization induces the emergence of bilinear relational structure in hidden representations. This structure not only overcomes the reversal curse but also enables logically consistent propagation of edits to related facts.

## Background & Motivation

### State of the Field
Language models exhibit strong performance on knowledge-intensive tasks, yet their reasoning often lacks logical consistency. A canonical example is the *reversal curse*: a model trained on "A is the father of B" fails to infer "B is the child of A." The field of model editing aims to update knowledge without retraining, but existing editing methods cannot propagate updates to logically entailed facts.

### Limitations of Prior Work

**The reversal curse is treated as a fundamental limitation**: The prevailing view attributes it to the directionality of the autoregressive training objective, i.e., the model only learns $P(B|A)$ rather than $P(A|B)$.

**Model editing cannot generalize logically**: After editing "A's spouse is C→D," the model cannot automatically infer "D's spouse is A," requiring explicit bidirectional editing.

**Existing solutions address symptoms rather than causes**: Data augmentation (generating reverse samples) or modifying the training objective are superficial fixes.

### Root Cause
Are these logical failures an inherent architectural deficiency of Transformers, or a product of how models represent knowledge?

### Paper Goals
1. Can the reversal curse be overcome through appropriate training?
2. What mathematical structure does the model use internally to encode relational knowledge?
3. How does this structure affect the logical consistency of model editing?

### Starting Point
The authors approach the problem from the perspective of the geometric structure of knowledge representations. Observing that bilinear models in knowledge graph embedding methods (e.g., RESCAL) naturally support relational inverses (matrix transpose) and compositions (matrix multiplication), they hypothesize that if a Transformer learns a bilinear relational structure, it can overcome the reversal curse and achieve editing generalization.

### Core Idea
The reversal curse and model editing failures are not inherent deficiencies of Transformers, but rather stem from the absence of proper knowledge representation structure. When models learn bilinear relational structure, these problems are naturally resolved.

## Method

### Overall Architecture
- **Input**: Synthetic family-relation knowledge graph (1,000 families, 10 members each, 8 relation types)
- **Training**: GPT-NeoX architecture trained from scratch (12 layers, 896-dim hidden states, 16 attention heads, ~206M parameters)
- **Key control variable**: Weight decay strength (0–6.0)
- **Evaluation**: Three probes (linear, translational, bilinear) to analyze the structure of hidden representations
- **Output**: Causal relationship between bilinear structure, overcoming the reversal curse, and model editing generalization

### Key Designs

1. **Carefully Designed Synthetic Knowledge Graph**:

    - Function: Constructs a family graph covering 8 relations: husband, wife, father, mother, son, daughter, brother, sister
    - Mechanism: Divides 1,000 families into two groups — Group 1 contains all 36 facts per family; Group 2 deliberately omits father/mother relations. The test set consists of the withheld relations from Group 2.
    - Design Motivation: These 8 relations form a minimal closed system encompassing both inverse relations (husband↔wife) and compositional relations (husband∘mother=father), ideally probing reverse and multi-hop reasoning.

2. **Comparison of Three Representation Probes**:

    - **Linear Relational Embedding**: $o_L \approx W_r s_l + b_r$, using the Jacobian matrix to predict the object's final-layer representation from the subject representation.
    - **Translational**: $s_l + v_r \approx o_l$, analogous to Word2Vec-style vector translation, with subject and object in the same layer.
    - **Bilinear**: $f_r(s_l, o_l) = s_l^\top M_r o_l$, modeling the interaction between subject and object via a relation matrix $M_r$, solved using RESCAL with ridge regression.
    - Design Motivation: The bilinear model naturally supports $M_r^\top$ for inverse relations and $M_{r_2} M_{r_1}$ for compositional relations, whereas linear and translational probes cannot.

3. **Weight Decay as the Key Regularization**:

    - Function: Sweeps weight decay in AdamW from 0 to 6.0.
    - Core Finding: All models achieve 100% training accuracy, but models with low weight decay (<1.0) completely fail at reverse inference, while high weight decay models achieve nearly 100% reverse inference accuracy.
    - Design Motivation: Regularization encourages the model to learn a more generalizable internal structure (bilinear) rather than simply memorizing training data.

4. **Verification of Algebraic Properties**:

    - Function: Verifies whether the learned $M_r$ matrices satisfy transpose = inverse and product = composition.
    - Mechanism: Tests whether $M_{\text{husband}}^\top$ serves as the relation matrix for wife, and whether $M_{\text{husband}} \cdot M_{\text{mother}}$ predicts the father relation.
    - Result: Non-reversal-curse models achieve >95% accuracy in layers 6–9; reversal-curse models remain at low accuracy throughout.

5. **Model Editing Experiment**:

    - Function: Edits a husband-relation fact (A, husband, B→B') and evaluates logical propagation.
    - Three metrics: Edit Success (direct edit success rate), Logical Generalization (rate of update to entailed facts), Locality (preservation rate of unrelated facts).
    - Core Finding: Both model types succeed on direct edits, but logical generalization differs dramatically — bilinear-structured models propagate updates to entailed facts such as (B', wife, A), while reversal-curse models almost entirely fail.
    - Quantitative Correlation: Correlation between bilinear probe accuracy and logical generalization success rate: $R^2 = 0.939$.

### Interesting Layer-wise Finding
The layers most effective for editing (layers 1–4) differ from those where bilinear structure is strongest (layers 6–9). Edits must be applied in early layers where structure is *forming*, rather than in layers where it is already *established*, in order to correctly update downstream representations.

## Key Experimental Results

### Main Results
Reverse inference accuracy vs. weight decay (evaluated on withheld father/mother relations in Group 2):

| Weight Decay | Training Accuracy | Test Accuracy (Reverse Inference) | Status |
|-------------|---------|-------------------|------|
| 0 | 100% | ~10% | Reversal curse |
| 0.5 | 100% | ~30% | Reversal curse |
| 1.0 | 100% | ~40–98% (seed-dependent) | Bifurcation point |
| 3.0+ | 100% | ~98% | Reversal curse overcome |

### Probe Accuracy Comparison (Intermediate Layers 6–9)

| Probe Type | Non-Reversal-Curse Model | Reversal-Curse Model |
|---------|--------------|-------------|
| Linear | ~33% (baseline) | ~33% (baseline) |
| Translational | ~33% (baseline) | ~33% (baseline) |
| Bilinear | >95% | ~33% (baseline) |

### Model Editing Results

| Metric | With Bilinear Structure | Without Bilinear Structure |
|------|-----------------|-----------------|
| Edit Success | ~100% | ~100% |
| Logical Generalization | High (~90%+ at best layer) | Near 0% |
| Locality | High | Low |

### Key Findings
- **The reversal curse is a representation problem, not an architectural one**: The same architecture yields drastically different reasoning capabilities under different regularization strengths.
- **Bilinear structure concentrates in intermediate layers**: Accuracy peaks in layers 6–9, consistent with prior findings on relational operations in attention heads.
- **The algebraic structure is functional**: Matrix transpose ≈ inverse relation; matrix multiplication ≈ relation composition — not merely a statistical correlation.
- **Optimal editing layer ≠ strongest structure layer**: Correct propagation requires editing at early layers where structure is still forming.

## Highlights & Insights
- **Most profound reframing**: The reversal curse is redefined from a "model capability defect" to a "missing representation structure," shifting the problem-solving paradigm away from better training objectives or data augmentation toward the geometry of knowledge representations.
- **Elegant control via synthetic data**: The family relation graph is a minimal complete system for testing relational reasoning; the 8 relations precisely cover inverses and compositions.
- **Systematic probe design**: The linear/translational/bilinear comparison clearly eliminates alternative hypotheses.
- **Transferable principle**: The paradigm of *"first verify whether the model possesses the representational structure required for reasoning, then decide which algorithm to apply"* generalizes to broader AI reliability problems.

## Limitations & Future Work
- **Synthetic vs. real data**: All experiments are conducted on clean synthetic data with a 206M-parameter model; whether analogous bilinear structure exists in large-scale pretrained models remains unverified.
- **Limited relation types**: Only 8 family relations are studied; real-world knowledge involves thousands of heterogeneous relations, and different knowledge types may adopt different geometric structures.
- **Simple editing method**: Only basic layer-wise fine-tuning is used; integration with advanced editing methods such as ROME/MEMIT is not explored.
- **Causality concern**: Whether high bilinear probe accuracy implies that the model *uses* this structure for inference remains an open question — correlation vs. causation.

## Related Work & Insights
- **vs. Berglund et al. (2024) reversal curse paper**: They characterize the reversal curse as a fundamental limitation of LMs; this work shows it is not an inherent defect but a consequence of representation structure.
- **vs. Hernandez et al. (2024) Linear Relational Embedding**: They argue that LMs encode knowledge via linear relations; this work finds that bilinear structure is the key to supporting reverse inference.
- **vs. ROME/MEMIT editing methods**: Those works focus on algorithmic design; this work argues that editing success depends more fundamentally on whether the model possesses appropriate representational geometry.
- **vs. Nishi et al. (2025)**: They find that editing can "shatter" internal topological structure; this work further clarifies what structure (bilinear) needs to be preserved.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Attributes the reversal curse to representational geometry rather than training objectives, proposing an entirely new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four progressively structured experiments with elegant design, but limited to synthetic data and small models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain, polished figures, concise formulations.
- Value: ⭐⭐⭐⭐ Offers important insights into LM knowledge representation and editing mechanisms, though practical applications require further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)

</div>

<!-- RELATED:END -->
