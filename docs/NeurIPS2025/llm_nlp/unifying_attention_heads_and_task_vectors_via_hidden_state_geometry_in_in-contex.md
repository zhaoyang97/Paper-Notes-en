---
title: >-
  [Paper Note] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning
description: >-
  [NeurIPS 2025][LLM (Other)][in-context learning] This paper proposes a unified framework based on hidden state geometry (separability + alignment) that bridges the two major explanatory lines of ICL — attention heads (PTH/IH) and task vectors — revealing a two-phase mechanism in classification tasks: early layers establish separability via PTH, while later layers improve alignment with label unembedding directions via IH.
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "in-context learning"
  - "attention heads"
  - "task vectors"
  - "hidden state geometry"
  - "separability"
  - "alignment"
date: 2026-05-08
content_hash: 1ec2035d5f073c13
---

# Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning

**Conference**: NeurIPS 2025  
**arXiv**: [2505.18752](https://arxiv.org/abs/2505.18752)  
**Code**: [https://github.com/HLYang2001/ICL_Hidden_Geometry](https://github.com/HLYang2001/ICL_Hidden_Geometry)  
**Area**: LLM/NLP  
**Keywords**: in-context learning, attention heads, task vectors, hidden state geometry, separability, alignment

## TL;DR
This paper proposes a unified framework based on hidden state geometry (separability + alignment) that bridges the two major explanatory lines of ICL — attention heads (PTH/IH) and task vectors — revealing a two-phase mechanism in classification tasks: early layers establish separability via PTH, while later layers improve alignment with label unembedding directions via IH.

## Background & Motivation
**Background**: ICL is among the most distinctive capabilities of LLMs. Existing research explains its mechanism from two independent perspectives: (1) the attention head view — key circuits such as PTH and IH; (2) the task vector view — task representations extracted from demonstrations that guide predictions.

**Limitations of Prior Work**: The two research lines remain disconnected. The former analyzes component effects on final outputs (black-box), while the latter focuses on manipulating intermediate-layer representations. A unified framework explaining how both lines jointly shape hidden states layer by layer to produce correct outputs is lacking.

**Key Challenge**: Attention heads transform hidden states layer by layer, and hidden states ultimately determine outputs — these are two perspectives on the same process and should not be analyzed in isolation.

**Goal**: Construct a unified framework that attributes the roles of attention heads and task vectors to their effects on the geometric properties of query hidden states.

**Key Insight**: LLM classification is essentially a mapping of hidden states through the unembedding matrix into label space. Classification accuracy depends on two geometric factors: whether hidden states are separable, and whether the separating direction is aligned with the label unembedding vectors.

**Core Idea**: ICL accuracy ≤ maximum separability $S^*$, with equality achieved when the separating direction aligns with the label unembedding difference direction.

## Method

### Overall Architecture
Define two geometric properties — Separability and Alignment — for LLM classification → Theoretically prove that accuracy is determined by these two properties (Theorem 1) → Design metric indicators to track layer-wise changes → Analyze how PTH and IH respectively drive each property.

### Key Designs

1. **Geometric Framework and Theoretical Guarantee (Theorem 1)**:

    - Function: Proves that classification accuracy ≤ maximum separability $S^*$
    - Mechanism: Defines separability $S(\bm{u})$ along direction $\bm{u}$ as the proportion of samples correctly linearly classified along that direction. Equality holds when the separating direction aligns with the label unembedding difference $\bm{E}_{y_A} - \bm{E}_{y_B}$
    - Design Motivation: Transforms ICL analysis from a "component → output" black-box into an interpretable "component → geometry → output" framework

2. **Metric System**:

    - Separability score: logistic regression classifier accuracy approximating $S^*$
    - Output alignment (logit lens): accuracy of directly decoding intermediate-layer hidden states via the unembedding matrix
    - Directional alignment: cosine similarity between SVD singular vectors and label difference directions; variance/mean/composite alignment metrics
    - Effective dimensionality: concentration of variance in hidden states

3. **Two-Phase ICL Mechanism**:

    - Phase 1 (early layers): separability increases rapidly; alignment remains low
    - Phase 2 (middle-to-late layers): separability stabilizes; alignment surges abruptly
    - **Key Finding**: The separability gap between ICL and zero-shot is negligible (despite an ~80% accuracy gap), indicating that alignment is the bottleneck for zero-shot performance

4. **PTH Drives Separability; IH Drives Alignment**:

    - Ablating the top 10% PTH → significant drop in separability; ablating the top 10% IH → severe collapse of alignment
    - PTH concentrates in early layers (mean ~layer 35); IH concentrates in later layers (mean ~layer 41); Mann-Whitney test $p < 10^{-7}$
    - Injecting IH outputs as task vectors into zero-shot hidden states recovers the ICL-specific alignment surge

### Semantic Analysis
- Post-transition layers progressively filter label-irrelevant semantics and retain label-relevant semantics
- Low-rank denoising (rank-10 SVD) can improve accuracy by 10%+ in early post-transition layers
- Semantic reversal appears in the final few layers: label-relevant directions are filtered out

## Key Experimental Results

### Main Results — 7 Models × 6 Datasets

| Finding | Specific Data |
|------|---------|
| ICL vs. zero-shot separability | Negligible gap (~95% vs. ~93%) |
| ICL vs. zero-shot accuracy | Large gap (~85% vs. ~5%) |
| Two-phase transition | Consistently observed across all 7 models |
| Low-rank denoising gain | >10% accuracy improvement in early post-transition layers |

### Ablation Study

| Ablation Target | Effect on Separability | Effect on Alignment | Effect on Accuracy |
|---------|------------|-----------|-----------|
| Top 10% PTH | **Significant drop** | Partial effect | Moderate drop |
| Top 10% IH | Largely unchanged | **Severe collapse** | Large drop |
| Random equal-size heads | Negligible | Negligible | Negligible |

### Key Findings
- **Alignment is the core contribution of ICL**: Separability is an intrinsic property of LLM inference; ICL primarily improves alignment
- The two-phase transition is robust across varying numbers of demonstrations (4–24), kNN selection, and symbolic label settings
- Mechanism of IH outputs as task vectors: copying demonstration label embeddings to the query position, directly performing "directional correction"

## Highlights & Insights
- **Elegance of the unified framework**: A single theorem (Acc ≤ $S^*$) ties together the entire analysis, subsuming two independent research lines into a single geometric framework
- **"Alignment is the bottleneck" finding**: Zero-shot hidden states are already sufficiently separable; the separating direction simply fails to align with label unembedding directions. Task vectors are essentially performing "directional correction"
- **Mechanistic explanation of IH as task vectors**: IH copies demonstration label embeddings to the query position, "pushing" hidden states toward label unembedding directions

## Limitations & Future Work
- Analysis focuses on classification tasks; only preliminary validation is provided for generative tasks
- Only inference-time behavior is analyzed; the formation of the two-phase pattern during training is not explored
- Metrics rely on a known label space and are not directly applicable to open-domain tasks

## Related Work & Insights
- **vs. Todd et al. (2024)**: They find that IH outputs can decode label tokens and serve as task vectors; this paper explains *why* — IH outputs align with label unembedding directions
- **vs. Hendel et al. (2023)**: They propose that ICL creates task vectors; this paper reveals the geometric essence underlying task vector effectiveness
- **vs. Olsson et al. (2022)**: The classic IH work focuses on copying patterns; this paper complements it by showing how IH influences downstream computation through geometric alignment

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies the two major explanatory lines of ICL
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models, 6 datasets, diverse ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are tightly integrated
- Value: ⭐⭐⭐⭐⭐ A landmark contribution to understanding ICL mechanisms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](../../ICML2025/llm_nlp/beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](../../ACL2025/llm_nlp/beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[NeurIPS 2025\] On the Role of Hidden States of Modern Hopfield Network in Transformer](on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)

</div>

<!-- RELATED:END -->
