---
title: >-
  [Paper Note] Mapping Semantic & Syntactic Relationships with Geometric Rotation
description: >-
  [ICLR 2026][Information Retrieval & RAG][Embedding Geometry] This paper proposes RISE (Rotor-Invariant Shift Estimation), a method that leverages Clifford algebra rotors to represent utterance-level semantic–syntactic tr…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Embedding Geometry"
  - "Hyperspherical Rotation"
  - "Cross-lingual Generalization"
  - "Semantic Transformation"
  - "Linear Representation Hypothesis"
date: 2026-05-08
content_hash: e7be55be076c7a03
---

# Mapping Semantic & Syntactic Relationships with Geometric Rotation

**Conference**: ICLR 2026
**arXiv**: [2510.09790](https://arxiv.org/abs/2510.09790)  
**Code**: [https://github.com/fuelix/RISE-steering](https://github.com/fuelix/RISE-steering)  
**Area**: Representation Learning / Embedding Interpretability
**Keywords**: Embedding Geometry, Hyperspherical Rotation, Cross-lingual Generalization, Semantic Transformation, Linear Representation Hypothesis

## TL;DR

This paper proposes RISE (Rotor-Invariant Shift Estimation), a method that leverages Clifford algebra rotors to represent utterance-level semantic–syntactic transformations (negation, conditionalization, and politeness) as consistent rotation operations on the unit hypersphere. Through systematic experiments across 7 languages × 3 embedding models × 3 transformation types, the paper demonstrates that these rotations transfer across languages and model architectures (77%–95% retention rate), extending the Linear Representation Hypothesis (LRH) from word-level to cross-lingual utterance-level and generalizing it to geodesic structures on curved manifolds.

## Background & Motivation

**Background**: In the word2vec era, semantic relationships could be intuitively expressed through vector arithmetic ("king − man + woman = queen"), and semantic spaces exhibited clear linear interpretability. The Linear Representation Hypothesis (LRH) formalized this observation, positing that semantic concepts are encoded as linear structures in embedding space. However, the high-dimensional representations of modern Transformer models have lost this intuitive geometric correspondence, rendering internal mechanisms opaque.

**Limitations of Prior Work**: (1) Existing interpretability methods (probes, steering vectors) are largely task-specific and lack a systematic geometric framework for mapping semantic relationships. (2) Steering vectors exhibit inconsistent behavior across contexts and poor generalizability. (3) The LRH has been validated primarily at the monolingual word level, with virtually no verification at the cross-lingual or utterance level. (4) A critical geometric mismatch exists: modern embeddings reside on curved manifolds (hyperspheres), while conventional methods operate in Euclidean space—a mismatch that may be the fundamental cause of poor generalization in steering vectors.

**Key Challenge**: Embedding space is curved (spherical), yet the tools for manipulation and analysis assume it is flat (Euclidean). A method for representing semantic transformations that respects manifold geometry is needed.

**Goal**: To develop a geometric framework for identifying utterance-level semantic–syntactic transformations, and to determine whether these transformations can be consistently mapped to geometric operations across languages and model architectures.

**Key Insight**: Normalized sentence embeddings reside on the unit hypersphere → semantic transformations = rotational displacements on the sphere → Clifford algebra rotors are used to canonicalize and align transformations across different sentences → a universal rotation prototype is learned to represent a given semantic transformation → this prototype is applied to new sentences for prediction.

**Core Idea**: Semantic–syntactic transformations correspond to consistent rotation operations on the unit hypersphere; once canonicalized via rotors, these rotations transfer across languages and models.

## Method

### Overall Architecture

RISE achieves geometric representation of semantic transformations in three steps: (1) **Canonicalization**—rotors are used to align each pair of sentence embeddings to a reference direction; (2) **Prototype Learning**—a unified rotation prototype is obtained by averaging all canonicalized transformations in the reference coordinate system; (3) **Prediction**—the prototype is applied to new sentences to predict their transformed counterparts. The entire process operates within a Riemannian geometric framework, employing logarithmic maps (log maps) and exponential maps (exp maps) to work in the tangent space.

### Key Designs

1. **Rotor Canonicalization and Prototype Learning**:

    - **Function**: For each neutral–transformed sentence embedding pair $(n_i, v_i)$, compute the orthogonal transformation $R(n_i)$ that maps $n_i$ to the reference direction $e_1$, then average all transformed tangent vectors in the reference coordinate system to obtain the prototype $\vec{p}$.
    - **Mechanism**: On the hypersphere, the semantic transformation from $n_i$ to $v_i$ can be represented as a tangent vector at $n_i$ via the Riemannian logarithmic map $\log_{n_i}(v_i)$. However, the tangent spaces at different $n_i$ are not directly comparable. The rotor $R(n_i)$ aligns all tangent spaces to a common reference frame—effectively "controlling for differences in the underlying semantic content" and isolating the transformation itself. The canonicalized tangent vectors $\xi = R(n_i)\log_{n_i}(v_i)$ can then be directly averaged to obtain a universal rotation prototype representing the given semantic transformation.
    - **Design Motivation**: Conventional additive vectors (Mean Difference Vectors) operate in Euclidean space and disregard the curved structure of the embedding space. RISE respects the intrinsic geometry of the manifold through Riemannian operations (log/exp maps + rotor alignment), making it theoretically more appropriate for semantic analysis on the hypersphere.

2. **Cross-lingual and Cross-model Transfer**:

    - **Function**: Rotation prototypes learned on language A are directly applied to embeddings of language B to evaluate cross-lingual transfer; statistical mapping (PCA + distributional alignment following Morris et al.) is used to transfer text-embedding-3-large prototypes into the bge-m3 space for cross-model transfer evaluation.
    - **Mechanism**: If a semantic transformation (e.g., negation) corresponds to the same geometric operation in the embedding spaces of different languages, then a rotation prototype learned on one language should directly apply to another. This tests the central hypothesis that semantic transformations have geometrically universal structure across languages.
    - **Design Motivation**: This constitutes the strongest test of the LRH—if the negation transformation in English and Japanese (typologically unrelated languages) corresponds to similar rotations in their respective embedding spaces, this geometric structure reflects properties of semantics itself rather than language-specific grammatical mechanisms.

3. **Commutativity Verification and Theoretical Support**:

    - **Function**: Demonstrates that sequentially applying multiple RISE transformations (e.g., negation followed by conditionalization vs. conditionalization followed by negation) commutes to first-order approximation.
    - **Mechanism**: Theorem A.1 proves that, under the small-angle approximation, the difference between two sequential RISE operations is $O(\|\vec{p}_A\| \cdot \|\vec{p}_B\|)$—a second-order quantity. This implies that semantic transformations behave like vector addition in the tangent space, providing theoretical support for extending the LRH to curved spaces.
    - **Design Motivation**: If RISE transformations satisfy commutativity, semantic transformations are composable—"a negated conditional" ≈ "a conditionalized negation"—consistent with the linear algebraic structure predicted by the LRH, but generalized to the geodesic framework.

### Experimental Design

- **3 semantic transformations**: negation ("P" → "not-P", a precise logical operation), conditionalization ("P" → "if P", introducing modal semantics), and politicization (increasing social formality, highly context- and culture-dependent).
- **7 languages**: English, Spanish, Japanese, Tamil, Thai, Arabic, and Zulu (5 language families, covering analytic, agglutinative, and inflectional morphological types).
- **3 embedding models**: text-embedding-3-large (3072-dim), bge-m3 (1024-dim), and mBERT (768-dim).
- **2 baselines**: Mean Difference Vectors (MDV, spherical mean difference vectors) and Procrustes alignment (global rotation fitting).
- **3 evaluation datasets**: a synthetic multilingual dataset (1,000 pairs per language × transformation, generated by GPT-4.5), BLiMP (English syntactic benchmark), and SICK (English semantic similarity).

## Key Experimental Results

### Main Results (Cross-lingual Transfer, Rotor Alignment Score = Cosine Similarity)

| Transformation | Mean Score | Performance Range | Variance | Characteristics |
|---|---|---|---|---|
| Negation | **0.788** | 0.686–0.918 | Moderate | Strongest cross-lingual |
| Conditionalization | 0.780 | More stable | **Lowest (0.038)** | Highest consistency |
| Politeness | 0.762 | Larger variation | Highest (0.060) | Clear cultural dependency |

### Cross-model Transfer (text-embedding-3-large → bge-m3)

| Language | Transfer Performance | Notes |
|---|---|---|
| English | 0.80–0.82 | Best; possibly reflects training data bias |
| Other languages | 0.70–0.75 | Reasonable, with ~20% performance gap |
| Zulu | 0.63–0.66 | Lowest; low-resource language challenge |

### Baseline Comparison

| Method | Monolingual Syntax (BLiMP) | Monolingual Semantics (SICK) | Cross-lingual Transfer |
|---|---|---|---|
| RISE | Strong (0.97) | Strong (0.84) | Moderate–Strong (0.74–0.89) |
| MDV | Strong (0.97) | Strong (0.83) | Moderate–Strong (0.72–0.91) |
| Procrustes | Strong (0.99) | Moderate (0.67) | **Failing–Weak (0.25–0.62)** |

### Key Findings

- Negation exhibits the most consistent rotation across all languages—suggesting that negation may be a "primitive operation" in semantic space, independent of its specific grammatical realization (particles, affixes, auxiliary verbs).
- Conditionalization shows the highest consistency (lowest variance: 0.038)—the encoding of modal semantics is highly stable across languages.
- Politeness exhibits the greatest variation—cultural and social factors make its cross-lingual geometric structure the least stable, as expected.
- Procrustes alignment fails severely in the cross-lingual setting—global rigid rotation is too coarse, confirming the necessity of manifold-based methods.
- Embedding dimensionality does not directly determine cross-lingual performance—the 1024-dim bge-m3 even outperforms the 3072-dim text-embedding-3-large in cross-lingual consistency, suggesting that training methodology and architectural choices matter more.
- On a downstream negation classification task: RISE achieves 93.0% accuracy vs. MDV's 87.2%, confirming that RISE's geometric representation captures more useful semantic information.

## Highlights & Insights

- "Rotation rather than translation" represents a core philosophical shift—on a sphere, translation (additive vectors) is not the natural operation; rotation is. RISE respects the intrinsic geometry of the embedding space, which may explain why conventional steering vectors perform inconsistently across different contexts.
- The cross-lingual geometric consistency of negation suggests it may be a "semantic primitive"—whether English uses "not," Japanese uses "ない," or Tamil uses verb-internal negation, all correspond to similar rotation directions in embedding space.
- **Spherical generalization of the LRH**—the conventional "linear direction = semantics" in flat space corresponds to "geodesic arc = semantics" in curved space. RISE extends the LRH from Euclidean space to Riemannian manifolds, representing a significant theoretical contribution.
- **Commutativity provides an algebraic foundation for semantic composition**—the approximate commutativity of multiple RISE transformations implies that semantic transformations possess algebraic properties analogous to a vector space, serving as an important theoretical guarantee for interpretability.

## Limitations & Future Work

- Only three semantic transformations are examined—extension to a broader range of semantic and pragmatic phenomena (e.g., tense, causality, entailment) is needed to test the framework's generality.
- The synthetic data is generated by GPT-4.5, which may introduce English-centric bias; validation by native speakers and on naturalistic corpora is required.
- Cross-model transfer shows a marked English advantage (approximately 20% higher than Zulu), reflecting imbalanced training data in current multilingual models.
- RISE achieves only moderate performance on semantic similarity tasks (SICK: 0.62–0.74), suggesting that rotation prototypes may better capture grammatical/syntactic structure than deep semantic similarity.
- Although computational complexity is $O(d)$, efficiency at very large embedding scales (>3072 dimensions) has not been tested in practice.
- MDV approaches RISE's performance in certain settings—more tasks are needed to better delineate the boundary conditions distinguishing the two methods.

## Related Work & Insights

- **Park et al. (2024, 2025)**: Formalized three notions of linear representation under the LRH—RISE extends these to geodesic structures on curved spaces.
- **Householder Pseudo-Rotation (HPR)**: Uses Householder reflections for LLM activation steering—RISE uses rotors for embedding space analysis; the geometric tools are similar, but the application domains are entirely distinct.
- **Steering vector research** (Turner, Zou, Li, et al.): Operates on LLM activation layers—RISE is the first to extend steering principles to embedding models and manifold spaces.
- **Word2Vec analogy reasoning**: RISE can be viewed as the natural generalization of word2vec vector analogies to modern embeddings with curved geometry.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to model semantic transformations as hyperspherical rotations, with cross-lingual validation and spherical generalization of the LRH.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full-matrix evaluation across 7 languages × 3 models × 3 transformations × 3 datasets × 2 baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete narrative arc from theoretical motivation to method design, large-scale experiments, and mathematical proofs.
- Value: ⭐⭐⭐⭐ Foundational contribution to embedding interpretability and multilingual understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding](../../AAAI2026/information_retrieval/magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)
- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](../../NeurIPS2025/information_retrieval/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)

</div>

<!-- RELATED:END -->
