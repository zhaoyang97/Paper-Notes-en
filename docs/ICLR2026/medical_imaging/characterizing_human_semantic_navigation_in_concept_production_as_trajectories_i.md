---
title: >-
  [Paper Note] Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space
description: >-
  [ICLR2026][Medical Imaging][semantic navigation] This paper proposes modeling the human concept production process as cumulative trajectories in Transformer embedding space, defining 5 kinematic metrics (distance…
tags:
  - "ICLR2026"
  - "Medical Imaging"
  - "semantic navigation"
  - "embedding trajectory"
  - "cognitive modeling"
  - "verbal fluency"
  - "neurodegenerative disease"
date: 2026-05-08
content_hash: a35e0b82d37ab2e7
---

# Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space

**Conference**: ICLR2026  
**arXiv**: [2602.05971](https://arxiv.org/abs/2602.05971)  
**Code**: [https://github.com/jesuinovieira/semtraj-iclr2026](https://github.com/jesuinovieira/semtraj-iclr2026)  
**Area**: Medical Imaging  
**Keywords**: semantic navigation, embedding trajectory, cognitive modeling, verbal fluency, neurodegenerative disease

## TL;DR
This paper proposes modeling the human concept production process as cumulative trajectories in Transformer embedding space, defining 5 kinematic metrics (distance, velocity, acceleration, entropy, and centroid distance). Evaluated on 4 datasets spanning 3 languages and covering neurodegenerative disease, taboo word fluency, and attribute listing tasks, the framework successfully distinguishes clinical groups and concept categories, with highly consistent results across different embedding models.

## Background & Motivation

**Background**: Human semantic retrieval is modeled in cognitive science as a "foraging" process in semantic space, balancing exploitation (clustering) and exploration (switching). Traditional methods analyze verbal fluency task data using a binary clustering/switching framework.

**Limitations of Prior Work**: (a) Clustering/switching analysis relies on time-consuming manual annotation and heterogeneous pipelines, rendering cross-study comparisons infeasible; (b) static word embeddings (e.g., fastText) ignore the cumulative nature of semantic retrieval—the semantics of each word is influenced by preceding words; (c) traditional analysis yields only coarse-grained categories (clustering vs. switching), lacking step-wise quantitative dynamics.

**Key Challenge**: Semantic retrieval is a history-dependent dynamic process (requiring working memory to suppress previously produced words), yet existing NLP methods embed each word independently, discarding sequential dependencies.

**Goal**: To establish a trajectory analysis framework based on cumulative embeddings, quantifying the step-wise dynamics of human semantic navigation using physical kinematic metrics.

**Key Insight**: Concept production sequences are treated as trajectories in embedding space—where each step's embedding is a cumulative encoding of all words produced so far. Concepts from physics such as distance, velocity, and acceleration are borrowed to characterize trajectory properties.

**Core Idea**: Model semantic retrieval as motion trajectories in high-dimensional space using cumulative Transformer embeddings, and use kinematic metrics to enable automated, cross-lingual analysis of semantic navigation.

## Method

### Overall Architecture
For each participant's concept production sequence (e.g., "cat, dog, shark…"), a Transformer text embedding model is used for cumulative encoding: $x_t$ encodes the concatenation of words 1 through $t$. The resulting embedding sequence $X = (x_1, \ldots, x_N)$ forms a trajectory in semantic space. Five metrics are computed over the trajectory and analyzed for inter-group/inter-category differences using GLMM statistical models.

### Key Designs

1. **Cumulative Embeddings**:

    - **Function**: Encode the sequential dependencies of concept production into the embeddings.
    - **Mechanism**: The embedding $x_t$ at step $t$ is not an independent embedding of word $t$, but rather a holistic embedding of "word1 word2 … word$t$". For example, after producing "cat dog," $x_2$ encodes the phrase "cat dog."
    - **Design Motivation**: Cognitive science research indicates that semantic retrieval depends on working memory and inhibitory control—previously produced words influence subsequent retrieval. Cumulative embeddings naturally capture this prefix dependency.

2. **5 Kinematic Metrics**:

    - **Distance to Next**: Cosine distance between consecutive embeddings—quantifies the magnitude of each "semantic jump."
    - **Velocity** $\mathbf{v}_t = \mathbf{x}_{t+1} - \mathbf{x}_t$: Vector difference preserving directional information—captures not only how far but also in which direction the trajectory moves.
    - **Acceleration** $\mathbf{a}_t = \mathbf{v}_{t+1} - \mathbf{v}_t$: Rate of change of velocity—quantifies the stability of search strategy (low acceleration = stable clustering; high acceleration = frequent switching).
    - **Entropy**: Shannon entropy computed after binarizing the distance sequence (above median = 1, below = 0)—quantifies the predictability of the search process.
    - **Distance to Centroid**: Distance from each embedding to the centroid of all embeddings for a given participant—quantifies the global dispersion of the search.

3. **Multi-Model Comparative Validation**:

    - **Function**: Verify that results are not dependent on a specific embedding model.
    - **Mechanism**: Three models are compared—OpenAI text-embedding-3-large, Google text-embedding-004, and Qwen3-Embedding-0.6B—alongside a fastText baseline, with cross-model correlations of trajectory metrics assessed.
    - **Design Motivation**: Consistent trajectory characteristics across different models indicate that the framework captures genuine cognitive phenomena rather than model-specific artifacts.

### Statistical Analysis
Generalized Linear Mixed Models (GLMMs) are employed with participants and concepts as random effects, and Tukey HSD correction applied for multiple comparisons. Log-normal distributions are fitted to distance, entropy, velocity, and acceleration; Gaussian distributions are fitted to centroid distance.

## Key Experimental Results

### Main Results (Inter-group/Inter-category Differences across 4 Datasets)

| Dataset | Comparison | Distance to Next | Velocity | Entropy | Distance to Centroid |
|--------|------|-----------------|----------|---------|---------------------|
| **Neurodegenerative Disease** | HC vs PD/bvFTD | HC **lower** | HC **lower** | HC **lower** | HC **higher** |
| **Taboo Word Fluency** | Animals vs Taboo | Taboo **highest** | Taboo **highest** | Taboo **highest** | Taboo **lowest** |
| **Italian** | Bird vs other categories | Bird **highest** | Bird **highest** | Selective category differences | Some categories higher/lower |
| **German** | Bird vs other categories | Bird **highest** | Bird **highest** | Selective category differences | Differs from Italian |

### Ablation Study (Cross-Model Consistency — Pearson Correlation)

| Metric | OpenAI vs Google | OpenAI vs Qwen3 | Notes |
|------|-----------------|-----------------|------|
| Velocity | >0.9 | >0.9 | Local dynamics highly consistent |
| Acceleration | >0.9 | >0.9 | Local dynamics highly consistent |
| Entropy | ~1.0 | ~1.0 | Rank-based; most stable |
| Distance to Centroid | 0.3–0.6 | 0.3–0.6 | Least consistent; dependent on model geometry |

### Key Findings
- **Kinematic Signatures of Neurodegenerative Disease**: Patients with PD and bvFTD exhibit higher velocity, acceleration, and entropy (disordered and unpredictable search) but lower centroid distance (restricted search space)—consistent with executive dysfunction manifesting as "more chaotic search within a smaller space."
- **Distinctive Semantic Topology of Taboo Words**: Taboo words form compact yet high-variability clusters in embedding space—lowest centroid distance (spatially compact) but highest distance, velocity, acceleration, and entropy (most irregular retrieval paths).
- **Cross-lingual Differences Reveal Cultural Encoding**: Italian and German datasets share the same protocol but yield different category effects, indicating that semantic organization is influenced by language and culture—differences that trajectory metrics can detect.
- **High Consistency Across Embedding Models**: Models with different training pipelines (causal vs. bidirectional attention) show high agreement on local trajectory dynamics ($r > 0.9$), but differ in global geometry (centroid distance)—suggesting that local semantic structure is shared across models.

## Highlights & Insights
- **Physics-Inspired Cognitive Metrics**: Directly mapping physical kinematics (velocity, acceleration) to semantic navigation is conceptually elegant, allowing cognitive scientists to describe semantic search using intuitive physical analogies.
- **Complementarity of Cumulative vs. Non-cumulative Embeddings**: Cumulative embeddings perform better for long sequences (capturing historical dependencies), while non-cumulative embeddings may be preferable for short sequences (insufficient context)—providing a practical selection guide.
- **Model Divergence in Distance to Centroid**: The lowest cross-model consistency of centroid distance can itself be leveraged to investigate how different models organize global semantic structure, serving as a tool for comparing LLM semantic spaces.
- **Extensibility to LLM Evaluation**: The framework can be directly applied to analyze semantic navigation patterns in LLM-generated text, enabling quantitative comparison of human vs. AI semantic search.

## Limitations & Future Work
- **Assumption of Euclidean Dynamics**: Applying Euclidean distance and derivatives in high-dimensional anisotropic embedding space is a simplification. Non-Euclidean geometries (e.g., hyperbolic space) may be more appropriate.
- **No Timestamps**: The datasets lack production timestamps for individual words; all steps are assumed to occur at equal time intervals. Data with timestamps would enable computation of true velocity and acceleration.
- **Limited to Fluency/Attribute Listing Tasks**: Coverage is restricted. Validation on more complex language production tasks such as free narration and dialogue is needed.
- **Potential Training Data Overlap**: Transformer models may have been trained on data containing similar semantic association knowledge, potentially artificially enhancing the discriminative power of cumulative embeddings.

## Related Work & Insights
- **vs. Traditional Clustering/Switching Analysis**: Traditional methods require manual annotation of subcategory boundaries. The proposed framework is fully automated and provides continuous step-wise dynamics rather than binary classification.
- **vs. Linz et al. (2017) Word Embedding Analysis**: That work applied static word2vec embeddings to verbal fluency analysis. This paper upgrades to cumulative Transformer embeddings combined with kinematic metrics, substantially increasing informational richness.
- **vs. Nour et al. on Schizophrenia Language Analysis**: That work also used embedding trajectories to analyze psychiatric disorders. This paper systematizes the metric framework and validates cross-model and cross-lingual robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework combining cumulative embeddings with kinematic metrics is novel, and the physics–cognition analogy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 4 datasets × 3 languages × 3 embedding models + 1 baseline, with rigorous statistics (GLMM + Tukey); however, validation on predictive tasks is lacking.
- Writing Quality: ⭐⭐⭐⭐ The bridging narrative between cognitive science and NLP is clear, though some sections presenting results are somewhat verbose.
- Value: ⭐⭐⭐⭐ Provides an automated analysis tool for cognitive science with potential clinical diagnostic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](../../NeurIPS2025/medical_imaging/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)

</div>

<!-- RELATED:END -->
