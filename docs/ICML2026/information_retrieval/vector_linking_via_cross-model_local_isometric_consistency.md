---
title: >-
  [Paper Note] Vector Linking based on Cross-Model Local Isometry Consistency
description: >-
  [ICML 2026][Information Retrieval & RAG][Vector Linking] This paper introduces the vector linking problem—discovering object correspondences between embedding clouds produced by two different encoders under black-box con…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Vector Linking"
  - "Local Geometric Consistency"
  - "Embedding Alignment"
  - "Multi-view Hashing"
  - "Bootstrapping"
date: 2026-05-08
content_hash: 5ac2790fe1512320
---

# Vector Linking based on Cross-Model Local Isometry Consistency

**Conference**: ICML 2026  
**arXiv**: [2605.31100](https://arxiv.org/abs/2605.31100)  
**Code**: https://github.com/DBgroup-Edinburgh/VecLinking  
**Area**: Information Retrieval / Vector Database / Embedding Alignment  
**Keywords**: Vector Linking, Local Geometric Consistency, Embedding Alignment, Multi-view Hashing, Bootstrapping

## TL;DR
This paper introduces the vector linking problem—discovering object correspondences between embedding clouds produced by two different encoders under black-box constraints. The core observation is that independently trained contrastive learning encoders maintain local isometry consistency over short distances (similarity preservation up to a scaling factor). Based on this, a multi-view geometric hashing bootstrapping framework is proposed, which recovers 79-90% of overlapping objects using only 15-30 seed pairs.

## Background & Motivation

**Background**: Embedding models evolve rapidly, and in practice, various systems adopt different fine-tuned encoders. Existing vector indices may contain the same objects, but their representations are incomparable, hindering cross-index retrieval, deduplication, and clustering.

**Limitations of Prior Work**: Traditional embedding alignment methods assume the existence of global isomorphism and rely on global linear/OT (Optimal Transport) transformations. However, vector linking involves partially unknown overlaps—non-overlapping regions are not simple outliers but can be structured and large. Global alignment often degrades correspondences in overlapping regions while trying to improve the fit for non-overlapping ones.

**Key Challenge**: Black-box constraints (access only to static vectors, with no model parameters, gradients, or training data) combined with partial unknown overlaps make a single global transformation unreliable.

**Goal**: Recover large-scale vector correspondences from a tiny seed set (15-30 pairs) under black-box constraints.

**Key Insight**: Independently trained contrastive encoders maintain strong correlation at short distances (Pearson > 0.8), while correlation decays rapidly at long distances. This suggests that local neighborhoods are significantly more stable than global arrangements.

**Core Idea**: Replace raw distances with "signatures from distance anchors"—relative distance patterns that remain similar (up to a scale factor) within local neighborhoods across models. Filter model-specific distortions through multi-view voting aggregation.

## Method

### Overall Architecture
GEH (Geometric Embedding Hashing) is a three-stage iterative process—(1) sample multiple small views from the current anchor pool, with each view inducing an independent hash space; (2) propose candidate links (mutual nearest neighbors) within the hash space; (3) aggregate evidence across views and use a Beta-Bernoulli posterior to promote high-confidence pairs as new anchors.

### Key Designs

1.  **Distance Anchor Geometric Hashing**:
    - **Function**: Generates encoder-invariant signatures for each vector. Given a pair of anchor sets $\mathcal{A}=\{(a_1,a'_1),\ldots,(a_k,a'_k)\}$, compute $\mathbf{r}_{\mathcal{A}}(u):=(\text{dist}(u,a_1),\ldots,\text{dist}(u,a_k))$, using scale-free similarity $\text{sim}_{\mathcal{A}}(u,v):=\langle\widehat{\mathbf{r}}_{\mathcal{A}}(u),\widehat{\mathbf{r}}'_{\mathcal{A}}(v)\rangle$.
    - **Mechanism**: Theorem 1 proves that two locally optimal contrastive encoders maintain $\|f_1(x)-f_1(y)\|=\kappa\cdot\|f_2(x)-f_2(y)\|+\mathcal{O}(d_{\mathcal{M}}(x,y)^2)$ at short distances, where $\kappa=\sqrt{\lambda_1/\lambda_2}$.
    - **Design Motivation**: Avoid globally unreliable absolute distances; use scale-invariant relative geometry of the local neighborhood to handle partial overlap.

2.  **Multi-view Voting Aggregation**:
    - **Function**: Counts the number of support votes $\nu_{(u,v),t}:=\sum_{r,k}Y_{r,k}(u,v)$ from multiple views for candidate pairs.
    - **Mechanism**: True links are proposed in views containing locally relevant anchors (Figure 2 shows a median of 48 votes), whereas pseudo-link support decays rapidly (following an exponential distribution). A Beta-Bernoulli posterior $\theta_{(u,v)}\mid\mathcal{Y}\sim\text{Beta}(1+\nu_{(u,v),t}, 1+N_{\leq t}-\nu_{(u,v),t})$ automatically determines confidence.
    - **Design Motivation**: Bypass hard-to-determine local thresholds $\delta_{\mathcal{M}}$ by filtering spurious collisions through the statistical stability of multi-view voting.

3.  **Adaptive Bootstrapping Schedule**:
    - **Function**: Iteratively grows the anchor pool, sampling $m_t:=\lceil m_0(1+c\log g_t)\rceil$ views at round $t$, each with size $s_t:=\lceil\rho_0|\mathcal{L}_{t-1}|/\text{sf}_t\rceil$.
    - **Mechanism**: As the anchor pool grows, the number of views increases while the size of individual views shrinks to maintain locality. Greedy FPS (Farthest Point Sampling) ensures anchor diversity, while Otsu's method adaptively determines $\tau_t$ without manual tuning.
    - **Design Motivation**: Tiny seeds (15 pairs) cannot cover the global space and require intelligent re-sampling; multi-view voting balances high local information density with sparse global coverage.

## Key Experimental Results

### Main Results

| Model Pair | Dataset | Prec/Rec/F1 (%) | Second Best Method | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Mistral-OpenAI | FiQA | **82.1/95.6/88.3** | Proc 52.5/11.8/19.3 | +68.9% F1 |
| GTE-OpenAI | ArguAna | **77.1/84.5/80.7** | Proc 30.8/4.8/8.4 | +71.8% F1 |
| Qwen-KaLM | FiQA | **79.8/79.9/79.8** | Proc 20.6/1.3/2.4 | +58.0% F1 |

(Overlap $\alpha=0.3$, seed size 15 pairs)

### Ablation Study (SciDocs, Mistral vs OpenAI, $\alpha=0.15$, 15 seeds)

| Configuration | Precision (%) | Recall (%) | F1 (%) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Full GEH | 62.1±1.1 | 81.7±0.7 | 70.5±0.6 | Baseline |
| w/o Kernel | 61.0±8.3 | 52.9±35.0 | 51.0±33.3 | Unstable without distance weighting |
| w/o FPS Sampling | - | - | - | Random sampling performance drop |
| w/o Posterior Aggregation | - | - | - | Fixed threshold fails |

### Key Findings
- **Ultra-low Seed Effectiveness**: Achieves performance with 15 pairs comparable to 30 pairs, whereas all baselines require 30-50 pairs.
- **Large-scale Scalability**: On FEVER (5.4 million texts), achieves 93.8% Precision and 68.9% Recall, with an end-to-end time of 3328 seconds on a single A100.
- **Cross-encoder Robustness**: Across 5 model pairs and 6 datasets, no voting variance > 33%, demonstrating that multi-view voting is the core of stability.

## Highlights & Insights
- **Local Isometry Theory**: Theorem 1 provides a rigorous proof that contrastive encoders maintain local distance ratios, breaking the assumption that "global isomorphism is required for black-box embedding alignment."
- **Statistical Multi-view Design**: The Beta-Bernoulli conjugate requires no hyperparameter tuning, and Otsu's adaptive threshold is entirely data-driven. The core insight is the signal/noise separation (median 48 vs exponential decay) shown in Figure 2.
- **Transferable Hashing Concept**: Distance anchor signatures are not limited to specific embeddings and can be applied to any vector collective; the multi-view voting framework is applicable to any model pairs with local consistency.

## Limitations & Future Work
- **Assumption Limitations**: The local positive sampling and isotropy assumptions may not hold for strong data augmentation or specialized domains; second-order Taylor expansion errors may be non-trivial in high dimensions.
- **Parameter Sensitivity**: Hyperparameters like $s_t, m_0, c$ for view scheduling have not been fully analyzed.
- **Future Directions**: Extending the theory to weak contrastive encoders; meta-learning for adaptive $s_t$ scheduling; hybrid offline-online strategies to accelerate large-scale deployment.

## Related Work & Insights
- **vs Traditional Point Set Registration (RANSAC/ICP/Geometric Hashing)**: The latter target 3D rigid bodies and low-dimensional spaces; this work handles high-dimensional heteroscedastic model distortion and partial overlap.
- **vs Global Alignment (Procrustes/OT)**: This work is local-first and does not require global isomorphism; multi-view voting is more robust against partial overlap destruction than global fitting.
- **Insight**: Alignment problems require a "problem-specific" geometric perspective rather than universal optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulates the vector linking problem for the first time; proves local isometry of contrastive encoders; unprecedented black-box multi-view bootstrapping framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 BEIR datasets × 5 model pairs × 9 configurations + large-scale 5.4M test + full ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear theory and comprehensive experiments; powerful visualizations; discussion on limitations is brief.
- Value: ⭐⭐⭐⭐⭐ Resolves a core challenge in the integration of cross-model vector databases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)

</div>

<!-- RELATED:END -->
