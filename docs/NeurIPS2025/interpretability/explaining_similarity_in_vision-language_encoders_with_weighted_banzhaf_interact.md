---
title: >-
  [Paper Note] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions
description: >-
  [NeurIPS 2025][Interpretability][CLIP interpretability] FIxLIP proposes a game-theoretic framework based on weighted Banzhaf interaction indices that unifies the decomposition of similarity predictions in vision-language encoders (e.g., CLIP, SigLIP-2) into first-order token attributions and second-order cross-modal/intra-modal interactions, surpassing existing first-order attribution methods in both efficiency and faithfulness.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "CLIP interpretability"
  - "Banzhaf interaction"
  - "game theory"
  - "cross-modal interaction"
  - "saliency map"
date: 2026-05-08
content_hash: c1a69b7c732d4d50
---

# Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions

**Conference**: NeurIPS 2025
**arXiv**: [2508.05430](https://arxiv.org/abs/2508.05430)  
**Code**: [hbaniecki/fixlip](https://github.com/hbaniecki/fixlip)  
**Area**: Interpretability
**Keywords**: CLIP interpretability, Banzhaf interaction, game theory, cross-modal interaction, saliency map

## TL;DR
FIxLIP proposes a game-theoretic framework based on weighted Banzhaf interaction indices that unifies the decomposition of similarity predictions in vision-language encoders (e.g., CLIP, SigLIP-2) into first-order token attributions and second-order cross-modal/intra-modal interactions, surpassing existing first-order attribution methods in both efficiency and faithfulness.

## Background & Motivation
Language-image pretraining (e.g., CLIP, SigLIP) has enabled a wide range of capabilities including zero-shot classification and cross-modal retrieval. However, these encoders are increasingly deployed in high-stakes decision-making (e.g., medical imaging), and the limitations of their internal representations have been extensively studied—such as CLIP's inability to correctly judge orientation, object count, or text in images.

Existing explanation methods (e.g., GAME, Grad-ECLIP) can only generate first-order attribution saliency maps capturing the importance of individual tokens. However, **the similarity predictions of vision-language encoders fundamentally depend on complex cross-modal interactions between image patches and text tokens**—a relationship that first-order methods cannot faithfully explain. User studies further confirm that visualizing second-order attributions (i.e., pairwise interactions) is necessary for understanding complex multimodal models.

**Core Problem**: How can the similarity predictions of VLEs be efficiently and faithfully decomposed into first-order attributions and second-order interactions?

## Method

### Overall Architecture
FIxLIP models the interpretation of vision-language encoders as a game-theoretic problem:
1. Input image patches and text tokens are treated as players in a cooperative game.
2. A FIxLIP game is defined: for all possible masks $M \subseteq N_{\mathcal{I}} \cup N_{\mathcal{T}}$, the similarity of the masked input is measured.
3. Weighted Banzhaf interaction indices are used to approximate the decomposition of game values into a constant term, first-order attributions, and second-order interactions.
4. Efficient approximation is achieved via weighted least-squares regression.

The resulting explanation takes the form of a complete graph: node weights represent token attributions, and edge weights represent pairwise interactions (including cross-modal and intra-modal).

### Key Designs

**Formal definition of the FIxLIP-p explanation:**

Given a VLE $f(x_{\mathcal{I}}, x_{\mathcal{T}}) = \cos(f_{\mathcal{I}}(x_{\mathcal{I}}), f_{\mathcal{T}}(x_{\mathcal{T}}))$, the FIxLIP game is defined as:

$$\nu(M) = f(x_{\mathcal{I}} \oplus_{M \cap N_{\mathcal{I}}} b_{\mathcal{I}}, x_{\mathcal{T}} \oplus_{M \cap N_{\mathcal{T}}} b_{\mathcal{T}})$$

The explanation is approximated by a second-order additive game: $\hat{\nu}_{\mathbf{e}}(M) = \mathbf{e}_0 + \sum_{i \in M} \mathbf{e}_i + \sum_{\{i,j\} \subseteq M} \mathbf{e}_{\{i,j\}}$

**p-faithfulness metric and weighted Banzhaf interactions:**
- p-faithfulness is defined as: $\mathfrak{F}_p(\nu, \hat{\nu}) = \sum_M p^{|M|}(1-p)^{n-|M|}(\nu(M) - \hat{\nu}(M))^2$
- The parameter $p$ controls the mask weight distribution: $p=0.5$ weights all masks equally; $p>0.5$ emphasizes in-distribution inputs (few tokens masked); $p<0.5$ emphasizes out-of-distribution inputs.
- FIxLIP-p = argmin p-faithfulness = **weighted Banzhaf interaction index**
- Advantages over Shapley interactions: (1) $p$ provides flexible in/out-of-distribution control; (2) masks can be factored into independent image/text distributions—a critical prerequisite for the cross-modal estimator.

**Cross-modal sampling strategy (core efficiency gain):**
- Conventional approach: sample $m$ joint masks $M \sim \mathbb{P}_p$, obtaining $m$ game values.
- FIxLIP proposes: independently sample $m_{\mathcal{I}}$ image masks and $m_{\mathcal{T}}$ text masks, and take all combinations.
- Only $m_{\mathcal{I}} + m_{\mathcal{T}}$ model forward passes are required (image and text encoders operate independently), yielding $m_{\mathcal{I}} \times m_{\mathcal{T}}$ game values.
- **Theoretical guarantee** (Theorem 2): the cross-modal estimator is unbiased, with variance bounded between model-agnostic estimators using $m$ and $m_{\mathcal{I}} \cdot m_{\mathcal{T}}$ samples.
- Practical speedup: **5–20×**.

**Scalability strategies:**
- The explanation basis grows quadratically: ViT-B/16 yields 196+30=226 tokens → 25,425 interactions.
- Two-step filtering: (1) compute interactions only for the top-k tokens by first-order attribution; (2) or restrict to cross-modal interactions only.
- Greedy subset selection: find the maximum/minimum similarity subgraph in the explanation graph for evaluation and visualization.

### Loss & Training
FIxLIP involves no model training and is a post-hoc explanation method. The core optimization objective is to approximate $\hat{\mathfrak{F}}_p^{(m_{\mathcal{I}}, m_{\mathcal{T}})}$ via weighted least-squares (WLS) regression.

Experimental configuration:
- Models explained: CLIP ViT-B/32, ViT-B/16; SigLIP, SigLIP-2 ViT-B/32, ViT-L/16
- Cross-modal estimator budget: $2^{21}$; Shapley interaction estimator budget: $2^{17}$ (comparable runtime)
- $p \in \{0.3, 0.5, 0.7\}$; different values of $p$ incur no additional computational cost.

## Key Experimental Results

### Main Results — Pointing Game Recognition (CLIP ViT-B/32, ImageNet-1k)

| Method | 1 object | 2 objects | 3 objects | 4 objects |
|--------|----------|-----------|-----------|-----------|
| GAME | .61 | .43 | .33 | .28 |
| Grad-ECLIP | .68 | .45 | .33 | .28 |
| Shapley values | .70 | .56 | .46 | .37 |
| exCLIP | .73 | **.88** | **.89** | **.92** |
| **FIxLIP (Shapley interaction)** | **.83** | .82 | .84 | .86 |
| **FIxLIP (w.Banzhaf p=0.7)** | .83 | .81 | .83 | .85 |

**Key Findings**: First-order methods degrade sharply in multi-object scenarios (approaching the random baseline of 0.25), whereas second-order interaction methods maintain high recognition accuracy.

### Insertion/Deletion Curves (CLIP ViT-B/32, MS COCO)

| Method | AID Score ↑ |
|--------|-------------|
| GAME | Low (cannot recover non-linear ranking) |
| Grad-ECLIP | Low |
| exCLIP | Medium (approximates cross-modal interactions only) |
| **FIxLIP (p=0.5)** | **Highest** (faithfully recovers optimal subset explanation) |

FIxLIP not only identifies the most important tokens whose removal causes a significant drop in similarity, but also identifies the least important tokens whose removal may improve predictions—a capability beyond gradient-based methods.

### Ablation Study — Computational Efficiency

| Estimator | Inference Speedup | Overall Speedup |
|-----------|------------------|-----------------|
| Model-agnostic | 1× | 1× |
| **Cross-modal** | **20×** | **5×** |

On SigLIP-2 ViT-B/32, compared to first-order attribution methods (~1 second), FIxLIP achieves acceptable computational overhead via cross-modal sampling under large budgets.

### Key Findings
- First-order attribution methods (e.g., Grad-ECLIP) achieve a p-faithfulness correlation of only ~0.5 for VLEs, whereas FIxLIP approaches 1.0.
- exCLIP fails AID ranking because it approximates only cross-modal interactions, neglecting first-order effects and intra-modal interactions.
- SigLIP-2 substantially outperforms CLIP in the Pointing Game (.90 vs .83 for the single-object setting), indicating that SigLIP-2 learns more accurate cross-modal correspondences.

## Highlights & Insights
1. **Rigorous game-theoretic foundation**: This work is the first to extend weighted Banzhaf interactions to VLE interpretation, satisfying desirable axiomatic properties including linearity, symmetry, and dummy player.
2. **Elegant cross-modal sampling strategy**: The independence of VLE image/text encoders is exploited to obtain $m^2/4$ game values with only $m$ forward passes, achieving both theoretical and practical efficiency.
3. **Evaluation metric contributions**: Pointing Game and Insertion/Deletion curves are generalized to second-order interaction explanations, filling a gap in the evaluation of interaction-based explanations.
4. **Practical visualization**: Multiple levels of understanding are enabled—viewing interaction heatmaps conditioned on individual tokens, or traversing the complete graph to find high/low similarity subsets.

## Limitations & Future Work
- Faithfulness degrades as image resolution increases (more patches)—further extension to high-resolution models is needed.
- The quadratic growth of the explanation basis limits scalability; the current top-k filtering is a heuristic solution.
- Evaluation is limited to the CLIP and SigLIP families; generative VLM architectures such as LLaVA have not been tested.
- Real-world application scenarios (e.g., explaining erroneous diagnoses in medical imaging VQA) have not been demonstrated.

## Related Work & Insights
- Compared to exCLIP (cross-modal interactions only), FIxLIP includes intra-modal interactions and first-order effects, providing a more complete decomposition.
- Relationship to SHAP/KernelSHAP: FIxLIP-0.5 is equivalent to Faith-Banzhaf (a KernelSHAP variant); $p \neq 0.5$ constitutes a weighted generalization.
- Implications for VLM development: comparing FIxLIP explanations across models can reveal architectural differences (e.g., attention patterns in CLIP vs. SigLIP-2).
- The framework generalizes to the interpretation of other multimodal encoders such as video-language and audio-language models.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of weighted Banzhaf interactions and cross-modal sampling is original, with substantial theoretical contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three evaluation metrics, multiple models, multiple datasets, and comprehensive efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, excellent visualizations, clear structure)
- Value: ⭐⭐⭐⭐ (Establishes a new standard for VLE interpretation, though practical impact remains to be demonstrated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[NeurIPS 2025\] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](../../ICLR2026/interpretability/conjuring_semantic_similarity.md)
- [\[ICLR 2026\] Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity](../../ICLR2026/interpretability/rethinking_layer_relevance_in_large_language_models_beyond_cosine_similarity.md)

</div>

<!-- RELATED:END -->
