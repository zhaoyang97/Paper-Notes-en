---
title: >-
  [Paper Note] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education
description: >-
  [AAAI2026][Causal Inference][Knowledge Tracing] This paper proposes KTCF, a counterfactual explanation generation method for Knowledge Tracing (KT) that leverages inter-concept relationships to produce sparse and actiona…
tags:
  - "AAAI2026"
  - "Causal Inference"
  - "Knowledge Tracing"
  - "Counterfactual Explanation"
  - "XAI"
  - "Actionable Recourse"
  - "Education"
date: 2026-05-08
content_hash: 8ecd352ff2a5e93b
---

# KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education

**Conference**: AAAI2026  
**arXiv**: [2601.09156](https://arxiv.org/abs/2601.09156)  
**Code**: To be confirmed  
**Area**: Causal Inference  
**Keywords**: Knowledge Tracing, Counterfactual Explanation, XAI, Actionable Recourse, Education  

## TL;DR
This paper proposes KTCF, a counterfactual explanation generation method for Knowledge Tracing (KT) that leverages inter-concept relationships to produce sparse and actionable counterfactuals, subsequently post-processed into sequentially ordered instructional recommendations. KTCF comprehensively outperforms baseline methods across validity, sparsity, and actionability metrics.

## Background & Motivation
- Deep learning-based Knowledge Tracing (KT) is widely applied in education to model student knowledge states and predict future performance.
- The EU AI Act classifies educational AI as high-risk, requiring interpretability; the U.S. Department of Education likewise emphasizes preserving human decision-making authority.
- Existing KT interpretability research primarily addresses *what* the model attends to (attention heatmaps) or whether it accurately captures learning processes (psychometric parameters).
- Methods addressing *why* and *how* questions—i.e., "why did the model produce this prediction" and "what should a student do to change the outcome"—remain largely unexplored.
- Counterfactual explanations are inherently causal and local, and are readily comprehensible to non-specialist educational stakeholders (students, teachers, parents), yet they have not been adequately explored in the KT domain.

## Core Problem
Given a student's learning history $X^{\text{orig}} = [(kc_1, r_1), \ldots, (kc_t, r_t)]$ and a trained KT model $f$, when the model predicts that the student will answer the target knowledge concept $kc_{\text{target}}$ incorrectly, the goal is to find a minimally modified counterfactual response sequence $R^{\text{cf}}$ that flips the prediction to correct, while satisfying:

1. **Intervention sparsity**: the number of modifications is minimized.
2. **Actionability**: only incorrect-to-correct flips are permitted (not the reverse).
3. **KC-relation consistency**: modified knowledge concepts are related to the target concept in the knowledge graph.
4. **Sequential output**: explanations are presented as ordered instructional steps that minimize the student's total learning burden.

## Method

### Counterfactual Explanation Generation (Algorithm 1)
KTCF formulates counterfactual generation as an optimization problem with a three-term loss:

$$\mathcal{L}_{\text{KTCF}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{spar}} \cdot \mathcal{L}_{\text{spar}} + \lambda_{\text{kc}} \cdot \mathcal{L}_{\text{kc}}$$

- **Prediction loss** $\mathcal{L}_{\text{pred}}$: binary cross-entropy between the KT model's output on the counterfactual input and the target probability of 1.0, driving prediction flipping.
- **Sparsity loss** $\mathcal{L}_{\text{spar}}$: Hamming distance between the original and counterfactual responses, encouraging minimal modification.
- **KC loss** $\mathcal{L}_{\text{kc}}$: penalizes modifications to KCs that are distant from the target KC in the predefined undirected KC relation graph $G_{\text{kc}}$, ensuring structural relevance.

Stochastic optimization is performed with the Adam optimizer. After each update step, an **actionability mask** $\mathbf{m}$ is applied as a projection to ensure that only originally incorrect responses can be modified, achieving 100% actionability.

### Initialization Strategies
Five initialization strategies are compared: Gaussian noise (-rn), random binary (-rand), soft relaxation (-sr), convex combination (-cc), and Gumbel-Sigmoid relaxation (-gs). Experiments show that soft relaxation-based initializations (-rn, -sr, -gs) achieve the best performance.

### Post-processing: Instructional Sequence Generation (Algorithm 2)
Counterfactual explanations are converted into an executable sequential learning path for students:

1. Extract the set of all modified KCs $\overline{V}^{\text{CF}}$ from the counterfactual.
2. Compute pairwise shortest-path distances between all KCs using Dijkstra's algorithm on the KC relation graph.
3. Construct a complete subgraph over these KCs with edge weights equal to shortest-path distances.
4. Apply a greedy strategy starting from the target KC to solve the Traveling Salesman Path Problem (TSPP), then reverse the path.
5. The resulting sequence specifies the recommended order in which the student should review KCs, minimizing total learning burden (total path distance).

## Key Experimental Results

### Datasets and Setup
- **Dataset**: XES3G5M, containing 5,549,635 interaction records from 18,066 students (mathematics domain).
- **KC relation graph**: 1,175 nodes, 1,304 edges.
- **KT model**: DKT (LSTM), test AUC 0.8226, accuracy 0.8253.
- **Evaluation**: 200 instances randomly sampled from students with error rates above 45%.

### Main Results (Table 1)

| Method | Validity↑ | Sparsity↓ | Actionability↓ |
|--------|----------|----------|----------------|
| Wachter-rand | 0.725 | 67.36 | 40.53 |
| DiCE-rand | 0.880 | 75.59 | 35.32 |
| **KTCF-rn** | **0.930** | **49.85** | **0.000** |
| **KTCF-gs** | 0.920 | 49.92 | **0.000** |

- KTCF-rn improves validity by 28.3% and reduces sparsity by 26.0% over Wachter; it improves validity by 5.7% and reduces sparsity by 34.0% over DiCE.
- All KTCF variants achieve an actionability score of 0 (fully eliminating non-actionable modifications), whereas baseline methods contain substantial non-actionable suggestions.
- KTCF-gs is the fastest variant (2.2s), 41.9% faster than Wachter.

### Qualitative Analysis (Table 2)
Using "distinguishing leap years from non-leap years" as the target KC:

- KTCF recommends only 5 KCs (modular arithmetic, calendar cycles, etc.) with a total path distance of 26.
- Wachter recommends 14 KCs (including irrelevant concepts such as tree diagrams and decimal units) with a total path distance of 66.
- DiCE recommends 20 KCs with a total path distance of 79.

## Highlights & Insights
- **First systematic** application of counterfactual XAI to knowledge tracing, providing a complete conceptualization framework (explanandum/explanans).
- **Actionability guarantee**: the masking mechanism achieves 100% actionability, ensuring recommendations exclusively involve practicing incorrect items to mastery and never suggest deliberate errors.
- **Educational theory grounding**: the method is connected to Bloom's mastery learning theory and the 2-Sigma problem, lending it pedagogical theoretical significance.
- **KC-relation constraints**: embedding KC graph structure into the loss function ensures that recommended modifications are coherent within the knowledge topology.
- **Post-processing design**: the use of TSPP for generating optimal learning paths elegantly minimizes the student's total learning burden.

## Limitations & Future Work
- Validation is limited to DKT (LSTM); Transformer-based KT models (e.g., AKT, SAINT) have not been tested, leaving generalizability unverified.
- The KC relation graph depends on predefined structures and is not applicable to datasets lacking such information.
- Counterfactual generation over binary responses remains sensitive to initialization strategies; although the KC loss improves robustness, the underlying issue is not fundamentally resolved.
- The greedy TSPP post-processing yields a suboptimal solution; path quality has room for improvement.
- No real user study has been conducted to validate actual instructional effectiveness (only qualitative case analysis is provided).
- Evaluation is based on only 200 instances, which is a relatively small scale.

## Related Work & Insights
- **vs. Wachter/DiCE**: these general-purpose counterfactual methods do not account for KC relationships or educational actionability constraints, resulting in redundant suggestions and non-actionable modifications. KTCF addresses both issues through the KC loss and actionability mask.
- **vs. attention heatmap explanations** (AKT, EKT, etc.): these answer only "where did the model attend" and cannot provide actionable instructional recommendations.
- **vs. psychometric parameter explanations** (IRT-based): difficulty and discrimination parameters support diagnosis but similarly provide no action guidance for improvement.
- **vs. SHAP/LRP post-hoc explanations** (Lu et al. 2024): these explain feature contribution directions but do not generate counterfactual guidance; Lu et al.'s user study demonstrates that explanations improve trust, and KTCF advances further by providing actionable steps.
- The post-processing idea of TSP-based path optimization generalizes to other scenarios requiring ranked recommendations, such as learning path planning in recommender systems.
- The KC-relation graph constraint is transferable to domains such as medical diagnosis, where a disease relation graph could constrain the plausibility of counterfactual suggestions.
- The actionability mask is a general technique applicable to any counterfactual generation problem with unidirectional actionability constraints.
- A promising future direction is combining KTCF with LLMs to convert counterfactual explanations into natural language instructional feedback.

## Rating
- Novelty: ⭐⭐⭐⭐ (first systematic application of counterfactual XAI to KT with a complete conceptual framework)
- Experimental Thoroughness: ⭐⭐⭐ (ablation studies are sufficient, but evaluation scale is small and user studies are absent)
- Writing Quality: ⭐⭐⭐⭐ (educational theory and technical methodology are naturally integrated; qualitative analysis is intuitive)
- Value: ⭐⭐⭐⭐ (meaningful practical contribution to interpretability in educational AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](../../CVPR2026/causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)

</div>

<!-- RELATED:END -->
