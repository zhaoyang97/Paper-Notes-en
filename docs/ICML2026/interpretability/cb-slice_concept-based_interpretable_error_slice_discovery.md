---
title: >-
  [Paper Note] CB-SLICE: Concept-Based Interpretable Error Slice Discovery
description: >-
  [ICML2026][Interpretability][Error Slice Discovery] CB-SLICE utilizes the concept prediction space of Concept Bottleneck Models (CBM) to discover and explain systematic error slices in deep learning models. Through a thr…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Error Slice Discovery"
  - "Concept Bottleneck Models"
  - "Model Debugging"
  - "Bias Detection"
  - "Explainable AI"
date: 2026-05-08
content_hash: 3fe49efaad7fda95
---

# CB-SLICE: Concept-Based Interpretable Error Slice Discovery

**Conference**: ICML2026  
**arXiv**: [2605.29836](https://arxiv.org/abs/2605.29836)  
**Code**: https://github.com/yaelkon/CB-SLICE  
**Area**: Interpretability  
**Keywords**: Error Slice Discovery, Concept Bottleneck Models, Model Debugging, Bias Detection, Explainable AI  

## TL;DR

CB-SLICE utilizes the concept prediction space of Concept Bottleneck Models (CBM) to discover and explain systematic error slices in deep learning models. Through a three-step pipeline—filtering error-prone concepts, clustering with GMM to form slices, and providing keyword-based concept explanations—it consistently outperforms existing methods across multiple benchmarks while providing faithful explanations directly rooted in the model's internal decision logic.

## Background & Motivation

**Background**: Although deep learning models excel in average performance, they often exhibit systematic errors on specific data subgroups (error slices). Existing Systematic Error Discovery Methods (SDM) such as Domino, GEORGE, and Spotlight have been able to identify these failure modes to some extent.

**Limitations of Prior Work**: Existing SDMs typically rely on auxiliary language models (e.g., ClipCap) to generate explanations. However, these explanations are decoupled from the internal reasoning process of the model under analysis—they are merely indirect approximations of error sources, which can be inaccurate or even misleading. Worse, the auxiliary models themselves may introduce additional biases, further reducing the reliability of explanations.

**Key Challenge**: Error slice discovery needs to solve two problems simultaneously: (i) finding subsets of error samples that share a semantic failure mode, and (ii) explaining the causes of failure in a human-understandable way. Existing methods separate these two steps, resulting in explanations that are unfaithful to the model's actual decision-making process.

**Goal**: To design a framework that unifies slice discovery and bias explanation within the model's internal representation space, making the explanations directly linked to the model's decision logic.

**Key Insight**: Concept Bottleneck Models (CBM) first predict human-understandable concepts (e.g., "dark skin", "asymmetric") and then perform classification based on these concept predictions. This structured prediction flow naturally establishes a transparent link between model decisions and semantic concepts. When downstream predictions depend on intermediate concept predictions, systematic errors must originate from the concept prediction stage.

**Core Idea**: Perform error slice discovery and explanation within the concept logit space of the CBM, transforming SDM from a "post-hoc description" into a "model-aware" process.

## Method

### Overall Architecture

CB-SLICE takes as input a trained CBM $\mathcal{M}_\theta = (g, f)$ (concept encoder $g$ + label predictor $f$) and a set of misclassified samples from the validation set $\Psi_{\text{val}}$. It outputs a set of error slices with keyword-based concept explanations through three steps. The entire process is completed entirely within the concept representation space of the CBM, without relying on any external auxiliary models.

### Key Designs

1.  **Error-Prone Concept Filtering (ECTP Filtering)**:
    *   **Function**: Filters the most likely subset of concepts $C_{\text{err}}$ from all $k$ concepts that lead to downstream misclassification.
    *   **Mechanism**: Uses the Expected Change in Target Prediction (ECTP) score to measure the change in the downstream prediction distribution after intervening on each concept. For each concept $i$, it calculates $T_i(\hat{\mathbf{c}}) = (1-\hat{c}_i) D_{\text{KL}}(\hat{y}_{\hat{c}_i=0} \| \hat{y}) + \hat{c}_i D_{\text{KL}}(\hat{y}_{\hat{c}_i=1} \| \hat{y})$, and selects the top-$t_e$ concepts based on the per-class average.
    *   **Design Motivation**: Restricting slice formation to error-prone concepts significantly improves discovery quality and avoids noise introduced by irrelevant concepts.

2.  **GMM Clustering to Form Error Slices**:
    *   **Function**: Groups error samples according to shared concept-level error patterns.
    *   **Mechanism**: Maps concept predictions of error samples from the probability space to the logit space $H_{\text{err}} = \sigma^{-1}(\hat{C}_{\text{err}})$, then clusters them using a Gaussian Mixture Model. The optimization objective consists of three parts: the GMM negative log-likelihood $\mathcal{L}_{\text{GMM}}$ to ensure semantic coherence, and two auxiliary classifier losses $\mathcal{L}_{c_{\text{true}}}$ and $\mathcal{L}_{c_{\text{pred}}}$ to ensure that samples within the same slice share both identical ground-truth concept values and predicted concept values. The total loss is $\mathcal{L} = \mathcal{L}_{\text{GMM}} + \lambda(\mathcal{L}_{c_{\text{true}}} + \mathcal{L}_{c_{\text{pred}}})$.
    *   **Design Motivation**: Concept logits encode the model's confidence in the presence of a concept and are approximately Gaussian-distributed, making them suitable for GMM modeling. Auxiliary losses force slices to capture consistent concept-level error patterns rather than just feature similarity.

3.  **Keyword Concept Explanation (ECSA Scoring)**:
    *   **Function**: Extracts the keyword concepts that best explain the formation of each slice.
    *   **Mechanism**: Proposes the Expected Change in Slice Assignment (ECSA) score to measure the change in sample slice assignment probability after intervening on a concept: $\text{ECSA}_i(\mathbf{x}) = \mathbb{E}_{v \sim \text{Bern}(\hat{c}_i)} [D_{\text{KL}}(P(S_j | \mathbf{x}, \hat{c}_i = v) \| P(S_j | \mathbf{x}))]$. The top-$t_k$ concepts by average ECSA across slice samples are chosen as keywords, and the correctness of each concept's prediction is annotated.
    *   **Design Motivation**: Keyword concepts not only indicate "what attributes this slice relates to" but also "whether the model's prediction on this attribute was correct or incorrect," distinguishing between two failure types: "errors caused by concept misprediction" and "errors caused by rare concept combinations."

### Slice Prioritization Strategy

To avoid the burden of analyzing too many slices, CB-SLICE proposes a Slice Informativeness score $\text{SI}_j = \rho \cdot \frac{1}{2}(\text{MC}_j + \frac{1+\text{SC}_j}{2})$, which considers Misprediction Consistency (MC, based on the entropy of predicted label distribution within the slice) and Semantic Compactness (SC, based on cosine similarity of slice members to the centroid), with a penalty factor $\rho$ to down-weight overly small slices.

## Key Experimental Results

### Main Results

On four datasets—Waterbirds, CelebA, MetaShift, and MNIST-Sum—CB-SLICE was compared against Domino, GEORGE, HiBug2, Spotlight, and K-Means using CBMs (trained with Sequential and Joint strategies):

| Dataset | Model | CB-SLICE Prec@10 | Prev. SOTA Prec@10 | CB-SLICE MGF | Prev. SOTA MGF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Waterbirds | CBM+Seq | **0.78** | 0.72 (Domino) | **0.70** | 0.25 (HiBug2) |
| Waterbirds | CBM+Joint | **0.83** | 0.62 (Domino) | **0.76** | 0.25 (HiBug2) |
| CelebA | CBM+Seq | **0.92** | 0.63 (Domino) | **0.66** | 0.51 (HiBug2) |
| MetaShift | CBM+Joint | **0.91** | 0.86 (Domino) | **0.86** | 0.72 (GEORGE) |
| MNIST-Sum | CBM+Joint | **1.00** | 0.50 (HiBug2) | **0.95** | 0.56 (HiBug2) |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Using all concepts (No ECTP) | Significant Drop | Irrelevant concepts introduce noise, reducing slice quality |
| Only $\mathcal{L}_{\text{GMM}}$ | Drop | Lack of concept-level error pattern alignment |
| $\mathcal{L}_{\text{GMM}} + \mathcal{L}_{c_{\text{true}}}$ | Suboptimal | Lack of predicted value consistency constraint |
| $\mathcal{L}_{\text{GMM}} + \mathcal{L}_{c_{\text{true}}} + \mathcal{L}_{c_{\text{pred}}}$ | **Optimal** | Synergy of three losses yields highest and most stable performance |
| GMM vs Linear Clustering | GMM Superior | GMM consistently outperforms linear alternatives on auxiliary classifier accuracy |

### Key Findings

*   CB-SLICE leads across the board in Precision@10, with particularly significant gains on CelebA (+29%) and MNIST-Sum (+50%), indicating highly accurate localization of error slices.
*   The massive advantage in the MGF metric (e.g., 0.70 vs 0.25 on Waterbirds) demonstrates that the slices discovered by CB-SLICE are highly homogeneous and do not mix samples from different failure modes.
*   Keyword concepts can distinguish between two failure modes: errors driven by concept misprediction (e.g., "medium size" being mispredicted in Waterbirds) and errors resulting from rare concept combinations (e.g., insufficient training for (1,1) combinations in MNIST-Sum).
*   The alignment between loss convergence points and evaluation metric saturation points provides a practical criterion for selecting the number of slices $t_g$ without needing labels.

## Highlights & Insights

*   **Model-Aware Explanation Paradigm**: CB-SLICE transforms error explanation from a "post-hoc description" to a "model-aware" process where explanations originate directly from the model's internal concept predictions, avoiding secondary biases introduced by auxiliary models. This approach can be generalized to any architecture with intermediate interpretable representations.
*   **Distinction Between Two Failure Modes**: By annotating the correctness of keyword concept predictions, CB-SLICE automatically distinguishes between "concept misprediction" and "under-training on rare combinations"—two fundamentally different failure causes that direct different mitigation strategies (refining the concept encoder vs. data augmentation).
*   **Generalization from ECTP to ECSA**: Generalizing the ECTP score (measuring concept influence on downstream predictions) to the ECSA score (measuring concept influence on slice assignment) provides a "intervention-observation" causal reasoning framework transferable to other scenarios requiring attribution analysis.

## Limitations & Future Work

*   CB-SLICE depends on the CBM architecture and requires complete and faithful concept annotations; performance may degrade with concept noise or incompleteness.
*   The requirement to train a CBM adds computational cost, though the performance gap between CBMs and standard DNNs is narrowing.
*   Future work could extend to scenarios with incomplete/noisy concept sets or form a closed loop with downstream bias mitigation strategies (e.g., resampling, data augmentation).

## Related Work & Insights

*   **SDM Series**: Domino discovers slices in CLIP space but uses external explanations; GEORGE uses embedding clustering without explanation; Spotlight finds high-loss regions but lacks discriminative power. CB-SLICE unifies discovery and explanation.
*   **CBM Bias Handling**: Bordt et al. mitigate spurious concepts via pruning; Kim et al. use VLMs to automatically filter concept libraries. CB-SLICE differs by aiming for comprehensive discovery of all failure modes rather than fixing a specific bias.
*   **Inspiration**: Concept bottlenecks are not just interpretability tools but a natural infrastructure for model debugging. Any architecture that decomposes the decision process into interpretable intermediate representations can utilize similar "error analysis in the intermediate representation space" strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)

</div>

<!-- RELATED:END -->
