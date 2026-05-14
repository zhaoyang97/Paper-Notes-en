---
title: >-
  [Paper Note] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps
description: >-
  [ICCV 2025][Interpretability][Concept explanation] CE-FAM is a concept explanation method that trains a branch network sharing activation maps with an image classifier to simulate VLM embeddings…
tags:
  - "ICCV 2025"
  - "Interpretability"
  - "Concept explanation"
  - "activation map fusion"
  - "Grad-CAM"
  - "VLM knowledge transfer"
date: 2026-05-08
content_hash: 38d85b91fc0b5063
---

# CE-FAM: Concept-Based Explanation via Fusion of Activation Maps

**Conference**: ICCV 2025
**arXiv**: [2509.23849](https://arxiv.org/abs/2509.23849)
**Code**: None
**Area**: Interpretability
**Keywords**: Concept explanation, activation map fusion, Grad-CAM, VLM knowledge transfer, interpretability

## TL;DR

CE-FAM is a concept explanation method that trains a branch network sharing activation maps with an image classifier to simulate VLM embeddings, establishing a one-to-one correspondence among concept prediction → concept region (weighted sum of activation maps) → concept contribution (effect on classification score). The paper also introduces a novel NRA evaluation metric and surpasses existing methods on zero-shot concept reasoning.

## Background & Motivation

In Explainable AI (XAI), three levels of explanation are needed, yet few methods satisfy all simultaneously:

**What**: Which human-understandable concepts has the model learned?

**Where**: Which image regions correspond to these concepts?

**How**: How much does each concept contribute to the final prediction?

- **Saliency map methods** (Grad-CAM, etc.): can only highlight important regions, leaving the interpretation of "what" to the user.
- **Concept bottleneck models** (CBM, TCAV): can quantify concept contributions but cannot localize concept regions.
- **Dissection methods** (CLIP-Dissect, WWW): associate concepts with individual neurons, but suffer from many-to-many mapping—one concept may relate to multiple neurons, and the optimal correspondence varies across samples.

Core insight: representing a concept with a single activation map is fundamentally limited; concept regions should be expressed via **weighted fusion of activation maps**.

## Method

### Overall Architecture

CE-FAM workflow:
1. **Training**: A branch network learns to map the classifier's multi-layer activation embeddings to the VLM (CLIP) image embedding space.
2. **Concept prediction**: Similarity between the mapped embeddings and concept text embeddings is computed.
3. **Concept region**: Gradients of the concept prediction score are back-propagated to obtain activation map weights, generating concept-specific region maps.
4. **Concept contribution**: The drop in classification score is measured by masking important channels.

### Key Designs

1. **Multi-Layer Concept Learning**

   Unlike conventional methods that use only the last-layer embedding, CE-FAM leverages embedding vectors from all CNN layers to capture features from low-level to high-level:

   $\mathbf{z}^l = \text{AvgPool}(A^l)$
   $\mathbf{z}_{\text{cat}} = \text{Concat}(\mathbf{z}^1, \mathbf{z}^2, \ldots, \mathbf{z}^L)$

   A translator function $h$ (a simple MLP) projects $\mathbf{z}_{\text{cat}}$ into the CLIP embedding space. The training loss is:

   $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{emb}} + \lambda \mathcal{L}_{\text{sim}}$

   where $\mathcal{L}_{\text{emb}} = \text{MSE}(h(\mathbf{z}_{\text{cat}}) - E_{\text{image}}(\mathbf{x}))$ simulates the VLM embedding, and $\mathcal{L}_{\text{sim}} = \text{MSE}(S^t - S_{\text{VLM}}^t)$ aligns concept prediction scores.

   Key finding: even without $\mathcal{L}_{\text{sim}}$ (i.e., without requiring a predefined concept set), the method still outperforms existing approaches—enabling **zero-shot reasoning** over concept labels.

2. **Concept Region via Activation Map Fusion**

   Extending the Grad-CAM idea, a region map is generated for each concept $t$ at each layer $l$:

   $R_t^l = \text{ReLU}\left(\sum_k \beta_k^t A_k^l\right)$
   $\beta_k^t = \frac{1}{\gamma} \sum_i \sum_j \frac{\partial S^t}{\partial A_k^l(i,j)}$

   Gradients of the concept prediction score $S^t$ (rather than the class prediction score $p^c$) serve as weights, producing concept-specific region maps.

   **Layer selection**: The most relevant layer is selected via a relevance score—Top-K important channels are masked one by one, the AUC of the resulting drop in $S^t$ is measured, and the layer with the highest AUC is selected.

3. **Concept Contribution Quantification**

   Concept contribution can be quantified for any layer, not only the last:
   - The computation mirrors the relevance score, but targets the class prediction score $p^c$.
   - The most important channels are progressively masked, and the AUC of the $p^c$ drop curve is recorded.
   - Negative contribution (score increases after masking) is also informative, indicating the concept has a negative influence on the prediction.

4. **NRA Evaluation Metric (Normalized Region Accuracy)**

   Limitations of existing metrics: IoU is sensitive to threshold selection; VEA (AUC) yields high scores even for random results when the mask region is large.

   NRA eliminates the influence of sample distribution through normalization:

   $\text{NRA} = \frac{\text{AUC} - \text{AUC}_{\text{low}}}{\text{AUC}_{\text{high}} - \text{AUC}_{\text{low}}}$

   where $\text{AUC}_{\text{high}}$ is the AUC under the ideal (GT segmentation) condition and $\text{AUC}_{\text{low}}$ is the AUC for a random region. NRA measures the relative position of the predicted region between random and ideal.

### Loss & Training

- Dataset: ImageNet for concept learning training
- Optimizer: SGD, initial LR 0.1, warmed up to 0.2
- Maximum 150 epochs + early stopping (LR × 0.1 if validation loss does not decrease for 4 consecutive epochs)
- $\lambda = 0.001$

## Key Experimental Results

### Main Results

Concept region evaluation on the Broden dataset (ResNet50 classifier + CLIP VLM):

| Method | EPG(Object) | EPG(Avg) | NRA(Object) | NRA(Avg) | Hit Rate(Object) | Hit Rate(Avg) |
|---|---|---|---|---|---|---|
| CLIP-Dissect | 0.197 | 0.146 | 0.327 | 0.334 | 0.215 | 0.199 |
| WWW | 0.179 | 0.117 | 0.322 | 0.278 | 0.154 | 0.114 |
| **CE-FAM** | **0.233** | **0.154** | **0.459** | **0.361** | **0.436** | **0.247** |

CE-FAM exceeds the best baseline by 8% on NRA and 13% on Hit Rate (proportion of samples with NRA > 0.5).

Evaluation on the ImageNet-S dataset (ViT-B/16 classifier):

| Method | EPG | NRA | Hit Rate |
|---|---|---|---|
| CLIP-Dissect | 0.047 | 0.105 | 0.013 |
| WWW | 0.076 | 0.232 | 0.017 |
| **CE-FAM** | **0.138** | **0.273** | **0.193** |

The advantage is particularly pronounced on ViT, where single-channel activation maps are noisier.

### Ablation Study

Concept region evaluation under different configurations (ResNet50):

| VLM | Sim Loss | Multi-Layer | EPG Avg | NRA Avg | Hit Rate Avg |
|---|---|---|---|---|---|
| CLIP | - | - | 0.152 | 0.338 | 0.209 |
| CLIP | ✓ | - | 0.156 | 0.348 | 0.234 |
| CLIP | ✓ | ✓ | 0.154 | 0.361 | 0.247 |
| SigLIP | - | - | 0.151 | 0.383 | 0.283 |
| SigLIP | ✓ | ✓ | **0.157** | **0.388** | **0.295** |

### Key Findings

- **Activation map fusion substantially outperforms single-channel association**: Hit Rate improves from 11.4% (WWW) to 24.7%, demonstrating that concepts should not be represented by individual neurons.
- **Multi-layer features are complementary**: Using multi-layer embeddings improves NRA by 2.3% and Hit Rate by 3.8% over using only the last layer; low-level features are important for concepts such as color.
- **Zero-shot concept reasoning is effective**: Without $\mathcal{L}_{\text{sim}}$, the method still outperforms existing methods that require concept labels.
- **VLM choice has a significant impact**: SigLIP outperforms CLIP; improvements in 2D representation quality in the VLM directly enhance concept learning.
- **High training efficiency**: The method surpasses existing approaches within only a few epochs.
- **Misclassification analysis**: Concept contributions can explain misclassification (e.g., an indigo bunting misclassified as a goldfinch due to negative contribution of the yellow concept).

## Highlights & Insights

1. **First to establish a one-to-one correspondence among concept label, region, and contribution**: completing the "What-Where-How" trinity of concept explanation.
2. **General framework**: applicable to any image classifier using activation maps (both CNN and ViT).
3. **Well-motivated NRA metric**: normalization effectively removes the influence of sample distribution on evaluation.
4. **Zero-shot capability**: no concept-annotated dataset is required; arbitrary concepts can be handled using VLM knowledge alone.

## Limitations & Future Work

- Concept expressiveness is bounded by VLM performance: CLIP tends to be dominated by salient image features, making fine-grained concept prediction difficult.
- Sensitivity to concept set selection: an overly large concept set introduces irrelevant concept noise.
- Concept contribution lacks ground-truth data for quantitative validation, making verification challenging.
- Computational cost scales with the number of layers and concepts; optimization is needed for large-scale deployment.

## Related Work & Insights

- The approach of extending Grad-CAM from "class explanation" to "concept explanation" is natural and elegant.
- Representing concepts as weighted fusions of activation maps effectively continues the multi-channel representation philosophy of Net2Vec.
- The NRA metric can be generalized to other XAI tasks requiring evaluation of region accuracy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Representing concept regions via activation map fusion and the one-to-one correspondence framework are novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets (Broden and ImageNet-S), multiple classifiers, multiple VLMs, ablation studies, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear, and the evaluation metric design is well-justified.
- **Value**: ⭐⭐⭐⭐ The general framework is applicable to model debugging and misclassification diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](../../NeurIPS2025/interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)

</div>

<!-- RELATED:END -->
