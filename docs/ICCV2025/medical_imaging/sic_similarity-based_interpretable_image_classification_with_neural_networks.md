---
title: >-
  [Paper Note] SIC: Similarity-Based Interpretable Image Classification with Neural Networks
description: >-
  [ICCV 2025][Medical Imaging][Interpretability] This paper proposes SIC, an inherently interpretable neural network that simultaneously provides local, global, and faithful explanations. By extracting class-representative support vectors from training images and computing input-to-support-vector similarities via B-cos transformations for classification, SIC achieves accuracy comparable to black-box models while delivering pixel-level contribution maps and case-based global explanations. On the FunnyBirds benchmark, SIC outperforms ProtoPNet on 8 out of 9 interpretability metrics.
tags:
  - ICCV 2025
  - Medical Imaging
  - Interpretability
  - Similarity-Based Classification
  - B-cos Networks
  - Support Vectors
  - Case-Based Reasoning
date: 2026-05-08
content_hash: e922853c46b4b610
---

# SIC: Similarity-Based Interpretable Image Classification with Neural Networks

**Conference**: ICCV 2025
**arXiv**: [2501.17328](https://arxiv.org/abs/2501.17328)
**Code**: [github.com/ai-med/SIC](https://github.com/ai-med/SIC)
**Area**: Medical Imaging
**Keywords**: Interpretability, Similarity-Based Classification, B-cos Networks, Support Vectors, Case-Based Reasoning

## TL;DR

This paper proposes SIC, an inherently interpretable neural network that simultaneously provides local, global, and faithful explanations. By extracting class-representative support vectors from training images and computing input-to-support-vector similarities via B-cos transformations for classification, SIC achieves accuracy comparable to black-box models while delivering pixel-level contribution maps and case-based global explanations. On the FunnyBirds benchmark, SIC outperforms ProtoPNet on 8 out of 9 interpretability metrics.

## Background & Motivation

### Problem Definition

Deploying deep learning in high-stakes domains such as medical imaging requires transparent decision-making processes. XAI methods fall into two categories:
- **Post-hoc explanations**: Approximate feature contributions for black-box models (e.g., SHAP, Grad-CAM), but suffer from unavoidable approximation errors on high-dimensional inputs.
- **Inherently interpretable models**: Explanations arise directly from the model design, requiring no approximation.

An ideal inherently interpretable model should simultaneously satisfy three properties: **local explanations** (rationale for individual predictions), **global explanations** (summary of overall model behavior), and **faithfulness** (explanations satisfying mathematical axioms such as completeness and sensitivity).

### Limitations of Prior Work

| Method | Local | Global | Faithful | Core Problem |
|--------|-------|--------|----------|--------------|
| NW-Head | ✓ | ✓ | ✗ | Uses softmax-normalized Euclidean distance with no pixel-level explanation; still assigns high confidence under adversarial attacks |
| ProtoPNet | ✓ | ✓ | ✗ | Upsamples distance maps from latent space to image space, but spatial correspondence is not guaranteed, potentially misleading explanations |
| BagNet | ✓ | ✗ | ✓ | Classifies local patches independently and aggregates, lacking global explanations |
| B-cos | ✓ | ✗ | (✓) | Summarizes each forward pass as a linear transformation for local explanations, but the weight matrix is input-dependent, lacking global explanations |

### Core Motivation

**Key Insight**: Combining a B-cos network (providing faithful pixel-level explanations) with a Nadaraya-Watson classification head (providing case-based global explanations) can simultaneously satisfy all three requirements. The key innovation is an *evidence predictor* that maps features to non-negative vectors, ensuring similarity computations naturally exclude negative contributions. Fixed class-representative support vectors are selected via k-means clustering, reducing complex model behavior to the intuitive reasoning: "which parts of this test image resemble which parts of which training images."

## Method

### Overall Architecture

SIC consists of three components:
1. **B-cos feature extractor** $\mathcal{F}_\theta$: Maps images to latent vectors $f$.
2. **Evidence predictor** $\mathcal{E}$: Transforms $f$ into a non-negative vector $f^+$ for similarity computation against support vectors.
3. **Class logits**: Obtained by summing per-class support vector similarities.

### Key Designs

#### 1. B-cos Feature Extractor

- **Function**: Extracts latent representations of images while ensuring each forward pass can be summarized as a single linear equation, enabling faithful pixel-level explanations.
- **Mechanism**:

  The B-cos transformation redefines the scalar product by introducing an alignment exponent $B$ to enhance weight-input alignment:

  $$\text{B-cos}(x; w) = \|x\| \cdot |\cos(x, \hat{w})|^B \times \text{sgn}(\cos(x, \hat{w}))$$

  where $\hat{w} = w/\|w\|$. Each layer's computation can be expressed as an input-dependent linear equation $\text{B-cos}(x, \mathbf{W}) = \tilde{\mathbf{W}}(x) x$, with:

  $$\tilde{\mathbf{W}}(x) = |\cos(x, \hat{\mathbf{W}})|^{B-1} \odot \hat{\mathbf{W}}$$

  For an $L$-layer network, the entire forward pass is summarized as:

  $$\mathcal{F}_\theta(x) = \left(\prod_{j=1}^{L} \tilde{\mathbf{W}_j}(a_j)\right) x = \mathbf{W}_{1 \rightarrow L}(x) \cdot x$$

  The contribution map for pixel $(m,n)$ is:

  $$\phi_j^l(x)_{(m,n)} = \sum_{ch} \left[ [\mathbf{W}_{1 \rightarrow l}]_j^T \odot x \right]_{(ch,m,n)}$$

  Inputs are encoded as 6-channel tensors $[R, G, B, 1-R, 1-G, 1-B]$ to uniquely encode color and avoid bias toward bright regions.

- **Design Motivation**: The B-cos transformation compels the network to focus on features most aligned with its weights, naturally learning to attend to the most salient data characteristics during training. The linear summarization property guarantees that explanations are exact descriptions of the forward pass, not approximations.

#### 2. Evidence Predictor and Support Vector Classification

- **Function**: Maps real-valued features to non-negative vectors, extracts fixed class-representative support vectors, and performs classification via similarity computation.
- **Mechanism**:

  The evidence predictor comprises two components:
  - A **non-negative mapping** $\oplus: \mathbb{R}^d \rightarrow \mathbb{R}_{\geq 0}^d$ transforming feature $f$ into $f^+ = \oplus(f)$.
  - A **similarity measure** $sim(f^+, v_i^c)$ computing the similarity between the input and each support vector.

  The class logit is defined as:

  $$\mu_c = b + \sum_{v_i^c} \frac{sim(f^+, v_i^c)}{\mathcal{T}}$$

  where $b$ is a fixed bias, $\mathcal{T}$ is a temperature parameter, and $v_i^c$ is the $i$-th support vector for class $c$.

  **Support vector selection**: After each epoch, k-means clustering is applied to the features of all training samples per class, and the nearest real sample feature to each cluster centroid is selected as a support vector:

  $$v_i^c = \arg\min_{f_k^+ | y_k = c} \|f_k^+ - \gamma_i^c\|_2$$

  This ensures support vectors correspond to actual training samples (rather than artificially constructed prototypes), facilitating case-based reasoning.

- **Design Motivation**:
  - The non-negative mapping $\oplus$ ensures only positive contributions can increase class probability, addressing the issue in NW-Head where adversarial attacks still yield high confidence under softmax + Euclidean distance.
  - K-means selection ensures support vectors capture intra-class diversity while limiting the number of explanations to be examined to $N_s$.
  - The temperature parameter $\mathcal{T}$ controls logit magnitude, preventing negative contributions in the B-cos transformation (an excessively small temperature causes the model to scale via negative contributions).

#### 3. Three-Level Explanation Mechanism

- **Function**: Provides a complete explanation chain combining global, local, and pixel-level interpretations.
- **Mechanism**:

  **Global explanations** (support vector contribution maps): For each support vector $v_i^c$, the B-cos contribution map is computed from the corresponding training image. Since $v_i^c$ is perfectly aligned with itself ($\cos = 1$), the B-cos transformation reduces to $\text{B-cos}(v_i^c, \hat{v}_i^c) = \|v_i^c\|$. The contribution map reveals what the model "learned" from each support sample — which pixel regions are encoded as representative features of that class.

  **Local explanations** follow a three-level structure:
  1. **Support evidence**: $\frac{sim(f^+, v_i^c)}{\mathcal{T}}$ quantifies the alignment between the test image and each support sample, as well as each support vector's proportional contribution to the log-probability.
  2. **Test contribution map**: Exploiting the linearity of B-cos, pixel-level contributions to the class logit are computed, revealing which parts of the test image align with the support vectors.
  3. **Support contribution map**: A global RGBA explanation of the support vector, allowing users to inspect the intersection of support and test contributions.

  **Theoretical guarantee**: SIC's explanations are proven to satisfy the 6 axioms proposed by Sundararajan et al. — completeness, sensitivity, implementation invariance, dummy, linearity, and symmetry-preserving.

- **Design Motivation**: The three-level explanation constitutes a complete reasoning chain — global explanations inform developers about "what the model learned," local explanations inform users about "why this prediction," and pixel-level contribution maps indicate "which specific part of the image the model attended to." Case-based reasoning ("this part of the image resembles that part of a training image") is the most intuitive mode of understanding for humans.

### Loss & Training

- **Loss function**: Binary cross-entropy loss (produces stronger alignment pressure than standard multi-class cross-entropy).
- **Training procedure**:
  1. Train the B-cos backbone and evidence predictor normally.
  2. After each epoch, recompute all training sample features and update support vectors via k-means clustering.
  3. During training, support vectors are randomly sampled per batch; fixed k-means support vectors are used at test time.
- **Architecture**: Supports three backbones — DenseNet121 (~8M), ResNet50 (~26M), and Hybrid ViT (~81M).
- **Number of support vectors** $N_s$: Fixed per class.

## Key Experimental Results

### Main Results

**Accuracy comparison across three datasets with ResNet50 backbone**:

| Method | Pascal VOC (mAP) | Stanford Dogs (Acc) | RSNA (Bal.Acc) | Interpretability |
|--------|-----------------|--------------------|--------------|-----------------| 
| Black-box ResNet50 | ~83% | ~83% | ~80% | None |
| ProtoPNet | ~80% | ~75% | ~76% | Local+Global (unfaithful) |
| BagNet17 | ~78% | ~72% | ~74% | Local (faithful) |
| NW-Head | ~82% | ~80% | ~78% | Local+Global (unfaithful) |
| B-cos | ~83% | ~82% | ~80% | Local (faithful) |
| **SIC** | **~83%** | **~79%** | **~79%** | **Local+Global+Faithful** |

SIC matches the black-box model on Pascal VOC and RSNA (≥+0.29% and −0.65%, respectively) and is slightly lower on Stanford Dogs (−3.7%), yet **outperforms 9 of the 14 compared models**.

### Ablation Study

**FunnyBirds interpretability benchmark evaluation (ResNet50 backbone)**:

| Metric | ProtoPNet | BagNet | B-cos | **SIC** |
|--------|----------|--------|-------|--------|
| Accuracy | ~92% | **~97%** | ~95% | ~96% |
| Background Independence | ~0.85 | ~0.95 | ~0.97 | **~0.99** |
| Completeness (avg) | ~0.90 | ~0.95 | ~0.93 | **~0.97** |
| Correctness | ~0.65 | **~0.76** | ~0.70 | ~0.72 |
| Contrastivity | ~0.85 | **~0.99** | ~0.90 | ~0.95 |
| CSDC | ~0.88 | ~0.94 | ~0.91 | **~0.96** |
| Preservation Check | ~0.92 | ~0.96 | ~0.95 | **~0.98** |
| Deletion Check | ~0.90 | ~0.96 | ~0.93 | **~0.97** |
| Distractability | ~0.88 | ~0.94 | ~0.94 | **~0.96** |

SIC **outperforms** ProtoPNet and B-cos on **8 out of 9** interpretability metrics, falling slightly behind BagNet only on Correctness and Contrastivity.

### Key Findings

1. **Accuracy-interpretability trade-off**: SIC incurs only a 0–3.7% accuracy loss across most datasets while gaining complete interpretability.
2. **Near-perfect background independence (0.99)**: Demonstrates that support vectors almost entirely exclude background features.
3. **High completeness (0.97)**: Indicates that support vectors encode all class-representative features.
4. **Diversity of support vectors**: K-means clustering effectively captures intra-class diversity (e.g., dogs from different angles, X-rays under different lighting conditions).
5. **Attention shifts across classes**: In multi-label classification (Pascal VOC), the model focuses on different image regions for different class predictions, demonstrating that latent vectors efficiently encode multi-class features.

## Highlights & Insights

1. **First to simultaneously satisfy all three interpretability requirements**: The unification of local, global, and faithful explanations represents an important theoretical contribution.
2. **Practical value of case-based reasoning**: In medical image diagnosis, explanations of the form "this prediction is based on this region of the X-ray resembling a corresponding region from a training patient" are highly intuitive for clinicians.
3. **Model debugging applications**: Analyzing the similarity matrix of support vectors and t-SNE projections can reveal class confusion in latent space (e.g., support vectors 0 and 3 from the RSNA dataset belong to different classes yet appear close in projection).
4. **Theoretical contribution**: SIC explanations are rigorously proven to satisfy 6 axioms, providing, for the first time, formal guarantees for case-based interpretable models.

## Limitations & Future Work

1. **Scalability with number of classes**: The overhead of k-means clustering grows with the number of classes, though experiments on Stanford Dogs (120 classes) demonstrate feasibility at moderate scale.
2. **Most pronounced accuracy drop on Stanford Dogs (−3.7%)**: The alignment constraint imposed by B-cos limits feature extraction flexibility in fine-grained classification tasks.
3. **Manual specification of support vector count**: The optimal value of $N_s$ depends on intra-class diversity; no adaptive selection mechanism currently exists.
4. **B-cos training constraints**: The B-cos transformation requires specialized scalar products across all layers, restricting the range of applicable backbone architectures.
5. **Temperature parameter sensitivity**: An excessively small temperature introduces negative contributions in support vector contribution maps, requiring careful tuning.

## Related Work & Insights

- **Fundamental distinction from ProtoPNet**: ProtoPNet learns class-specific prototype "parts" and provides explanations via upsampled distance maps. SIC extracts support vectors from complete images and provides faithful pixel-level explanations via B-cos linear summarization, avoiding spatial alignment issues.
- **Complementary relationship with B-cos**: B-cos provides faithful local explanations but lacks global interpretability. SIC extends it by introducing fixed support vectors that add a global explanation dimension.
- **Implications for medical applications**: The RSNA experiments demonstrate how to verify that the model genuinely attends to lesion regions (via the intersection of contribution maps and bounding boxes), a validation approach of considerable value for clinical deployment evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Although the combination of B-cos and NW-Head builds on existing work, the introduction of the evidence predictor and support vector selection achieves a unified three-level explanation with solid theoretical guarantees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across three datasets, three backbones, and the FunnyBirds interpretability benchmark; however, direct comparison with additional interpretable methods (e.g., Concept Bottleneck Models) is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear positioning (Table 1 is immediately informative), rich explanation examples with insightful analysis (e.g., multi-label analysis on Pascal VOC).
- **Value**: ⭐⭐⭐⭐ — Offers practical deployment value for interpretable classification in high-stakes domains such as medical imaging; theoretical guarantees enhance trustworthiness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](../../NeurIPS2025/medical_imaging/firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/medical_imaging/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)

</div>

<!-- RELATED:END -->
