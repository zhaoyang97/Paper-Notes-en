---
title: >-
  [Paper Note] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation
description: >-
  [AAAI 2026][Interpretability][Melanoma Diagnosis] This paper proposes the CEFM framework, which aligns ViT visual features with ABCD-rule-based clinical features (asymmetry, border…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Melanoma Diagnosis"
  - "Contrastive Learning"
  - "Explainable AI"
  - "ABCD Rule"
  - "Report Generation"
date: 2026-05-08
content_hash: e3e5bb65f6bba34f
---

# Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation

**Conference**: AAAI 2026
**arXiv**: [2512.06105](https://arxiv.org/abs/2512.06105)
**Code**: [https://eattt-wen.github.io/CEFM/](https://eattt-wen.github.io/CEFM/)
**Area**: Interpretability
**Keywords**: Melanoma Diagnosis, Contrastive Learning, Explainable AI, ABCD Rule, Report Generation

## TL;DR
This paper proposes the CEFM framework, which aligns ViT visual features with ABCD-rule-based clinical features (asymmetry, border, color) via cross-modal contrastive learning, and subsequently employs CLIP and DeepSeek to generate structured diagnostic reports. On the ISIC dataset, the framework achieves 92.79% accuracy and 0.961 AUC, with an expert-rated interpretability score of 4.6/5.

## Background & Motivation

**Background**: Deep learning has achieved expert-level performance (≥90% accuracy) in melanoma classification; however, these models remain black boxes, lacking interpretability and thereby limiting clinical adoption. Existing XAI methods such as Grad-CAM only highlight attention regions without establishing semantic associations with clinical diagnostic criteria (i.e., the ABCD rule).

**Limitations of Prior Work**: (a) Heatmap-based methods like Grad-CAM cannot provide clinically actionable explanations—"where the model looks" does not equate to "why it predicts malignancy"; (b) attention-based text generation approaches (e.g., LSTM descriptions) are not anchored to diagnostic standards; (c) methods requiring extensive manual annotation for alignment (e.g., CompA) exhibit poor scalability.

**Key Challenge**: A trust gap exists between a model's high accuracy and its low interpretability—clinicians require explanations aligned with ABCD diagnostic criteria before trusting AI-assisted diagnosis.

**Goal**: To design a framework that explicitly aligns the visual features of deep models with clinical ABC diagnostic criteria and automatically generates structured diagnostic reports.

**Key Insight**: Cross-modal contrastive learning is used to align ViT visual features with quantified ABC clinical features in a shared embedding space, followed by CLIP and an LLM for readable report generation.

**Core Idea**: Anchor black-box visual features to interpretable ABC clinical features via cross-modal contrastive learning, and subsequently employ an LLM to translate these into natural-language diagnostic reports.

## Method

### Overall Architecture
CEFM comprises four interconnected pipelines:
1. **DNN Classification Pipeline**: A ViT encoder extracts visual features → an MLP projection head maps them to a shared space → a classification head produces predictions.
2. **Clinical Interpretation Pipeline**: UltraLight VM-UNet performs coarse segmentation → SAM2 refines the segmentation → ABC features (asymmetry, border curvature, color variation) are computed from the mask.
3. **Contrastive Learning Module**: Visual embeddings are aligned with ABC clinical features in a shared latent space.
4. **Report Generation Module**: CLIP retrieves visual descriptors, and DeepSeek LLM generates structured diagnostic reports.

### Key Designs

1. **Coarse-to-Fine Lesion Segmentation**:

    - **Function**: Precisely segment melanoma regions from dermoscopic images.
    - **Mechanism**: In the first stage, UltraLight VM-UNet (pretrained on ISIC 2018) rapidly generates a coarse mask (DSC 0.89). In the second stage, the coarse mask is used as a pseudo-label to sample foreground/background point prompts for SAM2, which generates multiple candidate masks; the one with the highest IoU is selected.
    - **Design Motivation**: The lightweight UNet is fast but produces rough boundaries, while SAM2 is precise but requires point prompts. The two-stage combination achieves automated, high-accuracy segmentation.

2. **ABC Clinical Feature Quantification**:

    - **Function**: Compute three interpretable clinical metrics from the segmentation mask.
    - **Mechanism**: **Asymmetry (A)**: The lesion image is mirror-flipped along its principal axis, and the proportion of differing pixels is computed as $A = \sum|I(x,y) - I_{\text{mirror}}(x,y)| / \sum M(x,y)$. **Border (B)**: The mean curvature of the lesion contour is computed as $B_2 = \frac{1}{N}\sum \kappa_i$, where $\kappa_i = \Delta\theta_i / \Delta s_i$. **Color (C)**: The standard deviations of hue, saturation, and brightness in HSV space are computed as $\sigma_H, \sigma_S, \sigma_V$.
    - **Design Motivation**: The ABCD rule is the most widely used diagnostic criterion in dermatology; quantifying it as numerical features enables direct correspondence with clinical practice.

3. **Cross-Modal Contrastive Alignment**:

    - **Function**: Align ViT visual embeddings with ABC clinical features in a shared latent space.
    - **Mechanism**: A visual encoder $f_v$ extracts image features $v$, and an MLP $f_c$ encodes clinical features $u$. Both are mapped to $\mathbb{R}^d$ via projection heads $h_v, h_c$, L2-normalized, and bidirectionally aligned using the NT-Xent loss. ViT parameters are frozen; only the projection heads are trained.
    - **Design Motivation**: Contrastive learning is naturally suited for cross-modal alignment—drawing the visual and clinical embeddings of the same patient closer while pushing those of different patients apart, thereby endowing visual features with clinical semantic interpretability.

4. **CLIP + DeepSeek Report Generation**:

    - **Function**: Translate quantified diagnostic results into natural-language reports.
    - **Mechanism**: ABC numerical values are first discretized into five severity levels. Simultaneously, CLIP-ViT-B/16 retrieves clinical descriptors most similar to the image (e.g., "asymmetric shape," "blue-gray areas"). The quantified metrics and CLIP descriptors are combined into a prompt fed to DeepSeek to generate a complete report.
    - **Design Motivation**: Numerical metrics alone are insufficiently intuitive for clinicians; natural-language narratives are needed to support decision-making. CLIP supplements visual cues beyond ABC features (e.g., ulceration, satellite lesions).

### Loss & Training
- Contrastive loss: Bidirectional NT-Xent, with temperature $\tau$ controlling the sharpness of the similarity distribution.
- Training proceeds in two steps: contrastive pre-training to align the feature space (ViT frozen), followed by freezing the projection heads and training the classification head.
- Classification uses the ViT-projected embeddings, ensuring classification and interpretation share the same feature representation.

## Key Experimental Results

### Main Results
Classification performance on the ISIC 2020 dataset:

| Backbone | Accuracy | AUC | Precision | Specificity |
|----------|----------|-----|-----------|-------------|
| ResNet50 | - | - | - | - |
| EfficientNet-B2 | 94.26% | - | - | - |
| **ViT (CEFM)** | **92.79%±0.57%** | **0.961±0.004** | **88.19%** | **97.15%** |

Segmentation performance (UltraLight VM-UNet on ISIC 2018): DSC = 0.8909, Acc = 95.56%, Specificity = 0.9746.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Full CEFM | Structured, coherent, and clinically complete reports |
| w/o CLIP | Reports contain only ABC numerical values; lack visual contextual description |
| w/o Clinical Interpretation | Loss of ABC quantification; relies solely on visual descriptions without numerical grounding |
| w/o DeepSeek | Reports are fragmented; narrative coherence is lost |

### Key Findings
- After contrastive learning, the cosine similarity distributions of positive and negative pairs are clearly separated: positive pairs concentrate at >0.75, while negative pairs concentrate near 0.
- Three dermatology experts rated interpretability at 4.6/5, usefulness of ABC feature analysis at 4.4/5, and clinical applicability at 4.0/5.
- Experts identified the framework as particularly suitable for early triage, assistance for junior clinicians, and longitudinal lesion tracking.
- The D component (dermoscopic structures) of the ABCD rule was excluded due to the lack of fine-grained annotations—an honestly acknowledged limitation.

## Highlights & Insights
- The **clinical-criteria-driven interpretability design** is commendable: rather than post-hoc explanation of a black box, the ABC clinical standards are embedded into the model training process a priori. Contrastive learning enables visual features to "understand" asymmetry and border irregularity.
- The **end-to-end interpretable pipeline**—from segmentation → feature quantification → contrastive alignment → report generation—forms a complete closed loop, with each stage having a clear clinical counterpart.
- The coarse-to-fine segmentation strategy (lightweight UNet + SAM2) elegantly balances automation with precision.

## Limitations & Future Work
- The D (dermoscopic structures) component of the ABCD rule is excluded, leaving the framework incomplete.
- The classification accuracy of 92.79%, while strong, is lower than some purely classification-oriented methods (e.g., EfficientNet-B0 at 97%), suggesting that interpretability is achieved at the cost of some performance.
- Validation is limited to the ISIC dataset; real clinical scenarios (low-quality images, rare subtypes) remain uncovered.
- Report generation depends on external LLMs such as DeepSeek, introducing risks of hallucination and uncontrollable outputs; moreover, no systematic quantitative evaluation of clinical accuracy is performed.
- The expert evaluation involved only three clinicians, yielding a small sample size.

## Related Work & Insights
- **vs. CompA (Chanda et al.)**: CompA uses guided attention and Grad-CAM to align model attention with physician-annotated regions, but requires extensive manual annotation and only provides visual explanations. CEFM offers greater scalability through automated ABC feature extraction and LLM-based report generation.
- **vs. Grad-CAM-based methods**: Heatmaps can only indicate "where the model looks," not "why it predicts malignancy." CEFM's ABC quantification and structured reports provide clinically actionable explanations.
- **vs. R2GenGPT**: R2GenGPT directly maps image features to an LLM's text space for report generation but lacks anchoring to clinical criteria. CEFM's contrastive alignment ensures explicit binding between visual features and clinical semantics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The clinical-criteria-driven contrastive learning framework is original, though all individual components (ViT, contrastive learning, CLIP, LLM) are existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ Classification and segmentation performance are reported with quantitative data, but the ablation study relies primarily on qualitative analysis; quantitative interpretability metrics are lacking, and the expert evaluation sample is too small.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly, clinical motivation is well-articulated, and figures are well-designed.
- **Value**: ⭐⭐⭐⭐ The work offers a meaningful paradigm for medical AI interpretability—binding model features to clinical criteria through contrastive learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](../../NeurIPS2025/interpretability/llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](llm_circuit_analyses_consistent_across_training_and_scale.md)

</div>

<!-- RELATED:END -->
