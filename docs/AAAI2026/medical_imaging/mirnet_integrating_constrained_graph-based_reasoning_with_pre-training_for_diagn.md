---
title: >-
  [Paper Note] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging
description: >-
  [AAAI 2026][Medical Imaging][tongue diagnosis] MIRNet is a framework that integrates self-supervised masked autoencoder (MAE) pre-training with constraint-aware graph attention network (GAT) reasoning for multi-label ton…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "tongue diagnosis"
  - "graph attention network"
  - "self-supervised pre-training"
  - "clinically constrained optimization"
  - "multi-label classification"
date: 2026-05-08
content_hash: 4d4314b11098cd8b
---

# MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging

**Conference**: AAAI 2026
**arXiv**: [2511.10013](https://arxiv.org/abs/2511.10013)  
**Code**: [GitHub](https://github.com/zijie8247/MIRNet)  
**Area**: Medical Image Analysis / Tongue Diagnosis
**Keywords**: tongue diagnosis, graph attention network, self-supervised pre-training, clinically constrained optimization, multi-label classification

## TL;DR

MIRNet is a framework that integrates self-supervised masked autoencoder (MAE) pre-training with constraint-aware graph attention network (GAT) reasoning for multi-label tongue diagnosis. The paper also introduces TongueAtlas-4K, a benchmark dataset of 4,000 images with 22 labels, achieving a 77.8% improvement in Macro Recall and 33.2% in Macro-F1.

## Background & Motivation

Medical image diagnosis requires combining fine-grained visual pattern recognition with domain knowledge-driven reasoning, particularly for understanding complex relationships among statistically correlated diagnostic labels and clinical priors. Tongue analysis is a critical component of Traditional Chinese Medicine (TCM) diagnosis — for instance, a "pale tongue" frequently co-occurs with a "white coating" — yet such domain knowledge remains underutilized in existing methods.

Existing tongue diagnosis approaches suffer from four interrelated problems:

**Problem 1: Label Scarcity.** Annotating professional medical images is costly and time-consuming, severely limiting supervised learning. Prior work such as jiang2022deep used a dataset of 8,676 annotated images that was never publicly released and covered only 7 categories.

**Problem 2: Severe Label Imbalance.** The prevalence of different symptoms in tongue diagnosis varies enormously — e.g., "white coating" appears in 78.38% of cases while "dark-red tongue" appears in only 2.15% — resulting in extremely poor detection of rare conditions.

**Problem 3: Insufficient Label Correlation Modeling.** Diagnostic labels exhibit significant statistical co-occurrence patterns (e.g., "thin tongue" and "enlarged tongue" are mutually exclusive), yet existing approaches (e.g., independent ResNet with late fusion, bidirectional RNN modeling) lack systematic modeling of label dependencies.

**Problem 4: Absence of Clinical Plausibility Constraints.** Predictions may violate medical common sense (e.g., simultaneously predicting mutually exclusive diagnoses), and existing models have no mechanism to prevent such unreasonable outputs.

**Core Idea**: Design an end-to-end framework that systematically addresses all four challenges: MAE pre-training for label scarcity, GAT for label dependency modeling, constraint-aware optimization for clinical plausibility, and asymmetric loss for label imbalance.

## Method

### Overall Architecture

MIRNet adopts an encoder-decoder architecture: a MAE pre-trained ViT encoder extracts image features; a graph constructed from label co-occurrence is processed by a GAT decoder to model inter-label dependencies; and a constraint-aware multi-objective loss function is used for joint training. The overall pipeline is: Image → MAE Encoder → Visual Features → Label Graph + GAT → Graph-Refined Features → Constraint-Aware Optimization → Multi-label Prediction.

### Key Designs

1. **Masked Autoencoder (MAE) Pre-training**

    - **Function**: Learn transferable visual representations on large-scale unannotated tongue images to address label scarcity.
    - **Mechanism**: The input image is divided into $N$ non-overlapping patches; 75% are randomly masked. The ViT encoder processes visible patches to produce features, and a lightweight decoder reconstructs the masked regions. The loss is the pixel-level MSE over masked patches: $\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$
    - **Design Motivation**: Pre-training on 15,905 unannotated images enables the encoder to learn anatomical features of the tongue, providing a strong initialization for downstream tasks. The backbone is ViT-Base-Patch16-224 with embed_dim=768, depth=12, and num_heads=12.
    - The visual features produced by the pre-trained encoder are used to initialize label node embeddings in the GAT.

2. **Label Correlation Modeling via Graph Attention Network**

    - **Function**: Propagate information over a graph of diagnostic labels to capture high-order inter-label correlations.
    - **Mechanism**: A label co-occurrence graph is first constructed from the training annotation matrix. The co-occurrence matrix $\mathbf{M} = \mathbf{Y}^\top\mathbf{Y} - \text{diag}(\mathbf{Y}^\top\mathbf{Y})$ is thresholded dynamically (25th percentile of non-zero co-occurrence values) to form a sparse adjacency matrix. Two-layer GATv2Conv then performs label propagation, aggregating neighbor information via attention coefficients $\alpha_{ij}$ at each layer. The final prediction concatenates the initial MAE features $\mathbf{v}_k^{(0)}$ with the graph-refined features $\mathbf{v}_k^{(L)}$ and passes them through a classification head: $\hat{y}_k = \sigma(\mathbf{w}_k^\top [\mathbf{v}_k^{(0)} \| \mathbf{v}_k^{(L)}] + b_k)$
    - **Design Motivation**: This design preserves local visual evidence while injecting context-aware label correlations. Two further enhancements are introduced: **rare label augmentation** (rescaling attention weights by the log of inverse frequency) and **correlation confidence weighting** (weighting attention edges by normalized co-occurrence frequency).

3. **Constraint-Aware Optimization**

    - **Function**: Incorporate clinical knowledge into training as differentiable constraints to ensure clinically plausible predictions.
    - **Mechanism**: The unified optimization objective is $\min_{\theta,\phi} \mathcal{L}_{\text{ASL}} + \lambda_1 \mathcal{L}_{\text{constraint}} + \lambda_2 \mathcal{L}_{\text{prior}}$, comprising three components:
        - **Asymmetric Loss (ASL)**: Addresses positive-negative imbalance via frequency-based weighting $\gamma_k = \sqrt{\tau/\mathbb{P}(y_k=1)}$ and asymmetric focusing parameters $\zeta_+ < \zeta_-$.
        - **Clinical Knowledge Constraints**: Mutual exclusion ($p_a \cdot p_b$), co-occurrence ($|p_a - p_b|$), and implication ($p_a \cdot (1-p_b)$) constraints are encoded as differentiable penalty terms via Lagrangian relaxation, incurring loss only when violated.
        - **Statistical Prior Regularization**: KL divergence $\text{KL}(q(\mathbf{y}|\mathbf{X}) \| p_{\text{data}}(\mathbf{y}))$ aligns the predicted marginal distribution with empirical class priors.
    - **Design Motivation**: With $\lambda_1=0.1$ and $\lambda_2=0.05$, constraints are satisfied naturally through gradient optimization without hard-coded rules.

4. **Boosting Ensemble Strategy**

    - **Function**: Further improve classification performance on low-frequency labels.
    - **Mechanism**: A base model is trained on the full dataset; a second model is fine-tuned on augmented data exclusively for classes with F1 < 0.5. Final predictions use the second model's output for the 5 worst-performing labels and the base model's output for the remainder.

### Loss & Training

- AdamW optimizer, base learning rate $1 \times 10^{-3}$, batch size 200, layer-wise learning rate decay with $\text{layer\_decay}=0.75$, trained for 200 epochs.
- Training performed on an NVIDIA A800 GPU.
- Dataset split 80/10/10 into train/validation/test sets.
- Image preprocessing includes rigorous color correction, DeepLabV3+ segmentation, and manual refinement.

## Key Experimental Results

### Main Results

| Model | Example-F1 | Micro-F1 | Macro-F1 | Macro Recall | Macro PR-AUC |
|-------|-----------|----------|----------|-------------|-------------|
| LGAN | 0.634 | 0.640 | 0.397 | 0.369 | 0.492 |
| Faster R-CNN | 0.651 | 0.662 | 0.381 | 0.339 | 0.493 |
| DenseNet121 | 0.648 | 0.657 | 0.403 | 0.364 | 0.351 |
| C-GMVAE | 0.634 | 0.647 | 0.346 | 0.305 | 0.526 |
| **MIRNet** | **0.680** | **0.683** | **0.525** | **0.599** | 0.527 |
| **MIRNet-Boosting** | 0.675 | 0.678 | **0.537** | **0.655** | **0.543** |

### Ablation Study

| Configuration | Example-F1 Change | Macro-F1 Change | Macro Recall Change | Notes |
|--------------|------------------|----------------|--------------------|----|
| MIRNet-C (w/o constraints) | −3.2% | −4.4% | — | Label consistency degraded |
| MIRNet-G (GAT → MLP) | — | −3.2% | −8.1% | Label dependency modeling is indispensable |
| MIRNet-P (w/o pre-training) | — | **−23.0%** | **−29.0%** | Most severe impact; pre-training is critical |

### Key Findings

- MIRNet-Boosting outperforms the strongest baseline by 33.2% in Macro-F1 and 77.8% in Macro Recall, a remarkable margin.
- Even without Boosting, MIRNet surpasses all baselines (Macro Recall +62.5%, Macro-F1 +30.4%).
- Improvements are especially pronounced on rare labels: dark-red tongue (2.15% prevalence) improves from F1 < 0.25 across all baselines to 0.68; gray-black coating improves from near zero to 0.71.
- Dimension-level missed detections are substantially reduced: tongue shape missed detections drop from up to 209 cases in baselines to 14 in MIRNet-Boosting (a 93.3% reduction).
- Ablation analysis confirms that the three components are complementary: pre-training has the greatest impact on rare classes (Macro Recall −29%), constraints provide the largest overall gain (preventing inconsistent labels), and GAT maintains the precision-recall balance.
- MIRNet performs consistently across all four diagnostic dimensions: tongue color 0.81, tongue shape 0.77, coating texture 0.76, coating color 0.84, compared to baseline averages of 0.59, 0.43, 0.51, and 0.68, respectively.

## Highlights & Insights

- Introducing the "learning-with-reasoning integration" paradigm — analogous to physical constraints in DRNets/PINNs — into medical image diagnosis represents a valuable cross-domain transfer of ideas.
- The constraint-aware optimization design is elegant: softening hard constraints into differentiable loss terms via $\max(0, \phi_j)$ preserves the semantic meaning of the constraints without obstructing gradient flow.
- The two auxiliary techniques in the GAT — rare label augmentation and correlation confidence weighting — are simple yet effective and are worth adopting in other multi-label classification settings.
- The release of TongueAtlas-4K (4,000 images, 22 labels, consensus annotations from 10 experts) fills a critical gap in the absence of public benchmarks for tongue analysis.

## Limitations & Future Work

- Although the paper claims the framework generalizes to other medical imaging tasks, all experiments are conducted exclusively on tongue data; validation on other modalities such as X-ray and CT is absent.
- Constraint rules (mutual exclusion, co-occurrence, implication) must be defined manually, requiring domain experts to redesign them for new clinical settings.
- The granularity of 22 labels may still be insufficient for practical TCM clinical use.
- The threshold for the Boosting strategy (F1 < 0.5) and the number of replaced labels (5) appear to be empirical choices whose robustness has not been thoroughly validated.
- The label graph construction relies on training set statistics and may be unreliable when the training set is small or the label distribution is skewed.

## Related Work & Insights

- MIRNet shares the same philosophical lineage as DRNets, which embed thermodynamic priors into deep learning for materials discovery; MIRNet analogously embeds TCM clinical priors for tongue diagnosis.
- Compared to LGAN (CNN + bidirectional RNN for label correlation) and IFRCNet (dilated convolutions + attention), MIRNet's graph-based reasoning is more explicit and interpretable.
- The ASL loss function's strategy for handling imbalance in multi-label classification is broadly applicable to other domains.
- The decisive impact of pre-training on performance (−23% Macro-F1 in ablation) further corroborates the importance of self-supervised pre-training in annotation-scarce settings.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](../../ACL2026/medical_imaging/dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)

</div>

<!-- RELATED:END -->
