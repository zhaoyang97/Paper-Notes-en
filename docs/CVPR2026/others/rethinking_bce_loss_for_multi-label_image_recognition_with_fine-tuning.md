---
title: >-
  [Paper Note] Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning
description: >-
  [CVPR 2026][Others][Paper Note] The authors find that fine-tuning CLIP with BCE for multi-label recognition systematically disrupts the semantic geometry of text embeddings, leading to a breakdown in calibration (under-confidence in base classes and over-confidence in new classes). They propose **Class-wise Covariance Regularization (CCR)**—which use
tags:
  - CVPR 2026
  - Others
date: 2026-05-08
content_hash: 0cfc2a1a7608f4b3
---
# Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Rethinking_BCE_Loss_for_Multi-Label_Image_Recognition_with_Fine-Tuning_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Multimodal VLM / Multi-Label Image Recognition  
**Keywords**: CLIP Fine-tuning, Multi-Label Recognition, BCE Loss, Confidence Calibration, Covariance Regularization  

## TL;DR
The authors find that fine-tuning CLIP with BCE for multi-label recognition systematically disrupts the semantic geometry of text embeddings, leading to a breakdown in calibration (under-confidence in base classes and over-confidence in new classes). They propose **Class-wise Covariance Regularization (CCR)**—which uses predicted covariance estimated from "jointly inactive class pairs" within a batch to align with the text semantic correlation matrix. As a lightweight structural regularizer applied over BCE, it fixes calibration while enhancing generalization.

## Background & Motivation
**Background**: Using Vision-Language Models (VLMs) like CLIP for Multi-Label Image Recognition (MIR) has become mainstream, with prompt tuning (e.g., CoOp, TaI-DPT, T2I-PAL) being widely adopted due to low parameter counts and good transferability. Notably, most of these multi-label methods **abandon traditional BCE loss in favor of Ranking loss** as the optimization target—even though BCE and its variants are typically more effective in traditional multi-label deep networks.

**Limitations of Prior Work**: Why does BCE fail in CLIP fine-tuning? Prior work (such as TaI-DPT) blamed the "vision-language modality gap," suggesting that directly optimizing sigmoid probabilities with BCE exacerbates the distribution mismatch between training text and test images. Following the modality gap analysis framework, the authors fed image-text pairs from MS-COCO into zero-shot CLIP and various fine-tuned versions for SVD visualization. They discovered a more fundamental phenomenon: **BCE fine-tuning disrupts the spatial distribution of class text embeddings more severely than Ranking fine-tuning**, causing systematic miscalibration.

**Key Challenge**: The authors correlated this structural drift with confidence behavior. The gradient of BCE loss is $\frac{\partial L}{\partial z_c} = p(c|x) - y_c$. In multi-label scenarios, the overwhelming number of negative samples ($y_c=0$) for each label creates a global "cooling effect": head classes are continuously pushed down by the $p(c|x)-1<0$ gradient, becoming **under-confident**. Tail classes have sparse positive samples but receive strong update signals and align with high-frequency visual components through feature sharing, becoming **over-confident**. New classes receive no supervision ($y_c\equiv0$) during fine-tuning and are driven solely by negative gradients. While Ranking loss also raises logits overall, it **preserves the semantic neighborhood structure** (higher NP@K), which explains its superior performance. Existing temperature scaling or regularization-based calibration methods (DAC, DOR) are mostly ported from single-label scenarios and cannot simultaneously calibrate head and tail classes or balance the base-vs-new trade-off.

**Goal**: To **restore and maintain reliable inter-class semantic relationships** during the BCE fine-tuning process. The goal is to constrain the relative structure of class embeddings from a global, class-level perspective to improve both calibration and generalization.

**Core Idea**: Instead of relying on sparse positive sample co-occurrence to estimate inter-class dependencies, the authors use **dense and stable negative evidence—"the simultaneous absence of two classes ($y=0$)"**—to estimate predicted covariance. This is then aligned with the semantic correlation encoded in CLIP text embeddings.

## Method

### Overall Architecture
CCR is a **pure structural regularization term** that does not modify the CLIP backbone or introduce new branches; it is simply added to the original BCE objective used in prompt tuning. The core problem it solves is that BCE fine-tuning disrupts the geometric structure of class text embeddings, leading to systematic confidence shifts. CCR estimates an **inter-class predicted covariance matrix** $C_{pred}$ within each mini-batch, normalizes it into a correlation matrix, and pulls it toward the **semantic similarity matrix** $\Sigma_{text}$ calculated from zero-shot CLIP text embeddings using Frobenius distance. This ensures the fine-tuning process preserves the original semantic topology.

Input consists of a batch of image-label pairs. The model calculates logits $z_c = \tau\cdot\text{sim}(f_{img}(x), f_{text}(t_c))$ and probabilities $p(c|x)=\sigma(z_c)$ as usual. The output adds a covariance alignment loss to the original BCE classification loss. Since this is an improvement at the loss function level without a multi-stage pipeline, no architecture diagram is needed.

### Key Designs

**1. Estimating Predicted Covariance via "Class-wise Co-inactivation": Replacing Sparse Positive Evidence with Dense Negative Evidence**

Positive samples are extremely sparse in multi-label datasets, making inter-class dependency estimation via co-occurrence ($y=1$) unreliable and biased toward head classes. CCR shifts the perspective: instead of asking "how confident is the model that class $c$ **exists**," it asks "how confident is the model that class $c$ **is absent**." For any class pair, the magnitude of negative evidence is orders of magnitude larger than positive evidence, forming a dense and stable statistical signal (referred to as a shared "semantic background"). Based on this, the authors construct a symmetric covariance matrix $C_{pred}$ within the batch to characterize how the model jointly suppresses or co-activates classes. Although calculated locally at the batch level, the dominance of "inactive predictions" allows this covariance to stably reflect global structural tendencies, also providing robustness to batch size.

**2. Normalization into Correlation Matrix and Alignment with Text Semantic Similarity: Removing Scale and Retaining Relational Structure**

The raw $C_{pred}$ measures the "raw covariance" of confidence, which is not on the same scale as the semantic similarity $\Sigma_{text}(i,j)=\text{sim}\langle t_i, t_j\rangle$ calculated from zero-shot CLIP text embeddings. CCR first normalizes $C_{pred}$ into a correlation matrix to remove magnitude bias and retain only the relational structure between classes:

$$\tilde{C}_{pred}(i, j) = \frac{C_{pred}(i, j)}{\sqrt{C_{pred}(i, i)}\,\sqrt{C_{pred}(j, j)}}$$

Then, the normalized predicted correlation matrix is pulled toward the text semantic correlation matrix using the Frobenius norm:

$$L_{cov} = \big\|\tilde{C}_{pred} - \Sigma_{text}\big\|_F^2$$

This term explicitly constrains the relational structure between any two classes $i,j$ based on their "co-inactivation" behavior, preserving the semantic geometry encoded in CLIP text space.

**3. Application as a Structural Calibration Prior on BCE: Countering Over-Cooling Effects**

The final objective is to attach CCR directly to the base BCE loss:

$$L = L_{BCE} + \lambda \cdot L_{cov}$$

where $\lambda$ controls the strength of the structural regularization. CCR acts as a **structural calibration prior**: while BCE handles discriminative learning, CCR constrains the label-space covariance to offset the "over-cooling" effect (collective suppression of base class confidence) found in standard BCE fine-tuning. Notably, by using second-order statistics (covariance) rather than first-order (mean), CCR is insensitive to the value of $\lambda$, avoiding the over-regularization issues seen in DAC or DOR.

### Loss & Training
All experiments use CLIP-ViT-B/16 with 16-shot few-shot fine-tuning. Training lasts 10 epochs with a batch size of 32 using a unified configuration consistent with TaI-DPT. Few-shot results are averaged over 5 different splits for statistical significance. Evaluation uses accuracy at a 0.5 threshold (reflecting performance under good calibration); metrics like mAP are provided in the appendix.

> ⚠️ The paper also defines two **diagnostic metrics** (used for motivation analysis, not training objectives): Embedding Divergence $ED(t_i)=\frac{1}{k}\sum_{f_{text}(t_j)\in N_k}\text{dist}\langle f_{text}(t_i), f_{text}(t_j)\rangle$ measures the local dispersion of class text embeddings (higher dispersion leads to over-confidence); Neighborhood rank Preservation (NP@K) measures how well semantic neighbor rankings are maintained after fine-tuning. These explain why "higher dispersion → over-confidence, more compact → under-confidence" and why Ranking loss is overall better (higher NP@K).

## Key Experimental Results

### Main Results: Average Calibration Error across Six Datasets (×10⁻², lower is better)
On six multi-label benchmarks (MS-COCO, PASCAL-VOC, NUS-WIDE, COCO-LT, VOC-LT, Open-Images-V6), CCR is compared with two SOTA calibration methods, DAC and DOR (Conf represents baseline performance). CCR achieves the lowest calibration error across most backbones and all four metrics:

| Backbone | Metric | Conf | DAC | DOR | **CCR** |
|----------|------|------|-----|-----|---------|
| CoOp | ECE↓ | 13.25 | 10.85 | 9.92 | **7.35** |
| CoOp | PIECE↓ | 15.12 | 12.36 | 10.95 | **9.42** |
| TaI-DPT | ECE↓ | 6.02 | 5.35 | 5.08 | **4.76** |
| T2I-PAL | ECE↓ | 4.09 | 3.93 | 3.88 | **3.62** |
| T2I-PAL | MCE↓ | 1.37 | 1.18 | 1.26 | **1.04** |

Unlike DAC/DOR which only improve specific models, CCR is consistently effective across seven tuning frameworks, demonstrating its role as a universal and robust structural calibration prior. It reduces both the average error (ECE) and the worst-case error (MCE), mitigating severe over-confidence.

### base-to-new Generalization (Average Accuracy % across six datasets)
CCR improves both base and new classes, with an average Harmonic Mean (HM) gain of approximately **4.8%**; unlike DOR, it does not "improve new classes at the expense of base classes":

| Category | ZS-CLIP | CoOp | CoOp+CCR | TaI-DPT | TaI-DPT+CCR | T2I-PAL | T2I-PAL+CCR |
|------|---------|------|----------|---------|-------------|---------|-------------|
| Head | 80.15 | 81.23 | **82.76** | 81.67 | 81.42 | 84.91 | **86.24** |
| Tail | 63.83 | 64.92 | 64.45 | 72.54 | **76.91** | 78.95 | **81.73** |
| New | 72.46 | 71.15 | **73.82** | 77.13 | **79.84** | 81.86 | **83.97** |

### Calibration by Frequency (ECE %, simultaneous improvement for Head/Medium/Tail/New)
Existing regularizers create trade-offs within base classes (improving tail but worsening head). CCR improves all three segments—Head, Medium, and Tail—**simultaneously**, suggesting it captures the intrinsic link between "class frequency $\leftrightarrow$ calibration difficulty":

| Segment | CoOp Vanilla | +DOR | **+CCR** | T2I-PAL Vanilla | **+CCR** |
|------|--------------|------|----------|------------------|----------|
| Head | 6.92 | 7.45 | **3.67** | 2.04 | **1.81** |
| Tail | 8.37 | 6.92 | **4.89** | 2.78 | **2.43** |
| New | 21.58 | 19.75 | **11.03** | 6.14 | **5.43** |

### Domain Generalization (Trained on MS-COCO few-shot, tested on COCO derivatives)
CCR consistently reduces ECE and slightly increases accuracy on both the source domain and various target domains. For example, CoOp source ECE improved from 5.10% $\rightarrow$ 2.92% (⚠️ the text mentions 2.92 while the source column in the table shows 3.92; refer to original text). TaI-DPT source accuracy improved from 69.05 $\rightarrow$ 71.93:

| Method | Source ECE↓ | Source Acc↑ | COCO-2014 Acc↑ |
|------|-----------|-----------|----------------|
| CoOp | 5.10 | 69.44 | 63.55 |
| CoOp+CCR | **3.92** | **71.47** | **72.47** |
| TaI-DPT | 4.13 | 69.05 | 69.57 |
| TaI-DPT+CCR | **2.86** | **71.93** | **74.94** |

### Key Findings
- **Negative evidence is key**: CCR is stable because it estimates covariance using the dense signal of "jointly inactive class pairs" rather than sparse positive co-occurrences.
- **Second-order > First-order**: Constraining covariance (2nd order) rather than mean (1st order) makes CCR stable across a wide range of $\lambda$, avoiding over-regularization.
- **ED/NP Diagnosis**: Higher embedding dispersion leads to over-confidence, while tighter embeddings lead to under-confidence. Ranking loss expands the embedding space but preserves neighborhood ranking (high NP@K), which is why it outperforms BCE—CCR allows BCE to preserve this topology as well.

## Highlights & Insights
- **Clever perspective reversal**: Switching from "estimating co-occurrence with sparse positives" to "estimating covariance with massive negatives (co-inactivation)" solves the long-standing problem of sparse positive samples and head-class bias in multi-label recognition. This "dense signal from inverse questions" approach is transferable to any long-tail or sparsely supervised structure estimation.
- **Closed-loop diagnosis and method**: The paper thoroughly explains "BCE disrupts semantic geometry $\rightarrow$ miscalibration" using ED/NP metrics, then uses covariance alignment to precisely fix the geometry. The motivation and method are perfectly aligned.
- **Plug-and-play structural prior**: CCR is orthogonal to supervised losses and works across seven different tuning frameworks with near-zero additional cost.
- **Modality-agnostic**: The authors note that CCR regularizes the inter-class correlation structure independently of input modalities. It could theoretically extend to visual fine-tuning or cross-modal alignment.

## Limitations & Future Work
- **Dependence on reliable text semantic priors**: CCR uses semantic correlation in CLIP text space as an anchor; its applicability is limited when text embeddings themselves are weak.
- **Modeling only pairwise linear correlation**: Covariance only captures pairwise linear relationships, leaving richer high-order or non-linear dependencies unaddressed.
- **Calibration of Ranking loss remains open**: The authors admit Ranking loss lacks explicit probability outputs, making standard ECE calculation difficult. They construct "pseudo-confidences," acknowledging this is unrigorous.
- **Empirical nature of diagnostic metrics**: Conclusions about head/tail/new classes' under/over-confidence are based on MS-COCO + ViT-B/16 empirical analysis. Robustness across other datasets/architectures is not fully explored in the main text.

## Related Work & Insights
- **vs. Ranking loss (TaI-DPT, etc.)**: These methods bypass BCE's calibration issues by using relative ranking targets. CCR addresses BCE's structural drift to make the "more effective BCE objective" usable again.
- **vs. DAC (Distance-Aware Calibration)**: DAC uses text-related logit biases for instance-level adjustments (1st order, sample-specific). CCR is a 2nd order, structural-level global prior.
- **vs. DOR (Dynamic Outlier Regularization)**: DOR suppresses over-confidence via dynamic outlier regularization, often improving new classes at the expense of base classes. CCR preserves semantic geometry, benefiting both.
- **vs. Temperature Scaling/Label Smoothing**: These are sample-level methods. CCR is structural, preserving the global semantic topology across classes to guide the model toward a semantically coherent confidence manifold.

## Rating
- Novelty: ⭐⭐⭐⭐ The "co-inactivation for covariance alignment" perspective is solid and unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ High coverage across 6 datasets, 7 backbones, 4 calibration metrics, plus generalization tests.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from diagnosis to mechanism to method, though minor discrepancies exist between table values and text.
- Value: ⭐⭐⭐⭐ A plug-and-play, zero-cost structural calibration prior with direct utility for multi-label CLIP fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](prototype-based_causal_intervention_for_multi-label_image_classification.md)
- [\[CVPR 2026\] Cross-View Distillation and Adaptive Masking for Incomplete Multi-View Multi-Label Classification](cross-view_distillation_and_adaptive_masking_for_incomplete_multi-view_multi-lab.md)
- [\[CVPR 2026\] From Pixel to Precision: Enhancing Handwritten Mathematical Expression Recognition with Image-Level Reward](from_pixel_to_precision_enhancing_handwritten_mathematical_expression_recognitio.md)
- [\[CVPR 2026\] Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach](revisiting_f-measure_optimization_in_multi-label_classification_a_sampling-based.md)
- [\[CVPR 2026\] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective](rethinking_knowledge_transfer_in_image_quality_assessment_a_perceptual_preferenc.md)

</div>

<!-- RELATED:END -->
