---
title: >-
  [Paper Note] Exploring and Leveraging Class Vectors for Classifier Editing
description: >-
  [NeurIPS 2025][Medical Imaging][Class Vector] This paper proposes Class Vectors, which capture class-level adaptation by calculating the difference between the class centroids in the latent space of pre-trained and fine-tuned models. Leveraging two properties—linearity and independence—classifier editing (forgetting, domain adaptation, adversarial defense) is achieved via simple vector arithmetic. This allows latent space injection without retraining…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Class Vector"
  - "Classifier Editing"
  - "Latent Space Manipulation"
  - "Class Forgetting"
  - "Adversarial Defense"
  - "Neural Collapse"
date: 2026-05-08
content_hash: e50e0c2b8710e0df
---

# Exploring and Leveraging Class Vectors for Classifier Editing

**Conference**: NeurIPS 2025  
**arXiv**: [2510.11268](https://arxiv.org/abs/2510.11268)  
**Code**: Yes (based on CLIP ViT)  
**Area**: Medical Images  
**Keywords**: Class Vector, Classifier Editing, Latent Space Manipulation, Class Forgetting, Adversarial Defense, Neural Collapse

## TL;DR

This paper proposes Class Vectors, which capture class-level adaptation by calculating the difference between the class centroids in the latent space of pre-trained and fine-tuned models. Leveraging two properties—linearity and independence—classifier editing (forgetting, domain adaptation, adversarial defense) is achieved via simple vector arithmetic. This allows latent space injection without retraining, or weight space mapping using <1.5K parameters in less than 1.5 seconds.

## Background & Motivation

**Need for Classifier Editing**: The behavior of deep classifiers becomes fixed after extensive training. However, users often require post-hoc modifications (e.g., forgetting specific classes, adapting to new environments, correcting prediction errors). A one-size-fits-all classifier cannot satisfy these diverse requirements.

**Limitations of Prior Work**: (a) Computationally expensive: retraining ViTs is far more costly than CNNs; (b) Large sample requirements: modifying knowledge with few samples is difficult and easily introduces bias; (c) Coarse granularity: existing methods are restricted to image-by-image error correction, lacking class-level editing capabilities.

**Limitations of Task Vectors**: Task vectors capture task-level modifications in the weight space but cannot decouple the adaptation behavior of individual classes, making them unsuitable for fine-grained classifier editing.

**Core Idea**: Extract class-level representation shift vectors (Class Vectors) in the latent space, leveraging their linearity and orthogonality to achieve precise class-level editing.

## Method

### Class Vector Definition

For class $c$, the Class Vector $\kappa_c \in \mathbb{R}^m$ is defined as the expected difference in the representation of the final layer between the fine-tuned and pre-trained encoders:

$$\kappa_c = \mathbb{E}_{s \in S}[f(s, \theta_{\text{ft}}^e)] - \mathbb{E}_{s \in S}[f(s, \theta_{\text{pre}}^e)]$$

The class centroid of the fine-tuned model can be decomposed as: $z_{\text{ft}}^c = z_{\text{pre}}^c + \kappa_c$

### Theoretical Foundation

**Theorem 3.1 (CTL between Pre-training and Fine-tuning)**: Under the Cross-Task Linearity (CTL) condition, the path from pre-training to fine-tuning is more linear (with smaller CTL deviation) than the path between two fine-tuned models. When $\|\theta_i - \theta_{\text{pre}}\| < \|\theta_i - \theta_j\|$, we have $\delta_{\text{pre},i} < \delta_{i,j}$.

This implies:
$$f(x_c; \theta_{\text{pre}} + \alpha\tau) \approx f(x_c; \theta_{\text{pre}}) + \alpha\kappa_c, \quad x_c \in \mathcal{D}_c$$

That is, scaling $\kappa_c$ in the latent space is equivalent to moving along the task vector $\tau$ in the weight space—class-level adaptation can be achieved via simple vector arithmetic.

### Key Properties of Class Vectors

**Linearity**: Interpolating between two classes with $z_{\text{edit}} = -\alpha\kappa_{c_1} + \alpha\kappa_{c_2}$ results in smooth linear shifts in predictions and logits, switching cleanly from $c_1$ to $c_2$ at the midpoint without detouring through other classes.

**Independence** (based on Neural Collapse):

**Theorem 3.3**: Assume (i) pre-trained class embeddings collapse to a common mean $\bar{z}^{\text{pre}}$; (ii) fine-tuned embeddings form an ETF structure $z_c^{\text{ft}} = \mu + u_c$ with $\sum_c u_c = 0$; (iii) the global shift is negligible. Then:
$$\cos(\kappa_c, z_{c'}^{\text{ft}}) \approx 0, \quad c \neq c'$$

That is, the Class Vector of any class is approximately orthogonal to the fine-tuned embeddings of other classes—modifying one class does not affect the others.

### Editing Methods

**Latent Space Injection (Training-free)**:
1. Compute representation: $r = f(x, \theta_{\text{ft}}^e)$
2. Gating: $\beta = \mathbf{1}[\text{sim}(r) > \gamma]$, activated only when the cosine similarity between $r$ and the target class centroid $z_{\text{ft}}^c$ exceeds the threshold.
3. Inject edit: $\hat{y} = g(r + \beta \cdot z_{\text{edit}}, \theta^h)$

**Weight Space Mapping (Lightweight Training)**:
Learn a mapping $\phi_{\text{edit}}: \mathbb{R}^m \to \mathbb{R}^{d_e}$ such that:
$$\theta_{\text{edit}}^e = \arg\min_{\theta_{\text{edit}}^e} \|f(x, \theta_{\text{edit}}^e) - (f(x, \theta_{\text{ft}}^e) + z_{\text{edit}})\|^2$$

Only the LayerNorm of the final few layers of the encoder is trained, requiring only a single reference sample and taking <1.5 seconds. **Theorem 3.2** proves that for over-parameterized encoders ($d_e \gg m$), there exist infinitely many mappings that satisfy this condition.

## Key Experimental Results

### Class Forgetting (ViT-B/16)

| Method | MNIST $\text{ACC}_f$↓ | MNIST $\text{ACC}_r$↑ | EuroSAT $\text{ACC}_f$↓ | EuroSAT $\text{ACC}_r$↑ | GTSRB $\text{ACC}_f$↓ | GTSRB $\text{ACC}_r$↑ |
|------|-------|-------|---------|---------|-------|-------|
| Retrained | 0.1 | 76.4 | 0.0 | 85.7 | 41.8 | 57.5 |
| NegGrad | 0.0 | 43.4 | 0.0 | 11.6 | 0.0 | 15.6 |
| Random Vec | 99.9 | 99.8 | 99.9 | 80.9 | 99.6 | 98.2 |
| **Class Vec** | **0.0** | **99.7** | **0.0** | **99.5** | **0.0** | **98.6** |
| **Class Vec†** | **0.0** | **96.2** | **0.0** | **99.7** | **0.0** | **93.4** |

- Class Vectors almost perfectly retain performance on non-target classes ($\text{ACC}_r > 93\%$) while forgetting the target class ($\text{ACC}_f \to 0$).
- Comparison with NegGrad: although it can forget, it severely damages performance on non-target classes ($\text{ACC}_r$ plummets).
- Random vectors are completely ineffective, validating that Class Vectors point in meaningful directions.

### Snow Environment Adaptation

| Method | ViT-B/16 | ViT-B/32 | ViT-L/14 |
|------|----------|----------|----------|
| Pretrained | 55.2 | 53.4 | 60.2 |
| Retrained | 55.8 | 55.8 | 75.3 |
| DirMatch | 72.0 | 73.9 | 74.6 |
| **Class Vec** | **69.7** | **71.5** | **74.2** |
| **Class Vec†** | **75.2** | **74.5** | **80.2** |

Let $z_{\text{edit}} = \lambda(\kappa_{\text{snow}+c_1} - \kappa_{c_1})$ with $\lambda < 0$ suppressing snow features. It yields a 10-20% gains using only 4 external samples, and does not require image-by-image training (unlike DirMatch, which requires image-by-image alignment).

### Defense Against Typographic Attacks

Class Vectors defend against typographic attacks by eliminating the classification shift caused by typographic text—using class-level arithmetic to remove representation shifts in the direction of the attack vector.

## Highlights & Insights

- ⭐⭐⭐⭐⭐ **Extremely High Efficiency**: Latent space injection is completely training-free; weight mapping requires only <1.5K parameters, a single sample, and takes less than 1.5 seconds.
- ⭐⭐⭐⭐ **Solid Theoretical Foundation**: Backed by dual theoretical support from CTL and Neural Collapse, with strict proofs for linearity and independence.
- ⭐⭐⭐⭐ **Advanced Interaction**: Non-expert users can perform concept-level edits through intuitive vector arithmetic (addition, subtraction, and scaling).
- ⭐⭐⭐⭐ **Broad Application**: The same framework covers forgetting, domain adaptation, adversarial defense, and backdoor attack optimization.
- ⭐⭐⭐ **Architectural Universality**: Validated effectively across MLP, ResNet-18, ViT-B/16/32, and ViT-L/14.

## Limitations & Future Work

1. **Dependence on Fine-tuned Models**: Requires simultaneous access to both pre-trained and fine-tuned models to calculate the Class Vector, increasing storage requirements.
2. **Limitation on Number of Classes**: Independence is based on the Neural Collapse hypothesis—when there are extremely many classes, the ETF structure may not be perfect.
3. **Insufficient Fine-grained Control**: Class-level editing cannot handle differences within intra-class subpopulations (e.g., different breeds of dogs).
4. **Gating Threshold $\gamma$**: Latent space injection requires setting a cosine similarity threshold, which may vary across different tasks.
5. **Unvalidated on Generative Models**: Only validated on discriminative classifiers; extending this to generative models (such as CLIP-guided diffusion) remains to be explored.

## Rating ⭐⭐⭐⭐

The paper refines the concept of task vectors from the task level to the class level, with elegant theoretical derivations (CTL + NC) and comprehensive experimental coverage. The ultimate highlight is its extreme efficiency—the capability to edit with a single sample in 1.5 seconds is highly valuable for real-world deployment. A limitation is the sparse validation in medical scenarios; the reason it was categorized under the medical imaging track might be its potential application in editing disease detection classifiers.

## Related Work & Insights

## Related Papers

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](../../ICML2026/medical_imaging/factored_classifier-free_guidance.md)
- [\[CVPR 2025\] MoEdit: On Learning Quantity Perception for Multi-Object Image Editing](../../CVPR2025/medical_imaging/moedit_on_learning_quantity_perception_for_multi-object_image_editing.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[ICML 2025\] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](../../ICML2025/medical_imaging/raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)

</div>

<!-- RELATED:END -->
