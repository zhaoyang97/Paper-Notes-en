---
title: >-
  [Paper Note] Interpretable Generative Models through Post-hoc Concept Bottlenecks
description: >-
  [CVPR 2025][Image Generation][Concept Bottleneck Models] This paper proposes two low-cost post-hoc methods—Concept Bottleneck Autoencoders (CB-AE) and Concept Controllers (CC)—to convert pre-trained generative models into interpretable and controllable models without retraining from scratch or requiring ground-truth annotations. They outperform prior CBGM methods in steerability by an average of approximately 25% on CelebA/CelebA-HQ/CUB datasets…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Concept Bottleneck Models"
  - "Interpretable Generative Models"
  - "Post-hoc Training"
  - "Generative Control"
  - "Concept Intervention"
date: 2026-05-08
content_hash: 470c00facb14b43c
---

# Interpretable Generative Models through Post-hoc Concept Bottlenecks

**Conference**: CVPR 2025  
**arXiv**: [2503.19377](https://arxiv.org/abs/2503.19377)  
**Code**: [https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm)  
**Area**: Diffusion Models  
**Keywords**: Concept Bottleneck Models, Interpretable Generative Models, Post-hoc Training, Generative Control, Concept Intervention

## TL;DR
This paper proposes two low-cost post-hoc methods—Concept Bottleneck Autoencoders (CB-AE) and Concept Controllers (CC)—to convert pre-trained generative models into interpretable and controllable models without retraining from scratch or requiring ground-truth annotations. They outperform prior CBGM methods in steerability by an average of approximately 25% on CelebA/CelebA-HQ/CUB datasets, while being 4–15x faster to train.

## Background & Motivation

1. **Background**: Deep generative models (GANs, Diffusion Models) have achieved immense success in image generation, but their generation process remains a black box. Concept Bottleneck Models (CBMs) achieve inherent interpretability by embedding human-understandable concepts into intermediate layers, but they are currently mostly limited to classification tasks.

2. **Limitations of Prior Work**:
    - The only prior work extending CBMs to generative tasks, CBGM, requires training the entire generative model from scratch, incurring massive computational costs (e.g., 240 V100-hours for DDPM-256).
    - CBGM requires ground-truth concept annotations for images, which are expensive to obtain.
    - The steerability of CBGM is only around 25%, limiting its practical utility.

3. **Key Challenge**: To achieve interpretability, CBGM requires complete retraining, which runs counter to the recent trend of leveraging powerful pre-trained models. Meanwhile, CBGM's reliance on concept annotations restricts its scalability.

4. **Goal**: How to inject concept-level interpretability and steerability into any pre-trained generative model in a low-cost, post-hoc manner.

5. **Key Insight**: The intermediate latent space of generative models already encodes rich semantic info. Thus, one only needs to train a lightweight concept bottleneck layer to "decode" and "control" this information.

6. **Core Idea**: Freeze the pre-trained generator and train only a concept bottleneck autoencoder to be inserted into the latent space, utilizing pseudo-labels instead of ground-truth annotations to realize low-cost post-hoc interpretable generation.

## Method

Mechanism: The core idea is to split the pre-trained generative model $g = g_2 \circ g_1$ into two parts and insert a downloadable Concept Bottleneck Autoencoder $f = D \circ E$ in between, such that the new model $g_2 \circ f \circ g_1$ preserves generation quality while offering concept-level interpretability and intervention capabilities.

### Overall Architecture

Input: Random noise $z$. Passing through $g_1$ yields the intermediate latent $w$. The CB-AE encoder $E$ maps $w$ to a concept vector $c = E(w)$, and the decoder $D$ reconstructs the latent as $w' = D(c)$. Finally, $g_2(w')$ generates the image. The concept vector $c$ contains both logits of pre-defined concepts and unsupervised concept embeddings.

Two methods:
- **CB-AE**: A complete Concept Bottleneck Autoencoder that provides inherent interpretability.
- **CC**: Train only the encoder (Concept Controller), used in conjunction with optimization-based intervention to achieve steering.

### Key Designs

1. **Concept Bottleneck Autoencoder (CB-AE)**:
    - **Function**: Provides concept prediction and intervention capabilities while preserving generation quality.
    - **Mechanism**: CB-AE consists of an encoder $E$ (latent to concept vector) and a decoder $D$ (concept vector to reconstructed latent). Each binary concept in the concept vector $c$ is represented by two logits (e.g., "smiling" is represented as $[c_i^+, c_i^-]$), alongside a 40-dimensional unsupervised embedding to capture concepts that are not pre-defined. There are three training objectives: (1) reconstruction loss $\mathcal{L}_{r_1}(w, w') + \mathcal{L}_{r_2}(x, x')$ (MSE) to ensure generation quality; (2) concept alignment loss $\mathcal{L}_c(\hat{y}, c)$ (cross-entropy) to align concept predictions with pseudo-labels $\hat{y} = M(x)$; (3) intervention loss $\mathcal{L}_{i_1}, \mathcal{L}_{i_2}$ to ensure that the generated images actually change when concepts are modified.
    - **Design Motivation**: Post-hoc training avoids the prohibitive costs of training from scratch. Using pseudo-labels (obtained from zero-shot CLIP classification or few-shot labeling models) eliminates the need for ground-truth annotations.

2. **Optimization-based Interventions**:
    - **Function**: Performs precise manipulation of specific concepts in the latent space via gradient optimization.
    - **Mechanism**: Inspired by adversarial attacks, I-RFGSM (Iterative Randomized Fast Gradient Sign Method) is used to optimize over the predictions of the CB-AE encoder. Given the original latent $w$ and the target concept $c^*$, the optimization solves $w^* = w + \arg\max_{\delta \in \Delta}[-\mathcal{L}_c(E(w+\delta), c^*)]$, subject to $\|\delta\|_\infty \leq \epsilon$. Intuitively, it searches for a small perturbation $\delta$ such that the concept prediction of $w + \delta$ shifts to the target concept, while minimizing visual changes in the image.
    - **Design Motivation**: While simple logit-swapping interventions are intuitive, their steerability is limited. Optimization-based interventions offer higher intervention success rates and superior image quality.

3. **Concept Controller (CC)**:
    - **Function**: A lighter alternative to CB-AE, designed solely for concept prediction and optimization-based intervention.
    - **Mechanism**: Observing that optimization-based intervention does not utilize the decoder $D$, the decoder can be omitted, leaving only the training of a concept predictor $\Omega$. The training objective simplifies solely to the concept alignment loss $\min_\Omega[\mathcal{L}_c(\hat{y}, c)]$. Used alongside optimization-based interventions, CC generally performs better in steerability, even though it is not inherently interpretable (as it does not alter the generative pathway).
    - **Design Motivation**: If users only require control without the need for inherent interpretability, the training time of CC can be significantly shorter than that of CB-AE (8-30x faster than CBGM).

### Loss & Training

- **Total loss for CB-AE**: $\mathcal{L}_{r_1} + \mathcal{L}_{r_2} + \mathcal{L}_c + \mathcal{L}_{i_1} + \mathcal{L}_{i_2}$
- **Intervention training**: Randomly select a concept, swap its logits to generate $c_{intervened}$, and reconstruct the image to verify if the intervention is successful.
- **CC training**: $\mathcal{L}_c$ only.
- 50 epochs, batch size 64, 4-layer MLP/Conv, optimization-based intervention uses 50-step I-RFGSM ($\epsilon=0.1$).

## Key Experimental Results

### Main Results

**Steerability Comparison (8 concepts, CelebA GAN)**

| Method | Steerability (%)↑ | FID↓ | Training Time |
|------|-------------------|------|---------|
| CBGM | 25.60 | 9.10 | 50 V100-hrs |
| CB-AE | 47.34 | 9.52 | 14 V100-hrs (3.5×) |
| CB-AE+opt-int | 61.14 | — | — |
| CC+opt-int | 51.14 | 7.65 | 6 V100-hrs (8.3×) |

**Steerability (%) Across Models and Datasets**

| Method | CelebA GAN | CelebA-HQ DDPM | CelebA-HQ StyGAN2 | CUB GAN |
|------|-----------|----------------|-------------------|---------|
| CBGM | 25.60 | 13.80 | — | 21.30 |
| CB-AE+opt-int | 61.14 | 38.09 | 61.66 | 46.03 |
| CC+opt-int | 51.14 | 41.45 | 67.95 | 48.91 |

### Ablation Study

| Configuration | Concept Accuracy | Steerability | Description/Notes |
|------|----------|-------------|------|
| CB-AE (logit swap) | 86.56% | 47.34% | Basic intervention |
| CB-AE + opt-int | 86.56% | 61.14% | Optimization-based intervention +13.8% |
| CC + opt-int | 87.65% | 51.14% | Lighter but slightly worse |
| CLIP zero-shot pseudo-labels | Slightly lower | Still significantly outperforms CBGM | Zero concept supervision is feasible |
| TIP few-shot pseudo-labels (128 images) | Close to supervised | Close to supervised | Approximable with only 128 images |

### Key Findings
- **Optimization-based intervention is key**: Elevates the steerability of CB-AE from 47.34% to 61.14%, a relative improvement of 29%.
- **CC is optimal on StyleGAN2**: Reaches up to 67.95% on CelebA-HQ, as the clean latent space of GANs is more amenable to optimization.
- **Pseudo-labels are sufficient**: Utilizing zero-shot CLIP classifiers yields effective pseudo-labels, eliminating the need for any ground-truth annotation data.
- **Large-scale scenario (40 concepts)**: On all 40 attributes of CelebA, CB-AE+opt-int reaches 58.3% steerability, while CBGM only scales to 23.1%.
- User studies validate the reliability of automated evaluations; human consensus on CB-AE's concept accuracy aligns closely with automated metrics.

## Highlights & Insights
- **Post-hoc training paradigm**: The framework of freezing the pre-trained generator and training a lightweight bottleneck layer is extremely practical. It can be plugged directly into almost any generative model (GAN/DDPM/StyleGAN2). This design blueprint is highly transferable to interpretability research in other media, such as video or 3D generation.
- **Adversarial attack to concept intervention**: Creatively applying an adversarial attack metric (I-RFGSM) for concept steerability represents a highly clever and effective cross-domain analogy.
- **Minimalist design of CC**: Removing the decoder reveals that intervention can be thoroughly executed in the latent space via optimization, rather than learning an explicit concept-to-latent mapping, drastically curbing training costs.

## Limitations & Future Work
- Concept coupling exists (e.g., "young" and "bald"); modifying one concept can alter others, as orthogonality control remains limited in the current implementation.
- Relying heavily on pseudo-label quality; CLIP's zero-shot capability for specific concepts (e.g., fine-grained CUB bird attributes) remains constrained.
- Only verified in image generation scenarios; extension to video or 3D generation has not yet been explored.
- Optimization-based intervention requires multi-step gradient calculations (e.g., 50 steps), which might limit its real-time application.
- There could be an optimization conflict between the reconstruction loss and intervention loss within CB-AE.

## Related Work & Insights
- **vs CBGM**: CBGM trains the entire generative model from scratch and necessitates ground-truth concept annotations, whereas CB-AE uses post-hoc training with pseudo-labels. CB-AE outperforms CBGM in steerability by ~25% on average while being 4-15x faster to train.
- **vs LF-CBM/VLG-CBM**: These are post-hoc CBMs tailored for classification tasks. CB-AE is the pioneering work that extends the post-hoc CBM paradigm to generative models.
- **vs GAN latent manipulation**: Techniques such as InterfaceGAN directly manipulate the GAN latent space but lack concept-level interpretability and structured constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of post-hoc concept bottlenecks and optimization-based interventions is highly novel, and the method design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Very comprehensive, covering multiple generative models (GAN/DDPM/StyleGAN2), multiple datasets, and human user studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described clearly, and the comparison tables are highly informative.
- Value: ⭐⭐⭐⭐ It provides a practical, low-cost solution for the interpretability of generative models, with strong potential for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[CVPR 2026\] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](../../CVPR2026/image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)

</div>

<!-- RELATED:END -->
