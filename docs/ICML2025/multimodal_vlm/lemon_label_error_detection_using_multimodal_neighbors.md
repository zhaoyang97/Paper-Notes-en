---
title: >-
  [Paper Note] LEMoN: Label Error Detection using Multimodal Neighbors
description: >-
  [ICML 2025][Multimodal VLM][Label Error Detection] This paper proposes LEMoN, a method that leverages the multimodal neighborhood structure of image-text pairs in the embedding space of contrastively pre-trained multimodal models (such as CLIP). It automatically detects label errors in both classification and image captioning scenarios, achieving a 3-4% F1-score improvement over training-free baselines. Furthermore, downstream classification and captioning performance are enh…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Label Error Detection"
  - "Multimodal Noisy Labels"
  - "Contrastive Learning Embeddings"
  - "Nearest Neighbor Methods"
  - "Image Captioning"
date: 2026-05-08
content_hash: d5b902089ce125c5
---

# LEMoN: Label Error Detection using Multimodal Neighbors

**Conference**: ICML 2025  
**arXiv**: [2407.18941](https://arxiv.org/abs/2407.18941)  
**Code**: Yes (unreleased link)  
**Area**: Multimodal VLMs  
**Keywords**: Label Error Detection, Multimodal Noisy Labels, Contrastive Learning Embeddings, Nearest Neighbor Methods, Image Captioning

## TL;DR
This paper proposes LEMoN, a method that leverages the multimodal neighborhood structure of image-text pairs in the embedding space of contrastively pre-trained multimodal models (such as CLIP). It automatically detects label errors in both classification and image captioning scenarios, achieving a 3-4% F1-score improvement over training-free baselines. Furthermore, downstream classification and captioning performance are enhanced when trained on the filtered data.

## Background & Motivation

**Background**: The training of modern vision-language models relies on massive image-text pair datasets (such as LAION-400M and CC-12M). Most of these data are crawled from the web and inevitably contain a large number of **label errors**—where images and descriptions do not match. Label errors degrade the reliability of downstream models, which is especially critical in fields like medicine.

**Limitations of Prior Work**:
   - Most label error detection methods are **unimodal**: they only utilize image representations for detection, ignoring textual information.
   - Some high-performance methods (such as AUM, Datamap) require **training classifiers for several epochs** on downstream tasks, which is computationally expensive.
   - Existing methods assume labels are from a "one-of-k" discrete category set, making them **unable to handle natural language labels** (such as image captions).
   - Although the simplest CLIP similarity-based method is training-free, it ignores the rich information in the neighborhood structure.

**Key Challenge**: The larger the dataset, the harder it is to perform manual verification → automatic detection is needed → but existing automatic methods either require expensive training or are limited to unimodal and discrete labels.

**Goal**: Propose a **training-free method that leverages multimodal neighborhood information** for label error detection, which is applicable to both classification labels and natural language descriptions.

**Key Insight**: In the CLIP embedding space, correctly labeled image-text pairs should have similar neighbors (the text corresponding to a neighbor's image should also be similar to the current text), whereas mismatched label relations will expose inconsistency within the neighborhood.

**Core Idea**: Construct a label error detection score by combining the image-text multimodal distance with cross-modal neighborhood information from two directions.

## Method

### Overall Architecture
Given a dataset $\mathcal{D} = \{(\mathbf{x}, \mathbf{y})_i\}_{i=1}^N$ (image-text pairs), LEMoN outputs a "label error score" $s$ for each sample. The core process is as follows:
1. Encode all images and texts using a pre-trained CLIP model.
2. For each sample, compute three score components and linearly combine them.
3. The higher the score, the more likely the sample contains a label error.

### Key Designs

1. **Multimodal Distance $d_{mm}$ (Base Score)**:

    - Directly compute the cosine distance between the image embedding and text embedding:
    $$d_{mm}(\mathbf{x}, \mathbf{y}) = d_{cos}(h_\theta^\mathcal{X}(\mathbf{x}), h_\theta^\mathcal{Y}(\mathbf{y}))$$
    - This is the CLIP Similarity baseline—the larger the distance, the more likely it is a label error.
    - **Design Motivation**: This is the most fundamental and direct signal, which has been validated by prior work. LEMoN builds upon this by adding neighborhood information.

2. **Image Space Neighborhood Score $s_n$**:

    - Find the $k$ nearest neighbors $\{\mathbf{x}_{n1}, \ldots, \mathbf{x}_{nk}\}$ of $\mathbf{x}$ in the image embedding space.
    - Compute the weighted average of the distances between the current text $\mathbf{y}$ and the corresponding texts $\mathbf{y}_{nj}$ of these neighbors:
    $$s_n(\mathbf{x}, \mathbf{y}, \mathcal{D}) = \frac{1}{k} \sum_{j=1}^k d_\mathcal{Y}(\mathbf{y}, \mathbf{y}_{nj}) \cdot e^{-\tau_{1,n} d_\mathcal{X}(\mathbf{x}, \mathbf{x}_{nj})} \cdot e^{-\tau_{2,n} d_{mm}(\mathbf{x}_{nj}, \mathbf{y}_{nj})}$$
    - **Intuition**: If my image is highly similar to my neighbors' images, but my text is very different from their corresponding texts, it indicates that my label is highly likely incorrect.
    - **Weight Design**:
        - $e^{-\tau_{1,n} d_\mathcal{X}}$: Downweights neighbors that are far away (adaptive $k$).
        - $e^{-\tau_{2,n} d_{mm}}$: Downweights neighbors that themselves might be label errors.

3. **Text Space Neighborhood Score $s_m$**:

    - Find the $k$ nearest neighbors $\{\mathbf{y}_{m1}, \ldots, \mathbf{y}_{mk}\}$ of $\mathbf{y}$ in the text embedding space.
    - Compute the distance between the current image $\mathbf{x}$ and the corresponding images $\mathbf{x}_{mj}$ of these neighbors:
    $$s_m(\mathbf{x}, \mathbf{y}, \mathcal{D}) = \frac{1}{k} \sum_{j=1}^k d_\mathcal{X}(\mathbf{x}, \mathbf{x}_{mj}) \cdot e^{-\tau_{1,m} d_\mathcal{Y}(\mathbf{y}, \mathbf{y}_{mj})} \cdot e^{-\tau_{2,m} d_{mm}(\mathbf{x}_{mj}, \mathbf{y}_{mj})}$$
    - **Intuition**: If the images corresponding to other texts that are similar to my text have a large discrepancy with my image, it also indicates a label error.
    - **Design Motivation**: Complementary to $s_n$—$s_n$ starts from the image neighborhood, whereas $s_m$ starts from the text neighborhood. Signals from both directions jointly enhance the detection.

4. **Final Score**:
    $$s = f(\mathbf{x}, \mathbf{y}) = d_{mm}(\mathbf{x}, \mathbf{y}) + \beta \cdot s_n(\mathbf{x}, \mathbf{y}, \mathcal{D}) + \gamma \cdot s_m(\mathbf{x}, \mathbf{y}, \mathcal{D})$$
    - $\beta, \gamma \geq 0$ are hyperparameters.
    - **Generalizability**: When $\beta = \gamma = 0$, it degrades to CLIP Similarity; when $\beta$ is large, $\gamma = 0$, and discrete distances are used, it degrades to Deep k-NN.

### Loss & Training
- **LEMoN itself is completely training-free** and only requires a pre-trained CLIP model.
- Two configurations:
    - **LEMoN$_{opt}$**: Optimizes hyperparameters $k, \beta, \gamma, \tau_{1,n}, \tau_{2,n}, \tau_{1,m}, \tau_{2,m}$ on a labeled validation set.
    - **LEMoN$_{fix}$**: Uses fixed, reasonable hyperparameters ($k=30, \beta=\gamma=5, \tau_1=0.1, \tau_2=5$), requiring no validation set.
    - The performance gap between the two is only ~1.7% in AUROC.

## Key Experimental Results

### Main Results (Label Error Detection - Classification Scenario)

| Dataset | Method | Requires Training? | AUROC (%) | AUPRC (%) | F1 (%) |
|--------|------|---------|-----------|-----------|--------|
| CIFAR-10 (human noise) | AUM | Yes | 98.3 | 97.9 | 94.0 |
| | Datamap | Yes | 98.2 | 97.6 | 93.4 |
| | CLIP Sim. | No | 93.8 | 92.4 | 86.9 |
| | Deep k-NN | No | 96.2 | 93.8 | 89.3 |
| | **LEMoN$_{opt}$** | **No** | **98.1** | **97.4** | **93.1** |
| CIFAR-100 (human noise) | AUM | Yes | 92.2 | 89.9 | 83.9 |
| | CLIP Sim. | No | 78.5 | 72.1 | 69.2 |
| | **LEMoN$_{opt}$** | **No** | **90.8** | **87.4** | **81.3** |

### Main Results (Label Error Detection - Captioning Scenario)

| Dataset | Method | AUROC (%) | AUPRC (%) | F1 (%) |
|--------|------|-----------|-----------|--------|
| MSCOCO | CLIP Sim. | 93.8 | 93.0 | 87.5 |
| | LLaVA | 80.3 | 63.4 | 74.9 |
| | **LEMoN$_{opt}$** | **95.6** | **94.6** | **89.3** |
| MIMIC-CXR | CLIP Sim. | 64.1 | 51.7 | 48.6 |
| | **LEMoN$_{opt}$** | **70.4** | **60.3** | **57.0** |

### Ablation Study

| Configuration | mmimdb AUROC | mscoco AUROC | Explanation |
|------|-------------|-------------|------|
| Full LEMoN | 86.0% | 95.6% | All components |
| W/o $\tau_1, \tau_2$ | 85.3% (-0.7) | 94.9% (-0.7) | Adaptive weights contribute |
| W/o $s_n$ (Image neighborhood) | 85.4% (-0.6) | 94.6% (-1.0) | Text neighborhood is more important (mmimdb) |
| W/o $s_m$ (Text neighborhood) | 86.1% (-指) | 94.7% (-0.9) | Image neighborhood is more important (mscoco) |
| Only $d_{mm}$ (CLIP Sim.) | 85.1% (-0.9) | 93.8% (-1.8) | Total neighborhood contribution is ~1-2% |

### Downstream Filtering Performance

| Dataset | Method | BLEU-4 | CIDEr | ROUGE |
|--------|------|--------|-------|-------|
| MSCOCO | No filtering (40% noise) | 27.5 | 54.3 | 36.5 |
| | CLIP Sim. filtering | 31.1 | 64.8 | 39.8 |
| | **LEMoN$_{opt}$ filtering** | **31.4** | **65.4** | **40.1** |
| | Clean data (Upper bound) | 32.0 | 66.3 | 40.4 |

### Key Findings
1. **Training-free method approaches training-based methods**: LEMoN$_{opt}$ achieves an AUROC of 98.1% on CIFAR-10, which is only 0.2% lower than AUM (98.3%), without requiring any classifier training.
2. **Significant lead in captioning scenarios**: Improves over CLIP Sim. by 1.8% AUROC and 1.8% F1 on MSCOCO.
3. **LEMoN$_{fix}$ remains strong without a validation set**: The fixed hyperparameter version only loses 1.7% AUROC on average.
4. **Modality reliance varies across datasets**: mmimdb relies more on the text neighborhood (movie posters + plot summaries), while MSCOCO relies more on the image neighborhood.
5. **Filtering almost recovers clean-data performance**: On MSCOCO, CIDEr after LEMoN filtering is 65.4 vs. 66.3 for clean data.

## Highlights & Insights
- **Elegant and versatile method**: A unified framework that handles both classification and captioning scenarios, generalizing to various datasets (natural images, movie posters, chest X-rays).
- **Solid theoretical support**: Proposition 4.1 proves the Lipschitz robustness of CLIP loss to noisy labels; Proposition 4.2 proves that contrastive learning embeddings can naturally distinguish between correct and incorrect labels.
- **High practicality**: The LEMoN$_{fix}$ version does not require a labeled validation set at all, needing only a pre-trained CLIP model to run.
- **Validation in the medical domain**: Also effective on MIMIC-CXR (chest X-rays + radiology reports); training CLIP from scratch solely on noisy data can substitute out-of-domain pre-training.

## Limitations & Future Work
- The performance gain on specialized domains like MIMIC-CXR is relatively limited (~6% AUROC improvement), indicating that the embedding quality of out-of-domain CLIP is a bottleneck.
- Instance-dependent noise—a more realistic but challenging noise type—was not tested.
- Real-world label errors can be ambiguous and subjective; the binary "correct/incorrect" assumption may be overly simplified.
- The hyperparameter search space is large (7 hyperparameters); even though LEMoN$_{fix}$ is effective, the transferability of its optimal values requires more validation.
- Integrating LEMoN scores with downstream training loops (e.g., adaptive filtering) was not explored.

## Related Work & Insights
- LEMoN is a natural generalization of Deep k-NN and CLIP Similarity, unifying both directions through multimodal neighborhoods.
- Highly applicable to image caption quality control—it can be used for automatic filtering in large-scale dataset description construction pipelines.
- Insight: In other multimodal tasks (e.g., video-text, audio-text), similar neighborhood-based methods might be equally effective.
- Training in-domain CLIP from scratch (even on noisy data) surprisingly outperforms large-scale out-of-domain pre-training, which is food for thought.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces multimodal neighborhood information to label error detection, supported by theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, 12 baselines, theoretical analysis, thorough ablation studies, and real-world validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, logically progressing through theory, experiments, ablations, and real-world validation.
- Value: ⭐⭐⭐⭐ Highly practical method, good generalization, significant import for data quality control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](../../ACL2025/multimodal_vlm/error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ICML 2025\] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](lada_scalable_label-specific_clip_adapter_for_continual_learning.md)
- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](../../NeurIPS2025/multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](../../ACL2025/multimodal_vlm/unsolvable_problem_detection.md)

</div>

<!-- RELATED:END -->
