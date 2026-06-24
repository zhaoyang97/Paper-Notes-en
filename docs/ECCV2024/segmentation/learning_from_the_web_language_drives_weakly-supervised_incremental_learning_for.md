---
title: >-
  [Paper Note] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation
description: >-
  [ECCV 2024][Segmentation][weakly-supervised] This work is the first to propose using entirely web images (rather than well-curated dataset images) for weakly-supervised incremental semantic segmentation. By filtering web images with a Fourier domain discriminator and using a caption-driven rehearsal strategy to preserve old class knowledge, it achieves 73.4% mIoU under the PASCAL VOC 15-5 setting.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "weakly-supervised"
  - "incremental learning"
  - "web images"
  - "vision-language model"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 2e58207339c57f4f
---

# Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.13363](https://arxiv.org/abs/2407.13363)  
**Code**: [https://github.com/dota-109/Web-WILSS](https://github.com/dota-109/Web-WILSS)  
**Area**: Semantic Segmentation / Incremental Learning  
**Keywords**: weakly-supervised, incremental learning, web images, vision-language model, catastrophic forgetting

## TL;DR
This work is the first to propose using entirely web images (rather than well-curated dataset images) for weakly-supervised incremental semantic segmentation. By filtering web images with a Fourier domain discriminator and using a caption-driven rehearsal strategy to preserve old class knowledge, it achieves 73.4% mIoU under the PASCAL VOC 15-5 setting.

## Background & Motivation
**Background**: Class-incremental learning for semantic segmentation (CILSS) allows models to learn new classes step-by-step, but traditional approaches rely on expensive pixel-level annotations. Weakly-supervised incremental semantic segmentation (WILSS) reduces the annotation burden to image-level labels in incremental steps, but still requires the use of well-curated dataset images.

**Limitations of Prior Work**: (1) Existing WILSS methods such as WILSON and FMWISS still require training images from the target dataset; (2) current methods suffer from severe performance degradation under multi-step single-class incremental settings (e.g., 15-1); (3) they do not support incremental steps using only single-class new data and require negative samples.

**Key Challenge**: In practical scenarios, when pre-trained models need to adapt to new classes, target domain data may be extremely limited (e.g., due to privacy constraints), making web-harvested data an inevitable alternative. However, web data poses two key challenges: (1) domain distribution differs from the training data; (2) image-level labels are severely noisy (images searched by class names may contain multiple classes or even lack the target class).

**Goal**: To learn new classes during incremental steps entirely using web images (instead of dataset images), while also leveraging web images for old-class rehearsal to prevent catastrophic forgetting.

**Key Insight**: (1) Use Fourier domain features to train a domain discriminator that filters web images close to the training data distribution; (2) replace simple class name searches with caption models to provide multi-class supervision; (3) store captions instead of images to query old-class rehearsal images, addressing privacy and storage concerns.

**Core Idea**: A Fourier-domain discriminator combined with caption-driven web image selection and rehearsal, enabling weakly-supervised incremental segmentation without the original dataset.

## Method

### Overall Architecture
Based on the WILSON framework (shared encoder $E^t$ + segmentation decoder $D^t$ + localizer $L^t$), this method replaces the data source from dataset images to web images in incremental steps. Two parallel data pipelines are established: (1) New Class Learning Pipeline: search images from the web using class names $\rightarrow$ filter through a Fourier domain discriminator $\rightarrow$ provide multi-label supervision using a caption model; (2) Old Class Preservation Pipeline: search images from the web using previously saved captions $\rightarrow$ regenerate captions and filter semantically $\rightarrow$ train with pseudo-labels.

### Key Designs
1. **Fourier Domain-based Discriminator**

    - **Function**: Filter web images that share a similar distribution with the original training data.
    - **Mechanism**: An EfficientNet-B0 discriminator $M_D$ is trained at the initial step ($t=0$), taking the Fourier transform amplitude spectrum of the images $(p_{ds}, p_{web}) = M_D(|\mathcal{F}(\mathbf{x})|)$ as input, where $p_{ds}$ is the probability of belonging to the original dataset. A web image is retained only if $p_{ds}/p_{web} > 1$.
    - **Design Motivation**: The amplitude spectrum in the Fourier domain exhibits highly consistent statistical characteristics across different classes (mainly reflecting style/texture rather than semantics). Thus, the discriminator trained in the initial step remains effective in subsequent steps (even on unseen new classes). Compared to pixel-domain discrimination, the Fourier domain is more robust to class variations.

2. **Caption Labeling**

    - **Function**: Generate multi-class image-level labels for web images using visual-language models, replacing simple search keyword labels.
    - **Mechanism**: Generate image captions $w = M_{CAP}(\mathbf{x})$ using the OpenFlamingo model, and match the nouns in the description against a pre-defined class vocabulary $\mathcal{W}^c$ (including synonyms, plurals, etc.): if $\exists w_i \in w : w_i \in \mathcal{W}^c$, then $y^c = 1$.
    - **Design Motivation**: Web images searched by a class name might contain multiple classes (e.g., searching for 'boat' yields an image of 'a person standing on a boat'). Simple single-label supervision can cause errors. Captions can identify both 'person' and 'boat' simultaneously to provide accurate multi-labels, and can also identify and discard images that do not contain the target class.

3. **Caption-based Querying**

    - **Function**: Store captions of old images instead of the images themselves, and use these captions to search for similar images from the web for rehearsal.
    - **Mechanism**: Captions are generated and saved for all training images during the initial step. During incremental steps, these captions are used as search queries to download images from the web: $\mathcal{X}_r^{web} = \{\mathbf{x} = \mathcal{D}^{web}(q') | q' = M_{CAP}(\mathbf{x}) : \mathbf{x} \in \mathcal{X}\}$.
    - **Design Motivation**: (1) Storing captions drastically reduces storage and avoids privacy issues compared to storing images; (2) images searched using captions contain richer semantic context (co-occurring classes), which is closer to the original distribution than images searched by class names alone.

4. **Caption-based Filtering**

    - **Function**: Verify whether the downloaded rehearsal images preserve the core semantic content of the original images.
    - **Mechanism**: Regenerate caption $q''$ for the downloaded images, extract the first two nouns $(n_1', n_2')$ and $(n_1'', n_2'')$ from both captions (using Penn TreeBank parsing), construct vector descriptors $v$ by retrieving hypernyms using WordNet, and calculate the cosine similarity. The image is kept if the similarity of any noun pair exceeds a threshold $T=0.6$.
    - **Design Motivation**: Caption-based searches do not guarantee content matches, necessitating double-verification. Using WordNet's semantic hierarchy instead of exact matching allows for flexible matching of synonyms and hypernyms/hyponyms (e.g., dog/animal).

### Loss & Training
- **Loss Function**: $\mathcal{L} = \mathcal{L}_{SEG} + \mathcal{L}_{CLS} + \mathcal{L}_{KDE} + \mathcal{L}_{KDL}$
    - $\mathcal{L}_{SEG}$: Pixel-level segmentation loss (pseudo-label supervision)
    - $\mathcal{L}_{CLS}$: Image-level classification loss (multi-label soft margin loss)
    - $\mathcal{L}_{KDE}$: Encoder feature distillation loss (MSE between $E^t$ and $E^{t-1}$)
    - $\mathcal{L}_{KDL}$: Consistency loss between the localizer and the old model
- **Pseudo-label Generation**: Merges localizer predictions and old model predictions. The localizer is used for new classes, and the old model is used for old classes.
- **Network Setup**: DeepLabV3 with ResNet-101 (VOC) / Wide-ResNet-38 (COCO); SGD optimizer, 30 epochs for the initial step and 40 epochs for incremental steps.
- **Web Data**: Download 10K candidates per class $\rightarrow$ filter 500 for training; for rehearsal, download 20 images per caption and retain 100 images in total.

## Key Experimental Results

### Main Results (PASCAL VOC Single-step Multi-class Setting)

| Method | Training Data | Rehearsal | 15-5 Disjoint All | 15-5 Overlap All | 10-10 Disjoint All | 10-10 Overlap All |
|------|---------|-----------|-------------------|------------------|--------------------|--------------------|
| WILSON | VOC | - | 67.3 | 67.2 | 60.8 | 65.0 |
| RaSP | VOC | - | - | 70.0 | - | 65.9 |
| FMWISS | VOC | VOC(50) | 70.7 | 73.3 | 64.6 | 69.1 |
| **Ours** | **VOC** | **WEB(100)** | **71.1** | **73.3** | **61.7** | **65.7** |
| **Ours** | **VOC** | **WEB(500)** | **72.0** | **73.4** | **61.0** | **65.3** |
| WILSON | WEB | - | 68.9 | 67.8 | 58.6 | 62.1 |
| **Ours** | **WEB** | **WEB(100)** | **70.5** | **71.7** | **60.4** | **65.3** |

### Ablation Study

| Configuration | 15-5 Overlap All | Description |
|------|------------------|------|
| Baseline WILSON (WEB) | 67.8 | No filtering and no caption for web images |
| + Fourier Domain Discriminator | 68.4 | +0.6, domain filtering is effective |
| + Caption Labeling | 69.5 | +1.7, multi-label supervision is critical |
| + Caption Rehearsal | **71.7** | +3.9, highly effective for old class preservation |

### Key Findings
- Using entirely web images (both training and rehearsal from the web) achieves performance close to that using the original dataset (71.7% vs 73.4% in 15-5 overlap), proving the viability of web data.
- Caption labeling contributes the most (+1.7%), indicating that the multi-class co-occurrence problem in web images is indeed the core challenge.
- The Fourier domain discriminator can generalize to domain filtering of unseen new classes after being trained only in the initial step.
- More rehearsal images (500 vs. 100) help when using the original dataset, but perform slightly worse under the pure web data setting, demonstrating that web data quality is unstable and quality matters more than quantity.
- Rehearsal images queried by captions contain much richer semantic context than those queried by class names.

## Highlights & Insights
- **First entirely web-based WILSS framework**: It lowers data requirements for incremental learning from well-curated datasets to merely class names, significantly extending the practical applicability (e.g., privacy-sensitive scenarios, new domain adaptation). Achieving reasonable performance under the Web+Web setting is a crucial proof of its practical value.
- **Caption as a lightweight memory medium**: Replacing image storage with captions for rehearsal is a clever design—storage cost is virtually zero, with no privacy risks, while diverse images with accurate semantic contexts can be reconstructed through search-and-filter.
- **Cross-class generalization in the Fourier domain**: Exploiting the style/texture statistical properties of the amplitude spectrum (independent of semantic content) to achieve cross-class domain discrimination requires training only once in the initial step for lifelong deployment.

## Limitations & Future Work
- The quality and diversity of web images are constrained by search engines, and results can vary significantly across different engines or search queries in different languages.
- The description quality of the caption model (OpenFlamingo) directly affects labeling and querying; incorrect captions will introduce noise.
- The Fourier domain discriminator may fail under extreme distribution shifts (e.g., from natural scenes to medical images).
- Performance still suffers a major drop under multi-step single-class incremental settings (e.g., the 15-1 setting), and compounding errors in long step sequences remain to be addressed.
- Evaluation is currently limited to PASCAL VOC and COCO; more diverse datasets are yet to be assessed.

## Related Work & Insights
- **vs WILSON**: WILSON serves as the baseline framework for this work but requires dataset images. By replacing them with web images, the proposed method improves performance under the 15-5 setting from 67.3% to 71.1%-72.0% (with VOC), and from 68.9% to 70.5% (pure WEB).
- **vs FMWISS**: FMWISS relies on DINO+MaskCLIP for pixel-level pseudo-supervision and requires VOC images for rehearsal. In contrast, this method uses only image-level labels and web images, outperforming FMWISS on 15-5 disjoint (72.0% vs. 70.7%).
- **vs RECALL**: RECALL is the first method to utilize web data in CILSS, but it uses it only for old class rehearsal with pixel-level pseudo-labels. This method applies web data to both new class learning and old class preservation under a purely weakly-supervised setting.

## Rating
- Novelty: ⭐⭐⭐⭐ It is the first to propose a purely web-based WILSS setting. Both the Fourier domain discriminator and caption rehearsal are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both VOC and COCO datasets across various incremental settings (15-5, 10-10, 15-1) with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and intuitive methodology flowcharts.
- Value: ⭐⭐⭐⭐ Lowering the data barrier for incremental segmentation is of great practical significance, and the paradigm of leveraging web data is highly noteworthy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[ECCV 2024\] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](../../CVPR2026/segmentation/from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)

</div>

<!-- RELATED:END -->
