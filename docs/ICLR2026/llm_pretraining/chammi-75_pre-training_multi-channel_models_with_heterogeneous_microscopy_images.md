---
title: >-
  [Paper Note] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images
description: >-
  [ICLR2026][LLM Pretraining][microscopy] This work introduces CHAMMI-75—the largest heterogeneous multi-channel microscopy image pre-training dataset (2.8M images, 75 sources, 25 channel types, 16 species)—and demonstrates that imaging modality diversity is the key factor for improving generalization of multi-channel models. The trained MorphEm model achieves state-of-the-art performance on 6 out of 7 benchmarks.
tags:
  - ICLR2026
  - LLM Pretraining
  - microscopy
  - multi-channel imaging
  - dataset curation
  - self-supervised learning
  - cell morphology
date: 2026-05-08
content_hash: 30f981dc44a41150
---

# CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images

**Conference**: ICLR2026  
**arXiv**: [2512.20833](https://arxiv.org/abs/2512.20833)  
**Code**: [https://github.com/CaicedoLab/CHAMMI-75](https://github.com/CaicedoLab/CHAMMI-75)  
**Area**: LLM Pre-training  
**Keywords**: microscopy, multi-channel imaging, dataset curation, self-supervised learning, cell morphology

## TL;DR
This work introduces CHAMMI-75—the largest heterogeneous multi-channel microscopy image pre-training dataset (2.8M images, 75 sources, 25 channel types, 16 species)—and demonstrates that imaging modality diversity is the key factor for improving generalization of multi-channel models. The trained MorphEm model achieves state-of-the-art performance on 6 out of 7 benchmarks.

## Background & Motivation

**State of the Field**: Microscopy imaging is a foundational tool in biological research. Unlike RGB natural images, microscopy images have a variable number of channels (1 to tens), each encoding distinct fluorescence signals. Deep learning is widely used for microscopy image analysis, but typically requires models with a fixed number of input channels—different experiments use different channel configurations, making models non-transferable across experiments.

**Limitations of Prior Work**: (a) **Fixed channels**—existing models adapt RGB architectures to a specific channel count and cannot handle new channel combinations; (b) **Data fragmentation**—multi-channel microscopy images are scattered across various public platforms with inconsistent formats and metadata, making unified use difficult; (c) **Insufficient scale**—existing datasets such as IDRCell100k contain only 100K images.

**Root Cause**: Training a generalizable cell morphology foundation model requires large-scale data covering diverse imaging modalities, species, and channel combinations—yet no such dataset exists.

**Paper Goals**: Construct the first large-scale heterogeneous multi-channel microscopy image dataset and systematically evaluate its effectiveness as a pre-training resource.

**Starting Point**: Images from 75 biological studies are collected across 18 public data-hosting platforms, unified with metadata annotations, and carefully curated with deduplication to construct a high-quality pre-training dataset.

**Core Idea**: Data diversity—especially imaging modality diversity—is the critical factor for training channel-adaptive cell morphology models, and CHAMMI-75 provides precisely this diversity.

## Method

### Overall Architecture
The work comprises three components: (1) CHAMMI-75 dataset construction—data acquisition → metadata integration → data curation (deduplication + quality control) → cell segmentation annotation; (2) six evaluation benchmarks (including three newly proposed) covering different channel configurations and domain transfer scenarios; (3) systematic experiments evaluating the pre-training value, influencing factors, and scalability of the dataset.

### Key Designs

1. **Data Curation Pipeline**:

    - Function: Filters ~26M downloaded images down to 2.8M high-quality, low-redundancy pre-training images.
    - Mechanism: Four-step deduplication—(a) random sampling of 2D slices from 3D images; (b) random frame sampling from live-cell time-lapse videos; (c) random well sampling from replicate control conditions; (d) K-means clustering to select a diverse, high-quality subset.
    - Design Motivation: Microscopy data contains abundant near-duplicate samples (e.g., adjacent slices of the same 3D volume, consecutive frames of time-lapse videos), which would cause severe overfitting if used directly. Systematic metadata-guided curation ensures diversity.

2. **Bag of Channels (BoC) vs. Multi-Channel Attention (MCA)**:

    - Function: Evaluates two multi-channel processing strategies.
    - Mechanism: BoC independently feeds each channel into the backbone for feature extraction and then concatenates the results—channel-agnostic and scalable; MCA organizes all channel tokens into a long sequence to model cross-channel relationships—more informative but 3–5× more computationally expensive.
    - Design Motivation: BoC consistently outperforms MCA in SSL settings by up to 19%, indicating that learning cross-channel associations in an unsupervised setting is difficult. BoC is more practical and easier to scale.

3. **MorphEm Model**:

    - Function: The best pre-trained model based on CHAMMI-75.
    - Mechanism: ViT-small + DINO self-supervised learning + BoC strategy, trained on the full CHAMMI-75 (2.8M images). Training required 2,352 GPU hours.
    - Design Motivation: Systematic scaling experiments show that DINO > MAE > SimCLR, BoC > MCA, and ViT-small is both feasible and effective under academic compute budgets.

4. **Evaluation Benchmark Design**:

    - Function: Six benchmarks covering different generalization scenarios.
    - Mechanism: Includes in-distribution channel tasks (CHAMMI, HPAv23, JUMP-CP), channel generalization tasks (CellPHIE with 14 channels—unseen combinations during training), and cross-modality + cross-domain tasks (RBC-MC brightfield imaging flow cytometry).
    - Design Motivation: In practice, new experiments frequently employ novel channel combinations or even new imaging modalities. CellPHIE and RBC-MC specifically test this most challenging form of generalization.

### Loss & Training
DINO-BoC self-supervised learning: single-channel input to ViT-small, student-teacher framework. After feature extraction, weights are frozen without fine-tuning, and models are evaluated directly on downstream tasks via linear probing or nearest-neighbor retrieval.

## Key Experimental Results

### Main Results (Comparison across 6 benchmarks)

| Model | Multi-channel | Pre-train Data | CHAMMI ↑ | HPAv23 ↑ | JUMP-CP1 ↑ | CellPHIE ↑ | RBC-MC ↑ |
|-------|--------------|----------------|---------|---------|-----------|-----------|---------|
| SubCell (WSL, ViT-B) | Manual select | HPAv23 | 53.38 | **69.33** | 77.60 | 71.23 | 59.10 |
| DINOv2 | BoC | LVD-142M | 37.93 | 53.76 | 75.84 | 72.27 | 59.41 |
| OpenPhenom | MCA | RxRx+JUMP | 38.22 | 49.13 | 74.26 | 75.56 | 64.43 |
| IDRCell100k | BoC | IDRCell | 37.38 | 44.05 | 72.37 | 79.14 | 55.85 |
| **MorphEm** | BoC | **CHAMMI-75** | **48.75** | **58.87** | **76.32** | **80.51** | **68.34** |

### Ablation Study (Impact of data factors — relative performance differences)

| Factor | With Factor | Without Factor | Impact |
|--------|------------|----------------|--------|
| Heterogeneous vs. specialized data | +38% | −27% | **Largest** |
| Multiple imaging modalities vs. fluorescence only | +15% | −13% | **Second largest** |
| Varied magnification levels | +3% | −3% | Moderate |
| Varied cell lines | +1% | −1% | Small |
| Varied channel counts | +1% | −1% | Small |

### Key Findings
- **Data heterogeneity >> data volume**: 100K heterogeneous images (IDRCell100k) vs. 2.8M heterogeneous images (CHAMMI-75)—the latter leads by a large margin. Meanwhile, specialized data of comparable scale cannot compete with heterogeneous data at all.
- **Imaging modality diversity is the key factor**: A model trained on a small number of non-mainstream imaging modalities (12 types) outperforms one trained on only two mainstream modalities by 28%. This suggests that models acquire more robust representations by learning to handle variation across different physical imaging processes.
- **BoC >> MCA under SSL**: BoC consistently outperforms MCA by 19% while requiring 3–5× less computation. Learning cross-channel associations in an unsupervised setting is inherently difficult.
- **Small model + good data can surpass large models**: ViT-small MorphEm (SSL) outperforms SubCell (WSL, ViT-base) by 13% on CellPHIE and 15% on RBC-MC—data quality and diversity matter more than model size.
- **DINO > MAE > SimCLR**: DINO consistently achieves the best performance in microscopy image SSL, likely because its teacher-student framework is better suited for capturing global features of biological morphology.

## Highlights & Insights
- **Value of the data curation methodology**: The curation process itself—filtering from 26M to 2.8M images—constitutes a contribution in its own right. The systematic deduplication and diversity-preservation strategy can serve as a template for large-scale dataset construction in other domains.
- **Insight on imaging modality diversity**: The key is not simply "more data is better," but rather "more types of imaging processes are better." This directly informs data strategy for foundation models—priority should be given to collecting data from diverse physical imaging processes.
- **Channel generalization to 14 channels**: Although models are trained on at most 7 channels, they generalize zero-shot to the 14-channel CellPHIE benchmark. The channel-independent processing of BoC makes this generalization naturally achievable—analogous to patch-independent processing in ViT for natural images.

## Limitations & Future Work
- **Compute constraints**: Due to academic compute limitations, only ViT-small was tested. The paper's own scaling experiments show that ViT-large can yield an additional 10% improvement—larger-scale training remains an open opportunity.
- **BoC discards cross-channel information**: Although practical, the BoC strategy ignores biological co-localization information across channels (e.g., spatial relationships between DAPI and phalloidin). Future work should explore methods for effectively leveraging cross-channel information under SSL.
- **Metadata noise**: Despite extensive annotation efforts, metadata still contains noise, which degrades weakly supervised learning performance.
- **Long-tail distribution unresolved**: Channel combination distributions are extremely long-tailed (Figure 4b)—certain channels appear in only a small number of studies, and the representation quality of the model for such rare channels remains unknown.

## Related Work & Insights
- **vs. IDRCell100k**: Both datasets draw from a similar number of sources (79 vs. 75), but CHAMMI-75 contains 30× more images and higher curation quality. A BoC model trained on IDRCell100k consistently underperforms one trained on CHAMMI-75, confirming the value of data quality and scale.
- **vs. SubCell**: SubCell employs weakly supervised learning, a larger model, and manually curated channel combinations to achieve top performance on some tasks. However, in generalization scenarios (novel channels, novel domains), the SSL small model trained on CHAMMI-75 leads by a large margin—demonstrating that diverse training data is the foundation of generalization.
- **Analogy to ImageNet/LAION**: Just as ImageNet drove transformative advances in natural image understanding, CHAMMI-75 has the potential to become the "ImageNet" of microscopy imaging—systematic data engineering driving model breakthroughs.

## Rating
- Novelty: ⭐⭐⭐⭐ In-depth dataset construction and multi-factor ablation analysis, though methodologically primarily builds on existing techniques (DINO + BoC).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive: 7 benchmarks, 6-factor ablation, 3-dimensional scaling experiments, and BoC vs. MCA comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Dataset motivation, construction process, and experimental design are all clearly presented with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Pioneering contribution to the biological imaging foundation model field; dataset, code, and model are all fully open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](stochastic_self-organization_in_multi-agent_systems.md)
- [\[CVPR 2026\] Linking Modality Isolation in Heterogeneous Collaborative Perception](../../CVPR2026/llm_pretraining/linking_modality_isolation_in_heterogeneous_collaborative_perception.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_ethical_data_for_llm_pretraining.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)

<!-- RELATED:END -->
