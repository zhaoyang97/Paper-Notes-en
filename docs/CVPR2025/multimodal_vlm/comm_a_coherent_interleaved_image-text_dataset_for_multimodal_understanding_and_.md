---
title: >-
  [Paper Note] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation
description: >-
  [CVPR 2025][Multimodal VLM][Interleaved image-text dataset] Addressing the core issues of poor narrative coherence and inconsistent entity styles in existing interleaved image-text datasets (such as MMC4/OBELICS), this work constructs the CoMM dataset (227K documents, 2.28M images). By targeting instructional content collection combined with a multi-perspective quality filtering strategy, it ensures text coherence, image consistency, and image-text alignment…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Interleaved image-text dataset"
  - "Multimodal coherence"
  - "Few-shot learning"
  - "Image-text generation"
  - "Preference dataset"
date: 2026-05-08
content_hash: ed8d4d37f6f0746e
---

# CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation

**Conference**: CVPR 2025  
**arXiv**: [2406.10462](https://arxiv.org/abs/2406.10462)  
**Code**: [GitHub](https://github.com/HKUST-LongGroup/CoMM)  
**Area**: Multimodal VLM / Dataset  
**Keywords**: Interleaved image-text dataset, Multimodal coherence, Few-shot learning, Image-text generation, Preference dataset

## TL;DR

Addressing the core issues of poor narrative coherence and inconsistent entity styles in existing interleaved image-text datasets (such as MMC4/OBELICS), this work constructs the CoMM dataset (227K documents, 2.28M images). By targeting instructional content collection combined with a multi-perspective quality filtering strategy, it ensures text coherence, image consistency, and image-text alignment, while proposing four interleaved generation evaluation tasks.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made rapid progress in cross-modal generation, but generating coherent interleaved image-text sequences remains challenging. Models like Emu2 and DreamLLM suffer from narrative incoherence and inconsistent entity styles during interleaved generation.

**Limitations of Prior Work**: The root cause lies in the poor quality of training data. The two major existing interleaved image-text datasets, MMC4 and OBELICS, exhibit severe limitations: (1) **Sparse images**—the median number of images is only 2 images/document for MMC4 and 1 image/document for OBELICS; (2) **Weak image-text association**—web-crawled images are often advertisements or decorative pictures, unrelated to the main text; (3) **Inconsistent styles**—images within the same document originate from different sources, leading to fragmented visual styles.

**Key Challenge**: Narrative coherence and entity style consistency are intrinsic requirements for interleaved image-text content. However, large-scale automatically crawled web data naturally lacks these properties. Manual annotation is prohibitively expensive, necessitating automated yet effective quality control paradigms.

**Key Insight**: Rather than crawling random web pages, this work performs targeted collection from instructional (such as WikiHow tutorials) and visual storytelling websites. Due to a unified intent (e.g., "cooking") and structured presentation (Step 1/2/3), such content naturally possesses textual coherence and image consistency. On top of this, a multi-dimensional filtering strategy is applied to further enhance quality.

## Method

### Overall Architecture

The construction of the CoMM dataset consists of three stages: (1) Targeted collection of raw interleaved image-text data from instructional and visual storytelling websites; (2) Application of a multi-perspective quality filtering strategy comprising text sequence filtering, image sequence filtering, and image-text alignment filtering; (3) Creation of preference learning datasets by shuffling step sequences to construct negative samples. The final scale consists of 227K documents and 2.28M images, averaging 10.1 images per document (compared to 3.6 for MMC4 and 2.9 for OBELICS).

### Key Designs

1. **Multi-Perspective Filter Strategy**:
    - **Text Sequence Filtering**: An LLM (Llama3) is utilized to evaluate the progression and coherence between textual steps in documents, filtering out logically incoherent text sequences based on evaluation scores.
    - **Image Sequence Filtering**: A filtering metric based on the CLIP vision encoder is designed as: $\mathcal{F}(\{x_i\}) = \frac{1}{N-1}\sum_{i=2}^{N}\text{Sim}(x_i, x_{i-1}) - \frac{2}{(N-1)(N-2)}\sum_{i=2}^{N}\sum_{j=1}^{i-1}\text{Sim}(x_i, x_j)$. The first term measures the visual coherence of adjacent images (higher is better), while the second term measures the global image diversity (lower is better). Balancing these two terms ensures that the sequence is both coherent and progressive.
    - **Image-Text Alignment Filtering**: First, CLIP is used to calculate image-text similarity scores to eliminate poorly matched pairs with scores below 0.1. Then, GPT-4o or Llama3 is leveraged to evaluate image-text alignment in context (since the text in interleaved content describes operational progress rather than simple image captions, using CLIP alone is insufficient).

2. **Preference Dataset Construction**: Negative samples are constructed via four methods: shuffling text order (keeping images fixed), shuffling image order (keeping text fixed), independently shuffling both text and images, and shuffling the sequence of steps (keeping image-text alignment within steps but randomizing the inter-step order). Pairing these negative samples with the original positive samples forms a preference dataset, which can be deployed in post-training stages such as RLHF/DPO.

3. **Four Interleaved Generation Evaluation Tasks**:
    - Task 1 (Image-to-Text Sequence Generation): Given an image sequence $\{x_i\}_{i=1}^N$, generate the corresponding text sequence.
    - Task 2 (Text-to-Image Sequence Generation): Given a text sequence $\{t_i\}_{i=1}^N$, generate the corresponding image sequence.
    - Task 3 (Interleaved Content Continuation): Given the first $k$ steps of the interleaved image-text content, generate the subsequent steps.
    - Task 4 (Question-guided Interleaved Generation): Given only a prompt/question, generate the complete interleaved image-text content.

### Loss & Training

The dataset can be employed for interleaved image-text modeling during the Supervised Fine-Tuning (SFT) phase, as well as for RLHF post-training via the preference dataset. Regarding evaluation metrics, METEOR/ROUGE is utilized for text, while FID/IS/SSIM/PSNR is used for images. Interleaved content is evaluated using GPT-4o across four dimensions: image coherence, image quality, document completeness, and image-text relevance. Additionally, the Illustration Relevance Score (IRS) is proposed to evaluate the association between images and text.

## Key Experimental Results

### Main Results: Comparison of Dataset Quality

| Metric | MMC4 | OBELICS | **CoMM** |
|------|------|---------|----------|
| Progression (DLP, GPT-4o) | 4.75 | 5.97 | **7.64** |
| Completeness (CPL, GPT-4o) | 5.12 | 5.88 | **7.07** |
| Image-Text Alignment (ITA, GPT-4o) | 4.66 | 3.81 | **8.91** |
| Image Sequence Quality (ImgS, $\mathcal{F}$) | 0.21 | 1.00 | **4.27** |
| Median Images/Doc | 2 | 1 | **4** |
| Mean Images/Doc | 3.6 | 2.9 | **10.1** |

### Ablation Study: Few-shot Downstream Tasks (CIDEr / Accuracy, Representative Results Selected)

| Training Data Source | COCO 0-shot | COCO 32-shot | TextVQA 32-shot | VQAv2 32-shot |
|---------------|-------------|-------------|----------------|---------------|
| Baseline | 79.5 | 99.7 | 30.6 | 56.9 |
| + MMC4 | 97.1 | 107.0 | 27.2 | 49.6 |
| + OBELICS | 83.5 | 102.2 | 29.2 | 53.2 |
| **+ CoMM** | **100.3** | **112.9** | **36.3** | **57.5** |

### Key Findings

- CoMM significantly outperforms MMC4 and OBELICS across all four quality dimensions, particularly in Image-Text Alignment (ITA), which increases from 4.66/3.81 to 8.91.
- Models trained on CoMM consistently outperform those trained on MMC4 and OBELICS in few-shot performance across seven downstream tasks, with the advantage being particularly pronounced in long-context (16/32-shot) scenarios.
- In contrast to the sparse image counts in MMC4/OBELICS (median of 1-2 images/document), CoMM averages 10.1 images/document, showing a positive correlation between document length and the number of images.
- The targeted collection strategy (from instructional websites) secures high content quality from the source, while the multi-perspective filtering strategy further eliminates noise.

## Highlights & Insights

- **The "Data Source Selection Over Filtering" Strategy**: Instead of crawling and filtering arbitrary web pages, this work performs direct collection from naturally high-quality instructional and visual storytelling websites. This decision ensures a high-quality baseline before any downstream filtering is applied.
- **Ingenious Image Sequence Filtering Metric**: Designed as a difference between the coherence term and the diversity term, it avoids degenerate cases like completely identical image sequences (which would score highly but carry no information) while maintaining visual style consistency.
- **Zero-Cost Preference Dataset Construction**: Negative samples are generated automatically by shuffling step configurations, bypassing any need for manual annotation.
- **Proposing Four Evaluation Tasks and Frameworks**: Fills the gap in evaluating interleaved image-text generation capability.

## Limitations & Future Work

- The data sources are dominated by English instructional websites, causing the domain coverage to lean heavily toward life skills and DIY, while knowledge-intensive domains like academic and news remain underrepresented.
- The NSFW filtering threshold is set to 0.1 (highly strict), which may excessively filter out relevant medical or educational content.
- The actual performance of the preference dataset during RLHF/DPO post-training remains unevaluated (constructed but not yet trained on).
- Visual quality evaluations rely heavily on subjective grading via GPT-4o, introducing potential evaluation biases.
- Promising Direction: Video instructional content (e.g., YouTube tutorials) can be explored as richer sources for interleaved multimodal data.

## Related Work & Insights

- **vs MMC4 (Zhu et al.)**: MMC4 utilizes CLIP to retrieve and insert illustrations based on the C4 text corpus, leading to sparse images with weak associations. In contrast, CoMM employs targeted collection from the source, resulting in abundant and naturally aligned illustrations.
- **vs OBELICS (Laurençon et al.)**: OBELICS preserves the original structure of web documents but suffers from highly variable quality. CoMM focuses on coherent narrative scenarios coupled with active multi-perspective filtering.
- **vs OpenLEAF (An et al.)**: OpenLEAF utilizes GPT-4 and SDXL to generate interleaved content at inference time in a training-free manner, but heavily relies on the generation models' understanding of textual descriptions, which is less robust than direct training on high-quality interleaved data.
- **Insight**: Data quality outperforms data scale—CoMM with 227K high-quality documents (averaging 10.1 images/document) demonstrates superior performance compared to MMC4/OBELICS containing millions of noisy documents.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty ⭐⭐⭐⭐: The data source selection strategy and the multi-perspective filtering design are insightful, although the core idea (better data yields better models) is naturally expected.
- Experimental Thoroughness ⭐⭐⭐⭐: High coverage, including dataset quality comparison, 7 downstream few-shot tasks, and a novel evaluation benchmark.
- Writing Quality ⭐⭐⭐⭐: Clear structure, well-defined quality metrics, and rich data analysis visualizations.
- Value ⭐⭐⭐⭐⭐: High utility as the dataset is open-source and ready for use, while the four proposed evaluation tasks fill a critical evaluation gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICLR 2026\] A High Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation](../../ICLR2026/multimodal_vlm/a_high_quality_dataset_and_reliable_evaluation_for_interleaved_image-text_genera.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)

</div>

<!-- RELATED:END -->
