---
title: >-
  [Paper Note] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Contrastive Data Synthesis] Proposes a data synthesis method inspired by contrastive learning. It automatically generates similar image pairs containing subtle object differences along with their difference descriptions. After fine-tuning MLLMs on this data, it outperforms GPT-4V and Gemini on MMVP by 12 points, achieving an average improvement of 3.06% across 8 general MLLM benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Contrastive Data Synthesis"
  - "Image Difference Captioning"
  - "Multimodal Large Language Models"
  - "Fine-grained Visual Understanding"
  - "Data Augmentation"
date: 2026-05-08
content_hash: a47a52aa537de5ad
---

# Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2408.04594](https://arxiv.org/abs/2408.04594)  
**Code**: [https://github.com/modelscope/data-juicer/tree/ImgDiff](https://github.com/modelscope/data-juicer/tree/ImgDiff)  
**Area**: Multimodal VLM  
**Keywords**: Contrastive Data Synthesis, Image Difference Captioning, Multimodal Large Language Models, Fine-grained Visual Understanding, Data Augmentation

## TL;DR
Proposes a data synthesis method inspired by contrastive learning. It automatically generates similar image pairs containing subtle object differences along with their difference descriptions. After fine-tuning MLLMs on this data, it outperforms GPT-4V and Gemini on MMVP by 12 points, achieving an average improvement of 3.06% across 8 general MLLM benchmarks.

## Background & Motivation

**Background**: The performance of Multimodal Large Language Models (MLLMs) highly depends on the quality of training data. Current training data for MLLMs mainly consists of image-text pairs in the pre-training phase and VQA instruction-following data in the fine-tuning phase, which still leaves a gap in fine-grained image recognition.

**Limitations of Prior Work**: (1) Existing VQA datasets focus on single-image understanding, lacking discrimination training on subtle differences between similar images; (2) Vision encoders like CLIP suffer from the "CLIP-blind" issue, failing to distinguish between images that have similar CLIP features but different content; (3) Existing image difference datasets (e.g., Spot-the-Diff, CLEVR-Change) are small in scale, narrow in domain, and their full-image level difference descriptions are not sufficiently precise.

**Key Challenge**: Improving the fine-grained visual discrimination of MLLMs requires large-scale "similar but different" image pairs. However, manual annotation is extremely costly, and existing synthesis methods (such as InstructPix2Pix) produce difference descriptions that are imprecise and lack spatial focus.

**Goal**: Designs a fully automatic data synthesis pipeline to generate high-quality "object substitution" image pairs and precise region-level difference descriptions.

**Key Insight**: Reference the idea of contrastive learning — allowing the model to learn to distinguish fine-grained differences by comparing similar images.

**Core Idea**: Utilize SDXL + Prompt-to-Prompt to generate similar image pairs, extract different regions through multi-stage filtering, and leverage MLLMs to generate precise region-level difference descriptions, forming high-quality contrastive training data.

## Method

### Overall Architecture
A three-step pipeline: (1) Image Pair Generation: using an LLM for object replacement to produce caption pairs, and generating similar image pairs using SDXL + Prompt-to-Prompt; (2) Difference Area Generator: locating different regions via image similarity filtering, FastSAM instance segmentation, image-text matching filtering, and difference detection; (3) Difference Captions Generator: a two-stage process using MLLMs to annotate region content and generate precise difference descriptions.

### Key Designs

1. **Image Pair Generation (Prompt-to-Prompt + SDXL)**:
    - **Function**: Generating highly similar image pairs where only a small number of objects are replaced.
    - **Mechanism**: (i) Retrieve image captions from caption databases (e.g., MSCOCO or LLaVA pre-training data); (ii) Use an LLM to replace an object name in the caption (e.g., "dog" $\rightarrow$ "cat"); (iii) Generate image pairs based on the original and replaced captions using Stable-Diffusion-XL configured with Prompt-to-Prompt — which ensures only the replaced object changes while keeping the rest consistent by sharing cross-attention layers.
    - **Design Motivation**: Directly comparing to InstructPix2Pix-style methods: (1) Utilizes the more advanced SDXL (instead of SD-1.5) to generate more realistic images; (2) Prompt-to-Prompt offers better control over the scope of modification than editing methods. Out of 118K image pairs, a multi-stage filtering process yields 38,533 highly similar but non-identical image pairs.

2. **Difference Area Generator**:
    - **Function**: Precisely locating the regions (bounding boxes) representing object differences between the image pairs.
    - **Mechanism**: A four-step cascading filter: (i) **Image Similarity Filtering**: Calculate the cosine similarity of the image pair using CLIP, retaining pairs within a specific threshold range (too similar = no difference, too different = completely different); (ii) **FastSAM Segmentation**: Query FastSAM for instance segmentation on each image to obtain all bounding boxes; (iii) **Image-Text Matching Filtering**: Use BLIP to perform image-text matching between cropped sub-images and object names to confirm that the sub-image indeed contains the target object; (iv) **Difference Detection**: Crop both images at the same bounding box position, compute the similarity between these sub-images using CLIP, retain only regions with significant differences, and apply IoU filtering to remove overlapping boxes.
    - **Design Motivation**: Avoids using object detection (such as YOLO) due to limited object categories. This segmentation-and-similarity-comparison-based approach is unrestricted by predefined categories and can detect differences of arbitrary objects. Multi-stage filtering ensures the quality of the 117,779 valid bounding boxes.

3. **Difference Captions Generator**:
    - **Function**: Generating precise textual descriptions for the different regions.
    - **Mechanism**: **Stage-1 (Object Labeling & Filtering)**: Select the top-5 bounding boxes with the lowest similarity, utilize LLaVA-NEXT to describe the content of each region, and filter them via image-text matching and description similarity (using CLIP to compute the similarity between both descriptions, where excessive similarity implies no real difference). **Stage-2 (Difference Captioning)**: Draw red bounding boxes on the images to mark different regions, and feed both images along with region descriptions and visual cues (red boxes) into LLaVA-NEXT to generate descriptions of the specific differences within the red-boxed regions.
    - **Design Motivation**: The key innovation is **region-level difference descriptions** instead of full-image difference; full-image descriptions often omit details or remain too general, while red-box prompting and localized descriptions ensure each difference is precisely captured. Ultimately, 12,688 high-quality "object substitution" samples are selected from 117,779 bounding boxes.

### Loss & Training
- Mix Img-Diff data into the original MLLM visual instruction tuning data for fine-tuning.
- For LLaVA-1.5 and MGM: Re-tune after mixing.
- For InternVL2: Conduct secondary fine-tuning on top of the original tuning.
- Perform an additional 2 epochs of domain-adaptation fine-tuning on Spot-the-Diff and Image-Edit-Request.
- No extra loss function designed; standard VQA training objectives are used.

## Key Experimental Results

### Main Results

#### MMVP Benchmark (Image Difference Recognition)

| Model | Original Score | +Img-Diff | Gain |
|------|---------|----------|------|
| LLaVA-1.5-7B | ~18 | ~32 | +14 |
| MGM-7B | ~27 | ~42 | +15 |
| InternVL2-8B | ~40 | ~45 | +5 |
| GPT-4V | ~30 | - | - |
| Gemini | ~30 | - | - |

#### 8 General MLLM Benchmarks (Average Gain $\Delta$)

| Model | VQAv2 | GQA | POPE | MMBench | MM-Vet | SciQA | SEED | Average $\Delta$ |
|------|-------|-----|------|---------|--------|-------|------|--------|
| LLaVA-1.5-7B | 78.5→79.3 | 62→62.8 | 85.9→86.4 | 64.3→66.1 | 30.5→33.2 | 66.8→68.2 | 58.6→61.7 | **+3.06%** |
| MGM-7B | 80.4→80.7 | 62.6→62.7 | 86→86.2 | 69.3→68.7 | 40.8→44.1 | 70.6→71.7 | 63.5→63.2 | **+1.28%** |
| InternVL2-8B | 81.8→81.8 | 62.6→62.6 | 87.7→88.0 | 82.5→82.7 | 49.2→52.6 | 96.5→96.6 | 69.5→69.9 | **+1.01%** |

#### Spot-the-Diff Benchmark

| Model | BLEU | METEOR | CIDEr-D | ROUGE-L |
|------|------|--------|---------|---------|
| MGM-7B | 9.9 | 12.0 | 46.3 | 31.5 |
| MGM-7B + RP | **10.8** | **13.1** | **53.5** | **33.0** |
| VACC (Prev. SOTA) | 9.7 | 12.6 | 41.5 | 32.1 |

### Key Findings
- **Img-Diff brings higher improvement to weaker models**: LLaVA-1.5-7B gains 3.06%, while InternVL2-8B (which already has plenty of high-quality data) only gains 1.01%, indicating diminishing marginal returns of data.
- **Quality evaluation results**: Human annotation of 1,000 samples reveals that 79.6% of bounding boxes contain valid object differences, 80.1% of region descriptions are accurate, and over 70% of the difference descriptions are completely correct.
- **Considerable diversity**: The dataset covers 1,203 object categories and 3,680 unique "object replacement pairs".
- **Data quantity vs. quality trade-off**: Appendix experiments indicate that reasonably scaling up the data quantity can further boost performance, but quality takes precedence over quantity.
- **"Object removal" data is also effective**: Aside from "object substitution," the "object removal" variant can also bring additional improvements (Appendix Section 16).

## Highlights & Insights
- **Sophisticated design of the entire data synthesis pipeline** — from image pair generation and multi-stage filtering to region-level descriptions, every step incorporates quality-assurance mechanisms.
- **Region-level difference descriptions** (bounding boxes + local descriptions) are much more precise than global descriptions — resolving the limitation where "a single caption cannot encompass all differences."
- **As a general fine-tuning dataset, Img-Diff is "harmless"** — it enhances difference-recognition ability without hurting (and even slightly improving) model performance on general VQA.
- **Outperforming GPT-4V and Gemini** is highly convincing — the ~12-point improvement comes purely from data without altering model architectures.

## Limitations & Future Work
- The data scale is relatively small (12,688 samples); scaling up while maintaining quality remains a challenge.
- Dependency on the generation quality of SDXL — synthetic images may contain artifacts that affect difference detection.
- Difference descriptions rely heavily on LLaVA-NEXT, which itself has limitations in producing fine-grained descriptions.
- Only focuses on "object substitution/removal" differences, lacking a systematic coverage of attribute changes (like color, texture, or pose).
- The filtering pipeline is relatively complex, and end-to-end efficiency needs to be improved.

## Related Work & Insights
- **vs InstructPix2Pix**: Both utilize Prompt-to-Prompt to generate image pairs, but Img-Diff employs the superior SDXL, multi-stage filtering, and region-level descriptions, yielding significantly higher data quality.
- **vs Spot-the-Diff Dataset**: Real-world captures from stationary cameras, where differences represent natural temporal changes. Img-Diff consists of synthetic object changes, offering broader coverage but potentially deviating from clean real-world distributions.
- **vs VIXEN**: The first method using MLLMs for image difference captioning, focusing on model design. Img-Diff focuses on data synthesis, and the two are highly complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of using contrastive learning principles for data synthesis is novel, though the fundamental components (SDXL, Prompt-to-Prompt, FastSAM) are existing tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive evaluation across 3 difference benchmarks, 8 general benchmarks, and 3 MLLMs, alongside data quality/diversity assessment, ablation, and multiple variants.
- Writing Quality: ⭐⭐⭐⭐ The pipeline diagram is clear, and every component is explicitly explained, though the division between the main text and the appendix is somewhat heavy.
- Value: ⭐⭐⭐⭐ Provides a generalizable data augmentation strategy of direct value to improving the fine-grained visual capabilities of MLLMs, with open-sourced code and data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ECCV 2024\] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](../../ECCV2024/multimodal_vlm/freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)

</div>

<!-- RELATED:END -->
