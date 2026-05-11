---
title: >-
  [Paper Note] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
description: >-
  [NeurIPS 2025][Remote Sensing][change captioning] This work introduces RSCC — the first large-scale disaster-aware remote sensing change captioning dataset comprising 62…
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "change captioning"
  - "disaster monitoring"
  - "bi-temporal"
  - "vision-language model"
date: 2026-05-08
content_hash: 773c3d3180a4ab9a
---

# RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events

**Conference**: NeurIPS 2025
**arXiv**: [2509.01907](https://arxiv.org/abs/2509.01907)
**Code**: [https://github.com/Bili-Sakura/RSCC](https://github.com/Bili-Sakura/RSCC)
**Area**: Remote Sensing / Vision-Language
**Keywords**: remote sensing, change captioning, disaster monitoring, bi-temporal, vision-language model

## TL;DR
This work introduces RSCC — the first large-scale disaster-aware remote sensing change captioning dataset comprising 62,351 pre/post-disaster image pairs with detailed change descriptions, covering 31 global disaster events including earthquakes, floods, and wildfires. High-quality annotations are generated using the QvQ-Max visual reasoning model, and a comprehensive benchmark evaluation framework is established.

## Background & Motivation

**Background**: Bi-temporal remote sensing imagery is critical for disaster monitoring, requiring detailed textual description and analysis of pre- and post-disaster changes. Multimodal large language models (MLLMs) have achieved significant advances in natural image understanding, yet remain underexplored for bi-temporal remote sensing image understanding. **Limitations of Prior Work**: Existing remote sensing image-text datasets suffer from three major shortcomings — either they are single-temporal snapshots lacking temporal information (UCM-Captions, RSICD), or they have temporal coverage but no textual annotations (fMoW, SpaceNet 7), or they are small-scale with brief descriptions lacking disaster context (LEVIR-CC: 20K pairs/40 words; Dubai-CCD: 1K pairs/35 words). **Key Challenge**: There is no dataset that simultaneously provides large scale, disaster specificity, bi-temporal coverage, and high-quality long-form textual descriptions for training and evaluating vision-language models. **Key Insight**: The paper leverages image and annotation data from two building damage assessment datasets — xBD and EBD — combined with the QvQ-Max visual reasoning model to automatically generate detailed change description texts.

## Method

### Overall Architecture
The RSCC construction pipeline consists of four steps: (1) Pre/post-disaster image pairs and building damage annotations are acquired from two building damage assessment datasets, xBD (MAXAR OpenData) and EBD; xBD images are cropped from $1024\times1024$ to $512\times512$ without overlap, while EBD retains the original $512\times512$ resolution. (2) Building damage labels (four levels based on the Joint Damage Scale: no damage/minor/major/destroyed) are converted into structured auxiliary information and overlaid on post-disaster images as visual prompts using color-coded bounding boxes. (3) A structured text prompt comprising \<task instruction\>\<disaster description\>\<building damage details\>\<output format\> is constructed, and the QvQ-Max (qvq-max-2025-03-25) API is called to generate change description texts. (4) Post-processing is performed via Qwen2.5-Max for automatic metadata inconsistency correction, supplemented by expert validation through 10% random sampling with a three-person review to ensure annotation quality. The final dataset is split into a training set of 61,363 pairs (31 events) and a test set of 988 pairs (19 events).

### Key Designs

1. **Annotation Generation via Visual Reasoning**:

    - Function: Automatically generate high-quality change descriptions for 62,351 image pairs.
    - Mechanism: QvQ-Max (Alibaba's visual reasoning model) takes as input the original pre-disaster image, the post-disaster image annotated with color-coded building damage bounding boxes, and structured text instructions. The key innovation is incorporating building damage assessment labels (Joint Damage Scale) as in-context auxiliary information, with damage levels encoded as visual prompts through differently colored bounding boxes.
    - Design Motivation: QvQ-Max's structured reasoning capability enables inference of spatiotemporal relationships, whereas conventional MLLMs tend toward recognition-style outputs; the visual prompt engineering approach is inspired by the marking-based method of Shtedritski et al.

2. **Two-Stage Post-Processing and Quality Control**:

    - Function: Ensure reliability and consistency of generated annotations.
    - Mechanism: In the first stage, Qwen2.5-Max automatically corrects metadata inconsistencies (disaster type, damage descriptions), improving disaster type consistency from 93.2% to 100%. In the second stage, 10% of samples are randomly drawn and reviewed by three experts using binary scoring across four dimensions (disaster type accuracy, damage detail completeness, factual consistency, and clarity), achieving a 100% pass rate.
    - Design Motivation: LLM-generated content may be inconsistent with metadata; a dual-layer guarantee of automated and human verification is therefore necessary.

3. **Training a Dedicated Change Captioning Model**:

    - Function: Validate the effectiveness of the RSCC dataset for VLM training.
    - Mechanism: Full-parameter fine-tuning is performed on Qwen2.5-VL 7B using the 61,363-pair training set, with batch size 1, trained for 2 epochs on 2×H800 GPUs (40 GPU hours total). The LLM backbone uses a learning rate of 1e-6, the visual encoder uses 1e-5, with cosine decay.
    - Design Motivation: Demonstrate that the RSCC dataset can effectively improve the capability of general-purpose MLLMs in remote sensing bi-temporal understanding.

### Loss & Training
Standard autoregressive language modeling loss. The original RSCC resolution of $512\times512$ is preserved as model input.

## Key Experimental Results

### Main Results

| Model | ROUGE(%)↑ | METEOR(%)↑ | ST5-SCS(%)↑ | Avg_L |
|-------|-----------|------------|-------------|-------|
| **Ours (7B)** | **14.99** | **16.05** | **58.52** | 44 |
| InternVL 3 (8B) | 12.76 | 15.77 | 51.84 | 64 |
| TEOChat (7B) | 7.86 | 5.77 | 52.64 | 15 |
| Kimi-VL (3B) | 12.47 | 16.95 | 51.35 | 87 |
| CCExpert (7B) | 7.61 | 4.32 | 40.81 | 12 |
| Qwen2-VL (7B) | 11.02 | 9.95 | 45.55 | 42 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Zero-shot vs. text prompt vs. visual prompt | Visual prompt yields significant improvement | Building damage information as auxiliary input enhances performance |
| Different model scales (3B→72B) | Increases with scale but non-linearly | Kimi-VL 3B exceeds expectations (51.35% ST5-SCS) |
| Calibration decoding (VCD/DoLa/DeCo) | No notable improvement | RS change captioning requires complex visual reasoning rather than simple hallucination reduction |
| Human preference study | QvQ-Max win rate 80.7–99.0% | Significantly outperforms all baselines |

### Key Findings
- The 7B model fine-tuned on RSCC (ST5-SCS 58.52%) substantially outperforms both general-purpose models (Qwen2-VL: 45.55%) and task-specific models (CCExpert: 40.81%).
- Task-specific remote sensing models CCExpert and TEOChat perform poorly on long-form change captioning — their outputs are excessively short (Avg_L of only 12 and 15 words, respectively), indicating insufficient spatiotemporal reasoning and long-text generation capabilities in existing specialized models.
- BLIP-3 and LLaVA-OneVision exhibit severe repetitive generation (Avg_L reaching 456 and 221 words, respectively), pointing to deficiencies in decoding strategies.
- Training-free calibration decoding strategies (VCD/DoLa/DeCo) have limited effectiveness on remote sensing change captioning; the authors attribute this to the requirement for complex visual reasoning in the task, as opposed to simple object-level hallucination reduction.
- Evaluation employs Sentence T5-XXL Embedding with Sharpened Cosine Similarity ($q=0$, $p=3$) as a semantic similarity metric, which is more suitable for long-text evaluation than traditional n-gram metrics.
- Model scale is not the sole determining factor: Kimi-VL (3B) surpasses Qwen2-VL (7B, 45.55%) and LLaVA-OneVision (8B, 46.15%) on ST5-SCS (51.35%).

## Highlights & Insights
- This work fills the gap in disaster remote sensing + bi-temporal + long-form description datasets; its scale (62K pairs) far exceeds existing comparable datasets (LEVIR-CC 20K, Dubai-CCD 1K).
- The visual prompt engineering approach is novel: color-coded bounding boxes encode damage levels (Joint Damage Scale) as visual prompts for VLMs, significantly improving description quality.
- Dataset construction cost is manageable (approximately \$5 per 1,000 pairs), demonstrating scalability and providing a viable paradigm for large-scale remote sensing annotation.
- Human preference evaluation results (80–99% win rates) provide strong evidence for the quality of QvQ-Max-generated annotations.
- This is the first systematic study of calibration decoding strategies (VCD/DoLa/DeCo) in remote sensing scenarios, finding that training-free correction decoding has limited effect on tasks requiring complex visual reasoning.
- The dataset covers 31 global disaster events across 6 major disaster types (earthquake/flood/hurricane/tornado/volcano/wildfire), offering high geographic diversity.

## Limitations & Future Work
- Generated descriptions may contain vague depictions; even domain experts may have difficulty confirming details for certain complex change scenarios.
- Evaluation metrics are limited to text similarity measures (ROUGE/METEOR/ST5-SCS); dedicated multi-image description evaluation metrics are lacking, as existing image captioning metrics (FLEUR/SPARC/G-VEval) support only single images.
- VLMs still significantly underperform specialized CV models on vision-centric tasks such as change detection and multi-label classification.
- The distribution of disaster types is imbalanced (dominated by hurricanes and floods), which may affect model generalization to less represented disaster categories.
- The EBD dataset lacks manual building damage annotations, necessitating the use of a simplified naive prompt for description generation.

## Related Work & Insights
- **LEVIR-CC** (20K pairs/40 words) / **Dubai-CCD** (1K pairs/35 words): Previously the largest remote sensing change captioning datasets; their scale and description quality are far inferior to RSCC (62K pairs/72 words).
- **CCExpert**: A task-specific model based on LLaVA-OneVision that introduces a difference-focused integration component, but exhibits insufficient long-text generation capability (ST5-SCS of only 40.81%).
- **TEOChat**: Enhances LLaVA-1.5's temporal understanding via a shared visual encoder, but produces excessively short outputs (Avg_L = 15).
- **WHU-CDC** (14.8K pairs): Also provides change captioning annotations but lacks disaster-specific context.
- **Diffusion-RSCC**: Applies a probabilistic diffusion model to RSICC, focusing on pixel-level differences.
- Insights: The paradigm of using powerful commercial VLMs (QvQ-Max at ~\$5/K pairs) to automatically generate high-quality annotations is generalizable and can be extended to other remote sensing understanding tasks; visual prompt engineering (color-coded bounding boxes encoding semantic information) is an effective means of improving VLM understanding of remote sensing data. The xBD and EBD datasets originate from the MAXAR OpenData Program, ensuring data accessibility and reproducibility.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale disaster-aware remote sensing change captioning dataset; clever visual prompt design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10+ baseline models, includes human preference evaluation and analysis of multiple enhancement strategies.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is clearly described with detailed statistical documentation.
- Value: ⭐⭐⭐⭐⭐ Significant dataset contribution; code and data are open-sourced, with important implications for the remote sensing vision-language community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](../../CVPR2026/remote_sensing/olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)

</div>

<!-- RELATED:END -->
