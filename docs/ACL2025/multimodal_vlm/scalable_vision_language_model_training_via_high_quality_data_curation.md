---
title: >-
  [Paper Note] Scalable Vision Language Model Training via High Quality Data Curation
description: >-
  [ACL 2025][Multimodal VLM][Vision-Language Model] This work proposes the SAIL-VL family of open-source vision-language models (2B/8B). The core contributions lie in constructing the highest-quality SAIL-Caption dataset of 300 million images, being the first to reveal the log-linear scaling law of data volume in VLM pre-training (validated with 655B token experiments), and shifting the scaling curve from log to near-linear through a three-stage curriculum SFT…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Vision-Language Model"
  - "Data Quality"
  - "Data Scaling Law"
  - "Curriculum Learning"
  - "Pre-training Scaling"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: bd98abdc932ff7c0
---

# Scalable Vision Language Model Training via High Quality Data Curation

**Conference**: ACL 2025  
**arXiv**: [2501.05952](https://arxiv.org/abs/2501.05952)  
**Authors**: Hongyuan Dong*, Zijian Kang*, Weijie Yin*, Xiao Liang, Chao Feng, Jiao Ran (ByteDance Douyin)  
**Code**: [HuggingFace](https://huggingface.co/BytedanceDouyinContent)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Model, Data Quality, Data Scaling Law, Curriculum Learning, Pre-training Scaling, Instruction Tuning  

## TL;DR

This work proposes the SAIL-VL family of open-source vision-language models (2B/8B). The core contributions lie in constructing the highest-quality SAIL-Caption dataset of 300 million images, being the first to reveal the log-linear scaling law of data volume in VLM pre-training (validated with 655B token experiments), and shifting the scaling curve from log to near-linear through a three-stage curriculum SFT, achieving SOTA performance on 18 benchmarks.

## Background & Motivation

### Background
The fundamental reasons for the subpar performance of lightweight VLMs lie in two aspects: (1) insufficient visual understanding capability during the pre-training stage—either due to a limited training budget (e.g., the LLaVA series only undergoes lightweight pre-training), or low data quality (e.g., although MiniCPM-V and Qwen2-VL allocate tens of billions of tokens for pre-training, their improvement is held back by low-quality data); (2) a lack of systematic data quality evaluation and phased training methodologies in the SFT stage.

### Limitations of Prior Work
- The LLaVA series relies on small amounts of low-quality caption data for lightweight pre-training, which limits its visual understanding capabilities.
- Although MiniCPM-V-2.5 and Qwen2-VL allocate dozens of billions of tokens for pre-training, the lack of data quality prevents them from fully exploiting the potential of pre-training.
- Prior works rarely offer reliable conclusions on how pre-training budgets and data quality affect VLM performance.
- There is a lack of recognized methodology for distribution adjustment of SFT datasets and training phase partitioning.
- Although Infinity-MM explores multi-stage SFT, it lacks a widely acknowledged systemized framework.

### Design Motivation
To drive model performance through data engineering—achieving "high quality + high scalability" in both the pre-training and SFT stages, and quantitatively revealing the scaling relationship between data volume and model performance.

## Method

### Overall Architecture
SAIL-VL employs InternViT-300M as the visual encoder and the Qwen2.5 series as the language model, progressively building capabilities through five training stages:
1. **Pretrain-Alignment** (131B tokens): Trains only the MLP projection layer, aligning the vision-language space using SAIL-Caption and OCR data.
2. **Pretrain-Advance** (524B tokens): Unlocks the visual encoder for joint training, further enhancing visual understanding.
3. **SFT-Knowledge** (21M samples): Learns basic instruction-following and world knowledge.
4. **SFT-Instruction** (12M samples): Strengthens visual instruction-following capabilities.
5. **SFT-Preference** (3.5M samples): Handles complex reasoning tasks using a small volume of highly complex data.

### Key Design 1: Scalable High-Quality Data Curation Pipeline (SAIL-Caption)

The data construction is divided into four steps:
- **Data Collection**: Gather data extensively from public datasets such as LAION-COCO, TextCaps, and SA1B to ensure distributional diversity.
- **Reference Data Annotation**: Select a balanced subset and annotate high-quality detailed descriptions using GPT-4O, supplemented by alt-text to provide world knowledge.
- **Annotator Model Training**: Fine-tune InternVL2-8B with the reference data to obtain SAIL-Captioner, which possesses both captioning and recaptioning capabilities.
- **Large-Scale Production**: Deploy SAIL-Captioner via LMDeploy to achieve multi-task, multi-node, and multi-process asynchronous annotation.

The final output is SAIL-Caption: a detailed description dataset containing 300 million images. Human evaluation scores its quality as 87.2/88.2 (converted to a 10-point scale), significantly higher than DataComp-LLaVA-Caption (70.0) and BLIP3-KALE (73.2). It also achieves absolute superiority in linguistic diversity metrics (number of unique n-grams, nouns, verbs, and adjectives).

### Key Design 2: Data Volume Logarithmic Scaling Law

In the pre-training stage, as the data volume increases exponentially from billions of tokens to 655B tokens, model performance exhibits a clear log-linear scaling relationship:
- Pretrain-Alignment stage: Performance on caption and OCR tasks improves steadily, appearing approximately linear when the x-axis is on a logarithmic scale.
- Pretrain-Advance stage: Performance jumps significantly after unlocking the visual encoder, maintaining a similar logarithmic scaling curve.
- Post-SFT performance also maintains this scaling relationship with data volume—regardless of using proprietary SFT data or LLaVA-Next SFT data, a larger pre-training data volume consistently leads to better final post-SFT performance.

This is the first work to explicitly propose and validate the data volume scaling law in VLM pre-training.

### Key Design 3: Curriculum SFT and Data Complexity Scaling

**Three-Tier Data Quality Evaluation System**:
- **Quick Quality Evaluation**: Uses a 2M subset to quickly train and evaluate dataset quality, based on the assumption that performance rankings of different datasets remain consistent across different data scales (empirically verified).
- **Composition Evaluation**: Groups SFT data by format (closed-form VQA, open-ended VQA, document VQA, etc.) and tests them by halving each group to optimize proportions.
- **Incremental Evaluation**: Adds and evaluates new datasets one by one, retaining those that improve or maintain performance.

**Curriculum Learning Strategy**: The task difficulty ($1.90 \rightarrow 2.15 \rightarrow 2.20$), data complexity ($2.44 \rightarrow 2.62 \rightarrow 2.74$), and image-text relevance ($3.94 \rightarrow 4.45 \rightarrow 4.55$) of the three SFT stages increase progressively. Experiments demonstrate that the scaling curve of curriculum SFT is approximately linear, which is significantly better than the logarithmic scaling curve obtained by mixing all data together for training.

## Key Experimental Results

### Table 1: Comparison between SAIL-VL and Same-Scale SOTA Models on 18 Benchmarks

| Model | Scale | Average | General VQA | OCR VQA | Math & Knowledge | Hallucination |
|------|------|--------|-------------|---------|----------------|---------------|
| **SAIL-VL** | **2B** | **69.1** | 60.4 | **75.9** | **79.0** | **66.2** |
| InternVL2.5-MPO | 2B | 67.7 | **63.1** | 71.1 | 75.3 | 64.5 |
| Qwen2-VL | 2B | 64.4 | 58.3 | 72.5 | 59.0 | 62.9 |
| DeepSeekVL-2 | 2B | 67.0 | 59.4 | 74.4 | 71.3 | 63.6 |
| **SAIL-VL** | **8B** | **74.5** | 68.3 | **79.8** | **83.3** | 68.7 |
| InternVL2.5-MPO | 8B | 74.3 | **71.2** | 76.3 | 83.2 | **69.7** |
| Qwen2-VL | 8B | 73.0 | 68.5 | 79.6 | 71.0 | 67.5 |
| DeepSeekVL-2 | 8B | 72.7 | 66.8 | 79.0 | 79.0 | 65.3 |

SAIL-VL-2B outperforms the runner-up InternVL2.5-MPO-2B by 1.4 points (+2.06%) on the overall average, showing prominent advantages in the OCR and mathematical knowledge subdomains. The 8B model also leads, though the margin of advantage narrows.

### Table 2: Impact of Pre-training Data Quality on Model Performance

| Caption Data | OCR Data | Overall | Caption | OCR |
|------------|---------|---------|---------|-----|
| **SAIL-Caption** | High Quality | **54.36** | **51.80** | 55.38 |
| SA1B-QwenVL-Caption | High Quality | 48.43 | 39.57 | 51.97 |
| DataComp-LLaVA-Caption | High Quality | 49.08 | 42.70 | 51.63 |
| BLIP3-KALE | High Quality | 53.06 | 46.00 | **55.89** |
| SAIL-Caption | High + Low Quality | 52.13 | 51.22 | 52.50 |
| SAIL-Caption | High Quality (Repeated) | 54.05 | 52.63 | 54.62 |

SAIL-Caption significantly outperforms other open-source datasets in visual understanding performance. Notably, training with repeated high-quality OCR data yields better results than mixing in diverse but low-quality data—under the pre-training setup with a frozen LLM, repeating high-quality data does not lead to overfitting.

### Table 3: Evaluation of Training on 2M Subsets of Data from Various SFT Stages

| Training Data | Overall | General | OCR | Math | Hallucination |
|---------|---------|---------|-----|------|---------------|
| SFT-Knowledge | 57.8 | 53.2 | 60.9 | 56.9 | 63.9 |
| **SFT-Instruction** | **61.9** | 55.8 | **67.1** | **65.4** | 61.7 |
| SFT-Preference | 61.3 | **57.1** | 65.8 | 59.5 | 61.3 |

The SFT-Instruction dataset achieves the highest quality, while training on SFT-Preference alone leads to a slight decline in performance due to its excessive complexity—this validates the necessity of curriculum learning.

## Key Findings

- **Data Quality > Data Diversity**: In pre-training, repeating high-quality data yields better results than mixing in low-quality data. In SFT, the meticulously filtered 12M data outperforms datasets of several tens of times its size from other open-source projects.
- **2B Models Benefit from Data Scaling**: The 655B token pre-training experiments demonstrate that even compact models can continuously obtain performance gains from larger pre-training datasets, breaking the misconception that "small models do not require large-scale pre-training."
- **Curriculum SFT Shifts the Scaling Curve from Log to Linear**: A phased, easy-to-hard training strategy yields superior performance compared to one-shot mixed training under equivalent compute budgets.
- **Effectiveness of Quick Quality Evaluation**: The dataset quality ranking evaluated using the 2M subset is perfectly consistent with full-scale training, providing a practical method for efficient data filtering.

## Highlights & Insights

- **First Formulation of the Data Scaling Law for VLM Pre-training**: This discovery provides quantitative guidance for pre-training resource allocation and holds substantial engineering value.
- **Comprehensive Data Engineering Methodology**: Formulates a complete loop from data construction (SAIL-Caption pipeline) and data evaluation (three-tier evaluation system) to data utilization (curriculum SFT).
- **Replicable Data Construction Pipeline**: First annotating reference data with GPT-4O, and then distilling into InternVL2-8B for large-scale annotation—this paradigm of "strong model annotation $\rightarrow$ weak model distillation $\rightarrow$ large-scale production" is highly generalizable.
- **Excessively Complex Data Can Be Harmful**: Training independently on SFT-Preference yields lower performance than SFT-Instruction, revealing the crucial principle that training data complexity must match the model's current capacity.

## Limitations & Future Work

- **Scaling Law Validated Only at Specific Data Scales**: Although performance saturation was observed, it remains uncertain whether there is still room for improvement under better training configurations.
- **The Margin of Advantage for the 8B Model Narrows Significantly**: The authors admit that the training data volume for the 8B model is relatively small (only 52B pre-training tokens vs. 655B tokens for the 2B model), failing to fully exploit the potential of the larger model.
- **Key Models of Larger Scales Unexplored**: All conclusions are based on 2B/8B models; whether the scaling law holds at larger scales remains to be verified.
- **Suboptimal Performance in the General VQA Subdomain**: Performance on benchmarks requiring long-text generation such as MMVet is unstable.
- **Incomplete Release of the SAIL-Caption Dataset**: Although the models are open-sourced, the full 300M dataset is only partially released.
- **Potential for Hallucinations and Bias**: The authors acknowledge that the model may still produce incorrect information under certain scenarios.

## Related Work & Insights

- **Differences from the LLaVA Series**: While LLaVA adopts a route of lightweight pre-training combined with meticulous SFT, SAIL-VL demonstrates the irreplaceable importance of large-scale, high-quality pre-training.
- **Comparison with InternVL2.5-MPO**: MPO requires an additional reinforcement learning stage, whereas SAIL-VL achieves comparable or even superior performance relying solely on data quality and curriculum SFT.
- **Relationship with Infinity-MM**: SAIL-VL directly utilizes Stage 2 and Stage 4 data from Infinity-MM as part of the SFT data, but introduces a more systematic data quality evaluation framework on a methodological level.
- **Inspiration for Data Flywheels**: The construction process of SAIL-Captioner is inherently a data flywheel: high-quality reference data $\rightarrow$ training the annotator $\rightarrow$ large-scale production $\rightarrow$ better models $\rightarrow$ higher-quality data.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically reveals the VLM pre-training data volume scaling law for the first time; the curriculum SFT strategy is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 655B token pre-training, evaluation on 18 benchmarks, and multiple sets of ablation studies; represents an immense engineering effort.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, extensive tables and figures, and a systematic elaboration of the methodology.
- Value: ⭐⭐⭐⭐ — Provides practical guidance on data engineering for VLM training; the scaling law and curriculum SFT strategy are of broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](../../CVPR2025/multimodal_vlm/active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)

</div>

<!-- RELATED:END -->
