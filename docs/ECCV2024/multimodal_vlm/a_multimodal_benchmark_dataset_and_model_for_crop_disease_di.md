---
title: >-
  [Paper Note] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis
description: >-
  [ECCV 2024][Multimodal VLM][Crop disease diagnosis] This work constructs the CDDM dataset containing 137k crop disease images and 1 million question-answering pairs, and proposes a strategy to apply LoRA fine-tuning simultaneously to the vision encoder, adapter, and language model. This enables Qwen-VL-Chat and LLaVA to leap from single-digit accuracy to over $90\%$ in crop disease diagnosis.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Crop disease diagnosis"
  - "multimodal dataset"
  - "vision-language models"
  - "LoRA fine-tuning"
  - "agricultural AI"
date: 2026-05-08
content_hash: dd5980c6646e7125
---

# A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis

**Conference**: ECCV 2024  
**arXiv**: [2503.06973](https://arxiv.org/abs/2503.06973)  
**Code**: [https://github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench)  
**Area**: Object Detection  
**Keywords**: Crop disease diagnosis, multimodal dataset, vision-language models, LoRA fine-tuning, agricultural AI  

## TL;DR
This work constructs the CDDM dataset containing 137k crop disease images and 1 million question-answering pairs, and proposes a strategy to apply LoRA fine-tuning simultaneously to the vision encoder, adapter, and language model. This enables Qwen-VL-Chat and LLaVA to leap from single-digit accuracy to over $90\%$ in crop disease diagnosis.

## Background & Motivation
Current crop disease diagnosis mainly relies on unimodal methods (classification, detection), which only output diagnosis results without providing rich agricultural knowledge such as prevention and control suggestions. Although general large vision-language models (e.g., LLaVA, Qwen-VL) perform excellently in general scenarios, they perform poorly in the agricultural domain—Qwen-VL-Chat achieves only $28.4\%$ accuracy in crop species identification and a mere $5.0\%$ in disease classification. There are two core reasons: (1) Lack of specialized agricultural multimodal training data; (2) The visual differences between different crop diseases are extremely subtle (leaf shapes and colors are highly similar, and spots of different diseases look almost identical), making it difficult for general vision encoders to capture these fine-grained differences.

## Core Problem
How can general LVLMs be equipped with professional crop disease diagnosis capabilities? This is specifically decomposed into two sub-problems:
1. **Data Gap**: The agricultural domain lacks large-scale, high-quality multimodal instruction datasets to fine-tune LVLMs.
2. **Rigid Vision Encoders**: Existing fine-tuning strategies freeze the vision encoder, preventing the model from distinguishing visually highly similar diseases.

## Method

### Overall Architecture
Using Qwen-VL-Chat as the base model, it includes three components: a vision encoder (ViT), a position-aware vision-language adapter (cross-attention), and a language model. The input consists of a crop disease image and a natural language question, and the output is a natural language answer covering disease identification, cause analysis, and prevention/control strategies. The core modification is to apply LoRA to all three components simultaneously during the fine-tuning stage, rather than freezing the vision encoder.

### Key Designs

1. **CDDM Dataset Construction**:

    - **Image Data**: 62K open-source images collected from Kaggle and web crawling + 75K private images collected from actual fields, totaling 137k images covering 16 crop types and 60 disease categories. Agricultural experts annotated crop categories, disease categories, and visual appearance descriptions. The data distribution is relatively balanced, with 48 categories having over 500 images.
    - **Disease Diagnosis Instruction Data**: GPT-4 was utilized to generate few-shot question-answering pairs based on {crop category, disease category, appearance description}, with 8 rounds of Q&A per image. The novelty lies in the deliberate inclusion of questions requiring negative answers, as LVLMs were found to have a bias toward affirmative responses. Ultimately, over 1 million QA pairs were generated, with an average question length of 6.11 words and an answer length of 8.92 words.
    - **Disease Knowledge Instruction Data**: Based on collected agricultural disease knowledge texts (symptoms, pathogens, transmission pathways, and control methods), conversational QA pairs were also generated using GPT-4. The average answer length is longer (130.41 words), providing in-depth agricultural knowledge.

2. **Full-Component LoRA Fine-Tuning Strategy**:

    - Unlike the standard practice of LLaVA/Qwen-VL (freezing the vision encoder and only tuning the adapter and LLM), this work also applies LoRA to the vision encoder.
    - Motivation: Differences between crop disease images are extremely subtle; frozen general vision encoders cannot capture local details and patterns that differentiate various diseases.
    - By using LoRA, the vision encoder is adapted to the fine-grained visual features of the agricultural domain in a parameter-efficient manner.

3. **Negative Answer Data Augmentation**:

    - To address the bias of LVLMs toward affirmative answers, questions requiring negative answers (e.g., "Is this $XX$ disease?" $\to$ "No") were deliberately designed in QA generation to enhance the model's performance on discriminative tasks.

### Loss & Training
- Based on Qwen-VL-Chat: batch size 128, lr $1 \times 10^{-5}$, epochs 5, max seq len 2048, weight decay 0.1
- Based on LLaVA-v1.5-7B: batch size 128, lr $2 \times 10^{-4}$, epochs 5, max seq len 2048, weight decay 0
- Standard autoregressive language modeling loss is used.

## Key Experimental Results

| Model | Crop Classification | Disease Classification | Knowledge QA |
|------|---------|---------|---------|
| Qwen-VL-Chat (Original) | 28.4% | 5.0% | 41 |
| Qwen-VL-Chat-AG (Frozen VE) | 84.4% | 66.1% | 88.5 |
| Qwen-VL-Chat-AG (All LoRA) | **97.4%** | **91.5%** | 84 |
| LLaVA-v1.5-7b (Original) | 24.5% | 5.9% | 47.5 |
| LLaVA-AG (Frozen VE) | 94.3% | 82.1% | **98** |
| LLaVA-AG (All LoRA) | **98.0%** | **91.8%** | 96.5 |

> Test set: 3000 out-of-distribution images; Knowledge QA scale is 100 (GPT-4 score normalized).

### Ablation Study
- **Vision Encoder Fine-Tuning is Crucial**: Comparing unfrozen vs. frozen vision encoders, the disease classification accuracy increases by about 25 percentage points (Qwen: 66.1% $\to$ 91.5%, LLaVA: 82.1% $\to$ 91.8%), confirming the necessity of adapting fine-grained visual features.
- **Interesting Trade-off in Knowledge QA**: Qwen-VL-Chat shows a slight drop in Knowledge QA scores after unfreezing the vision encoder (88.5 $\to$ 84), which might be because visual encoder adjustments compromised semantic alignment.
- **Core Value of the Dataset**: Regardless of the fine-tuning strategy, model performance achieves a qualitative leap after using the CDDM dataset (crop classification rises from ~25% to 85%+).

## Highlights & Insights
- **Dataset Scale and Quality**: 137k images and 1 million QAs, covering 16 crop types and 60 disease categories, representing a pioneering resource in the agricultural multimodal domain.
- **Negative Sample Design**: Introducing negative answer QAs to address the affirmation bias of LVLMs. This insight is highly generalizable and can be transferred to other vertical-domain VLM adaptations.
- **Simple yet Effective Strategy**: Simply altering the fine-tuning strategy (unfreezing the vision encoder with LoRA) yields massive improvements, highlighting that choosing "which parameters to tune" is critical in vertical-domain adaptation.
- **High Practical Value**: Directly addresses the practical needs of farmers and agricultural practitioners, providing a one-stop solution from diagnosis to prevention/control.

## Limitations & Future Work
- **Poor Out-of-Domain Generalization**: The authors explicitly point out that the fine-tuned model performs poorly on out-of-distribution diseases not in the training set, indicating that generalization remains limited by data coverage.
- **Trade-off between Knowledge QA and Diagnosis**: Full-component LoRA causes a slight decline in Knowledge QA performance on Qwen, and the paper lacks in-depth analysis or mitigation solutions for this phenomenon.
- **Coarse Evaluation Metric**: Disease diagnosis is evaluated only by checking if the answer contains correct keywords, neglecting completeness and accuracy of the responses.
- **Bias Towards Chinese Crops**: Over 70% of the data consists of field-collected images, which may lead to geographic and climatic bias.
- **Lack of Comparison with More Base Models**: Only two 7B models, Qwen-VL and LLaVA, were evaluated.
- **Future Directions**: Exploring in-context learning to handle out-of-distribution diseases, employing more fine-grained evaluation metrics (e.g., BERTScore), and integrating RAG to introduce dynamic disease knowledge bases.

## Related Work & Insights
- **vs. Lan et al. (2023)**: Previous agricultural VQA works only supported a limited number of disease categories with restricted architectures (e.g., ResNet+BERT), outputting simple answers. In contrast, this work covers 60 disease categories and provides detailed control suggestions based on LVLMs.
- **vs. LLaVA-Med**: LLaVA-Med performed similar vertical-domain adaptation in the medical field but froze the vision encoder. This work demonstrates that unfreezing the vision encoder is necessary for vertical domains with substantial fine-grained visual differences.
- **vs. General LVLMs**: General models like Qwen-VL and LLaVA perform poorly in the agricultural domain, verifying that "general $\neq$ specialized" and highlighting the necessity of dedicated data and adaptation strategies for vertical fields.

## Related Work & Insights
- **Vertical-Domain VLM Adaptation Paradigm**: The workflow of dataset construction (GPT-4 generated instruction data) + full-component LoRA fine-tuning can be generalized to other vertical domains (e.g., healthcare, industrial inspection).
- **Should the Vision Encoder Be Frozen?** This question is particularly crucial in fine-grained tasks. This paper provides empirical evidence that it "should not be frozen," offering valuable references for fields like medical VLMs.
- **Negative Sample Strategy**: The affirmation bias of LVLMs is a common issue; the negative QA design of this work can be applied to any VLM fine-tuning scenario requiring discriminative capabilities.

## Rating
- Novelty: ⭐⭐⭐ Limited methodological innovation (LoRA + unfreezing the vision encoder), with the core contribution being the dataset.
- Experimental Thoroughness: ⭐⭐⭐ Only two base models evaluated, metrics are relatively coarse, and in-depth ablation studies are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed description of dataset construction.
- Value: ⭐⭐⭐⭐ An important foundational resource for agricultural AI, and the insights on full-component LoRA fine-tuning have general reference value for other vertical-domain VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](../../ACL2025/multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](../../NeurIPS2025/multimodal_vlm/multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)
- [\[ECCV 2024\] Dataset Growth (InfoGrowth)](dataset_growth.md)
- [\[CVPR 2026\] MeteorPred: A Meteorological Multimodal Large Model and Dataset for Severe Weather Event Prediction](../../CVPR2026/multimodal_vlm/meteorpred_a_meteorological_multimodal_large_model_and_dataset_for_severe_weathe.md)

</div>

<!-- RELATED:END -->
