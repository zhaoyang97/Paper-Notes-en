---
title: >-
  [Paper Note] A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection
description: >-
  [NeurIPS 2025 (Datasets and Benchmarks Track)][Multimodal VLM][greenwashing detection] This work introduces the first multimodal framing analysis benchmark for oil and gas (O&G) industry video advertisements…
tags:
  - "NeurIPS 2025 (Datasets and Benchmarks Track)"
  - "Multimodal VLM"
  - "greenwashing detection"
  - "framing analysis"
  - "video advertising"
  - "vision-language models"
  - "oil and gas industry"
date: 2026-05-08
content_hash: 28650e81df6ac234
---

# A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection

**Conference**: NeurIPS 2025 (Datasets and Benchmarks Track)  
**arXiv**: [2510.21679](https://arxiv.org/abs/2510.21679)  
**Code**: [GitHub](https://github.com/climate-nlp/multimodal-oil-gas-benchmark) / [HuggingFace](https://huggingface.co/datasets/climate-nlp/multimodal-oil-gas-benchmark)  
**Area**: Multimodal VLM / Computational Social Science  
**Keywords**: greenwashing detection, framing analysis, video advertising, vision-language models, oil and gas industry

## TL;DR

This work introduces the first multimodal framing analysis benchmark for oil and gas (O&G) industry video advertisements, comprising 706 videos, 13 framing categories, 50+ entities, and 20 countries. It systematically evaluates six VLMs on greenwashing-related framing detection, finding that GPT-4.1 achieves 79% F1 zero-shot on environmental labels but only 46% on green innovation, thereby exposing implicit framing analysis and cultural context understanding as core challenges for current VLMs.

## Background & Motivation

**Background**: Oil and gas (O&G) companies craft carefully designed public relations campaigns to shape brand image and are frequently accused of greenwashing—projecting a climate-friendly appearance. Framing analysis is an important tool for understanding corporate strategic communication; "framing is the selective presentation of certain aspects of reality."

**Limitations of Prior Work**: Existing greenwashing detection benchmarks focus exclusively on the textual modality (e.g., framing analysis of advertising copy), completely ignoring visual information. Qualitative research has shown, however, that visual strategies in video advertisements are a critical vehicle for greenwashing—images of wind turbines, smiling workers, and similar visual symbols convey implicit "green" impressions. Approximately 30% of advertising videos contain no speech at all and rely purely on visuals to communicate.

**Key Challenge**: Framing in video is often implicit—rather than stating "we are environmentally friendly" in text, companies imply it through footage of solar panels. VLMs must simultaneously interpret visual symbols, cultural context, and corporate strategy to classify frames accurately, yet no benchmark exists to assess this capability.

**Goal**: (1) Construct the first multimodal (video + text + transcript) framing analysis benchmark for O&G advertising; (2) cover the distinct advertising strategies of Facebook and YouTube; (3) systematically evaluate the capabilities and bottlenecks of current VLMs on this task.

**Key Insight**: Data are collected from two complementary platforms—Facebook political advertising versus YouTube corporate brand promotion—with fine-grained framing taxonomies defined for each, and an entity-aware 1-shot prompting strategy designed to improve VLM performance.

**Core Idea**: Construct the first multimodal O&G advertising framing benchmark and quantitatively expose the systematic deficiencies of VLMs in implicit framing and cultural understanding for video greenwashing detection.

## Method

### Overall Architecture

The task is formulated as multi-label classification: given a social media video published by an O&G entity, the model outputs a set of framing labels. The dataset is divided into a Facebook subset (7 fine-grained climate-obstruction framing categories, 320 videos) and a YouTube subset (6 impression-based framing categories, 386 videos). Evaluation is conducted via zero-shot and 1-shot VLM inference.

### Key Designs

1. **Dual-Platform, Dual-Taxonomy Framework**

    - **Function**: Comprehensively covers the differentiated advertising strategies employed across platforms by the O&G industry.
    - **Mechanism**: The **Facebook subset** inherits the climate-obstruction taxonomy from Holder et al. (7 label types: CA = community economy, CB = job creation, GA = emissions reduction and transition, GC = clean natural gas, PA = practical energy choice, PB = raw material uses, SA = domestic energy independence), with labels derived from prior text annotation. The **YouTube subset** introduces 6 newly defined impression-based labels (Community & Life, Work, Environment, Green Innovation, Economy & Business, Patriotism), annotated directly from video content by human experts.
    - **Design Motivation**: Facebook advertisements are short, politically oriented, and text-dominant; YouTube videos are longer, brand-focused, and visually implicit. The framing strategies on the two platforms differ fundamentally and cannot be adequately captured by a single taxonomy.

2. **Entity-Aware 1-Shot Prompt Construction**

    - **Function**: Selects the most relevant in-context example for each test video to facilitate in-context learning.
    - **Mechanism**: (1) **Entity Restriction (ER)**: Candidate examples are restricted to training samples from the same entity. (2) **Embedding-based Search (ES)**: CLIP encodes video frames and transcripts separately; a weighted sum yields the video representation $\bm{e} = 0.5\bm{e}_{\rm Frame} + 0.5\bm{e}_{\rm Transcript}$, and cosine similarity retrieves the most similar training sample as the 1-shot example.
    - **Design Motivation**: A given company's advertising strategy tends to be consistent (e.g., BP consistently emphasizes transition; ExxonMobil emphasizes employment), making entity-aware selection more informative than random selection.

3. **Transcript-Aligned Dynamic Frame Sampling**

    - **Function**: Selects the most representative frames from long videos together with their corresponding text.
    - **Mechanism**: Using timestamps from Whisper-1 transcript segments, the midpoint frame of each segment's time span is sampled, up to a maximum of $N_{\rm Frame}$ frames (10 for GPT-4.1/Qwen2.5-VL; 3 for InternVL2/DeepSeek). Each frame is paired with its corresponding transcript segment and fed to the VLM.
    - **Design Motivation**: Video lengths vary considerably (Facebook average 18 s; YouTube average 76 s), necessitating dynamic sampling. Frame–text alignment ensures that the VLM receives matched multimodal context.

### Loss & Training

No model training is involved—all VLMs are evaluated via zero-shot or few-shot inference. Annotation agreement is measured using Fleiss' Kappa; the YouTube subset achieves 0.61.

## Key Experimental Results

### Main Results

| Model | Parameters | YouTube (All) | Facebook (All) | Best Label F1 | Worst Label F1 |
|-------|-----------|--------------|----------------|--------------|----------------|
| GPT-4.1 (0-shot) | — | 71.0 | 61.1 | 84.9 (Comm.) | 46.1 (Green Innov.) |
| GPT-4.1 (1-shot) | — | 69.3 | 72.6 | 80.6 (Comm.) | 41.6 (Green Innov.) |
| Qwen2.5-VL (0-shot) | 32B | 60.7 | 49.0 | 73.5 (Env.) | 42.8 (Patrio.) |
| Qwen2.5-VL (1-shot) | 32B | 66.2 | 70.5 | 77.4 (Env.) | 45.8 (Green Innov.) |
| GPT-4o-mini (0-shot) | — | 60.5 | 54.2 | 72.8 (Env.) | 39.2 (Green Innov.) |
| DeepSeek-VL2 (1-shot) | 4.5B | 49.7 | 62.3 | 68.6 (Comm.) | 21.4 (Green Innov.) |

### Ablation Study

| Configuration | YouTube | Facebook | Notes |
|--------------|---------|----------|-------|
| Full (T+ES+ER) | 66.2 | 70.5 | Qwen2.5-VL 32B |
| w/o transcript (T=✗) | 61.2 | 60.6 | Transcripts matter more for Facebook |
| w/o embedding search (ES=✗) | 64.0 | 59.1 | Random example selection underperforms |
| w/o entity restriction (ER=✗) | 65.6 | 68.1 | Entity information offers limited gain on YouTube |

### Key Findings

- **Green Innovation is consistently the hardest label**: All models perform worst on this category (best F1 of only 46.1%), as its visual expression is highly implicit—laboratory scenes may indicate R&D or routine operations alike.
- **GPT-4.1 zero-shot outperforms 1-shot on YouTube**, whereas other models benefit from 1-shot prompting—suggesting that the strongest model has already internalized sufficient domain knowledge, and additional examples introduce noise.
- **Transcripts are critical on Facebook**: Removing transcripts drops F1 from 70.5 to 60.6 (−14%), because Facebook advertisement videos often serve merely as background while text carries the core message.
- **Small models can be effective**: DeepSeek-VL2 at 4.5B achieves 62.3% on Facebook 1-shot, approaching GPT-4o-mini, indicating that task-specific prompt design matters more than simply scaling model size.
- Approximately 30% of Facebook videos contain no transcript, making pure visual understanding essential for these samples.

## Highlights & Insights

- **The first multimodal greenwashing benchmark** fills a critical gap in the literature by extending the modality from text to video; the dataset design accounts for diversity across platforms (Facebook vs. YouTube), entities (50+ companies and advocacy groups), and cultures (20 countries).
- **The entity-aware prompting strategy** is simple yet effective—leveraging the consistency of a given company's advertising strategy to improve few-shot performance. This approach is transferable to any entity-level classification task, such as brand sentiment analysis or political advertisement detection.
- **The design philosophy of impression-based annotation**—YouTube subset labels are deliberately constructed to capture the subjective impression conveyed to viewers rather than objective facts, which precisely reflects the nature of greenwashing: it manipulates impressions rather than facts.

## Limitations & Future Work

- **Limited YouTube annotation coverage**: Only 386 videos, split evenly between training and test, yields insufficient statistical reliability for low-frequency labels such as Patriotism and Green Innovation.
- **Facebook labels are "distantly annotated"**: Original annotations are based on advertising text rather than the video itself, potentially introducing misalignment between video content and text-derived labels.
- **English only**: Although 20 countries are represented, the analysis focuses primarily on English-language content and does not address localized strategies in non-English-speaking markets.
- **Information loss from frame sampling**: Compressing full videos to 3–10 frames inevitably discards substantial temporal and narrative information, particularly for longer YouTube videos.
- **No fine-tuning baseline**: All experiments are zero-shot or few-shot; the performance ceiling attainable by fine-tuning on the training set is not explored.
- **Ambiguity in greenwashing judgment**: The dataset detects framing rather than greenwashing per se; inferring greenwashing from framing requires additional corporate behavior data.

## Related Work & Insights

- **vs. Rowlands et al. (2024)**: They defined a framing classification task for Facebook text advertisements; this work extends it to multimodal video and introduces the YouTube platform with a new label taxonomy.
- **vs. Holder et al. (2022)**: They qualitatively analyzed framing strategies in O&G Facebook advertising; this work translates those qualitative findings into a quantitatively evaluable NLP/VLM benchmark.
- **vs. general video understanding benchmarks (ActivityNet, Kinetics)**: This benchmark targets "strategic intent" rather than "physical actions"—detecting "what impression a video intends to convey" is considerably more difficult than detecting "what is happening in the video."
- The paper demonstrates the significant potential and shortcomings of VLMs in social science applications, contributing an important benchmark to the AI for Social Good literature.

## Rating

- Novelty: ⭐⭐⭐⭐ First multimodal greenwashing detection benchmark; dual-platform design and impression-based annotation are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six VLMs + ablation + error analysis; fine-tuning upper bound is absent.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction process is described in thorough and transparent detail.
- Value: ⭐⭐⭐⭐ Fills an important gap; open-source data and code directly advance AI + climate policy research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A3: Towards Advertising Aesthetic Assessment](../../CVPR2026/multimodal_vlm/a3_towards_advertising_aesthetic_assessment.md)
- [\[NeurIPS 2025\] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture](mirage_a_benchmark_for_multimodal_information-seeking_and_reasoning_in_agricultu.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[AAAI 2026\] OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding](../../AAAI2026/multimodal_vlm/omnipt_unleashing_the_potential_of_large_vision_language_models_for_pedestrian_t.md)

</div>

<!-- RELATED:END -->
