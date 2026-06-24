---
title: >-
  [Paper Note] VisionArena: 230K Real World User-VLM Conversations with Preference Labels
description: >-
  [CVPR 2025][Multimodal VLM][Vision-Language Models] VisionArena constructs a large-scale dataset containing 230K real-world user-VLM interaction records (including preference labels), covering 73K users, 45 VLMs, and 138 languages. It reveals the current limitations of VLMs in spatial reasoning and planning tasks, and demonstrates that fine-tuning on real dialogue data significantly outperforms LLaVA-Instruct.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Human Preferences"
  - "Benchmarking"
  - "Real-World User Interactions"
  - "Chatbot Arena"
date: 2026-05-08
content_hash: 364b6e462d6db42f
---

# VisionArena: 230K Real World User-VLM Conversations with Preference Labels

**Conference**: CVPR 2025  
**arXiv**: [2412.08687](https://arxiv.org/abs/2412.08687)  
**Code**: [https://huggingface.co/lmarena-ai](https://huggingface.co/lmarena-ai)  
**Area**: Recommendation Systems / VLM Evaluation  
**Keywords**: Vision-Language Models, Human Preferences, Benchmarking, Real-World User Interactions, Chatbot Arena

## TL;DR
VisionArena constructs a large-scale dataset containing 230K real-world user-VLM interaction records (including preference labels), covering 73K users, 45 VLMs, and 138 languages. It reveals the current limitations of VLMs in spatial reasoning and planning tasks, and demonstrates that fine-tuning on real dialogue data significantly outperforms LLaVA-Instruct.

## Background & Motivation

**Background**: While VLM capabilities are rapidly growing, existing benchmarks are mostly artificially constructed, failing to reflect real-world user scenarios and preferences. Chatbot Arena has been successfully applied to LLM evaluation, but the vision domain lacks a similar large-scale, real-world interaction dataset.

**Limitations of Prior Work**: (1) Gaps exist between artificial benchmarks and real-world scenarios; (2) There is a lack of large-scale user preference data to guide VLM training; (3) It remains unclear how users actually interact with VLMs and on which tasks these models perform poorly.

**Key Challenge**: There is a critical need for large-scale, realistic, and diverse user-VLM interaction datasets, but collecting such data is costly and involves privacy concerns.

**Goal**: To build the first large-scale, real-world user-VLM conversation dataset with preference labels to support both training and evaluation.

**Key Insight**: Leverage real-world user interaction records from the open-source Chatbot Arena platform, augmented with preference voting features.

**Core Idea**: Collect 230K real-world conversations from Chatbot Arena, categorized into three subsets: Chat (200K conversations), Battle (30K preference comparisons), and Bench (500 automated benchmark prompts).

## Method

### Overall Architecture
The dataset consists of three subsets: (1) VisionArena-Chat: 200K single-turn/multi-turn user-VLM conversations; (2) VisionArena-Battle: 30K records where users simultaneously chat with two anonymous VLMs and vote for their preference; (3) VisionArena-Bench: 500 automated benchmark prompts that efficiently approximate online Chatbot Arena rankings.

### Key Designs

1. **VisionArena-Chat (200K Conversations)**:

    - **Function**: Provide large-scale, real-world VLM training data.
    - **Mechanism**: Collect user-submitted dialogue records from the Chatbot Arena platform, covering 138 languages and 45 VLMs. Conversations contain user-uploaded images, text queries, and VLM responses.
    - **Design Motivation**: Real-world data aligns better with the actual distribution of user interactions than synthetic instruction data, yielding better fine-tuning performance.

2. **VisionArena-Battle (30K Preference Comparisons)**:

    - **Function**: Provide high-quality preference labels for RLHF training and model ranking.
    - **Mechanism**: Users send the same query to two anonymous VLMs simultaneously, then vote for the better response (or select a tie). This side-by-side comparison is the gold standard for acquiring reliable preference signals.
    - **Design Motivation**: Preference data can be directly utilized for RLHF training or building reward models.

3. **VisionArena-Bench (500 Automated Benchmark)**:

    - **Function**: Provide automated evaluation tools.
    - **Mechanism**: Select 500 diverse prompts from the Battle data and use a strong VLM as a judge to auto-score, efficiently approximating the online Arena's ELO rankings.
    - **Design Motivation**: Since online Arena rankings require substantial human labor, automated benchmarks allow for fast evaluation of new models.

### Loss & Training
Standard instruction fine-tuning was performed using VisionArena-Chat. Experiments demonstrate that, using the same base model, fine-tuning on VisionArena-Chat outperforms LLaVA-Instruct-158K by 17 points on MMMU and by 46 points on WildVision.

## Key Experimental Results

### Main Results

| Training Data | MMMU | WildVision |
|---------|------|-----------|
| LLaVA-Instruct-158K | Baseline | Baseline |
| VisionArena-Chat | **+17** | **+46** |

### Key Findings

| Finding | Details |
|------|------|
| Response Style Preference | Open-ended tasks (description, humor) highly rely on response style |
| Model Weaknesses | Current VLMs perform poorly on spatial reasoning and planning tasks |
| Data Quality | Real user data is far superior to synthetic instruction data for training |

### Key Findings
- User preferences for open-ended tasks (e.g., captioning, humor generation) heavily rely on response style rather than content correctness.
- Current VLMs generally perform poorly on spatial reasoning and planning tasks.
- The distribution of real-world user queries significantly differs from existing benchmarks, containing a large volume of non-English queries.

## Highlights & Insights
- **First large-scale, real-world VLM interaction dataset**: 230K conversations, 73K users, and 138 languages provide diversity far exceeding existing datasets.
- **Importance of training data quality**: Simply replacing the training data (VisionArena-Chat instead of LLaVA-Instruct) yielded substantial improvements, demonstrating the value of real-distribution data.
- **Complexity of preference signals**: The finding regarding the separation of style vs. content preferences offers critical implications for RLHF training strategies.

## Limitations & Future Work
- The data originates from Chatbot Arena users, which may introduce user demographic bias (biased toward technical users).
- Some conversations may contain privacy-sensitive content, and the data cleaning process is not described in detail.
- The 500 prompts in VisionArena-Bench may not be fully comprehensive.

## Related Work & Insights
- **vs WildVision**: An earlier, smaller-scale, real-world VLM evaluation dataset. VisionArena significantly outperforms it in both scale and diversity.
- **vs Chatbot Arena (LLM)**: VisionArena successfully migrates the Arena paradigm to the vision modality.
- Serves as a vital data resource supporting future VLM training and alignment research.

## Rating
- Novelty: ⭐⭐⭐⭐ The first large-scale real-world VLM dialogue dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed data analysis, with training experiments validating the dataset's value.
- Writing Quality: ⭐⭐⭐⭐ Clear descriptions of dataset construction and analysis.
- Value: ⭐⭐⭐⭐⭐ Extremely high value of the dataset for the VLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](sting-bee_towards_vision-language_model_for_real-world_x-ray_baggage_security_in.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/multimodal_vlm/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](../../ICCV2025/multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ICLR 2026\] WorldSense: Evaluating Real-World Omnimodal Understanding for Multimodal LLMs](../../ICLR2026/multimodal_vlm/worldsense_evaluating_real-world_omnimodal_understanding_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
