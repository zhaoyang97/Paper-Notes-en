---
title: >-
  [Paper Note] DrVideo: Document Retrieval Based Long Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Long Video Understanding] This paper proposes DrVideo, which reformulates **long video understanding as a long document understanding** task: it first converts video frames into text documents, locates keyframes and **augments information** via **document retrieval**, then uses a **Planning-Interaction dual-agent loop** to iteratively retrieve missing information, and finally answers questions in a CoT manner. It significantly outperforms exis…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Document Retrieval"
  - "LLM Agent"
  - "Information Augmentation"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: a566c4bf35eaad4e
---

# DrVideo: Document Retrieval Based Long Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2406.12846](https://arxiv.org/abs/2406.12846)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Long Video Understanding, Document Retrieval, LLM Agent, Information Augmentation, Chain-of-Thought

## TL;DR

This paper proposes DrVideo, which reformulates **long video understanding as a long document understanding** task: it first converts video frames into text documents, locates keyframes and **augments information** via **document retrieval**, then uses a **Planning-Interaction dual-agent loop** to iteratively retrieve missing information, and finally answers questions in a CoT manner. It significantly outperforms existing LLM-based SOTAs on EgoSchema (3 mins), MovieChat-1K (10 mins), and Video-MME long video division (average of 44 mins).

## Background & Motivation

Long video understanding is a core challenge in computer vision, requiring spatiotemporal information processing and long-range reasoning in videos spanning dozens of minutes or even hours.

**Limitations of Prior Work**:
- **Video-LM methods** (e.g., LLaVA-NeXT-Video): These methods encode video frames into visual token sequences and concatenate them with text tokens, but (i) they cannot take the entire long video as input due to token length limits, (ii) they usually employ uniform sampling with large strides (e.g., 1 frame per 16 frames), leading to the loss of keyframes, and (iii) the simple concatenation of tokens makes it difficult for LLMs to locate critical information within long sequences.
- **LLM-based methods** (e.g., LLoVi, VideoAgent): These methods convert videos into text descriptions and leverage LLMs for reasoning, but (i) **the keyframe localization is inaccurate**—VideoAgent relies on prior LLM knowledge to infer missing information and then uses CLIP similarity for frame selection, which lacks a holistic grasp of the video content; (ii) **information loss is severe**—even when keyframes are found, the generated captions fail to cover all key details (e.g., the description "a woman looking in the mirror" cannot answer "what she is wearing").

**Key Challenge**: Key information in long videos is sparse and scattered, requiring both global video understanding to locate keyframes and targeted information augmentation to prevent caption omissions.

**Core Idea**: To simulate how humans understand long videos—first scanning the entire video to grasp the general content, then locating key parts relevant to the question, and finally examining them closely to answer. This process is formalized as **document retrieval + augmentation + multi-stage Agent interaction**.

## Method

### Overall Architecture

DrVideo is composed of five components (Figure 1): (1) **Video-to-Document Conversion Module**: converts each frame to a short text description to construct the initial document; (2) **Retrieval Module**: calculates the semantic similarity between the question and the document to retrieve the top-K keyframes; (3) **Document Augmentation Module**: utilizes a VLM to generate detailed, targeted descriptions for the keyframes; (4) **Multi-stage Agent Interaction Loop**: a Planning Agent and an Interaction Agent iteratively search for and augment missing information; (5) **Answering Module**: generates answers in a CoT manner based on the final document.

### Key Designs

1. **Video-to-Document Conversion + Document Retrieval**
    - **Function**: Converts long videos into searchable text documents and locates keyframes relevant to the question.
    - **Mechanism**: 
     - First uses a VLM (e.g., LLaVA-NeXT) to generate a brief description of within 50 words for each frame, constructing the initial document $Doc_{init} = \{\{1, S_{V_1}\}, \{2, S_{V_2}\}, \ldots, \{T, S_{V_T}\}\}$.
     - Encodes the document and the question using an OpenAI embedding model, and retrieves top-K (default K=5) keyframes via cosine similarity.
    - **Design Motivation**: Semantic retrieval in the text space is more **precise** than CLIP's image-text similarity (fully leveraging the long-text retrieval capabilities of LLMs), and can be further improved after document augmentation.

2. **Document Augmentation Module**
    - **Function**: Supplements the retrieved keyframes with detailed, question-relevant information, compensating for the information loss in the initial short descriptions.
    - **Mechanism**: For each keyframe $t'$, LLaVA-NeXT is employed with different prompts to generate a detailed description $L_{V_{t'}}$. The initial augmented prompt is a general QA prompt, while subsequent agent loops generate more targeted prompts (e.g., requesting specific types of information).
    - **Design Motivation**: While the initial 50-word descriptions inevitably omit many details, generating detailed descriptions for all frames is prohibitively expensive (T can be hundreds of frames). Augmenting only the top-K keyframes is the most cost-efficient solution.

3. **Multi-Stage Agent Interaction Loop**
    - **Function**: Iteratively discovers and supplements key information that is still missing.
    - **Mechanism**: 
     - **Planning Agent**: Given question Q and the current augmented document $\mathcal{AD}_i$, determines whether the current information is sufficient to confidently answer. If not, it analyzes the reason and updates the analysis history $\mathcal{H}_i$.
     - **Interaction Agent**: Based on the analysis history and the current document, identifies $N$ keyframes with missing information ($n \notin \text{topk\_doc}$, $N < K$), determines the type of information needed for each frame (A: Image Captioning or B: Visual Question Answering), and then interacts with the document augmentation module to retrieve the information.
     - Loops until the Planning Agent deems the information sufficient or the maximum number of iterations is reached.
    - **Design Motivation**: A single retrieval + augmentation pass may miss frames that are indirectly related yet crucial for reasoning. The agent loop discovers "missing pieces" through in-context reasoning, making it more intelligent than static retrieval.

## Key Experimental Results

### Main Results

| Benchmark (Video Length) | Method | LLM | Core Metrics |
|----------------|------|-----|---------|
| **EgoSchema Subset** (3min) | VideoAgent | GPT-4 | 60.2% |
| | LLoVi | GPT-4 | 61.2% |
| | **DrVideo** | **GPT-4** | **66.4%** (+5.2%) |
| **MovieChat-1K Global** (10min) | LLoVi | GPT-4 | 58.3% Acc |
| | **DrVideo** | **GPT-4** | **93.1%** (+34.8%) |
| **MovieChat-1K Breakpoint** (10min) | VideoAgent* | GPT-4 | 31.6% Acc |
| | **DrVideo** | **GPT-4** | **56.4%** (+24.8%) |
| **Video-MME Long Videos w/o subs** (44min) | LLoVi* | GPT | 45.4% |
| | **DrVideo** | **DeepSeek** | **51.7%** (+6.3%) |
| **Video-MME Long Videos w/ subs** (44min) | GPT-4o mini | - | 63.4% |
| | Gemini 1.5 Flash | - | 68.8% |
| | **DrVideo** | **DeepSeek** | **71.7%** |

- In the Video-MME + subtitles setting, DrVideo even **outperforms** GPT-4o mini (71.7% vs 63.4%) and Gemini 1.5 Flash (71.7% vs 68.8%).

### Ablation Study (EgoSchema Subset, GPT-3.5)

| Configuration | Accuracy (%) |
|------|----------|
| w/o Retrieval Module + w/o Agent Loop + w/ CoT | 57.4 |
| + Retrieval Module | 60.6 (+3.2) |
| + Agent Loop | 62.6 (+2.0) |
| Full DrVideo (w/o CoT) | 62.2 |
| Full DrVideo | **62.6** |

- The Retrieval Module contributes the most (+3.2%), while the Agent Loop brings further improvement (+2.0%).
- Both the VQA and Caption augmentation types are necessary (removing VQA: -2.2%, removing Caption: -0.8%).
- Top-K=5 performs the best; scaling up K introduces noise instead.

### Key Findings

- It is highly feasible to reformulate long video understanding as document understanding, fully **leveraging the long-text retrieval and reasoning capabilities of LLMs**.
- The Agent interaction loop can find keyframes missed by pure similarity-based retrieval through in-context reasoning.
- Subtitle information is extremely critical for long video understanding—using subtitles alone (w/o vision) can achieve 68.5%, while the visual information augmentation contributes an additional 3.2%.
- DrVideo is training-free and can be reproduced on a single RTX 4090 with a reasonable number of API calls.

## Highlights & Insights

- **Paradigm Shift**: Transitioning from "video $\to$ visual tokens $\to$ LLM" to "video $\to$ text document $\to$ LLM", which cleverly reformulates long video questions into long document understanding tasks where LLMs excel.
- **Progressive Information Gathering**: The "retrieval $\to$ augmentation $\to$ agent supplementation" pipeline mimics the human cognitive process of understanding long videos, where the sufficiency of information is autonomously judged by the LLM.
- **Strong Experimental Results**: The accuracy boost on MovieChat-1K from 58.3% to 93.1% (Global mode) is remarkable.
- It is training-free and reproducible on a single GPU, making it extremely friendly to researchers.

## Limitations & Future Work

- It is heavily reliant on LLM APIs (GPT-4/DeepSeek), with **inference cost** scaling proportionally to the video length and the number of Agent loop iterations.
- The caption quality in the video-to-document conversion stage sets the upper bound of performance—if the captioner overlooks key information and subsequent Agents fail to detect it, the loss is irreversible.
- Relying solely on reasoning within the language space, it may struggle with tasks requiring precise spatial localization or strict visual detail judgment (e.g., "where is the object located in the scene").
- The frame sampling rate is fixed (0.5 FPS / 0.2 FPS), which might miss keyframes in videos containing rapid actions.

## Related Work & Insights

- **LLoVi** pioneered the paradigm of "short-clip description $\to$ LLM summarization $\to$ answering", on top of which DrVideo introduces document retrieval and information augmentation.
- The agent-based design of **VideoAgent** inspired DrVideo's multi-stage loop, but DrVideo replaces CLIP similarity localization with document retrieval, rendering it more precise.
- The success of **RAG (Retrieval-Augmented Generation)** in NLP has been successfully transferred to video understanding scenarios.
- Insights: Other long-sequence understanding tasks (such as long audio understanding or multi-turn dialogue history understanding) can also be formulated as document retrieval problems.

## Rating

⭐⭐⭐⭐ — Formulating long video understanding as document retrieval and augmentation is a clean and elegant framework design. The experimental results significantly outperform the SOTA on multiple benchmarks, particularly the 34.8% improvement on MovieChat-1K and the convincing results on Video-MME that surpass GPT-4o mini and Gemini 1.5 Flash. The training-free and single-GPU-reproducible nature is highly practical. However, heavy reliance on LLM APIs is a practical limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [\[CVPR 2025\] VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video SpatioTemporal Augmentation](vista_enhancing_long-duration_and_high-resolution_video_understanding_by_video_s.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)

</div>

<!-- RELATED:END -->
