---
title: >-
  [Paper Note] Video Summarization with Large Language Models
description: >-
  [CVPR 2025][Video Understanding][Video Summarization] LLMVS proposes an LLM-based video summarization framework. It first employs a multimodal LLM to convert video frames into textual descriptions, and then uses an LLM to evaluate the local importance scores of each frame via sliding-window in-context learning. Finally, it aggregates the global context through a global self-attention mechanism to generate the final predictions, achieving SOTA performance on SumMe and TVSum.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Summarization"
  - "Large Language Models"
  - "Keyframe Selection"
  - "Local-to-Global"
  - "In-Context Learning"
date: 2026-05-08
content_hash: 408b8454ff882d48
---

# Video Summarization with Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2504.11199](https://arxiv.org/abs/2504.11199)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Summarization, Large Language Models, Keyframe Selection, Local-to-Global, In-Context Learning

## TL;DR

LLMVS proposes an LLM-based video summarization framework. It first employs a multimodal LLM to convert video frames into textual descriptions, and then uses an LLM to evaluate the local importance scores of each frame via sliding-window in-context learning. Finally, it aggregates the global context through a global self-attention mechanism to generate the final predictions, achieving SOTA performance on SumMe and TVSum.

## Background & Motivation

**Background**: Video content is growing exponentially. Efficient video navigation, search, and retrieval require advanced video summarization technologies. Existing methods are mainly divided into two categories: vision-only methods (such as VASNet, DSNet, CSTA) that select keyframes based on visual features and temporal dynamics, and multimodal methods (such as CLIP-It, A2Summ) that integrate visual and textual features, where text only serves as an auxiliary enhancement for visual representations.

**Limitations of Prior Work**: Vision-only methods primarily rely on visual saliency and fail to capture semantic information of video content, leading to incomplete or incoherent summaries. Multimodal methods, although incorporating text, still center around vision—textual features are used as keys/values in cross-attention to enhance visual queries, which essentially boils down to "detecting which frames are visually conspicuous" rather than "understanding which frames are important."

**Key Challenge**: The definition of "what is a keyframe" in video summarization is inherently subjective and semantic-driven—demanding an understanding of the narrative structure and informational value rather than relying solely on visual saliency. Traditional methods lack deep semantic understanding capabilities.

**Goal**: Leverage the knowledge learned by LLMs from massive data to evaluate the importance of video frames, with the expectation that LLM judgments align better with diverse human-annotated ground truths.

**Key Insight**: LLMs excel at contextual understanding and cross-domain reasoning, acting as natural "importance evaluators." However, LLMs cannot directly process videos and require converting visual signals into the textual space.

**Core Idea**: After translating video frames into textual descriptions, the model leverages the in-context learning of LLMs to assess frame importance within a local window, and then refines the predictions across the global video context using a trainable global self-attention module. The key is to use the LLM's output embeddings (rather than the final textual answer) as an intermediate representation.

## Method

### Overall Architecture

The input is a sequence of video frames $\mathbf{F} = [F_1, ..., F_T]$, and the output is the importance score for each frame $\mathbf{s} \in \mathbb{R}^{T \times 1}$. The pipeline consists of three steps: (1) using a frozen M-LLM (LLaVA-1.5-7B) to generate a single-sentence textual description for each frame; (2) using a frozen LLM (Llama-2-13B-chat) to evaluate the importance of the center frame within a local window via in-context learning, extracting the output embeddings; (3) feeding the embeddings of all frames into a trainable global self-attention module to output the final scores. Both the M-LLM and the LLM are frozen, while only the self-attention blocks are trained.

### Key Designs

1. **Text Description Generation (M-LLM to Text Space Conversion)**:

    - **Function**: Converts video frames from the visual space to the text space, enabling processing by the LLM.
    - **Mechanism**: Utilizes a frozen LLaVA-1.5-7B to describe each frame, prompted with "Provide a detailed one-sentence description," limiting each frame to a maximum of 77 tokens. The output is a text sequence $\mathbf{C} = \phi(\mathbf{F})$.
    - **Design Motivation**: Directly prompting the M-LLM for importance scoring is less effective than explicitly generating descriptions first and then passing them to a dedicated LLM (as verified by the ablation study). General and simple descriptions perform better than region-based descriptions (e.g., center vs. background) because overall descriptions more easily capture scene dynamics.

2. **Local Importance Scoring (LLM + Sliding Window In-Context Learning)**:

    - **Function**: Evaluates the relative importance of each frame within a local temporal context.
    - **Mechanism**: For a frame at time step $t$, the descriptions $C_{t-3:t+3}$ within a window of size $w=7$ are sent to Llama-2-13B-chat. The importance of the center frame is evaluated via in-context learning (using an instruction-example-query prompt with 3 examples). **Key Innovation**: Instead of using the LLM's final textual answer, the query embedding $\mathbf{q}_t \in \mathbb{R}^{L^q \times D}$ and answer embedding $\mathbf{a}_t \in \mathbb{R}^{L^a \times D}$ are extracted from after the RMS Norm layer, preserving richer contextual and semantic information.
    - **Design Motivation**: Video frames are highly redundant and need to be compared within a local context to filter out keyframes. The intuition behind using embeddings instead of the LLM's final answer is that the final answer is highly compressed information (a single number), whereas the embeddings retain the full details of the LLM's internal reasoning. The ablation study confirms that the embedding-based method significantly outperforms directly using the LLM's answers.

3. **Global Context Aggregation (Self-Attention Blocks)**:

    - **Function**: Refines local importance scores from the perspective of the entire video context to generate coherent summaries.
    - **Mechanism**: The query and answer embeddings of each frame are concatenated as $\mathbf{x}_t = \text{concat}(\mathbf{q}_t, \mathbf{a}_t)$, followed by max pooling and dimensionality reduction via an MLP to $\mathbb{R}^{1 \times M}$ ($M=2048$). The embeddings of all $T$ frames form a sequence $\mathbf{x}' \in \mathbb{R}^{T \times M}$. Global dependencies are modeled using a 3-layer, 2-head self-attention module $\psi$, and the final scores are output through an MLP: $\mathbf{s} = \text{MLP}(\psi(\mathbf{x}'))$.
    - **Design Motivation**: The local window of an LLM cannot perceive the global narrative structure. For example, a frame that appears unimportant locally might be a turning point globally. The self-attention mechanism enables the model to capture dependencies across windows. Only this lightweight module (3-layer SA) is trained, preserving the general knowledge of the LLM.

### Loss & Training

The Mean Squared Error (MSE) loss is employed: $\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T}(s_t - \hat{s}_t)^2$. The model is optimized using AdamW for 200 epochs on 5 A100 GPUs, with a batch size of 1. The learning rates are set to 1.19e-4 for SumMe and 7e-5 for TVSum. The M-LLM and LLM are fully frozen, with only the global attention module and MLPs being trained. The total training time is approximately 10 hours.

## Key Experimental Results

### Main Results

| Method | Type | SumMe τ↑ | SumMe ρ↑ | TVSum τ↑ | TVSum ρ↑ |
|------|------|----------|----------|----------|----------|
| Human | - | 0.205 | 0.213 | 0.177 | 0.204 |
| CSTA | Vision | 0.246 | 0.274 | 0.194 | 0.255 |
| MSVA | Vision | 0.200 | 0.230 | 0.190 | 0.210 |
| SSPVS | Vision+Text | 0.192 | 0.257 | 0.181 | 0.238 |
| LLM (zero-shot) | LLM | 0.170 | 0.189 | 0.051 | 0.056 |
| **LLMVS** | **LLM** | **0.253** | **0.282** | **0.211** | **0.275** |

### Ablation Study

| Configuration | SumMe τ↑ | SumMe ρ↑ |
|------|----------|----------|
| LLaVA Direct Scoring (w/o LLM) | 0.119 | 0.132 |
| LLaVA* Fine-tuned Direct Scoring | 0.140 | 0.156 |
| LLaVA→Llama (w/o Global Aggregation) | 0.170 | 0.189 |
| LLaVA→Llama* Fine-tuned (w/o Global Aggregation) | 0.181 | 0.201 |
| **LLaVA→Llama + SA* (LLMVS)** | **0.253** | **0.282** |

Embedding Selection Ablation:

| Configuration | τ↑ | ρ↑ |
|------|-----|-----|
| Answer embedding $\mathbf{a}$ only + SA | 0.233 | 0.260 |
| Query embedding $\mathbf{q}$ only + SA | 0.238 | 0.265 |
| **$\mathbf{q} + \mathbf{a}$ + SA** | **0.253** | **0.282** |
| $\mathbf{q} + \mathbf{a}$ + MLP (w/o SA) | 0.182 | 0.203 |

### Key Findings

- Global self-attention is the most critical module: The improvement from zero-shot LLM (0.170) to LLMVS (0.253) primarily stems from the global aggregator, demonstrating that local windows are insufficient for producing coherent summaries.
- The LLM's output embeddings are far superior to direct answers: LLMVS using embeddings significantly outperforms the version using only the LLM's final numerical answers, proving that intermediate representations indeed retain more useful information.
- Zero-shot LLM is competitive on SumMe (0.170 vs. Human 0.205) but performs poorly on TVSum (0.051), as TVSum is evaluated individually for each annotator—LLMs are suited for generic summaries but struggle to capture personal preferences.
- Query embeddings are slightly better than answer embeddings (0.238 vs. 0.233), indicating that the context of all frames within the window is more informative than the final evaluation.
- Numerical scoring prompting outperforms text summarization prompting (0.253 vs. 0.239), showing that a direct importance scoring task is more straightforward.
- Extracting embeddings from the RMS Norm layer is superior to the Linear layer (0.253 vs. 0.241).

## Highlights & Insights

- **"Using embeddings instead of answers"**: The capability of an LLM is reflected not only in its final output; the representations in its intermediate layers contain richer information about the reasoning process. This finding is transferable to any downstream task using LLMs for decision-making—utilize the embeddings rather than just the answers.
- **Ultralight architectural design**: The M-LLM and LLM are completely frozen; only a 3-layer self-attention block is trained, completing training in 10 hours. This proves that the paradigm of "retrieving LLM knowledge + lightweight adaptation" is efficient in video understanding.
- **Language-centric video understanding**: Reframing video summarization as a language understanding problem assigns the "protagonist" role to the LLM rather than a vision model. This stands in sharp contrast to traditional multimodal approaches (text assisting vision).

## Limitations & Future Work

- Textual descriptions inevitably lose some visual information and may be insensitive to purely vision-driven elements of importance (such as aesthetics or composition).
- The token cost of the sliding window is high—each frame requires a full forward pass of the LLM, leading to high inference overhead for long videos.
- Evaluation is conducted only on two small benchmarks, SumMe and TVSum. These datasets are limited in scale and widely used, carrying a risk of overfitting.
- The 3 examples for in-context learning are randomly sampled from the training set; the quality of these examples may impact the assessment of the LLM.
- Freezing the LLM limits its potential to adapt to dataset characteristics—fine-tuning the LLM might yield further performance gains.

## Related Work & Insights

- **vs CSTA**: A CNN-based visual method that processes frame sequences as images and extracts spatial-temporal attention using a 2D CNN; LLMVS replaces visual analysis with language understanding, making the two approaches complementary.
- **vs CLIP-It**: Uses the cross-attention of CLIP to fuse visual and textual features, with text acting as an auxiliary; LLMVS reverses this paradigm, being language-centric.
- **vs AntGPT/MovieChat**: Also leverages LLMs for video understanding, but for Q&A or action prediction; LLMVS is the first to utilize LLMs for frame-level importance evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of using LLM embeddings for video summarization is novel, and the local-to-global two-stage design is rational.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are highly detailed (embedding selection, prompting strategies, extraction positions, etc.), but restricted to only two small datasets.
- **Writing Quality**: ⭐⭐⭐⭐ The article is clearly structured with intuitive charts and detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ Demonstrates the effectiveness of LLMs in video summarization; the finding that "embeddings outperform answers" has broad inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](../../NeurIPS2025/video_understanding/monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)

</div>

<!-- RELATED:END -->
