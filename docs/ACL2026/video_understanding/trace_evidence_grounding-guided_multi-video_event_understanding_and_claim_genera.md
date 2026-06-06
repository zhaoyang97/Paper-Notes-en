---
title: >-
  [Paper Note] TRACE: Evidence Localization-based Multi-Video Event Understanding and Statement Generation
description: >-
  [ACL 2026][Video Understanding][Multi-video event understanding] TRACE achieves SOTA on multi-video event understanding tasks by adopting a "locate-then-reason" pipeline. It first constructs a text-searchable video timel…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Multi-video event understanding"
  - "evidence localization"
  - "claim generation"
  - "video citation"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: 83bcbaaf4ff9c2a4
---

# TRACE: Evidence Localization-based Multi-Video Event Understanding and Statement Generation

**Conference**: ACL 2026  
**arXiv**: [2605.16740](https://arxiv.org/abs/2605.16740)  
**Code**: https://github.com/pengyu965/TRACE  
**Area**: Video Understanding  
**Keywords**: Multi-video event understanding, evidence localization, claim generation, video citation, Large Vision-Language Models

## TL;DR

TRACE achieves SOTA on multi-video event understanding tasks by adopting a "locate-then-reason" pipeline. It first constructs a text-searchable video timeline using OCR and object detection, then utilizes a text LLM for query-conditioned evidence localization, and finally employs an LVLM to generate cited claims. This approach improves the F1 score from 0.705 to 0.811.

## Background & Motivation

**Background**: Multi-video event understanding requires models to not only recognize visual content but also localize and attribute discrete evidence fragments distributed across long video corpora. Recent Large Vision-Language Models (LVLMs) demonstrate strong performance in general video understanding but face two core bottlenecks in this specific scenario.

**Limitations of Prior Work**: Directly processing raw videos with LVLMs faces three difficulties. First, models tend to focus on visually salient content (e.g., main characters, background landscapes) while ignoring specific query-relevant evidence (casualty numbers on news scrolls, total votes in broadcast captions, scoreboard data). Second, even state-of-the-art LVLMs are forced to perform aggressive temporal sampling on long videos due to context window limits, causing brief segments containing critical information—such as a fleeting news ticker with vital statistics—to be missed. Third, these models struggle to precisely localize "event-relevant" moments.

**Key Challenge**: The crux is that event videos are replete with structured semantic signals (broadcast captions, detected object categories, OCR text) that can be extracted cheaply, yet existing LVLM pipelines mostly fail to fully exploit them. Increasing the LVLM context window alone does not solve the problem, as the challenge is not just "seeing more frames," but "identifying which frames matter."

**Goal**: To design a system capable of precisely localizing evidence within long, heterogeneous video collections and generating claims with citation attributes. Key requirements include: (1) performing evidence localization efficiently in the text space to avoid frame-by-frame LVLM calls; (2) leveraging OCR and detection signals to guide the LVLM toward evidence fragments; and (3) consolidating evidence and citations across multiple videos to avoid double counting.

**Key Insight**: OCR text in event videos (lower thirds, scoreboards, graphic overlays) is often more semantically precise than the raw visual appearance itself. These signals can be cheaply extracted via YOLOv12 object detection and OCR, providing an interpretable text serialization for downstream reasoning.

**Core Idea**: A "locate-then-reason" paradigm is adopted. Instead of making the LVLM perform evidence discovery and generation simultaneously, the system first constructs a text-searchable video timeline (via OCR + detection). A text LLM is used for query-conditioned evidence localization, followed by guided LVLM generation and citation integration.

## Method

### Overall Architecture

The TRACE pipeline consists of three sequential stages. **Stage 1** constructs a structured localization representation for each input video: YOLOv12 object detection and OCR text recognition are run on sampled frames to generate a timestamp-detection-OCR triplet timeline. **Stage 2** segments this timeline into fixed-size windows, serializes them into text, and feeds them into a text LLM along with the user query and persona information. The LLM determines which frames within each window are relevant. **Stage 3** involves the LVLM receiving a hybrid frame set (uniformly sampled frames + evidence localization frames) and structured localization annotations to generate query-conditioned claims. Finally, claims across multiple videos are deduplicated and citations are integrated through semantic clustering and LLM verification.

### Key Designs

1. **Structured Video Localization Representation**:

    - **Function**: Transitions long videos into a text-searchable, efficiently filterable representation, avoiding frame-by-frame visual reasoning.
    - **Mechanism**: YOLOv12 detection and OCR are run on uniformly sampled frames. Detection outputs are sets of triplets $(l_i, c_i, \mathbf{b}_i)$, where $l_i$ is the COCO-80 label, $c_i$ is confidence, and $\mathbf{b}_i$ is the bounding box. Object co-occurrence patterns (e.g., detecting people, microphones, and podiums simultaneously) allow for scene type inference without a dedicated scene classifier. OCR extracts textual content like broadcast captions, numbers on scoreboards, and entity names. The two streams merge into a timeline $\mathcal{F}=\{(t, \mathcal{D}_t, \mathcal{T}_t)\}_{t=0}^T$.
    - **Design Motivation**: Directly processing raw frames with an LVLM misses critical short segments due to context budgets. Building a serializable text representation with lightweight detection and OCR preserves semantic information while enabling subsequent query-conditioned localization using fast text LLMs.

2. **Query-Conditioned Evidence Localization**:

    - **Function**: Identifies which video segments contain relevant evidence in the text space based on user queries and personas, directing the subsequent LVLM generation.
    - **Mechanism**: The system segments the timeline into non-overlapping windows $\{\mathcal{F}_j\}$ of size $C$ frames. Each window is serialized into compact text (including timestamps, detected objects, and OCR strings) and sent to a text LLM alongside query $q$ and persona $p$. The LLM outputs a subset of relevant frames $\mathcal{S}_j$ and their supporting strings. The union of relevant frames $\mathcal{S}=\bigcup_j \mathcal{S}_j$ forms the query-relevant keyframe set. This stage operates entirely in the text space without visual encoding, making it orders of magnitude more efficient than dense LVLM inference.
    - **Design Motivation**: Using a text LLM for low-cost primary filtering allows it to learn the semantic bridge between queries and signals (e.g., relating "vote count" to "percentage sign in OCR"), reducing irrelevant frames for the LVLM and freeing context capacity for critical moments.

3. **Hybrid Frame Selection and Evidence Fusion**:

    - **Function**: Constructs the visual input for the LVLM to maintain global temporal coverage while focusing attention on evidence fragments, ultimately generating cited claims.
    - **Mechanism**: The LVLM visual input is the union of uniformly sampled frames and localization frames $\mathcal{I}_v = \mathcal{I}_{\text{unif}} \cup \{\hat{i}_s : t_s \in \mathcal{S}\}$, where $\mathcal{I}_{\text{unif}}$ contains $N_{\text{unif}}=100$ linearly spaced frames. Crucially, frame indices are passed as explicit positional metadata (rather than dense rank $0,1,...,N-1$) to preserve correct temporal intervals in the LVLM's rotary positional embeddings. Finally, five evidence streams (hybrid frames, query, persona, structured localization annotations, and ASR transcripts) are combined into a single prompt.
    - **Design Motivation**: Uniform sampling provides global temporal insurance against localization errors; evidence frames focus visual capacity; explicit temporal metadata ensures temporal alignment between text annotations and corresponding visual tokens, preventing cross-modal axis drift.

## Key Experimental Results

### Main Results

Quantitative comparison on the MAGMaR 2026 Oracle Track validation set (8 event topics):

| Method | Avg. F1 | Info Prec | Info Rec | Info F1 | Cite Prec | Cite Rec | Cite F1 |
|------|---------|---------|---------|---------|---------|---------|---------|
| Qwen3.5-9B | 0.472 | 0.437 | 0.756 | 0.554 | 0.875 | 0.251 | 0.390 |
| Qwen3-VL-8B | 0.723 | 0.870 | 0.802 | 0.835 | 0.930 | 0.452 | 0.608 |
| Qwen3-VL-30B (Baseline) | 0.705 | 0.883 | 0.731 | 0.800 | 0.990 | 0.440 | 0.609 |
| **TRACE (Full)** | **0.811** | **0.863** | **0.876** | **0.869** | **0.939** | **0.628** | **0.753** |

Compared to the strongest baseline (Qwen3-VL-30B), TRACE improves Avg. F1 by +0.106 (+15%). Notably, citation recall increases from 0.440 to 0.628 (+42.7%), indicating that localization guidance enables the model to discover and attribute evidence from multiple videos.

### Ablation Study

| Configuration | Keyframe Augment | Clustering Strategy | Avg. F1 | Info F1 | Cite F1 |
|------|-----------|---------|---------|---------|---------|
| w/o Loc. Guidance + LLM Clust. | ✗ | LLM | 0.802 | 0.859 | 0.745 |
| w/o Loc. Guidance + Embed-Sim Clust. | ✗ | Embed-Sim | 0.808 | 0.868 | 0.748 |
| w/ Loc. Guidance + LLM Clust. | ✓ | LLM | 0.804 | 0.867 | 0.741 |
| Full Model | ✓ | Embed-Sim | **0.811** | **0.869** | **0.753** |

### Key Findings

- **Localization guidance is the primary contributor**: All four variants significantly outperform the baseline (Avg. F1 ≥0.802 vs 0.705), indicating that structured localization is the main driver of improvement.
- **Embedding similarity clustering is more precise**: Embed-Sim clustering outperforms pure LLM clustering under both frame selection settings, with the difference being more pronounced in citation F1.
- **Localization frames provide complementary gains**: Adding localization frames improves info recall from 0.858 to 0.885 under LLM clustering, though improvements are limited under Embed-Sim, suggesting that text localization already captures most evidence context at the prompt level.
- **Cross-dataset generalization**: On WikiVideo (52 queries), TRACE achieves 0.879 Avg. F1 (vs 0.854 for Qwen3-VL-30B), with a significant advantage in citation recall (0.838 vs 0.792).

## Highlights & Insights

- **Innovation of the "Locate-then-Reason" Paradigm**: Multi-video event understanding is reformulated as an evidence localization problem rather than direct generation. This decomposition allows a lightweight text LLM to handle low-cost primary filtering, drastically reducing LVLM inference costs and context waste. This concept is transferable to any task requiring precise localization in long contexts.
- **Reuse Value of OCR as High-Precision Semantic Signals**: Traditional LVLMs often ignore structured text like broadcast captions or scoreboards. TRACE proves these signals are often more informative for event understanding than visual appearance itself.
- **Efficiency of Evidence Localization in Text Space**: By performing complex query alignment in the text space, the system avoids visual encoding for every potential keyframe. In practice, this makes the localization stage over 50x faster than dense LVLM inference.

## Limitations & Future Work

**Limitations**: (1) The YOLO detector is restricted to the COCO-80 vocabulary, failing to recognize domain-specific entities in many news queries. (2) Stages in the pipeline are non-differentiable and sequential; localization errors propagate forward without recovery. (3) Cross-video clustering based on embedding similarity and LLM verification may misclassify claims that are semantically similar but factually distinct.

(Self-identified limits): (1) Gain from localization is limited on short video collections (e.g., WikiVideo) where uniform sampling is already dense enough. (2) ASR transcription quality significantly impacts long video context, yet its handling is not discussed. (3) Generated claim length and detail correlate with input frame count, risking excessive wordiness.

**Future Work**: (1) Integrate open-vocabulary detectors (e.g., GroundingDINO) to expand entity coverage. (2) Design backpropagation or reinforcement learning schemes for end-to-end optimization of localization and generation. (3) Introduce adaptive sampling strategies for the localization stage, dynamically adjusting windows and sampling rates based on event frequency.

## Related Work & Insights

- **vs. Long-Context LVLMs (Video-LLaVA / VideoChat / Qwen3-VL)**: These works improve visual capacity via memory compression or adaptive frame selection, assuming better memory allows LVLMs to auto-localize evidence. TRACE's insight is that the issue lies in the attention bias of LVLMs toward raw visual content rather than capacity.
- **vs. Retrieval-Augmented Generation (RAG)**: RAG systems retrieve relevant documents before generation. TRACE adopts a similar decomposition but in the multimodal domain: using lightweight structured signals instead of dense embeddings for retrieval.
- **vs. Modular Multimodal Reasoning (Visual Programming / ViperGPT)**: These works improve multimodal reasoning by decomposing perception and reasoning into specialized modules. TRACE generalizes this by treating evidence discovery as a dedicated localization module.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying the "locate-then-reason" paradigm to multi-video event understanding is innovative, though the basic ideas of RAG and modular reasoning are known.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detailed evaluation across two benchmarks with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-organized with informative figures and detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ Achieved SOTA on the MAGMaR 2026 official leaderboard with substantial improvements in citation recall (+42.7%).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](../../CVPR2026/video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](../../AAAI2026/video_understanding/emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)
- [\[ICML 2026\] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](../../ICML2026/video_understanding/videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)
- [\[ACL 2026\] Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning](rethinking_the_idiomaticity_decomposability_hypothesis_evidence_from_distributio.md)

</div>

<!-- RELATED:END -->
