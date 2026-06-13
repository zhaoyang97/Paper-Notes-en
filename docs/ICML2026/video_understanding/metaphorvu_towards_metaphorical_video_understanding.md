---
title: >-
  [Paper Note] MetaphorVU: Towards Metaphorical Video Understanding
description: >-
  [ICML 2026][Video Understanding][Metaphorical Video Understanding] This paper proposes the first metaphorical video understanding benchmark, MetaphorVU-Bench (860 videos + 8 metaphor categories)…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Metaphorical Video Understanding"
  - "Multimodal Large Language Models (MLLMs)"
  - "Cross-domain Mapping"
  - "Knowledge Graph Enhancement"
date: 2026-05-08
content_hash: 4c97955633de89db
---

# MetaphorVU: Towards Metaphorical Video Understanding

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.25461](https://arxiv.org/abs/2605.25461)  
**Code**: To be confirmed  
**Area**: Video Understanding / High-level Cognition  
**Keywords**: Metaphorical Video Understanding, Multimodal Large Language Models (MLLMs), Cross-domain Mapping, Knowledge Graph Enhancement

## TL;DR
This paper proposes the first metaphorical video understanding benchmark, MetaphorVU-Bench (860 videos + 8 metaphor categories), and an enhancement method, MetaphorBoost. By utilizing a metaphor knowledge graph with 54K nodes and 200K edges as an external cognitive scaffold, the study quantitatively reveals that the core bottleneck for MLLMs in metaphorical video tasks is the "absence of cross-domain mapping" rather than visual recognition errors. The optimal model still lags behind humans (83.4) by 17 points.

## Background & Motivation

**Background**: Metaphorical videos are prevalent in social media and public communication, serving as vital media for conveying complex ideas. However, existing MLLM research primarily focuses on literal perception tasks (object recognition, event description), lacking systematic studies on high-level cognitive abilities.

**Limitations of Prior Work**: Current MLLMs struggle to accurately understand metaphorical videos. The state-of-the-art Gemini-3-Pro scores only 63.8 (compared to 83.4 for humans), and many existing reasoning enhancement methods (long CoT, test-time scaling) provide almost no help for metaphor understanding—indicating that the problem is not a lack of "thinking more."

**Key Challenge**: Error analysis reveals that most MLLM failures **do not originate from visual element recognition errors**, but rather from a lack of **cross-domain mapping capability** to link visual elements to underlying concepts—the essence of understanding metaphors.

**Goal**: (1) Construct a systematic metaphorical video understanding benchmark; (2) Diagnose the root causes of current model failures; (3) Design targeted methods to enhance cross-domain mapping.

**Key Insight**: Rather than letting MLLMs perform cross-domain mapping blindly, an external metaphor knowledge graph can serve as a cognitive scaffold to guide models in establishing links from visual elements to metaphorical concepts—transforming what is "unthinkable" into "searchable."

**Core Idea**: Use a metaphor knowledge graph as an external cognitive scaffold for inference-time enhancement, helping MLLMs perform cross-domain mapping more effectively.

## Method

### Overall Architecture
Two main contributions: (1) **MetaphorVU-Bench**—the first systematic metaphorical video understanding benchmark; (2) **MetaphorBoost**—an inference-time enhancement framework based on a metaphor knowledge graph. The former defines the problem and evaluation, while the latter addresses the deficiency in cross-domain mapping.

### Key Designs

1.  **Metaphorical Video Taxonomy**:
    -   **Function**: Systematically defines types of metaphorical videos, providing a theoretical foundation for benchmark construction.
    -   **Mechanism**: Designed 8 metaphor types based on multimodal metaphor theory—Body Language, Atmosphere Language, Cultural Symbol, Naturalistic Symbol, Causal Montage, Analogical Montage, Surreal Narrative, and Performative Narrative.
    -   **Design Motivation**: Different metaphor types correspond to different cognitive difficulties, allowing for fine-grained evaluation of MLLM metaphor understanding capabilities.

2.  **Multi-stage High-Quality Benchmark Construction**:
    -   **Function**: Efficiently filters 860 high-quality metaphorical videos from billions of videos.
    -   **Mechanism**: A four-layer filtering pipeline: (1) Filtering by comment count (> 150) results in 70K; (2) GPT-5 analysis of video info + subtitles + comments to judge metaphor logic results in 16K; (3) Gemini-3-Pro verification reduces candidates to 4K; (4) Final manual filtering results in 860. Annotations use a uniform format constraint (specifying which visual elements convey which metaphorical meanings), with cross-validation by three people to ensure consistency.
    -   **Design Motivation**: Multi-stage processing reduces costs while ensuring quality; uniform formatting and cross-validation ensure evaluation reliability.

3.  **Metaphor Knowledge Graph + Inference-time Enhancement**:
    -   **Function**: Externally enhances MLLM cross-domain mapping capabilities without retraining, ready for out-of-the-box use.
    -   **Mechanism**: Construct a metaphor knowledge graph with 54,687 nodes and 200,268 edges. During inference: (1) MLLMs identify video keywords $\mathcal{K} = \{k_1, \ldots, k_m\}$; (2) Query the graph with max-h-hops to obtain $\mathcal{R} = \text{Top-}z(\bigcup_{i=1}^m \mathcal{N}_\mathcal{G}^h(k_i), \deg(\cdot, \mathcal{K}))$ (selecting the $z$ target nodes with the most connections to keywords); (3) The retrieved metaphorical concepts $\mathcal{R}$ are appended as auxiliary information to the video and caption to generate the metaphorical interpretation $\hat{\tau}, \hat{o} = \text{Generate}(v \oplus t \oplus \mathcal{R})$.
    -   **Design Motivation**: Metaphor understanding requires multi-hop connections (which KGs naturally support); metaphor-specific knowledge is superior to general common sense (proven by ConceptNet comparative experiments).

### Loss & Training
MetaphorBoost is an **inference-time training-free enhancement**, requiring no updates to MLLM parameters. The knowledge graph construction is completed offline via LLM distillation and manual verification.

## Key Experimental Results

### Main Results

| Model | Body L. | Atmosp. | Cultural | Natural | Causal M. | Analog M. | Surreal | Perform. | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Human | 87.8 | 87.5 | 89.1 | 83.8 | 72.0 | 81.5 | 78.1 | 78.0 | **83.4** |
| GPT-5 | 69.9 | 76.3 | 77.4 | 66.6 | 45.0 | 55.4 | 54.9 | 46.1 | 63.7 |
| Gemini-3-Pro | 71.2 | 74.0 | 75.1 | 66.9 | 49.4 | 58.9 | 51.1 | 48.1 | 63.8 |
| Qwen3-VL-8B | 56.0 | 66.1 | 68.8 | 60.8 | 33.2 | 45.0 | 39.3 | 29.2 | 52.0 |
| **MetaphorBoost (Gemini-3-Pro)** | 71.5 | 76.3 | 77.5 | 66.9 | **57.2** | 59.1 | **57.3** | 50.8 | **66.1** |
| **MetaphorBoost (Qwen3-VL-8B)** | **61.8** | **71.0** | **71.8** | 61.3 | 36.7 | 47.1 | **45.7** | 31.5 | **55.9** |

Key observations: (1) All MLLMs perform particularly poorly on Causal Montage and Analogical Montage (45.0 and 55.4)—categories requiring the most cross-domain mapping, confirming the necessity of enhancing mapping; (2) MetaphorBoost consistently improves all models, with the largest gains in types requiring the most cross-domain mapping (Causal Montage +7.8).

### Ablation Study

| Configuration | Average Score | Description |
| :--- | :--- | :--- |
| MetaphorBoost Full | 55.9 | Full model |
| w/o External Enhancement | 53.4 | Direct MLLM query without KG, -2.5 |
| w/o Graph Structure | 54.3 | Raw text retrieval instead of KG, -1.6 |
| w/o Metaphor-oriented | 52.5 | General common sense KG (ConceptNet) instead, -3.4 |

All three key factors are effective—external knowledge compensates for MLLM defects (-2.5), graph structures are more effective than text (-1.6), and metaphor-specific knowledge is superior to general common sense (-3.4).

### Key Findings
-   Improving cross-domain mapping requires fine-grained, structured, and metaphor-specific knowledge—all three characteristics are indispensable.
-   The worse the base model, the larger the improvement (Qwen3-VL-8B +3.8% > Gemini-3-Pro +2.3%)—MetaphorBoost acts as a compensatory enhancement.
-   The best combination (Gemini-3-Pro + MetaphorBoost = 66.1) still lags behind humans (83.4) by 17.3 points—indicating that cross-domain mapping is only a partial bottleneck and significant room for improvement remains.

## Highlights & Insights
-   **Systematic + Complete Benchmark**: The first metaphorical video understanding benchmark combines a theoretical foundation (8 categories), data scale (860 videos), and rigorous quality control (multi-stage screening + cross-validation).
-   **Diagnostic Error Analysis**: Through quantitative decomposition (83% of failures stem from mapping deficiencies rather than recognition errors), the study precisely identifies the problem in MLLM metaphor understanding, providing a clear direction for subsequent improvements.
-   **Effectiveness of Inference-time Enhancement**: Achieves stable improvements across any MLLM without retraining; the multi-hop characteristics of knowledge graphs are more effective than flat text.
-   **Value of Metaphorical Knowledge**: Ablations clearly prove a 3.4-point advantage of metaphor-specific knowledge over general common sense, inspiring the use of domain-specific knowledge bases for high-level cognitive tasks.

## Limitations & Future Work
-   Knowledge Graph Coverage: 54K nodes may insufficient for novel metaphors.
-   Keyword Recognition Accuracy: The first step of MetaphorBoost relies on MLLMs; recognition errors contaminate subsequent queries.
-   Model Scale Constraints: Only assessed MLLMs $\leq$ 235B; the performance of larger-scale models remains unknown.
-   The optimal combination (66.1) is still 17.3 points behind humans (83.4), suggesting exploration of deeper multi-hop reasoning or visual-concept alignment.

## Related Work & Insights
-   **vs. Advertising Metaphor Work** (Kalarani 2024, Long 2025): These focus on specific domains (advertising), while this work offers a systematic classification and multi-source data coverage for a broader range, along with fine-grained error diagnosis.
-   **vs. MMR-V** (Zhu 2026): MMR-V evaluates a broad spectrum of reasoning abilities (metaphor being just one), whereas this work focuses deeply on metaphors to provide more detailed analysis.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ The first systematic metaphorical video understanding benchmark + diagnostic analysis + knowledge enhancement method, a three-pronged approach to solving MLLM high-level cognitive bottlenecks.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 MLLMs + 5 reasoning enhancement methods + detailed error analysis + multi-angle ablations, covering both open/closed source and large/small models.
-   Writing Quality: ⭐⭐⭐⭐ Logically clear, progressing deeply from problem diagnosis to solution design and comparative validation; finely designed ablation experiments.
-   Value: ⭐⭐⭐⭐⭐ The systematic benchmark fills a research gap; diagnostic results provide direct guidance for MLLM improvement; knowledge enhancement ideas are transferable to other high-level cognitive tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[ICML 2026\] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking](videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t.md)
- [\[ICML 2026\] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

</div>

<!-- RELATED:END -->
