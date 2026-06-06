---
title: >-
  [Paper Note] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding
description: >-
  [ACL2026][Video Understanding][Streaming Video Understanding] Response-G1 uses query-guided online scene graphs, historical scene graph retrieval…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "Streaming Video Understanding"
  - "Proactive Responding"
  - "Scene Graph"
  - "Retrieval-Augmented Generation"
  - "Video-LLM"
date: 2026-05-08
content_hash: 624f9e5145a6ca16
---

# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding

**Conference**: ACL2026  
**arXiv**: [2605.07575](https://arxiv.org/abs/2605.07575)  
**Code**: https://github.com/kadmkbl/Response-G1  
**Area**: video_understanding  
**Keywords**: Streaming Video Understanding, Proactive Responding, Scene Graph, Retrieval-Augmented Generation, Video-LLM

## TL;DR
Response-G1 uses query-guided online scene graphs, historical scene graph retrieval, and timestamped trigger prompts to explicitly align visual evidence with the response conditions of user queries, significantly improving the ability of Video-LLMs to judge "whether to answer now" without fine-tuning.

## Background & Motivation
**Background**: Video-LLMs are already capable of video question answering and long-video understanding. Streaming Video Understanding further requires models to perform incremental perception, reasoning, and interaction as video continuously arrives. Most existing systems remain reactive: the user asks at a specific time, and the model immediately answers based on observed segments.

**Limitations of Prior Work**: Many real-world interactions are anticipatory, such as "tell me when someone starts doing X." In such cases, the answer conditions may not yet have appeared when the question is asked; the model must remain silent and wait for the evidence to meet the conditions before responding. Existing proactive methods either train EOS/binary trigger heads, which rely on fine-grained frame-level annotations, or use frame-difference thresholds or multi-agent prompts, which easily ignore query semantics.

**Key Challenge**: The key to proactive responses is not "whether the video has changed," but "whether the currently accumulated evidence satisfies the response conditions implicit in the query." If both visual evidence and query conditions exist only implicitly in Video-LLM hidden states or prompts, it is difficult for the model to stably align them or explain why it triggered at a specific moment.

**Goal**: The authors aim to design a framework that requires no fine-tuning or frame-level labels, allowing Video-LLMs to explicitly model query-related evidence in streaming videos, retrieve historical evidence, and decide on silence/response accordingly.

**Key Insight**: User queries typically describe a target scene composed of objects, attributes, and relations (e.g., "a boy in red is talking to someone"). This can naturally be represented as a scene graph. If video segments are also converted into scene graphs, evidence-condition matching can be performed in the same structural space.

**Core Idea**: Convert both streaming video evidence and query conditions into scene graphs. Use top-$K$ scene graph retrieval to feed the most relevant historical evidence back to the Video-LLM, followed by trigger prompts for frame-by-frame response timing decisions.

## Method
Response-G1 is a fine-tuning-free pipeline. Instead of training a new trigger classifier, it uses the Video-LLM itself as the scene graph generator, text encoder, and final decision-maker. The key is compressing long video history into structured, query-related, and retrievable graph memory rather than stuffing all visual tokens into the context.

### Overall Architecture
Input consists of streaming video frames and a user query. Output includes a `silence` or `response` decision at each timestep and the final natural language answer upon triggering. The framework comprises three steps: query-guided scene graph generation for the current clip; storing historical scene graphs in a memory bank and calculating similarity with the query condition graph to retrieve top-$K$ segments; and inputting visual tokens, timestamped retrieved scene graphs, and trigger instructions into the Video-LLM to decide whether to respond.

### Key Designs
1. **Query-guided Online Scene Graph Generation**:
    - **Function**: Converts each streaming clip into structured evidence retaining only query-related details.
    - **Mechanism**: For a video clip $C_t$ near time $t$, the Video-LLM generates a scene graph $G_t=(O_t,P_t)$ based on the original query $Q$, where nodes are objects/attributes and edges are spatio-temporal relations. The graph is represented as a set of triplets $G_t=\{(o_i,p_{ij},o_j)\}$. The query is injected into the prompt to prioritize describing objects and relations relevant to the trigger condition.
    - **Design Motivation**: Scene graphs without query guidance generate excessive irrelevant triplets, increasing retrieval noise. Direct injection of target objects might induce hallucination. Using the original query as soft guidance balances relevance and truthfulness.

2. **Memory-based Scene Graph Retrieval**:
    - **Function**: Identifies evidence from historical video clips that best supports the current response condition.
    - **Mechanism**: Each triplet is linearized into a natural language phrase, and the full graph is represented as a concatenation of phrases $\Phi_t$. The query is parsed into a query condition graph $G_q$ and linearized into $\Phi_q$ to maintain format consistency. The Video-LLM text encoder performs mean pooling to obtain graph vectors $g_t$ and query vectors $g_q$, followed by cosine similarity retrieval of the top-$K$ scene graphs.
    - **Design Motivation**: Using raw query text and video graphs directly leads to format inconsistency; treating both as graph text ensures retrieval focuses on matching object relations.

3. **Timestamped Retrieval-Augmented Trigger Decision**:
    - **Function**: Enables the model to judge if sufficient evidence exists to answer at each timestep.
    - **Mechanism**: Retrieved scene graphs are augmented with textual timestamps (e.g., `<2.0s>`) and encoded into the context. The trigger phase input includes visual frame embeddings, retrieved scene graph context, and an instruction like "Should I answer now? Yes or No." If "silence" is output, it proceeds to the next frame; if "response," it generates the final answer using the same context.
    - **Design Motivation**: Proactive response requires knowing not just if the target relationship appeared, but the temporal order and sufficiency of evidence. Timestamps turn graph memory into an evidence chain for temporal grounding.

### Loss & Training
Response-G1 involves no parameter training; it is an inference-time pipeline. Experiments use Qwen3-VL-8B as the Video-LLM backbone. OVO-Bench utilizes 1 FPS. StreamingBench follows official rules: 1 FPS for short videos, 0.5 FPS for medium, and 0.2 FPS for long videos. All experiments run on A100 80GB in FP16. Latency analysis shows the original version takes ~825ms/frame (1.2 FPS), while using streaming KV-Cache reduces it to ~473ms (2.1 FPS).

## Key Experimental Results

### Main Results
On OVO-Bench, Response-G1's advantage over open-source streaming Video-LLMs is concentrated in Forward Active Responding (the core proactive capability). While overall scores do not surpass closed-source models like Gemini 1.5 Pro, it leads significantly among open-source streaming models.

| Model | Params | Real-Time Visual Perception Avg↑ | Backward Tracing Avg↑ | Forward Active Responding Avg↑ | Overall Avg↑ |
|------|------|----------------------------------|-----------------------|--------------------------------|--------------|
| GPT-4o | - | 63.6 | 58.7 | 53.4 | 58.6 |
| Gemini 1.5 Pro | - | 70.8 | 62.3 | 57.2 | 65.3 |
| TimeChat-Online | 7B | 58.6 | 42.0 | 36.4 | 45.6 |
| StreamAgent | 7B | 61.3 | 41.7 | 45.4 | 49.4 |
| Response-G1 | 8B | 73.6 | 52.1 | 58.2 | 61.3 |

On StreamingBench, Response-G1 achieves the highest Overall score among open-source models and improves the proactive output (PO) from ~29 to 44.

| Model | Params | Real-Time Visual Understanding Avg↑ | PO↑ | Overall Avg↑ | Note |
|------|------|-------------------------------------|-----|--------------|------|
| GPT-4o | - | 73.3 | 56.9 | 70.5 | Strong closed-source baseline |
| LLaVA-OneVision | 7B | 71.1 | 29.6 | 66.3 | Strong open-source Video-LLM |
| TimeChat-Online | 7B | 75.4 | 28.8 | 70.9 | Open-source streaming baseline |
| StreamAgent | 7B | 74.3 | 28.9 | 70.2 | Multi-agent prompt baseline |
| Response-G1 | 8B | 77.5 | 44.0 | 73.7 | Highest overall/PO among open-source |

### Ablation Study
Both retrieval augmentation and timestamps are effective. Removing retrieval causes drops in both proactive and reactive tasks. Removing timestamp encoding impacts tasks requiring temporal localization (e.g., CRR/PO) more severely.

| Config | OVO ACR↑ | OVO HLD↑ | OVO CRR↑ | Streaming CS↑ | Streaming PR↑ | Streaming PO↑ |
|------|----------|----------|----------|---------------|---------------|---------------|
| w/o Retrieval Augmentation | 66.1 | 28.0 | 55.4 | 83.6 | 79.6 | 36.8 |
| w/o Timestamp Encoding | 74.0 | 33.6 | 60.4 | 87.7 | 82.9 | 43.6 |
| Full | 74.3 | 33.9 | 61.7 | 88.0 | 83.3 | 44.0 |

Query guidance is critical. Directly inserting parsed object relations into the prompt increases relevance but risks inducing the model to "see" targets before they appear.

| SGG Strategy | Streaming PO↑ | OVO REC↑ | OVO SSR↑ | OVO CRR↑ | Explanation |
|----------------|---------------|----------|----------|----------|------|
| w/o Guidance | 38.8 | 34.1 | 66.9 | 59.4 | Generates many irrelevant triplets |
| Object-Guidance | 43.6 | 40.2 | 67.9 | 61.3 | More relevant, but hallucination risk |
| Query-Guidance | 44.0 | 41.9 | 71.1 | 61.7 | Best balance of relevance/factuality |

### Key Findings
- Explicit structured evidence is particularly useful for proactive timing. Response-G1 shows the most significant gains in OVO FAR and StreamingBench PO, proving scene graphs help judge condition fulfillment.
- Retrieval is not just for context compression; it reorders long video history based on query semantics, ensuring trigger decisions see the most relevant evidence.
- Format consistency for graph text is vital. Retrieval based on graph-text representations outperforms raw query text.
- KV-Cache brings the method closer to real-time deployment (from 1.2 FPS to 2.1 FPS).

## Highlights & Insights
- A key strength is reformulating proactive response timing as evidence-condition alignment rather than a vague judgment call.
- Scene graphs here act as open-vocabulary structured memory generated by the Video-LLM itself, trading some rigor for generic applicability without detectors or fine-tuning.
- Query-guided SGG provides a proper balance; soft guidance from the original query works best to avoid both noise and hallucination.

## Limitations & Future Work
- Scene graph representations cannot cover all reasoning needs, especially "why-style" questions or causal explanations.
- Fixed clip sizes may split semantic events. Event-level triggers or semantic change detection could be explored.
- LLM-based open-set SGG remains prone to hallucinations.
- The method depends on the text encoding and prompt-following capabilities of the specific Video-LLM used.
- Currently validated at low FPS; latency and safety are insufficient for high-frequency robot control or autonomous driving.

## Related Work & Insights
- **vs VideoLLM-online / Flash-Vstream**: These focus on token efficiency; Response-G1 focuses on proactive timing using structured graph memory.
- **vs Dispider / StreamBridge**: These train activation models; Response-G1 relies on prompts and retrieval without frame-level trigger labels.
- **vs StreamAgent**: StreamAgent uses multi-agent prompting; Response-G1 provides better interpretability and PO gains by explicitly aligning evidence and conditions via scene graphs.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/video_understanding/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)

</div>

<!-- RELATED:END -->
