---
title: >-
  [Paper Note] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model
description: >-
  [NeurIPS 2025][LLM/NLP][Streaming Video Understanding] StreamBridge proposes a simple and generalizable framework that enables multi-turn streaming interaction via a memory buffer with round-decayed compression, and achieves proactive response through a decoupled lightweight activation model. Combined with the purpose-built Stream-IT dataset, it successfully converts offline Video-LLMs (e.g., Qwen2-VL, LLaVA-OV) into streaming assistants, surpassing GPT-4o and Gemini 1.5 Pro on OVO-Bench and Streaming-Bench.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Streaming Video Understanding
  - Video-LLM
  - Proactive Response
  - Multi-turn Interaction
  - Activation Model
date: 2026-05-08
content_hash: 1bdf23bf52367e24
---

# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model

**Conference**: NeurIPS 2025
**arXiv**: [2505.05467](https://arxiv.org/abs/2505.05467)
**Code**: N/A
**Area**: Video Understanding / Streaming Video LLM
**Keywords**: Streaming Video Understanding, Video-LLM, Proactive Response, Multi-turn Interaction, Activation Model

## TL;DR

StreamBridge proposes a simple and generalizable framework that enables multi-turn streaming interaction via a memory buffer with round-decayed compression, and achieves proactive response through a decoupled lightweight activation model. Combined with the purpose-built Stream-IT dataset, it successfully converts offline Video-LLMs (e.g., Qwen2-VL, LLaVA-OV) into streaming assistants, surpassing GPT-4o and Gemini 1.5 Pro on OVO-Bench and Streaming-Bench.

## Background & Motivation

Current Video-LLMs typically process pre-recorded complete videos, whereas emerging applications (robotics, autonomous driving, etc.) require **online causal perception** capabilities. Adapting offline models to streaming settings poses two major challenges:

**Multi-turn real-time understanding**: Users issue queries at different time points, requiring the model to attend to the latest video segments while retaining historical context.

**Proactive response mechanism**: The model must continuously monitor the visual stream and proactively generate outputs at appropriate moments, rather than passively awaiting user queries.

Limitations of existing approaches:
- Streaming models trained from scratch (e.g., VideoLLM-Online) perform poorly on offline tasks.
- Embedding the activation mechanism into the main model leads to optimization conflicts and probability calibration issues.
- Existing benchmarks reduce multi-turn streaming to independent single-turn offline tasks, discarding historical context.

Core Idea: **Rather than retraining, enhance a pretrained offline Video-LLM into a streaming model at minimal cost — via memory management, compression strategy, and a decoupled activation model.**

## Method

### Overall Architecture

StreamBridge consists of three plug-and-play components:
1. Memory Buffer → stores accumulated visual-text embeddings
2. Round-Decayed Compression → controls input length
3. Activation Model → determines when to respond

### Key Designs

1. **Memory Buffer**:

    - Adopts a producer-consumer paradigm: the encoder continuously produces frame features, which the LLM consumes on demand.
    - Each frame is independently encoded and appended to the buffer along with associated query embeddings.
    - After a response is generated, the response embeddings are also appended to the buffer, preserving the complete multi-turn interaction history.
    - At inference time, buffer contents are flattened into a single sequence and fed into the LLM.

2. **Round-Decayed Compression**:

    - A maximum embedding length MaxLen is predefined.
    - When input exceeds MaxLen, visual tokens from the earliest dialogue rounds are merged via average pooling, one frame at a time.
    - Core Idea: **Coarse retention for distant history, fine-grained preservation of recent context.**
    - Ensures real-time response accuracy while avoiding complete loss of historical visual context.
    - Significantly reduces memory usage and inference latency.

3. **Plug-and-play Activation Model**:

    - Uses an independent small MLLM (e.g., LLaVA-OV-0.5B) as the activation judge.
    - Architecture modification: replaces the language modeling head with a score head for binary classification and introduces a learnable `<ACT>` activation token.
    - Input format: `<Q> <V1> <A1> <V2> <A2> ...`, with aggressive pooling on visual tokens for efficiency.
    - Training data: collected from dense video captioning, sequential step recognition, foundational video QA, and other temporally annotated datasets.
    - Annotation strategy: only the last $P\%$ of frames in each video segment are labeled as positive samples ($P$ is dynamically sampled from 0% to 50%).
    - At inference: when the predicted score exceeds threshold $\alpha$, the main LLM is triggered to respond.
    - Key advantage: runs fully in parallel with the main LLM without interfering with language generation.

4. **Stream-IT Dataset**:

    - **Proactive understanding data**: Collects publicly available datasets including dense video captioning, sequential step recognition, and foundational VideoQA, unified into an interleaved `<Q> <V1> <A1> <V2> <A2>` format.
    - **StreamingQA-120K**: Filters 1.28 million short clips from WebVid-10M, Panda-70M, and InternVid-10M, concatenates them into long videos (avg. 150s+) by semantic similarity, and uses GPT-4o to generate multi-turn QA across 8 task types.
    - Data augmentation: Random QA Drop ($P_{\text{drop}} = 0.55$) prevents overfitting to fixed QA positions; QA Interval Shift ($P_{\text{shift}} = 0.1$) simulates proactive response scenarios.

### Loss & Training

- Main Video-LLM: standard next-token prediction loss, fine-tuned on Stream-IT combined with ~600K samples (LLaVA-178K, etc.).
- Activation model: binary cross-entropy loss.
- Video sampling rate: 1 FPS.
- Supports three backbone models: LLaVA-OV-7B, Qwen2-VL-7B, and Oryx-1.5-7B.

## Key Experimental Results

### Main Results

| Benchmark / Model | Metric | Qwen2-VL†+Stream-IT | GPT-4o | Gemini 1.5 Pro |
|---|---|---|---|---|
| OVO-Bench (multi-turn streaming) | Avg | **71.30** | 64.46 | 69.32 |
| Streaming-Bench (multi-turn streaming) | Avg | **77.04** | 73.28 | 75.69 |

| Video-LLM Backbone | OVO Original → StreamBridge → +Stream-IT |
|---|---|
| Qwen2-VL-7B | 55.98 → 63.35 → **71.30** (+15.32) |
| Oryx-1.5-7B | 59.25 → 59.25 → **71.17** (+11.92) |
| LLaVA-OV-7B | 61.64 → 61.64 → **69.93** (+8.29) |

### Ablation Study

| Configuration | OVO-Bench | Notes |
|---|---|---|
| Single-turn offline evaluation | ~63 | Original simplified evaluation protocol |
| Multi-turn streaming (w/o Stream-IT) | ~63 | Framework effective but without training |
| Multi-turn streaming + Stream-IT | **71.30** | Synergistic gain from data and framework |

### Key Findings

- The StreamBridge framework alone (without Stream-IT fine-tuning) already benefits certain models: Qwen2-VL improves from 55.98 to 63.35, suggesting its interleaved multimodal pretraining makes it naturally amenable to streaming inputs.
- LLaVA-OV shows a slight performance drop under streaming settings (64.02 → 61.64), as interleaved sequences are less represented in its pretraining data.
- Gains from Stream-IT fine-tuning are consistent across all three backbone models.
- Performance on general video benchmarks improves rather than degrades: Oryx-1.5 increases from 58.8 to 65.5 (+6.7) on VideoMME.
- The decoupled activation model design preserves the main LLM's language fluency.

## Highlights & Insights

- **Minimalist framework design**: All three components — memory buffer, compression, and activation model — are plug-and-play and require no modification to the backbone architecture.
- **Decoupled activation model**: Separating "when to respond" from "how to respond" avoids optimization conflicts and allows independent upgrading.
- **Round-decayed compression**: The coarse-to-fine information retention strategy aligns with the temporal attention distribution in video understanding.
- **Stream-IT construction pipeline**: The approach of concatenating short clips and generating QA with GPT-4o is highly scalable.
- **Generalizability**: Successful adaptation across three Video-LLMs with distinct architectures validates the broad applicability of the method.

## Limitations & Future Work

- Multi-turn QA in Stream-IT is generated by GPT-4o, which may introduce quality inconsistencies and hallucinations.
- A 1 FPS sampling rate may be insufficient for rapidly changing scenes.
- The activation threshold $\alpha$ requires manual tuning and may need adjustment across different scenarios.
- The compression strategy uses simple average pooling, which may discard important visual details.
- End-to-end performance has not been validated in real-time systems such as robotics or autonomous driving.

## Related Work & Insights

- **vs. VideoLLM-Online**: VideoLLM-Online introduces specialized online objectives and trains from scratch, resulting in poor offline task performance; StreamBridge leverages existing model capabilities with minimal adaptation.
- **vs. MMDuet**: MMDuet adds dedicated heads to the main model for proactive response, potentially interfering with language capability; StreamBridge's decoupled design is safer.
- **vs. Flash-VStream**: Flash-VStream designs a dedicated memory architecture, whereas StreamBridge directly reuses the existing model's KV-cache mechanism.
- **vs. Dispider**: Dispider also operates in a streaming setting but achieves lower performance; StreamBridge attains larger gains through the synergy of data and framework design.

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupled activation model design is creative, though the overall approach leans toward engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three backbone models with both streaming and offline evaluations, detailed ablations, and comprehensive dataset analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is precise, method description is clear, and pseudocode is complete.
- Value: ⭐⭐⭐⭐⭐ Provides a practical solution for converting offline Video-LLMs into streaming models; the Stream-IT dataset is directly usable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](../../ICCV2025/llm_nlp/vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](../../ACL2026/llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](../../ACL2026/llm_nlp/the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)

</div>

<!-- RELATED:END -->
