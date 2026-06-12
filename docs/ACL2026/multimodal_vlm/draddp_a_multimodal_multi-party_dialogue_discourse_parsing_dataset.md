---
title: >-
  [Paper Note] DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset
description: >-
  [ACL 2026][Multimodal VLM][Multi-party dialogue] DraDDP constructs the first publicly available English multimodal multi-party dialogue discourse parsing dataset and systematically evaluates the distinct contributions of…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multi-party dialogue"
  - "Discourse parsing"
  - "Multimodal dataset"
  - "Audio-visual cues"
  - "SDRT"
date: 2026-05-08
content_hash: 0254801ef5f4e408
---

# DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset

**Conference**: ACL 2026  
**arXiv**: [2606.00012](https://arxiv.org/abs/2606.00012)  
**Code**: https://github.com/DraDDP  
**Area**: Multimodal Dialogue Understanding / Discourse Parsing  
**Keywords**: Multi-party dialogue, Discourse parsing, Multimodal dataset, Audio-visual cues, SDRT

## TL;DR
DraDDP constructs the first publicly available English multimodal multi-party dialogue discourse parsing dataset and systematically evaluates the distinct contributions of text, audio, and video cues to the identification of dependency edges and discourse relations using traditional parsers, LLMs, and MLLM systems.

## Background & Motivation
**Background**: Multi-party dialogue (MPD) discourse parsing aims to identify dependency structures and relation types (e.g., Comment, Background, Question-Answer Pair) between Elementary Discourse Units (EDUs). Previous mainstream datasets and methods (e.g., STAC, Molweni, DialogueDSA, MSDC) primarily focus on text, relying on BERT, structural Transformers, incremental LLaMA parsers, or specialized multi-task learning frameworks.

**Limitations of Prior Work**: Real-world MPDs do not rely solely on text. Interactions involve parallel topics, gaze shifts, tonal variations, and scene actions. Relying only on text often leads to mislinking irrelevant responses to the wrong context. Existing multimodal discourse parsing resources are insufficient: JDDC 2.1 and MODDP are biased toward dyadic dialogues and are primarily Chinese-based, failing to support English multi-party multimodal research.

**Key Challenge**: Multi-party dialogues increase topic branching and long-distance dependencies, while multimodal information may provide both critical cues and scene noise. The research community lacks a benchmark that simultaneously features English, multi-party interactions, audio-visual synchronization, and human-annotated discourse structures to systematically determine the utility of different modalities.

**Goal**: The authors aim to fill this data gap and establish a reproducible experimental benchmark. This involves constructing an English MPD discourse parsing dataset containing text, video, and audio while evaluating Link-F1 and Link&Rel-F1 using various models to analyze the roles of audio and video across different speaker counts and relation types.

**Key Insight**: The paper selects the first season of the TV series *Friends* as the data source due to its stable subtitles, timestamps, rich multi-party interactions, emotional expressions, physical movements, and scene transitions. This choice allows for the collection of dialogue structures closer to real face-to-face communication while ensuring alignment quality.

**Core Idea**: Construct an aligned text-video-audio discourse parsing dataset from timestamped sitcom MPDs and decompose the question of "whether multimodality is useful" into quantifiable aspects such as dependency edges, relation types, speaker counts, and modality combinations through a systematic benchmark.

## Method
DraDDP is essentially a dataset and benchmark paper. Its technical contribution lies in decomposing multimodal MPD discourse parsing into a process of annotation, training, and comparison: extracting EDUs from subtitles, annotating dependency graphs and 16 categories of discourse relations via the SDRT framework, and comparing text, audio, video, and their combinations under a unified evaluation protocol.

### Overall Architecture
The pipeline consists of four steps. First, data preparation: extracting dialogue segments from 24 episodes of *Friends* Season 1, using subtitle lines as EDUs, and aligning them with video frames and audio clips using timestamps. Second, human annotation: labeling parent nodes and relation types for each utterance based on the SDRT system. Third, quality control: using a pre-labeling model as assistance while maintaining consistency through multi-person annotation, discussion, and third-party arbitration. Fourth, benchmarking: evaluating RLTST, BERTLine, MODDP, LLaMIPa, and the Qwen series (text/audio/video/omni models) on DraDDP and MODDP.

### Key Designs
1. **Multimodal Data Construction for Multi-party Interactions**:
    - **Function**: Converts multi-party dialogues from English TV series into synchronized text, video, and audio discourse parsing samples.
    - **Mechanism**: Subtitle lines are treated as EDUs because of their moderate length, correspondence to conversational turns, and timestamp availability for alignment. The final dataset includes 495 dialogue segments, 6,374 utterances, and 9.1 hours of parallel video.
    - **Design Motivation**: Forum or game text data lack interactive cues like facial expressions, gaze, and tone. TV dialogues provide stable, dense, and aligned samples of multi-party interactions.

2. **Four-stage Human Annotation with Pre-labeling Support**:
    - **Function**: Obtains reliable discourse dependency edges and relation types in complex multi-party scenarios.
    - **Mechanism**: LLaMA3 fine-tuned on STAC is used for text-based pre-labeling, followed by correction from PhD and Master students watching the video. The process includes collaborative labeling of 1/6 of the data for standardization, independent labeling of 1/3 with conflict discussion, and final arbitration by a third party.
    - **Design Motivation**: Pure human annotation is costly, while total reliance on models introduces bias. Pre-labeling achieved 72.69% Link-F1, significantly reducing the workload for short-distance dependencies.

3. **Benchmark Decomposed by Modality and Speaker Count**:
    - **Function**: Evaluates the actual benefits of multimodal information in identifying dependency edges and relation types.
    - **Mechanism**: The study uses micro F1, where Link-F1 requires correct edges and Link&Rel-F1 requires both correct edges and relations. Qwen-series models (Qwen2.5, Qwen2.5-VL, Qwen2-Audio, Qwen2.5-Omni) are used to test combinations of T, V, and A.
    - **Design Motivation**: Multimodality is not universally beneficial. Video may capture gaze but also introduce noise. Analyzing results by speaker count and relation type reveals the specific conditions under which modalities contribute.

### Loss & Training
The paper does not propose a new loss function; training strategies follow established baselines. LLM-related models use LoRA via LLaMA-Factory with rank 8 and a scaling factor of 16. The AdamW learning rate is $1\times10^{-4}$, batch size is 1 per GPU with 8 gradient accumulation steps over 3 epochs. Video is sampled at 1 fps (max 16 frames), and audio is converted to 16 kHz, 80-channel Mel spectrograms. Checkpoints are selected based on the best Link&Rel-F1 on the development set.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | Key Results (Ours) | Comparison Target | Note |
|--------|------|------|----------|------|
| DraDDP Scale | Dialogues / Utterances / Video | 495 / 6,374 / 9.1h | MODDP: 864 / 18K / CN Dyadic | DraDDP is smaller but covers English MPD and T+V+A |
| DraDDP | Link-F1 / Link&Rel-F1 | LLaMIPa†: 85.03 / 54.58 | LLaMIPa: 84.71 / 53.39 | Relation F1 improved by 1.19 without history structure concatenation |
| DraDDP | Link-F1 / Link&Rel-F1 | Qwen2-Audio: 84.90 / 55.09 | Qwen2.5 text: 84.14 / 53.55 | Audio brings 1.54 Link&Rel-F1 gain |
| MODDP | Link-F1 / Link&Rel-F1 | Qwen2-Audio: 92.43 / 54.88 | Qwen2.5 text: 91.26 / 52.82 | Audio also yields 2.06 gain on Chinese dyadic data |
| DraDDP | Link-F1 / Link&Rel-F1 | Qwen2.5-Omni: 84.55 / 53.34 | Qwen2-Audio: 84.90 / 55.09 | Omni-modal fusion underperforms T+A, indicating video noise offsets gains |

### Ablation Study
| Configuration | Link-F1 | Link&Rel-F1 | Note |
|------|---------|------|------|
| T | 84.67 | 53.69 | Text is the strongest single-modality base |
| V | 43.38 | 22.21 | Visual info alone is insufficient for discourse parsing |
| A | 47.39 | 38.83 | Pure audio is closer to relation discrimination than pure visual |
| T+V | 83.61 | 52.97 | Video addition performs worse than pure text |
| T+A | 84.83 | 54.76 | Optimal dual-modality combination (+1.07 Rel-F1 over text) |
| V+A | 50.12 | 40.39 | Difficult to parse dependency structures without text |
| T+A+V | 84.55 | 53.34 | Tri-modal fusion is affected by visual noise |

### Key Findings
- The multi-party nature of DraDDP significantly increases difficulty: the Qwen2.5 text model's Link-F1 on DraDDP is 7.12 lower than on MODDP.
- Audio becomes more important as the number of speakers increases. In scenarios with $s>6$, Qwen2-Audio improves Link-F1 by 7.69 and Link&Rel-F1 by 5.77 compared to the text model.
- Video is better suited for dyadic or few-speaker scenarios. At $s\leq2$, Qwen2.5-VL's Link&Rel-F1 is 2.08 higher than the text model; however, background and motion noise interfere in complex MPD scenarios.
- Error analysis shows audio significantly reduces confusion related to emotions and questions (e.g., `{Comt -> Clafi}` errors reduced by 71.4%, `{QAP -> Comt}` by 75%).

## Highlights & Insights
- The dataset is clearly positioned to fill the gap of "English + Multi-party + T/V/A + Discourse Structure." There were previously almost no public resources at this intersection.
- The most valuable finding is that "multimodality is conditionally useful." Audio is more reliable in complex multi-party interactions, while video is more focused in dyadic exchanges. Omni-modal fusion may hurt recognition due to noise.
- The use of pre-labeling is restrained; LLaMA3 output was used only to reduce repetitive work for short-distance dependencies, while final decisions were human-made based on audio-visual context.
- For future tasks, multimodal dialogue models should not simply concatenate all modalities but should dynamically weight them based on speaker count, relation types, and noise levels.

## Limitations & Future Work
- The data scale remains small (495 segments, 6,374 utterances), which is insufficient for training large models, leading to long-tail issues for rare relation types.
- The single-source sitcom data might carry specific humor styles and cultural biases, which may not generalize to meetings or natural social scenarios.
- Coarse video processing (1 fps, max 16 frames) struggles to capture micro-expressions or the flow of gaze, potentially underestimating the potential of visual information.
- Future work should explore modal gating, speaker tracking, and temporal visual encoding specifically for discourse relations.

## Related Work & Insights
- **vs STAC / Molweni**: These provide MPD text discourse resources; DraDDP adds A/V cues and utilizes face-to-face interactions to study non-verbal signals.
- **vs MODDP**: While MODDP is a Chinese dyadic multimodal dataset, DraDDP focuses on English and multi-party scenarios with more topic branching.
- **vs LLaMIPa**: LLaMIPa is an incremental parser; the authors found that its history structure concatenation could propagate early errors, suggesting a need for better confidence control in MPD parsing.
- **Insights for MLLMs**: General MLLMs might perceive images and audio but do not necessarily understand "who is responding to whom and why." This dataset serves as a diagnostic benchmark for fine-grained interaction understanding.

## Rating
- Novelty: ⭐⭐⭐⭐☆ First public English multimodal MPD discourse dataset; new task-data combination.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive benchmarks and ablation; limited by data scale and single source.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and process; high information density in tables.
- Value: ⭐⭐⭐⭐⭐ Highly valuable resource for multimodal dialogue understanding, meeting parsing, and MLLM diagnostics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance](guidedog_a_real-world_egocentric_multimodal_dataset_for_blind_and_low-vision_acc.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](../../ICCV2025/multimodal_vlm/toolvqa_a_dataset_for_multi-step_reasoning_vqa_with_external_tools.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](../../CVPR2026/multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)
- [\[AAAI 2026\] Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Reports](../../AAAI2026/multimodal_vlm/pharos-esg_a_framework_for_multimodal_parsing_contextual_narration_and_hierarchi.md)

</div>

<!-- RELATED:END -->
