---
title: >-
  [Paper Note] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models
description: >-
  [ACL2026][Audio & Speech][Omni-modal Large Language Models] This paper identifies that end-to-end omni-modal large language models (OLLMs) tend to miscopy slide text as speech content when performing SlideASR. It propose…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Omni-modal Large Language Models"
  - "Slide-Enhanced Speech Recognition"
  - "Reinforcement Learning"
  - "Visual Anchoring"
  - "Contextual ASR"
date: 2026-05-08
content_hash: b0c75ee8a2c66db0
---

# VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models

**Conference**: ACL2026  
**arXiv**: [2510.08618](https://arxiv.org/abs/2510.08618)  
**Code**: https://github.com/isruihu/SlideASR-Bench  
**Area**: reinforcement_learning  
**Keywords**: Omni-modal Large Language Models, Slide-Enhanced Speech Recognition, Reinforcement Learning, Visual Anchoring, Contextual ASR

## TL;DR
This paper identifies that end-to-end omni-modal large language models (OLLMs) tend to miscopy slide text as speech content when performing SlideASR. It proposes VAPO, which utilizes a "Look-then-Listen" structured reasoning chain and multi-objective reinforcement learning to transform slide text into semantic anchors for speech recognition rather than sources of interference.

## Background & Motivation
**Background**: Traditional ASR is already robust for general speech but remains prone to missing domain terminology, rare entities, and proper nouns in academic reports, technical demonstrations, and professional lectures. Slides typically contain these keywords; hence, SlideASR aims to use slide images as visual context to improve speech transcription.

**Limitations of Prior Work**: Current mainstream SlideASR methods mostly employ a pipeline: first using OCR to extract slide text, then selecting keywords, and finally feeding these texts as context to an audio language model. Such cascaded systems have many modules, are complex to implement, and errors in OCR or keyword selection propagate through the system. While OLLMs seem capable of processing images, audio, and text simultaneously to perform end-to-end SlideASR naturally, challenges remain.

**Key Challenge**: End-to-end OLLMs do not automatically integrate modalities reasonably. The authors discovered a phenomenon termed visual interference: when prominent text exists on a slide, models bias towards the visual text, even outputting slide words that were never spoken as part of the transcript. In other words, visual context, which should assist in identifying proper nouns, instead suppresses the auditory signal.

**Goal**: The paper aims to establish a truly end-to-end SlideASR paradigm that takes audio, slide images, and instructions as direct inputs while explicitly distinguishing between visual perception and auditory transcription during the reasoning process to avoid "copying whatever is seen."

**Key Insight**: The authors draw inspiration from human habits when attending presentations: humans typically scan the slide first to form thematic and entity priors, then listen to the speech to align heard words with visual priors. VAPO explicitly formulates this process as a "Look-then-Listen" chain and uses reinforcement learning to reward the model for adhering to it.

**Core Idea**: Use `<think>` to extract visual priors from the slides and `<answer>` to generate the speech transcript, optimizing this structured strategy through four rewards: format, OCR, ASR, and visual anchoring.

## Method
The core of VAPO is not simply making the model "look" at an image, but changing the temporal sequence of how it utilizes the image. When a vanilla OLLM receives images and audio simultaneously, strong visual text may dominate during decoding, leading the model to recite slide text. VAPO forces the model to write visual content into `<think>` first, then generate the transcript in `<answer>` based on the audio. Thus, the visual signal becomes a referable intermediate memory rather than competing with the audio signal at the same moment.

The paper defines and quantifies visual interference. Given the set of slide words $V_{slide}$ and the set of ground-truth audio words $V_{audio}$, words appearing only in the slide but not the audio are identified as $V_{exclusive}=V_{slide}\setminus V_{audio}$. If these words appear in the model prediction $V_{pred}$, the sample is considered interfered. The Visual Interference Rate (VIR) is the proportion of such samples. This metric aligns closely with the task essence, as it measures the specific failure mode where "visual text overwhelms audio" rather than just looking at WER.

### Overall Architecture
The input to VAPO includes audio, corresponding slide images, and task instructions. The output is a structured sequence: `<think>visual context</think><answer>speech transcript</answer>`. In the `<think>` stage, the model performs a visual scan similar to OCR to extract words, phrases, and entities from the slide potentially related to the speech. In the `<answer>` stage, the model prioritizes the audio while using the entities extracted in `<think>` as semantic anchors to assist in identifying technical terms.

During training, the authors used Qwen2.5-Omni 3B / 7B as base models and performed policy optimization via GRPO on the SlideASR-S training set. Instead of just learning a format template, the model is pulled toward four objectives: correct format, accurate visual priors, accurate final transcription, and the actual citation of key entities between the look and listen phases.

The authors also constructed SlideASR-Bench to address data scarcity. SlideASR-S is extended from ContextASR-Bench, using entities and domain labels to generate slide-style text, which is then rendered into slide images using Matplotlib, totaling 8,467 samples (6,413 training, 2,054 testing). SlideASR-R consists of 60 real academic report segments covering chemistry, medicine, biology, and AI, with 200 manually annotated domain entities for testing in real-world complex scenarios.

### Key Designs
1. **Look-then-Listen Reasoning Chain**:
	- **Function**: Decouples visual perception and speech transcription temporally to prevent the model from directly treating slide text as the transcript.
	- **Mechanism**: The model first extracts slide text and entities in `<think>` to form a visual prior, then generates the final transcript in `<answer>` based on audio, citing candidate entities from `<think>` when encountering technical terms.
	- **Design Motivation**: Humans also establish context by looking at slides before confirming content through listening. This structure allows visual signals to serve as "anchors" rather than directly determining the output.

2. **Multi-objective Policy Optimization with Four Rewards**:
	- **Function**: Ensures the model adheres to the structure while genuinely learning to see images, hear audio, and connect entities.
	- **Mechanism**: Format Reward checks the `<think><answer>` format; OCR Reward calculates $R_{OCR}=max(1-WER,0)$ between `<think>` text and ground-truth slide text; ASR Reward calculates $R_{ASR}=max(1-WER,0)$ between `<answer>` and ground-truth transcript; Visual Anchoring Reward counts whether key entities appear in both `<think>` and `<answer>`, approximating target entity recall.
	- **Design Motivation**: Using only ASR rewards might lead the model to ignore visuals; using only OCR rewards encourages copying slides. The VA reward specifically constrains whether "seen entities are correctly used for heard speech," acting as the bridge between phases.

3. **Entity-dense SlideASR-Bench**:
	- **Function**: Provides data that better evaluates professional entity recognition compared to existing SlideSpeech.
	- **Mechanism**: The synthetic part uses domain labels and entity lists to generate formal slide text rendered as images; the real part collects audio and slides from public report videos with manual cross-modal entity annotation.
	- **Design Motivation**: If professional entities are too sparse, models can mask issues with generic ASR scores. SlideASR-Bench uses NE-WER and NE-FNR as core metrics to test the method on truly difficult entity recognition.

### Loss & Training
VAPO utilizes GRPO for reinforcement learning-style post-training, with a total reward of $R_{total}=\lambda_1R_{Format}+\lambda_2R_{OCR}+\lambda_3R_{ASR}+\lambda_4R_{VA}$. In experiments, the four weights are set to 1 by default. Training spans 800 steps using AdamW, a learning rate of $1e^{-6}$, a global batch size of 32 on 4 A100 GPUs, a group size of 4, a sampling temperature of 1.0, and a KL penalty coefficient of 0.01.

## Key Experimental Results

### Main Results
The authors first compared contextless, pipeline, and end-to-end settings on SlideSpeech. Surprisingly, simply adding slide text or images often worsened baselines, while VAPO simultaneously lowered WER and improved keyword recall.

| Method | Setting | Dev WER | Dev Recall | Test WER | Test Recall |
|------|------|---------|------------|----------|-------------|
| Qwen2.5-Omni-7B | Audio-only | 11.75 | 94.78 | 11.75 | 94.78 |
| Qwen3-Omni-30B-A3B | Audio-only | 10.87 | 95.04 | 11.71 | 95.50 |
| Qwen3-Omni-30B-A3B | Slide text pipeline | 50.43 | 96.45 | 57.12 | 96.34 |
| Qwen3-Omni-30B-A3B | Slide image end-to-end | 19.85 | 95.59 | 24.13 | 94.74 |
| VAPO-3B | Slide image end-to-end | 9.84 | 96.54 | 10.73 | 96.57 |
| VAPO-7B | Slide image end-to-end | 8.62 | 97.61 | 10.31 | 97.32 |

On SlideASR-Bench, VAPO's advantage in entity-related metrics is more pronounced, particularly on the synthetic English/Chinese test sets and the real SlideASR-R.

| Method | SlideASR-S en WER | en NE-WER | en NE-FNR | SlideASR-S zh WER | zh NE-WER | zh NE-FNR | SlideASR-R NE-WER | R NE-FNR |
|------|-------------------|-----------|-----------|-------------------|-----------|-----------|-------------------|----------|
| Qwen3-Omni-30B-A3B Audio-only | 9.06 | 14.61 | 15.53 | 20.77 | 23.31 | 22.49 | 40.43 | 41.09 |
| Qwen3-Omni-30B-A3B End-to-end image | 101.45 | 59.64 | 12.08 | 79.21 | 46.45 | 5.54 | 32.26 | 24.75 |
| VAPO-3B | 4.90 | 3.19 | 3.73 | 2.47 | 4.21 | 2.22 | 27.28 | 19.31 |
| VAPO-7B | 4.60 | 2.83 | 2.97 | 2.13 | 3.78 | 1.36 | 26.48 | 15.35 |

### Ablation Study
Reward ablation shows that the ASR reward is the foundation for stable generation, the OCR reward improves visual priors, and the Visual Anchoring reward further reduces entity omissions.

| Model | ASR Reward | OCR Reward | VA Reward | SlideASR-R NE-WER | SlideASR-R NE-FNR |
|------|------------|------------|-----------|-------------------|------------------|
| Qwen2.5-Omni-3B | No | No | No | 49.00 | 53.47 |
| Qwen2.5-Omni-3B | Yes | No | No | 37.23 | 31.19 |
| Qwen2.5-Omni-3B | Yes | Yes | No | 29.97 | 22.28 |
| Qwen2.5-Omni-3B | Yes | Yes | Yes | 27.28 | 19.31 |
| Qwen2.5-Omni-7B | No | No | No | 41.77 | 35.15 |
| Qwen2.5-Omni-7B | Yes | Yes | Yes | 26.48 | 15.35 |

Weight sensitivity experiments show that the default 1:1:1:1 is most stable. Increasing the VA reward slightly improves Chinese entity recall but increases overall WER; increasing the ASR reward may suppress visual citation.

| Weight $\lambda_1:\lambda_2:\lambda_3:\lambda_4$ | en WER | en NE-WER | en NE-FNR | zh WER | zh NE-WER | zh NE-FNR |
|---------------------------------------------------|--------|-----------|-----------|--------|-----------|-----------|
| 1:1:1:1 | 4.90 | 3.19 | 3.73 | 2.47 | 4.21 | 2.22 |
| 1:1:1:2 | 5.27 | 3.34 | 3.78 | 2.50 | 4.30 | 2.09 |
| 1:1:2:1 | 5.32 | 4.12 | 3.91 | 2.48 | 4.38 | 2.09 |
| 1:2:1:1 | 5.17 | 3.45 | 3.80 | 2.51 | 4.23 | 1.99 |

### Key Findings
- Visual interference is a widespread problem. On SlideSpeech test, MiniCPM-o-2.6 reached a VIR of 63.28%, Megrez-Omni 44.90%, and even Qwen2.5-Omni-7B had 12.87%.
- Directly adding visual context can lead to catastrophic degradation. Qwen3-Omni reached a WER of 101.45 under the SlideASR-S en end-to-end image setting, clearly treating slide content as audio to be transcribed.
- VAPO achieves the greatest improvement in entity recognition. On SlideASR-R, VAPO-7B reduced NE-FNR to 15.35, outperforming the strongest baselines (24.75 or 28.22).
- RL is more effective than SFT. While SFT with `<think>` reduced Qwen2.5-Omni-7B's WER from 10.58 to 6.73, VAPO further reduced it to 3.37, indicating reward optimization helps the model truly utilize `<think>` in `<answer>`.
- Mismatched slide experiments show VAPO does not blindly copy images. After randomly swapping slides, VAPO-3B's WER was approximately 6.70, close to the audio-only baseline, suggesting it can fallback to audio when visual cues are unreliable.

## Highlights & Insights
- The paper breaks the intuition that "more multimodal data is always better." Omni-modal models do not naturally fuse modalities; without reasoning constraints, strong visual text can suppress audio.
- VIR is an excellent diagnostic metric. Standard WER only indicates the error amount, whereas VIR points to the cause: "exclusive slide words were incorrectly output," making it highly precise for SlideASR.
- The structure of VAPO resembles migrating CoT to cross-modal perception. `<think>` is not just an explanation but a cache of visual entities observable by `<answer>`.
- The Visual Anchoring Reward design is critical. It prevents the model from only performing pretty OCR in `<think>` while ignoring visual info in the final transcript, and vice versa.
- SlideASR-Bench has independent value for future research. Being entity-dense and distinguishing NE-WER / NE-FNR makes the task evaluation more practical than simple WER.

## Limitations & Future Work
- The method currently relies heavily on slide text. For charts, diagrams, drug images, math formulas, or non-textual visual cues, the current OCR-centric design of the "Look" stage is insufficient.
- Training data is still primarily synthetic SlideASR-S. Although validated as effective on several real sets, the layout, noise, occlusions, and complex charts of synthetic slides might not cover real meeting environments.
- Higher inference latency. The structured `<think><answer>` increases output length and computational overhead; the paper acknowledges it is better suited for offline transcription. Real-time captioning might require distillation or pre-caching visual priors.
- Rewards require entity and slide text annotations. VA and OCR rewards depend on relatively clear entity sets and text supervision, requiring adaptation for weakly annotated or open video scenarios.
- Prevention of over-anchoring is still needed. Weight experiments show excessively high VA rewards harm overall WER; future work needs to help the model judge the reliability of visual cues better.

## Related Work & Insights
- **vs Pipeline SlideASR**: Traditional methods use OCR first then treat text as context; modular but suffers from error accumulation. VAPO maintains end-to-end input while explicitly separating visual priors and transcription in the output structure.
- **vs Contextual ASR**: Contextual ASR often utilizes entity lists, domain labels, or dialogue history. VAPO automatically extracts context from slide images, making it more suitable for reporting scenarios without manual lexicons.
- **vs Vanilla OLLM**: Normal OLLMs process images and audio simultaneously, leading to modal competition. VAPO uses temporal decoupling and RL rewards to make visuals a supporting anchor.
- **vs SFT with CoT**: SFT can teach formats but does not guarantee that intermediate visual content is utilized by the final answer. VAPO's VA reward directly rewards cross-stage entity reuse, resulting in stronger connections.
- **Insights**: This "Look-then-Listen" paradigm can be extended to tasks like meeting video understanding, medical imaging report dictation, or classroom recording transcription, and can be further expanded to "View chart, listen to explanation, then generate structured notes."

## Rating
- Novelty: ⭐⭐⭐⭐☆ Problem insight and reward design are highly targeted; the method itself is a structural innovation combining reasoning chains with RL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes failure mode quantification, two types of primary datasets, reward ablation, weight sensitivity, SFT comparison, mismatch robustness, and attention visualization.
- Writing Quality: ⭐⭐⭐⭐☆ The narrative is clear, the motivation chain from Visual Interference to VAPO is smooth, and tables are numerous but data-rich.
- Value: ⭐⭐⭐⭐⭐ Highly practical for professional ASR scenarios and omni-modal model training, especially by reminding us to design modality fusion processes explicitly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)

</div>

<!-- RELATED:END -->
