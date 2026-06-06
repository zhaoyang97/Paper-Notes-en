---
title: >-
  [Paper Note] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation
description: >-
  [ACL2026][Social Computing][Speaker attribution] This paper converts public Zoom meeting videos into a civic deliberation corpus that is cross-video traceable and contains action tags and persona metadata. Using QLoRA to…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Speaker attribution"
  - "Action tags"
  - "Persona modeling"
  - "Civic deliberation simulation"
  - "QLoRA"
date: 2026-05-08
content_hash: 56f62cd8d72daf1c
---

# Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation

**Conference**: ACL2026  
**arXiv**: [2511.17813](https://arxiv.org/abs/2511.17813)  
**Code**: Not released  
**Area**: audio_speech  
**Keywords**: Speaker attribution, Action tags, Persona modeling, Civic deliberation simulation, QLoRA

## TL;DR
This paper converts public Zoom meeting videos into a civic deliberation corpus that is cross-video traceable and contains action tags and persona metadata. Using QLoRA to fine-tune LLMs for specific participant utterance generation, the approach reduces perplexity by up to 67%, and humans find it difficult to distinguish simulated dialogues from real meeting segments in Turing-style tests.

## Background & Motivation
**Background**: LLM multi-agent simulations can model debates, negotiations, meetings, and policy discussions. However, many systems still rely on manually written persona prompts, providing each agent with a description of identity, goals, and style for turn-taking. While low-cost, this approach struggles to capture stable speaking habits, parliamentary procedures, and long-term interaction structures in real institutions.

**Limitations of Prior Work**: There are many public meeting videos (e.g., courts, school boards, city councils), but automatic speech recognition (ASR) typically outputs anonymous labels like `Speaker_1`. Without real identities traceable across videos, models can only learn a generalized speaking style of an "official/board member/judge," failing to stably simulate a specific individual's issue preferences, phrasing habits, and procedural actions.

**Key Challenge**: Realistic civic simulation requires large-scale real dialogue data, which is naturally noisy: varying video resolutions, shifting speaker tiles, ASR errors, and mixtures of procedural speech and impromptu debates. Simple persona prompts are insufficient, and pure audio diarization cannot provide cross-meeting identities or pragmatic actions.

**Goal**: The authors aim to solve three sub-problems: first, recovering stable speaker identities from public videos; second, extending transcripts into training samples containing persona, topic, and action tags; third, evaluating multi-agent simulation using metrics that measure fluency, persona fidelity, and identity distinctiveness.

**Key Insight**: Zoom gallery view provides a usable weak supervision signal: the active speaker's tile is highlighted, and the tile corner usually contains a name. By capturing this visual cue, the authors combine OCR, audio transcription, and textual context to transform anonymous ASR transcripts into speaker-attributed transcripts.

**Core Idea**: Replace thin persona prompts with "traceable identity + action tags + persona metadata," allowing the LLM to learn from real deliberation corpora who speaks with which procedural action under specific agendas.

## Method
This paper proposes a complete pipeline covering data construction, metadata extraction, model fine-tuning, and simulation evaluation rather than just a prompt. The key idea is to transform public videos into structured dialogue data and inject this structured information into training and inference, enabling the model to not only speak like a government official but specifically like a particular participant at a certain agenda stage.

### Overall Architecture
The input consists of public YouTube/Zoom meeting recordings, and the output includes a speaker-labeled civic deliberation dataset for training and simulation, as well as persona agents fine-tuned for each speaker. The workflow consists of four steps: identifying active speakers from video and performing OCR on names; using Whisper to transcribe audio and align segments to normalized speakers; using GPT-5 to extract persona profiles, meeting topics, and action tags; and finally serializing context with metadata into ChatML-style samples for QLoRA fine-tuning to predict the target speaker's next turn.

### Key Designs
1.  **Multimodal Speaker Linking**:
    - **Function**: Converts anonymous ASR transcripts into consistent real speaker transcripts across videos.
    - **Core Idea**: Zoom gallery view is processed at 1 FPS, utilizing highlighted borders to locate active tiles and cropping the name area for OCR; low-resolution videos are enhanced with EDSR super-resolution. Audio is transcribed via Whisper and assigned to speakers based on timestamps. Cross-video names are normalized via fuzzy matching into canonical speaker labels.
    - **Design Motivation**: Traditional diarization depends on acoustic features and often fragments identities in multi-party, noisy, remote settings. The Zoom interface explicitly encodes the active speaker in the visual layer; utilizing this UI signal is more robust than pure audio and requires no additional metadata from meeting hosts.

2.  **Action-Aware Persona Data Format**:
    - **Function**: Enables training samples to contain long-term speaker style, meeting topics, and current pragmatic actions.
    - **Core Idea**: Persona profiles are derived from the 25 longest monologues of each speaker, where GPT-5 extracts goals, tone, boundaries, and policy stances into a stable profile. Topic extraction segments transcripts into 1024-token chunks for summarization. Action tag extraction uses 15–30 compact speech-act labels for each utterance (e.g., *move a motion*, *ask for clarification*, *cite material*, *call for vote*). During training, tiap utterance is preceded by alphabetically sorted action tags while retaining prior speaker labels.
    - **Design Motivation**: Real deliberation follows procedures rather than free chat. Including actions allows the model to know the "function" of the next sentence, providing stronger constraints on form than identity alone and explaining why action tags help reduce PPL even for non-fine-tuned models.

3.  **Complementary Fidelity Evaluation**:
    - **Function**: Avoids relying solely on perplexity, as low PPL does not equate to being "like" a specific person.
    - **Core Idea**: Three types of metrics are used. PPL measures linguistic distribution fit. Classifier Fool Rate (CFR) trains one-vs-all DeBERTa classifiers to calculate the proportion of generated speech classified as the target speaker's real speech. Speaker Attribution Accuracy (SAA) trains multi-classifiers to see if generated speech can be correctly attributed to the right speaker. CFR measures "likeness," while SAA measures "distinctiveness" from other participants.
    - **Design Motivation**: Many people in government meetings share institutional phrasing. Generating "city council-sounding" sentences is easy, but preserving a personal rhetorical fingerprint is hard. Thus, the combination of CFR and SAA is more reliable than fluency alone.

### Loss & Training
The training objective is standard autoregressive language modeling: predicting the target utterance conditioned on speaker-labeled context, persona, topic, and action tags. The authors compare GPT-OSS-120B, LLaMA-3.1-70B, and Qwen-2.5-72B. Fine-tuning uses QLoRA, with hyperparameters tuned via grid search over 5 epochs, selecting checkpoints based on held-out PPL. Context length is truncated to 1024 tokens. During simulation, agents speak round-robin, with a time-aware version adding agenda timestamps to test if temporal anchoring improves agenda coverage and decision-making closure.

## Key Experimental Results

### Main Results
The paper constructs three real-world civic deliberation datasets covering courts, school boards, and a New Zealand district council. While not massive, every transcript includes identity, topics, action tags, and speaker profiles.

| Dataset | # Transcripts | Avg. Participants | Total Words | Avg. Words/File |
| :--- | :--- | :--- | :--- | :--- |
| DC Court of Appeals | 10 | 8.6 | 193,712 | 19,371 |
| Albemarle School Board | 32 | 21.38 | 594,253 | 18,570 |
| Waipā District Council | 80 | 12.125 | 933,272 | 11,666 |

In human data validation, utterance speaker identification reached 98.5%, and cross-video speaker consistency reached 95.3%. Subjective transcription quality was high, with 55.9% "Strongly agree" and 29.8% "Agree" on accuracy, indicating a reliable data foundation.

| Model/Config | Albemarle PPL ↓ | Albemarle CFR ↑ | Albemarle SAA ↑ | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| GPT baseline w/o tags | 24.68 ± 1.00 | 0.21 ± 0.03 | 0.36 ± 0.02 | Hard to stabilize persona with prompts alone |
| GPT fine-tuned w/o tags | 12.89 ± 0.50 | 0.64 ± 0.04 | 0.48 ± 0.03 | Fine-tuning significantly improves style fit |
| GPT fine-tuned w/ tags | 7.73 ± 0.31 | 0.82 ± 0.05 | 0.63 ± 0.04 | Fine-tuning + action tags yields max gain |
| Qwen baseline w/o tags | 24.64 ± 1.35 | 0.18 ± 0.03 | 0.27 ± 0.02 | Weak speaker traits without fine-tuning |
| Qwen fine-tuned w/ tags | 5.32 ± 0.19 | 0.58 ± 0.03 | 0.31 ± 0.02 | Lowest PPL, but distinctiveness remains hard |

### Ablation Study
The ablation focuses on prompt components, action tags, and time anchoring. System message ablation shows that models rely heavily on persona prompts before fine-tuning; after fine-tuning, models internalize traits, and excessive prompting may introduce noise.

| Config/Analysis | Key Result | Description |
| :--- | :--- | :--- |
| Full prompt + baseline w/o tags | PPL 20.37 ± 1.92, CFR 0.29 ± 0.02, SAA 0.23 ± 0.02 | Full prompts help non-fine-tuned models maintain basic roles |
| No system prompt + baseline w/o tags | PPL 18.22 ± 2.19, CFR 0.08 ± 0.01, SAA 0.09 ± 0.01 | Without persona conditions, CFR/SAA drops sharply |
| Full prompt + fine-tuned w/ tags | PPL 6.64 ± 0.47, CFR 0.64 ± 0.03, SAA 0.45 ± 0.04 | Fine-tuning and tags provide the best robust quality |
| No micro + fine-tuned w/ tags | PPL 6.53 ± 0.45, CFR 0.89 ± 0.03, SAA 0.61 ± 0.04 | Simplified prompts often work better after fine-tuning |
| Time-aware simulation | Topic coverage: LLaMA 71.4%→94.4%, Qwen 81.4%→99.2% | Time anchoring helps models advance the agenda |
| Human Turing test | Sim. segments correctly identified ~44.5% | Human accuracy below or near chance, indicating high realism |

### Key Findings
- Fine-tuning is the primary source of improvement: PPL, CFR, and SAA generally improved across all models and datasets.
- Action tags are not just training labels but also natural language control signals; baseline PPL decreases significantly when tags are included even without fine-tuning.
- CFR is often higher than SAA, suggesting models find it easier to generate speech that "looks like" the target person than speech that is uniquely distinguishable from fellow participants.
- Time-aware simulation significantly helps LLaMA and Qwen with agenda progression, whereas GPT shows smaller gains due to its inherent planning strength.

## Highlights & Insights
- The most clever aspect is treating the Zoom UI as a weak supervision sensor. Instead of expensive end-to-end video understanding, it uses stable interface signals like highlighted borders and name tiles to obtain low-cost cross-video speaker linking.
- Action tags push persona modeling from "speaking like someone" to "acting like someone within a procedure." This is critical for civic simulation, as realism stems largely from procedural actions and turn-taking structures.
- The CFR/SAA design is insightful: one measures plausibility against a "not-this-person" set, while the other measures distinctiveness among candidates.
- Time anchoring experiments show that real meeting simulation requires agenda-level progress, not just coherent individual sentences.

## Limitations & Future Work
- The method depends heavily on the Zoom gallery view; it may not apply to non-Zoom videos or those without stable name tiles or layouts.
- Whisper ASR still makes errors with overlapping speech, noise, and low-quality recordings; if errors are used in profiles or tags, fine-tuning inherits these biases.
- Extraction of action tags and personas relies on GPT-5, raising concerns about cost, reproducibility, and potential bias.
- Round-robin turn-taking is used, but in real meetings, interruptions and silence are part of the behavior model. Future work could include next-speaker prediction.
- Civic persona simulation is a double-edged sword: it is useful for education but carries risks of misuse for faking public discourse or manipulating opinions.

## Related Work & Insights
- **vs. Traditional Speaker Diarization**: Traditional methods cluster audio; this work uses the Zoom visual interface for linking, which is better for cross-meeting identity tracking of public figures but restricted by video layout.
- **vs. Prompt-only Persona Simulation**: Prompt-only methods are lightweight but prone to persona drift; fine-tuning internalizes traits into model weights or LoRA adapters for better stability.
- **vs. Reflexion/ReAct Multi-agent Debate**: Those systems focus on reasoning; this work focuses on personified style, procedural actions, and agenda progression in institutional settings.
- **Insights for Other Tasks**: Medical consultations, classroom discussions, and corporate meetings can reuse the "identity linking + action tag + persona fine-tuning + dual-metric evaluation" structure with domain-specific action taxonomies.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines speaker linking, action tags, and persona PEFT into a practical civic simulation pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Model evaluation, human tests, and ablations are comprehensive, though cross-platform generalization could be stronger.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative; evaluation metrics are well-explained.
- Value: ⭐⭐⭐⭐⭐ High reference value for multi-party simulation, public policy analysis, and agent persona modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)

</div>

<!-- RELATED:END -->
