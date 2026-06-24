---
title: >-
  [Paper Note] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding
description: >-
  [ACL 2025][LLM (Other)][Auditory attention decoding] This paper proposes the Intention-aware Auditory Scene Understanding (II-ASU) paradigm and the AAD-LLM prototype system. By decoding which speaker the listener is attending to from intracranial electroencephalography (iEEG) signals and injecting this attention state into an auditory LLM, the model generates responses aligned with the listener's perception in multi-speaker scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Auditory attention decoding"
  - "Brain-computer interfaces"
  - "iEEG"
  - "Auditory LLMs"
  - "Intention awareness"
  - "Multi-speaker scenarios"
  - "Qwen2-Audio"
date: 2026-05-08
content_hash: cf5d7a38bec2cfb2
---

# AAD-LLM: Neural Attention-Driven Auditory Scene Understanding

**Conference**: ACL 2025  
**arXiv**: [2502.16794](https://arxiv.org/abs/2502.16794)  
**Code**: [Project Page](https://aad-llm.github.io)  
**Area**: Multimodal LLMs / Brain-Computer Interfaces / Auditory Scene Understanding  
**Keywords**: Auditory attention decoding, Brain-computer interfaces, iEEG, Auditory LLMs, Intention awareness, Multi-speaker scenarios, Qwen2-Audio  

## TL;DR

This paper proposes the Intention-aware Auditory Scene Understanding (II-ASU) paradigm and the AAD-LLM prototype system. By decoding which speaker the listener is attending to from intracranial electroencephalography (iEEG) signals and injecting this attention state into an auditory LLM, the model generates responses aligned with the listener's perception in multi-speaker scenarios.

## Background & Motivation

**Background**: Auditory foundation models (e.g., LTU, SALMONN, Qwen2-Audio) have made significant progress in general auditory scene understanding, enabling audio description, transcription, and question answering. However, these models process all acoustic inputs equally without distinguishing the listener's focus, which is disconnected from the human selective attention mechanism.

**Limitations of Prior Work**: Existing auditory LLMs indiscriminately transcribe and analyze all sound sources in multi-speaker environments, failing to distinguish the user's attended speech from background conversations. The standard Qwen2-Audio model exhibits a Word Error Rate (WER) of 90.1% when transcribing the foreground speaker, but drops to 6.6% when given the target speaker, indicating a massive performance gap due to the lack of listener intention information. Previous auditory attention decoding (AAD) research has been confined to the signal enhancement level, failing to guide AI in semantic-level scene understanding and reasoning.

**Key Challenge**: The human auditory system possesses selective attention capabilities (the cocktail party effect), but auditory AI systems remain in a passive "hear-all, record-all" state. The model neither knows what the user is listening to nor can adapt its output accordingly, leading to a severe mismatch between the output and user perception.

**Goal**: (1) How to decode the listener's attentional intention (i.e., which speaker is being attended to) from brain signals; (2) How to inject the decoded attention state into large language models to enable them to generate responses aligned with the listener's perception in multi-speaker scenarios, rather than processing all sound sources equally.

**Key Insight**: Instead of modifying the acoustic signal (as in traditional speech enhancement-based AAD), the attention signal is integrated into the language model to guide selective reasoning over the auditory scene. This decouples the training of brain signal decoding and language model alignment: brain data requires only a few minutes to train a speaker predictor, while the language model can be trained independently for intention alignment on large-scale speech data.

**Core Idea**: Decoding the listener's attention from intracranial EEG into discrete speaker identity tokens acts as the vehicle. Injecting these tokens into auditory LLMs achieves intention-aware multi-speaker scene understanding.

## Method

### Overall Architecture

AAD-LLM is a three-input multimodal LLM system: textual question $Q$ + speech mixture $S$ + brain signal $Z \rightarrow$ intention-aligned response $A$. Built on Qwen2-Audio (Whisper audio encoder + Qwen2 text LLM), the system introduces an intention decoding module and an auxiliary speech separation module. The overall workflow consists of: (1) An auxiliary separator pre-processes the mixed speech into two separated streams; (2) An intention decoder predicts the attended speaker's identity from iEEG brain signals and outputs a speaker identity token; (3) The speaker token is mapped to the LLM embedding space via a projector and concatenated with the embeddings of the two speech streams and the text question; (4) The LLM first generates speaker label information via Chain-of-Thought (CoT) before generating the final response aligned with the listener's intention. The modules are trained decoupingly, with the audio encoder and LLM fine-tuned via LoRA.

### Key Designs

1. **Intention Decoding Module**:
    - **Function**: Decodes the speaker identity attended by the listener from intracranial EEG signals.
    - **Mechanism**: First, K-means clustering ($K=8$) is applied to the x-vectors (512-dimensional embeddings) of a large-scale speaker corpus to obtain discrete representations of speaker types. Then, a bidirectional LSTM maps the iEEG signal to the predicted cluster index, outputting the corresponding cluster centroid vector as the "intention token."
    - **Design Motivation**: Utilizing discrete speaker identity tokens instead of continuous speech reconstruction aligns well with how LLMs process discrete tokens. This design decouples intention decoding from intention alignment, bypassing the limitation of scarce brain data (only a few minutes long) while allowing the LLM's alignment to be trained on massive speech datasets (85.3 hours).

2. **Intention Alignment Module**:
    - **Function**: Enables the LLM to selectively process the target speaker's content based on decoded attention identity information.
    - **Mechanism**: (a) The speaker token is embedded into the LLM space via a linear projector and concatenated with the speech/text embeddings; (b) During training, attention is simulated by randomly designating a foreground speaker and using the corresponding x-vector centroid as the intention input; (c) Chain-of-Thought prompting forces the model to output speaker labels and the attended target label before generating the final answer, addressing the issue where LLMs tend to ignore the attention token.
    - **Design Motivation**: Directly embedding the intention token into the input is insufficient for LLMs to automatically perform selective processing. CoT-forced reasoning compels the model to explicitly utilize attention information.

3. **Auxiliary Speech Separation Module (Mamba-TasNet)**:
    - **Function**: Pre-separates mixed speech into two independent streams, reducing the processing difficulty for the LLM.
    - **Mechanism**: Based on Mamba-TasNet blind source separation. The separator itself is unaware of the listener's intention; the LLM selects the correct stream based on the brain-decoded token.
    - **Design Motivation**: Ablation experiments show that the separator significantly improves the LLM's speaker discrimination ability.

### Loss & Training

- Audio Encoder (Whisper) and LLM (Qwen2): Fine-tuned using LoRA (rank=512) with the standard autoregressive language model loss.
- Speech Separator: Trained to maximize the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) of the separated speech.
- Speaker Predictor: Trained using cross-entropy loss to predict the speaker class among $K=8$ clusters.
- Decoupled Training: The brain decoding model is trained on limited clinical data, whereas the LLM alignment is trained independently on large-scale speech datasets.

## Key Experimental Results

### Auditory Attention Decoding and Speech Extraction

| Method | AAD Accuracy ↑ | SNR ↑ | WER ↓ | Speaker Similarity ↑ |
|------|-----------|------|------|-------------|
| Original Mixed Speech | - | 0.1 | 37.4 | 84.4 |
| Blind Separation + Mel Reconstruction | 92.0 | 12.0 | 15.2 | 94.1 |
| Blind Separation + Envelope Reconstruction | 88.0 | 11.2 | 20.4 | 90.7 |
| Target Speaker Extraction | 96.0 | 12.8 | 14.3 | 94.8 |
| **AAD-LLM (Brain Decoding)** | **94.4** | **12.2** | **14.7** | **94.1** |
| AAD-LLM (Oracle) | 95.8 | 12.3 | 13.0 | 94.3 |
| Oracle Speaker Upper Bound | 100.0 | 13.0 | 8.8 | 95.5 |

### Intention-Aware Scene Understanding (Comprehensive Evaluation of Four Tasks)

| System Configuration | Description AVG ↑ | Transcription WER ↓ (Foreground) | Summarization ROUGE-L ↑ (Foreground) | QA ROUGE-L ↑ (Foreground) |
|----------|---------|---------------|-------------------|------------------|
| Qwen2-Audio (Mixed) | 50.9 | 90.1 | 27.5 | 39.9 |
| Qwen2-Audio (Random Speaker) | 69.3 | 71.8 | 30.2 | 50.0 |
| Qwen2-Audio (Extracted Speaker) | 88.1 | 18.5 | 54.5 | 62.3 |
| **AAD-LLM (Brain Decoding)** | **89.3** | **14.4** | **58.3** | **63.1** |
| AAD-LLM (Oracle) | 89.9 | 12.5 | 59.7 | 63.0 |
| Qwen2-Audio (Oracle Upper Bound) | 91.7 | 6.6 | 59.7 | 64.9 |

### Ablation Study

| Ablation Variant | Description AVG ↑ | Transcription WER ↓ |
|----------|---------|---------|
| Full AAD-LLM | 89.3 | 14.4 |
| W/o CoT Prompting | Significant Decrease | Significant Increase |
| W/o Speech Separator | Significant Decrease | Significant Increase |
| Adding Clinical Data (15 mins) | 89.2 | **6.0** |

### Key Findings

- The brain-decoded version of AAD-LLM achieves a description accuracy of 89.3%, close to the Oracle attention upper bound (91.7%), demonstrating that the attention decoded from brain signals is highly accurate.
- The foreground speaker's WER drops from 90.1% (without intention awareness) to 14.4%, representing a **performance gain of 75.7 percentage points**.
- In subjective evaluations, 83.8% to 92.2% of the generated responses align closely with the target speaker, significantly outperforming the baselines.
- Adding 15 minutes of clinical iEEG data further reduces the WER to 6.0%, approaching the Oracle upper bound.

## Highlights & Insights

- **Paradigm Innovation**: A shift from passive auditory processing to listener intention-driven auditory AI, pioneering the II-ASU direction.
- **New Application for Brain-AI Interfaces**: For the first time, brain signals are used to steer auditory scene understanding in LLMs, going beyond traditional signal-level enhancement.
- **Modular Decoupled Design**: Intention decoding and intention alignment are trained separately, artfully resolving the conflict between the scarcity of brain data and the data-hungry nature of LLM training.
- **CoT-Forced Attention Utilization**: A simple yet effective engineering strategy that tackles the common issue where LLMs tend to ignore newly introduced modality tokens.

## Limitations & Future Work

- Dependence on intracranial EEG (invasive BCI); non-invasive EEG may lack sufficient precision, limiting practical deployment.
- Validation was restricted to dual-speaker scenarios; scenarios with three or more speakers remain untested.
- Attention was simulated by randomly designating a foreground speaker during training, failing to capture the dynamic switching of attention in real-world environments.
- The iEEG data was collected from a small cohort of clinical epilepsy patients (6 subjects); individual variability and cross-subject generalization require further validation.
- The speaker clustering employs only $K=8$, which may be insufficient for scenarios with higher speaker diversity.

## Related Work & Insights

- **vs. Auditory LLMs (e.g., Qwen2-Audio/SALMONN)**: Prior models lack intention awareness and process all acoustic sources equally, while AAD-LLM selectively refines processing based on brain signals.
- **vs. Traditional AAD (O'Sullivan 2015; Geirnaert 2021)**: Traditional AAD is restricted to speech enhancement or extraction, whereas AAD-LLM scales this up to semantic-level scene understanding and QA.
- **vs. EEG-to-Text (Jiang 2024a; Kim 2024)**: EEG-to-Text directly generates text from brain signals with the goal of decoding language; AAD-LLM uses brain signals as an intentional guide to align the outputs of auditory AI.
- **Insights**: The modular intention injection concept can be generalized to other BCI scenarios—decoding arbitrary physiological signals (e.g., eye gaze, head orientation) into discrete intention tokens to steer LLMs.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)
- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment](maps_personalized_search.md)

</div>

<!-- RELATED:END -->
