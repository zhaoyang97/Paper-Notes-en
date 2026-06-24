---
title: >-
  [Paper Note] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Large Language Models] This paper proposes the CAT model, which captures fine-grained audio-visual features via a question-aware Clue Aggregator. Combined with a hybrid multimodal training strategy and an AI-assisted Vagueness-aware Direct Preference Optimization (ADPO) strategy, it significantly improves MLLM question-answering accuracy in dynamic audio-visual scenarios, achieving SOTA performance on multiple AVQA benchmarks.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Audio-Visual Question Answering"
  - "Clue Aggregator"
  - "Preference Optimization"
  - "Vagueness Elimination"
date: 2026-05-08
content_hash: d1d1fcdd0424734e
---

# CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios

**Conference**: ECCV 2024  
**arXiv**: [2403.04640](https://arxiv.org/abs/2403.04640)  
**Code**: [GitHub](https://github.com/rikeilong/Bay-CAT)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Audio-Visual Question Answering, Clue Aggregator, Preference Optimization, Vagueness Elimination

## TL;DR

This paper proposes the CAT model, which captures fine-grained audio-visual features via a question-aware Clue Aggregator. Combined with a hybrid multimodal training strategy and an AI-assisted Vagueness-aware Direct Preference Optimization (ADPO) strategy, it significantly improves MLLM question-answering accuracy in dynamic audio-visual scenarios, achieving SOTA performance on multiple AVQA benchmarks.

## Background & Motivation

The real world consists of a mixture of auditory and visual information. Although multimodal large language models (MLLMs) can respond to audio-visual content, they still face two core limitations:

**Insufficient Association between Audio-Visual Content and Questions**: Existing MLLMs (e.g., Video-LLaMA, ChatBridge) primarily process each modality independently using multiple branches and then concatenate the modal embeddings with prompts to feed into the LLM. This paradigm fails to enable low-level interaction between textual information and audio-visual cues, preventing the network from focusing on question-relevant details.

**Vague Descriptions caused by Multimodal Alignment Difficulties**: Aligning multimodal and textual corpora is extremely challenging, often causing the model to generate vague answers—either using ambiguous vocabulary to describe audio-visual content or producing excessive useless text. For instance, when asked "what instrument is being played in the video", both Video-LLaMA and ChatBridge fail to correctly identify "bagpipes".

Key Challenge: Coarse-grained modal bridging methods fail to meet the need for precise localization of specific audio-visual objects in AVQA tasks. This work addresses this from three aspects: question-guided feature aggregation, hybrid audio-visual training, and preference optimization for vagueness elimination.

## Method

### Overall Architecture

CAT consists of three information streams that are fed into a frozen LLaMA2-7B:
- **Visual Branch**: ImageBind-encoded video $\rightarrow$ linear projection $\rightarrow$ visual token $x^{vid}$
- **Audio Branch**: ImageBind-encoded audio $\rightarrow$ linear projection $\rightarrow$ audio token $x^{aud}$
- **Clue Branch**: Clue Aggregator interacts question text with audio-visual features $\rightarrow$ Q-Former compression $\rightarrow$ clue token $x^{cue}$

These three types of tokens are concatenated with the text query and fed into the LLM with LoRA to generate answers.

### Key Designs

1. **Clue Aggregator (CA)**:

    - Function: Dynamically extracts fine-grained clues relevant to the current question from audio-visual features.
    - Mechanism: Works in two steps—
        - **Step 1 Perception**: Employs a Perceiver (a mini-Transformer network) containing a forward block $\mathcal{B}_1$ and a backward block $\mathcal{B}_2$. $\mathcal{B}_1$ uses the question text $h_t$ as the query to locate question-related audio-visual regions via cross-attention. $\mathcal{B}_2$ consolidates the original audio-visual representation at the attention level to restore the original modal sequence length. Specifically:
       - $\mathcal{B}_1(h_t; X) = \text{XA}(h_t, \text{FFN}(\text{SA}(X)))$
       - $\mathcal{B}_2(X; \mathcal{B}_1) = \text{SA}(\text{FFN}(\text{XA}(\text{FFN}(\text{SA}(X)), \mathcal{B}_1)))$
        - **Step 2 Aggregation**: Adopts a Q-Former-like architecture, setting K=48 learnable query vectors to extract fixed-length clue tokens from question-aware audio-visual features.
    - Design Motivation: The visual and audio Perceivers share parameters to learn cross-modal latent associations; conducting cross-attention with the question for localization first and then restoring the original representation avoids information loss.

2. **Hybrid Multimodal Training Strategy**:

    - Function: Trains CAT in two stages to enable simultaneous understanding of vision and audio.
    - Mechanism:
        - **Stage-I Feature Alignment**: First, train the visual projector using WebVid 2.5M video-text data (with the audio part frozen), and then train the audio projector using WavCap audio-text data (with the visual part frozen), while keeping the LLM and ImageBind frozen throughout.
        - **Stage-II Audio-Visual Joint Instruction Tuning**: Freeze the visual and audio projectors, fine-tune only the Clue Aggregator and LoRA parameters, utilizing 100k video instructions along with the self-constructed AVinstruct dataset.
    - Design Motivation: Stage-wise training prevents multimodal conflicts. AVinstruct collects real-world videos from YouTube and VGGSound, leveraging GPT-4 to synthesize joint audio-visual QA pairs from human-written exemplars.

3. **AI-Assisted Vagueness-Aware Direct Preference Optimization (ADPO)**:

    - Function: Retrains the model to prefer precise descriptions and reject vague ones.
    - Mechanism: Works in two steps—
        - **Collecting Vague Samples and Rewriting**: Use the trained CAT to generate answers on the training set, let GPT evaluate whether each answer is vague (i.e., significantly deviating from the ground truth), label vague answers as negative responses $y_{neg}$, and have GPT rewrite them into precise positive responses $y_{pos}$.
        - **DPO Training**: Optimize jointly using DPO loss and SFT auxiliary loss: $\mathcal{L} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{SFT}$, where $\lambda=0.1$.
    - Design Motivation: Pure SFT training cannot resolve the vagueness of MLLMs in describing specific audio-visual objects. Since the standard DPO loss has limited effectiveness when the difference between positive and negative responses is small, the auxiliary SFT loss is introduced to stabilize the learning process towards prioritizing positive responses.

4. **Question Tagging Mechanism**:

    - Function: Uses `<Q>` and `</Q>` tags in the prompt to mark the start and end of the question.
    - Mechanism: Reformats the input structure as: `USER:<system><Q></Q><video><audio><clues> Assistant:`
    - Design Motivation: Allows the Clue Aggregator to locate exactly which part of the input represents the question, facilitating more accurate question-aware feature extraction.

### Loss & Training

- Stage-I: Standard language modeling loss, training only the projectors.
- Stage-II: Standard language modeling loss + LoRA (r=64, alpha=128), batch size=128, learning rate=2e-5.
- ADPO Stage: $\mathcal{L}_{DPO} + 0.1 \cdot \mathcal{L}_{SFT}$, LoRA (r=64, alpha=16), batch size=1, learning rate=4e-6, $\beta=0.1$.
- All training is conducted on a single NVIDIA A100 GPU.

## Key Experimental Results

### Main Results: Video-Text Generation & Zero-Shot Video QA

| Method | LLM Size | Correctness | Detail | Context | Temporal | Consistency | MSRVTT-QA Acc | ActivityNet-QA Acc |
|------|---------|--------|------|--------|------|--------|--------------|-------------------|
| Video-LLaMA | 7B | 39.2 | 43.6 | 43.2 | 36.4 | 35.8 | 29.6 | 12.4 |
| Video-ChatGPT | 7B | 48.0 | 50.4 | 52.4 | 39.6 | 47.4 | 49.3 | 35.2 |
| LLaMA-VID | 7B | 59.2 | 60.0 | 70.6 | 49.2 | 50.2 | 57.7 | 47.1 |
| **CAT (Ours)** | **7B** | **61.6** | **62.0** | **69.8** | **56.2** | **57.8** | **62.1** | **50.2** |

### Main Results: Closed-ended AVQA (Fully Supervised Music-AVQA)

| Method | Language Model | Trainable Params | Audio Avg. | Visual Avg. | Audio-Visual Avg. | Overall Avg. |
|------|---------|---------|---------|---------|-----------|---------|
| PSTP-Net | BERT | 4.3M | 70.9 | 77.3 | 72.6 | 73.5 |
| LAVISH | CLIP | 21.1M | 77.1 | 77.3 | 77.0 | - |
| VAST | BERT | - | - | - | - | 80.7 |
| **CAT-7B** | **LLaMA2** | **5.8M** | **84.9** | **86.1** | **83.2** | **84.3** |

### Ablation Study

| Input Modality Combination | Correctness | Detail | Temporal |
|-------------|--------|------|------|
| Visual Only $x^{vid}$ | 38.6 | 40.4 | 34.0 |
| Visual + Audio $x^{vid}+x^{aud}$ | 40.6 | 44.8 | 35.2 |
| Clue Only $x^{cue}$ | 58.6 | 57.4 | 55.8 |
| All $x^{vid}+x^{aud}+x^{cue}$ | **61.6** | **59.0** | **56.2** |

| ADPO Ablation (AVSD Dataset) | B@4 | METEOR | CIDEr |
|----------------------|-----|--------|-------|
| Video-LLaMA w/o ADPO | 18.4 | 40.1 | 63.7 |
| Video-LLaMA + ADPO | 22.2 | 48.2 | 69.2 |
| CAT w/o ADPO | 28.9 | 56.2 | 74.8 |
| CAT + ADPO | **34.2** | **59.8** | **79.0** |

### Key Findings

- The clue token alone outperforms the combined visual + audio input, demonstrating the critical importance of question-guided feature extraction.
- In the Clue Aggregator, removing visual features has a far greater negative impact than removing audio (with B@4 dropping from 30.8 to 10.0 vs. 18.8), indicating that the visual modality contains more exploitable question-related details.
- ADPO exhibits strong generalizability across different MLLMs, yielding a 5.5 CIDEr point improvement even when applied to Video-LLaMA.
- The optimal number of clue tokens is K=48, as values either too high or too low degrade performance.

## Highlights & Insights

- **Elegant Clue Aggregator Design**: The forward-backward dual-block structure achieves question-aware localization without losing the original modal information, offering greater specificity than simple projectors or a standard Q-Former.
- **Pragmatic and Effective ADPO Strategy**: It reformulates the MLLM vagueness issue as a preference optimization task, leveraging GPT to automatically synthesize positive and negative response pairs, thereby bypassing manual annotation costs.
- **State-of-the-Art with Minimal Trainable Parameters (5.8M)**: Thanks to LoRA and the frozen encoder/LLM paradigm, training costs remain highly manageable.
- **Construction of the AVinstruct Dataset**: Provides dedicated instruction-tuning data specifically for joint audio-visual QA, filling a gap in this research area.

## Limitations & Future Work

- ImageBind's audio encoder produces only a single token, which might result in over-compressed audio information, limiting the model's understanding of complex acoustic scenes.
- ADPO relies on GPT for vagueness evaluation and rewriting, which potentially introduces biases from external models.
- Evaluation is conducted solely on LLaMA2-7B, without testing on larger LLMs (e.g., 13B, 70B).
- The visual and audio Perceivers in the Clue Aggregator share parameters, which might not be an optimal design given the substantial differences between the two modalities.
- More complex temporal modeling methods (e.g., video segmentation, timestamp localization) have not been explored.

## Related Work & Insights

- **Video-LLaMA** (Zhang et al., 2023): Directly bridges modalities using linear layers, serving as the primary baseline for CAT.
- **Q-Former** (Li et al., 2023): Queries-based feature compression proposed in BLIP-2, which is integrated into the second stage of CAT's clue aggregation.
- **DPO** (Rafailov et al., 2023): Optimizes preferences directly by bypassing the reward model, which inspired the design of the ADPO strategy.
- **ChatBridge** (Zhao et al., 2023): A 13B model that still underperforms CAT-7B on AVQA, demonstrating that model scale is not the sole key factor.
- **Insight**: In multimodal reasoning, **question-guided feature extraction is more crucial than simply packing more modal inputs**—the clue token on its own outperformed the visual + audio combination.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the dual-block design of the Clue Aggregator and the ADPO strategy are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four major benchmarks: video-text generation, zero-shot VQA, and closed/open-ended AVQA.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined motivation, and intuitive visualization designs.
- Value: ⭐⭐⭐⭐ Provides an effective model architecture and training strategies for audio-visual question answering tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ECCV 2024\] Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[ECCV 2024\] LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers](loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)
- [\[CVPR 2026\] VKG-QA: Visual Knowledge Graph-based Question Answer for Large Multimodal Models](../../CVPR2026/multimodal_vlm/vkg-qa_visual_knowledge_graph-based_question_answer_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
