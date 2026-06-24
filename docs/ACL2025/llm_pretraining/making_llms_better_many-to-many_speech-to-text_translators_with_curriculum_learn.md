---
title: >-
  [Paper Note] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning
description: >-
  [ACL 2025][LLM Pretraining][Speech-to-Text Translation] Proposes LLM-SRT, which reformulates the Speech-to-Text Translation (S2TT) task as a joint Speech Recognition and Translation (SRT) task. Through a three-stage curriculum learning strategy (ASR→SMT→SRT), it effectively leverages the machine translation capabilities of LLMs to achieve state-of-the-art many-to-many speech translation performance across $15 \times 14$ language pairs in extremely low-resource scenarios (less…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Speech-to-Text Translation"
  - "Curriculum Learning"
  - "MLLM"
  - "Multilingual"
  - "Low-resource"
date: 2026-05-08
content_hash: e7c41d38e03c88c3
---

# Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning

**Conference**: ACL 2025  
**arXiv**: [2409.19510](https://arxiv.org/abs/2409.19510)  
**Code**: [Available](https://github.com/yxduir/LLM-SRT)  
**Area**: LLM Pre-training  
**Keywords**: Speech-to-Text Translation, Curriculum Learning, MLLM, Multilingual, Low-resource

## TL;DR

Proposes LLM-SRT, which reformulates the Speech-to-Text Translation (S2TT) task as a joint Speech Recognition and Translation (SRT) task. Through a three-stage curriculum learning strategy (ASR→SMT→SRT), it effectively leverages the machine translation capabilities of LLMs to achieve state-of-the-art many-to-many speech translation performance across $15 \times 14$ language pairs in extremely low-resource scenarios (less than 10 hours of data per language).

## Background & Motivation

### Background

Speech-to-Text Translation (S2TT) traditionally relies on cascaded systems (ASR + MT), which suffer from error propagation. Recently, Multimodal Large Language Models (MLLMs) have shown advantages in simplifying architectures and reducing error propagation, but they face two critical challenges:

**Data Scarcity**: Existing S2TT datasets are English-centric (e.g., MuST-C), and datasets supporting many-to-many translation (e.g., FLEURS) only have about 10 hours of training data per language.

**Capability Transfer Issue**: LLMs already possess strong multilingual machine translation capabilities; however, transfering this capability to the S2TT task under limited data remains unresolved.

### Key Insight

LLMs already possess "translation" capabilities (MT); what they lack is the ability to "hear" (Speech-to-Text alignment). If MLLMs can be trained to "understand" speech with a small amount of data and connect this to their existing translation capabilities, many-to-many S2TT in low-resource settings can be achieved.

### Novelty

Redefine the S2TT task as a joint SRT (Speech Recognition and Translation) task: inputting speech and **simultaneously outputting both transcription and translation**. This design allows the MLLM to generate the transcription first during inference, then utilize the LLM's inherent MT capability to translate, combining the advantages of both cascaded and end-to-end approaches.

## Method

### Overall Architecture

The architecture of LLM-SRT consists of three parts:
- **Speech Encoder**: A frozen Whisper-large-v3 to extract speech features.
- **Speech Adapter**: Q-Former + MLP to compress speech feature dimensions and align them with the hidden space of the LLM.
- **LLM**: Qwen series (3B/7B/32B), frozen or fine-tuned via LoRA.

### Key Designs

1. **Three-stage Curriculum Learning**

   **Stage 1: ASR (Automatic Speech Recognition)**  
    - Goal: Multimodal alignment to make the model learn to "hear".
    - Input: Speech + language tag instructions (e.g., `<|eng|>`).
    - Output: Transcription text.
    - Train all target languages using the Common Voice dataset.
    - Only the speech adapter is trained.

   **Stage 2: SMT (Speech-assisted Machine Translation)**  
    - Goal: Activate the cross-lingual translation capabilities of the LLM.
    - Input: Speech + transcription + translation instructions (e.g., `Will it rain tomorrow?<|eng|><|deu|>`).
    - Output: Translation text.
    - Establish a bridge connecting MT and S2TT.
    - Resume from the ASR checkpoint, training only the adapter.

   **Stage 3: SRT (Speech Recognition and Translation)**  
    - Goal: Ultimately activate end-to-end S2TT capabilities.
    - Input: Speech + task instructions (e.g., `<|eng|><|deu|>`).
    - Output: Transcription + translation (e.g., `Will it rain tomorrow?<|eng|><|deu|>Regnet es morgen?`).
    - Resume from the SMT checkpoint, optionally unfreezing the LLM (LoRA).

2. **Minimalist Instruction Design**

    - ASR: `<|eng|>` denotes "transcribing English".
    - SMT: `transcription<|source_lang|><|target_lang|>` denotes "translating this".
    - SRT: `<|source_lang|><|target_lang|>` denotes "recognize first and then translate".
    - Language tags appear simultaneously in both instructions and generated targets, naturally separating transcription and translation content.
    - Design Motivation: Reduce instruction token length to improve efficiency.

3. **Speech Adapter Optimization**

    - Q-Former compresses variable-length speech features into a fixed length of 80 queries ($\mathbf{Q'} \in \mathbb{R}^{n_q \times D_q}$).
    - The MLP maps the Q-Former output to the LLM hidden space dimension ($\mathbf{E^X} \in \mathbb{R}^{n_q \times d_{LLM}}$).
    - Compared to the variable-length features of Qwen2-Audio, the fixed length significantly reduces the number of LLM input tokens.
    - Inference speed is increased by about 3x, and larger batch sizes are supported.

### Loss & Training

- Three-stage progressive training, where each stage resumes from the checkpoint of the previous stage.
- The speech encoder is frozen at all times.
- Stages 1-2: Train only the adapter.
- Stage 3: Optionally extra unfreeze the LLM via LoRA.
- Trained using bf16 and DDP, with $lr=1e-4$ and the AdamW optimizer.
- 4x A100 GPUs: ~3 days for 3B/7B models, ~7 days for 32B model.

## Key Experimental Results

### Main Results — BLEU across 6x12 Directions on FLEURS

| Model | S2TT Data | Eng | Deu | Fra | Jpn | Zho | Average |
|------|-----------|-----|-----|-----|-----|-----|------|
| Whisper+Qwen2.5-32B (Cascade) | - | 29.9 | 26.0 | 24.6 | 17.3 | 18.5 | 23.3 |
| SeamlessM4T-V2 (2.3B) | 351K Hours | 33.1 | 20.5 | 19.6 | 13.2 | 15.2 | 20.2 |
| Qwen2-Audio (7B) | Internal | 22.6 | 20.1 | 20.6 | 4.0 | 13.7 | 16.0 |
| Baseline-3B | 52 Hours | 11.8 | 9.0 | 9.5 | 5.2 | 6.2 | 8.6 |
| **LLM-SRT-3B** | **52 Hours** | **27.2** | **22.6** | **22.0** | **14.3** | **16.5** | **20.6** |
| **LLM-SRT-32B** | **52 Hours** | **32.5** | **26.8** | **26.1** | **17.5** | **19.2** | **24.6** |

Using only 52 hours of data, LLM-SRT-3B (BLEU 20.6) outperforms SeamlessM4T-V2 (20.2), which was trained on 351K hours of data.

### Ablation Study — Effect of Each Stage in Curriculum Learning (CoVoST-2)

| Setting | Deu | Jpn | Zho | Average |
|------|-----|-----|-----|------|
| LLM-SRT-7B (Full) | 28.7 | 41.6 | 47.1 | 39.1 |
| w/o ASR | 26.4 (-2.3) | 38.6 (-3.0) | 45.5 (-1.6) | 36.8 (-2.3) |
| w/o SMT | 27.6 (-1.1) | 39.7 (-1.9) | 46.5 (-0.6) | 38.0 (-1.1) |
| w/o SRT | 25.6 (-3.1) | 36.7 (-4.9) | 40.4 (-6.7) | 34.2 (-4.9) |

Removing the SRT stage has the most critical impact (-4.9), verifying the key role of extending MT capabilities to S2TT.

### Inference Speed Comparison

| Model | Strategy | Batch | Time to Process 1000 Items |
|------|------|-------|-------------|
| Qwen2-Audio | Greedy | 4 | 59s |
| Qwen2-Audio | Greedy | 8 | OOM |
| LLM-SRT-7B | Greedy | 4 | 74s |
| LLM-SRT-7B | Greedy | 64 | 19s |
| LLM-SRT-7B | Beam 5 | 12 | 56s |

Inference speed of LLM-SRT under a large batch size is approximately **3x** that of Qwen2-Audio, and it does not trigger OOM.

### Key Findings

1. **Significant Impact of Curriculum Learning**: Direct fine-tuning of Baseline-3B yields only 8.6 BLEU, whereas LLM-SRT-3B improves this to 20.6 (+140%) via the three-stage strategy.
2. **Conformation with LLM Scaling Laws**: Performance increases steadily from 3B to 7B to 32B (20.6 → 21.4 → 24.6).
3. **S2TT Performance Strongly Correlated with MT**: BLEU scores between S2TT and MT across 210 translation directions show a strong positive correlation.
4. **Effective Scaling of Data Volume**: Scaling data from 52 to 430 hours improves the 3B model from 34.3 to 36.6, and the 7B model from 35.0 to 39.1.
5. **SRT Task Does Not Impair ASR**: SRT even slightly improves ASR performance (WER 11.1 → 10.9).
6. **SMT Task Validation**: Given ground-truth transcriptions, SMT achieves highly competitive BLEU scores (e.g., Zho 55.6), proving that the MT capabilities of the LLM are successfully activated.

## Highlights & Insights

- **Ingenity of Task Redefinition**: Reformulating S2TT into SRT allows the model to "transcribe first, then translate", naturally harnessing the MT capabilities of the LLM.
- **Extreme Data Efficiency**: Merely 52 hours of multilingual data (<10h per language) outperforms SeamlessM4T-V2, which was trained on 350K hours.
- **End-to-End Beats Cascade**: Under the same LLM, the end-to-end framework of LLM-SRT outperforms the Whisper+LLM cascade method.
- **Practical Value of Q-Former Compression**: Fixing the length to 80 queries not only boosts inference speed by 3x but also resolves the OOM issue when Qwen2-Audio uses batched inference with batch size > 4.
- **First Many-to-Many S2TT MLLM at 32B Scale**

## Limitations & Future Work

1. **Weaker Performance in English-to-X Directions**: FLEURS contains only ~10h per language; the lack of English data leads to lower performance on Eng→X than SeamlessM4T-V2.
2. **Frozen Speech Encoder**: The potential benefits of unfreezing Whisper weight have not been fully explored.
3. **Information Bottleneck of Q-Former**: Whether fixing length to 80 queries discards critical acoustic information warrants further research.
4. **Limitations of Evaluation Metrics**: Relying solely on BLEU without adopting more robust translation quality metrics like COMET.
5. **Streaming Inference**: The current method requires complete audio input and does not support real-time streaming translation.

## Related Work & Insights

- Relation to Qwen2-Audio: LLM-SRT adopts a similar architecture but significantly boosts performance using the curriculum learning strategy.
- Comparison with SeamlessM4T: While the latter relies on massive amounts of data (351K hours), LLM-SRT demonstrates outstanding data-efficiency advantages by exploiting the LLM's inherent translation capabilities.
- Insight: The "capability transfer" logic of curriculum learning (MT → S2TT) might be applicable to other cross-modality tasks (e.g., image captioning → video captioning).

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | The SRT task formulation and three-stage curriculum learning strategy are novel and effective. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Comprehensive experiments conducted across multiple datasets, model sizes, translation directions, and detailed ablation studies. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure and rich tabular configurations. |
| Value | ⭐⭐⭐⭐⭐ | Holds significant practical value for low-resource multilingual speech translation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] Byte Latent Transformer: Patches Scale Better Than Tokens](byte_latent_transformer.md)

</div>

<!-- RELATED:END -->
