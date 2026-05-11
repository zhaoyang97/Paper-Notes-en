---
title: >-
  [Paper Note] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
description: >-
  [ICLR 2026][LLM Pretraining][speech tokenization] This paper proposes TASTE (Text-Aligned Speech Tokenization and Embedding), which aligns speech tokens with text transcriptions via a cross-attention mechanism…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "speech tokenization"
  - "spoken language model"
  - "text-speech alignment"
  - "joint modeling"
  - "speech reconstruction"
date: 2026-05-08
content_hash: 365634661ebfe672
---

# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

**Conference**: ICLR 2026
**arXiv**: [2504.07053](https://arxiv.org/abs/2504.07053)
**Code**: [GitHub](https://mtkresearch.github.io/TASTE-SpokenLM.github.io)
**Area**: LLM Pretraining
**Keywords**: speech tokenization, spoken language model, text-speech alignment, joint modeling, speech reconstruction

## TL;DR

This paper proposes TASTE (Text-Aligned Speech Tokenization and Embedding), which aligns speech tokens with text transcriptions via a cross-attention mechanism, enabling high-quality speech reconstruction at an extremely low bitrate (~150 bps). This design makes text-speech joint modeling straightforward and efficient; the resulting 1.3B-parameter TASLM outperforms 7B pretrained SLMs.

## Background & Motivation

Speech tokenization is the central challenge in Spoken Language Modeling (SLM). Existing approaches suffer from two major problems:

**Length mismatch**: Speech token sequences are typically 10–50× longer than their corresponding text (e.g., ~50 Hz vs. ~3 Hz), making joint modeling difficult.

**Information redundancy**: Existing speech tokens (SSL-quantized or codec-based) are extracted independently of text, inevitably encoding overlapping information with text tokens.

Common mitigation strategies include:
- Token interleaving (Spirit LM)
- Padding to synchronize sequence lengths (Moshi, MiniOmni)
- Additional alignment training stages

All of these approaches add complexity and are fundamentally remedial patches applied after tokenization.

The core idea of TASTE is to **resolve the alignment problem at the tokenization stage itself**. Speech tokens should:
1. Avoid redundantly encoding textual content (already carried by text tokens) and instead focus on paralinguistic information.
2. Correspond one-to-one with text tokens, enabling joint modeling without heuristic rules or explicit alignment.

## Method

### Overall Architecture

TASTE consists of two main components:

1. **Text-aligned speech tokenizer** (Section 3.1.1): Encodes speech into speech tokens of the same length as text tokens.
2. **Speech decoder** (Section 3.1.2): Reconstructs speech from text tokens and the aligned speech tokens.

### Key Designs

**Tokenizer — three sub-modules:**

**Encoder**: A frozen Whisper ASR encoder that extracts hidden representations from two layers:
- Last layer $\mathbf{h}^{(L)}$: rich text-speech alignment cues.
- Shallow layer $\mathbf{h}^{(l)}$ (first half of layers): supports high-quality speech reconstruction.

**Aggregator**: The core innovation — a cross-attention mechanism for text-aligned aggregation:

$$Q = \text{text transcription } \mathbf{v}, \quad K = \text{encoder last layer } \mathbf{h}^{(L)}, \quad V = \text{encoder shallow layer } \mathbf{h}^{(l)}$$

Intuition: the last-layer representations serve as Keys to provide alignment cues, guiding attention to aggregate acoustic information from the shallow-layer Values. The output length naturally follows the text transcription (Query), yielding $\mathbf{z} \in \mathbb{R}^{N \times d_z}$ aligned to the $N$ text tokens.

**Quantizer**: Residual Vector Quantization (RVQ) for discretization:

$$\mathbf{q}, \hat{\mathbf{z}} = \text{Quantizer}(\mathbf{z}), \quad \mathbf{q} = [\mathbf{q}^{(1)}, \ldots, \mathbf{q}^{(R)}], \quad \hat{\mathbf{z}} = \sum_{r=1}^R \hat{\mathbf{z}}^{(r)}$$

$R=4$ RVQ layers are used, with codebook size 512 and dimension 256.

**Speech decoder**: A Transformer decoder predicts speech units, followed by a flow model + HiFiGAN to synthesize the waveform:

$$\mathbf{y} = \text{UnitDecoder}(\hat{\mathbf{z}}, \mathbf{v})$$

### Loss & Training

The overall training objective is the sum of reconstruction and quantization losses:

$$\mathcal{L}_{\text{taste}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{rvq}}$$

Cross-entropy reconstruction loss:

$$\mathcal{L}_{\text{ce}}(\theta) = \frac{1}{|T'|} \sum_{t=1}^{T'} -\log p_\theta(y_t^{\text{target}} | \hat{\mathbf{z}}, \mathbf{v}; \mathbf{y}_{<t}^{\text{target}})$$

Quantization commitment loss:

$$\mathcal{L}_{\text{rvq}}(\theta) = \sum_{r=1}^R \|\mathbf{z}^{(r)} - \hat{\mathbf{z}}^{(r)}\|$$

**Joint language model training** — two variants:

- **Token mode** ($\text{TASLM}_{\text{token}}$): Multi-head prediction jointly predicting the next text token and $R$ RVQ codes.
- **Embedding mode** ($\text{TASLM}_{\text{emb}}$): Predicts continuous embeddings $\mu_i, \sigma_i$ with a regularization and KL divergence loss.

## Key Experimental Results

### Main Results

**Speech reconstruction quality** (LibriSpeech test-clean):

| Method | Freq. | Bitrate | WER↓ | UTMOS | DNSMOS | ViSQOL | Duration Consistency | Speaker Similarity | MUSHRA |
|--------|-------|---------|------|-------|--------|--------|----------------------|--------------------|--------|
| Encodec (75Hz, 2RVQ) | 75 | 3000 | 2.6% | 2.35 | 3.48 | 3.81 | 0.96 | 0.78 | 25.6 |
| SpeechTokenizer (2RVQ) | 50 | 2000 | 3.0% | 3.56 | 3.60 | 3.65 | 0.97 | 0.80 | 53.9 |
| Mimi | 12.5 | 1000 | 3.1% | 3.60 | 3.60 | 3.62 | 0.96 | 0.82 | 67.6 |
| S3 token (topline) | 25 | 600 | 3.0% | 4.18 | 3.90 | 3.30 | 0.96 | 0.82 | 70.2 |
| Text-only (baseline) | ~3 | ~50 | 5.9% | 4.31 | 4.11 | 2.44 | 0.57 | 0.78 | 42.6 |
| **TASTE** | **~3** | **~150** | **4.4%** | **4.29** | **4.10** | 3.05 | **0.91** | 0.80 | **68.3** |

TASTE achieves quality comparable to or better than high-bitrate methods at the **lowest frequency and bitrate**.

**Spoken language model performance** (speech continuation + likelihood evaluation):

| Method | Params | GPT-4o | UTMOS | Human MOS | SALMON | StoryCloze | Overall |
|--------|--------|--------|-------|-----------|--------|-----------|---------|
| TWIST 7B | 7B | 1.44 | 3.27 | 2.04 | 63.4 | 64.7 | 64.1 |
| Spirit LM 7B | 7B | 2.79 | 3.41 | 2.38 | 59.1 | 72.0 | 65.6 |
| Spirit LM Expr. 7B | 7B | 1.90 | 3.40 | 2.41 | 69.0 | 66.2 | 67.6 |
| **TASLM 1B (token)** | **45M/1.3B** | **3.08** | **4.07** | **3.93** | 60.8 | **76.5** | **68.7** |
| **TASLM 1B (embed.)** | **45M/1.3B** | **3.16** | **4.22** | **4.16** | 57.7 | 76.7 | 67.2 |

With only LoRA fine-tuning, the 1.3B TASLM **comprehensively outperforms 7B-scale pretrained SLMs** on continuation evaluation.

### Ablation Study

**Tokenizer module ablation** (S3 token top-5 reconstruction accuracy):

| Module | Frequency | Accuracy |
|--------|-----------|----------|
| Encoder only | 50Hz | 0.98 |
| Encoder + Aggregator | ~3Hz | 0.88 |
| Encoder + Agg + Quantizer | ~3Hz | 0.76 |
| Encoder (last layer only) | 50Hz | 0.84 |
| Encoder + Agg (last layer only) | ~3Hz | 0.78 |
| Text-only | ~3Hz | 0.65 |

Key findings:
- The aggregator reduces frequency from 50 Hz to ~3 Hz with only a 0.10 drop in accuracy.
- Using shallow representations as Values (0.88) outperforms using only the last layer (0.78).
- After quantization, accuracy remains well above the text-only baseline (0.76 vs. 0.65).

### Key Findings

1. **Core value of text-aligned tokenization**: Directly using S3 tokens for joint modeling performs very poorly despite better reconstruction quality, demonstrating that tokenization design matters more for joint modeling than reconstruction quality alone.
2. **TASTE makes joint modeling straightforward**: No interleaving, padding, or delayed decoding tricks are needed; one-to-one correspondence suffices.
3. **TASTE enables text-aligned speech editing**: Swapping TASTE tokens between two utterances with identical transcriptions precisely exchanges the paralinguistic features (e.g., duration, prosody) of the corresponding words.
4. **Few-shot spoken QA capability**: TASLM is the only pretrained SLM that demonstrates few-shot spoken question-answering ability.
5. **TASLM is the only SLM that preserves or exceeds the text LLM backbone's performance**.

## Highlights & Insights

1. **Elegant design philosophy**: Rather than patching length mismatches at the joint modeling stage, TASTE addresses the root cause at tokenization — exemplifying the principle of solving problems at the correct level of abstraction.
2. **Ingenious K/V separation**: Using the last encoder layer as Keys provides alignment priors while shallow layers as Values supply acoustic information — each component serving a distinct role.
3. **Extremely low bitrate** (~150 bps vs. typical 1000+ bps) confirms that text already carries the vast majority of information, and speech tokens need only encode "residual" paralinguistic information.
4. **LoRA suffices**: Full-parameter training is unnecessary; a 1.3B model trained with LoRA surpasses fully trained 7B baselines.
5. **Speech editing experiments** directly verify that TASTE tokens encode word-level paralinguistic information rather than semantic content.

## Limitations & Future Work

1. Validated only on English data; multilingual generalization is unknown.
2. Lacks conversational turn-taking and instruction-following capabilities.
3. Only handles single-speaker speech with lexical content; multi-speaker speech, overlapping speech, and non-lexical events are not covered.
4. Dependent on ASR quality — ASR errors cascade into TASTE tokens.
5. The tokenization scheme is designed specifically for joint SLM; applicability to purely generative speech tasks (e.g., TTS) has not been explored.

## Related Work & Insights

- Compared to Moshi (Défossez et al., 2024): Moshi trains a custom codec to reduce frame rate, whereas TASTE achieves alignment more fundamentally through text-guided tokenization.
- Compared to the interleaving strategy of Spirit LM (Nguyen et al., 2025): TASTE's one-to-one correspondence is more natural.
- Inspiration: The idea of joint tokenization may generalize to other multimodal settings (e.g., video + text, music + score).
- The shallow/deep layer information separation architecture may also be valuable in other Transformer encoder contexts.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first end-to-end text-aligned speech tokenization scheme; resolves joint modeling challenges at the tokenization level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional evaluation covering reconstruction, language modeling, editing, and QA, with complete ablations.
- **Practicality**: ⭐⭐⭐⭐ — Code and models are open-sourced; LoRA-compatible; however, ASR dependency remains a constraint.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation.
- **Overall**: ⭐⭐⭐⭐⭐ (4.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICLR 2026\] Imagine How To Change: Explicit Procedure Modeling for Change Captioning](imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)
- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)

</div>

<!-- RELATED:END -->
