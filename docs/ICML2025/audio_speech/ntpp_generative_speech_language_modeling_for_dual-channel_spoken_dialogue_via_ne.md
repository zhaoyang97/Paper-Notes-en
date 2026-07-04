---
title: >-
  [Paper Note] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
description: >-
  [ICML2025][Audio & Speech][spoken dialogue] Proposes the Next-Token-Pair Prediction (NTPP) paradigm, which models the joint distribution of dual-channel spoken dialogue in a speaker-independent manner using a decoder-only architecture for the first time, achieving more natural turn-taking, lower inference latency, and stronger speaker independence.
tags:
  - "ICML2025"
  - "Audio & Speech"
  - "spoken dialogue"
  - "dual-channel speech"
  - "Next-Token-Pair Prediction"
  - "decoder-only"
  - "turn-taking"
  - "full-duplex dialogue"
date: 2026-05-08
content_hash: e402296e6b53e008
---

# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction

**Conference**: ICML2025  
**arXiv**: [2506.00975](https://arxiv.org/abs/2506.00975)  
**Code**: [Demo & Code](https://audio-3059.pages.dev)  
**Area**: speech_language_model  
**Keywords**: spoken dialogue, dual-channel speech, Next-Token-Pair Prediction, decoder-only, turn-taking, full-duplex dialogue

## TL;DR

Proposes the Next-Token-Pair Prediction (NTPP) paradigm, which models the joint distribution of dual-channel spoken dialogue in a speaker-independent manner using a decoder-only architecture for the first time, achieving more natural turn-taking, lower inference latency, and stronger speaker independence.

## Background & Motivation

- **Core Problem**: How to make speech language models (SLMs) as natural as human-to-human conversation—supporting full-duplex interactions including overlaps, interruptions, and pauses?
- **Limitations of Prior Work**:
    - **Single-channel methods** (SpeechGPT, LLama-Omni, etc.): Model the turn-based $p(S^b|S^a)$, requiring VAD for turn segmentation, failing to handle real-time streaming dialogue.
    - **dGSLM**: Models the joint distribution $p(S^a, S^b)$, but uses a Siamese encoder-decoder dual-tower architecture, which has low parameter efficiency.
    - **LSLM**: Decoder-only, but only predicts one channel (conditional distribution), lacking speaker independence.
    - **Moshi**: RQ-transformer that relies on an extra encoder and requires maintaining two KVCaches, resulting in poor inference efficiency.
- **Motivation**: Dual-channel speech naturally contains conversational dynamics such as overlaps, pauses, gaps, and interruptions (as shown in Figure 1), which are mixed and indistinguishable in a single channel. Can a pure decoder-only architecture be used to directly model the dual-channel joint distribution?

## Method

### 1. NTPP Modeling Paradigm

Core Idea: Simultaneously predict the token pair $(s_t^a, s_t^b)$ of both speakers at each time step $t$ to learn the joint distribution:

$$p(S^a, S^b) = \prod_{t=1}^{T} p(s_t^a, s_t^b \mid s_{t-1}^a, \ldots, s_1^a, s_{t-1}^b, \ldots, s_1^b)$$

Decompose the pair through the **conditional independence assumption**:

$$p(s_t^a, s_t^b \mid \cdot) = p(s_t^a \mid s_{t-1:1}^a, s_{t-1:1}^b) \cdot p(s_t^b \mid s_{t-1:1}^a, s_{t-1:1}^b)$$

Key inductive bias: A speaker's utterance is jointly influenced by what they previously said and what they heard from the counterpart.

Comparison with existing methods (Table 1): NTPP is the only method that simultaneously satisfies four conditions: **Speaker-Independent + Encoder-Free + VAD-Free + Single KVCache**.

### 2. Autoregressive Dual-channel Speech Transformer

The dual-channel sequence is arranged in an interleaved style: $S = ((s_1^a, s_1^b), (s_2^a, s_2^b), \ldots, (s_T^a, s_T^b))$, predicting a pair of tokens at each step. Only two lightweight modifications are made to the standard decoder-only Transformer:

**Token Pair Embedding**: Each token pair contains three types of embeddings:
- **Codebook embedding** $\mathbf{z}_t$: Retrieved via table lookup from the VQ/RVQ codebook.
- **Positional embedding** $\mathbf{p}_t$: Adopted from RoPE, where $s_t^a$ and $s_t^b$ at the same time step share the same positional encoding.
- **Channel embedding** $\mathbf{c}_t$: One-hot encoding to distinguish speaker a/b.

Calculation of Query and Key: $\mathbf{q} = \mathbf{W}_Q[\mathbf{z}_t^a, \mathbf{z}_t^b] + [\mathbf{p}_t^a, \mathbf{p}_t^b] + [\mathbf{c}_t^a, \mathbf{c}_t^b]$

**Pair-wise Causal Masking**: The causal mask matrix $\mathbf{M} \in \mathbb{R}^{2T \times 2T}$ uses a $2 \times 2$ block strategy on its diagonal: $s_t^a$ and $s_t^b$ within the same time step are **invisible to each other** (leaving only themselves on the diagonal) to ensure conditional independence.

### 3. Generalization to RVQ Tokenizer

RVQ unpacks each time step into $D$ depth tokens, rendering the sequence as $(s_{t,1}^a, \ldots, s_{t,D}^a, s_{t,1}^b, \ldots, s_{t,D}^b)$. The following are introduced:
- **Cyclic Depth Embedding**: $\mathbf{d} = (\sin(2\pi i / D), \cos(2\pi i / D))$, which cycles with period $D$, helping the model recognize the depth position of the current token.
- **RVQ Causal Masking**: In the $2D \times 2D$ block diagonal, an upper triangular mask is applied (shallow layers do not look at deep layers), and the bottom-left $D \times D$ submatrix is fully masked (channels a/b are invisible to each other).

### 4. Streaming Conditional Inference

To accommodate real-time interaction, chunk-wise streaming inference is adopted: Given $\lambda$ user input tokens, the model begins generating $\lambda$ response tokens, repeating the process cyclically.

## Experimental Setup & Main Results

### Training Setup

- **Two-stage Training**: Stage 1 trains the base SLM using about 140,000 hours of single-channel speech (textless, without text alignment). Stage 2 performs NTPP fine-tuning using the Fisher dataset (2200 hours of dual-channel telephone dialogue).
- **Audio tokenizer**: Train an RVQ tokenizer based on SoundStream, achieving 40 tokens per second with a codebook size of 4096.
- **LLM Backbone**: LLaMA 3.1-8B / Mistral-7B / Gemma-2-9B, among which LLaMA 3.1 converges the fastest.
- **Training Hyperparameters**: 16 × A100, cosine scheduler (lr: 4e-6 → 4e-4), batch size 64, 40k steps per epoch.
- **Ablation Finding**: Audio-only training (w/o Text) converges faster and yields lower perplexity than adding ASR text (w Text), indicating that text transcripts might introduce modal interference.

### Main Results

**1. Turn-Taking Event Distribution (Table 2)**  
Comparing turn-taking statistics (IPU count/duration, Pause, Gap, Overlap) between generated dialogues and real dialogues on the Fisher test set, evaluated by |Δ| (mean absolute difference from ground truth). NTPP achieves the best balance at a temperature of 0.9:

| Model | IPU Count \|$\Delta$\| | Pause Count \|$\Delta$\| | Gap Count \|$\Delta$\| | Overlap Count \|$\Delta$\| |
|------|---------|---------|---------|----------|
| Cascaded | 4.1 | 7.0 | 7.4 | 6.5 |
| dGSLM w/o CA | 3.9 | 2.9 | 3.6 | 1.0 |
| dGSLM | 1.6 | 3.4 | 2.0 | 2.9 |
| LSLM | 2.2 | 3.6 | 2.4 | 3.2 |
| **NTPP (t=0.9)** | **1.3** | **2.3** | **1.5** | **0.9** |

NTPP leads significantly in the Overlap metric, indicating that it better captures overlap dynamics in conversation.

**2. Human Evaluation (Table 3)**  
25 annotators evaluated meaningfulness (M-MOS) and naturalness (N-MOS) on a 5-point MOS scale:

| Model | M-MOS (Overall) | N-MOS (Overall) | M-MOS (Fisher) | N-MOS (Fisher) | M-MOS (CANDOR) | N-MOS (CANDOR) |
|------|--------|--------|--------|--------|--------|--------|
| dGSLM | 1.38 | 3.85 | 1.82 | 4.10 | 1.51 | 2.85 |
| SyncLLM | 3.85 | 4.10 | 4.10 | 4.33 | 3.85 | 3.91 |
| Moshi | 3.90 | 3.95 | 3.20 | 3.90 | 3.95 | 3.95 |
| **NTPP** | **3.95** | **4.15** | **4.10** | **4.42** | **4.05** | **4.05** |
| GT | 4.90 | 4.95 | 4.90 | 4.90 | 4.90 | 4.95 |

NTPP achieves the highest M-MOS and N-MOS among all baselines, with a naturalness of 4.42 on Fisher in-domain data.

**3. Speaker Independence (Table 4)**  
Measure metric changes $|\Delta M_{\text{original}} - \Delta M_{\text{swapped}}|$ after swapping dual-channel inputs (lower is more robust):
- dGSLM and NTPP exhibit changes close to 0 on the training set and remain low on the test set (all NTPP metrics < 0.45).
- Moshi shows significant changes (Gap duration deviation reaches 0.84), suggesting its dependence on speaker-conditioned generation.

**4. Inference Latency (Figure 7)**  
NTPP consistently maintains a response latency of under 220ms as dialogue turns increase, whereas Moshi grows linearly. Reason: NTPP only requires a single KVCache, while Moshi requires two.

**5. Ablation Study (Figure 8)**  
- Two-stage training vs. single-stage: Removing Stage 1 (single-channel pretraining) or Stage 2 (NTPP fine-tuning) both lead to a significant increase in perplexity.
- RVQ vs. VQ: RVQ tokenizer training loss is consistently lower than VQ.

## Highlights & Insights

- **Elegant Paradigm Shift in Modeling**: Shifted from conditional distribution $p(S^b|S^a)$ to joint distribution $p(S^a, S^b)$, elegantly implemented within a decoder-only architecture via pair-wise causal masking, presenting a minimal but highly effective modification.
- **Deployment-Friendly**: Single KVCache + No extra encoder + No VAD module = simple deployment and highly efficient inference.
- **Reasonable Conditional Independence Assumption**: Although the conditional independence of the two channels in the same time step seems to discard information, it actually leverages the conversational prior that "speech is influenced by historical context." Experiments demonstrate that this does not compromise performance.
- **Better Performance with Textless Training**: Ablation studies show that not adding ASR texts leads to better results, suggesting that speech tokens themselves contain sufficient semantic information, and text alignment might introduce noise.

## Limitations & Future Work

1. **Data Bottleneck**: Dual-channel speech data is scarce, with Fisher containing only 2200 hours. Scaling up to larger multi-channel datasets remains a core challenge.
2. **Limitations of the Conditional Independence Assumption**: The conditional independence assumption of $s_t^a$ and $s_t^b$ at the same time step may not hold in highly interactive scenarios (e.g., heated arguments).
3. **Limited to Dual-Channel**: Not extended to multi-party conversation (>2 speakers) scenarios.
4. **Limited Evaluation Dimensions**: Lacks objective metrics such as speech quality (MOS for speech quality) and semantic accuracy (WER).
5. **Potential Misuse Risks**: The paper mentions risks of being used in scenarios such as telecommunication fraud, and the discussion on safety mechanisms is insufficient.

## Related Work & Insights

- **dGSLM** (Nguyen et al., 2023): The first dual-channel textless dialogue generation model, utilizing a dual-tower Siamese encoder-decoder. NTPP replaces it uniformly with a decoder-only architecture.
- **LSLM** (Ma et al., 2024): Uses a token fusion strategy to merge dual channels but only models conditional distribution. It proves the feasibility of using a decoder-only model to process dual channels.
- **Moshi** (Défossez et al., 2024): RQ-transformer + text alignment—a full-stack solution with poor inference efficiency. NTPP holds a structural advantage in latency.
- **SyncLLM** (Veluri et al., 2024): A full-duplex conversational agent that achieves overlapping speech through streaming. It is close to NTPP in human evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The pair-wise prediction paradigm is novel, and the conditional independent decomposition is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple dimensions (turn-taking/human/speaker-independence/latency/ablation), but lacks objective metrics for speech quality.
- Writing Quality: ⭐⭐⭐⭐ — Formulations are derived clearly, the diagrams are rich, and the motivation is thoroughly explained.
- Value: ⭐⭐⭐⭐ — Great practical significance (real-time voice interaction), simple and reproducible method, though data bottlenecks limit its impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Aligning Spoken Dialogue Models from User Interactions](aligning_spoken_dialogue_models_from_user_interactions.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](long-form_speech_generation_with_spoken_language_models.md)

</div>

<!-- RELATED:END -->
