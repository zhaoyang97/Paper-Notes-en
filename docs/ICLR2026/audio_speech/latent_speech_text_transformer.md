---
title: >-
  [Paper Note] Latent Speech-Text Transformer
description: >-
  [ICLR 2026 Oral][Audio & Speech][speech-text modeling] The authors propose Latent Speech-Text Transformer (LST), which aggregates discrete speech tokens into higher-level "latent speech patches" as autoregressive units (similar to how BLT processes bytes). This aligns the sequence modeling granularity of speech and text (reducing from 20× to ~1:1), achieving a +6.5% absolute gain on Speech HellaSwag with sustained gains from 420M to 7B parameters…
tags:
  - "ICLR 2026 Oral"
  - "Audio & Speech"
  - "speech-text modeling"
  - "latent patches"
  - "autoregressive"
  - "ASR"
  - "TTS"
  - "cross-modal alignment"
  - "BLT"
date: 2026-05-08
content_hash: d6af360c547fd2fa
---

# Latent Speech-Text Transformer

**Conference**: ICLR 2026 Oral  
**arXiv**: [2510.06195](https://arxiv.org/abs/2510.06195)  
**Code**: [GitHub](https://github.com/facebookresearch/lst)  
**Area**: Audio & Speech  
**Keywords**: speech-text modeling, latent patches, autoregressive, ASR, TTS, cross-modal alignment, BLT

## TL;DR
The authors propose Latent Speech-Text Transformer (LST), which aggregates discrete speech tokens into higher-level "latent speech patches" as autoregressive units (similar to how BLT processes bytes). This aligns the sequence modeling granularity of speech and text (reducing from 20× to ~1:1), achieving a +6.5% absolute gain on Speech HellaSwag with sustained gains from 420M to 7B parameters, while simultaneously reducing ASR/TTS inference computational costs.

## Background & Motivation
**Background**: Discrete speech tokens (e.g., HuBERT 25Hz, 501 codebook) have enabled autoregressive speech LMs. However, speech token sequences are significantly longer than corresponding text (10-20×), leading to training and inference efficiencies far below those of text LLMs—it is estimated that speech requires three orders of magnitude more data to reach equivalent capability.

**Limitations of Prior Work**:
   - **Information Density Mismatch**: The severe asymmetry in sequence length between speech and text tokens hinders cross-modal knowledge transfer.
   - **Unequal Compute Allocation**: During pre-training and inference, most computation is spent on long speech sequences rather than meaningful semantic modeling.
   - **Insufficient Alignment Attempts**: While warm initialization (from text LLMs) and interleaved training help, a significant performance gap remains between speech-to-speech and text-to-text.
   - BPE fails on speech tokens (reported by Cuervo & Marxer 2024)—simple sub-word segmentation is unsuitable for speech.

**Key Challenge**: Speech modeling requires fine-grained tokens (25Hz), but autoregressive modeling is inefficient over long sequences and exhibits poor cross-modal alignment.

**Core Idea**: Borrowing the concept from Byte Latent Transformer (BLT)—aggregating speech tokens into "latent patches" (high-level autoregressive units). A global Transformer models at the patch level, while a lightweight decoder expands patches back into speech tokens. Patch granularity is aligned with text tokens.

## Method

### Overall Architecture
LST first uses a lightweight Patch Encoder to aggregate a discrete speech token sequence $\{s_0,\ldots,s_T\}$ into a significantly smaller number of "latent speech patches" $\{z_0,\ldots,z_{T'}\}$ ($T'\ll T$). This allows the primary Global Transformer to perform unified autoregression on units of comparable granularity (patches and text tokens). Finally, a lightweight Patch Decoder restores each patch into speech tokens and calculates standard NTP loss. The architecture follows the BLT strategy—"fine-grained token $\rightarrow$ high-level patch $\rightarrow$ global modeling $\rightarrow$ decoder expansion"—simply replacing bytes with speech tokens, thereby compressing the 20× length difference relative to text down to approximately 1:1.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}%%
flowchart TD
    A["Discrete Speech Token Sequence<br/>HuBERT 25Hz · 501 Codebook"] --> PA
    subgraph PAT["Curriculum Patching (Design 1)"]
        direction TB
        PA["Partition by Alignment/Static boundaries<br/>P(u) linearly decays from 1 to 0"] --> PB["Aggregate into Latent Speech Patches<br/>Seq length ≈ 1/p of original"]
    end
    PB --> ENC
    subgraph ARCH["Lightweight Encoder/Decoder + Heavy Global Transformer (Design 2)"]
        direction TB
        ENC["Lightweight Patch Encoder<br/>Sliding Window Self-attn + Cross-attn"] --> GT["Global Transformer<br/>Patch-level Autoregressive Backbone"]
        GT --> DEC["Lightweight Patch Decoder<br/>Expand Patch to Speech Tokens"]
    end
    TX["Interleaved Text Tokens<br/>Llama 2 tokenizer"] -->|Patch-level Cross-modal Alignment (Design 3)| GT
    DEC --> L["Token-level NTP Loss"]
```

### Key Designs

**1. Three Patching Strategies and Curriculum Transition: Balancing "Semantic Alignment" and "Inference Independence"**

How patches are segmented determines whether speech granularity can match text. The simplest, Static Patching, uses a fixed size $p$ for non-overlapping segmentation (e.g., $p=3$, every 3 speech tokens form one patch), which is efficient and requires no external models during inference, but segment points are unrelated to semantic boundaries. Alignment Patching uses Wav2Vec2+CTC forced alignment to obtain speech-text timestamps, making each text unit (word/BPE) correspond to one patch (with silences forming their own patches). While alignment quality is superior, it requires an alignment model during inference, which is impractical. LST’s final solution is Curriculum Patching: training begins using alignment to leverage semantic correspondence, then the probability of using alignment segmentation $P(u)$ linearly decays from $1$ to $0$ over a training step interval $[\tau_1, \tau_2]$, smoothly switching to static. This "distills" the alignment signal into the patch representation early on, allowing the model to retain learned semantic correspondence even after switching to static segmentation—achieving alignment quality without inference-time dependency. Notably, directly applying BPE sub-word segmentation to speech tokens fails (Cuervo & Marxer 2024), which is why LST utilizes the patching approach instead of vocabulary compression.

**2. Lightweight Encoder / Decoder + Heavy Global Transformer: Spending Compute on Patch-level Semantics rather than Redundant Tokens**

The Patch Encoder consists of sliding window self-attention and cross-attention, aggregating token embeddings within a window into a single patch embedding. The Patch Decoder is a lightweight Transformer that receives patch information via cross-attention in each layer, restoring speech tokens within a 512-token self-attention window. The key lies in compute allocation: major FLOPs are concentrated in the patch-level Global Transformer, while the Encoder and Decoder are kept lightweight. Since global modeling occurs on a patch sequence that is only $1/p$ of the original length, the primary autoregressive overhead drops significantly, and long-distance dependencies are easier to learn. This is why LST reduces ASR/TTS inference costs without sacrificing reconstruction quality.

**3. Patch-level Cross-modal Alignment: Consistent Granularity within the Same Sequence**

By compressing speech into patches, its sequence length becomes consistent with text tokens, allowing both to be treated equally in the same autoregressive sequence. Training employs interleaved data—alternating speech and text segments from the same corpus—where some speech segments are replaced by their corresponding text. This forces the model to predict across modalities. Consequently, patches automatically learn correspondences with syllables/words, opening a channel for speech$\leftrightarrow$text knowledge transfer. This explains why cross-modal training does not degrade text performance but instead slightly improves text-to-text results.

### Loss & Training
The entire model is trained end-to-end (Patch Encoder + Global Transformer + Patch Decoder optimized together). The objective is the standard token-level NTP loss applied to the Patch Decoder output. HuBERT 25Hz with a 501-codebook tokenizer is used for speech discretization, while the Llama 2 tokenizer is used for text.

## Key Experimental Results

### Main Results (Speech HellaSwag, story completion)

| Setting | Condition | LST Gain |
|------|------|---------|
| Compute-controlled (Equal training steps) | 420M | +6.5% absolute |
| Data-controlled (Equal data volume) | 420M | +5.3% absolute |
| Compute-optimal scaling | 420M → 1.8B | **Gains grow with scale** |
| Fixed-token budget | 7B, 70B tokens | Sustained gains |

Key Insight: Gains do not saturate; they **continue to grow as the model size increases**, indicating that LST improves compute-optimal scaling.

### Downstream Tasks

| Task | Effect | Description |
|------|------|------|
| ASR Adaptation | More Stable | Patch-level modeling reduces long-distance dependency issues |
| TTS Inference | Shorter Sequence, Lower Compute | Direct benefit of sequence length compression |
| Reconstruction Quality | No Degradation | Proves patch compression is lossless |
| Text→Text | Improvement | Cross-modal training inversely improves text capability |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| No patching (baseline) | Baseline | Standard interleaved training |
| BPE on speech tokens | No improvement/Degradation | Confirms BPE is unsuitable for speech tokens |
| Static patching (p=3) | Significant improvement | Even simple segmentation is effective |
| Alignment patching | Best but requires aux model | The value of semantic alignment |
| **Curriculum patching** | **Best Balance** | Retains alignment benefits + No inference aux model |

### Key Findings
- **Information Density Alignment is Core**: Bringing speech and text to similar sequence lengths significantly improves cross-modal knowledge transfer—supporting the hypothesis that granularity mismatch is a primary bottleneck.
- Even simple static patching is effective—suggesting the problem is less about precise semantic alignment and more about reducing redundancy in speech sequences.
- **Patches Automatically Learn Semantic Correspondence**: Curriculum patching starts with alignment but switches to static; the model maintains learned correspondences, showing alignment signals can be "distilled" into patch representations.
- Gains grow with model scale—this has important implications for scaling laws: LST may shift the compute-optimal point for speech LMs.
- Text performance also improves—speech patching does not harm text; rather, it indirectly enhances it through better cross-modal training.

## Highlights & Insights
- **Successful Transfer of BLT Paradigm from Text to Speech**: The core idea of Byte Latent Transformer (aggregating fine-grained tokens into patches for global modeling) is equally effective in the speech domain, and potentially more useful given that speech redundancy is higher than that of bytes.
- **Win-Win for Efficiency and Quality**: Reducing sequence length while improving quality is not a trade-off but a win-win. Reason: Shorter sequences make it easier for the Global Transformer to learn long-range dependencies.
- **Clever Curriculum Design**: Alignment patching requires an inference auxiliary model (impractical), while static patching loses semantic alignment (sub-optimal). Curriculum smoothly transitions from the former to the latter—using alignment for training and static for inference.

## Limitations & Future Work
- The robustness of patch size selection across different languages (with varying syllable structures) and speech rates has not been fully verified.
- Only HuBERT semantic tokens were used; codec-based acoustic tokens (e.g., SoundStorm) were not tested.
- No direct comparison was made with end-to-end speech LLMs like Moshi or Spirit-LM.
- Hyperparameters $\tau_1, \tau_2$ for the curriculum schedule require tuning.
- 7B experiments were conducted with a sub-optimal token budget (70B vs. ~140B optimal); full compute-optimal experiments are costly.

## Related Work & Insights
- **vs BLT (Pagnoni 2024)**: LST transfers the byte$\rightarrow$patch concept of BLT to speech token$\rightarrow$speech patch, serving as a direct inspiration.
- **vs Nguyen 2025 (Interleaved Training)**: Interleaved training is a baseline method; LST adds patching on top of this as an orthogonal improvement.
- **vs Moshi (End-to-end Speech LLM)**: Moshi uses multi-stream modeling, whereas LST uses patch compression—different paths to solving the information density mismatch.

## Rating
- Novelty: ⭐⭐⭐⭐ The latent patch concept is concise and effective; the transfer from BLT to speech is natural but includes non-obvious design considerations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale (420M$\rightarrow$7B) $\times$ two control settings $\times$ downstream tasks $\times$ comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and easy to follow; the logic from motivation to design and experiments is fluid.
- Value: ⭐⭐⭐⭐⭐ ICLR Oral is well-deserved; provides significant guidance for joint speech-text modeling and improves scaling behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](../../ACL2026/audio_speech/immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ICLR 2026\] Towards True Speech-to-Speech Models Without Text Guidance](towards_true_speech-to-speech_models_without_text_guidance.md)
- [\[AAAI 2026\] SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](../../AAAI2026/audio_speech/spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)
- [\[ICLR 2026\] Closing the Gap Between Text and Speech Understanding in LLMs](closing_the_gap_between_text_and_speech_understanding_in_llms.md)

</div>

<!-- RELATED:END -->
