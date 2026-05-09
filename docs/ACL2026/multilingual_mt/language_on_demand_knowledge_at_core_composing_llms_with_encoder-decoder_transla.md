---
title: >-
  [Paper Note] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality
description: >-
  [ACL 2026][Multilingual LLM] This paper proposes XBridge, an architecture that composes pretrained multilingual encoder-decoder translation models (e.g., NLLB) with English-centric LLMs — the encoder handles multilingual understanding, the LLM handles knowledge reasoning, and the decoder handles multilingual generation. Lightweight mapping layers and optimal transport alignment are employed to bridge cross-model semantic gaps, yielding significant improvements over baselines on low-resource and unseen languages.
tags:
  - ACL 2026
  - Multilingual LLM
  - Model Composition
  - Encoder-Decoder Translation Model
  - Optimal Transport Alignment
  - Low-Resource Languages
date: 2026-05-08
content_hash: 67571b3ee73a9aa6
---

# Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality

**Conference**: ACL 2026
**arXiv**: [2603.17512](https://arxiv.org/abs/2603.17512)
**Code**: [GitHub](https://github.com/ictnlp/XBridge)
**Area**: Multilingual Translation
**Keywords**: Multilingual LLM, Model Composition, Encoder-Decoder Translation Model, Optimal Transport Alignment, Low-Resource Languages

## TL;DR

This paper proposes XBridge, an architecture that composes pretrained multilingual encoder-decoder translation models (e.g., NLLB) with English-centric LLMs — the encoder handles multilingual understanding, the LLM handles knowledge reasoning, and the decoder handles multilingual generation. Lightweight mapping layers and optimal transport alignment are employed to bridge cross-model semantic gaps, yielding significant improvements over baselines on low-resource and unseen languages.

## Background & Motivation

**Background**: LLMs exhibit strong general intelligence and reasoning capabilities, yet their multilingual performance is severely imbalanced — excelling in English and a handful of high-resource languages while frequently failing in low-resource and unseen ones. Meanwhile, pretrained encoder-decoder translation models (e.g., NLLB) already provide balanced translation coverage across hundreds of languages.

**Limitations of Prior Work**: (1) Data-level approaches (translating instruction data for multilingual fine-tuning) may introduce translation noise and disrupt existing language capabilities, making it difficult to balance performance across high- and low-resource languages. (2) Existing encoder-augmentation methods (e.g., MindMerger, LayAlign) inject multilingual encoder representations only at the input side to improve understanding, but generation still relies on the LLM's original language distribution (typically English). (3) The natural extension of incorporating a multilingual decoder introduces representation space mismatch, as inserting a frozen LLM between the encoder and decoder causes its outputs to no longer conform to the decoder's cross-attention expectations.

**Key Challenge**: The core limitation of LLMs is not a lack of knowledge, but an inability to effectively interface knowledge in their unified semantic space with diverse linguistic representation spaces. Encoder-decoder translation models offer complementary multilingual understanding and generation capabilities, yet the two models' representation spaces are heterogeneous and misaligned.

**Goal**: To construct an Encoder-LLM-Decoder compositional architecture that offloads multilingual understanding and generation to external translation models while keeping the LLM frozen as an English-centric knowledge core.

**Key Insight**: The modular nature of translation model encoders and decoders is exploited — the encoder maps multilingual inputs into a shared semantic space, and the decoder projects shared representations into the target language — naturally corresponding to the input-processing-output pipeline of LLMs. The key challenge lies in cross-model representation alignment.

**Core Idea**: A Semantic Bridge is constructed via lightweight mapping layers that transform representations from the multilingual encoder space into the LLM input space, process them through the LLM for knowledge reasoning, and map them into the decoder generation space, with an optimal transport objective enforcing token-level fine-grained semantic alignment.

## Method

### Overall Architecture

XBridge adopts a three-stage Encoder-LLM-Decoder architecture: (1) A multilingual encoder (e.g., the NLLB encoder) receives input in any language and produces contextual representations $H_x$; (2) An encoder-side mapping layer projects $H_x$ into the LLM representation space, which is then fed together with English instructions into the frozen LLM for knowledge processing; (3) While the LLM generates an English response, its penultimate-layer hidden states are projected via a decoder-side mapping layer into the decoder representation space, serving as cross-attention input to the multilingual decoder.

### Key Designs

1. **Cross-Model Mapping**:

    - **Function**: Bridges the representation space gap between the encoder→LLM and LLM→decoder interfaces.
    - **Mechanism**: $\text{Mapping}_{enc}$ linearly projects encoder representations $H_x \in \mathbb{R}^{n \times d_e}$ into the LLM dimension $\tilde{H}_x \in \mathbb{R}^{n \times d_l}$; $\text{Mapping}_{dec}$ projects the LLM's penultimate-layer hidden states $H_{z'} \in \mathbb{R}^{m \times d_l}$ into the decoder dimension $\tilde{H}_{z'} \in \mathbb{R}^{m \times d_d}$. The penultimate layer is chosen over the final layer because the final layer is over-aligned to the output vocabulary space, whereas the penultimate layer retains richer semantic information.
    - **Design Motivation**: Directly coupling the representation spaces of heterogeneous models is infeasible; lightweight mapping layers provide a bridging solution with minimal parameter overhead.

2. **Optimal Transport Alignment**:

    - **Function**: Enforces token-level fine-grained semantic consistency at the LLM-decoder interface.
    - **Mechanism**: The English sequence $z$ generated by the LLM is re-encoded as encoder representations $H_z$, and the optimal transport distance between $H_z$ and the decoder-side mapped representations $\tilde{H}_{z'}$ is computed. Since the two models use different tokenizers (and may produce sequences of different lengths), OT provides flexible many-to-many soft matching, with cosine distance as the transport cost.
    - **Design Motivation**: Simple projection cannot resolve sequence length mismatches caused by heterogeneous tokenizers. OT provides length-robust token-level alignment supervision, ensuring that the representations seen by the decoder are semantically consistent with the encoder-decoder shared space.

3. **Three-Stage Progressive Training**:

    - **Function**: Stably aligns heterogeneous representation spaces and adapts the model to downstream tasks.
    - **Mechanism**: Stage 1 trains both mapping layers and decoder cross-attention on trilingual translation data (source language–English–target language) to establish coarse-grained cross-model alignment. Stage 2 freezes the decoder side and fine-tunes the encoder-side mapping layer on task instruction data, teaching the LLM to leverage multilingual representations for task execution. Stage 3 freezes the encoder side and fine-tunes the decoder-side mapping layer with OT loss and decoder generation loss to improve multilingual generation quality.
    - **Design Motivation**: Decoupling the optimization objectives of the LLM and the decoder — first stabilizing the LLM's conditional distribution, then optimizing decoder performance — avoids conflicts between the two optimization targets.

### Loss & Training

The total loss comprises three terms: LLM English generation cross-entropy $\mathcal{L}_{CE\_LLM}$, decoder multilingual generation cross-entropy $\mathcal{L}_{CE\_Dec}$, and optimal transport alignment loss $\mathcal{L}_{OT}$, with different subsets used at different stages. The LLM remains frozen throughout; only the mapping layers and decoder cross-attention parameters are trained.

## Key Experimental Results

### Main Results

| System (LLM=LLaMA3-8B) | Low-res X→En | Low-res En→X | High-res X→En | High-res En→X |
|------------------------|-------------|-------------|--------------|--------------|
| LLaMA3-8B (base) | 29.83 | 13.18 | 45.28 | 36.24 |
| MindMerger | 33.86 | — | 42.52 | — |
| LayAlign | 32.95 | — | 41.29 | — |
| **XBridge** | **37.09** | **28.42** | **45.75** | **35.45** |
| NLLB-200-1.3B | 37.78 | 32.83 | 46.23 | 39.91 |

### Ablation Study

| Configuration | Low-res BLEU | High-res BLEU | Note |
|---------------|-------------|--------------|------|
| Full XBridge | 37.09 / 28.42 | 45.75 / 35.45 | Full model |
| w/o OT alignment | −∼2–3 pts | −∼1–2 pts | Token-level alignment is important |
| w/o three-stage training | Unstable | Unstable | Progressive training is necessary |
| Using final-layer hidden states | −∼1–2 pts | −∼1 pt | Penultimate layer is superior |

### Key Findings

- XBridge yields the most significant gains on low-resource languages (En→X improves from 13.18 to 28.42 over base LLaMA3-8B), demonstrating the effectiveness of model composition.
- Consistent improvements are observed across four different LLMs (MetaMath-7B, LLaMA3-8B, Aya-23-8B, Qwen2.5-7B).
- Low-resource generation performance approaches that of the dedicated NLLB translation model (28.42 vs. 32.83), substantially closing the gap.
- Existing methods (MindMerger, LayAlign) support only the X→En direction and cannot perform multilingual generation.

## Highlights & Insights

- The design philosophy of "language on demand, knowledge at core" is elegant and concise — the LLM only needs to master English reasoning, while multilingual capability is entirely delegated to the translation model, allowing each component to excel at what it does best.
- Optimal transport alignment elegantly resolves the sequence length mismatch caused by heterogeneous tokenizers, offering greater precision than simple linear projection.
- The decoupling philosophy of three-stage training is transferable to other model composition scenarios — first aligning representation spaces, then separately adapting the input and output sides.

## Limitations & Future Work

- Keeping the LLM frozen throughout prevents the architecture from leveraging the LLM's implicit multilingual knowledge.
- Maintaining an additional translation model increases inference-time computational overhead and deployment complexity.
- Evaluation is currently limited to translation and simple tasks; performance on complex reasoning combined with multilingual generation remains unknown.
- The computational complexity of OT alignment scales with sequence length, which may become a bottleneck in long-document settings.

## Related Work & Insights

- **vs. MindMerger/LayAlign**: These methods inject multilingual encoder representations only at the input side, with generation still relying on the LLM's English distribution; XBridge additionally incorporates a decoder to enable genuine multilingual generation.
- **vs. Data-level Augmentation**: Multilingual fine-tuning via translated instruction data may degrade high-resource language performance; XBridge leaves LLM parameters unchanged, avoiding such regression.
- **vs. NLLB**: NLLB provides balanced multilingual coverage but lacks general reasoning capability; XBridge combines the strengths of both.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The Encoder-LLM-Decoder composition paradigm is novel, though OT alignment and mapping layers are applications of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation spans four LLMs, multiple tasks, and multiple languages, though complex reasoning tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly derived, method description is precise, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides an elegant approach to multilingual LLM adaptation that requires no modification of LLM parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)

</div>

<!-- RELATED:END -->
