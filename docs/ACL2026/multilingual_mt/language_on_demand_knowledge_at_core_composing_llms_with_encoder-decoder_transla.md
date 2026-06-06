---
title: >-
  [Paper Note] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual LLM] This paper proposes XBridge, an architecture that composes pre-trained multilingual encoder-decoder translation models (e.g.…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual LLM"
  - "Model Composition"
  - "Encoder-Decoder Translation Models"
  - "Optimal Transport Alignment"
  - "Low-resource Languages"
date: 2026-05-08
content_hash: 4762afbca8356211
---

# Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality

**Conference**: ACL 2026  
**arXiv**: [2603.17512](https://arxiv.org/abs/2603.17512)  
**Code**: [GitHub](https://github.com/ictnlp/XBridge)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual LLM, Model Composition, Encoder-Decoder Translation Models, Optimal Transport Alignment, Low-resource Languages

## TL;DR

This paper proposes XBridge, an architecture that composes pre-trained multilingual encoder-decoder translation models (e.g., NLLB) with English-centric LLMs. The encoder handles multilingual understanding, the LLM performs knowledge reasoning, and the decoder manages multilingual generation. Cross-model semantic bridging is achieved through lightweight mapping layers and Optimal Transport alignment, significantly outperforming baselines on low-resource and unseen languages.

## Background & Motivation

**Background**: LLMs have demonstrated powerful general intelligence and reasoning capabilities, but their multilingual performance is severely unbalanced—excelling in English and a few high-resource languages while often failing in low-resource and unseen languages. Meanwhile, pre-trained encoder-decoder translation models (e.g., NLLB) have developed balanced translation capabilities across hundreds of languages.

**Limitations of Prior Work**: (1) Data-level methods (multilingual fine-tuning via translated instruction data) may introduce translation noise and interfere with existing linguistic capabilities, making it difficult to balance high- and low-resource performance. (2) Existing encoder-enhancement methods (e.g., MindMerger, LayAlign) only inject multilingual encoder representations at the input side to improve understanding, but generation still relies on the LLM's original language distribution (typically English). (3) A natural extension is to add a multilingual decoder, but inserting a frozen LLM between the encoder and decoder introduces representation space mismatch—the LLM output no longer matches the cross-attention expectations of the decoder.

**Key Challenge**: The core limitation of LLMs is not a lack of knowledge, but rather the inability to effectively interface knowledge within their unified semantic space with diverse language representation spaces. Encoder-decoder translation models provide complementary multilingual understanding and generation capabilities, but their representation spaces are heterogeneous and unaligned with the LLM.

**Goal**: To build an Encoder-LLM-Decoder composition architecture that offloads multilingual understanding and generation tasks to an external translation model while keeping the LLM frozen as an English-centric knowledge core.

**Key Insight**: Leverage the modular nature of translation model encoders and decoders—the encoder maps multilingual input to a shared semantic space, and the decoder projects shared representations into the target language. This naturally corresponds to the input-processing-output pipeline of LLMs. The critical challenge lies in cross-model representation alignment.

**Core Idea**: Construct a "Semantic Bridge" using lightweight mapping layers to convert representations from the multilingual encoder space to the LLM input space, and from the LLM processing space to the decoder generation space. Fine-grained token-level semantic alignment is enforced using an Optimal Transport objective.

## Method

### Overall Architecture

XBridge adopts a three-stage Encoder-LLM-Decoder architecture: (1) A multilingual encoder (e.g., NLLB encoder) receives input in any language and generates contextual representations $H_x$; (2) An encoder-side mapping layer projects $H_x$ into the LLM representation space, which is then fed into the frozen LLM along with English instructions for knowledge processing; (3) While the LLM generates an English response, its penultimate hidden states are projected through a decoder-side mapping layer into the decoder representation space, serving as the cross-attention input for the multilingual decoder.

### Key Designs

1.  **Cross-Model Mapping**:
    *   **Function**: Bridges representation space differences between Encoder $\to$ LLM and LLM $\to$ Decoder.
    *   **Mechanism**: $\text{Mapping}_{enc}$ linearly projects encoder representations $H_x \in \mathbb{R}^{n \times d_e}$ to LLM dimensions $\tilde{H}_x \in \mathbb{R}^{n \times d_l}$. $\text{Mapping}_{dec}$ projects LLM penultimate hidden states $H_{z'} \in \mathbb{R}^{m \times d_l}$ to decoder dimensions $\tilde{H}_{z'} \in \mathbb{R}^{m \times d_d}$. The penultimate layer is chosen over the final layer because the latter is overly aligned with the output vocabulary space, whereas the penultimate layer retains richer semantic information.
    *   **Design Motivation**: Directly interfacing the representation spaces of heterogeneous models is infeasible; lightweight mapping layers provide a bridging solution with minimal parameter overhead.

2.  **Optimal Transport Alignment**:
    *   **Function**: Achieves fine-grained token-level semantic consistency at the LLM-decoder interface.
    *   **Mechanism**: The English sequence $z$ generated by the LLM is re-encoded into encoder representations $H_z$. The Optimal Transport (OT) distance is then calculated between $H_z$ and the mapped decoder-side representations $\tilde{H}_{z'}$. Since the two models use different tokenizers (leading to different sequence lengths), OT provides flexible many-to-many soft matching, with cosine distance used as the transport cost.
    *   **Design Motivation**: Simple projections cannot resolve sequence length mismatches caused by heterogeneous tokenization. OT provides token-level alignment supervision robust to length variations, ensuring the representations seen by the decoder are semantically consistent with the encoder-decoder shared space.

3.  **Progressive Training Strategy**:
    *   **Function**: Stably aligns heterogeneous representation spaces and adapts to downstream tasks.
    *   **Mechanism**: **Stage 1** uses trilingual translation data (Source-English-Target) to train both mapping layers and decoder cross-attention, establishing coarse-grained cross-model alignment. **Stage 2** freezes the decoder side and fine-tunes the encoder-side mapping layer using task instructions to teach the LLM how to utilize multilingual representations. **Stage 3** freezes the encoder side and fine-tunes the decoder-side mapping layer using OT loss and decoder generation loss to improve multilingual generation quality.
    *   **Design Motivation**: Decoupling LLM and decoder optimization objectives stabilizes the LLM's conditional distribution before optimizing decoder performance, preventing conflicting optimization goals.

### Loss & Training

The total loss consists of three components: LLM English generation cross-entropy $\mathcal{L}_{CE\_LLM}$, decoder multilingual generation cross-entropy $\mathcal{L}_{CE\_Dec}$, and Optimal Transport alignment loss $\mathcal{L}_{OT}$. Different subsets of these are used across the stages. The LLM remains frozen throughout; only the mapping layers and decoder cross-attention parameters are trained.

## Key Experimental Results

### Main Results

| System (LLM=LLaMA3-8B) | Low-res X→En | Low-res En→X | High-res X→En | High-res En→X |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA3-8B Vanilla | 29.83 | 13.18 | 45.28 | 36.24 |
| MindMerger | 33.86 | - | 42.52 | - |
| LayAlign | 32.95 | - | 41.29 | - |
| **Ours (XBridge)** | **37.09** | **28.42** | **45.75** | **35.45** |
| NLLB-200-1.3B | 37.78 | 32.83 | 46.23 | 39.91 |

### Ablation Study

| Configuration | Low-res BLEU | High-res BLEU | Description |
| :--- | :--- | :--- | :--- |
| Full XBridge | 37.09 / 28.42 | 45.75 / 35.45 | Complete model |
| w/o OT Alignment | Decreased ~2-3 pts | Decreased ~1-2 pts | Token-level alignment is critical |
| w/o 3-Stage Training | Unstable | Unstable | Progressive training is necessary |
| Using Final Layer | Decreased ~1-2 pts | Decreased ~1 pt | Penultimate layer is superior |

### Key Findings

*   XBridge shows the most significant improvements in low-resource languages (e.g., En→X increased from 13.18 to 28.42 compared to vanilla LLaMA3-8B), proving the effectiveness of model composition.
*   The method is consistently effective across four different LLMs (MetaMath-7B, LLaMA3-8B, Aya-23-8B, Qwen2.5-7B).
*   Low-resource generation performance approaches that of the specialized NLLB translation model (28.42 vs 32.83), drastically narrowing the gap.
*   Existing methods (MindMerger, LayAlign) only support the X→En direction and cannot perform multilingual generation.

## Highlights & Insights

*   The "Language on Demand, Knowledge at Core" philosophy is elegant—the LLM only needs to excel at English reasoning, while multilingual capabilities are outsourced to the translation model, leveraging the strengths of both.
*   Optimal Transport alignment cleverly addresses the sequence length mismatch caused by heterogeneous tokenizers, providing finer granularity than simple linear projections.
*   The decoupling idea in the three-stage training is transferable to other model composition scenarios—aligning representation spaces first before independently adapting the input and output ends.

## Limitations & Future Work

*   Keeping the LLM frozen throughout means its internal, latent multilingual knowledge cannot be fully exploited.
*   Maintaining an additional translation model increases computational overhead during inference and deployment complexity.
*   Evaluation is currently focused on translation and simple tasks; performance in joint complex reasoning + multilingual generation scenarios remains unknown.
*   The computational complexity of OT alignment grows with sequence length, which may become a bottleneck for long-text scenarios.

## Related Work & Insights

*   **vs MindMerger/LayAlign**: These only add a multilingual encoder at the input; generation still depends on the LLM's English distribution. XBridge adds a decoder for true multilingual generation.
*   **vs Data-level Enhancement**: Multilingual fine-tuning with translated data can hurt high-resource language performance. XBridge avoids this degradation by not modifying LLM parameters.
*   **vs NLLB**: NLLB has balanced multilingual capabilities but lacks general reasoning; XBridge combines the advantages of both.

## Rating

*   Novelty: ⭐⭐⭐⭐ The Encoder-LLM-Decoder composition approach is novel, though OT alignment and mapping layers are applications of existing techniques.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across four LLMs, multiple tasks, and many languages, though complex reasoning tasks are missing.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, precise methodological description, and intuitive charts.
*   Value: ⭐⭐⭐⭐ Provides an elegant solution for LLM multilinguality without modifying LLM parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimizing Language Models for Crosslingual Knowledge Consistency](../../ICML2026/multilingual_mt/optimizing_language_models_for_crosslingual_knowledge_consistency.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)

</div>

<!-- RELATED:END -->
