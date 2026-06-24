---
title: >-
  [Paper Note] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][Multilingual preference alignment] This paper proposes utilizing implicit reward signals from a well-aligned English DPO model to annotate preference relationships through cross-lingual instruction-response pairs. Combined with iterative DPO training, this approach achieves efficient multilingual preference alignment, resulting in an average win rate improvement of 12.72% on X-AlpacaEval.
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "Multilingual preference alignment"
  - "implicit reward model"
  - "cross-lingual transfer"
  - "DPO"
  - "iterative training"
date: 2026-05-08
content_hash: 7f90160ab23b84d5
---

# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2503.04647](https://arxiv.org/abs/2503.04647)  
**Code**: [GitHub](https://github.com/ZNLP/Implicit-Cross-Lingual-Rewarding)  
**Area**: Alignment RLHF / Multilingual  
**Keywords**: Multilingual preference alignment, implicit reward model, cross-lingual transfer, DPO, iterative training

## TL;DR

This paper proposes utilizing implicit reward signals from a well-aligned English DPO model to annotate preference relationships through cross-lingual instruction-response pairs. Combined with iterative DPO training, this approach achieves efficient multilingual preference alignment, resulting in an average win rate improvement of 12.72% on X-AlpacaEval.

## Background & Motivation

**Background**: DPO (Direct Preference Optimization) has become the mainstream method for aligning LLMs with human preferences, achieving significant progress in English LLM alignment. However, multilingual preference alignment progress is constrained by data scarcity.

**Limitations of Prior Work**: Obtaining multilingual preference data is extremely expensive, requiring the collection of human preference annotations for each language individually, which is practically unfeasible for low-resource languages. Existing solutions attempt to translate English preference data into other languages, but the translation process introduces noise and bias, leading to the distortion of preference signals.

**Key Challenge**: On one hand, English models have already learned high-quality preference knowledge through DPO; on the other hand, a reliable bridge is lacking to transfer this preference knowledge to other languages—direct translation distorts reward signals, while re-collecting data for each language is excessively costly.

**Goal**: To design a cross-lingual preference transfer method that does not rely on translation, directly utilizing the preference knowledge learned by English models to guide multilingual alignment.

**Key Insight**: The authors observe that the logit difference between a DPO-aligned model and its reference model inherently encodes an implicit reward function. This implicit reward model can directly score cross-lingual responses—evaluating multilingual responses using English instructions, thereby avoiding translation-induced distortion.

**Core Idea**: Extract implicit reward signals from an English DPO model to construct cross-lingual instruction-response pairs. The implicit reward model evaluates preference relationships of multilingual responses under English instructions, which are then used as annotated data for iterative DPO training.

## Method

### Overall Architecture

The method is divided into three stages that form an iterative loop: (1) generating multilingual responses from the current multilingual model; (2) constructing cross-lingual instruction-response pairs and annotating preferences using the implicit reward model; (3) performing DPO fine-tuning on the annotated data. The framework starts with Llama-3-Base-8B-SFT-DPO, which has already completed DPO training on English preference data.

### Key Designs

1. **Implicit Reward Model**:

    - **Function**: Extracts preference scoring signals from the aligned English DPO model and its reference model.
    - **Mechanism**: Based on the theoretical derivation of DPO, the log probability difference $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ between the DPO-trained policy model $\pi_\theta$ and the reference model $\pi_{\text{ref}}$ serves as the implicit reward. This reward does not require training an additional reward model and is computed directly from the logits of the two models.
    - **Design Motivation**: Avoids the overhead of training a separate reward model while preserving the complete reward signals learned from English preference data.

2. **Cross-Lingual Preference Annotation**:

    - **Function**: Transfers English preference knowledge to target language preference pairs.
    - **Mechanism**: For a multilingual prompt $x_l$ (where $l$ is the target language), the corresponding English instruction $x_{\text{en}}$ is identified. The current model generates multiple target-language response pairs $(y_l^1, y_l^2)$, and then the implicit reward model evaluates the preference relationship of these responses under the English instruction—i.e., if $r(x_{\text{en}}, y_l^1) > r(x_{\text{en}}, y_l^2)$, then $y_l^1$ is selected as the preferred response.
    - **Design Motivation**: Assessing multilingual responses with English instructions ensures that the anchor for preference scoring remains in English (the language the model is most proficient in), avoiding evaluation noise that may arise from target-language instructions.

3. **Iterative Preference Transfer Training**:

    - **Function**: Progressively improves the quality of multilingual alignment through multiple iterations.
    - **Mechanism**: In each round, the current best model generates new multilingual responses, preference pairs are re-annotated, and combined DPO + NLL training is conducted. The NLL loss $\mathcal{L}_{\text{NLL}} = -\log \pi_\theta(y_w | x)$ is applied to the preferred response, preventing the model from deviating from the correct output distribution during DPO training.
    - **Design Motivation**: Since single-round training is limited by the response quality of the initial model, iterative training progressively enhances both response quality and preference annotation accuracy, establishing a positive feedback loop.

### Loss & Training

The total loss function is a weighted combination of the DPO loss and the NLL loss: $\mathcal{L} = \mathcal{L}_{\text{DPO}} + \alpha \mathcal{L}_{\text{NLL}}$. The DPO loss is the standard preference optimization loss, while the NLL loss calculates the negative log-likelihood on the preferred response. The training process undergoes two iterations (M0 → M1), with the reference model of each round updated to the model obtained from the previous round of training.

## Key Experimental Results

### Main Results

Evaluated on X-AlpacaEval, covering high-resource (es, ru, de, fr) and low-resource (bn, sw, th) languages:

| Model | Avg Win Rate | Avg LC Win Rate | Iterations |
|------|-------------|----------------|---------|
| Llama-3-Base-8B-SFT | 5.36% | 5.78% | - |
| Llama-3-Base-8B-SFT-DPO (English) | 11.14% | 11.96% | - |
| ICR-M0 (DPO) | 18.62% | 15.40% | 1 |
| ICR-M1 (DPO) | 23.86% | 17.93% | 2 |
| ICR-M0 (KTO) | 17.29% | 14.85% | 1 |
| ICR-M1 (KTO) | 21.44% | 16.72% | 2 |

After two iterations, the average Win Rate increased by 12.72%, and the LC Win Rate increased by 5.97%.

### Ablation Study

| Configuration | Avg Win Rate | Description |
|------|-------------|------|
| ICR-M1 Full Model | 23.86% | Two-round DPO iterations |
| M0 only (one round) | 18.62% | No iteration, 5.24% lower |
| Without NLL loss | ~20.5% | Drops by ~3% |
| 1000 samples per language | ~19.8% | Performance drops with reduced data |
| 5000 samples per language | ~22.1% | Close to full performance |
| Using t-1 reference model | ~21.5% | Reference model policy affects performance |

### Key Findings

- Iterative training significantly boosts performance—M1 shows substantial improvements over M0 across all languages, demonstrating the effectiveness of the iterative generation-annotation-training loop.
- The inclusion of NLL loss is crucial for preventing model degradation during DPO training.
- The performance gain in low-resource languages (bn, sw, th) even exceeds that in high-resource languages, indicating that implicit reward transfer is highly effective for low-resource languages.
- Both DPO and KTO preference optimization methods benefit from implicit cross-lingual rewarding, with DPO showing a slight advantage.

## Highlights & Insights

- **Cross-Lingual Transferability of Implicit Rewards** is the most core insight of this paper: preference knowledge encoded in the DPO model can be used across languages without requiring translated data or multilingual reward models. This suggests that preference evaluation is, to some extent, language-agnostic.
- The **training-free reward model** design is highly elegant—leveraging the byproduct of DPO training (the logit difference between the policy and reference models) as the reward signal, obtaining reward scoring capabilities with zero extra cost.
- The iterative training paradigm can be transferred to other "strong-to-weak knowledge transfer" scenarios, such as transferring preference knowledge from large to small models, or from general to domain-specific models.

## Limitations & Future Work

- The method relies on pairing English instructions with multilingual responses; if the linguistic characteristics of the target language differ drastically from English (e.g., honorific systems in Japanese), English preference scoring might not accurately reflect target language preferences.
- Evaluated only on Llama-3-8B, meaning the generalizability to larger-scale models or non-Llama architectures remains unknown.
- Evaluation primarily relies on LLM-as-a-judge (GPT-4 evaluation), lacking verification via human evaluation.
- Future work could explore extending the method to more languages (e.g., Chinese, Arabic) and preference alignment on culturally sensitive topics.

## Related Work & Insights

- **vs SimPO/ORPO**: These methods also do not require an explicit reward model but still necessitate preference data in the target language. This paper further eliminates the need for multilingual preference data.
- **vs Translation-based Alignment**: Directly translating English preference data into target languages introduces translation bias. This paper avoids this issue by evaluating responses in the English instruction space.
- **vs Multilingual RLHF**: Traditional methods require training reward models or collecting preference data for each language. The proposed method significantly reduces this cost.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of cross-lingual transfer of implicit rewards is novel, though the iterative DPO framework is relatively common.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple languages and ablation dimensions, but lacks human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear reasoning and complete methodology description.
- **Value**: ⭐⭐⭐⭐ Provides a low-cost, practical solution for multilingual alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)
- [\[ACL 2025\] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](modular_sentence_encoders.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)

</div>

<!-- RELATED:END -->
