---
title: >-
  [Paper Note] Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU
description: >-
  [CVPR 2026][Multimodal VLM][LVLM] Layer-wise linear probing analysis reveals a significant gap between internal representations and generated responses in LVLMs for visual document understanding (VDU). Intermediate layer…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "LVLM"
  - "visual document understanding"
  - "linear probing"
  - "internal representations"
  - "intermediate layers"
date: 2026-05-08
content_hash: 7bbfd30a621fb3fa
---

# Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU

**Conference**: CVPR 2026
**arXiv**: [2604.04411](https://arxiv.org/abs/2604.04411)
**Code**: None
**Area**: Multimodal Large Language Models / Document Understanding
**Keywords**: LVLM, visual document understanding, linear probing, internal representations, intermediate layers

## TL;DR

Layer-wise linear probing analysis reveals a significant gap between internal representations and generated responses in LVLMs for visual document understanding (VDU). Intermediate layers encode more linearly accessible task-relevant information than final layers, and fine-tuning intermediate layers simultaneously improves accuracy and narrows the gap.

## Background & Motivation

Large vision-language models (LVLMs) have made progress on VDU tasks, yet performance evaluation has primarily relied on the correctness of generated responses. However, response accuracy may not fully reflect whether the model has internally captured the information necessary to answer a question.

Prior work has hinted that internal representations may carry richer information than generated responses, but a layer-wise investigation of this phenomenon in LVLMs—and whether the most informative representations reside in the final or earlier layers—remains underexplored. VDU, requiring the integration of multimodal and structured reasoning, provides an ideal testbed for analyzing how multimodal information is represented within LVLMs.

## Method

### Overall Architecture

Linear classifiers are constructed at every layer of three LVLMs (Qwen2.5-VL 32B, Gemma3 27B, LLaVA-NeXT 13B) to assess the linear encodability of information at each layer, and the results are compared against the models' text response accuracy.

### Key Designs

1. **Layer-wise Linear Probing**: Four types of classifiers (image-token, text-token, all-token, last-token) are constructed at each LLM layer. A single linear transformation $\mathbf{z} = W\mathbf{h} + \mathbf{b}$ maps the hidden state to a binary classification output, evaluating the linear separability of information at each layer. Four task types are covered: visual attribute recognition (easy-VQA), text recognition (MJSynth), layout understanding (PubLayNet), and chart understanding (FigureQA). Datasets are specifically constructed using samples for which the model originally generates incorrect answers (78% of samples are filtered out), ensuring that only non-trivial cases are analyzed. Each task comprises 100K training samples and 10K test samples.

2. **Gap Quantification**: Linear probing accuracy (measuring internal information) and text response accuracy (measuring output behavior) are systematically compared to reveal the gap between the two, demonstrating that information can be linearly encoded internally without being reflected in the response.

3. **Intermediate-Layer Fine-Tuning**: Based on the linear probing findings, intermediate layers are selectively fine-tuned rather than all layers. Experiments show that full-layer fine-tuning is insufficient to fully close the gap, whereas intermediate-layer fine-tuning more effectively improves both linear probing accuracy and response accuracy.

### Loss & Training

Linear probes are trained with cross-entropy loss while LVLM parameters are frozen. Standard VQA training loss is used during fine-tuning. Datasets are specifically constructed using samples for which the model originally generates incorrect answers (78% filtered), ensuring analysis of non-trivial cases.

## Key Experimental Results

### Main Results

| Model | Task | Best Probing Layer | Best Accuracy | Response Accuracy | Gap |
|---|---|---|---|---|---|
| Qwen2.5-VL 32B | Chart Understanding | Intermediate | ~85% | ~50% | ~35% |
| Gemma3 27B | Layout Understanding | Intermediate | ~80% | ~50% | ~30% |

### Key Findings

- A significant gap exists between internal representations and generated responses: information is encoded internally but not utilized.
- Information required for VDU tasks is more linearly accessible in intermediate layers than in final layers.
- Full-layer fine-tuning is insufficient to close the gap; intermediate-layer fine-tuning is more effective.
- Image tokens contain rich task-relevant information in early-to-middle layers.

## Highlights & Insights

- The first systematic layer-wise linear probing analysis in the VDU domain.
- The finding that "the model knows but does not say" carries profound implications, suggesting potential directions for improving generation strategies.
- The finding that intermediate-layer fine-tuning outperforms full-layer fine-tuning has practical value.
- Provides a new perspective for understanding the internal mechanisms of LVLMs.
- Only samples on which the model generates incorrect answers are retained (78% filtered), ensuring the study focuses on non-trivial scenarios.
- Image tokens carry rich task-relevant information in early-to-middle layers but are suppressed by language priors in deeper layers.
- Experiments cover three mainstream LVLMs: Qwen2.5-VL 32B, Gemma3 27B, and LLaVA-NeXT 13B.

## Limitations & Future Work

- Linear probing captures only linearly encodable information and may underestimate non-linearly encoded information.
- The fine-tuning strategy requires prior probing analysis to identify target layers, increasing deployment overhead.
- The root causes of the gap (decoding bias? attention allocation?) warrant further investigation.
- Retaining only samples with incorrect generated answers during dataset construction may introduce selection bias.
- Scalability to more complex VDU tasks (e.g., multi-page documents, table reasoning) remains to be verified.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First layer-wise analysis in the VDU domain, revealing the gap between internal representations and generated responses.
- **Technical Depth**: ⭐⭐⭐⭐ — Systematic and comprehensive analysis covering 4 tasks and multiple token types.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple models and tasks.
- **Value**: ⭐⭐⭐ — The intermediate-layer fine-tuning strategy offers practical reference, though it requires prior probing analysis.

The analysis employs large-scale binary classification datasets of 100K training and 10K test samples, ensuring statistical significance of the results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts](scipostgen_bridging_the_gap_between_scientific_papers_and_poster_layouts.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[ICCV 2025\] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs](../../ICCV2025/multimodal_vlm/sparsemm_head_sparsity_emerges_from_visual_concept_responses_in_mllms.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)

</div>

<!-- RELATED:END -->
