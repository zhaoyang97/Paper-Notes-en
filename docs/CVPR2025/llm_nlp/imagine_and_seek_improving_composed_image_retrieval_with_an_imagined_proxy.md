---
title: >-
  [Paper Note] Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy
description: >-
  [CVPR 2025][LLM (Other)][Composed Image Retrieval] Proposed the IP-CIR method, which translates Composed Image Retrieval (CIR) into a standard image retrieval problem by using large language models to generate an "imagined target text description" as a proxy, achieving zero-shot SOTA on benchmarks such as CIRR and FashionIQ.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Composed Image Retrieval"
  - "Virtual Proxy Image"
  - "CLIP"
  - "Zero-Shot Generalization"
  - "Text-Image Alignment"
date: 2026-05-08
content_hash: 60a610801ff5ea9b
---

# Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy

**Conference**: CVPR 2025  
**arXiv**: [2411.16752](https://arxiv.org/abs/2411.16752)  
**Code**: TBD  
**Area**: Image Retrieval / Multimodal Learning  
**Keywords**: Composed Image Retrieval, Virtual Proxy Image, CLIP, Zero-Shot Generalization, Text-Image Alignment

## TL;DR
Proposed the IP-CIR method, which translates Composed Image Retrieval (CIR) into a standard image retrieval problem by using large language models to generate an "imagined target text description" as a proxy, achieving zero-shot SOTA on benchmarks such as CIRR and FashionIQ.

## Background & Motivation

**Background**: The task of Composed Image Retrieval (CIR) is to retrieve a target image that meets modification requirements given a reference image and a modification text. Existing methods require a large number of annotated triplets (reference image, modification text, target image), which is extremely costly to annotate.

**Limitations of Prior Work**:
   - Supervised CIR methods require expensive triplet data and have limited generalization capabilities.
   - Zero-shot CIR methods (such as Pic2Word, SEARLE) do not require triplets but perform significantly worse than supervised methods.
   - Existing methods struggle to effectively fuse image content with text modification intents.

**Key Insight**: If one can imagine what the modified target image looks like (i.e., generate a "virtual proxy" description), standard text-to-image retrieval can replace complex composed retrieval. LLMs can accomplish this "imagination" process—reasoning the description of the target image based on the reference image description and the modification text.

## Method

### Overall Architecture
Reference image → BLIP2 generates image description → LLM (GPT-4) combines modification text to generate target description → CLIP text encoder encoding → Matching and retrieval with database image features.

### Key Designs

1. **Imagined Proxy Generation**

    - Use BLIP2 to convert the reference image into a textual description.
    - Input the image description and modification text into the LLM to generate the "imagined target image description".
    - Example: Reference image description "red dress" + modification text "change to blue" $\rightarrow$ Imagined description "blue dress".
    - Design Motivation: Leverage the reasoning capability of LLMs to achieve visual imagination, avoiding complex multimodal fusion.

2. **Feature Fusion and Matching**

    - Encode the imagined description using the CLIP text encoder to obtain the proxy feature.
    - Fuse this proxy feature with the CLIP visual feature of the original reference image via weighted fusion.
    - Calculate the similarity with the CLIP visual features of all images in the database.
    - Design Motivation: Retain the visual details of the reference image while injecting the text modification intent.

3. **Training Strategy**

    - Zero-shot setting: No training data of CIR triplets is required.
    - The LLM is only utilized during inference to generate the proxy description.
    - Optional fine-tuning: The alignment module can be further fine-tuned when annotated data is available.

### Loss & Training
No training is required in the zero-shot setting. For the supervised setting, contrastive learning loss is employed to align the proxy features with the target image features.

## Key Experimental Results

### Main Results: Zero-Shot CIR

| Dataset | Metric | IP-CIR | Pic2Word | SEARLE |
|--------|------|--------|----------|--------|
| CIRR | Recall@10 | **70.07** | 58.2 | 62.1 |
| CIRR | Recall@50 | **87.3** | 79.6 | 82.5 |
| FashionIQ (Dress) | Recall@10 | **32.4** | 26.8 | 28.9 |

### Ablation Study

| Configuration | Recall@10 | Explanation |
|------|----------|------|
| Text modification only | 55.3 | Ignores reference image |
| Image features only | 48.7 | No text modification |
| Imagined proxy (text) | 65.2 | LLM-generated description |
| Imagined proxy + Image fusion | **70.07** | Full method |

### Key Findings
- The imagined proxy significantly outperforms direct text modification ($70.07$ vs $55.3$), proving the value of LLM reasoning for CIR.
- Image feature fusion contributes an additional ~5 points to Recall, indicating the importance of retaining visual details.
- Zero-shot performance is close to or even exceeds some supervised methods.

## Highlights & Insights
- **Paradigm Shift**: Transforms the composed retrieval problem into "imagine then retrieve", bypassing the bottleneck of triplet annotations.
- **LLM as a Visual Reasoner**: Leverages the common sense reasoning capability of LLMs to accomplish visual imagination.
- **Plug-and-Play**: The method is independent of specific CLIP models and can adapt to any multimodal foundation model.

## Limitations & Future Work
- Relies on the reasoning quality of LLMs; complex modifications (e.g., spatial relationship transformations) might result in inaccurate imagination.
- The image descriptions from BLIP2 might miss critical visual details.
- LLM inference increases inference latency.
- Performance on fine-grained attribute modifications (e.g., texture changes) remains to be verified.

## Related Work & Insights
- **vs Pic2Word**: Pic2Word projects images into the text space but lacks reasoning; this work employs LLMs for explicit reasoning.
- **vs SEARLE**: SEARLE uses retrieval augmentation, whereas this work uses generative augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of an imagined proxy is intuitive and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets with clear ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation.
- Value: ⭐⭐⭐⭐ A practical solution for zero-shot CIR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ACL 2026\] OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning](../../ACL2026/llm_nlp/toward_robust_in-context_learning_leveraging_out-of-distribution_proxies_for_tar.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](../../ACL2025/llm_nlp/improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](../../ACL2025/llm_nlp/uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)

</div>

<!-- RELATED:END -->
