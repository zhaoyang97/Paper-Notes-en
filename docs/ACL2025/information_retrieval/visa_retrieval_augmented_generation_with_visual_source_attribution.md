---
title: >-
  [Paper Note] VISA: Retrieval Augmented Generation with Visual Source Attribution
description: >-
  [ACL 2025][Information Retrieval & RAG][Visual Source Attribution] VISA proposes a RAG method based on visual source attribution, which leverages large vision-language models (VLMs) to highlight the precise region supporting the generated answer with bounding boxes on retrieved document screenshots, and constructs two datasets, Wiki-VISA and Paper-VISA, to verify its effectiveness.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Visual Source Attribution"
  - "Retrieval-Augmented Generation"
  - "Vision-Language Models"
  - "Document Screenshots"
  - "Bounding Box Localization"
date: 2026-05-08
content_hash: e8d10ee25b2648ab
---

# VISA: Retrieval Augmented Generation with Visual Source Attribution

**Conference**: ACL 2025  
**arXiv**: [2412.14457](https://arxiv.org/abs/2412.14457)  
**Code**: Yes (the paper mentions that code, data, and model checkpoints will be released)  
**Area**: Information Retrieval  
**Keywords**: Visual Source Attribution, Retrieval-Augmented Generation, Vision-Language Models, Document Screenshots, Bounding Box Localization

## TL;DR

VISA proposes a RAG method based on visual source attribution, which leverages large vision-language models (VLMs) to highlight the precise region supporting the generated answer with bounding boxes on retrieved document screenshots, and constructs two datasets, Wiki-VISA and Paper-VISA, to verify its effectiveness.

## Background & Motivation

1. **Background**: RAG systems enhance generation reliability by retrieving external documents. Recent work introduces "generation with citation", enabling models to cite source documents while generating answers. However, existing methods mainly focus on **text-level citation**—linking answers to document identifiers.

2. **Limitations of Prior Work**: Document-level citations impose a heavy cognitive burden on users. Upon receiving a citation, users still need to manually locate the specific paragraph, table, or image supporting the answer in a long, multi-page document. Even paragraph-level citations have issues—they require additional engineering development to match chunks to original document positions, and cannot naturally highlight within formats like PDFs.

3. **Key Challenge**: The attribution granularity of existing RAG is too coarse (document-level) and is limited by the text format, making it impossible to visually show the location of evidence. On the other hand, the recently emerged **document screenshot retrieval paradigm** (e.g., DSE) directly uses screenshots for retrieval, preserving visual information but lacking attribution capability.

4. **Goal**: Is it possible to achieve end-to-end source attribution in the visual RAG pipeline—allowing the model to not only generate the answer but also precisely locate the answer on the document screenshot?

5. **Key Insight**: Leverage the existing image understanding and bounding box prediction capabilities of VLMs to define source attribution as outputting the bounding box coordinates of supporting evidence within document screenshots.

6. **Core Idea**: Transform RAG source attribution from a text citation paradigm to a visual localization paradigm—VLMs directly draw bounding boxes on document screenshots to point to the source of evidence for the answers.

## Method

### Overall Architecture

The input is a user text query + retrieved document screenshots (1 to many). After processing the multimodal inputs, the VLM autoregressively generates three outputs simultaneously: (1) the textual answer; (2) the identifier of the relevant document; and (3) bounding box coordinates (top-left and bottom-right $(x_1,y_1,x_2,y_2)$). Finally, the bounding box is drawn on the document screenshot and presented to the user.

### Key Designs

1. **Visual Source Attribution Task Definition**:

    - **Function**: Formalize the visual attribution task in RAG
    - **Mechanism**: Given a query $q$ and a retrieved candidate document set $D=\{d_1,...,d_n\}$, the system must simultaneously return the answer $a$, the most relevant document identifier $i$, and the bounding box of the evidence in that document $B_{d^*} = [(x_1,y_1),(x_2,y_2)]$. All inputs are screenshot images, and the entire process is unified under next-token prediction modeling
    - **Design Motivation**: Unify attribution and generation into a single autoregressive generation task to avoid multi-stage pipelines

2. **Wiki-VISA and Paper-VISA Dataset Construction**:

    - **Function**: Provide high-quality training and evaluation data
    - **Mechanism**: **Wiki-VISA** is based on the Natural Questions dataset, using Selenium to render screenshots of original Wikipedia pages (980px width $\times$ up to 3920px height), with NQ's short answers as the target answers, and the screenshot positions of the HTML elements corresponding to the long answers as the target bounding boxes (87k training / 3000 test). **Paper-VISA** is based on PubLayNet (medical paper PDF pages), leveraging VLMs to synthesize queries and answers for each pre-annotated layout element (100k training / 2160 test). In addition, **FineWeb-VISA** (60k scraped webpage screenshots) was constructed as supplementary training data
    - **Design Motivation**: Wiki-VISA provides human-annotated quality general knowledge evaluation, while Paper-VISA covers the scientific paper domain. The large layout differences between them test generalization capabilities

3. **Multi-candidate Document Training Setup**:

    - **Function**: Simulate real RAG scenarios with multiple retrieval candidates
    - **Mechanism**: Randomly sample $m-1$ incorrect documents as hard negatives from the top-20 results of the DSE retriever, and mix them with the correct document for input. A 20% random probability is used to replace the correct document to test the model's ability to identify "no-answer" scenarios. The model is first trained on single documents for two epochs, and then multi-candidate document training is initialized using the single-document weights for one epoch
    - **Design Motivation**: Random sampling instead of directly taking the top-$m$ prevents the model from relying on specific retrievers and document positions; adding no-answer scenarios tests the model's ability to abstain

### Loss & Training

Standard next-token prediction + cross-entropy loss. Fine-tune Qwen2-VL-2B and Qwen2-VL-7B using LoRA, with a learning rate of 1e-4, batch size of 64, and 4×H100 GPUs. Random cropping augmentation (cropping outside the bounding box) is applied during training to improve the model's generalization to different input sizes. The image encoder is frozen during multi-candidate training to save VRAM.

## Key Experimental Results

### Main Results

**Single-document Setup**:

| Model | Wiki-VISA bbx Avg | Wiki-VISA ans Avg | Paper-VISA bbx Avg | Paper-VISA ans Avg |
|------|-------------------|-------------------|--------------------|--------------------|
| QWen2-VL-72B (zero-shot) | 1.5% | 60.4% | 1.5% | 43.1% |
| VISA-2B-single | 37.5% | 57.1% | 63.0% | 38.3% |
| VISA-7B-single | 54.2% | 65.2% | 68.2% | 43.8% |

**Multi-document Setup** (VISA-7B, 3 candidates, including unanswerable samples):

| Setup | Wiki-VISA bbx | Wiki-VISA ans | Paper-VISA bbx | Paper-VISA ans |
|------|---------------|---------------|----------------|----------------|
| Multi-candidate, Full | 41.6% | 51.1% | 66.8% | 50.3% |

### Ablation Study

| Training Data | Wiki-VISA bbx | Wiki-VISA ans | Paper-VISA bbx | Paper-VISA ans |
|----------|---------------|---------------|----------------|----------------|
| Wiki only | 54.2% | 65.2% | 27.8% | 36.2% |
| Paper only | 0.2% | 42.6% | 68.2% | 43.8% |
| FineWeb only | 37.6% | 50.2% | 22.0% | 43.3% |
| Wiki+Paper+FineWeb | 58.1% | 64.8% | 67.6% | 44.3% |

### Key Findings

- **Zero-shot is extremely difficult**: Although QWen2-VL-72B can generate reasonable answers (60.4%), its bounding box accuracy is only 1.5%, indicating that existing VLMs are far from possessing zero-shot visual attribution capabilities.
- **Document position heavily impacts performance**: Bounding box accuracy is 75.6% for first-page paragraphs vs. 50.1% for non-first-page paragraphs (Wiki-VISA); attribution in multi-page long documents remains a major challenge.
- **Cross-domain generalization is difficult**: Paper $\rightarrow$ Wiki transfer bounding box accuracy is near zero (0.2%), while Wiki $\rightarrow$ Paper is 27.8%. The multi-page nature of Wiki provides richer training signals.
- **Multi-candidate vs. Single-document**: Moving from single-document to three-candidate documents, bounding box accuracy drops from 54.2% to 37.7% (a decrease of 17 percentage points), indicating that multi-document attribution is significantly harder.
- **FineWeb data augmentation is effective**: Wiki+FineWeb improves bbx on Wiki-VISA from 54.2% to 58.2%, showing that diverse layouts aid generalization.

## Highlights & Insights

- **Pioneering definition of the visual attribution paradigm**: Transforming RAG source attribution from text citations to visual localization is the first instance of unifying answer generation and precise visual localization into a single task within a VLM-RAG framework. This idea can be transferred to any scenario requiring search backtracks to information sources.
- **Ingenious dataset construction strategy**: Wiki-VISA naturally obtains bounding box annotations using HTML elements from NQ's long answers without additional manual annotation; Paper-VISA utilizes VLMs to synthesize QAs, avoiding the high cost of manual annotation for scientific papers.
- **Meaningful evaluation of VLM capabilities**: VISA is not only an application but also a benchmark to test the self-explanation and precise localization capabilities of VLMs.

## Limitations & Future Work

- Cross-domain generalization is highly insufficient (Paper $\rightarrow$ Wiki is almost zero), limiting practical deployment.
- Currently, only single bounding boxes are supported, whereas answer evidence might be scattered across multiple regions in real-world scenarios.
- Only the Qwen2-VL series was tested; other VLMs (such as InternVL, LLaVA) were not evaluated.
- Localization on multi-page long documents remains a major bottleneck (accuracy on non-first-page paragraphs drops significantly).
- Currently, only screenshot-level inputs are supported, and integration with OCR pipelines has not been explored.

## Related Work & Insights

- **vs. Traditional Text RAG Attribution**: Citation generation from Gao et al. only provides document IDs, requiring users to search for themselves; VISA directly points to the specific location on the document screenshot, significantly reducing the cognitive burden.
- **vs. DSE (Ma et al.)**: DSE uses document screenshots for retrieval but lacks attribution capabilities; VISA fills this visual attribution gap on top of DSE, forming an end-to-end visual RAG system.
- **vs. GUI Grounding**: Lin et al. and Cheng et al. perform UI element localization in GUI interfaces; VISA applies similar capabilities to content-dense documents, which is more challenging due to more complex layouts and denser text.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to propose visual source attribution within a VLM-RAG framework, making the problem definition highly pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Detailed analysis conducted across two datasets and multiple settings (single-document/multi-document/cross-domain).
- **Writing Quality**: ⭐⭐⭐⭐ Clear description of methods and detailed process for dataset construction.
- **Value**: ⭐⭐⭐⭐ Proposes a new paradigm for RAG verifiability, but cross-domain generalization issues limit immediate practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
