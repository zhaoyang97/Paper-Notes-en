---
title: >-
  [Paper Note] DocoPilot: Improving Multimodal Models for Document-Level Understanding
description: >-
  [CVPR 2025][Multimodal VLM][Document Understanding] This paper constructs Doc-750K—a high-quality, document-level multimodal dataset containing 758K question-answer pairs and 3.1M images. Based on this, the authors train Docopilot, a native document understanding model. It outperforms InternVL2-8B by 19.9 percentage points on MM-NIAH, processing multi-page documents efficiently without relying on RAG.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Document Understanding"
  - "Long Context"
  - "Multimodal Dataset"
  - "Native Document Model"
  - "Multi-page Reasoning"
date: 2026-05-08
content_hash: c7e0edafc1034f16
---

# DocoPilot: Improving Multimodal Models for Document-Level Understanding

**Conference**: CVPR 2025  
**arXiv**: [2507.14675](https://arxiv.org/abs/2507.14675)  
**Code**: [https://github.com/OpenGVLab/Docopilot](https://github.com/OpenGVLab/Docopilot)  
**Area**: Information Retrieval  
**Keywords**: Document Understanding, Long Context, Multimodal Dataset, Native Document Model, Multi-page Reasoning

## TL;DR

This paper constructs Doc-750K—a high-quality, document-level multimodal dataset containing 758K question-answer pairs and 3.1M images. Based on this, the authors train Docopilot, a native document understanding model. It outperforms InternVL2-8B by 19.9 percentage points on MM-NIAH, processing multi-page documents efficiently without relying on RAG.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in image-level tasks (such as OCR, VQA, and image captioning), yet they still perform poorly in document-level understanding, which requires extracting and integrating key information across multiple pages. Existing open-source MLLMs are primarily trained on image-level data and lack long-context handling capabilities.

**Limitations of Prior Work**: Retrieval-Augmented Generation (RAG) is currently the mainstream solution for handling long documents, but it suffers from three core weaknesses: (1) **Fragmented retrieval**—retrieved information lacks the overall document layout and structure; (2) **Multi-stage error propagation**—incorrect retrieval propagates directly to downstream answers; (3) **Extra latency overhead**—the retrieval step increases response latency, limiting real-time interaction capabilities.

**Key Challenge**: High-quality, document-level multimodal datasets are extremely scarce due to high labeling costs and the lack of structured pipelines. Without sufficient training data, it is impossible to train native long-context document understanding models, leaving developers reliant on defective RAG approaches.

**Goal** (1) How to efficiently construct large-scale, high-quality document-level multimodal training data? (2) How to train native document-level MLLMs without relying on RAG?

**Key Insight**: The authors leverage the structured nature of academic papers (which have well-defined organizations including titles, abstracts, and experimental sections) to design an automated data-construction pipeline. This pipeline extracts realistic QA pairs from sources like Sci-Hub, arXiv, and OpenReview, avoiding the prohibitive cost of human annotation.

**Core Idea**: By constructing Doc-750K, a large-scale, high-quality document-level dataset, and combining it with engineering optimizations (multimodal data packing + Ring Attention + Liger Kernel), the authors train a native document MLLM. This model simultaneously outperforms RAG-based approaches in both accuracy and efficiency on document-understanding tasks.

## Method

### Overall Architecture

The entire system consists of two major parts: (1) a Data Engine—an automated pipeline transforming raw documents into training data; and (2) Model Training—efficient training and inference of long-context documents based on the ViT-MLP-LLM architecture, implemented through engineering optimizations. The input to the model comprises the document content (either interleaved text-image or multi-image format) along with the question, and the output is the answer.

### Key Designs

1. **Data Engine and the Doc-750K Dataset**:

    - **Function**: Automatically construct large-scale, document-level QA training data.
    - **Mechanism**: The data engine operates in three main steps. **First**, raw documents (PDF/HTML) are collected from sources such as Sci-Hub, arXiv, and OpenReview. **Second**, document content extraction is performed, processing each document into two formats: interleaved text-image format (extracted using the MinerU tool, e.g., `<text>\n<image>\n<text>`) and multi-image format (where each page is rendered as an image). **Third**, QA pairs are constructed: direct extraction of real review Q&As for OpenReview papers; design of 5 proxy tasks (abstract writing, title generation, table/figure captioning, experimental section writing, translation) for structured papers; and GPT-4o-generated QA pairs for other documents (comprising only 4.8%). The final dataset contains 758K questions, 3.1M images, and 251K dialogues, with 31.6% being real-world QA pairs.
    - **Design Motivation**: Harnessing the natural hierarchical structure of academic papers enables the construction of high-quality, diverse document-level QA data without manual annotation. Real QA pairs (e.g., peer-review comments and responses) guarantee dataset quality, while proxy tasks (such as writing an abstract based on the body text) inherently require the model to comprehend cross-page information across the entire document.

2. **The Training Efficiency Optimization Trio**:

    - **Function**: Solve GPU memory bottlenecks and training efficiency challenges during long-document training.
    - **Mechanism**: **(a) Multimodal Data Packing**—using a priority queue to pack multiple short samples into longer sequences, constrained by an image threshold $T_{img}$ and a token threshold $T_{tok}$ to maximize GPU utilization and eliminate padding waste. **(b) Ring Attention**—distributing long sequences into blocks across multiple GPUs, overlapping communication with attention computation to bypass single-device memory limits. **(c) Liger Kernel**—further reducing memory usage and boosting training throughput through kernel fusion, in-place operations, and input chunking.
    - **Design Motivation**: Document-level inputs contain significantly more tokens than conventional image-level inputs (averaging 11,245 text tokens and 6,178 image tokens). Without optimization, training on existing hardware is computationally infeasible.

3. **SFT Data Recipe**:

    - **Function**: Prevent the model from overfitting to the document domain and preserve general capabilities.
    - **Mechanism**: Doc-750K is blended with other open-source datasets to cover four scenarios: multi-page document QA (the core task, including MP-DocVQA, DUDE, etc.), multi-image general QA (MMDU-45K), single-page document QA (DocVQA, ChartQA, etc.), and text-only QA (LongAlpaca, LongCite, etc.).
    - **Design Motivation**: Exclusively training on Doc-750K would cause the model to over-specialize in academic paper contexts. Mixing multi-source data increases model robustness across various document types.

### Loss & Training

Standard next-token prediction and conversational SFT (Supervised Fine-Tuning) are applied. Built on the ViT-MLP-LLM architecture (using InternVL as the base), the model employs a Visual Transformer to encode images, a two-layer MLP for projection alignment, and a pre-trained LLM to generate answers.

## Key Experimental Results

### Main Results

| Model | MM-NIAH Overall | MP-Doc ANSL↑ | MMLong-Doc Acc↑ | DocGenome SP Acc↑ |
|------|----------------|-------------|-----------------|-------------------|
| InternVL2-8B | 41.9 | 79.5 | 18.6 | 60.3 |
| InternVL2-26B | 48.4 | - | - | - |
| Docopilot-2B | **49.2** | 76.2 | 21.8 | 45.1 |
| Docopilot-8B | **61.8** | 84.5 | 31.4 | 66.2 |
| GPT-4o | - | - | 42.8 | 71.8 |

Docopilot-8B outperforms InternVL2-8B by +19.9 points on MM-NIAH and beats InternVL2-26B while requiring only 31% of the latter's inference latency. With less than 10% of the parameters, Docopilot-2B achieves comparable performance to InternVL2-26B.

### Ablation Study

| Configuration | MM-NIAH Overall | Description |
|------|----------------|------|
| InternVL2-8B baseline | 41.9 | No document-level training |
| + Doc-750K only | ~55 | Document data brings a significant boost |
| + SFT mixed data | 61.8 | Mixed training brings further improvements |
| InternVL2-8B + RAG | 51.0 | RAG provides limited gains and increases latency |

### Key Findings

- **Native long-context training significantly outperforms RAG**: Docopilot not only achieves higher accuracy but also dramatically reduces inference latency (by avoiding the retrieval step), exhibiting a pronounced advantage especially in multi-turn interactions.
- **Data quality is more important than data quantity**: The 31.6% of real QA pairs in Doc-750K are critical for performance improvements; purely synthetic data yields limited effectiveness.
- **Small models can also achieve strong document understanding**: Training a 2B model on high-quality document data allows it to match a 26B model, demonstrating that training data quality is more critical for document understanding capabilities than model size.
- **Multimodal data packing significantly enhances training efficiency**, enabling document-level MLLM instruction tuning on consumer-grade GPUs (e.g., dual RTX 4090 level).

## Highlights & Insights

- **Ingenious Data Engine Design**: Leveraging the layout of academic papers (Title -> Abstract -> Experiments -> Fig/Tab captions) to design proxy tasks implicitly trains the model to integrate cross-page information. This "structure-as-supervision" concept can easily transfer to other structured document domains such as legal contracts and medical reports.
- **The optimization trio** turns long-document training from impossible to highly viable, and the recipe is fully reusable for other multimodal tasks requiring long-context capabilities.
- **Utilization of OpenReview peer-review data**: Direct inclusion of real review Q&As guarantees both the depth and diversity of inquiries, serving as a zero-cost source of high-quality data.

## Limitations & Future Work

- **Domain bias toward academic papers**: Doc-750K is predominantly constructed from Sci-Hub, arXiv, and OpenReview, leading to insufficient coverage of business documents (e.g., contracts, financial reports), legal documents, and multilingual materials.
- **Benchmark limitations**: Evaluations like MM-NIAH primarily focus on "needle-in-a-haystack" information retrieval, rather than thoroughly testing complex reasoning processes (such as cross-page causal reasoning or multi-step calculations).
- **Potential hallucinations in GPT-4o-generated QA**: Even though it only accounts for 4.8% of the dataset, this portion has not been manually verified for quality.
- **No structural architectural novelty**: The model adopts the standard ViT-MLP-LLM pipeline; the primary contributions lie in the corpus and the training strategy.

## Related Work & Insights

- **vs M3DocRAG**: M3DocRAG is a retrieval-augmented model. While its ANSL of 84.4 on MP-Doc is slightly higher than Docopilot's 76.2 (the 2B variant), RAG-based systems increase inference latency and suffer on tasks demanding global understanding like MM-NIAH.
- **vs Docmatix**: Docmatix features 9.5 million QA pairs but focuses predominantly on image-level tasks. Doc-750K, while smaller in scale, is dedicated to the document level (with average tokens exceeding 11K) and includes authentic QA pairs.
- **vs mPLUG-DocOwl2**: DocOwl2 only scores 6.6 on MM-NIAH, demonstrating that pure architectural innovation yields poor results without document-level training data, highlighting the critical value of Doc-750K.

## Rating

- Novelty: ⭐⭐⭐ No major architectural innovations; the core contributions lie in the dataset and engineering optimizations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluated on multiple document-level benchmarks with fair comparisons against RAG solutions.
- Writing Quality: ⭐⭐⭐⭐ The data engine is clearly described, making the pipeline highly reproducible.
- Value: ⭐⭐⭐⭐⭐ Represents the first large-scale, high-quality document-level multimodal dataset, filling a critical gap in the field and providing immense value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](../../ACL2025/multimodal_vlm/moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)

</div>

<!-- RELATED:END -->
