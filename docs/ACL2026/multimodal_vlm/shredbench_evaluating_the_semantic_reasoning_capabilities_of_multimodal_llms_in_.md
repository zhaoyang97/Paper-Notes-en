---
title: >-
  [Paper Note] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction
description: >-
  [ACL2026][Multimodal VLM][Document Reconstruction] ShredBench constructs an evaluation benchmark that tasks multimodal large models with "restoring content from shredded documents." It demonstrates that current MLLMs…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Document Reconstruction"
  - "MLLM Evaluation"
  - "Shredded Documents"
  - "Semantic Reasoning"
  - "OCR Robustness"
date: 2026-05-08
content_hash: f0c2d9d33a5c6188
---

# ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction

**Conference**: ACL2026  
**arXiv**: [2604.23813](https://arxiv.org/abs/2604.23813)  
**Code**: https://github.com/ythere-y/ShredBench  
**Area**: Multimodal VLM / Document Understanding / Multimodal Reasoning  
**Keywords**: Document Reconstruction, MLLM Evaluation, Shredded Documents, Semantic Reasoning, OCR Robustness

## TL;DR
ShredBench constructs an evaluation benchmark that tasks multimodal large models with "restoring content from shredded documents." It demonstrates that current MLLMs, despite strong performance in conventional OCR, generally lack the ability to perform integrated reasoning across visual fragments, reading order, and semantic context.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) already cover scenarios such as OCR, table parsing, information extraction, and visual document Q&A. Common benchmarks often assume input documents are clear, complete, and have stable layouts. Models only need to read text in high-resolution images, parse layouts, and output structured content.

**Limitations of Prior Work**: Real-world document processing does not always involve perfect scans. Paper may be torn, occluded, folded, or shuffled. Models must not only recognize local text but also infer the sequential relationship between fragments. Existing OCR or document parsing benchmarks typically only measure "clarity of vision" and rarely test the "ability to bridge broken visual evidence with linguistic common sense."

**Key Challenge**: Shredded document reconstruction lies between visual jigsaw puzzles and linguistic reasoning. Traditional jigsaws rely more on edge matching, whereas document fragments often feature black text on a white background with sparse boundary information. The truly usable clues are instead grammar, semantics, code structures, and 2D table layouts. Whether MLLMs can integrate these clues is a question that has not been systematically evaluated.

**Goal**: The authors aim to construct an automated, scalable evaluation set with controlled contamination risks, covering natural language, code, and tables. It observes how model capabilities degrade with structural destruction through varying fragment counts.

**Key Insight**: The paper defines document restoration as a set-to-sequence task: the input is a set of unordered fragment images, and the output is the text of the original document. This preserves the complexity of visual input while allowing repeatable evaluation using text similarity and structural metrics.

**Core Idea**: Use a controllable physical rendering pipeline to generate shredded documents, forcing models to rely on cross-fragment semantic bridging rather than clean OCR or simple edge matching.

## Method

### Overall Architecture
The workflow of ShredBench is divided into three steps. First, multi-source documents are collected, including English news, Chinese news, C++/Java/Python code, and scientific tables. Second, the original content is rendered into high-resolution pages, and irregular fragments are generated using Voronoi partitioning and 3D physical rendering. Third, the scattered fragments are provided to the MLLM as a single visual input, and the model is required to output text or table content as close to the original document as possible.

The dataset contains a total of 756 documents, with each document generating three fragment granularities: 8, 12, and 16. The authors emphasize that the data sources can be flexibly replaced with the latest or unseen text to mitigate the impact of training set contamination on evaluation validity.

### Key Designs
1.  **Physical Shredding Generation Pipeline**:
    - **Function**: Converts ordinary Markdown or text content into visual inputs that approximate real torn paper.
    - **Mechanism**: Documents are first rendered into 1600px wide pages via a browser, followed by random sampling of $N \in \{8,12,16\}$ Voronoi seed points to cut the image. Subsequently, paper thickness, wrinkles, shadows, and random rotations are added in Blender, outputting scattered fragments on a 4K canvas.
    - **Design Motivation**: Ordinary rectangular cropping preserves strong pixel continuity, allowing models to bypass semantic reasoning through low-level edge matching. Physical rendering reduces these shortcuts, making the task closer to real-world damaged documents.

2.  **Cross-modal Semantic Recovery Task**:
    - **Function**: Evaluates whether a model can recover a complete text sequence from unordered fragments rather than just recognizing local text on each fragment.
    - **Mechanism**: Given a set of fragments $\mathcal{I}=\{f_1,\dots,f_N\}$, the model must generate text $\hat{T}$ consistent with the original document $D$. The task does not require explicit prediction of fragment coordinates; instead, it measures implicit stitching capability through final text quality.
    - **Design Motivation**: This simultaneously examines OCR, reading order, language model priors, code syntax, and spatial table structures, exposing MLLM global reasoning weaknesses more effectively than simple document OCR.

3.  **Multi-dimensional Evaluation & Semantic Ablation**:
    - **Function**: Measures restoration quality across text similarity, table structure, and code structure.
    - **Mechanism**: Standard text uses NED, BLEU, and ROUGE-L; tables additionally use TEDS; and code includes CodeBLEU in the appendix. A "meaningless text" control set is also constructed, maintaining layout and character length while removing actual semantics.
    - **Design Motivation**: If models solve the task via visual edges, meaningless text should not perform significantly worse. A sharp decline in performance indicates that models rely primarily on semantic priors when successful, while visual matching is insufficient.

### Loss & Training
This paper does not propose a new training method but focuses on benchmarking and evaluation. Inference uses a unified zero-shot prompt, requiring the model to ignore physical noise and restore content verbatim. The temperature is set to 0 or the lowest value supported by the API, and outputs undergo unified post-processing to ensure metrics reflect content recovery rather than formatting noise.

## Key Experimental Results

### Main Results
Overall results show that Gemini 3 Pro and Gemini 3 Flash lead significantly, but even the strongest models degrade as the number of fragments increases. Open-source and specialized OCR models are generally weak at fragment restoration, indicating that "OCR capability" does not equate to "reconstruction capability."

| Model | 8-frag NED↓ / BLEU↑ / ROUGE↑ | 12-frag NED↓ / BLEU↑ / ROUGE↑ | 16-frag NED↓ / BLEU↑ / ROUGE↑ | Observation |
| :--- | :--- | :--- | :--- | :--- |
| Gemini 3 Pro | 0.33 / 0.51 / 0.83 | 0.37 / 0.48 / 0.81 | 0.41 / 0.44 / 0.76 | Strongest overall; degradation is relatively graceful as fragments increase. |
| Gemini 3 Flash | 0.34 / 0.47 / 0.82 | 0.40 / 0.44 / 0.77 | 0.44 / 0.41 / 0.74 | Close to Pro; even stronger in table scenarios. |
| Qwen-VL-Plus | 0.59 / 0.26 / 0.58 | 0.63 / 0.22 / 0.53 | 0.65 / 0.20 / 0.50 | Moderate level; significant performance drop as fragments increase. |
| GLM-4.6v | 0.67 / 0.20 / 0.45 | 0.70 / 0.17 / 0.40 | 0.71 / 0.15 / 0.37 | Recovers some semantics, but global order is unstable. |
| DeepSeek-OCR | 0.86 / 0.02 / 0.12 | 0.87 / 0.01 / 0.09 | 0.87 / 0.01 / 0.10 | Specialized OCR effectively fails on shredded inputs. |

Performance across document types is also insightful. In code, the average NED for Java and C++ is better than Python. The authors suggest that explicit structures like braces and semicolons provide more restoration anchors. In table scenarios, Gemini 3 Flash's NED of 0.49 outperformed Gemini 3 Pro's 0.59, suggesting that semantic-oriented models are not necessarily the best at rigid 2D layouts.

### Ablation Study
The semantic ablation in the appendix directly addresses whether models are merely performing visual jigsaws. The authors constructed 50 documents of meaningless text, keeping layout, character length, and shredding processes constant, and re-evaluated under the 16-fragment condition.

| Model | Real English ROUGE↑ | Meaningless ROUGE↑ | ROUGE Drop | Real English NED↓ | Meaningless NED↓ | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini 3 Pro | 0.73 | 0.33 | -0.40 | 0.35 | 0.65 | Strongest models rely heavily on semantic bridging. |
| Gemini 3 Flash | 0.67 | 0.29 | -0.38 | 0.41 | 0.71 | Visual clues insufficient without semantics. |
| Qwen-VL-Plus | 0.38 | 0.13 | -0.25 | 0.65 | 0.75 | Mid-tier models also show significant degradation. |
| GLM-4.6v | 0.30 | 0.18 | -0.12 | 0.70 | 0.74 | Initially weak semantic utilization leads to smaller drop. |
| GPT-5.1 | 0.15 | 0.08 | -0.07 | 0.80 | 0.81 | Overall weak restoration; minimal gap in control set. |

### Key Findings
- Fragment count is a stable difficulty knob: Gemini 3 Pro's NED only increased by 0.08 from 8 to 16 fragments, while Qwen-VL-Plus increased by approximately 0.14, indicating that stronger models have flatter degradation curves.
- Chinese news is more difficult than English news due to high information density (semantic loss when single characters are cut) and BLEU/ROUGE sensitivity to Chinese segmentation boundaries.
- Code recovery failures mainly stem from line order errors and omitted content; narrow fragments are particularly prone to being ignored as visual noise.
- Semantic ablation shows models do not solve tasks via simple edge jigsaws; without real semantics, models converge to a similar low-performance range.

## Highlights & Insights
- This paper pushes "document understanding robustness" from noise and rotation to the level of structural destruction. The task setup is natural and closer to real-world damaged document processing than traditional OCR benchmarks.
- The data generation pipeline is clever: it uses replaceable text sources to lower contamination risk, 3D rendering to weaken visual shortcuts, and three granularities to create a continuous difficulty gradient.
- Analysis of code and tables is valuable as it demonstrates that semantic reasoning is not a panacea. Code requires syntactic constraints and tables require 2D structural constraints; future models may need explicit structural search or constrained decoding.
- The "meaningless text" ablation is the most critical insight: current MLLM success is highly dependent on language priors, while pure visual stitching capability remains weak when semantics are removed.

## Limitations & Future Work
- Data remains synthetic; although physical rendering is used, real-world shredded paper may include occlusions, folds, stains, material variations, and scan angle biases.
- Evaluation focuses primary on final text similarity without explicitly evaluating fragment ordering or the geometric reconstruction process, making it difficult to distinguish if a model "stitches then reads" or "guesses while reading."
- Metrics for tables and code are still imperfect; string metrics penalize formatting differences, while structural metrics may not cover semantic equivalence.
- Future work could combine ShredBench with search-based reordering, OCR candidate graphs, program syntax checkers, or table structure parsers to construct stronger multi-stage document restoration systems.

## Related Work & Insights
- **vs OmniDocBench / WildDoc**: These benchmarks focus on full document parsing or document robustness in natural scenes; ShredBench completely shuffles the input structure, emphasizing cross-fragment semantic bridging.
- **vs Jigsaw-Puzzles / RePAIR**: Traditional reconstruction tasks rely more on visual or geometric matching; ShredBench's core lies in the constraints imposed by text, code, and table semantics on stitching.
- **vs Pure OCR Models**: Models like DeepSeek-OCR and Hunyuan-OCR may be strong at standard text recognition but perform poorly on shredded inputs, indicating that document restoration requires a global reasoning module.
- **Insight**: For automated research systems, "structural destruction robustness" can serve as an important evaluation dimension for document parsing models when dealing with paper scans, damaged tables, or low-quality PDFs in the future.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The benchmark task is novel and clearly defined; using shredded documents as a probe for MLLM semantic reasoning is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 756 documents, 4 scenarios, 3 fragment granularities, and 14 models, supplemented by semantic ablation and code structure metrics.
- Writing Quality: ⭐⭐⭐⭐☆ The logic is smooth and charts are information-dense; some model naming and timelines feel speculative but do not hinder core understanding.
- Value: ⭐⭐⭐⭐⭐ directly valuable for document understanding, OCR robustness, multimodal reasoning, and real-world damaged document recovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)
- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](scimdr_advancing_scientific_multimodal_document_reasoning.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)
- [\[ACL 2026\] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
