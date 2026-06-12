---
title: >-
  [Paper Note] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction
description: >-
  [ACL2026][Multimodal VLM][Page-to-LaTeX] This paper advances scientific PDF OCR from "text/Markdown conversion" to "reconstructing zero-human-intervention compilable page-level LaTeX." It proposes TEXOCR-Bench…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Page-to-LaTeX"
  - "Document OCR"
  - "Compilable LaTeX"
  - "RLVR"
  - "Unit Test Reward"
date: 2026-05-08
content_hash: f793e85d7b63176b
---

# TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction

**Conference**: ACL2026  
**arXiv**: [2604.22880](https://arxiv.org/abs/2604.22880)  
**Code**: To be confirmed (The paper page contains Data & Models / Code anchors, but current caches and arXiv abs do not retain specific links)  
**Area**: Document OCR / Reinforcement Learning  
**Keywords**: Page-to-LaTeX, Document OCR, Compilable LaTeX, RLVR, Unit Test Reward

## TL;DR
This paper advances scientific PDF OCR from "text/Markdown conversion" to "reconstructing zero-human-intervention compilable page-level LaTeX." It proposes TEXOCR-Bench, TEXOCR-Train, and a two-stage SFT+RLVR training approach, enabling a Qwen3-VL-2B derivative model to significantly outperform open-source baselines of the same scale in structural consistency, citation validity, and compilation success rate.

## Background & Motivation
**Background**: Scientific papers are still predominantly disseminated as PDFs, but the truly reusable research assets are often the LaTeX source codes. These codes preserve formulas, tables, sections, citations, floats, and numbering structures, while also allowing for re-compilation, editing, and integration into publishing pipelines. Recent document OCR has shifted from traditional modular pipelines to end-to-end MLLM recognition, with many systems capable of converting PDFs to plain text or Markdown.

**Limitations of Prior Work**: While Markdown-level OCR is helpful for "text-like" outputs, many errors in scientific documents are more complex than simple character mistakes. A missing closing bracket, incorrect environment boundaries, corrupted `\ref{}` tags, or misaligned table column separators can prevent the entire LaTeX project from compiling or, worse, silently alter the semantics of citations and numbering.

**Key Challenge**: Existing OCR evaluations focus primarily on local transcription similarity, whereas usable LaTeX requires global invariants: section hierarchies must be correct, floats must be placed on reasonable pages, formula and table syntax must be closed, label-reference links must be resolvable, and the final project must compile without human intervention. Pure SFT learns "LaTeX-like strings," which does not equate to learning these executable constraints.

**Goal**: The authors aim to establish a benchmark for evaluating page-level PDF-to-LaTeX, construct a large-scale page-aligned training set, and verify whether verifiable rewards can push models from token imitation toward functional correctness.

**Key Insight**: The paper treats LaTeX reconstruction as an "OCR with unit tests" problem. On the evaluation side, nine metrics are defined to cover transcription, structure, and end-to-end usability. On the training side, these metrics are reformulated as page-level pass/fail unit tests, using RLVR to directly reward compilable, parsable, and citation-consistent outputs.

**Core Idea**: Utilizing LaTeX compilers and structural checkers as verifiable supervision, the OCR model training objective is shifted from "generating similar text" to "generating a LaTeX project that passes document unit tests."

## Method
The TeXOCR methodology consists of three components: TEXOCR-Bench for defining tasks and evaluation; TEXOCR-Train for providing large-scale page-level supervision; and the TeXOCR model training, which performs SFT for transcription followed by RLVR for LaTeX unit test optimization.

### Overall Architecture
The input is a rendered image of a single scientific PDF page, and the output is the corresponding LaTeX or BibTeX snippet. During training, authors collected LaTeX source packages and PDFs from arXiv (January 2022 to October 2025), parsed `.tex` dependencies, merged source files, and recovered section/float structures. PDF pages were then cropped into screenshots and aligned with LaTeX snippets. Based on Qwen3-VL-2B, the model undergoes full-parameter SFT on 404K page image-LaTeX pairs, followed by sampling multiple outputs for each page to receive rewards from automatically constructed unit tests for group-relative policy optimization style updates.

During evaluation, all models follow a unified page-level inference protocol: each page is generated independently, concatenated in document order, and evaluated against nine metrics and a final Overall score. The authors compared single-image, multi-image, and merged multi-page image inference granularities, finding single-page images to be the most stable.

### Key Designs
1.  **TEXOCR-Bench with Three Dimensions and Nine Metrics**:
    - **Function**: Decomposes PDF-to-LaTeX quality into transcription accuracy, structural faithfulness, and end-to-end usability, rather than relying solely on character similarity.
    - **Mechanism**: Transcription Fidelity includes Complex Text Preservation (CTP), Formula Accuracy (FA), and Table Accuracy (TA). Structural Faithfulness includes Section Accuracy (SA), Citation Coverage (CC), and Reference Validity (RV). End-to-End Usability includes Document-level Similarity (DS), basic sanity checks, and Compilation Success Rate (CSR). Evaluation merges page outputs into a project to run structural parsing and standard LaTeX compilation.
    - **Design Motivation**: Critical risks in scientific document OCR often occur where things "look similar but are unusable." The nine metrics prevent models from scoring high purely through body text OCR, forcing them to handle engineering constraints like sections, citations, floats, formulas, tables, and compilers.

2.  **Page-level LaTeX Alignment Data Construction**:
    - **Function**: Aligns multi-file arXiv source code and final PDFs into supervisable single-page training samples.
    - **Mechanism**: Authors parse LaTeX dependencies and merge them into a canonical source. For body pages, GPT-5-mini assists in identifying page start/end tokens to align with source snippets. Floats like figures/tables are localized using pdf2figure and assigned to the most appropriate page. For reference pages, the body area uses LaTeX supervision while the reference area is converted to BibTeX supervision.
    - **Design Motivation**: LaTeX source order and PDF display order often mismatch, especially for figure/table floats. Without explicit handling of page-source misalignment, models learn incorrect mappings, making it difficult for even strong training objectives to recover structural consistency.

3.  **Verifiable Reward Training via SFT + RLVR**:
    - **Function**: Teaches high-fidelity generation first, then optimizes functional correctness using automated unit tests.
    - **Mechanism**: The SFT stage uses standard next-token prediction, maximizing $\log \pi_\theta(y_t \mid x,p,y_{<t})$. The RLVR stage samples $K$ completions per page; each output passes a set of page-level tests $T(x)$, with the reward defined as the pass rate: 
    $$
    R(x,y)=|T(x)|^{-1}\sum_{\tau\in T(x)}\mathbb{I}[\tau(y)=pass]
    $$
    Optimization uses group-relative advantage with a KL constraint to prevent the policy from deviating from the SFT reference.
    - **Design Motivation**: Token loss struggles to express whether "this LaTeX compiles," "this label exists," or "table columns are closed." Unit test rewards convert these discrete constraints into training signals, particularly suitable for programmable verification in OCR.

### Loss & Training
Training is divided into two stages. Stage I involves full-parameter SFT on Qwen3-VL-2B for 1 epoch at a learning rate of $1e-5$; each sample consists of a page screenshot, formatting prompt, and LaTeX/BibTeX target. Stage II performs RLVR on the SFT model: $K$ outputs are sampled per page to execute three sets of binary unit tests (transcription, structure, usability). The reward is the pass ratio, used to update the policy via group-relative advantage alongside a KL penalty for style stability. Analysis of group sizes $K\in\{4,8,12,16,20,24\}$ showed that larger $K$ reduces variance and stabilizes RLVR gains.

## Key Experimental Results

### Main Results
TEXOCR-Bench comprises 2,135 expert-annotated documents; TEXOCR-Train contains 57K papers and 404K page image-LaTeX/BibTeX pairs across 181K images, 231K tables, and 488K formulas. The main experiment evaluates 21 frontier MLLM/OCR models, including GPT-5.3, Qwen3-VL, InternVL, DeepSeek-OCR, olmOCR-2, and others.

| Model | Structural Avg | Usability Avg | Transcription Avg | CSR | Overall | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.3 | 78.2 | 84.6 | 72.7 | 82.7 | 78.5 | Most stable closed-source, 1st overall |
| **Ours** (SFT + RLVR) | 83.1 | 68.4 | 73.5 | 45.2 | 75.0 | Strongest open-source, highest structural faithfulness |
| **Ours** (SFT) | 74.0 | 66.0 | 70.1 | 44.3 | 70.0 | Significantly stronger than base, lacks structural constraints |
| Qwen3-VL-32B | 55.5 | 74.7 | 76.1 | 58.9 | 68.8 | Strong transcription, unstable citations/structure |
| Qwen3-VL-8B | 39.4 | 74.6 | 72.1 | 59.0 | 62.2 | Decent text/formulas, low structural score |
| Qwen3-VL-2B | 24.3 | 68.5 | 63.8 | 57.4 | 52.2 | Weak structural ability without LaTeX training |
| olmOCR-2-7B | 14.8 | 66.2 | 61.4 | 36.5 | 47.5 | Markdown/PDF OCR strengths do not transfer |
| DeepSeek-OCR | 1.5 | 59.5 | 31.5 | 50.1 | 30.8 | Outputs Markdown style, LaTeX structure nearly invalid |

RLVR gains are concentrated in structure and usability. Overall score improved from 70.0 to 75.0; Structural Avg rose from 74.0 to 83.1, Reference Validity from 74.1 to 86.8, and Citation Coverage from 74.5 to 85.9. This indicates that verifiable rewards effectively push the model to prioritize labels, references, and sections—constraints that token loss fails to optimize stably.

### Ablation Study
The ablation of unit test rewards is direct: removing a category leads to a drop in corresponding capabilities, and the Overall score falls from 75.0 to approximately 70-71. Specifically, removing Structural Faithfulness tests drops the structural average from 83.1 to 73.1. Removing Transcription Fidelity drops formula accuracy from 58.4 to 53.4.

| RLVR Configuration | Structural Avg | Usability Avg | Transcription Avg | CSR | Overall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SFT+RLVR | 83.1 | 68.4 | 73.5 | 45.2 | 75.0 | Full unit test reward |
| w/o Transcription Fidelity | 77.4 | 67.5 | 68.9 | 48.9 | 71.3 | Significant drop in CTP/FA/TA |
| w/o Structural Faithfulness | 73.1 | 67.2 | 70.6 | 46.4 | 70.3 | Sections, citations, references damaged |
| w/o End-to-End Usability | 75.1 | 66.5 | 70.1 | 46.3 | 70.5 | Logic and doc similarity decrease |

Inference granularity analysis suggests that although the task is document-centric, feeding multiple pages at once yields worse results. Multi-image leads to cross-page interference, while merged images suffer from loss of resolution and visual clarity.

| Model | Single-Image | Multi-Image | Merged | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-2B | 52.2 | 39.1 | 36.9 | Single-page preserves layout, more stable |
| GPT-5.3 | 78.5 | 56.9 | 42.6 | Resolution loss affects even closed models |

### Key Findings
- Compilable LaTeX reconstruction is significantly harder than Markdown OCR. Many models preserve body text but fail on section, citation, reference, and table syntax.
- The role of RLVR is not merely to improve text similarity but to push the model toward LaTeX invariants: closed environments, valid citations, stable numbering, and compilable snippets.
- GPT-5.3 remains first Overall, but **Ours** (SFT+RLVR) surpasses it in structural faithfulness average, showing targeted training can compensate for smaller model capacity.
- Compilation Success Rate (CSR) is a rigorous metric. **Ours** achieves a CSR of only 45.2 vs GPT-5.3's 82.7, indicating substantial room for improvement in reliable automated publishing.
- Single-page inference currently outperforms multi-page input, though this highlights a limitation: document-level consistency is not fully resolved.

## Highlights & Insights
- The major contribution is defining document OCR objectives with "engineering realism." While many benchmarks end at Markdown, research workflows require files accepted by LaTeX compilers that maintain citation semantics.
- Using unit tests for RLVR rewards is natural. LaTeX is among the few generation tasks where syntax, citations, tables, and formulas can be automatically verified, removing dependence on fragile preference models.
- Handling float placement during data construction is vital. The mismatch between float positions in source vs. PDF is a fundamental LaTeX trait; failing to resolve this introduces noise into page-to-LaTeX supervision.
- The evaluation dimensions are transferable to other "executable document/code generation" tasks, such as notebook OCR, HTML/CSS restoration, and CAD script reconstruction.

## Limitations & Future Work
- The paper focuses on page-level reconstruction. Real LaTeX projects involve cross-page structures, global macros, shared bibliographies, and long-distance references, which single-page concatenation cannot fully guarantee.
- CSR is crucial for usability but is sensitive to compilation environments, missing packages, and preamble strategies. Custom templates further complicate the task.
- Data largely comes from arXiv; coverage for non-English papers, complex templates, handwritten annotations, and low-quality scans needs expansion.
- RLVR rewards utilize pass/fail tests, which are sparse. Future work could incorporate differentiable rendering similarity and compiler log localization for finer-grained rewards.
- Code/data links are not clearly preserved in the current cache; replication requires model weights or further confirmation from the HTML anchors.

## Related Work & Insights
- **vs. PDF-to-Markdown OCR**: READoc and olmOCRBench evaluate text extraction; this work requires LaTeX structure and compilation, aligning closer to scientific publishing workflows.
- **vs. formula/table LaTeX transcription**: CMER-Bench and Table2LaTeX-RL focus on local elements; TeXOCR unifies formulas, tables, and full-text compilation.
- **vs. RL in olmOCR2 / DianJin-OCR-R1**: These utilize task-specific rewards for reading order or Markdown; this work extends verifiable rewards to compilable LaTeX invariants.
- **vs. General MLLM OCR**: While Qwen and InternVL provide strong visual recognition, they lag in structural metrics without training on LaTeX engineering constraints.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Page-to-LaTeX is not entirely new, but the combination of benchmark, dataset, and RLVR unit test rewards is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 21 models, nine metrics, training ablations, and error analysis with a solid chain of evidence.
- **Writing Quality**: ⭐⭐⭐⭐☆ Motivations and designs are clear, though some resource links require more clarity.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for scientific OCR, executable document generation, and RLVR applications; serves as a strong baseline for future PDF-to-LaTeX work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](scimdr_advancing_scientific_multimodal_document_reasoning.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](../../CVPR2026/multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)

</div>

<!-- RELATED:END -->
