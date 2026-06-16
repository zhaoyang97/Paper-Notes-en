---
title: >-
  [Paper Note] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction
description: >-
  [ACL 2026][Multimodal VLM][Page-to-LaTeX] This paper advances scientific PDF OCR from "converting to text/Markdown" to "reconstructing zero-human-intervention compilable page-level LaTeX." It proposes TEXOCR-Bench, TEXOCR-Train, and a two-stage SFT+RLVR training pipeline, enabling a Qwen3-VL-2B derivative model to significantly outperform open-source baselines
tags:
  - ACL 2026
  - Multimodal VLM
  - Page-to-LaTeX
  - RLVR
date: 2026-05-08
content_hash: d0b0e889653b9e87
---
# TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction

**Conference**: ACL2026  
**arXiv**: [2604.22880](https://arxiv.org/abs/2604.22880)  
**Code**: To be confirmed (The paper page contains Data & Models / Code anchors, but current caches and arXiv abs do not retain specific links)  
**Area**: Document OCR / Reinforcement Learning  
**Keywords**: Page-to-LaTeX, Document OCR, Compilable LaTeX, RLVR, Unit Test Reward

## TL;DR
This paper advances scientific PDF OCR from "converting to text/Markdown" to "reconstructing zero-human-intervention compilable page-level LaTeX." It proposes TEXOCR-Bench, TEXOCR-Train, and a two-stage SFT+RLVR training pipeline, enabling a Qwen3-VL-2B derivative model to significantly outperform open-source baselines of similar scale in structural consistency, citation validity, and compilation success rate.

## Background & Motivation
**Background**: Scientific papers are still predominantly distributed as PDFs, but the truly reusable research assets are LaTeX source files. LaTeX preserves formulas, tables, sections, citations, floats, and numbering structures, while allowing for recompilation, editing, and integration into publishing workflows. Recent document OCR has shifted from traditional modular pipelines to end-to-end recognition using MLLMs, with many systems capable of converting PDFs to plain text or Markdown.

**Limitations of Prior Work**: Markdown-level OCR is helpful for "text-like" outputs, but many errors in scientific documents go beyond surface-level character mistakes. A missing closing bracket, incorrect environment boundaries, broken `\ref{}` tags, or misaligned table column separators can render the entire LaTeX project uncompilable or, worse, silently alter the semantics of citations and numbering.

**Key Challenge**: Existing OCR evaluations mostly focus on local transcription similarity, whereas usable LaTeX requires global invariants: correct section hierarchies, floats placed on reasonable pages, closed formula and table syntax, resolvable label-reference links, and a final project that compiles without manual intervention. Simply learning "LaTeX-like strings" via SFT does not equate to learning these executable constraints.

**Goal**: The authors aim to establish a benchmark for evaluating page-level PDF-to-LaTeX, construct a large-scale page-aligned training set, and verify whether verifiable rewards can push models from token imitation toward functional correctness.

**Key Insight**: The paper treats LaTeX reconstruction as an "OCR with Unit Tests" problem. On the evaluation side, nine metrics are defined to cover transcription, structure, and end-to-end usability. On the training side, these metrics are reformulated as page-level pass/fail unit tests, using RLVR to directly reward outputs that are compilable, parsable, and citation-consistent.

**Core Idea**: Using LaTeX compilers and structural checkers as verifiable supervision to shift the OCR model training objective from "generating similar text" to "generating LaTeX projects that pass document unit tests."

## Method
The TeXOCR method consists of three components: TEXOCR-Bench for task definition and evaluation; TEXOCR-Train for providing large-scale page-level supervision; and TEXOCR model training, which performs transcription learning via SFT followed by reinforcement optimization against LaTeX unit tests using RLVR.

### Overall Architecture
The input is a rendered image of a single scientific PDF page, and the output is the corresponding LaTeX or BibTeX fragment. During training, the authors collect LaTeX source packages and PDFs from arXiv (January 2022 to October 2025), parse `.tex` dependencies, merge source files, recover section/table structures, and then slice the PDFs into single-page screenshots aligned with LaTeX fragments. Based on Qwen3-VL-2B, the model undergoes full-parameter SFT on 404K page image-LaTeX pairs, followed by sampling multiple outputs per page and assigning rewards based on automatically constructed unit tests for group-relative policy optimization style updates.

During evaluation, all models follow a unified page-level inference protocol: each page is generated independently, concatenated in document order, and evaluated against nine metrics and an Overall score. The authors compare three inference granularities (single-image, multi-image, and merged multi-page images), finding single-page images to be the most stable.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph BENCH["TEXOCR-Bench: 3 Dimensions & 9 Metrics"]
        direction TB
        M["Transcription Fidelity<br/>CTP / FA / TA"]
        N["Structural Faithfulness<br/>SA / CC / RV"]
        O["End-to-End Usability<br/>DS / sanity / Compilation Success Rate CSR"]
    end
    A["arXiv PDF + LaTeX Source<br/>2022.01–2025.10"]
    subgraph DATA["Page-level LaTeX Alignment Data Construction"]
        direction TB
        B["Parse Dependencies + Merge Canonical Source"] --> C["Body Page Alignment<br/>GPT-5-mini labels page start/end tokens"]
        C --> D["Float Positioning<br/>pdf2figure allocates to pages"]
        D --> E["Reference Pages<br/>Body→LaTeX / Refs→BibTeX"]
    end
    A --> DATA
    DATA --> F["404K Page Image–LaTeX/BibTeX Pairs"]
    F --> G["SFT Transcription Pre-training<br/>Qwen3-VL-2B full-parameter next-token"]
    subgraph RLVR["SFT + RLVR Verifiable Reward Training"]
        direction TB
        H["Sample K LaTeX completions per page"] --> I["Run page-level unit tests T(x)"]
        I --> J["reward = pass rate<br/>Group-relative advantage + KL constraint"]
    end
    G --> RLVR
    BENCH -->|9 metrics converted to unit tests| I
    RLVR --> K["TEXOCR Model<br/>Single-page inference → Sequential concat → Bench Eval"]
```

### Key Designs

**1. TEXOCR-Bench with 3 Dimensions and 9 Metrics: Quantifying "Usability" into Engineering Constraints**

The real risks in scientific document OCR often hide in outputs that "look similar but are unusable"—where high text OCR scores coexist with missing environment boundaries or broken `\ref{}` tags that prevent compilation. To address this, the benchmark splits quality into three dimensions across nine metrics: Transcription Fidelity (Complex Text Preservation CTP, Formula Accuracy FA, Table Accuracy TA); Structural Faithfulness (Section Accuracy SA, Citation Coverage CC, Reference Validity RV); and End-to-End Usability (Document Similarity DS, basic sanity checks, and Compilation Success Rate CSR).

Crucially, evaluation is not just page-by-page scoring; page outputs are merged into a full project for structural parsing and standard LaTeX compilation. Engineering constraints such as sections, citations, floats, formulas, tables, and the compiler itself are thus integrated into the assessment—a model that only writes "LaTeX-like strings" but fails to compile is exposed via the CSR metric.

**2. Page-level LaTeX Alignment: Resolving Natural Mismatches Between Source and Visual Order**

The writing order in LaTeX source and the display order in PDFs often mismatch, especially for floating elements like figures and tables. Failing to handle this page-source misalignment leads to incorrect mapping that subsequent training cannot rectify. Data construction handles this categorically: LaTeX source dependencies are parsed and merged into a canonical source; for body pages, GPT-5-mini assists in identifying start/end tokens to align with source fragments; floats are positioned in the global PDF using pdf2figure and then assigned to the most appropriate page; for reference pages, body areas use LaTeX supervision while reference areas are converted to BibTeX. This ensures that every single-page sample has truly aligned image and LaTeX/BibTeX targets.

**3. SFT + RLVR Verifiable Reward Training: Transforming "Compilability" into Training Signals**

Standard token loss cannot express discrete constraints like "is this LaTeX compilable," "does this label exist," or "are table columns closed." Step 1 (SFT) uses standard next-token prediction to learn high-fidelity transcription, maximizing $\log \pi_\theta(y_t \mid x,p,y_{<t})$. Step 2 (RLVR) samples $K$ completions per page and runs a set of page-level unit tests $T(x)$. The reward is the proportion of passed tests:

$$R(x,y)=|T(x)|^{-1}\sum_{\tau\in T(x)}\mathbb{I}[\tau(y)=\text{pass}]$$

Optimization uses group-relative advantage with a KL constraint to prevent the policy from drifting from the SFT reference. This verifiable reward directly transforms discrete programmable constraints—compilation, citations, structure—into gradient signals, addressing gaps ignored by token loss.

### Loss & Training
Training occurs in two stages. Stage I involves full-parameter SFT on Qwen3-VL-2B for 1 epoch at a learning rate of $1e-5$, with samples consisting of single-page PDF screenshots, format instructions, and LaTeX/BibTeX targets. Stage II applies RLVR on the SFT model: $K$ completions are sampled per page, binary unit tests are executed for transcription, structure, and usability, and the policy is updated using group-relative advantage with KL penalty. Analysis of group sizes $K\in\{4,8,12,16,20,24\}$ indicates that larger $K$ reduces variance and stabilizes RLVR gains.

## Key Experimental Results

### Main Results
TEXOCR-Bench contains 2,135 expert-annotated documents. TEXOCR-Train includes 57K papers and 404K page image-LaTeX/BibTeX pairs, covering 181K images, 231K tables, and 488K formulas. The main experiment evaluates 21 frontier MLLM/OCR models, including GPT-5.3, Qwen3-VL, Qwen2.5-VL, InternVL, DeepSeek-OCR, olmOCR-2, etc.

| Model | Structural Avg | Usability Avg | Transcription Avg | CSR | Overall | Key Conclusion |
|------|----------------|---------------|-------------------|-----|---------|----------|
| GPT-5.3 | 78.2 | 84.6 | 72.7 | 82.7 | 78.5 | Most stable closed-source, rank 1 |
| TEXOCR (SFT + RLVR) | 83.1 | 68.4 | 73.5 | 45.2 | 75.0 | Strongest open-source, highest structural faithfulness |
| TEXOCR (SFT) | 74.0 | 66.0 | 70.1 | 44.3 | 70.0 | Significantly stronger than same-sized base models |
| Qwen3-VL-32B | 55.5 | 74.7 | 76.1 | 58.9 | 68.8 | Strong transcription, unstable structure/citations |
| Qwen3-VL-8B | 39.4 | 74.6 | 72.1 | 59.0 | 62.2 | Decent text/formula, low structural scores |
| Qwen3-VL-2B | 24.3 | 68.5 | 63.8 | 57.4 | 52.2 | Very weak structural ability without LaTeX training |
| olmOCR-2-7B | 14.8 | 66.2 | 61.4 | 36.5 | 47.5 | Markdown/PDF OCR strengths don't transfer to LaTeX |
| DeepSeek-OCR | 1.5 | 59.5 | 31.5 | 50.1 | 30.8 | Outputs Markdown-style; LaTeX structure is invalid |

RLVR gains are concentrated in structure and usability. Moving from SFT to SFT+RLVR increased the Overall score from 70.0 to 75.0, Structural Avg from 74.0 to 83.1, Reference Validity from 74.1 to 86.8, and Citation Coverage from 74.5 to 85.9. This demonstrates that verifiable rewards effectively force the model to prioritize constraints like labels, references, and sections.

### Ablation Study
Ablating unit test rewards shows that removing any category leads to a drop in the corresponding capability, with the Overall score falling from 75.0 to approximately 70-71. Specifically, removing Structural Faithfulness unit tests dropped the structural average from 83.1 to 73.1. Removing Transcription Fidelity dropped formula accuracy from 58.4 to 53.4.

| RLVR Config | Structural Avg | Usability Avg | Transcription Avg | CSR | Overall | Description |
|-----------|----------------|---------------|-------------------|-----|---------|------|
| SFT+RLVR | 83.1 | 68.4 | 73.5 | 45.2 | 75.0 | Full unit test rewards |
| w/o Transcription Fidelity | 77.4 | 67.5 | 68.9 | 48.9 | 71.3 | Transcription metrics (CTP/FA/TA) drop |
| w/o Structural Faithfulness | 73.1 | 67.2 | 70.6 | 46.4 | 70.3 | Sections, citations, references impaired |
| w/o End-to-End Usability | 75.1 | 66.5 | 70.1 | 46.3 | 70.5 | Doc similarity and usability drop |

Inference granularity ablation confirms that while the task is document-centric, feeding multiple pages simultaneously yields worse results due to cross-page interference and resolution loss in merged images.

| Model | Single-Image | Multi-Image | Merged | Conclusion |
|------|--------------|-------------|--------|------|
| Qwen3-VL-2B | 52.2 | 39.1 | 36.9 | Single-page is more stable via local layout preservation |
| GPT-5.3 | 78.5 | 56.9 | 42.6 | Even strong models suffer from multi-page interference |

### Key Findings
- Compilable LaTeX reconstruction is far more difficult than Markdown OCR. Models often preserve text but fail on sections, citations, references, and table syntax.
- RLVR does not just improve text similarity; it pushes the model toward LaTeX invariants: closed environments, valid citations, stable numbering, and compilable fragments.
- GPT-5.3 remains Overall #1, but TEXOCR (SFT+RLVR) surpasses it in Structural Faithfulness Avg, proving targeted training can compensate for smaller model capacity.
- Compilation Success Rate (CSR) is a rigorous metric. TEXOCR's CSR is only 45.2 (vs. GPT-5.3's 82.7), indicating significant room for improvement in automated scientific publishing.
- Single-image inference is currently superior, though it highlights the limitation that document-level consistency is not fully resolved.

## Highlights & Insights
- The paper's major contribution is defining the document OCR objective with "engineering realism." Most benchmarks stop at Markdown, but research workflows require compiler-ready source files that maintain citation semantics.
- Using unit tests as RLVR rewards is intuitive. LaTeX is one of the few generation tasks where syntax, citations, tables, and formulas can be automatically verified, making RL independent of fragile preference models.
- Handling float placement in data construction is critical. The mismatch between source and PDF positions is a fundamental LaTeX trait; failing to resolve this introduces noise into page-level supervision.
- The evaluation dimensions are transferable to other "executable document/code generation" tasks, such as notebook OCR, HTML/CSS restoration, or CAD script reconstruction.

## Limitations & Future Work
- The work focuses on page-level reconstruction. Real LaTeX projects involve cross-page structures, global macros, shared bibliographies, and long-distance references that simple page concatenation cannot fully guarantee.
- CSR is vital but sensitive to compilation environments, missing packages, and preamble strategies. Different templates or custom commands increase complexity.
- Data mainly comes from arXiv; coverage of non-English papers, complex templates, handwritten annotations, and low-quality scans needs expansion.
- RLVR rewards are pass/fail, which are sparse. Future work could incorporate differentiable rendering similarity or compilation log localization for finer-grained rewards.

## Related Work & Insights
- **vs. PDF-to-Markdown OCR**: READoc, OmniDocBench, and olmOCRBench evaluate text/Markdown extraction. This work adds requirements for LaTeX structure, citations, and compilation.
- **vs. Formula/Table LaTeX Transcription**: CMER-Bench and Table2LaTeX-RL focus on local elements. TeXOCR unifies formulas, tables, sections, and citations into page-level reconstruction.
- **vs. RL in olmOCR2 / DianJin-OCR-R1**: These works use task-specific rewards for reading order or Markdown; this paper extends verifiable rewards to compilable LaTeX invariants.
- **vs. General MLLM OCR**: Models like Qwen, InternVL, and LLaVA have strong visual recognition but lag in structural metrics without training on LaTeX engineering constraints.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Page-to-LaTeX is known, but the combination of benchmark, training set, and RLVR unit test rewards is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 21 models, 9 metrics, training ablations, and inference granularities with solid evidence.
- Writing Quality: ⭐⭐⭐⭐☆ Motivations and designs are clear.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for scientific OCR and executable document generation; sets a baseline for future PDF-to-LaTeX research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](scimdr_advancing_scientific_multimodal_document_reasoning.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)
- [\[CVPR 2026\] Reading or Reasoning? Format Decoupled Reinforcement Learning for Document OCR](../../CVPR2026/multimodal_vlm/reading_or_reasoning_format_decoupled_reinforcement_learning_for_document_ocr.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)

</div>

<!-- RELATED:END -->
