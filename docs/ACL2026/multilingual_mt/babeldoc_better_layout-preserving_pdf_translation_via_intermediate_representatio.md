---
title: >-
  [Paper Note] BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation
description: >-
  [ACL 2026][Multilingual & Machine Translation][PDF Translation] This paper proposes BabelDOC: a layout-preserving PDF translation system based on "Intermediate Representation (IR)". By decoupling visual layout from seman…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "PDF Translation"
  - "Intermediate Representation"
  - "Adaptive Typesetting"
  - "Formula Placeholders"
  - "Terminology Consistency"
date: 2026-05-08
content_hash: 7ce4c796a171c80f
---

# BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation

**Conference**: ACL 2026  
**arXiv**: [2605.10845](https://arxiv.org/abs/2605.10845)  
**Code**: https://github.com/funstory-ai/BabelDOC  
**Area**: Multilingual Machine Translation / Document Translation / Layout-aware NLP  
**Keywords**: PDF Translation, Intermediate Representation, Adaptive Typesetting, Formula Placeholders, Terminology Consistency

## TL;DR
This paper proposes BabelDOC: a layout-preserving PDF translation system based on "Intermediate Representation (IR)". By decoupling visual layout from semantic content, it allows NLP operations such as LLM translation, terminology extraction, cross-page context, and formula placeholders to occur at the semantic layer, before re-anchoring to the original layout using an adaptive typesetting engine. On a 200-page benchmark, it outperforms PDFMathTranslate and DeepL Document Translation in BIoU, layout fidelity, and terminology consistency.

## Background & Motivation
**Background**: As cross-linguistic scientific collaboration surges, PDF remains the dominant format for scientific, legal, and technical documents. However, its "designed-for-display" imperative syntax makes translation difficult. Existing approaches fall into two categories: (i) CAT/MT systems (Google, DeepL) focus on text streams, losing significant layout metadata during extraction; (ii) Document parsers (Doc2X, MinerU, Mathpix) excel at one-way PDF → Markdown/LaTeX extraction but do not support reverse re-typesetting, making them unable to "restore the PDF after translation."

**Limitations of Prior Work**: The authors' previous work, PDFMathTranslate, implemented the first end-to-end layout-preserving translation. However, its monolithic pipeline lacked an explicit IR layer, making document-level NLP interventions nearly impossible. This led to terminology inconsistency in long documents, broken context across pages/columns, and difficulty in unified handling of nested XObject/Form/clipping paths. End-to-end models treat translation as a black box, resulting in poor extensibility.

**Key Challenge**: There is a structural trade-off between "translation quality" and "layout fidelity." Operations at the text layer (CAT, LLM) damage the layout, while layout parsers cannot perform reverse reconstruction. There is a lack of an intermediate layer that allows both sides to operate at their most suitable abstraction levels.

**Goal**: (1) Design a bi-directionally operable IR that can be deconstructed from and reconstructed into PDFs; (2) Implement various document-level NLP interventions on the IR (terminology extraction, glossary injection, cross-page stitching, formula placeholders); (3) Use adaptive typesetting to fit translated (usually longer) text into the original bounding boxes; (4) Ensure the system is fully open-source with hot-swappable modules.

**Key Insight**: The translation pipeline is divided into four stages: parser → IR → semantic engine → typesetting. The IR simultaneously carries spatial coordinates + stylistic attributes + semantic content, decoupling upstream parsing from downstream reconstruction. NLP interventions (like glossary injection) are performed only on the IR, without polluting the layout.

**Core Idea**: Use an "explicit IR" to bridge the Document Understanding (DU) and NLP communities, transforming translation into a plugin-friendly transparent pipeline rather than a black-box conversion.

## Method

### Overall Architecture
Five modules operate in sequence: (1) **Decoupled IR Parser**: Standardizes and parses the input PDF into a unified IR, where each element per page (characters, text lines, graphic blocks, inline images) carries bbox, coordinates, and font/style attributes; (2) **Formula & Multimodal Processing**: Identifies formulas and multimodal segments, masking them as placeholders (to prevent LLMs from corrupting mathematical symbols during translation); (3) **Semantic Engine**: Performs LLM translation on the IR, automatically extracting terms to build dynamic glossaries, stitching segments across pages/columns, and performing glossary-constrained generation; (4) **Adaptive Typesetting**: Iteratively searches for a local scaling factor $\gamma$ to fit the translated (often longer) text back into the original bbox; (5) **Nested Structure & CTM Reconstruction**: Manages the nested stack for XObject/Form/clipping paths and the Current Transformation Matrix (CTM), re-rendering the graphics state layer by layer.

### Key Designs

1.  **Bi-directional IR + Formula Placeholders**:
    - **Function**: Deconstructs the PDF into a structured IR that preserves all spatial + stylistic metadata. Formula parsing follows three steps: (a) a script detection unit calculates font-size variance between adjacent characters to identify sub/superscripts; (b) an offset calculation unit determined fragment offsets based on baseline coordinates; (c) a vector reconstruction unit uses offsets to rebuild vector formulas. These non-translated segments are masked as placeholders during the NLP phase and accurately restored according to the IR after translation.
    - **Mechanism**: Traditional LLMs often mistranslate or delete segments when encountering $\int$ or sub/superscripts, damaging mathematical notation. This system marks all non-linguistic content (formulas, inline images, special characters) with placeholders at the IR layer, allowing the LLM to see only "text stream + placeholder IDs". After translation, elements are filled back precisely via the IR. Unlike one-way parsers (Doc2X / MinerU), this IR maintains two sets of metadata: one to enable translation and one to close the loop for reconstruction.
    - **Design Motivation**: Formula corruption is the primary failure mode in PDF translation. The explicit IR resolves the fundamental conflict between "not modifying formulas during translation" and "restoring formulas to original positions," serving as the foundation of the system.

2.  **Semantic Engine: Terminology Extraction + Cross-page + Glossary Constraints**:
    - **Function**: Scans the IR before translation to extract domain terms → constructs a dynamic glossary (or accepts user-uploaded ones) → enforces terminology constraints in the LLM prompt. It also uses the reading order in the IR to merge logical paragraphs split across columns or pages before translation, preventing semantic fragmentation.
    - **Mechanism**: Conventional CAT/MT performs per-paragraph translation, leading to inconsistent terms in long documents (e.g., "Current Transformation Matrix" translated differently across sections). This system uses the IR as a document-level view, where the glossary is explicitly injected into the prompt so all paragraphs share the same constraints. Cross-column stitching uses spatial + reading order information to identify when a sentence at the bottom of one column continues at the top of the next.
    - **Design Motivation**: Terminology consistency and contextual coherence are critical for the readability of technical documents, which paragraph-level pipelines like PDFMathTranslate cannot address. By providing a document-level view through the IR, these interventions become prompt-engineering tasks rather than architectural overhauls, making modules independently replaceable.

3.  **Adaptive Typesetting Engine**:
    - **Function**: Translations (e.g., English to Spanish) typically expand text length (10–30%), causing original bboxes to overflow. This system uses an iterative bisection search to find the minimum feasible local scaling factor: starting from $\gamma = 1.0$, it checks if the translated text fits within the original bbox. If it overflows, it re-typesets with $\gamma \leftarrow \gamma - 0.05$ (or 0.10) until it fits or reaches a lower bound (typically $\gamma = 0.85$).
    - **Mechanism**: Local scaling is applied per-paragraph based on the bbox constraints provided by the IR, rather than global uniform scaling. This ensures long paragraphs are compressed while short paragraphs remain unchanged, preventing a cluttered visual appearance. Compared to DeepL Document's approach of "inserting text regardless of overflow," adaptive scaling significantly improves layout fidelity.
    - **Design Motivation**: "Text expansion" in cross-linguistic translation is the biggest obstacle to layout-preserving translation. Without local scaling, one must either break the layout or truncate content. This system provides an engineerable solution through simple iterative search; ablations show that removing adaptive typesetting drops LF from 4.5 to 3.0.

### Loss & Training
This is a systems/engineering paper and does not include training objectives. The LLM uses off-the-shelf models with role-play prompts. OCR / layout detection are hot-swappable (e.g., DocLayout-YOLO, YOLOv10).

## Key Experimental Results

### Main Results (200-page Benchmark: 80 Scientific, 60 Technical, 60 Patents)

| System | BIoU ↑ | LF (human) ↑ | TP ↑ | VA ↑ | TC ↑ | UTB (avg untranslated blocks) ↓ |
|---|---|---|---|---|---|---|
| DeepL Document | 19.8% | 3.44 | 3.62 | 3.63 | 4.21 | 2.33 |
| PDFMathTranslate | 48.7% | 3.29 | 3.40 | 3.28 | 3.34 | 6.25 |
| **BabelDOC** | **50.0%** | **4.59** | **4.28** | **4.46** | **4.47** | 2.85 |

LLM-as-a-judge (Gemini-2.5-Flash) follows the same trend: BabelDOC scores highest in LF (4.46), VA (4.49), and TC (4.43), while tying with DeepL at 4.19 for TP. A BIoU of 50% for geometric layout elements directly validates the effectiveness of the IR + adaptive typesetting in "placing elements in the right positions."

### Ablation Study (80-page Representative Subset)

| Variant | LF ↑ | VA ↑ | TC ↑ | Meaning |
|---|---|---|---|---|
| Full BabelDOC | 4.50 | 4.50 | 5.00 | Complete |
| w/o adaptive typesetting | 3.00 | 2.50 | 4.00 | Typesetting reverts; LF/VA both drop |
| w/o glossary/context control | 4.50 | 4.50 | 3.00 | Terminology consistency collapses |

### Key Findings
- **Layout is BabelDOC's strongest selling point**: BIoU is 30 points higher than DeepL, and human-rated LF is more than 1 point higher than all baselines; this is a direct benefit of the IR + adaptive typesetting.
- **BabelDOC ties with DeepL on TP**: This indicates that the value of BabelDOC is not in replacing the MT engine, but in providing a layout-aware controllable framework for the MT engine.
- **UTB is slightly worse than DeepL**: The few untranslated blocks in BabelDOC stem from failures in upstream OCR/layout detection (text inside figures, scanned pages), showing that the layout-preserving pipeline is constrained by the robustness of the parsing end.
- **Perfect division of labor in ablations**: Adaptive typesetting mainly affects LF/VA, while glossary/context control mainly affects TC—proving that the two modules solve orthogonal problems and can be independently upgraded.
- **Ecosystem Impact**: With 8.4K stars and 17 contributors, the extensibility of the IR-based design has successfully attracted the community.

## Highlights & Insights
- **"IR-as-interface" finally enables dialogue between the DU and NLP communities**: Previously, document translation was an island where DU produced parsers and NLP produced MT, joined only by engineering glue. By elevating IR to a first-class citizen, both sides can work at their own levels. This paradigm can be extended to PowerPoint, Word, and Web layouts.
- **Engineering simplicity of adaptive typesetting**: Instead of complex layout optimization or learning, it uses a simple iterative search with a 0.05 step size. It is engineeringly minimal yet outperforms DeepL significantly. It reminds us that "getting the trivial baseline right" is often more important than rushing to neural networks.
- **Formula placeholders + glossary injection are reusable prompt techniques**: These are applicable to any task requiring long documents with mathematical notation to be fed into LLMs (paper translation, textbook rewriting, academic summarization).
- **Open-source ecosystem strategy**: The 8.4K stars and 17 contributors are not just for show. The authors intentionally designed a backend + multiple UIs + plugin architecture, turning a system paper into a long-term ecosystem asset— a release pattern the NLP community should learn from.

## Limitations & Future Work
- IR construction incurs computational overhead; inference latency is significantly higher than pure text API calls (1.63 s/page vs Google 0.38 s/page), making it unsuitable for high-concurrency real-time scenarios.
- Completely dependent on the robustness of upstream OCR / layout detection. PDFs with poor scan quality or eccentric layouts still fail, as evidenced by the UTB results compared to DeepL.
- Translation quality is inherently limited by the LLM; BabelDOC is not an MT engine. Typesetting still needs optimization for language pairs with extreme morphological differences (e.g., vertical Traditional Chinese ↔ Latin scripts).
- Evaluation focused primarily on scientific literature / technical docs / patents; layout-heavy scenarios like novels or graphics-rich magazines were not covered.
- The adaptive scaling step size of 0.05 is empirical; in cases of extreme expansion (e.g., English → German), font sizes might drop to unreadable levels. There is no hard constraint discussion for font size thresholds.

## Related Work & Insights
- **vs PDFMathTranslate (Previous work by same authors)**: Monolithic black box, no IR. BabelDOC decomposes it into IR + modules, allows independent replacement of LLM/OCR, and adds terminology extraction + cross-page context + adaptive typesetting.
- **vs DeepL Document / Google Doc Translate**: Commercial closed-source + text-stream focused, poor layout restoration. BabelDOC is open-source + IR-based + explicit layout control.
- **vs Doc2X / MinerU / Mathpix**: One-way parsing (PDF → Markdown), cannot perform reverse reconstruction. BabelDOC's bi-directional IR is the key differentiator.
- **vs LayoutReader / DocLayout-YOLO**: These are upstream layout models for the IR; BabelDOC treats them as hot-swappable plugins.
- **vs LLM Long-context Translation**: Pure long-context LLM translation loses layout and formulas. BabelDOC adds layout-aware scaffolding to the LLM, representing a pragmatic engineering combination.

## Rating
- Novelty: ⭐⭐⭐⭐ The IR paradigm is not new to document processing, but achieving bi-directional operation + pluginization for PDF translation is a first for system implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 200-page benchmark + human evaluation + LLM-as-judge + component ablations; both quantitative and qualitative analyses are well-executed.
- Writing Quality: ⭐⭐⭐⭐ The five modules are explained clearly, and tables correspond closely with case studies.
- Value: ⭐⭐⭐⭐⭐ 8.4K stars, strong community impact, and industrial usability make it an excellent example of an ACL system demo.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)

</div>

<!-- RELATED:END -->
