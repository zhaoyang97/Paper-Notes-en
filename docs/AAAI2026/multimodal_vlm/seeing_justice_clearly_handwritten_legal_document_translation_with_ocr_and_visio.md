---
title: >-
  [Paper Note] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Handwritten document recognition] This paper systematically compares traditional OCR+machine translation (OCR-MT) pipelines against vision large language models (vLLMs) on the task of translat…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Handwritten document recognition"
  - "OCR"
  - "vision-language models"
  - "legal document translation"
  - "low-resource languages"
date: 2026-05-08
content_hash: 92b14824e0098a2c
---

# Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2512.18004](https://arxiv.org/abs/2512.18004)  
**Code**: [github](https://github.com/anviksha-lab-iitk/SJC)  
**Area**: Multimodal VLM
**Keywords**: Handwritten document recognition, OCR, vision-language models, legal document translation, low-resource languages

## TL;DR

This paper systematically compares traditional OCR+machine translation (OCR-MT) pipelines against vision large language models (vLLMs) on the task of translating handwritten Marathi legal documents into English. The study finds that neither approach meets legal-grade deployment requirements: OCR-MT suffers severely from cascading errors, while vLLMs exhibit critical hallucination issues. Nevertheless, vLLMs demonstrate potential for unified end-to-end processing.

## Background & Motivation

### State of the Field
The Indian judicial system is among the most complex legal systems in the world. Grassroots courts and police stations still rely heavily on handwritten documents, including First Information Reports (FIRs), case diaries, witness statements, and court proceedings. These documents are critical to criminal and civil proceedings, yet their handwritten and unstructured nature renders filing, retrieval, and analysis extremely difficult.

### Limitations of Prior Work

**Challenges in handwritten text recognition**: Handwriting styles vary widely and writing quality is inconsistent; conventional OCR systems (e.g., Tesseract, EasyOCR, PaddleOCR) perform poorly on handwritten legal documents.

**Low-resource language challenges**: Indian languages such as Marathi lack large-scale digitized corpora, posing data scarcity problems for both OCR and translation models.

**Cascading error propagation**: In OCR-MT pipelines, recognition errors from the OCR stage directly degrade downstream translation quality, causing loss of legal semantics.

**Specificity of legal terminology**: Legal documents contain specialized terminology, official stamps, signatures, and structured tables, increasing the difficulty of recognition and translation.

### Root Cause
The digitization of legal documents urgently requires accurate and scalable translation systems, yet existing technical approaches—whether modular OCR-MT or end-to-end vLLMs—face fundamental limitations when handling handwritten, low-resource language legal documents.

### Starting Point
The paper constructs a unified evaluation framework that systematically compares the two paradigms (OCR-MT vs. vLLM) in realistic legal document scenarios, providing actionable baselines and directional guidance for future research.

## Method

### Overall Architecture
Rather than proposing a novel method, this work constructs a **systematic comparative experimental framework** to evaluate two major categories of approaches on the task of handwritten Marathi legal document translation.

### Key Designs

1. **OCR-MT Pipeline (6 combinations)**:

    - **OCR tools**: Three OCR engines—Tesseract, EasyOCR, and PaddleOCR
    - **Translation models**: IndicTrans2 (a Transformer encoder-decoder model supporting 22 Indian languages) and Sarvam-1 (a 2B-parameter model optimized for 10 Indian languages)
    - **Workflow**: Scanned document image → OCR text extraction → machine translation → English output
    - **Design Motivation**: The modular architecture facilitates pinpointing performance bottlenecks—whether quality degradation originates in the OCR stage or the translation stage

2. **vLLM End-to-End Translation (3 models)**:

    - **Model selection**: Chitrarth (an Indian-language vision-language bridging model), Maya-8B (a multilingual instruction-tuned model), and Ovis2 (34B int4-quantized and 16B variants)
    - **Mechanism**: Handwritten document images are fed directly into vLLMs, with zero-shot prompting instructing the model to produce English translations
    - **Design Motivation**: Bypasses the intermediate OCR step to avoid cascading errors, leveraging the multimodal reasoning capabilities of vLLMs

3. **Evaluation Protocol Design**:

    - **OCR evaluation**: Character Error Rate (CER) and Word Error Rate (WER) are used to measure the fidelity of Marathi text extraction
    - **Translation evaluation**: Human evaluation along three dimensions—**fluency** (grammatical correctness), **adequacy** (preservation of original meaning), and **correctness** (alignment with the gold standard)
    - **Dataset**: Approximately 60 scanned PDF Marathi legal documents from real legal sources, translated by native speakers and reviewed by legal language experts

## Key Experimental Results

### Main Results

| Method | Representative Model | Handwritten Text Performance | Translation Quality | Main Issues |
|--------|---------------------|------------------------------|---------------------|-------------|
| OCR-MT | EasyOCR + IndicTrans2 | Acceptable for print; poor for handwriting | Severely affected by OCR errors | Cascading errors; loss of legal semantics |
| OCR-MT | PaddleOCR + Sarvam-1 | Worst | Mixed-language output | Weakest handwriting support |
| OCR-MT | Tesseract + IndicTrans2 | Moderate | Incomplete translations | Lack of handwriting adaptation |
| vLLM | Chitrarth | Fails to recognize | Complete hallucination | Generates fictitious meeting content |
| vLLM | Maya-8B | Partial recognition | Irrelevant output | Misidentifies legal documents as study guides |
| vLLM | Ovis2-34B (int4) | Partial recognition | Partially correct but fabricated content | Recognizes structure but introduces semantic errors |
| vLLM | Ovis2-16B | Relatively best | Partial translation | Incomplete and partially incoherent |

### Ablation Study (OCR Model Comparison)

| OCR Model | Print Performance | Handwriting Performance | Overall Assessment |
|-----------|------------------|-------------------------|--------------------|
| EasyOCR | Good | Moderate (still struggles) | Best among the three |
| PaddleOCR | Moderate | Poor | Errors in digit and date recognition |
| Tesseract | Moderate | Poor | Limited low-resource language support |

### Key Findings

1. **The OCR stage is the primary bottleneck in the OCR-MT pipeline**: EasyOCR achieves the best performance among the three OCR tools, yet still fails to handle inconsistent handwriting styles effectively.
2. **Severe error propagation**: OCR transliterates "Gaav" (meaning "village") as "Gaon" rather than translating it as "Village," causing complete failure in downstream translation.
3. **Hallucination in vLLMs**: Chitrarth generates descriptions of fictitious meetings, including non-existent names, dates, and locations; Maya-8B outputs legal documents as study guides.
4. **Structural recognition advantage of vLLMs**: The Ovis2 series partially recognizes document structure (e.g., account numbers, names, locations), but content accuracy remains insufficient.
5. **High-stakes nature of legal documents**: In the legal domain, vLLM hallucinations pose serious risks—generating plausible-sounding yet entirely fabricated text.

## Highlights & Insights

1. **Clear problem definition**: The work is grounded in the real needs of the Indian judicial system, targeting a task of genuine practical value.
2. **Comprehensive comparative framework**: Covers 9 combinations across the OCR-MT and vLLM paradigms, with rich evaluation dimensions.
3. **Exposes fundamental issues of vLLMs in high-stakes domains**: Hallucination is not merely a performance concern but a matter of safety and trustworthiness.
4. **Dataset contribution**: A high-quality handwritten Marathi legal document dataset is constructed, translated by native speakers and reviewed by legal experts.
5. **Future research directions**: Concrete directions are proposed, including hybrid OCR-vLLM pipelines, domain-specific fine-tuning, and prompt engineering.

## Limitations & Future Work

1. **Small dataset scale**: Only approximately 60 documents, insufficient to support large-scale quantitative evaluation.
2. **Absence of automatic evaluation metrics**: Translation quality relies primarily on human assessment, limiting reproducibility.
3. **No fine-tuning experiments**: All vLLMs are evaluated under zero-shot settings; the potential of fine-tuning remains unexplored.
4. **Single language pair**: Coverage is limited to Marathi→English; other Indian languages are not addressed.
5. **Hybrid approaches not implemented**: Although combining OCR structural cues with vLLM contextual translation is suggested, no corresponding experiments are conducted.
6. **Missing edge deployment analysis**: Despite claims of concern for low-resource deployment environments, no computational efficiency or model compression experiments are performed.

## Related Work & Insights

- **VISTA-OCR / olmOCR**: Introduces generative, layout-aware OCR pipelines that may be better suited to the complex layouts of legal documents.
- **Nirnayak**: A pioneering work on OCR applications in the Indian legal domain, though constrained by OCR error propagation.
- **TransDocAnalyser**: A framework specifically targeting FIR documents, combining FastRCNN+ViT encoders with a BERT decoder.
- **PLATTER**: An end-to-end handwriting OCR framework supporting 10 Indian languages, serving as a potential upgrade to the OCR module in this work.
- **Insight**: A hybrid OCR+vLLM approach—using OCR for structural detection and vLLMs for contextual translation—may represent the most promising direction.

## Rating
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](../../CVPR2026/multimodal_vlm/seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](../../ACL2026/multimodal_vlm/texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[ACL 2026\] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models](../../ACL2026/multimodal_vlm/when_seeing_overrides_knowing_disentangling_knowledge_conflicts_in_vision-langua.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)

</div>

<!-- RELATED:END -->
