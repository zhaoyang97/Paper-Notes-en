---
title: >-
  [Paper Note] SciMDR: Advancing Scientific Multimodal Document Reasoning
description: >-
  [ACL2026][Multimodal VLM][Scientific Document Reasoning] SciMDR proposes a synthesize-and-reground data construction framework. It first synthesizes credible QA and reasoning chains on atomic claims and then re-embeds th…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Scientific Document Reasoning"
  - "Multimodal QA"
  - "Data Synthesis"
  - "Long Document Understanding"
  - "Evidence Localization"
date: 2026-05-08
content_hash: 79aa5ffebd75386e
---

# SciMDR: Advancing Scientific Multimodal Document Reasoning

**Conference**: ACL2026  
**arXiv**: [2603.12249](https://arxiv.org/abs/2603.12249)  
**Code**: No public code link found  
**Area**: Multimodal VLM / Scientific Document Understanding  
**Keywords**: Scientific Document Reasoning, Multimodal QA, Data Synthesis, Long Document Understanding, Evidence Localization

## TL;DR
SciMDR proposes a synthesize-and-reground data construction framework. It first synthesizes credible QA and reasoning chains on atomic claims and then re-embeds them into full scientific papers for training, allowing a 7B VLM to approach GPT-5 series performance in scientific multimodal document reasoning.

## Background & Motivation
**Background**: Scientific document understanding is evolving from summary-level QA and chart QA toward full-paper-level reasoning. Real-world scientific questions often require simultaneous reading of body text, figures, tables, captions, and experimental descriptions, alongside locating evidence within long documents.

**Limitations of Prior Work**: High-quality scientific QA data faces a triple contradiction: human annotation is high quality but small scale; data constructed from charts or snippets is more credible but less realistic; generating questions directly from full documents is closer to real usage, but long contexts dilute attention and increase hallucinations, leading to unreliable answers and reasoning chains.

**Key Challenge**: Training scientific assistants requires both faithful supervision signals and realistic full-document tasks. Models trained only on short snippets fail to learn how to find evidence in full papers, while synthesis from full papers directly makes it difficult to guarantee annotation correctness.

**Goal**: To build a large-scale training set SciMDR and an expert-annotated evaluation set SciMDR-Eval, enabling models to learn to locate evidence, link textual and visual elements, and perform multi-step scientific reasoning in full scientific documents, while verifying if synthetic data truly improves scientific QA capabilities.

**Key Insight**: The authors decouple "generating credible QA" from "constructing realistic training tasks." In the first stage, QA and CoT are generated only within small, verifiable atomic contexts. In the second stage, using recorded evidence locations in claims, the QA is re-embedded into full documents with added information localization steps.

**Core Idea**: Lock answers and evidence to atomic claims first, then place the same supervision signal back into the full paper environment, allowing the model to learn to "find evidence first, then reason, and finally answer" within high-noise long contexts.

## Method
The focus of SciMDR is not a new model architecture but a training data generation paradigm for scientific multimodal documents. It parses scientific papers into text, sections, figures, and captions, generates three types of QA (VQA/TQA/MQA) around claims, and transforms these into full-document training samples via document-scale regrounding.

### Overall Architecture
The input consists of filtered scientific paper PDFs from arXiv CoRR and Nature Communications. OCR via MinerU2.5 extracts the body text, sections, figures, tables, and captions, serializing them into JSON. GPT-5.1 then determines if the paper is an original experimental study, filtering out surveys, position papers, tutorials, and purely conceptual articles. The final training data covers approximately 20K papers and 300K QA pairs; the evaluation set comprises 907 high-quality QA pairs manually constructed from 300 arXiv papers by three CS graduate students.

The framework involves two stages. Claim-Centric QA Synthesis is responsible for generating credible data in small contexts; Document-Scale Regrounding is responsible for turning this data into full-paper-level training tasks. The final training format is `(Full Document Context, Question) -> (Information Localization + Reasoning + Final Answer)`.

### Key Designs
1.  **Claim-Centric QA Synthesis**:
    - **Function**: Generates credible QA, answers, and reasoning chains in small, verifiable contexts.
    - **Mechanism**: Each multimodal context unit contains a segment of text, related figures/tables, and captions. The system first identifies sentences in the text that cite visual elements, then temporarily hides visual information to let the LLM extract discrete declarative claims from the text. Finally, visual information is restored, and cross-modal grounding determines if the claim has a visual counterpart, routing it to VQA, TQA, or MQA accordingly.
    - **Design Motivation**: Open-ended QA generation from full papers by LLMs is high-risk. Claims provide an "answer blueprint," turning reasoning chain generation from open inference into "explaining why the answer holds," thereby reducing hallucinations and evidence mismatch.

2.  **Backward Reasoning Chain Construction**:
    - **Function**: Generates imitable and verifiable reasoning chains for each QA.
    - **Mechanism**: The authors treat the claim as a ground-truth conclusion, prompting the model to construct a reasoning process from the question to the evidence and then to the answer around the known conclusion. In other words, the LLM does not need to discover the answer itself but logically connects the question, evidence, and claim.
    - **Design Motivation**: The difficulty of scientific QA lies in evidence retrieval and open-ended inference. Backward construction outsources these difficulties to claim extraction and localization, resulting in more stable CoT supervision.

3.  **Document-Scale Regrounding**:
    - **Function**: Transforms atomic QA into realistic training samples within full documents.
    - **Mechanism**: Since the claim bound to each QA records the locations of text and visual evidence, the system automatically generates Information Localization steps, such as "First check Section X, then cross-reference Table Y." These localization statements are prepended to the synthetic reasoning chain and used as training samples alongside the full paper context.
    - **Design Motivation**: Real users do not crop relevant paragraphs for the model. Re-embedding maintains the noise of long documents while the answer chain remains supported by precise evidence, solving the problem of short contexts being faithful but not realistic.

### Loss & Training
The paper employs supervised fine-tuning rather than new loss functions. The main experiments use Qwen2.5-VL-7B as the base model, trained in two stages: Stage 1 uses VQA and TQA data for 1 epoch with a peak learning rate of $1\times10^{-5}$ and batch size 64; Stage 2 continues with MQA data for 1 epoch at a learning rate of $1\times10^{-6}$. During fine-tuning, the vision encoder and projector are frozen, and only the language model is trained. The SPIQA baseline is also reproduced using the same base model to isolate data quality differences.

## Key Experimental Results

### Main Results
The main table demonstrates the improvement of SciMDR training on Qwen2.5-VL-7B. While some dataset names for paper PDFs in the table are cluttered, the last column corresponds to the authors' SciMDR-Eval; `+ SciMDR` is the model fine-tuned using the 300K data from this study.

| Model | ChartQA | CharXiv-D | CharXiv-R | SPIQA-A | SPIQA-B | SPIQA-C | SciMDR-Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.1 | - | 90.9 | 58.3 | 79.4 | 79.8 | 71.6 | 47.2 |
| GPT-5.2 | - | 95.2 | 73.1 | 79.9 | 75.4 | 74.0 | 49.9 |
| Qwen-3-VL-8B | 87.4 | 74.2 | 40.1 | 73.2 | 64.0 | 62.3 | 34.2 |
| Qwen2.5-VL-7B | 84.6 | 65.0 | 37.7 | 66.4 | 56.6 | 48.9 | 19.8 |
| Qwen2.5-VL-7B + SPIQA | 81.8 | 50.9 | 33.3 | 62.7 | 44.7 | 40.0 | 5.6 |
| Qwen2.5-VL-7B + SciMDR | 86.3 | 75.6 | 37.9 | 68.6 | 58.8 | 47.3 | 49.1 |

Direct comparison with proprietary models shows that SciMDR-Eval is quite difficult and that specialized 7B training can significantly close the gap.

| Model | SciMDR-Eval |
| :--- | :--- |
| GPT-5.2 | 49.9 |
| GPT-5.1 | 47.2 |
| GPT-4o | 24.7 |
| Qwen2.5-VL-7B | 19.8 |
| Qwen2.5-VL-7B + SciMDR | 49.1 |

### Ablation Study
The key analysis in the paper uses LLaVA-1.5-7B as a data quality probe. The authors compare original SPIQA, SciMDR VQA, and SPIQA re-annotated using this claim-centric pipeline at the same scale of 50K samples. The text reports that re-annotating SPIQA improves performance from 35.7 to 39.8, and the output length on CharXiv is approximately 5 times that of the original data, indicating that gains come from reasoning chain quality rather than just data sources.

| Configuration | Key Result | Description |
| :--- | :--- | :--- |
| Qwen2.5-VL-7B base | SciMDR-Eval 19.8 | General VLM struggles with full scientific paper reasoning |
| + SPIQA data | SciMDR-Eval 5.6 | Short-context synthetic data degrades when migrated to full documents |
| + SciMDR data | SciMDR-Eval 49.1 | Info localization + reasoning chains significantly boost real document QA |
| SPIQA re-annotation | 39.8 vs original 35.7 | Claim-centric annotation quality is superior for same-source documents |

### Key Findings
- `+ SciMDR` provides the largest gain on SciMDR-Eval, increasing by 29.3 points from 19.8 to 49.1, nearly matching GPT-5.2's 49.9.
- `+ SPIQA` leads to a decline in most metrics, particularly SciMDR-Eval dropping from 19.8 to 5.6, suggesting that existing short-context synthetic data cannot naturally teach models to find evidence in full papers.
- On CharXiv-D, SciMDR increases from 65.0 to 75.6, showing that claim-centric data serves not only self-built evaluation sets but also transfers to chart-based scientific QA.
- A slight decrease of 1.6 on SPIQA-C suggests that specialized training for full-document localization might sacrifice performance on some original sub-tasks or indicates differences in skill distribution across evaluation sets.

## Highlights & Insights
- The core insight of the paper is decoupling the two goals of data synthesis: faithfulness is guaranteed in small contexts, while realism is restored in full documents. This approach is more stable than "direct long-document generation" and more practical than "snippet-only QA."
- The claim as an intermediate representation is highly effective. It serves both as an answer blueprint for QA generation and as an information localization map for the re-embedding stage, bridging "annotation quality control" and "training task construction."
- Information Localization supervision is an often-overlooked step in scientific document assistant training. While many datasets only provide the final answer and CoT, SciMDR explicitly makes the model state which section/table/figure should be checked first, which is closer to real scientific reading.
- The results serve as a reminder that data scale does not equal data effectiveness. Synthetic data like SPIQA can harm models when tasks are mismatched; a training format specifically oriented toward full documents is key.

## Limitations & Future Work
- The authors acknowledge that training data quality is limited by the proprietary teacher GPT-5.1. Even if atomic claims reduce hallucinations, subtle errors by the teacher in niche scientific fields may still be hard-coded into the student model.
- Experiments focus primarily on STEM, especially Computer Science and Natural Sciences. Whether the SciMDR pipeline applies to Humanities and Social Sciences, where argumentative structures and evidence forms differ, remains unverified.
- Data construction relies heavily on OCR, chart parsing, and section structure extraction. Parsing errors from MinerU2.5 could affect claims, evidence locations, and re-embedding quality; the paper does not systematically quantify this error propagation.
- SciMDR-Eval uses an LLM judge to score open-ended answers. While reasonable, this may introduce judge bias. Future work could include manual reviews, factual consistency checks, and cross-judge stability analysis.

## Related Work & Insights
- **vs ChartQA / CharXiv**: These benchmarks emphasize chart or scientific image understanding; SciMDR emphasizes textual-visual evidence localization and reasoning within full papers, closer to research assistant scenarios.
- **vs SPIQA**: SPIQA is recent scientific paper QA data, but its synthesis method is biased toward short contexts. SciMDR results show that if the goal is full-document reasoning, full-document regrounding must be explicitly included.
- **vs Human-Annotated Scientific QA**: ExpertQA and QASPER have high quality but limited scale. SciMDR scales to 300K QA pairs using claim-centric synthesis while validating effectiveness with a 907-item manual evaluation set.
- **Insight**: For long multimodal materials like medical documents, legal documents, and patent documents, the construction paradigm of "atomic credible annotation + document-level re-embedding" can be adopted.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The core is a practical data construction paradigm rather than a new model; the design of claims as both answer blueprints and evidence maps is clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Strong main results, proprietary model comparisons, and data quality analyses, though OCR error and judge stability analysis are lacking.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear logic; the faithfulness-realism dilemma is well-articulated. There is some rendering pollution of dataset names in the PDF text, but it doesn't hinder primary understanding.
- **Value**: ⭐⭐⭐⭐⭐ Highly inspiring for scientific document VLM training, especially for building research assistants capable of reading full papers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)

</div>

<!-- RELATED:END -->
