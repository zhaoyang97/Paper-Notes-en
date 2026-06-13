---
title: >-
  [Paper Note] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers
description: >-
  [ACL2026][Medical NLP][Biomedical VLM] Ryze automatically converts biomedical paper PDFs into evidence-enriched QA data that retains charts, captions, structured extractions…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Biomedical VLM"
  - "Evidence-Enriched Data Synthesis"
  - "Scientific PDF Understanding"
  - "Chart-Aware OCR"
  - "GRPO"
date: 2026-05-08
content_hash: bed5b016f3afd4a0
---

# Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers

**Conference**: ACL2026  
**arXiv**: [2606.00902](https://arxiv.org/abs/2606.00902)  
**Code**: https://github.com/Chivier/Ryze  
**Area**: Biomedical NLP  
**Keywords**: Biomedical VLM, Evidence-Enriched Data Synthesis, Scientific PDF Understanding, Chart-Aware OCR, GRPO

## TL;DR
Ryze automatically converts biomedical paper PDFs into evidence-enriched QA data that retains charts, captions, structured extractions, and referring paragraphs. By training BioVLM-8B with progress-gated SFT+GRPO, it achieves $48.0\%$ weighted accuracy on LAB-Bench, outperforming the Qwen3-VL-8B base by $12.6$ percentage points and GPT-5.2 by $3.8$ percentage points.

## Background & Motivation
**Background**: General VLMs are capable of handling everyday vision-language tasks, but scientific paper understanding is distinct from standard visual QA. In biomedical papers, answers are often scattered across multi-column text, figure captions, coordinate axes, legends, multi-row table headers, and in-text explanations of figures. Models must extract these entire evidence chains simultaneously to answer questions regarding experimental design, sequence analysis, protocol tracing, or literature synthesis.

**Limitations of Prior Work**: The bottleneck for domain-specific VLMs is not just model scale, but training data. Expert-annotated biomedical QA pairs are expensive and have narrow coverage. Directly reusing PubMedQA or MedQA results in the loss of visual and structural evidence. Common OCR or Markdown conversion tools frequently misidentify gene names, chemical formulas, chart values, and figure/table references, which subsequent QA synthesis then inherits.

**Key Challenge**: Scientific QA requires "evidence integrity," whereas common data synthesis pipelines only retain local text or figure-caption pairs. Without referring prose, table structures, and chart annotations, training samples might contain answers but actually train the model to memorize shallow patterns rather than learning cross-element, evidence-grounded reasoning.

**Goal**: The authors aim to solve a systemic problem: given a batch of open-access biomedical PDFs, a base VLM, and a target benchmark, can high-quality domain QA data be automatically generated without relying on human annotation to train an 8B-scale model like Qwen3-VL-8B into a locally deployable BioVLM?

**Key Insight**: The key observation of Ryze is that the minimum unit for scientific document data synthesis should not be "text fragments" or "image-caption pairs," but a complete evidence package: visual elements, captions, extracted structures, the paragraphs in the text that cite them, and the context after terminology repair and consistency checks.

**Core Idea**: Replace ordinary text synthesis with evidence-enriched scientific document extraction and QA synthesis, then use SFT to inject domain knowledge and GRPO to strengthen complex evidence-based reasoning.

## Method
Ryze is an end-to-end workflow rather than a single model architecture. Starting from raw PDFs, it performs chart-aware extraction and cleaning, generates QA based on complete evidence packages, uses a progress-gated strategy to decide when to switch from SFT to GRPO, and finally feeds weaknesses exposed by evaluation back into the data generation phase.

### Overall Architecture
Input includes biomedical paper PDFs, a base VLM (Qwen3-VL-8B in the paper), and a target benchmark (LAB-Bench). Ryze first segments the PDF into text blocks, figures, tables, and captions, and restores in-text figure/table cross-references. It then retrieves associated evidence for each question to generate QA with complete evidence. Synthesis, SFT, and evaluation are performed iteratively in increments of approximately 1M tokens; when SFT improvement plateaus, the system switches to GRPO. Finally, diagnostic analysis of weak benchmark categories triggers a new round of paper searching and data augmentation.

### Key Designs
1. **Chart-Aware Extraction and Three-Stage Cleaning**:
	- **Function**: Converts scientific PDFs into a reliable structured evidence store, avoiding OCR misreads and broken cross-element relationships.
	- **Mechanism**: Ryze uses Surya for layout detection to segment pages into text, figures, tables, and captions. Text regions are converted to Markdown retaining sectional structures. In-text references like "Table 1 / Figure 3" are repaired to bind visual elements with captions and related paragraphs. Figures and tables undergo chart/table-aware extraction via GLM-OCR, with tables converted to HTML retaining merged cells and multi-row headers. Finally, Qwen3 is used for hallucination detection, domain terminology repair, and cross-element consistency checks.
	- **Design Motivation**: An incorrect gene name or chart coordinate in a biomedical paper will contaminate all subsequent QA. Ensuring the integrity of the extraction structure and terminology prevents the amplification of OCR errors into model knowledge during synthesis.

2. **Evidence-Enriched QA Synthesis**:
	- **Function**: Generates training samples that are independent of human annotation but still traceable to original paper evidence.
	- **Mechanism**: Question seeds come from two sources: general domain questions from original papers and skill categories abstracted from the target benchmark (e.g., chart interpretation, protocol tracing, literature synthesis). Ryze does not copy benchmark questions or answers; instead, it uses Qwen3-VL-235B to rewrite and diversify these coarse-grained skills, strictly grounding answers to visual elements, captions, OCR annotations, HTML tables, and referring paragraphs retrieved from the source PDF corpus.
	- **Design Motivation**: This approach functions like curriculum-aware active learning: the benchmark informs the system on what capabilities to cover without providing specific questions or answers, thereby targeting LAB-Bench improvements while reducing the risk of direct data leakage.

3. **Progress-Gated SFT-to-GRPO Training Loop**:
	- **Function**: Automatically switches between data synthesis cost and reasoning capability, avoiding blind accumulation of SFT tokens.
	- **Mechanism**: Ryze trains and evaluates an SFT checkpoint every ~1M tokens of synthetic data. When accuracy plateaus, SFT is considered saturated. The data is then frozen and converted to RL format, using GRPO to train the model to generate more coherent reasoning chains. SFT focuses on learning terminology, common sense, and basic biological concepts, while GRPO strengthens reasoning across complex charts, literature, and protocols.
	- **Design Motivation**: Experiments show SFT-only approaches GPT-5.2, but the $+4.3$pp gain that actually surpasses GPT-5.2 primarily comes from GRPO, indicating that "knowing facts first, then learning to reason based on evidence" is more effective than simply increasing synthetic samples.

### Loss & Training
Training is divided into LoRA SFT and GRPO phases. SFT alternates between text QA and visual QA batches, allowing the model to absorb text terminology and chart evidence simultaneously. GRPO does not rely on a separate reward model; it converts accumulated evidence-enriched SFT data into a format that reinforces reasoning chains, focusing on tasks requiring inference across charts, tables, captions, and text. All training configurations use the same token budget: $8,051,591$ tokens for SFT and $1,584,412$ tokens for GRPO. Hardware includes AMD EPYC 7313P CPUs and 4 NVIDIA RTX A6000 48GB GPUs.

## Key Experimental Results

### Main Results
LAB-Bench contains $1,967$ samples across 8 biology categories. Starting from Qwen3-VL-8B, BioVLM-8B reaches $48.0\%$ weighted average accuracy, a $+12.6$pp improvement over the base model and a $+3.8$pp improvement over GPT-5.2.

| Category | Qwen3-VL-8B | GPT-5.2 | BioVLM-8B (SFT only) | BioVLM-8B |
|------|-------------|---------|----------------------|-----------|
| Cloning | 24.2 | 36.4 | 34.5 | 38.4 |
| DbQA | 31.2 | 41.7 | 44.7 | 48.9 |
| FigQA | 24.7 | 36.5 | 31.8 | 35.2 |
| LitQA2 | 38.7 | 45.7 | 58.2 | 65.5 |
| ProtocolQA | 38.3 | 65.7 | 68.1 | 72.3 |
| SeqQA | 43.4 | 47.0 | 39.5 | 42.8 |
| SuppQA | 24.8 | 48.8 | 40.9 | 44.2 |
| TableQA | 34.0 | 36.9 | 40.3 | 45.6 |
| **Weighted Avg** | **35.4** | **44.2** | **43.7** | **48.0** |

### Ablation Study
Ryze validated the data source, OCR pipeline, and cross-model generalization.

| Configuration | Key Metric | Description |
|------|----------|------|
| BioVLM-8B Full Model | 48.0 weighted accuracy | Final result after SFT followed by GRPO |
| BioVLM-8B (SFT only) | 43.7 weighted accuracy | Nearly matches GPT-5.2 (44.2) but lacks reasoning gains |
| PubMedQA SFT | 26.6 weighted accuracy | Far below evidence-enriched data at same token budget |
| MedQA SFT | 29.0 weighted accuracy | Shows standard QA data cannot replace scientific evidence packages |
| Ours OCR pipeline | ChartQA 75.8 | Significantly outperforms general OCR on chart-intensive tasks |
| Without OCR / Marker / DeepSeek OCR | ChartQA 68.0 / 69.3 / 69.1 | Replacing the specialized extraction drops performance by up to ~7.8pp |

### Key Findings
- The largest gains for Ryze come from retaining the complete evidence chain: outperforming GPT-5.2 by $+19.8$pp, $+8.7$pp, and $+7.2$pp on LitQA2, TableQA, and DbQA, respectively.
- GPT-5.2 still leads in FigQA, SeqQA, and SuppQA, indicating that BioVLM's visual understanding and sequence analysis are not yet dominant in all areas.
- Migrating the same evidence-enriched SFT data to other base models also showed improvements: Qwen2.5-7B (from 33.1 to 35.1), LLaMA-3.2 (from 31.3 to 34.4), Gemma-2 (from 31.8 to 33.5).
- Low cost is a systemic highlight: OCR+cleansing ~$18, QA synthesis ~$143, SFT ~$24, GRPO ~$12, totaling less than $200.

## Highlights & Insights
- The most valuable aspect of this paper is not the introduction of a new VLM backbone, but defining the "scientific document evidence package" as the core object of data synthesis. For scientific tasks, data format is the ceiling for model capability.
- Progress gating is practical: it avoids wasting budget on redundant samples after SFT saturation, shifting computation toward GRPO to teach the model how to reason based on existing evidence.
- The boundary regarding benchmark contamination is clear: skill categories are used rather than specific questions/answers. This approach is suitable for domain-customized models, though additional held-out benchmarks are still needed to prove generalization.
- Ryze's pipeline is friendly to small laboratories. With training costs below $200, use of 8B models, and local deployment, it is more suitable than closed-source APIs for privacy-sensitive lab records, internal reports, or unpublished papers.

## Limitations & Future Work
- Current experiments only cover biology/biomedicine. While the authors mention expansion into climate change, geoscience, and civil engineering, systematic reports on these areas are not yet available.
- BioVLM-8B still lags behind GPT-5.2 in FigQA, SeqQA, and SuppQA, suggesting that visual details, sequence analysis, and supportive evidence localization require stronger multimodal RL or better visual extraction.
- The scaling behavior of the progress-gating strategy on larger models is unclear. SFT saturation points and GRPO gains on 8B models may not directly transfer to 32B or 70B models.
- Data generation referenced coarse-grained skill categories from LAB-Bench; future verification should be conducted on benchmarks that were completely uninvolved in curriculum design.

## Related Work & Insights
- **vs LLaVA-Med / PMC-VQA**: These works mostly use medical images or figure-caption pairs to adapt VLMs. Ryze emphasizes the binding of captions, chart structures, and in-text referring prose, making it better for fine-grained scientific reasoning.
- **vs PubMedQA / MedQA SFT**: Standard QA data acts more like text-based knowledge injection. Ryze's data comes from complete evidence packages of original PDFs. PubMedQA and MedQA significantly underperform at the same token budget, showing that data structure is more critical than whether the source is nominally "medical."
- **vs General OCR/Document Parsing Tools**: Tools like Marker or DeepSeek OCR focus on general conversion quality. Ryze is designed for charts and cross-references in scientific papers, making it particularly effective for building chart/table-heavy training data.
- **Insight**: This paradigm can be reused for other scientific fields: first define the domain's evidence package, then perform task-aware synthesis, and finally use weakness feedback to drive incremental data generation rather than simply feeding PDF chunks to an LLM.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Strong system design; core innovation lies in evidence-enriched synthesis and progress-gated training rather than model architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, data source comparisons, OCR ablations, cross-model generalization, and cost analysis are complete, though cross-domain validation is pending.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and system flow with concentrated experimental data; active discussion on benchmark leakage boundaries.
- Value: ⭐⭐⭐⭐⭐ Highly practical for scientific VLM adaptation, especially for low-cost, local, and privacy-sensitive domain model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ICLR 2026\] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](../../ICLR2026/medical_nlp/medagentgym_agentic_training_biomedical.md)
- [\[ACL 2026\] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)
- [\[ACL 2026\] Language Reconstruction with Brain Predictive Coding from fMRI Data](language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)

</div>

<!-- RELATED:END -->
