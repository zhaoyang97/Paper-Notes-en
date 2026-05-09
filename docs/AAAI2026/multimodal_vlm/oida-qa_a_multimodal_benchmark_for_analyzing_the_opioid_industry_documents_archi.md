---
title: >-
  [Paper Note] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive
description: >-
  [AAAI 2026][Multimodal VLM][Document QA] This paper constructs OIDA-QA, a multimodal document question-answering benchmark based on the UCSF-JHU Opioid Industry Documents Archive (OIDA), comprising 400K training documents and 370K multi-hop QA pairs. A domain-specialized LLM system integrating content recitation and a page finder module is developed to effectively handle multi-turn QA and answer page localization over extremely long documents.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Document QA
  - Multimodal Benchmark
  - Long Context
  - Page Localization
  - Opioid Crisis
date: 2026-05-08
content_hash: 63588140c7621e26
---

# OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive

**Conference**: AAAI 2026  
**arXiv**: [2511.09914](https://arxiv.org/abs/2511.09914)  
**Code**: [HuggingFace](https://huggingface.co/datasets/opioidarchive/oida-qa)  
**Area**: Multimodal VLM  
**Keywords**: Document QA, Multimodal Benchmark, Long Context, Page Localization, Opioid Crisis

## TL;DR
This paper constructs OIDA-QA, a multimodal document question-answering benchmark based on the UCSF-JHU Opioid Industry Documents Archive (OIDA), comprising 400K training documents and 370K multi-hop QA pairs. A domain-specialized LLM system integrating content recitation and a page finder module is developed to effectively handle multi-turn QA and answer page localization over extremely long documents.

## Background & Motivation

**Social Context**: The opioid crisis constitutes a major public health emergency. In 2019, 10.1 million Americans reported opioid misuse, and 90% of the 108,000 drug overdose deaths recorded between June 2021 and May 2022 involved opioids. The UCSF-JHU OIDA contains extensive internal communications and corporate documents from the opioid industry, making it a critical data source for analyzing this crisis.

**Technical Limitations**:
1. OIDA documents are multimodal (scanned documents containing text, images, and layout information) with long-context characteristics, posing challenges for existing LLMs in terms of complex reasoning and hallucination risks.
2. Most LLMs cannot effectively handle multi-turn interactions, making it difficult to process sequential user queries.
3. LLMs frequently fail to ground answers to specific pages or paragraphs within documents, resulting in low answer traceability and credibility.
4. Existing medical QA datasets lack multi-turn dialogue and answer grounding information (cf. Table 1).

**Key Challenge**: The OIDA dataset is large and continuously growing, with documents often spanning dozens of pages across multiple specialized domains including law, medicine, and corporate governance. A low-cost, reliable, and scalable multimodal LLM is needed to assist the public and researchers in analyzing these documents.

**Key Insight**:
1. Systematically extracting three modalities from OIDA PDFs: text, visual, and layout information.
2. Introducing personas to generate diverse questions, ensuring broad coverage.
3. Developing content recitation and page finder mechanisms to address the challenges of extremely long documents.

## Method

### Overall Architecture

The OIDA-QA construction pipeline consists of three major stages: data collection and extraction → QA pair generation → model training and evaluation. The model system comprises an instruction-tuned LLM (Mistral-7B-Instruct-v0.2) and an independent page finder module.

### Key Designs

#### 1. **Data Collection and Multimodal Information Extraction**

**Data Distribution Analysis**: A CLIP model pre-trained on ADOPD and a classification taxonomy are used to label the first page of each document. The cosine similarity between page visual features and classification label text embeddings is computed, and Top-5 labels are selected for document grouping. The 20 largest clusters are ultimately selected.

**Balanced Sampling**: 20K training documents per cluster (400K total) + 500 test documents per cluster (10K total), with balanced sampling across sub-categories and page count dimensions.

**Multimodal Information Extraction** (Figure 3):
- **Text**: OCR extracts raw text → heuristic rules + Doc2Box model groups text lines into semantic paragraphs.
- **Visual**: CLIP labels capture high-level document attributes + Doc2Mask model identifies entity masks.
- **Layout**: Each paragraph $\mathbf{p}_{k,i,j}$ is associated with positional coordinates $\mathbf{l}_{k,i,j} = (p_{k,i,j}, b_x^l, b_y^t, b_x^r, b_y^b)$.

**Design Motivation**: Raw OCR text lines fail to capture semantic relationships (cf. Figure 3); Doc2Box better preserves semantic structure. The combination of three modalities provides a comprehensive document representation for subsequent QA generation and model training.

#### 2. **Persona-Driven Multi-Hop QA Generation**

**Persona Setup**: Personas are sampled from Persona Hub (1B+ personas); GPT-4o generates an average of 48 detailed personas per cluster (including name, age, gender, professional background, experience, and hobbies).

**QA Generation Pipeline** (Algorithm 1):
1. **Question Generator** (GPT-4o): generates questions based on document content and persona attributes.
2. **Answer Generator** (GPT-4o): assesses answerability and generates answers with page number citations.
3. **QA Decomposer** (GPT-4o): decomposes individual QA pairs into multi-turn dialogue sequences.

The pipeline yields 360K+ QA pairs, each answer containing a corresponding page number citation. Additionally, medical professionals (physicians and nurses) were recruited to annotate and refine 100K QA pairs.

**Design Motivation**: Persona diversity ensures question coverage across different professional backgrounds; multi-hop decomposition simulates realistic sequential query scenarios; page citations enhance answer verifiability.

#### 3. **Long Document Processing — Content Recitation and Page Finder**

**Instruction Fine-Tuning**: The LLM is fine-tuned for multi-turn QA with a negative log-likelihood training objective:
$$\mathcal{L}_{\text{QA}}(\theta) = -\sum_{i=1}^N \sum_{j=1}^{K_i} \log P(a_i^j | C_i, q_i^j, H_i^{<j}; \theta)$$

**Content Recitation Augmentation**: The model is trained to output page numbers and relevant context excerpts alongside answers:
$$P(p_i^j, \tilde{C}_i^j | C_i, q_i^j, H_i^{<j}; \theta)$$

A page localization loss $\mathcal{L}_{\text{PF}}$ is jointly trained with $\mathcal{L}_{\text{QA}}$. An additional 64K content recitation training samples are generated.

**Page Finder Module**: An independent retrieval module based on Sentence Transformer (multi-qa-mpnet-base-dot-v1). Training employs Multiple Negatives Ranking Loss:
$$\mathcal{L}_{\text{MNRL}} = -\frac{1}{B}\sum_{b=1}^B \log \frac{\exp(s_{b,b}/\tau)}{\sum_{k=1}^B \exp(s_{b,k}/\tau)}$$

At inference: query-page relevance scores are computed → Top-K pages are selected → adjacent pages are expanded until the context limit $L_{\max}$ is reached → the condensed context is fed into the LLM for answer generation.

**Design Motivation**: Long documents may exceed the model's maximum context window; ground-truth pages are available during training but not at test time. Content recitation bridges this train-test mismatch. The page finder addresses hardware constraints and irrelevant information interference for extremely long documents.

### Loss & Training
- Base model: Mistral-7B-Instruct-v0.2, full-parameter fine-tuning
- 8× NVIDIA H100 GPUs
- Batch size 12, learning rate $5 \times 10^{-6}$, maximum sequence length 8192
- AdamW optimizer + cross-entropy loss
- Page finder fine-tuned independently: batch size 16, learning rate $2 \times 10^{-5}$, warmup ratio 0.1

## Key Experimental Results

### Main Results — QA Performance Under Different Configurations

| Window Size | Recitation | Page Finder | BLEU-1 | METEOR | ROUGE-L | BERTScore |
|-------------|-----------|-------------|--------|--------|---------|-----------|
| None | ✗ | ✗ | 65.9% | 56.8% | 53.7% | 88.5% |
| None | ✗ | ✓ | 73.5% | 63.9% | 61.7% | 90.7% |
| 1 page | ✗ | ✗ | 74.6% | 66.1% | 63.7% | 91.7% |
| 1 page | ✓ | ✓ | **77.0%** | **68.9%** | **66.5%** | **92.3%** |
| 3 pages | ✓ | ✓ | 75.9% | 68.1% | 65.8% | 91.8% |
| Max | ✓ | ✓ | 76.5% | 68.7% | 66.4% | 92.2% |

### Page Localization Performance

| Window Size | Recitation | Page Finder | Page Generation Rate | Page Accuracy |
|-------------|-----------|-------------|----------------------|---------------|
| None | ✗ | ✗ | 68.7% | 83.2% |
| 1 page | ✓ | ✓ | 83.4% | 99.2% |
| 3 pages | ✓ | ✓ | 88.1% | 98.0% |
| Max | ✓ | ✓ | **88.5%** | 97.6% |

### Ablation Study

| Configuration | Key Change | Notes |
|---------------|-----------|-------|
| No context window | BLEU-1 drops to 65.9% | Lack of context causes unstable optimization |
| + Content recitation | BLEU-1: 74.6% → 75.6% (window=1) | Enhances reading comprehension and page localization |
| + Page finder | BLEU-1: 75.6% → 77.0% (window=1) | Further improves precision |
| Both combined | Best overall performance | Complementary gains |
| Window 1 > Window 3 > Window Max | Small window + augmentation > large window | Focused information outperforms redundant information |

### Key Findings
1. **Context window size is critical**: Training without a context window yields the worst performance (65.9% BLEU-1); adding a window brings significant improvements.
2. **Content recitation and page finder are complementary**: They respectively address answer quality and page localization; their combination yields the best results.
3. **Small window + augmentation > large window**: Window=1 with content recitation + page finder (77.0%) outperforms maximum window without augmentation (72.3%), indicating that information focus is more important than information abundance.
4. **Page accuracy reaches 99.2%**: The model can precisely localize the source page of answers, substantially improving answer credibility.
5. Qualitative comparisons with GPT-4 demonstrate comparable multi-hop QA capability and superior page localization.

## Highlights & Insights
1. **First multimodal QA benchmark for opioid industry documents**: Fills a data gap in AI-assisted public health analysis.
2. **Three-modality extraction (text + visual + layout)**: Establishes a comprehensive digitization scheme for scanned documents.
3. **Persona-driven QA generation**: Leverages Persona Hub to simulate diverse user populations, ensuring broad question coverage.
4. **Engineering value of the page finder**: Enables the model to process extremely long documents in GPU-constrained environments such as mobile devices.
5. **Content recitation strategy**: Trains the model to explicitly learn page associations, effectively bridging the train-test mismatch.
6. **Leading data scale**: 370K multi-turn QA pairs, far exceeding existing medical QA datasets.

## Limitations & Future Work
1. QA pairs are generated by GPT-4o, which may introduce model-specific biases; the 100K human-annotated subset represents a limited proportion.
2. The current approach relies solely on text information (OCR) and does not fully leverage the extracted visual and layout information for model training.
3. Mistral-7B may underperform specialized medical LLMs in understanding domain-specific medical terminology.
4. The page finder relies on semantic matching via Sentence Transformer, which may be less effective for layout-dense tabular documents.
5. No systematic comparison is conducted against recent long-context models (e.g., Claude, Gemini).
6. The sequence length is limited to 8192 tokens, still requiring truncation for documents spanning dozens of pages.

## Related Work & Insights
- The multimodal document information extraction pipeline (OCR + Doc2Box + Doc2Mask + CLIP labels) is transferable to other document understanding tasks.
- The persona-driven QA generation methodology offers a reference for constructing conversational datasets in other domains.
- The content recitation strategy is generalizable to other QA systems requiring answer provenance.
- The "retrieve-then-generate" paradigm of the page finder represents a sound RAG practice for document QA.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](../../CVPR2026/multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)
- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](../../ACL2026/multimodal_vlm/when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)
- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](../../ICCV2025/multimodal_vlm/analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)

<!-- RELATED:END -->
