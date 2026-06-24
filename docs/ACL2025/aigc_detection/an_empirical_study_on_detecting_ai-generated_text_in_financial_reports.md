---
title: >-
  [Paper Note] An Empirical Study on Detecting AI-Generated Text in Financial Reports
description: >-
  [ACL 2025][AIGC Detection][AI-Generated Text Detection] Focusing on the highly regulated domain of financial reports, this paper systematically evaluates the performance of various AI-generated text detection methods (statistical features, neural network classifiers, watermark detection, etc.) in identifying AI-generated content in financial documents, revealing the significant impact of domain specificity on detection effectiveness.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "AI-Generated Text Detection"
  - "Financial Reports"
  - "LLM Detection"
  - "Text Authenticity"
  - "Financial NLP"
date: 2026-05-08
content_hash: a9854091dc2e2764
---

# An Empirical Study on Detecting AI-Generated Text in Financial Reports

**Conference**: ACL 2025  
**Area**: AIGC Detection  
**Keywords**: AI-Generated Text Detection, Financial Reports, LLM Detection, Text Authenticity, Financial NLP

## TL;DR
Focusing on the highly regulated domain of financial reports, this paper systematically evaluates the performance of various AI-generated text detection methods (statistical features, neural network classifiers, watermark detection, etc.) in identifying AI-generated content in financial documents, revealing the significant impact of domain specificity on detection effectiveness.

## Background & Motivation

**Background**: As LLMs are widely utilized to assist in or automatically generate financial reports, analyst reports, and compliance documents, detecting AI-generated content in financial text is becoming increasingly crucial. Financial regulatory agencies mandate disclosure authenticity and accountability, whereas AI-generated text may lack a factual analytical basis. Although general-domain AI text detection methods (such as GPTZero, DetectGPT, etc.) have made certain progress, their applicability in the financial domain remains unverified.

**Limitations of Prior Work**: Financial texts exhibit unique stylistic characteristics—heavy use of professional terminology, numerical representations, standardized formats, and conservative language styles. These features may cause financial texts themselves to "look like they are AI-generated" (since humans also use highly templated language) or make AI-generated text easier to disguise as authentic. Consequently, the accuracy of general-purpose detectors may drop significantly when applied to financial texts.

**Key Challenge**: General-purpose AI detectors perform well in domains such as news and academic papers, but the highly formatted and specialized characteristics of the financial domain invalidate the underlying assumptions of these detectors (e.g., that AI text is more "regular" than human text).

**Goal**: (1) To construct a benchmark dataset for AI-generated text detection in the financial domain; (2) to systematically evaluate the performance of existing detection methods in the financial domain; and (3) to analyze failure modes and propose domain adaptation improvements.

**Key Insight**: Real financial reports are collected and corresponding financial text is generated using various LLMs to construct a human-AI paired dataset for systematic evaluation.

**Core Idea**: Comprehensively evaluate and improve the reliability of AI text detection in the specific high-value vertical scenario of the financial domain.

## Method

### Overall Architecture
Data Construction Phase: Real financial reports (10-K, quarterly analysis reports, ESG reports, etc.) are collected from data sources such as SEC EDGAR and Bloomberg, and corresponding texts are generated using GPT-4, Claude 3.5, and Llama 3-70B given the same financial data. Evaluation Phase: A systematic evaluation is conducted on 8 mainstream detection methods, including statistical methods, training-based classifiers, and zero-shot detectors. Analysis Phase: Failure cases are thoroughly analyzed to study the impact of factors such as financial terminology, numerical density, and formatting degree on detection accuracy.

### Key Designs

1. **Financial AI Text Benchmark Dataset (FinAIText Benchmark)**:

    - **Function**: Provides a standardized evaluation set for AI text detection in the financial domain.
    - **Mechanism**: The dataset contains three types of financial text pairs: (1) full-text AI generation—LLMs generate complete financial analysis paragraphs given corporate financial data; (2) paragraph-level AI rewriting—paragraphs from real reports are rewritten by LLMs; (3) mixed text—AI-generated paragraphs are interspersed within real reports. Each type covers multiple financial document categories (financial reports, research reports, ESG reports), totaling approximately 5000 sample pairs.
    - **Design Motivation**: The financial domain lacks a specialized AI text detection benchmark, and the mixed text scenario is closer to practical usage patterns.

2. **Multi-dimensional Detection Method Evaluation**:

    - **Function**: Comprehensively compares the detection capabilities of different technical approaches.
    - **Mechanism**: Evaluates eight methods across three categories—(1) statistical feature methods: perplexity-distribution-based (DetectGPT, Fast-DetectGPT) and vocabulary-diversity-based (GLTR); (2) training-based classifiers: RoBERTa-based binary classifiers, and classifiers pre-trained on general data and fine-tuned on financial data; (3) commercial/zero-shot methods: GPTZero, OpenAI Text Classifier. Each method is evaluated separately on the three text types.
    - **Design Motivation**: Different technical approaches rely on different assumptions; a comprehensive comparison is necessary to understand which approach is most effective in the financial domain.

3. **Domain Adaptation Analysis**:

    - **Function**: Quantifies the impact of financial domain specificity on detection effectiveness.
    - **Mechanism**: Compares the performance of detection methods on general datasets (such as HC3) with that on financial datasets to calculate the "domain gap". It further analyzes specific factors driving this gap: numerical density in financial texts (abundant numbers affect perplexity calculation), formatting degree (highly templated text reduces "naturalness" signals in human text), and professional terminology density (the low-frequency nature of professional vocabularies may mislead statistical methods).
    - **Design Motivation**: To understand the factors causing general-purpose detectors to fail in the financial domain, providing directions for domain adaptation.

### Loss & Training
The fine-tuned classifiers employ standard binary cross-entropy loss, pre-trained on a general AI detection dataset, and subsequently fine-tuned on financial data. Zero-shot methods do not require training.

## Key Experimental Results

### Main Results

| Detection Method | General Domain F1 | Financial Full-Text F1 | Financial Rewritten F1 | Financial Mixed F1 | Domain Gap |
|---------|---------|----------|----------|----------|---------|
| DetectGPT | 89.5 | 72.3 | 65.8 | 51.2 | -17.2 |
| Fast-DetectGPT | 91.2 | 76.1 | 69.4 | 54.7 | -15.1 |
| GPTZero | 87.8 | 68.5 | 62.3 | 48.9 | -19.3 |
| RoBERTa (General) | 93.4 | 74.8 | 71.2 | 56.3 | -18.6 |
| RoBERTa (Fin Fine-tuned) | - | 85.6 | 79.8 | 67.4 | - |
| GLTR | 82.1 | 63.7 | 58.2 | 45.1 | -18.4 |

### Failure Mode Analysis

| Influencing Factor | High Numerical Density (F1) | Low Numerical Density (F1) | Difference | Explanation |
|---------|-------------|-------------|------|------|
| DetectGPT | 64.5 | 78.8 | -14.3 | Numbers degrade the perplexity signal |
| Highly Formatted vs. Lowly Formatted | 67.2 | 79.1 | -11.9 | Templates weaken detection signals |
| Domain-Term Dense vs. General | 69.8 | 77.5 | -7.7 | Specialized vocabulary interferes with statistical features |
| GPT-4 Generated vs. Llama Generated | 68.3 | 76.9 | -8.6 | GPT-4 is more difficult to detect |

### Key Findings
- **General detectors suffer significant accuracy drops in the financial domain**: The F1 score of all methods drops from 82–93 in the general domain to 64–76 in the financial domain, with a domain gap of 15–19 percentage points.
- **Mixed text presents the biggest challenge**: In scenarios where AI paragraphs are interspersed with real text, the detection F1 drops to 45–67, as the authenticity of the surrounding context masks the characteristics of the AI-generated paragraphs.
- **Domain fine-tuning is significantly effective**: The RoBERTa classifier fine-tuned on financial data improves the F1 score from 74.8 to 85.6 ($+10.8$), demonstrating that domain adaptation is critical.
- **High numerical density is a detector "blind spot"**: Text containing a high frequency of numbers (e.g., "revenue grew $15.3\%$ to $\$4.2\text{B}$") significantly degrades the performance of perplexity-based methods, as the low predictability of the numbers themselves masks the statistical patterns of the AI-generated text.

## Highlights & Insights
- First to systematically reveal the challenges of AI text detection in the financial vertical, providing a reliable benchmark evaluation for this regulatorily required scenario.
- The discovery that "numerical density causes detection failure" offers important methodological insights—any detection method relying on perplexity might fail in highly numerical contexts (not only in finance but also in science, engineering, etc.).

## Limitations & Future Work
- The dataset primarily covers English financial reports; the financial text characteristics in other languages (such as Chinese or Japanese) may vary.
- The study only considers text generated by current mainstream LLMs; stronger future models may further narrow the human-AI gap.
- Detection in mixed-text scenarios remains the greatest unresolved issue, necessitating fine-grained paragraph-level detection methods.
- Financial report generation is typically based on actual data; hence, AI-generated content might be factually correct, posing challenges for detection methods based on factual consistency.
- Future studies could incorporate document metadata (such as longitudinal consistency of writing style, comparison with historical reports of the company) to improve detection accuracy.

## Related Work & Insights
- **vs DetectGPT (Mitchell et al., 2023)**: The perturbation-based perplexity method of DetectGPT is ineffective in the financial domain, primarily due to the low-entropy nature of financial text.
- **vs HC3 (Guo et al., 2023)**: HC3 is a general human-ChatGPT comparison dataset; this work fills the gap in the financial domain.
- **vs Binoculars**: Binoculars utilizes a dual-model comparison strategy, which might be more robust in the financial domain but incurs higher computational costs.

## Rating
- Novelty: ⭐⭐⭐⭐ AI detection in the financial domain is a novel and important research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely complete, featuring 8 methods, 3 scenarios, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ The empirical analysis is clear and well-structured.
- Value: ⭐⭐⭐⭐⭐ Holds direct guiding significance for financial regulation and AI detection practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2025\] Cognitive Framework for Detecting AI-Generated Fiction](cognitive_framework_for_detecting_ai-generated_fiction.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)
- [\[ACL 2025\] Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigt_social_media_monitoring.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)

</div>

<!-- RELATED:END -->
