---
title: >-
  [Paper Note] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts
description: >-
  [ICML 2025][Social Computing][Multimodal Fact-Checking] DEFAME is proposed, which is a modular, zero-shot multimodal LLM pipeline. By using a six-stage dynamic workflow (Plan -> Execute -> Summarize -> Develop -> Predict -> Justify) combined with external multimodal tool retrieval for evidence, it achieves end-to-end joint text-image fact-checking, reaching new SOTA performance on three benchmarks: AVeriTeC, MOCHEG, and VERITE.
tags:
  - "ICML 2025"
  - "Social Computing"
  - "Multimodal Fact-Checking"
  - "Large Language Model Agent"
  - "Retrieval-Augmented Generation"
  - "Zero-Shot Reasoning"
  - "Explainable AI"
date: 2026-05-08
content_hash: d931ef00bd0ceda8
---

# DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

**Conference**: ICML 2025  
**arXiv**: [2412.10510](https://arxiv.org/abs/2412.10510)  
**Code**: [GitHub](https://github.com/multimodal-ai-lab/DEFAME/tree/icml)  
**Area**: Social Computing  
**Keywords**: Multimodal Fact-Checking, Large Language Model Agent, Retrieval-Augmented Generation, Zero-Shot Reasoning, Explainable AI

## TL;DR

DEFAME is proposed, which is a modular, zero-shot multimodal LLM pipeline. By using a six-stage dynamic workflow (Plan -> Execute -> Summarize -> Develop -> Predict -> Justify) combined with external multimodal tool retrieval for evidence, it achieves end-to-end joint text-image fact-checking, reaching new SOTA performance on three benchmarks: AVeriTeC, MOCHEG, and VERITE.

## Background & Motivation

Misinformation is spreading at an unprecedented scale and quality, exceeding the capabilities of manual fact-checking. Approximately 80% of professional fact-checking involves multimodal content (text + image). However, existing Automatic Fact-Checking (AFC) systems suffer from the following key limitations:

**Mostly Text-Only**: The vast majority of AFC systems cannot handle multimodal claims and multimodal evidence.

**Lack of Explainability**: Many methods rely on surface-level pattern matching or lexical/visual similarity, failing to generate human-understandable explanations.

**Reliance on Parametric Knowledge**: Without external evidence retrieval, these systems suffer from knowledge cutoff issues and fail when facing recent claims.

**Fragmentation**: Prior works focus individually on subtasks such as evidence retrieval, summarization, or ranking, lacking a unified end-to-end solution.

**Lack of Dynamic Planning**: Most systems adopt a fixed pipeline and cannot flexibly adjust search strategies as needed.

The core motivation of DEFAME is to unify the fragmented research achievements in the AFC field into an end-to-end framework, supporting simultaneous processing of multimodal claims and multimodal evidence for the first time, and achieving transparent and explainable fact-checking through dynamic planning and external tool retrieval.

## Method

### Overall Architecture

DEFAME consists of three core components: a **Multimodal Large Language Model (MLLM)**, a **Multimodal Toolset**, and a **Structured Fact-Checking Report**. The overall framework operates as a dynamic multi-step RAG system inspired by professional fact-checking workflows. At each MLLM invocation, the current status of the fact-checking report is input as context, combined with task-specific descriptions to achieve context-aware reasoning.

The entire fact-checking process is decomposed into six manageable stages:

### Key Designs

#### Stage 1: Plan Actions

Upon receiving a claim, the MLLM is prompted to generate a targeted sequence of actions to retrieve missing information. Since the action space of certain tools is infinite (e.g., Web Search allows arbitrary queries), the planner's goal is to minimize the number of actions and costs. DEFAME tracks executed actions to avoid redundancy and adaptively adjusts when encountering a "dead end." In-context learning is used to guide the model to select tools: Web Search, Image Search, Reverse Image Search (RIS), or Geolocation.

#### Stage 2: Execute Actions

Four specialized tools are invoked based on the planning results:

| Tool | Input | Function | Implementation |
|------|------|------|----------|
| **Web Search** | Text query | Returns Top-3 relevant web pages | Google Search via Serper API |
| **Image Search** | Text caption | Returns up to 3 web page URLs containing matching images | Google Image Search |
| **Reverse Image Search** | Image | Returns up to 3 web page URLs containing the identical image | Google Vision API |
| **Geolocation** | Image | Estimates the most likely country of origin of the image | GeoCLIP model |

Key anti-leakage design: All web-based tools restrict search results to sources published before the claim's publication date; primary fact-checking websites and sites that prohibit automated access are excluded. For each retrieved URL, Firecrawl is used to scrape page content, and the scraper is extended to identify and download images referenced in the page, ensuring full context.

#### Stage 3: Summarize Results

The collected evidence is integrated into the fact-checking report. The MLLM generates abstract summaries of key findings for each tool output, maintaining conciseness and aligning with the existing report. Relevant images are retrieved and incorporated into the report, while irrelevant results are filtered by instructing the MLLM to return NONE.

#### Stage 4: Develop the Fact-Check

Combining the claim with the summarized evidence, the MLLM is guided to step-by-step discuss the veracity of the claim based on the evidence, marking any missing information as "incomplete." This stage provides room for complex reasoning to derive new insights via natural language inference, preparing for the next stage.

#### Stage 5: Predict a Verdict

The MLLM summarizes key findings and selects a verdict category. Key dynamic mechanism: If the model returns NEI (Not Enough Information), the system returns to Stage 1 to retrieve more evidence, allowing up to three iterations. This mimics the iterative nature of human fact-checking.

#### Stage 6: Justify the Verdict

A concise summary is generated to distill key findings and pivotal evidence (with hyperlinks), appended to the end of the full report. This provides a human-readable explanation for end-users while serving as an assistant tool for further manual verification.

### Loss & Training

DEFAME is a **fully zero-shot** system requiring no fine-tuning or training data. Core configuration:

- Temperature is set to 0.01, and top-p is set to 0.9 to control response complexity.
- A maximum of 32 images per scraped webpage are processed to avoid image flooding.
- Interleaved text-image inputs are processed, preserving the original image locations.
- Inputs exceeding the MLLM's maximum context window are truncated.
- Multiple MLLM backbones are supported (GPT-4o, GPT-4o mini, LLaVA-1V, Llama 4).

## Key Experimental Results

### Main Results

Comparison with SOTA methods and GPT-4o baseline on four benchmark datasets (all reporting mean $\pm$ standard deviation over three runs):

| Dataset | Metric | DEFAME | Prev. SOTA | Gain |
|--------|------|--------|-----------|------|
| AVeriTeC | Accuracy | **70.5±0.6** | 65.6 (DeBERTa) | +4.9% |
| MOCHEG | Accuracy | **59.2±0.4** | 48.6 (MetaSum) | +10.6% |
| VERITE (T/F) | Accuracy | **83.9±0.5** | 58.0 (AITR) | +25.9% |
| VERITE (T/OOC) | Accuracy | 78.4±1.0 | 82.7 (AITR) | -4.3% |
| VERITE (T/MC) | Accuracy | **83.3±1.1** | 59.3 (CHASMA) | +24.0% |
| ClaimReview2024+ | Accuracy | **69.7±2.5** | 35.2 (GPT-4o) | +34.5% |

Comparison of different backbones:

| Backbone Model | AVeriTeC | MOCHEG | VERITE | CR2024+ |
|----------|----------|--------|--------|---------|
| GPT-4o | **70.5** | **59.2** | **83.9** | **69.7** |
| GPT-4o mini | 68.8 | 55.5 | 67.1 | 47.7 |
| LLaVA-1V (7B) | 49.3 | 42.1 | 59.3 | 32.6 |
| Llama 4 Scout | 67.0 | 55.0 | 72.3 | 48.8 |

### Ablation Study

| Configuration | MOCHEG (Acc) | VERITE T/F (Acc) | CR2024+ (Acc) | Description |
|------|-------------|-----------------|--------------|------|
| DEFAME (Full) | **59.2** | **83.9** | **69.7** | Baseline |
| w/o Web Search | 42.0 | 81.8 | 59.7 | Key tool for verifying textual claims |
| w/o Image Search | 57.8 | 81.4 | 63.7 | Multimodal evidence retrieval is important |
| w/o Reverse Search | 58.2 | 73.7 | 64.0 | Most impact on VERITE |
| w/o Geolocation | 58.3 | 80.6 | 65.7 | Crucial for image-centric tasks |
| Single Turn | 47.7 | 82.8 | 63.3 | Multi-turn iteration is vital |
| w/o Planning | 58.7 | 83.0 | 68.0 | Dynamic planning improves efficiency + performance |
| w/o Develop | 57.4 | 83.8 | 67.0 | Intermediate reasoning stage helps |
| Unimodal Develop | 56.1 | 82.0 | 65.7 | Multimodal reasoning outperforms text-only |

### Key Findings

1. **Web Search is the most critical tool**: Upon removal, accuracy on MOCHEG drops drastically by 17.2%, as a large number of textual claims reply on web evidence.
2. **Multi-turn iteration mechanism is vital**: The single-turn variant's performance drops significantly (by 11.5% on MOCHEG), confirming the importance of in-depth retrieval.
3. **ClaimReview2024+ reveals the limitations of parametric knowledge**: Direct verification by GPT-4o yields only 35.2%, and adding CoT even decreases it to 31.4%; in contrast, DEFAME reaches 69.7%, proving that external evidence retrieval mitigates temporal dependency.
4. **Human Evaluation**: Across 185 ratings, DEFAME shows no significant difference in coherence compared to GPT-4o CoT, but significantly outperforms the baseline in completeness (whether the verdict is supported by sufficient evidence).
5. **Gap with open-source models is narrowing**: Llama 4 Scout is close in performance to GPT-4o mini, though GPT-4o still leads by a large margin.

## Highlights & Insights

1. **First truly end-to-end multimodal fact-checking system**: Simultaneously handles multimodal claims and multimodal evidence, which no prior work has achieved.
2. **Exquisitely designed six-stage pipeline**: Mimics the human fact-checking workflow, where each stage has clear functions and can be independently validated via ablation.
3. **Temporal generalization capability**: Demonstrated through the ClaimReview2024+ benchmark (containing claims post GPT-4o's knowledge cutoff date) that the system is not constrained by the backbone model's knowledge cutoff.
4. **Fully zero-shot**: Reaches SOTA on multiple heterogeneous benchmarks without requiring any training data or fine-tuning.
5. **First place in AVeriTeC Challenge**: Participated in the competition under the name InFact and achieved the top performance, validating the system's flexibility.
6. **Transparent and explainable**: Generates detailed fact-checking reports including traceable evidence sources and hyperlinks.

## Limitations & Future Work

1. **Credibility of external evidence**: Relying on search engines may introduce unreliable information; it lacks an independent source credibility evaluation module.
2. **System stability**: Web scraping is constrained by access restrictions and large document sizes; open-source models are sensitive to prompt formatting.
3. **Risk of hallucination**: Although human evaluation did not reveal severe hallucinations, the inherent hallucination problem of LLMs has not been fully analyzed.
4. **Failure mode analysis**: Label ambiguity (Refuted vs. Misleading), missing evidence (inability to retrieve video content), reasoning errors (numerical confusion), and premature verdicts.
5. **Cost and latency**: Multi-turn iterative API calls and web scraping incur high computational costs; more efficient planning strategies could be considered.
6. **Video/Audio evidence**: Currently only text and images are supported, which cannot handle crucial evidence embedded in videos.

## Related Work & Insights

- **RAGAR** (Khaliq et al., 2024): The closest predecessor work, but retrieve only textual evidence and converts images into text descriptions, losing visual information.
- **PACAR** (Zhao et al., 2024): Agent-based fact-checking in the textual domain with planning and tool use, but without multimodal support.
- **VERITE/CHASMA** (Papadopoulos et al., 2024): OOC detection benchmarks and methods, but rely on fixed models like CLIP and lack evidence retrieval.
- **GeoCLIP** (Cepeda et al., 2023): Geolocation tool integrated by DEFAME, demonstrating the value of embedding specialized tools into agent frameworks.
- **Insights**: This work demonstrates the powerful potential of LLM Agent + tool use + dynamic planning in knowledge-intensive tasks, which can be extended to medical knowledge verification, legal fact-checking, and other fields.

## Rating

- Novelty: ⭐⭐⭐⭐ — Excellent framework design but the individual components (web search, RIS, etc.) are not entirely new; the core contribution lies in system integration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive with four datasets + new benchmark + ablation + human evaluation + failure analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and structured with detailed method descriptions and an extremely complete Related Work comparison table.
- Value: ⭐⭐⭐⭐ — High practical value, but reliance on commercial APIs (GPT-4o, Serper, Google Vision) limits openness.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](../../ACL2026/social_computing/veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[ACL 2026\] Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking](../../ACL2026/social_computing/diagnosing_llm_arbitration_behavior_over_pre-evidence_epistemic_states_in_rag-ba.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/social_computing/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[NeurIPS 2025\] Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation](../../NeurIPS2025/social_computing/redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)

</div>

<!-- RELATED:END -->
