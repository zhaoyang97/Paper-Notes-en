---
title: >-
  [Paper Note] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction
description: >-
  [ACL 2025][Instruction tuning] Proposes Web Reconstruction (WebR), a framework that fully automatically synthesizes high-quality instruction-tuning data from raw web documents. Through a dual-perspective paradigm of "Web as Instruction" and "Web as Response", it generates IT data superior to existing SOTA without human annotation.
tags:
  - "ACL 2025"
  - "Instruction tuning"
  - "Data synthesis"
  - "Web reconstruction"
  - "Dual-perspective paradigm"
  - "Fully automated"
date: 2026-05-08
content_hash: e082db070bd2cdce
---

# Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction

**Conference**: ACL 2025  
**arXiv**: [2504.15573](https://arxiv.org/abs/2504.15573)  
**Code**: [GitHub](https://github.com/YJiangcm/WebR)  
**Area**: Others  
**Keywords**: Instruction tuning, Data synthesis, Web reconstruction, Dual-perspective paradigm, Fully automated

## TL;DR

Proposes Web Reconstruction (WebR), a framework that fully automatically synthesizes high-quality instruction-tuning data from raw web documents. Through a dual-perspective paradigm of "Web as Instruction" and "Web as Response", it generates IT data superior to existing SOTA without human annotation.

## Background & Motivation

The instruction-following capability of LLMs heavily relies on the quality of instruction-response pairs. Existing data synthesis methods have obvious limitations:

**Human Annotation Methods** (e.g., ShareGPT, DOLLY): High cost, limited scale.

**Semi-automated Methods** (e.g., Self-Instruct, Alpaca): Rely on the quality of seed data, limited diversity.

**Fully Automated Methods** (e.g., WebInstruct, Backtranslation): Impose overly strong assumptions on web document structures.

WebInstruct assumes the existence of explicit Q&A pairs in web pages, which limits its applicability. Backtranslation directly treats web content as responses, but web pages often contain irrelevant content and inappropriate expressions. These methods can only process a limited range of web documents, leading to insufficient diversity and restricting downstream performance.

The core motivation of WebR is: Can raw, noisy web documents be fully automatically converted into high-quality instruction-tuning data with minimal assumptions about web page content?

## Method

### Overall Architecture

WebR adopts a Dual-Perspective Paradigm, treating each web document as either an instruction or a response to trigger the reconstruction process. The framework leverages an off-the-shelf powerful LLM to complete all conversions without human intervention or seed data.

### Key Designs

#### 1. Web as Instruction

- **Function**: Combines the raw web page with a synthesized rewrite request to form a complete instruction.
- **Mechanism**: Uses an LLM to generate detailed rewrite requests (such as style, format, or structural adjustments) for the web page content, and then tasks the LLM to reconstruct the web page content according to the request.
- **Design Motivation**: The complexity of rewrite requests naturally covers various NLP tasks like summarization, information extraction, and semantic understanding, forcing the model to exhibit reasoning and comprehension capabilities.
- **Diversity Enhancement**: Generates rewrite requests targeting specific parts of the web page (rather than the entire text) with a 50% probability, simulating scenarios where real users modify only a portion of the text.

#### 2. Web as Response

- **Function**: Treats web content as a potential response and infers the corresponding latent instruction.
- **Mechanism**: Introduces a two-stage refinement process—first letting the LLM generate an initial response (rollout) for the inferred instruction, and then refining the response in combination with the raw web content.
- **Design Motivation**: The initial rollout ensures the response has human-style fluency, while the refinement step integrates key information from the web page, making the final response both precise and comprehensive.
- **Difference from Traditional Backtranslation**: Traditional methods directly treat web content as responses. WebR overcomes the issue of web content being unsuitable as a direct response via rollout+refinement.

#### 3. Persona-driven Instruction Synthesis

- The LLM first generates a persona for the web document, then guides subsequent instruction synthesis with this persona.
- Improves the diversity and targeted nature of the synthesized instructions.

#### 4. Data Construction Details

- Data Sources: 70% Common Crawl (General) + 15% OpenWebMath (Math) + 15% GitHub (Code)
- The ratio of Web as Instruction to Web as Response is 2:1
- Deduplication using MinHash (signature size 128, similarity threshold 0.7)
- Finally synthesized 100k instruction-response pairs

### Loss & Training

- Used Llama3-70B-Instruct to generate WebR-Basic, and GPT-4o-mini to generate WebR-Pro.
- The total call cost for GPT-4o-mini is only $38.57.
- Unified training hyperparameters for a fair comparison.

## Key Experimental Results

### Main Results: Instruction-Following Performance (Table 1)

| IT Data | Data Size | Human Effort | AlpacaEval 2 | Arena-Hard | MT-Bench | IFEval Avg |
|---------|--------|---------|-------------|-----------|---------|-----------|
| Untuned | - | - | 0.18 | 0.31 | 1.78 | 7.31 |
| ShareGPT | 112k | High | 9.89 | 6.49 | 6.34 | 22.70 |
| WildChat | 652k | High | 14.62 | 8.73 | 6.60 | 23.03 |
| Magpie (Llama3) | 100k | None | 23.62 | 13.98 | 6.26 | 24.15 |
| **WebR-Basic** | 100k | None | **25.33** | **16.50** | **6.95** | **28.17** |
| IT Mix (GPT-4o) | 100k | Medium | 30.39 | 28.03 | 7.36 | 31.29 |
| Magpie (GPT-4o) | 100k | None | 32.61 | 27.97 | 7.26 | 29.95 |
| **WebR-Pro** | 100k | None | **34.36** | **31.10** | **7.57** | **33.71** |
| (IT+WebR) Merge | 200k | Medium | **35.40** | **35.12** | **7.59** | **36.36** |

WebR-Basic improves over SOTA Magpie by an average of 16.65% with zero human effort; WebR-Pro outperforms IT Mix and Magpie by 7.73% and 12.55% respectively under the same response generator.

### Ablation Study (Table 3)

| Setting | AlpacaEval 2 | MT-Bench | IFEval | MMLU | MATH |
|------|-------------|---------|--------|------|------|
| WebR-Pro (Full) | 34.17 | 7.50 | 28.41 | 61.15 | 24.94 |
| -w/o Persona | 33.30 | 6.93 | 28.31 | 60.98 | 24.03 |
| -w/o Part (Full text only) | 33.89 | 7.53 | 28.01 | 61.05 | 22.73 |
| -w/o Refinement | 31.61 | 7.42 | 27.92 | 59.83 | 24.36 |
| -w/o MinHash | 32.43 | 7.29 | 27.58 | 60.69 | 24.82 |
| Ratio 1:0 (Instruction only) | 29.15 | 7.10 | 25.27 | 58.79 | 25.74 |
| Ratio 0:1 (Response only) | 33.41 | 6.68 | 27.54 | 52.68 | 23.30 |

### Key Findings

1. **Refinement is most critical**: Removing the refinement step drops AlpacaEval 2 by nearly 3 points.
2. **The two perspectives are complementary**: Web as Instruction enhances reasoning (ARC, MATH), and Web as Response enhances instruction following and Q&A (IFEval, AlpacaEval).
3. **Optimal ratio is 2:1**: A larger proportion of Instruction yields better performance.
4. **Data efficiency**: Performance increases linearly with the logarithmic growth of training data.
5. **Scalability**: Tested on Qwen2.5-1.5B/3B/7B/14B. The larger the model, the more obvious the gain (AlpacaEval +4.12 at 14B).
6. **Compatibility**: Merging WebR with existing IT data can further improve performance.

## Highlights & Insights

- **Minimal Assumption Principle**: Does not rely on the presence of Q&A pairs or clean content in web pages, and can process any web document.
- **Dual-Perspective Complementarity** is a key innovation—different perspectives train different dimensions of capabilities.
- **Extremely Low Cost**: Synthesizing 100k data with GPT-4o-mini costs only $38.57, offering extremely high engineering practicality.
- **Data Diversity** (embedding diversity 0.93) reaches the level of human data, indicating that the natural diversity of web pages is effectively utilized.

## Limitations & Future Work

- Relies on powerful teacher models (Llama3-70B or GPT-4o-mini), performance on weaker models remains unverified.
- Data sources are still limited to English-dominated web pages; multilingual generalization remains to be explored.
- Quality filtering of web pages relies solely on deduplication; no explicit quality filtering is applied.
- The two-stage Web as Response process increases inference overhead.

## Related Work & Insights

- Difference from Self-Instruct/Alpaca: WebR requires absolutely no seed data.
- Difference from WebInstruct: Does not assume the existence of Q&A pairs in web pages.
- Difference from Backtranslation: Introduces a two-stage refinement of rollout+refinement.
- The persona-driven strategy is borrowed from Ge et al. (2024).
- Insight: In the future, domain adaptation (adjusting the ratio of web page sources) can be combined to generate domain-specific IT data rapidly.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-perspective paradigm is an effective new paradigm, and conceptualizing web reconstruction as an IT synthesis task is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Compared with 10+ baselines across 4 benchmarks, comprehensive ablation studies, and validated across multiple model scales.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich diagrams, and well-articulated motivation.
- **Value**: ⭐⭐⭐⭐⭐ — Fully automatic synthesis of high-quality IT data at an extremely low cost, indicating very high practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[ACL 2025\] Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)

</div>

<!-- RELATED:END -->
