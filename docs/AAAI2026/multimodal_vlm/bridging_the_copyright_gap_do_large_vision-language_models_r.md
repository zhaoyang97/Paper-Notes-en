---
title: >-
  [Paper Note] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?
description: >-
  [AAAI 2026][Multimodal VLM][VLM copyright compliance] This paper presents the first systematic evaluation of LVLMs' ability to recognize and respect copyrighted content in multimodal contexts. It constructs a large-scale…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "VLM copyright compliance"
  - "multimodal benchmark"
  - "copyright notice"
  - "CopyGuard"
  - "tool-augmented defense"
date: 2026-05-08
content_hash: a0ee2927d5885aa3
---

# Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?

**Conference**: AAAI 2026 Oral  
**arXiv**: [2512.21871](https://arxiv.org/abs/2512.21871)  
**Code**: [https://github.com/bluedream02/CopyGuard](https://github.com/bluedream02/CopyGuard)  
**Area**: Multimodal VLM / AI Safety / Copyright Compliance  
**Keywords**: VLM copyright compliance, multimodal benchmark, copyright notice, CopyGuard, tool-augmented defense  

## TL;DR
This paper presents the first systematic evaluation of LVLMs' ability to recognize and respect copyrighted content in multimodal contexts. It constructs a large-scale benchmark of 50,000 multimodal query–content pairs, finds that 11 out of 12 SOTA LVLMs fail to refuse infringing requests even when explicit copyright notices are present, and proposes CopyGuard—a tool-augmented framework that raises the infringement rejection rate from ~3% to ~62%.

## Background & Motivation

**Background**: LVLMs such as GPT-4o, Gemini, and Claude are widely deployed in multimodal RAG pipelines, web agents, and search engines, where they inevitably encounter copyrighted content (book excerpts, news articles, song lyrics, code documentation, etc.). Yet these models' ability to recognize and comply with copyright has received almost no systematic study.

**Limitations of Prior Work**: While LVLMs perform reasonably well at refusing direct infringement requests (e.g., "write me a passage from Harry Potter"), when copyrighted content appears as an image in context—such as a scanned book page retrieved via RAG—models reproduce, translate, or paraphrase the content without hesitation, even when a "©" copyright notice is clearly visible in the image.

**Key Challenge**: Safety alignment in existing LVLMs primarily targets coarse-grained risks such as violence and explicit content, with little copyright awareness. Furthermore, copyright law is complex—varying by material type, jurisdiction, and fair use exceptions—making naive fine-tuning prone to over-refusal.

**Goal**: Three research questions are addressed—RQ1: How do LVLMs comply with copyright in multimodal contexts? RQ2: Do copyright notices affect model behavior? RQ3: How can LVLM copyright compliance be improved?

**Key Insight**: Copyrighted content is rendered as images (simulating RAG/screenshot scenarios), and LVLM behavior is evaluated across four infringement scenarios—reproduction, extraction, paraphrasing, and translation—under conditions with and without copyright notices.

**Core Idea**: Construct a large-scale copyright compliance benchmark and propose the tool-augmented CopyGuard defense framework (OCR + search-based copyright verification + query risk analysis + compliance reminding).

## Method

### Overall Architecture
The work comprises two components: (1) a benchmark for systematically evaluating copyright compliance across 12 LVLMs, and (2) the CopyGuard defense framework, which performs copyright checking and risk notification prior to LVLM inference.

### Key Designs

1. **Benchmark Construction**:

    - Function: Constructs 50,000 multimodal query–content pairs covering 4 copyright material types × 5 copyright notice forms × 4 infringement tasks × multiple query variants.
    - Mechanism: 250 copyrighted sources (100 books + 50 news articles + 50 song lyrics + 50 code documentation files); each source is paired with 5 copyright notice forms (no notice, text "All Rights Reserved," text original notice, image "All Rights Reserved," image original notice); 4 infringement tasks (reproduction, extraction, paraphrasing, translation); 10 query variants per task.
    - Design Motivation: To capture the diversity of real-world copyrighted materials and varying presentations of copyright notices.

2. **Evaluation Metric Suite**:

    - Function: Assesses copyright compliance from two dimensions—content similarity and rejection rate.
    - Mechanism: ROUGE-L measures verbatim copying for reproduction/extraction tasks; BERTScore measures semantic similarity for paraphrasing; multilingual XLM-R embedding cosine similarity is used for translation; GPT-4 automatically determines whether the model refused the request.
    - Design Motivation: Different forms of infringement require different metrics—verbatim copying and semantic-level reproduction constitute distinct infringement modes.

3. **CopyGuard Defense Framework**:

    - Function: Detects copyright risk and provides compliance reminders before LVLM inference.
    - Mechanism: Four components operate in parallel—
        - **Copyright Notice Identifier**: Applies PaddleOCR to extract text from images and detects copyright indicators (©, Copyright, All Rights Reserved).
        - **Copyright Status Verifier**: When no copyright notice is detected, identifies the text source via the Google Search API and uses DeepSeek-R1 to verify current copyright status (e.g., whether the copyright has expired).
        - **Query Risk Analyzer**: Analyzes whether the user query may lead to infringement (e.g., "reproduce" is infringing; "summarize" is fair use) and suggests query reformulations.
        - **Copyright Status Reminder**: Delivers a clear copyright warning to the LVLM when an infringement risk is detected.
    - Design Motivation: Relying solely on fine-tuning causes over-refusal; the tool-augmented approach enables real-time copyright status queries and adapts to the dynamic nature of copyright law.

### Latency Optimization
The four CopyGuard components execute in parallel: OCR + search ($T_1$) and query risk analysis ($T_2$) proceed simultaneously, so the additional latency is only $\max(T_1, T_2)$. No overhead is incurred when no risk is detected.

## Key Experimental Results

### Main Results (Rejection Rate Without Copyright Notice)

| Model | Reproduction (%) | Extraction (%) | Paraphrasing (%) | Translation (%) | Average |
|-------|-----------------|----------------|------------------|-----------------|---------|
| GPT-4o | **90.65** | **52.78** | 7.84 | **95.36** | **61.66** |
| GPT-4o-mini | 69.31 | 2.10 | 5.66 | 11.82 | 22.22 |
| Gemini-2.0 | 0.09 | 1.25 | 0.04 | 1.06 | 0.61 |
| Claude-3.7 | 21.86 | 0.03 | 2.03 | 2.79 | 6.68 |
| Qwen2.5-VL-7B | 2.35 | 2.39 | 1.32 | 3.98 | 2.51 |

### CopyGuard Effectiveness

| Model | Original Rejection Rate | + CopyGuard Rejection Rate | Gain |
|-------|------------------------|---------------------------|------|
| GPT-4o-mini | 22.22% | **66.40%** | +44.18% |
| GPT-4o | 61.66% | **76.79%** | +15.13% |
| Gemini-2.0 | 0.61% | **61.61%** | +61.00% |
| Claude-3.7 | 6.68% | **63.66%** | +56.98% |
| LLaVA-1.5-7B | 5.08% | **79.76%** | +74.68% |
| Qwen2.5-VL-7B | 2.51% | **54.54%** | +52.03% |

### Key Findings
- **11 out of 12 LVLMs show near-zero rejection rates for infringement requests without copyright notices**: With the exception of GPT-4o, most models exhibit rejection rates below 10%, with Gemini-2.0 as low as 0.61%.
- **Copyright notices offer marginal improvement but are far from sufficient**: 9 out of 12 models show some improvement when copyright notices are added, yet most rejection rates remain below 5%; only GPT-4o, GPT-4o-mini, and Claude respond meaningfully.
- **Detailed copyright notices outperform generic "All Rights Reserved" notices**: Original notices containing author name and year produce lower ROUGE-L scores than generic notices.
- **CopyGuard produces zero false positives**: It issues no rejections for non-copyrighted content or non-infringing queries (e.g., summarization, commentary), and does not degrade model performance on standard benchmarks such as MMMU and MMBench.
- **Code documentation is the most vulnerable copyright material category**: Almost no model refuses infringement requests targeting code documentation.
- **Model architecture has a greater impact on copyright compliance than model scale**: Models of the same 7B parameter size (Qwen, DeepSeek, Janus) exhibit substantially different compliance behaviors.

## Highlights & Insights
- **First systematic study of LVLM copyright compliance**, bridging the gap between existing LLM copyright research and the multimodal setting.
- **The 50,000-pair benchmark** features a thorough cross-product design—5 copyright notice forms × 4 infringement types—providing comprehensive coverage.
- **CopyGuard's tool-augmented approach** avoids the over-refusal problem inherent to fine-tuning and can adapt to dynamic copyright status changes (e.g., expiring copyrights), which fine-tuning cannot address.
- **The discovery that LVLMs selectively ignore copyright notices in the visual modality** is a significant safety finding: models refuse when "©" appears in text but ignore the same symbol when embedded in an image.

## Limitations & Future Work
- CopyGuard relies on external tools (Google Search, DeepSeek-R1), introducing network latency and API costs.
- Copyright law varies by jurisdiction; the benchmark is primarily grounded in U.S. copyright law, limiting generalizability.
- The definition of fair use is simplified (e.g., "summarization = non-infringing; reproduction/translation = infringing"), whereas actual legal determinations are more nuanced.
- Only text-in-image content is evaluated; audio, video, and other multimodal copyright materials are not covered.
- CopyGuard's defense against paraphrasing is relatively weak (e.g., GPT-4o-mini improves only from 5.66% to 26.84%).

## Related Work & Insights
- **vs. CopyBench (LLM)**: CopyBench evaluates pure-text copyright compliance in LLMs; this work extends the scope to multimodal scenarios where copyrighted content is embedded in images, more closely reflecting real RAG usage.
- **vs. LlamaGuard**: LlamaGuard provides a general-purpose safety guardrail, while CopyGuard is a copyright-specialized guardrail; the two are complementary and can be used together.
- **Implications for RAG systems**: Multimodal RAG pipelines should incorporate copyright checking at the retrieval stage rather than deferring the decision to the LVLM generation phase.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic study of LVLM copyright compliance is a novel contribution, though the CopyGuard framework offers limited technical innovation (tool composition).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models, 50K dataset, 4 infringement scenarios, 5 copyright notice forms, and harmlessness validation on standard benchmarks—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, research design is rigorous, and the appendix is highly detailed.
- Value: ⭐⭐⭐⭐⭐ Copyright compliance is a hard legal requirement for AI deployment; both the benchmark and CopyGuard offer direct practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies](remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)
- [\[CVPR 2026\] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts](../../CVPR2026/multimodal_vlm/scipostgen_bridging_the_gap_between_scientific_papers_and_poster_layouts.md)
- [\[AAAI 2026\] SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](../../CVPR2026/multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)
- [\[ICML 2026\] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](../../ICML2026/multimodal_vlm/cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)

</div>

<!-- RELATED:END -->
