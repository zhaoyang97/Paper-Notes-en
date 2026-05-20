---
title: >-
  [Paper Note] LiveWeb-IE: A Benchmark For Online Web Information Extraction
description: >-
  [ICLR 2026][Multimodal VLM][Web information extraction] This paper introduces LiveWeb-IE, the first benchmark for online web information extraction (WIE), covering multi-type data extraction including text, images…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Web information extraction"
  - "online evaluation"
  - "visual grounding"
  - "XPath generation"
  - "multimodal agent"
date: 2026-05-08
content_hash: 91836e626112bf0b
---

# LiveWeb-IE: A Benchmark For Online Web Information Extraction

**Conference**: ICLR 2026
**arXiv**: [2603.13773](https://arxiv.org/abs/2603.13773)  
**Code**: [GitHub](https://github.com/sbY99/LiveWeb-IE)  
**Area**: Multimodal VLM
**Keywords**: Web information extraction, online evaluation, visual grounding, XPath generation, multimodal agent

## TL;DR

This paper introduces LiveWeb-IE, the first benchmark for online web information extraction (WIE), covering multi-type data extraction including text, images, and hyperlinks. It further proposes the Visual Grounding Scraper (VGS) framework, which simulates human cognitive processes—visual scanning to locate regions → precise element localization → XPath generation—to achieve robust information extraction on dynamic webpages.

## Background & Motivation

Web information extraction (WIE) refers to the task of automatically extracting structured data from webpages. Existing WIE benchmarks (e.g., SWDE, WEIR, PLAtE) are all constructed from **static HTML snapshots**, which suffer from fundamental limitations:

**Temporal mismatch**: Webpage layouts and structures evolve continuously over time; static snapshots fail to reflect the current state of a webpage.

**Unreliable performance**: LLM-based wrapper methods exhibit an average F1 drop of over 15% on the same websites after structural changes.

**Limited data types**: Existing benchmarks focus exclusively on text extraction, neglecting the need for image and hyperlink extraction.

**Lack of complexity dimensions**: No systematic task complexity stratification exists.

Furthermore, existing WIE methods rely excessively on HTML parsing. As webpage structures grow increasingly complex, the redundancy of HTML makes accurate information localization progressively more difficult.

## Method

### Overall Architecture

This paper makes two contributions: (1) the **LiveWeb-IE benchmark**, an online WIE evaluation benchmark; and (2) the **VGS method** (Visual Grounding Scraper), a multi-stage agent framework that simulates human cognitive processes.

### Key Designs

#### 1. LiveWeb-IE Benchmark Design

**Four core properties**:
- **Online evaluation**: WIE systems are required to directly access target URLs at evaluation time, processing the live DOM structure of the webpage.
- **Diverse and authorized websites**: 15 licensed websites spanning 8 domains, with robots.txt compliance, terms-of-service review, and direct administrator authorization.
- **Multi-type data extraction**: Covers text, image, and hyperlink extraction.
- **Multi-dimensional task complexity**: Four complexity levels defined based on the number of attributes and value cardinality.

**Four task types**:
- **Type I**: Single attribute, single value (e.g., "What is this professor's email address?")
- **Type II**: Multiple attributes, single value (e.g., "What are this player's height and weight?")
- **Type III**: Single attribute, list value (e.g., "All paper titles on this page")
- **Type IV**: Multiple attributes, list value (e.g., "Names and prices of all products")

**Data construction pipeline**: Website selection → page grouping (layout-based clustering) → data annotation → manual cross-validation. The final benchmark contains 342 queries, 97 unique attributes, and 46 page groups.

**Content stability design**: Queries target factual information (e.g., the score of the 2022 World Cup final) whose answers remain invariant even as webpage layouts change, ensuring the benchmark's long-term validity.

#### 2. VGS (Visual Grounding Scraper) Framework

VGS simulates the human cognitive process of locating information on a webpage through four stages that progressively narrow the observation space:

**Stage 1: Attribute identification.** An LLM decomposes the natural language query into a structured set of target attributes:
$$\hat{\mathcal{A}} = \text{LLM}(I_a, Q)$$

**Stage 2: Visual grounding.** The webpage is rendered as a sequence of vertical regions (fixed-size screenshots). For each attribute, a VLM locates the relevant region within the region sequence:
$$r'_i = \text{VLM}(I_g, \mathcal{R}, \hat{a}_i)$$
The key value lies in **substantially reducing the observation space**.

**Stage 3: Precise element localization.** The target value is precisely located within the identified region via a two-step strategy:
1. Candidate bounding boxes are generated (via VLM scanning for text attributes; via HTML tag localization for non-text attributes).
2. Set-of-Mark Prompting overlays numbered markers, allowing the VLM to select the correct element subset:
$$\mathcal{B}_i^* = \text{VLM}(I_p, r_i^*, \hat{a}_i)$$

**Stage 4: XPath synthesis.** DOM elements corresponding to the precisely localized bounding boxes are identified; a local HTML snippet (within proximity distance $d$) is extracted; the VLM then combines visual and structural information to generate a reusable XPath:
$$x_i = \text{VLM}(I_x, \mathcal{H}_i, \hat{r}_i, \hat{a}_i)$$
The resulting XPath set constitutes a reusable wrapper.

### Loss & Training

VGS is a **training-free** agent framework that relies entirely on the inference capabilities of pretrained LLMs/VLMs. Evaluation metrics include Precision, Recall, and F1.

## Key Experimental Results

### Main Results

**Overall F1 comparison on LiveWeb-IE**:

| Backbone | Method | Type I F1 | Type II F1 | Type III F1 | Type IV F1 | Overall F1 |
|----------|--------|-----------|------------|-------------|------------|------------|
| GPT-4o | COT | 47.54 | 40.84 | 8.15 | 7.24 | 24.60 |
| GPT-4o | AutoScraper | 55.22 | 42.65 | 9.10 | 6.92 | 26.76 |
| GPT-4o | **VGS** | **65.87** | **46.35** | **45.38** | **41.50** | **48.58** |
| Gemini-2.5-Flash | **VGS** | **49.02** | **44.82** | **42.92** | **38.13** | **43.44** |

**Open-source model comparison (Overall F1)**:

| Backbone | COT | AutoScraper | VGS |
|----------|-----|-------------|-----|
| Qwen-2.5-7B | 11.67 | 16.04 | **21.74** |
| Qwen-2.5-32B | 17.74 | 21.61 | **35.05** |
| Gemma-3-27B | 16.65 | 19.04 | **30.79** |

### Ablation Study

Contributions of individual VGS stages:
1. **Removing visual grounding**: Directly performing element localization without prior region identification leads to significant performance degradation.
2. **Removing precise element localization**: Skipping the Set-of-Mark step causes marked deterioration on complex task types.
3. **Replacing visual information with HTML**: F1 drops substantially for Type III and Type IV tasks.

### Key Findings

1. **Static-to-online performance gap**: LLM-based methods exhibit an average F1 drop exceeding 15% after structural evolution, confirming the necessity of online evaluation.
2. **Large complexity gap**: VGS yields the greatest advantage on complex task types—GPT-4o+VGS achieves a Type III F1 of 45.38%, compared to only 8.15% for COT.
3. **Critical role of visual information**: Pure HTML-based methods fail on complex webpages; VGS bypasses HTML noise through visual grounding.
4. **Open-source vs. closed-source gap**: Even with VGS, Qwen-2.5-32B (35.05%) trails GPT-4o (48.58%) by a significant margin.
5. **Wrapper reusability**: XPaths generated by VGS generalize across pages of the same type.

## Highlights & Insights

- **Novel problem formulation**: This work is the first to shift WIE evaluation from offline to online settings, addressing annotation longevity through content stability design.
- **Cognition-inspired design**: The four-stage VGS pipeline faithfully mirrors the human process of locating information on a webpage.
- **Dual-channel visual and structural fusion**: XPath generation elegantly combines visual grounding results with local HTML context.
- **Multi-type data coverage**: Incorporating images and hyperlinks into WIE evaluation aligns with real-world requirements.

## Limitations & Future Work

1. Limited benchmark scale: Only 15 websites and 342 queries; large-scale expansion would be valuable.
2. Content stability assumption: Some websites may undergo redesigns that render URLs inaccessible, requiring periodic maintenance.
3. High VLM inference cost: Each of the four stages requires VLM inference, posing efficiency challenges for large-scale extraction.
4. XPath fragility: Generated XPaths remain dependent on DOM structure and may break after major website redesigns.
5. Insufficient handling of dynamic content: Processing of JavaScript-rendered dynamic content is not thoroughly addressed.

## Related Work & Insights

LiveWeb-IE differs in objective from web agent benchmarks such as WebArena—the latter focuses on multi-step task completion, whereas LiveWeb-IE targets precise information extraction from individual pages. The visual grounding approach in VGS, combined with Set-of-Mark Prompting, demonstrates the potential of VLMs for webpage understanding and can be extended to applications such as automated webpage testing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Online WIE benchmarking is a novel and practically valuable contribution.
- **Technical Quality**: ⭐⭐⭐⭐ — VGS is well-designed, though its technical innovations lean toward engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-backbone comparisons are comprehensive, though ablations could be more systematic.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly addresses real-world web data collection scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — The benchmark design motivation and method pipeline are clearly articulated.
- **Overall**: ⭐⭐⭐⭐ (8.0/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](../../AAAI2026/multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[NeurIPS 2025\] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture](../../NeurIPS2025/multimodal_vlm/mirage_a_benchmark_for_multimodal_information-seeking_and_reasoning_in_agricultu.md)
- [\[ICLR 2026\] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)
- [\[ICLR 2026\] Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)

</div>

<!-- RELATED:END -->
