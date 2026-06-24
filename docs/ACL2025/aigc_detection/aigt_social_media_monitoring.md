---
title: >-
  [Paper Note] Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media
description: >-
  [ACL 2025][AIGC Detection][AI-Generated Text] This work offers the first large-scale quantification of the changing proportion of AI-Generated Text (AIGT) on social media. By collecting 2.4 million posts across Medium, Quora, and Reddit, constructing the AIGTBench dataset, and training the optimal detector OSM-Det, the study reveals that the AIGT proportion on Medium and Quora zoomed from ~2% to ~37-39% between 2022 and 2024, whereas Reddit's proportion only increased from 1.…
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "AI-Generated Text"
  - "Social Media Monitoring"
  - "Text Detection"
  - "Longitudinal Analysis"
  - "Platform Differences"
date: 2026-05-08
content_hash: 077048059f66d9ab
---

# Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media

**Conference**: ACL 2025  
**arXiv**: [2412.18148](https://arxiv.org/abs/2412.18148)  
**Code**: [https://github.com/TrustAIRLab/AIGT_on_Social_Media](https://github.com/TrustAIRLab/AIGT_on_Social_Media)  
**Area**: AIGC Detection  
**Keywords**: AI-Generated Text, Social Media Monitoring, Text Detection, Longitudinal Analysis, Platform Differences  

## TL;DR
This work offers the first large-scale quantification of the changing proportion of AI-Generated Text (AIGT) on social media. By collecting 2.4 million posts across Medium, Quora, and Reddit, constructing the AIGTBench dataset, and training the optimal detector OSM-Det, the study reveals that the AIGT proportion on Medium and Quora zoomed from ~2% to ~37-39% between 2022 and 2024, whereas Reddit's proportion only increased from 1.3% to 2.5%.

## Background & Motivation
**Background**: The rapid advancement of LLMs has made AI-generated text increasingly prevalent on social media. AIGT could potentially be utilized to spread misinformation and manipulate public opinion.

**Limitations of Prior Work**: (a) It remains unclear how much AIGT actually exists on social media due to a lack of systematic quantitative research; (b) the penetration rate of AIGT might vary significantly across different platforms but remains untracked; (c) existing detection datasets lack diversity and fail to provide benchmarks tailored to social media text.

**Key Challenge**: The growth rate of AIGT may drastically outpace the improvement of detection capabilities, yet there is a lack of empirical data to support this hypothesis.

**Goal**: To establish a quantitative monitoring framework for AIGT on social media by constructing detectors and datasets to track the longitudinal changes of AIGT.

**Key Insight**: Leveraging large-scale data collection (2.4M posts) + a multi-source detection benchmark (12 LLMs) + longitudinal tracking (2022-2024).

**Core Idea**: While approximately one-third of the content on Medium and Quora is already AI-generated, Reddit exhibits a remarkably slow growth, indicating that platform culture significantly influences AIGT penetration.

## Method

### Overall Architecture
(1) Collection of the SM-D dataset containing 2.4 million posts from Medium, Quora, and Reddit (from Jan 2022 to Oct 2024); (2) Construction of AIGTBench, which combines public datasets with paired AIGT data generated from social media texts using 12 LLMs; (3) Training of the optimal detector OSM-Det; (4) Longitudinal analysis performed on SM-D using OSM-Det.

### Key Designs

1. **AIGTBench Benchmark**:

    - **Function**: Provides training and evaluation benchmarks for social media AIGT detection.
    - **Mechanism**: Collects human-written text on social media and uses 12 different LLMs to generate corresponding AI texts, forming paired data.
    - The 12 LLMs cover various scales and architectures, such as GPT-4, Claude, Llama, and Mistral.
    - **Design Motivation**: Detectors trained on a single LLM exhibit poor generalization, necessitating multi-source training.

2. **OSM-Det Detector**:

    - **Function**: Selects/trains the optimal social media AIGT detector from multiple existing detectors.
    - **Mechanism**: Evaluates various detection methods (statistical, pre-trained, fine-tuned) on AIGTBench and selects the best solution.
    - **Design Motivation**: Performance of different detectors varies significantly across scenarios, requiring a systematic comparison.

3. **AI Attribution Rate (AAR) Longitudinal Tracking**:

    - **Function**: Quantifies the monthly proportion change of AIGT on each platform.
    - **Mechanism**: Uses OSM-Det to detect monthly posts in SM-D and calculates the proportion of AIGT.
    - **Key Findings**: Medium (1.77% → 37.03%), Quora (2.06% → 38.95%), Reddit (1.31% → 2.45%).

### Loss & Training
- OSM-Det is fine-tuned based on RoBERTa with a standard binary classification objective.
- The evaluation includes a systematic comparison of multiple detection methods.

## Key Experimental Results

### Main Results (Longitudinal changes in AAR, 2022.1-2024.10)

| Platform | 2022.1 AAR | 2024.10 AAR | Growth Multiplier |
|------|-----------|------------|---------|
| Medium | 1.77% | **37.03%** | 20.9x |
| Quora | 2.06% | **38.95%** | 18.9x |
| Reddit | 1.31% | **2.45%** | 1.9x |

### Analytical Dimensions

| Dimension | Differences between AIGT and Human Text |
|------|-------------------|
| Linguistic Patterns | AIGT is more formal, longer, and lexically more diverse |
| Topic Distribution | AIGT is clustered in technology, science, and business topics |
| Engagement Level | AIGT posts receive lower engagement (likes/comments) |
| Author Follower Distribution | AIGT authors tend to have fewer followers |

### Key Findings
- Medium and Quora exhibit massive growth in AIGT, with about 1/3 of the content already being AI-generated. Reddit's growth is significantly slower, likely because Reddit's community culture places a higher value on originality and discussion.
- AIGT authors typically have fewer followers, suggesting that AI text is used for rapid content scaling rather than by established influencers.
- AIGT posts receive lower engagement, indicating that users may implicitly recognize characteristics of AI-generated content and interact less.
- The launch of ChatGPT (Nov 2022) accelerated the growth of AAR, demonstrating a direct causal relationship.

## Highlights & Insights
- **First quantitative evidence of social media AIGT**—the finding that "approx. 1/3 of content is already AI" holds substantial citation and policy-making value.
- **Platform differences** reflect how community culture moderates AIGT penetration: anonymous/long-form platforms (Medium/Quora) are more susceptible to penetration, whereas community-driven platforms (Reddit) possess natural resilience.
- AIGTBench serves as a valuable detection benchmark, where the diversity of 12 LLMs ensures the detector's generalization capability.
- The finding that AIGT receives lower engagement suggests that platform algorithms might already implicitly de-prioritize AI-generated content.
- The topic distribution preference of AIGT (primarily technology and science) provides strategic insights for content moderation.

## Limitations & Future Work
- The accuracy of OSM-Det cannot be 100%—false positive and false negative rates during large-scale applications affect AAR estimation.
- Only three English platforms are covered—major platforms such as X/Twitter and YouTube are not included.
- Changes in AAR may be partially influenced by shifts in platform policies (e.g., Medium's moderation policies).
- The study does not differentiate between different usage intentions of AIGT (e.g., AI-assisted writing vs. SPAM).

## Related Work & Insights
- **vs MultiSocial**: MultiSocial constructs detection benchmarks, while this paper goes a step further by conducting longitudinal quantitative monitoring.
- **vs Perez et al.**: While they study information collapse in iterative generation, this paper focuses on the propagation of AIGT over social media.
- This work establishes an empirical foundation for platform content governance and AI text regulation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Offers the first large-scale longitudinal quantification of AIGT on social media, with highly citable empirical data.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 2.4M posts + 12 LLMs + 3 platforms + longitudinal tracking + multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear data presentation with impactful findings.
- Value: ⭐⭐⭐⭐⭐ Holds significant policy implications for AI governance and content regulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](multisocial_mgt_detection.md)
- [\[ICLR 2026\] EditLens: Quantifying the Extent of AI Editing in Text](../../ICLR2026/aigc_detection/editlens_quantifying_the_extent_of_ai_editing_in_text.md)
- [\[ACL 2025\] An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)

</div>

<!-- RELATED:END -->
