---
title: >-
  [Paper Note] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation
description: >-
  [ACL 2026][Social Computing][Short-video rumors] This paper constructs a high-quality evaluation set of 200 Chinese health-related short-video rumors. It systematically evaluates 8 cutting-edge MLLMs using evidence chain…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Short-video rumors"
  - "MLLM"
  - "Chinese health information"
  - "cognitive biases"
  - "authority bias"
date: 2026-05-08
content_hash: 7098dee9cde7414a
---

# Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation

**Conference**: ACL 2026  
**arXiv**: [2601.06600](https://arxiv.org/abs/2601.06600)  
**Code**: [GitHub](https://github.com/penguinnnnn/Fine-VDK)  
**Area**: Multimodal VLM / Video Rumor Detection / Model Cognitive Bias Evaluation  
**Keywords**: Short-video rumors, MLLM, Chinese health information, cognitive biases, authority bias

## TL;DR
This paper constructs a high-quality evaluation set of 200 Chinese health-related short-video rumors. It systematically evaluates 8 cutting-edge MLLMs using evidence chains, error types, and social cues. The study finds that Gemini-2.5-Pro is the most stable, while most models remain influenced by label bias, authority accounts, and traffic metrics during multimodal rumor judgment.

## Background & Motivation

**Background**: Short-video platforms has become the primary channel for the dissemination of health, food, and lifestyle information. Video rumors are typically not just textual; they construct credibility through experimental demonstrations, subtitles, oral broadcasting, account certification, and like counts.

**Limitations of Prior Work**: Existing misinformation benchmarks are mostly concentrated in narrow fields such as news, image-text, or COVID-19, and often rely on external fact-checking. It is difficult to evaluate whether a model can discover errors from internal experimental logic, argumentation chains, and visual evidence within the video. Many datasets also lack fine-grained annotations for error causes.

**Key Challenge**: MLLMs are strong on general video understanding benchmarks, but short-video rumors resemble "real-world reasoning problems with social signals." Models must both understand multimodal content and resist human-like heuristic biases, such as perceiving high play counts as more credible or official accounts as more authoritative.

**Goal**: To establish a fine-grained evaluation framework for Chinese short-video rumors, addressing three questions: which modality is most important; whether model reasoning covers human-annotated error causes; and whether models exhibit bandwagon effects and authority biases.

**Key Insight**: The authors start from professional debunking cases, trace back to the original Douyin / Kuaishou videos, and provide error types and reasons for each false video using national standards, academic papers, legal documents, and common sense/Wikipedia as evidence.

**Core Idea**: To elevate short-video rumor evaluation from "binary truth-false classification" to a joint diagnosis of "multimodal content + evidence chain + social bias," observing whether the model truly understands the error rather than making judgments based solely on label tendencies.

## Method

### Overall Architecture
The paper constructs the Fine-VDK dataset and designs five input settings and two types of cognitive bias analysis around it. The dataset contains 200 short videos (100 false and 100 real), covering four public health-related domains. Each video is decomposed into visual frames, OCR text, and ASR oral text, while retaining social metadata such as titles, channel IDs, and likes/shares. During evaluation, the model provides a credibility score from $1$ to $7$ using a CoT prompt. The authors then convert these scores into a Belief Score, measuring the model's ability to doubt false content and believe real content respectively.

### Key Designs

1.  **High-Quality Short-Video Rumor Data Construction**:
    -   Function: Provides a compact, high-quality evaluation set of Chinese short-video rumors with evidence chains.
    -   Mechanism: 100 false cases are first collected from professional debunking channels, followed by retrieving original rumor videos. The real set is extracted from verified real statements on the China Internet Joint Rumor-Refuting Platform, matched with videos of similar themes and visual styles. False samples are labeled into three categories: experimental errors, logical fallacies, and fictional claims.
    -   Design Motivation: Automatically crawled or weakly labeled data makes it difficult to judge "why it is wrong." The authors sacrifice scale for annotation quality to enable model evaluation to delve into reasoning paths.

2.  **Multimodal Inputs and Belief Score Metric**:
    -   Function: Dissects the contribution of visual, subtitle, audio, and human statements to the model's judgment.
    -   Mechanism: The five settings are Claim, Textual, Aural, Visual, and Multimodal. Claim uses human-distilled core statements; Textual uses OCR; Aural uses ASR; Visual uses $0.5$ FPS sampling with a maximum of $32$ frames; Multimodal uses both frames and transcripts. The model outputs a $1-7$ Likert credibility score. The Belief Score only rewards scores that correctly reflect "doubt toward fake content" or "belief toward true content."
    -   Design Motivation: In short videos, subtitles, visual experiments, and oral broadcasts are often redundant yet mutually interfering; modality-specific testing reveals which type of information the model relies on.

3.  **Bandwagon Effect and Authority Bias Analysis**:
    -   Function: Evaluates whether the model is influenced by non-content cues in its authenticity judgment.
    -   Mechanism: For the bandwagon effect experiment, metrics such as plays, likes, comments, and shares are scaled by order of magnitude and added to the prompt. For the authority bias experiment, channel IDs and certification statuses are revealed, including unverified, Yellow V (individual), Blue V (enterprise), and Red V (organization).
    -   Design Motivation: Rumor propagation on real-world platforms is not just a content issue. If models are influenced by "many people have seen it" or "the account looks authoritative" like humans, they will amplify platform biases in practical applications.

### Loss & Training
This paper is an evaluation framework rather than training a new model. The core metric is the normalized Belief Score: for false videos, points are awarded only when the model score is above the neutral point; for real videos, points are awarded only when the score is below the neutral point; otherwise, it is recorded as $0$. This design simultaneously punishes "believe all" and "disbelieve all" label biases.

## Key Experimental Results

### Main Results
The All Belief Scores of 8 MLLMs under five input settings are as follows. Claim serves as an upper-bound reference after human extraction of core statements and does not represent the model's ability to process original videos.

| Model | Claim | Textual | Aural | Visual | Multimodal |
|-------|-------|---------|-------|--------|------------|
| GPT-4o-20241120 | 67.5 | 57.3 | 52.0 | 47.3 | 59.3 |
| o3-20250416 | 55.2 | 34.0 | 23.0 | 40.7 | 35.2 |
| Gemini-2.5-Flash | 74.8 | 56.7 | 45.2 | 67.5 | 64.5 |
| Gemini-2.5-Pro | 79.0 | 74.0 | 56.3 | 79.7 | 71.5 |
| Qwen-VL-Max | 55.7 | 55.5 | 55.2 | 60.3 | 50.8 |
| Qwen-2.5-VL-72B | 53.8 | 53.0 | 38.3 | 44.5 | 52.2 |
| Claude-Sonnet-4 | 62.0 | 48.3 | 44.3 | 31.3 | 47.5 |
| Seed-1.6-Thinking | 74.5 | 60.8 | 53.0 | 71.0 | 65.5 |
| Average | 65.3 | 55.0 | 45.9 | 55.3 | 55.8 |

### Ablation Study

| Analysis Object | Key Figure | Conclusion |
|----------|----------|------|
| Domain Difficulty | Avg Multimodal BS: Agriculture 41.5, Food Processing 59.1, Chem/Tools 46.8, Health/Medical 66.5 | Health common sense is easiest; commodity/material updates are harder |
| Error Type | Logical Fallacy Multimodal BS is 45.9, lower than Experimental Error (51.8) and Fictional Claim (53.0) | Models have the most difficulty capturing breaks in the argumentation chain |
| Certification Status | Avg BS for false subset with channel ID: Unverified 73.0, Org-certified 36.8 | Authority accounts significantly reduce model suspicion of false content |
| ID Reliability Judgment | GPT-4o perceived 2.25% of false-set IDs as reliable vs 28.8% for true-set; Qwen-2.5 was 1.12% vs 19.2% | Models implicitly judge account credibility first, which then influences truth judgment |

### Key Findings
- Gemini-2.5-Pro is the most stable model, reaching an All BS of 71.5 under Multimodal, but not all models benefit from multimodal input.
- The Aural setting averaged only 45.9, approximately 10 points lower than Textual, Visual, and Multimodal, indicating that oral text noise and information gaps significantly affect models.
- The Qwen series tends to trust videos, scoring high on the real video subset and low on the false subset; o3 exhibits excessive conservatism, being reluctant to give high trust even to real videos.
- Although multimodal input does not necessarily yield the highest truth-false judgment scores, it improves CoT coverage of human-annotated error causes, suggesting that video context aids explanation quality.
- High traffic metrics did not simply induce models to believe fake videos but rather strengthened the model's original judgment confidence; authority accounts influenced authenticity judgments more directly.

## Highlights & Insights
- Although the dataset size is only 200, the annotation density is very high: every false video has an error reason, evidence type, and error pattern, which is more suitable for diagnosing model reasoning than large-scale weakly labeled video sets.
- The Belief Score design exposes label bias. When looking only at accuracy, a model that always says "true" might perform well on the real subset; BS separates the true and false sides for observation.
- The paper introduces the concept of cognitive bias into MLLM rumor evaluation, which is of great practical significance. In short-video scenarios, accounts, likes, and shares are part of the content; models cannot be evaluated only on decontextualized data.
- The phenomenon that "Claim is best, but Multimodal explanation is better" suggests that human extraction of statements reduces noise but also strips away the real-world complexity that models need to handle.

## Limitations & Future Work
- The data originates from the Simplified Chinese short-video ecosystem. Many rumors are related to culture, platform mechanisms, and lifestyle habits; cross-linguistic and cross-platform generalization requires re-verification.
- The sample size is small. Although the authors performed significance analysis, statistical power is limited when subdivided into domain $\times$ error type $\times$ truth/false cross-cells.
- Evaluation depends on current cutting-edge models and prompt templates; future model versions, video input interfaces, and CoT strategies may affect the conclusions.
- The dataset primarily targets health and lifestyle rumors, with insufficient coverage of high-risk information dissemination scenarios such as politics, finance, and disasters.
- Experiments on channel IDs and traffic metrics are controlled analyses; complex social contexts in real recommendation systems, such as ranking, comment sections, and follower profiles, have not yet been included.

## Related Work & Insights
- **vs FakeSV / FMNV**: These datasets lean more toward news or broad fake-video detection; this paper emphasizes Chinese short-video health rumors, error types, and evidence chains.
- **vs Textual Rumor Detection**: Textual tasks usually only require judging the relationship between statements and evidence; this paper also handles video demonstrations, subtitles, oral broadcasts, and social cues.
- **vs General MLLM benchmarks**: MMMU, MMBench, etc., evaluate multimodal understanding capabilities; this paper evaluates the robustness of model factual judgment under real-world noise and cognitive biases.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines short-video rumors, fine-grained evidence chains, and cognitive biases; the problem definition is highly realistic.
- Experimental Thoroughness: ⭐⭐⭐⭐ Integration of modality, domain, error type, and social cue analysis is complete, though the sample scale is small.
- Writing Quality: ⭐⭐⭐⭐ Data construction and findings are clearly explained; tables are numerous but the main line is clear.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for evaluating the factual judgment and platform-context robustness of Chinese multimodal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[ICML 2026\] Self-Debias: Self-correcting for Debiasing Large Language Models](../../ICML2026/social_computing/self-debias_self-correcting_for_debiasing_large_language_models.md)

</div>

<!-- RELATED:END -->
