---
title: >-
  [Paper Note] Can Community Notes Replace Professional Fact-Checkers?
description: >-
  [ACL 2025][Social Computing][Community Notes] A large-scale analysis of 664k Twitter/X Community Notes reveals that their reliance on professional fact-checking is 5 times higher than previously reported ($\ge$5-7%). Content involving conspiracy theories/false narratives is twice as likely to cite fact-checking sources compared to other content, demonstrating that high-quality community moderation is deeply intertwined with and irreplaceable by professional fact-checking.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Community Notes"
  - "Fact-checking"
  - "Misinformation"
  - "Crowdsourced Moderation"
  - "Twitter/X"
date: 2026-05-08
content_hash: 1f7807817fee4d44
---

# Can Community Notes Replace Professional Fact-Checkers?

**Conference**: ACL 2025  
**arXiv**: [2502.14132](https://arxiv.org/abs/2502.14132)  
**Code**: None  
**Area**: NLP / Social Computing & Misinformation Governance  
**Keywords**: Community Notes, Fact-checking, Misinformation, Crowdsourced Moderation, Twitter/X

## TL;DR

A large-scale analysis of 664k Twitter/X Community Notes reveals that their reliance on professional fact-checking is 5 times higher than previously reported ($\ge$5-7%). Content involving conspiracy theories/false narratives is twice as likely to cite fact-checking sources compared to other content, demonstrating that high-quality community moderation is deeply intertwined with and irreplaceable by professional fact-checking.

## Background & Motivation

- **Task Definition**: Quantifying the reliance of Twitter/X Community Notes on the work of professional fact-checking organizations, and identifying the characteristics of posts and notes that rely on fact-checking sources.
- **Background**: Meta's 2025 announcement to end partnerships with fact-checking organizations in favor of a community moderation model implies that the two strategies are independent or even adversarial; Twitter/X has fully implemented Community Notes as its primary tool for misinformation governance since 2022.
- **Limitations of Prior Work**: Kangur et al. (2024) reported that only 1% of Community Notes cite fact-checking sources, but their list of fact-checking organizations was too small, and they classified fact-checking sections of news media (e.g., AP Fact Check) as "news", leading to a severe underestimation.
- **Core Problem**: (RQ1) To what extent do Community Notes rely on professional fact-checking? (RQ2) What types of posts and notes rely more on fact-checking sources?

## Method

### Overall Architecture

The authors downloaded all raw Community Notes data from Twitter/X from 2021.1 to 2025.1 (1.5 million notes), filtered by language (removing 526k non-English) $\rightarrow$ removed "not misleading" notes (268k) $\rightarrow$ removed ads/spam (44k), ultimately retaining 664k English notes. URLs in the notes were classified into 13 source categories. A subsample of 25.5k "helpful" notes was selected to fetch their corresponding post texts (denoted as $\mathcal{S}_\text{text}$), which was used for topic analysis and narrative/conspiracy theory annotation.

### Key Designs

**1. Five-step Cascade URL Source Classification Pipeline**

This addresses the issue where domain-name matching alone cannot capture fact-checking columns in news media. Classification is performed iteratively based on priority: ① Domain matching against a manually curated list of fact-checking organizations (Snopes, PolitiFact, AFP Fact Check, etc., 30+ in total); ② Searching for "fact-check" or its variants in the URL path (capturing paths like AP News' `/fact-checking/`); ③ Domain matching against the top-100 common domains manually annotated by the authors; ④ Using GPT-4o to classify remaining domains; ⑤ Marking as "unknown" if GPT-4 fails. Ultimately, 95% of URLs were successfully classified into 13 categories.

**2. Zero-Shot Topic Classification and Human Verification**

The ModernBERT-large-zeroshot model was applied to the $\mathcal{S}_\text{text}$ subset, taking the concatenated format "Tweet:\<post\>; Note:\<note\>" as input, to perform zero-shot classification into 13 topics (health, politics, technology, etc.). The authors' manual evaluation showed an accuracy of 90%, with the primary error being that AI-generated image content was misclassified under "technology".

**3. LLM-Driven Narrative and Conspiracy Theory Detection**

GPT-4o was used to determine whether an 8k balanced sample of \<post, note\> pairs involved broader false narratives or conspiracy theories. Two authors independently annotated 100 pairs for validation (agreement rate of 0.88, with discrepancies resolved through discussion), yielding a model F1 = 0.85. Additionally, the authors performed fine-grained manual annotation on 400 pairs to analyze rebuttal strategies (providing missing context, questioning sources, citing scientific evidence, etc.).

## Key Experimental Results

### RQ1: To what extent do Community Notes rely on professional fact-checking?

| Note Type | Proportion Citing Fact-Checking Sources | Remarks |
|:---------|:---:|:-----|
| All English Notes | ≥5% | Previously reported as only 1.2% (Kangur et al.) |
| Notes Rated "Helpful" | 7% | Fact-checking sources are positively correlated with high quality |
| Notes Rated "Not Helpful" | 1% | Low-quality notes rarely cite fact-checking |

- Compared to the 1.2% reported by Kangur et al. (2024), this study finds a rate up to **5 times** higher.
- Notes containing fact-checking sources scored significantly higher on the "HelpfulGoodSources" dimension of user ratings.
- Fact-checking citation rates are higher in high-risk topics (health, science, scams) and lower in technology and sports.

### RQ2: What is the relationship between content involving narratives/conspiracy theories and fact-checking?

| | Contains Fact-Checking Sources | Does Not Contain Fact-Checking Sources |
|:---|:---:|:---:|
| Involves Broader Narrative/Conspiracy | 22% | 11% |
| Does Not Involve | 28% | 39% |

- Content involving broader narratives/conspiracy theories is **twice** as likely to cite fact-checking sources compared to other content.
- Fine-grained annotation of 400 pairs further reveals differences in rebuttal strategies: complex narratives rely more on external fact-checking links, whereas misleading media content is refuted directly by providing counterexamples or missing context.
- Fact-checking sources are primarily used to question the credibility of a claim's source and to provide scientific evidence, and are rarely used to supplement missing context.

### Classification Distribution of Note Sources (Top-5 Categories)

| Source Category | Proportion in All Notes | Proportion in "Helpful" Notes |
|:---------|:---:|:---:|
| News | Highest | Highest |
| Social Media | High | High |
| Reference | Medium | Medium |
| Fact-Checking | ≥5% | 7% |
| Academic | Low | Low |

## Highlights & Insights

1. **Policy Responsiveness**: Directly using data to address Meta's decision to terminate fact-checking partnerships—demonstrating that community moderation and professional fact-checking exist in a symbiotic relationship rather than a substitutional one, where weakening fact-checking will cascade into reducing the quality of community notes.
2. **Methodological Improvements**: The five-step cascade classification pipeline identifies 5 times more fact-checking citations than simple domain matching, revealing a systematic underestimation in previous studies.
3. **Symbiotic Mechanism**: Professional fact-checkers conduct in-depth investigative research $\rightarrow$ Community Notes cite and disseminate these research findings $\rightarrow$ forming a closed loop in the information governance ecosystem.
4. **Partisan Dilemma**: Only 11% of Community Notes achieve a "helpful" status (requiring cross-ideological consensus), taking an average of 15.5 hours, with particularly low efficiency on partisan issues.

## Limitations & Future Work

1. The analysis is restricted to English notes (excluding over 500k non-English notes), which may bias findings toward Anglosphere public discourse.
2. The original tweet text is unavailable for most notes (only the $\mathcal{S}_\text{text}$ subset has corresponding post texts), limiting deep analysis.
3. The scale of manual annotation is limited (400 pairs for fine-grained annotation, 100 pairs for validation set), which could be expanded via crowdsourcing in the future.
4. The professional backgrounds of the Community Notes contributors were not differentiated—some might be professional fact-checkers themselves.
5. The criteria for determining conspiracy theories are based on a Western scientific perspective, potentially introducing cultural bias.

## Related Work & Insights

- **Community Notes Analysis**: Pröllochs (2022) analyzed the relationship between source credibility and "helpful" ratings; this study deepens the quantitative analysis from the perspective of fact-checking.
- **Fact-checking Ecosystem**: Graves & Anderson (2020) studied collaboration models between platforms and fact-checking organizations; this study supplements empirical evidence from the user perspective.
- **Crowdsourced Verification**: Martel et al. (2024) demonstrated that crowdsourcing is effective at identifying misinformation; Zhao & Naaman (2023) found that lay verifiers tend to refer to professional fact-checking in specialized domains such as medicine.
- **Insights**: Community-driven knowledge verification systems (e.g., academic peer review, Wiki editing) might share a similar implicit reliance on professional auditing.

## Rating

- **Novelty**: ★★★★☆ — First study to systematically quantify the reliance of Community Notes on fact-checking, revealing a 5-fold underestimation.
- **Technical Depth**: ★★★☆☆ — Primarily relies on statistical analysis and LLM annotation, without proposing new models or algorithms.
- **Experimental Thoroughness**: ★★★★☆ — Large-scale data of 664k notes + human validation + multi-angle, multi-grained analysis.
- **Practicality**: ★★★★★ — Offers direct reference value for misinformation governance policies on social media platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](../../ACL2026/social_computing/beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/social_computing/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2025\] How does Misinformation Affect Large Language Model Behaviors and Preferences?](how_does_misinformation_affect_large_language.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)

</div>

<!-- RELATED:END -->
