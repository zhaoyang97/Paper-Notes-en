---
title: >-
  [Paper Note] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus
description: >-
  [ACL 2025][Podcast Dataset] This study builds and releases SPoRC, a large-scale dataset containing transcripts of 1.1 million podcast episodes (complete with metadata, inferred speaker roles, and acoustic features for 370k episodes). Through topic analysis, guest co-occurrence network analysis, and responsiveness analysis during the George Floyd protests, it provides the first comprehensive characterization of the content, structure, and responsiveness of the podcast ecosyste…
tags:
  - "ACL 2025"
  - "Podcast Dataset"
  - "Speech Transcription"
  - "Topic Modeling"
  - "Social Network Analysis"
  - "Media Ecosystem"
  - "Collective Attention"
date: 2026-05-08
content_hash: 8e8c49cac73d6682
---

# Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus

**Conference**: ACL 2025  
**arXiv**: [2411.07892](https://arxiv.org/abs/2411.07892)  
**Code**: [https://github.com/blitt2018/SPoRC_data](https://github.com/blitt2018/SPoRC_data)  
**Data**: [https://huggingface.co/datasets/blitt/SPoRC](https://huggingface.co/datasets/blitt/SPoRC)
**Authors**: Benjamin Litterer, David Jurgens, Dallas Card
**Affiliation**: University of Michigan
**Area**: Misc / Computational Social Science / NLP Resources  
**Keywords**: Podcast Dataset, Speech Transcription, Topic Modeling, Social Network Analysis, Media Ecosystem, Collective Attention

## TL;DR

This study builds and releases SPoRC, a large-scale dataset containing transcripts of 1.1 million podcast episodes (complete with metadata, inferred speaker roles, and acoustic features for 370k episodes). Through topic analysis, guest co-occurrence network analysis, and responsiveness analysis during the George Floyd protests, it provides the first comprehensive characterization of the content, structure, and responsiveness of the podcast ecosystem.

## Background & Motivation

**Background**: Podcasts have become a vital part of the modern media landscape—in 2023, 42% of the US population over age 12 reported listening to a podcast in the past month. Podcasting covers topics such as education, news, true crime, comedy, etc., exerting real-world influence on listener behavior (60% changed their media consumption habits, 36% changed their lifestyle, and 28% purchased products).

**Limitations of Prior Work**:
   - The largest existing research corpus is the 200k episode dataset released by Spotify, which has been deprecated and is no longer accessible.
   - PodcastRE focuses primarily on archival preservation and does not widely distribute data.
   - The Brookings Institution's PPPD only covers about 120 political podcasts.
   - There is a lack of **comprehensive, multimodal, large-scale open podcast datasets with structured metadata**.

**Core Motivation**: To establish a large-scale public research dataset similar to those in domains like Twitter, Reddit, and Amazon, covering both textual and acoustic dimensions, enhancing metadata, and supporting interdisciplinary research spanning communication, computational social science, and NLP.

## Method

### Dataset Construction Pipeline

#### 1. Initial Data Collection

- **Data Source**: Podcast Index (a public database containing information on over 4 million podcasts)
- **Time Period**: May–June 2020 (selected to capture a critical historical window providing rich context, including major events like the George Floyd protests and COVID-19)
- **Scale**: Identified episodes released during this period by 273k English podcasts, successfully downloading 1.3 million episodes of audio and metadata across 247k podcasts

#### 2. Speech Transcription

- Automatic speech recognition is performed using **Whisper (whisper-base.en)**
- Primary error source: Phrase repetitions during music, silence, or non-English speech (not hallucinations), which are removed using an n-gram filter
- Validation against professional transcripts: Word Error Rate (WER) < 10% (with over half of the errors arising from disfluency edits in professional transcription)
- After filtering, **1.1 million transcribed episodes containing 6.6 billion words** are retained

#### 3. Prosodic Feature Extraction

The openSMILE toolkit is used to extract the following acoustic features:
- **Fundamental Frequency (F0)**: Related to pitch, used to signal emphasis and questioning
- **First Formant (F1)**: Related to vowel articulation, describing variations in dialect, gender, and age
- **MFCC 1-4**: Mel-Frequency Cepstral Coefficients, capturing the short-term power spectrum, useful for downstream tasks such as emotion recognition
- High-frequency features are aggregated to the token level (averaged over the duration of each word)

#### 4. Speaker Identification

Three-stage pipeline:

**Step 1: Speaker Diarization**
- Audio is segmented into speaker turns using pyannote
- Diarization is applied to 370k episodes (due to computational resource constraints)
- Validation shows a diarization error rate of only 2.1%
- After filtering out speakers with < 5% talk time: 37% are single-speaker, and 39% are two-speaker episodes

**Step 2: Host/Guest Name Identification**
- Persons' names are extracted from the first 350 words of transcripts using spaCy Named Entity Recognition (NER)
- Crowdsourced human annotations for 2,000 entities (Host/Guest/Neither) are collected via Prolific, yielding a Krippendorff's $\alpha = 0.77$
- A RoBERTa classifier is fine-tuned: achieving 87% in cross-validation and 88% accuracy on the test set
- Identification results: 550k episodes (49%) have at least one identified Host or Guest

**Step 3: Voice-Name Matching**
- For diarized podcasts where exactly one Host is identified, the name is heuristically mapped to the voice that first utters the Host's name

### Dataset Summary

| Level | Contents |
|------|---------|
| Episode Level (1.1M episodes) | Transcripts, categories, durations, release dates, inferred hosts/guests |
| Turn Level (370k episodes) | Diarized transcripts, timestamps, acoustic features, speaker roles |

## Key Experimental Results

### Topic Analysis

- Topic modeling is performed using a **200-topic LDA** on the first 1,000 words of each episode
- Zero-shot classification of news categories is conducted using the IPTC NewsCode taxonomy
- **Category Cohesion**: Categories like Sports, Religion, and Business exhibit highly coherent internal topics (e.g., niche communities in baseball, wrestling, real estate, Bitcoin, Judaism, Islam, etc.)
- **Cross-Category Topics**: Topics like COVID-19, racial justice, and mind-body/self-improvement span across multiple category boundaries
- **Religion is the most voluminous podcast category** (composed heavily of recorded Christian sermons)

### Guest Co-occurrence Network Analysis

A podcast-guest bipartite graph is constructed and projected onto a one-mode network:
- 10,480 nodes (podcasts) and 26,589 edges (shared guest connections)

**Category Modularity** (measuring how tightly guests are shared within a category):

| Category | Modularity |
|------|---------|
| Sports | 0.155 |
| Business | 0.134 |
| News | 0.064 |
| Religion | 0.045 |
| Society | 0.013 |
| Education | 0.011 |

- Sports and Business form highly cohesive guest network communities
- Although Religion and Society are the largest categories, their guest networks are sparse—podcasts within these categories invite guests less frequently

### George Floyd Event Responsiveness Analysis

- **Rapid Response of Collective Attention**: Discussion of Floyd-related topics peaked roughly 10 days after the event, while BLM topics peaked with a lag of about 4 days
- **Slow Decay**: Consistently follows a "media storm" pattern but decays slower than news media (which typically peak around Day 3 and decay by Day 10)
- **Broad Penetration**: **21% of all podcasts** mentioned George Floyd by name in at least one episode
- **Cross-Category Discussion**: While all categories showed significant discussion during the peak period, categories that regularly cover these topics (e.g., Society) maintained a higher mention rate in later stages
- **News's Uniqueness**: News is the only category that paid sustained, substantial attention to "policing, law, protest" topics

## Highlights & Insights

1. **A Truly Large-Scale Podcast Corpus**: 1.1 million transcripts + 370k diarized and feature-extracted episodes, far exceeding prior resources (with Spotify's 200k corpus no longer available).
2. **Multimodal Data Fusion**: The combination of text transcripts, prosodic features, speaker diarization, and metadata supports research into *how* people communicate, rather than just *what* they say.
3. **Speaker Identification Pipeline**: A complete pipeline ranging from entity recognition to role classification and voice matching, successfully identifying participants in 49% of the episodes despite imperfect accuracy.
4. **Cross-Category Topic Discovery**: Highly impactful topics (e.g., COVID-19, racial justice) transcend category boundaries, revealing the potential for information to flow across communities in the podcast ecosystem.
5. **Media Ecosystem Comparisons**: The temporal dynamics of "media storms" in podcasts are similar to but slower than those in traditional news media, establishing a baseline for understanding information propagation in new media.

## Limitations & Future Work

1. **Limited Temporal Scope**: The data covers only May–June 2020, precluding long-term evolutionary analysis; the insights may not generalize to other historical periods.
2. **Incomplete Platform Coverage**: Excludes Spotify-exclusive podcasts (e.g., *The Joe Rogan Experience*) and podcasts not distributed via RSS feeds.
3. **Transcription Quality Limitations**: Whisper's transcription quality may degrade with non-standard accents, code-switching, or low-quality recordings; additionally, n-gram filtering may disproportionately affect marginalized speaker groups.
4. **Speaker Labeling Constraints**: Some hosts do not introduce themselves, and the name detection is limited to two-word names; furthermore, cascading errors exist in role labeling.
5. **Lack of Causal Inference**: The time-series analyses describe observational patterns but do not explain *why* these patterns emerge.

## Related Work & Insights

- **Podcast Research**: Media studies of podcasting by Berry (2006, 2015), podcast misinformation studies by Wirtschafter (2023), and listener surveys by the Pew Research Center (2023).
- **Large-scale Media Corpora**: Pushshift Reddit dataset (Baumgartner et al. 2020), Congressional Speeches (Gentzkow et al. 2019).
- **Spotify Podcast Corpus**: The 100k-episode dataset of Clifton et al. (2020), which is no longer maintained.
- **Podcast Guest Networks**: Contemporary work by DeMets & Spiro (2025) studying the guest networks of around 120 political podcasts in PPPD.
- **Media Storms**: News media storm theory by Boydstun et al. (2014), computational analysis by Litterer et al. (2023).
- **Topic Modeling**: The LDA model for topic discovery by Blei et al. (2003).

## Rating

⭐⭐⭐⭐ (4/5)

- **Dataset Value** ⭐⭐⭐⭐⭐: Fills a critical data gap in podcast research; its scale and multimodal coverage far exceed earlier resources.
- **Depth of Analysis** ⭐⭐⭐⭐: Fully maps out the podcast ecosystem through topic analysis, network analysis, and time-series analysis.
- **Methodology** ⭐⭐⭐: The processing pipeline is solid (Whisper + pyannote + RoBERTa), though individual steps rely on relatively standard methodologies.
- **Social Science Significance** ⭐⭐⭐⭐: Opens up brand-new possibilities for large-scale analysis in communication, information diffusion, and community studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] CoAM: Corpus of All-Type Multiword Expressions](coam_corpus_of_all-type_multiword_expressions.md)
- [\[ACL 2025\] Graph-Structured Trajectory Extraction from Travelogues](graph-structured_trajectory_extraction_from_travelogues.md)
- [\[ACL 2025\] DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)

</div>

<!-- RELATED:END -->
