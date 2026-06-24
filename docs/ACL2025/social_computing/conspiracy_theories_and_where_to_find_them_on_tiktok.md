---
title: >-
  [Paper Note] Conspiracy Theories and Where to Find Them on TikTok
description: >-
  [ACL 2025][Social Computing][Conspiracy theories] The first systematic analysis of conspiracy theories on TikTok: collecting 1.5 million US long videos via the official API, identifying conspiracy theory content using hashtag enrichment and distant supervision (around 1,000 new videos per month), evaluating the impact of the TikTok Creator Rewards Program, and testing the effectiveness of open-source LLMs (Llama3, Mistral, Gemma) in detecting conspiracy theories based on audi…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Conspiracy theories"
  - "TikTok"
  - "LLM content moderation"
  - "hashtag enrichment"
  - "distant supervision"
  - "Creator Rewards Program"
date: 2026-05-08
content_hash: 8399f6fb430b3403
---

# Conspiracy Theories and Where to Find Them on TikTok

**Conference**: ACL 2025  
**arXiv**: [2407.12545](https://arxiv.org/abs/2407.12545)  
**Code**: [https://anonymous.4open.science/r/ct_tt-FC7E](https://anonymous.4open.science/r/ct_tt-FC7E) (de-anonymized dataset + replication code)  
**Authors**: Francesco Corso, Francesco Pierri, Gianmarco de Francisci Morales
**Institutions**: Politecnico Di Milano, CENTAI
**Area**: Social Media Analysis / Content Safety  
**Keywords**: Conspiracy theories, TikTok, LLM content moderation, hashtag enrichment, distant supervision, Creator Rewards Program

## TL;DR

The first systematic analysis of conspiracy theories on TikTok: collecting 1.5 million US long videos via the official API, identifying conspiracy theory content using hashtag enrichment and distant supervision (around 1,000 new videos per month), evaluating the impact of the TikTok Creator Rewards Program, and testing the effectiveness of open-source LLMs (Llama3, Mistral, Gemma) in detecting conspiracy theories based on audio transcriptions (achieving a precision up to 96% but overall performance comparable to fine-tuned RoBERTa).

## Background & Motivation

**TikTok's Influence**: Almost half of TikTok users (17% of US adults) regularly use the platform to get news and information. TikTok's massive user base and viral dissemination mechanism make it a breeding ground for malicious content.

**The Harm of Conspiracy Theories**: Conspiracy theories are false or unverified narratives claiming secret (usually malevolent) conspiracies. The risk is that they distort consumers' perception of reality, exacerbate misinformation and polarization, and can even lead to dangerous real-world consequences.

**Research Gap**: Prior studies on TikTok conspiracy theories were limited to qualitative analyses of specific topics (e.g., far-right narratives, mpox-related conspiracy content), lacking a quantitative and systematic approach.

**Three Research Questions**:
   - RQ1: How prevalent is the sharing of conspiracy theory videos on TikTok?
   - RQ2: Does the Creator Rewards Program (originally Creativity Program) affect the supply of conspiracy theory content?
   - RQ3: Can LLMs be leveraged to detect conspiracy theories on TikTok?

## Method

### 3.1 Data Collection

The TikTok official Research API was utilized to collect US data from 2021 to 2023:
- **Total Volume**: 1,605,696 videos (1,494,831 unique videos after deduplication)
- **Creators**: 1,178,303 unique users
- **Extra Samples**: Around 10K random videos before and after the launch of the Creator Rewards Program

### 3.2 Conspiracy Hashtag Enrichment

**Step 1: Seed Selection**
From the top 30 frequent seed words of the LOCO conspiracy theory dataset (90,000+ conspiracy and non-conspiracy articles), 10 seeds directly associated with conspiracy theories (e.g., illuminati, reptilians) were manually selected, while overly broad terms (e.g., cancer, AIDS, 5G) were excluded.

**Step 2: Hashtag Similarity Calculation**
The similarity between 281,510 hashtags and the seed hashtags was computed using:

$$sim(h_s, h_t) = \frac{\alpha \cos(W_s, W_t) + (1-\alpha) \cos(H_s, H_t)}{1 + \log(df(h_t))}$$

Using the hashtag-hashtag co-occurrence matrix $H$ and the hashtag-word co-occurrence matrix $W$ (mixing parameter $\alpha=0.3$), the calculation applies a discount based on document frequency to mitigate the impact of overly common hashtags (such as #fyp, #viral).

**Step 3: Manual Verification**
A total of 197 candidate hashtags were produced. Five related videos were checked for each hashtag, and they were classified into:
- **CT** (Conspiracy Theory): 92 hashtags
- **NOCT** (Non-Conspiracy Theory): 68 hashtags
- **DW** (Dog Whistling): 28 hashtags—seemingly harmless tags used by conspiracy theorists as covert signals (e.g., #radiowaves associated with weather modification and chemtrails)
- **HJ** (Hashtag Hijacking): Conspiracy theory users borrowing trending hashtags to gain traffic
- **RHJ** (Reverse Hashtag Hijacking): Users spreading debunking information under conspiracy theory hashtags

**Final Annotation Rule**: Videos containing CT or DW hashtags, and not containing NOCT/HJ/RHJ hashtags, were labeled as conspiracy theories $\rightarrow$ 1,363 videos.

**Quality Verification**: 200 random videos were manually inspected, yielding a Cohen's kappa of 0.81 (strong agreement).

### 3.3 Estimation of Conspiracy Video Volume

The Good-Turing frequency estimator was used to estimate the lower bound of the total number of long videos on TikTok:

$$M = \frac{N}{1 - N_1/K}$$

where $N$ is the number of unique videos, $N_1$ is the number of videos appearing only once, and $K$ is the total number of sampled videos. Maximum likelihood estimation was also used to verify, and the difference between the two methods was $<1\%$.

### 3.5 Video Transcription

- Only about 70K videos (approx. 5%) had available `voice_to_text` (VTT) fields from the TikTok API.
- **OpenAI Whisper** (medium size) was used to expand transcripts.
- Verification: The median word error rate (WER) between Whisper transcripts and native VTT transcripts was about 0.15.
- Filtering: No-speech/music-dominated, and non-English content were excluded.

### 3.5 LLM Detection Pipeline

**Model Selection**: Llama3 (8B), Mistral (7B), and Gemma (7B)—comparable in parameter size and context window, and fully reproducible.

**Three Prompting Strategies**:
1. **Simple**: Directly judge whether the transcript discusses conspiracy theories.
2. **With Definition**: Append the definition of conspiracy theories by Douglas & Sutton (2023).
3. **Step-by-step**: Chain-of-thought—first extract the narratives/assertions, then judge if they represent conspiracy theories, and finally provide the answer.

**Three Data Configurations**:
- **C1** (Balanced + Distant Labeling): 887 positive + 779 negative
- **C2** (Imbalanced + Distant Labeling): 100 positive + 779 negative
- **C3** (Imbalanced + Manual Labeling): 100 manually annotated positive + 779 negative

**Baseline**: Fine-tuned RoBERTa-base

## Key Experimental Results

### RQ1: Conspiracy Theory Prevalence

- A peak in uploads appeared in Q3 2021, where conspiracy theory videos accounted for about 0.2% of the sample (N=542).
- It then stabilized at around 0.1%.
- The absolute number of conspiracy theory videos continued to grow in 2023, reaching **about 1,000 videos per month**.
- **Lower bound estimation**: Conspiracy theory videos account for about 1 in every 500 uploaded videos.

### RQ2: Impact of the Creator Rewards Program

- After the program launched (May 3, 2023), long videos (>1 minute) increased from 0.39% to 1.03% of the sample.
- Both Mann-Whitney and Chi-square tests confirmed that the distributions before and after were statistically significantly different (p<0.001).
- **However**: The proportion of conspiracy theory videos remained stable—the rewards program affected overall content behavior rather than specifically promoting conspiracy theory content.

### RQ3: LLM Detection Performance

**Balanced Setup (C1)**:

| Model + Prompt | Precision | Recall |
|---|---|---|
| Llama3 + Step-by-step | **0.96** | — |
| Gemma + Definition | — | **0.87** |
| RoBERTa (Fine-tuned Baseline) | 0.83 | 0.83 |

- Regarding **precision**: Llama3 + Step-by-step reached 0.96, and most LLM configurations outperformed the RoBERTa baseline.
- Regarding **recall**: Only Gemma + Simple/Definition and Llama3 + Simple outperformed the baseline; other configurations were significantly lower than RoBERTa (-1 to -49pp).

**Imbalanced + Manual Labeling (C3)**:
- Overall performance dropped significantly.
- Llama3 still achieved the highest precision (0.77, Step-by-step).
- Best overall balance: Mistral + Simple (both precision and recall exceeded the baseline).
- The recall of most models was far below the RoBERTa baseline.

**Impact of Text Length**:
- Longer transcripts generally led to higher precision and recall (with a consistent trend).
- However, from Q3 to Q4 (longest transcripts), the improvement in recall stagnated or even decreased.

**Ensemble Models** (majority vote of three LLMs): Achieved a good trade-off between precision and recall, but increased deployment complexity and runtime.

## Highlights & Insights

1. **First Systematic Analysis of Conspiracy Theories on TikTok**: Utilizing the official API for large-scale longitudinal data collection and analysis, with a reproducible methodology.
2. **Discovery of "Dog Whistling" Hashtags**: 28 seemingly harmless hashtags were actually used by conspiracy theorists as covert communication signals—this reveals an important evasion mechanism against detection.
3. **Unexpected Finding on the Creator Rewards Program**: The rewards program led to an platform-wide growth in long videos, but the proportion of conspiracy content remained unchanged—indicating that financial incentives affect content formats rather than content quality.
4. **High Precision but Insufficient Recall of LLMs**: Llama3 achieved a zero-shot precision of up to 0.96, but its recall was often lower than fine-tuned RoBERTa—for content moderation practices, missed conspiracy videos can be more dangerous than false positives.
5. **Pragmatic Experimental Setup**: The three configurations C1/C2/C3 simulated different deployment scenarios from ideal to real-world, providing valuable references for practical content moderation systems.

## Limitations & Future Work

1. **API Transparency**: The TikTok Research API is an opaque system, meaning the data may contain biases or be incomplete.
2. **Seed Dependency**: Different initial seed hashtags might produce different results.
3. **Limited Causal Inference**: The analysis of the Creator Rewards Program demonstrates correlation rather than causation (potential hidden confounding variables exist).
4. **Model Scale Constraints**: Only smaller models with 7-8B parameters were evaluated; larger models might perform better.
5. **No Differentiation of Conspiracy Theory Types**: The dissemination patterns of different conspiracy theories (e.g., illuminati vs. chemtrails) might differ.
6. **Text-only Modality**: Only audio transcriptions were used, without utilizing multimodal information such as video keyframes.

## Related Work & Insights

- **Conspiracy Theory Research**: Psychological analysis by Douglas et al. (2019), linguistic feature analysis by Fong et al. (2021), and the LOCO dataset (Miani et al., 2021).
- **TikTok Content Analysis**: Far-right extremism by Weimann & Masri (2023), mpox conspiracy theories by Zenone & Caulfield (2022), and COVID vaccine misinformation by Basch et al. (2021).
- **LLM Content Moderation**: Reddit conspiracy theory classification by Diab et al. (2024), and social media analysis with LLMs by Plaza-del Arco et al. (2024).

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — The first systematic analysis of TikTok conspiracy theories; the discovery of dog whistling hashtags is a highlight.
- **Value**: ⭐⭐⭐⭐⭐ — Directly serves content moderation practices; the methodology and experimental configuration designs carry strong reference value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The design of the three RQs is reasonable, and the three configurations C1/C2/C3 are practical, though limited by smaller models.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured, with a detailed methodology and comprehensive ethical considerations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Among Us: Language of Conspiracy Theorists on Mainstream Reddit](../../ACL2026/social_computing/among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [\[ACL 2025\] Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)

</div>

<!-- RELATED:END -->
