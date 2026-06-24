---
title: >-
  [Paper Note] Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents
description: >-
  [ACL 2025][Time Series][Sentiment Forecasting] Proposed a multi-perspective role-playing framework (MPR) based on LLMs. By using subjective agents to simulate user posting and an objective agent (a fine-tuned "psychologist" LLM) to audit behavioral consistency, it forecasts social media users' future emotional responses to real-time events through iterative rectification, significantly outperforming traditional methods at both macro and micro levels.
tags:
  - "ACL 2025"
  - "Time Series"
  - "Sentiment Forecasting"
  - "Role-Playing"
  - "Multi-Agent"
  - "Social Media"
  - "Behavioral Psychology"
date: 2026-05-08
content_hash: 6c6ec22468730b85
---

# Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents

**Conference**: ACL 2025  
**arXiv**: [2505.24331](https://arxiv.org/abs/2505.24331)  
**Code**: [https://github.com/ManFanhang/Context-Aware-Sentiment-Forecasting-via-LLM-based-Multi-Perspective-Role-Playing-Agents](https://github.com/ManFanhang/Context-Aware-Sentiment-Forecasting-via-LLM-based-Multi-Perspective-Role-Playing-Agents)  
**Area**: Time Series  
**Keywords**: Sentiment Forecasting, Role-Playing, Multi-Agent, Social Media, Behavioral Psychology

## TL;DR
Proposed a multi-perspective role-playing framework (MPR) based on LLMs. By using subjective agents to simulate user posting and an objective agent (a fine-tuned "psychologist" LLM) to audit behavioral consistency, it forecasts social media users' future emotional responses to real-time events through iterative rectification, significantly outperforming traditional methods at both macro and micro levels.

## Background & Motivation

**Background**: Social media sentiment analysis is a popular traditional research direction in NLP, where existing works primarily focus on **retrospective** analysis—determining the sentiment polarity of an existing comment. Research on sentiment evolution also largely models the mutual influence between users (e.g., SINN, DeGroot models).

**Limitations of Prior Work**: (a) Existing methods only consider direct interactions between users, **ignoring the driving force of external event contexts** on sentiment—for instance, a hurricane landfall or election results announcement can drastically shift public emotion; (b) Traditional methods model sentiment as a continuous evolution process, making it difficult to capture **sudden emotional jumps** (e.g., from -2 to +2); (c) The personal characteristics of social media users (tone style, attitudes toward events) are difficult to acquire and model.

**Key Challenge**: Sentiment forecasting is inherently a forward-looking reasoning task that requires a comprehensive understanding of complex event semantics, individual user traits, and social influence, which traditional numerical models and shallow-layered neural networks lack.

**Goal**: Formulate sentiment forecasting as a reasoning problem: given all information up to time $t$ (user historical comments, event context, followers' comments), predict the user's sentiment $\sigma_{t'}$ at a future time $t'$.

**Key Insight**: Leverage the role-playing capabilities of LLMs to simulate user behavior. However, simple role-playing suffers from randomness, so a "psychologist" LLM is introduced as an objective auditor to ensure behavioral consistency through iterative rectification.

**Core Idea**: Use a dual-agent (subjective user + objective psychologist) role-playing framework to convert sentiment forecasting into a reasoning workflow of "user-posting simulation $\rightarrow$ consistency auditing $\rightarrow$ iterative rectification".

## Method

### Overall Architecture
Input: User historical comments $\mathcal{C}_t^u$, user attributes $\mathcal{A}^u$, event context $\mathcal{E}_t^u$, and followers' comments $\mathcal{F}_t^u$. Output: User's sentiment score $\sigma_{t'}$ at a future time $t'$. The entire pipeline is divided into four stages: Feature Extraction $\rightarrow$ Subjective Agent comment generation $\rightarrow$ Objective Agent consistency auditing $\rightarrow$ Iterative Rectification. Ultimately, a BERT sentiment classifier is applied to the generated comments to retrieve sentiment labels.

### Key Designs

1. **Feature Extraction**:

    - **Function**: Extract two types of implicit features from historical comments—**textual tone style** $\nu_t^u$ (e.g., sarcastic, humorous, formal) and **attitude toward the event** $\alpha_t^u$.
    - **Mechanism**: Directly leverage an LLM to analyze historical comments. Tone is extracted as 3 descriptive adjectives. Attitude extraction integrates tone, historical comments, and event context, allowing attitudes to evolve dynamically alongside the event.
    - **Design Motivation**: Social media users are highly anonymous, making demographic labels unavailable. However, key features influencing sentiment expression can be inferred from user-generated content. Tone style remains relatively stable (reflecting social persona consistency), while attitudes change dynamically with the event.

2. **Subjective Role-Playing Agent (Subjective Agent)**:

    - **Function**: LLM plays the role of the target user, reviews followers' comments, and then generates a future comment $\phi_{t'}^u$.
    - **Mechanism**: Inject the extracted features (tone, attitude, attributes) and users' historical comments as context into the LLM, prompting it under a few-shot setting to mimic the user's posting style. It first browses followers' comments to simulate the information acquisition process, and then generates a future comment based on the latest developments of the event.
    - **Design Motivation**: Directly predicting sentiment scores (as in the MPR-RP variant) yields performance close to random guessing, showing the necessity of simulating the human process of "acquiring information $\rightarrow$ thinking $\rightarrow$ expressing". Gemma 2 9B and Mistral NeMo 12B are chosen over the GPT series, as GPT models filter negative or offensive content, making them less suitable for simulating real-world social media users.

3. **Objective Role-Playing Agent (Objective Agent)**:

    - **Function**: A fine-tuned "behavioral psychologist" LLM audits comments generated by the subjective agent, assessing tone consistency and the plausibility of attitude changes.
    - **Mechanism**: First, 3 behavioral psychology experts annotate comment consistency (Fleiss' Kappa = 0.796). Next, GPT-4o is used to expand this dataset to 25,000 auditing samples to perform LoRA fine-tuning on Llama 3 8B Instruct. The auditing output includes "whether consistent" and "inconsistency analysis".
    - **Design Motivation**: Pure role-playing has high stochasticity, and generated comments may align poorly with users' historical behaviors. Introducing an LLM equipped with professional psychological knowledge as a "reviewer" constrains randomness and ensures the behavioral plausibility of the generated comments.

4. **Iterative Rectification**:

    - **Function**: For comments that fail the consistency check, the objective agent's analysis is fed back to the subjective agent to regenerate the comment, iterating up to $n=3$ times.
    - **Mechanism**: Resembling the "revising-and-resubmitting" loop in paper peer review, the objective agent's detailed analysis (rather than a simple yes/no response) guides the subjective agent to adjust its generation path.
    - **Design Motivation**: Balancing computational efficiency and rectification performance, 3 iterations are determined experimentally as the optimal trade-off point.

### Loss & Training
The fine-tuning of the objective agent uses standard LoRA supervised fine-tuning loss with a learning rate of $\eta = 1 \times 10^{-4}$. The subjective agent requires no training and directly performs zero/few-shot role-playing using general-purpose LLMs.

## Key Experimental Results

### Main Results
Evaluated on two Twitter datasets (2012 Hurricane Sandy and 2020 US Election), measuring performance at both macro (JSD distribution distance) and micro (Accuracy / Macro F1) levels.

**Macro Results (JSD, lower is better)**:

| Method | Sandy-NJ T1 | Sandy-NY T1 | Election T3 | Election T4 |
|------|-------------|-------------|-------------|-------------|
| SINN | 0.1673 | 0.1504 | 0.0554 | 0.0625 |
| NN | 0.1904 | 0.1733 | 0.0482 | 0.0441 |
| **MPRG (Ours)** | **0.0243** | **0.0456** | **0.0097** | **0.0053** |
| **MPRM (Ours)** | **0.0148** | **0.0220** | **0.0106** | **0.0068** |

**Micro Results (Accuracy / Macro F1)**:

| Method | Sandy-NJ T1 Acc/F1 | Sandy-NY T1 Acc/F1 | Election T3 Acc/F1 | Election T4 Acc/F1 |
|------|---------------------|---------------------|---------------------|---------------------|
| SINN | 0.353/0.179 | 0.385/0.168 | 0.476/0.193 | 0.485/0.183 |
| **MPRG** | **0.413/0.302** | **0.396/0.292** | **0.615/0.374** | **0.596/0.397** |
| **MPRM** | **0.445/0.312** | **0.482/0.310** | **0.593/0.368** | **0.581/0.370** |

### Ablation Study

| Configuration | Sandy-NJ T1 Acc/F1 | Description |
|------|---------------------|------|
| MPR (Full) | 0.413/0.342 | Full model |
| MPR-OB (w/o Objective Agent) | 0.408/0.294 | F1 drops by 4.8%; objective review mainly improves F1 |
| MPR-FE (w/o Feature Extraction) | 0.343/0.266 | Acc drops by 7%, F1 drops by 7.6%; feature extraction contributes significantly |
| MPR-RP (w/o Comment Generation) | 0.212/0.186 | Close to random guessing, showing the necessity of generating comments before extracting sentiment |

### Key Findings
- **Comment generation is central**: Excluding comment generation (directly predicting sentiment scores) causes performance to collapse to random levels, validating the necessity of "simulating human behavioral processes".
- **Feature extraction contributes the most**: Removing it leads to significant drops in both Accuracy and F1, indicating that user tone and attitude are critical cues for sentiment forecasting.
- **Objective Agent improves F1 instead of Accuracy**: This indicates it primarily reduces "highly biased erroneous predictions", making predictions more consistent and stable.
- **Event context dependency**: Performance on the Election dataset is markedly better than on the Hurricane dataset. This is because election discussions heavily rely on news and social media information (accessible to LLMs), whereas in the hurricane context, 15%+ of users post based on personal, physical experiences (inaccessible to LLMs).
- Accuracy can reach **63.9%** when predicting only sentiment polarity (positive/neutral/negative).

## Highlights & Insights
- **Transforming sentiment forecasting into behavioral simulation**: Instead of directly regressing sentiment scores, it models users' posting behavior first before extracting sentiment. This "process simulation" paradigm is closer to human cognition than end-to-end prediction, and can transfer to other human behavior prediction tasks (e.g., purchase intent and voting behavior prediction).
- **Dual-agent mutual auditing mechanism**: Subjective agents focus on creative generation, while the objective agent controls quality. This "generation-audit-rectification" loop is a highly versatile pattern in LLM agent systems.
- **Fine-tuning small models with professional psychological knowledge**: LoRA fine-tuning on Llama 3 8B with only 25K samples enables it to act effectively as a "behavioral consistency auditor", providing a low-cost yet highly effective solution.

## Limitations & Future Work
- **Limited source of information**: Agents only access textual information on social media and lack knowledge of users' real-life physical experiences (e.g., personal encounters during a hurricane), making 15%+ of user sentiments unpredictable.
- **Text-only modality support**: Social media features increasingly more image/video content; multimodal integration is a straightforward direction for improvement.
- **Constrained LLM choices**: Common commercial models like GPT series filter negative or aggressive content due to safety guardrails, restricting the design space for user simulation.
- **Scalability**: Each user demands separate feature extraction and multi-turn generation-audit processes, making computational complexity scale linearly with the number of users.
- **Evaluation dependency on the BERT sentiment classifier**: Ground-truth labels are pseudo-generated from a BERT classifier (with 87% accuracy), introducing label noise into the process.

## Related Work & Insights
- **vs SINN**: Traditional work like SINN uses sociological models (e.g., Stochastic bound confidence models) to guide neural networks in modeling sentiment evolution, but only considers user-user interactions without event contexts. This paper implicitly integrates event semantic information directly via LLM role-playing.
- **vs Traditional LLM Role-Playing**: Existing role-playing works (such as simulating anime characters or historical figures) require extensive character-specific training data. In contrast, this work simulates anonymous users using only comment history and implicit feature extraction, making it highly applicable to large-scale social media scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of modeling sentiment forecasting as role-playing + behavioral simulation is highly novel, and the dual-agent mutual auditing mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested across multiple time points on two datasets with comprehensive ablation tests, though it suffers from a lack of comparison with broader datasets and other LLMs.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and formulation is rigorous, though some symbol definitions are somewhat redundant.
- Value: ⭐⭐⭐⭐ Sentiment forecasting is an important application area, and the framework paradigm can easily extend to other human behavior prediction tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading](../../ICML2025/time_series/event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_.md)
- [\[ACL 2025\] Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement](time-mqa_time_series_multi-task_question_answering_with_context_enhancement.md)
- [\[ICLR 2026\] COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting](../../ICLR2026/time_series/cosa_context-aware_output-space_adapter_for_test-time_adaptation_in_time_series_.md)
- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](../../ICLR2026/time_series/hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)
- [\[CVPR 2025\] Competition-Aware CPC Forecasting with Near-Market Coverage](../../CVPR2025/time_series/competition-aware_cpc_forecasting_with_near-market_coverage.md)

</div>

<!-- RELATED:END -->
