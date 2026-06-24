---
title: >-
  [Paper Note] Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse
description: >-
  [AAAI2026][Audio & Speech][Climate Discourse] This paper analyzes 226k climate change videos and 2.75M comments on Brazilian YouTube using a psycholinguistic framework, revealing that emotional/moral rhetoric significantly drives user engagement. It also demonstrates that fine-tuned LLMs can automatically generate highly engaging climate-denial comments, warning of the potential risks of generative AI in public opinion manipulation.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Climate Discourse"
  - "Persuasion"
  - "Theory of Mind"
  - "YouTube"
  - "LLM-generated Manipulation"
  - "Social Media Analysis"
date: 2026-05-08
content_hash: 7a4ceccf83f22f07
---

# Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse

**Conference**: AAAI2026  
**arXiv**: [2511.06091](https://arxiv.org/abs/2511.06091)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: Climate Discourse, Persuasion, Theory of Mind, YouTube, LLM-generated Manipulation, Social Media Analysis  

## TL;DR

This paper analyzes 226k climate change videos and 2.75M comments on Brazilian YouTube using a psycholinguistic framework, revealing that emotional/moral rhetoric significantly drives user engagement. It also demonstrates that fine-tuned LLMs can automatically generate highly engaging climate-denial comments, warning of the potential risks of generative AI in public opinion manipulation.

## Background & Motivation

Climate change is a global threat, and addressing it requires evidence-based policy-making and public understanding. Social media (particularly YouTube) has increasingly become a major channel for disseminating climate narratives, while also serving as a breeding ground for misinformation. As a representative country of the Global South, Brazil holds a significant ecological position due to the Amazon rainforest, and YouTube covers approximately 68% of its population, making it an ideal scenario for studying climate discourse.

The rapid development of LLMs in recent years has introduced new risk dimensions: existing research indicates that AI-generated texts are persuasive and can even influence beliefs in conspiracy theories. This raises a core concern: can generative AI be exploited to manipulate climate discourse at scale and in an automated manner, such as by fabricating a false consensus of "climate denial"?

The motivation of this study is to: (1) systematically quantify the impact of psycholinguistic features (persuasion strategies + Theory of Mind) on user engagement; and (2) evaluate whether these patterns can be exploited by LLMs to automatically generate highly engaging, manipulative content.

## Core Problem

1. Which psychological content features (persuasion strategies) are most effective at driving viewer engagement in Brazilian climate YouTube videos?
2. To what extent can these psychological features predict content popularity?
3. Can these insights be exploited to design automated, persuasive synthetic content (e.g., climate denial campaigns)?

## Method

### Dataset Construction

* **Scale**: Metadata of 226,775 Brazilian Portuguese YouTube videos + 2,756,165 user comments, spanning from 2019 to 2025
* **Collection Process**: Retrieved via YouTube Data API v3 based on 65 climate-related keywords, filtered non-Portuguese content using FastText for language identification, and then filtered low-relevance videos using GPT-4.1-mini (temperature = 0)
* **Video Classification**: Categorized by duration into Short videos (<3 minutes) and Long videos ($\ge 3$ minutes); shorts have become the mainstream format for climate topics since 2023

### Psycholinguistic Annotation

#### Persuasion Strategy Annotation (Video Level)

Using GPT-4.1 via 5-shot prompting to annotate 10 types of persuasion strategies on video content:

| Strategy | Description |
|---|---|
| Logical Appeal | Persuasion based on reason and evidence |
| Emotional Appeal | Evoking emotional responses |
| Statistical Evidence | Providing concrete data and statistics |
| Social Norm | Exerting pressure through social proof/consensus |
| Authority | Citing experts, institutions, and official reports |
| Personal Stories | Sharing personal experiences |
| Moral Appeal | Appealing to moral responsibility |
| Reciprocity | Emphasizing reciprocal benefits |
| Scarcity | Presenting temporal urgency and irreversible impacts |
| Common Ground | Establishing shared identity and values |

Human validation results: Average F1 = 0.93, Accuracy = 0.98.

#### Theory of Mind Annotation (Comment Level)

Using GPT-4.1-mini to annotate user comments with 7 Theory of Mind (ToM) categories: Belief, Intention, Desire, Emotion, Knowledge, Percept, and Non-literal. Human validation: F1 = 0.66, Accuracy = 0.83.

### Case Study 1: Engagement Modeling

Evaluating the impact of psycholinguistic features on user engagement in three stages:

1. **Video Level**: Using linear regression to analyze the impact of persuasion strategy vector $\mathbf{p}_i$ on standardized comment rate $R_i$ and like rate $L_i$, controlling for confounding factors such as video duration and channel attributes.
2. **Strategy-Mind Association**: Aggregating the comment ToM vector of each video as $\bar{\mathbf{t}}_i = (1/|C_i|)\sum_{c_k \in C_i} \mathbf{t}_k$, and computing the partial correlation between persuasion strategies and ToM categories.
3. **Comment Level**: Treating the number of likes and replies on comments as dependent variables and ToM annotations as independent variables, while controlling for comment length and time delta.

### Case Study 2: Popularity Prediction

Pairing comments as $(c_i, c_j)$ and defining a binary label $y_{ij}^{(\ell)} = \mathbb{I}[\ell_i > \ell_j]$ to predict which comment will be more popular. Three types of methods are used:

- **LLM-as-a-Judge**: GPT-4.1, o4-mini, Phi-4, Llama-3.1-8B, Llama-4-Maverick
- **Encoder Model Fine-tuning**: BERTimbau (Brazilian Portuguese BERT), DeBERTa V3
- **Bradley-Terry Model**: Training linear classifiers based on comment embeddings

### Case Study 3: Comment Generation

Fine-tuning Llama-3-8B to generate targeted comments across three scenarios:

1. **Sampling by Persuasion Strategy**: Controlling video-level effects
2. **Sampling by ToM Profile**: Generating comments reflecting specific mental states
3. **Subdivision by Belief/Stance**: Disentangling three models: "Believe Climate Change", "Climate Denial", and "Extreme Denial"

Evaluation Method: Retrieving $K$ most similar real comments for each generated comment, using their average number of likes/replies as proxy evaluation metrics.

## Key Experimental Results

### Impact of Persuasion Strategies on Engagement

- The most frequently used strategies (Logical Appeal 51%, Authority 47%, Common Ground 36%) are all associated with **lower** user engagement.
- Emotional Appeal (33%) and Moral Appeal (26%) are associated with **significantly higher** engagement; specifically, moral appeal yields an average of 2.1% increase in video likes.
- The effectiveness of moral rhetoric in short videos has grown steadily over time.

### Popularity Prediction

| Model | Best Accuracy | Condition |
|---|---|---|
| BERTimbau | **88%** | No context, random pairings |
| GPT-4.1 | 82% | With video context + few-shot |
| DeBERTa V3 | 84% | With video context |

- Emotional ToM improves prediction performance by **4.69%** on average.
- BERTimbau achieves 88% accuracy relying on comment text alone, indicating that comment content itself contains sufficient signals to predict engagement.

### Comment Generation

| Model | Estimated Likes $\hat{\ell}_{gen|1}$ |
|---|---|
| Baseline (Random Comments) | 2.20 |
| Engaging (Fine-tuned on High-like Comments) | **7.25** (3.3x increase) |
| Believe (Believe Climate Change) | 3.23 |
| Denial (Climate Denial) | 1.91 |
| Extreme (Extreme Denial) | 2.37 |

Comments generated by the Extreme Denial model contain more details and rhetorical intensity, making them more engaging than those from the standard Denial model.

## Highlights & Insights

1. **Large-scale Psycholinguistic Dataset**: Releasing a Brazilian climate discourse dataset containing 226k videos and 2.75M comments with persuasion strategy and ToM annotations, representing one of the largest non-English resources in this domain.
2. **Clear Causal Chain of Persuasion and Engagement**: A complete analysis pipeline from "persuasion strategy $\rightarrow$ user mental state $\rightarrow$ engagement behavior" developed across three progressive case studies.
3. **Empirically Uncovering AI Manipulation Risks**: Moving beyond theoretical discussions, this study demonstrates the feasibility of automated opinion manipulation via actual LLM fine-tuning; the outputs from the extreme denial model exhibit a striking and alarming level of realism.
4. **Unique Brazil + Portuguese Perspective**: Filling a gap in Global South representation within climate discourse research. BERTimbau outperforming GPT-4.1 on Portuguese comments highlights the crucial importance of language specificity.
5. **In-depth Analysis of Short Video Trends**: Revealing that short climate videos have surpassed long ones since 2023, a format shift that further compresses the space available for fact-checking.

## Limitations & Future Work

- **Text-Only Analysis**: Ignores the persuasive effects of visual, audio, and other multimodal elements, a limitation that is particularly pronounced for YouTube as a video platform.
- **Incomplete Engagement Metrics**: Does not account for crucial factors affecting engagement such as recommendation algorithms, individual psychological differences, and user profiles.
- **Geographic and Language Limitations**: All findings are restricted to Brazilian Portuguese YouTube content; cross-lingual and cross-platform generalizability remains unverified.
- **Moderate ToM Annotation Quality**: The F1 score of only 0.66 is substantially lower than the 0.93 achieved for persuasion strategy annotations, potentially affecting the reliability of downstream analyses.
- **Indirect Evaluation of Generated Comments**: Uses proxy evaluations via nearest neighbor retrieval rather than staging real-world platform testing, preventing confirmation of actual engagement effects.
- Future work can expand to cross-lingual and cross-platform (e.g., TikTok/X) comparisons and introduce a multimodal analysis framework.

## Related Work & Insights

| Aspect | Ours | Prior Climate Discourse Research |
|---|---|---|
| Language/Region | Brazilian Portuguese | Predominantly English |
| Analysis Framework | Dual-dimensional (Persuasion Strategy + ToM) | Typically single-dimensional (stance detection or sentiment analysis) |
| Manipulation Risk Assessment | LLM fine-tuning generation experiments | Primarily theoretical discussion |
| Data Scale | 226k videos + 2.75M comments | Typically <150k tweets |
| Platform | YouTube (Videos + Comments) | Predominantly Twitter/X |

Compared to the AI persuasion study by Costello et al. (2024), this work shifts the focus from controlled experiments to real-world social media settings. Compared to the LLM persuasiveness analysis by Breum et al. (2024), this study introduces the ToM dimension and practical generation experiments.

## Implications & Takeaways

- **Warning for Generative AI Governance**: The high engagement efficacy of emotional and moral rhetoric + the generative capacity of LLMs = the possibility of low-cost, large-scale public opinion manipulation, calling for establishing a governance framework for synthetic media.
- **Escalating Challenges for Fact-checking**: The trend toward short-form videos compresses the space for disseminating in-depth information. Algorithmic recommendations further amplify emotional content, raising concerns about the efficacy of fact-checking.
- **Finding of BERTimbau > GPT-4.1**: This suggests that in specific linguistic and cultural scenarios, localized smaller models can be more effective than general large models, offering important insights for multilingual AI research.
- **Transferable Methodology**: The dual-dimensional analytical framework of persuasion strategies + ToM can be applied to other societal issues (e.g., vaccine hesitancy, political polarization) and other platforms.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The dual-dimensional framework of persuasion strategies + ToM is novel, and the empirical analysis of LLM manipulation risks is forward-looking.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Three Case Studies are comprehensively designed with a large data scale, though there is room for improvement in ToM annotation quality and generation evaluation methods.)
- **Writing Quality**: ⭐⭐⭐⭐ (The structure is clear with rich cases, presenting a coherent logical chain from analysis to risk warning.)
- **Value**: ⭐⭐⭐⭐ (Holds significant warning value for AI ethics and social media governance, and the open release of the dataset constitutes a valuable community contribution.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)
- [\[AAAI 2026\] Aligning Generative Music AI with Human Preferences: Methods and Challenges](aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)
- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](../../ICML2026/audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)
- [\[NeurIPS 2025\] Accelerate Creation of Product Claims Using Generative AI](../../NeurIPS2025/audio_speech/accelerate_creation_of_product_claims_using_generative_ai.md)
- [\[ICML 2026\] Probing Token Spaces under Generator Shift in AI-Generated Music Detection](../../ICML2026/audio_speech/probing_token_spaces_under_generator_shift_in_ai-generated_music_detection.md)

</div>

<!-- RELATED:END -->
