---
title: >-
  [Paper Note] Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse
description: >-
  [AAAI2026][Audio & Speech][Climate Discourse] Through a psycholinguistic framework, this work analyzes 226,775 Brazilian YouTube climate change videos and 2,756,165 comments…
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
content_hash: 2987a7a2ac97a2e8
---

# Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse

**Conference**: AAAI2026
**arXiv**: [2511.06091](https://arxiv.org/abs/2511.06091)
**Code**: To be confirmed
**Area**: Audio & Speech
**Keywords**: Climate Discourse, Persuasion, Theory of Mind, YouTube, LLM-generated Manipulation, Social Media Analysis

## TL;DR

Through a psycholinguistic framework, this work analyzes 226,775 Brazilian YouTube climate change videos and 2,756,165 comments, revealing that emotional and moral rhetoric significantly drives user engagement. It further demonstrates that fine-tuned LLMs can automatically generate high-engagement climate denial comments, warning of the potential risks of generative AI in public opinion manipulation.

## Background & Motivation

Climate change is a global threat that demands evidence-based policymaking and adequate public understanding. Social media platforms—YouTube in particular—have become primary channels for the dissemination of climate narratives, while simultaneously serving as breeding grounds for misinformation. Brazil, as a representative country of the Global South with significant ecological importance due to the Amazon rainforest, and with YouTube reaching approximately 68% of its population, constitutes an ideal setting for studying climate discourse.

The rapid development of LLMs in recent years has introduced a new risk dimension: prior research has shown that AI-generated text is persuasive and can even influence the formation of beliefs in conspiracy theories. This raises a central concern—can generative AI be weaponized to automate large-scale manipulation of climate discourse, for instance by fabricating a false consensus of "climate denial"?

The motivation of this paper is twofold: (1) to systematically quantify the effect of psycholinguistic features (persuasion strategies and Theory of Mind) on user engagement; and (2) to assess whether these patterns can be exploited by LLMs to automatically generate high-engagement manipulative content.

## Core Problem

1. Which psychological content features (persuasion strategies) most effectively drive audience engagement in Brazilian climate YouTube videos?
2. To what extent can these psychological features predict content popularity?
3. Can these insights be leveraged to design automated persuasive synthetic content (e.g., climate denial campaigns)?

## Method

### Dataset Construction

- **Scale**: 226,775 Brazilian Portuguese-language YouTube video metadata entries and 2,756,165 user comments, spanning 2019–2025.
- **Collection Pipeline**: Videos retrieved via the YouTube Data API v3 using 65 climate-related keywords; non-Portuguese content filtered using FastText language identification; low-relevance videos subsequently filtered by GPT-4.1-mini (temperature = 0).
- **Video Classification**: Videos categorized as short-form (< 3 minutes) or long-form (≥ 3 minutes); since 2023, short-form videos have become the dominant format for climate topics.

### Psycholinguistic Annotation

#### Persuasion Strategy Annotation (Video Level)

GPT-4.1 with 5-shot prompting is used to annotate 10 persuasion strategies at the video level:

| Strategy | Description |
|---|---|
| Logical Appeal | Persuading through reasons and evidence |
| Emotional Appeal | Eliciting emotional responses |
| Statistical Evidence | Providing concrete data and statistics |
| Social Norm | Applying pressure through social conformity |
| Authority | Citing experts, institutions, and official reports |
| Personal Stories | Narrating personal experiences |
| Moral Appeal | Appealing to moral responsibility |
| Reciprocity | Emphasizing mutual benefits |
| Scarcity | Presenting time-limited and irreversible consequences |
| Common Ground | Building shared identity and values |

Manual validation yields an average F1 = 0.93 and accuracy = 0.98.

#### Theory of Mind Annotation (Comment Level)

GPT-4.1-mini is used to annotate user comments with 7 Theory of Mind categories: Belief, Intention, Desire, Emotion, Knowledge, Percept, and Non-literal. Manual validation yields F1 = 0.66 and accuracy = 0.83.

### Case Study 1: Engagement Modeling

The impact of psycholinguistic features on user engagement is assessed through a three-stage analysis:

1. **Video level**: Linear regression is applied to analyze the effect of the persuasion strategy vector $\mathbf{p}_i$ on normalized like rate $L_i$ and comment rate $R_i$, controlling for confounders such as video duration and channel identity.
2. **Strategy–Mind correlation**: Each video's comment ToM vectors are aggregated as $\bar{\mathbf{t}}_i = (1/|C_i|)\sum_{c_k \in C_i} \mathbf{t}_k$, and partial correlations between persuasion strategies and ToM categories are computed.
3. **Comment level**: Comment like counts and reply counts serve as dependent variables, with ToM annotations as independent variables, controlling for comment length and temporal distance.

### Case Study 2: Popularity Prediction

Comment pairs $(c_i, c_j)$ are constructed with binary label $y_{ij}^{(\ell)} = \mathbb{I}[\ell_i > \ell_j]$, predicting which comment is more popular. Three classes of methods are employed:

- **LLM-as-a-Judge**: GPT-4.1, o4-mini, Phi-4, Llama-3.1-8B, Llama-4-Maverick
- **Fine-tuned Encoder Models**: BERTimbau (Brazilian Portuguese BERT), DeBERTa V3
- **Bradley-Terry Model**: A linear classifier trained on comment embeddings

### Case Study 3: Comment Generation

Llama-3-8B is fine-tuned to generate targeted comments under three scenarios:

1. **Sampling by persuasion strategy**: Controlling for video-level effects.
2. **Sampling by ToM profile**: Generating comments that reflect specific mental states.
3. **Sampling by belief stance**: Three distinct models corresponding to "belief in climate change," "climate denial," and "extreme denial."

Evaluation: For each generated comment, $K$ most similar real comments are retrieved, and their average like/reply counts serve as a proxy evaluation metric.

## Key Experimental Results

### Effect of Persuasion Strategies on Engagement

- The most frequently used strategies (Logical Appeal 51%, Authority 47%, Common Ground 36%) are all associated with **lower** user engagement.
- Emotional Appeal (33%) and Moral Appeal (26%) are associated with **significantly higher** engagement; moral appeals increase video likes by an average of 2.1%.
- The effectiveness of moral rhetoric in short-form videos has grown consistently over time.

### Popularity Prediction

| Model | Best Accuracy | Condition |
|---|---|---|
| BERTimbau | **88%** | No context, random pairing |
| GPT-4.1 | 82% | With video context + few-shot |
| DeBERTa V3 | 84% | With video context |

- Emotional ToM improves prediction performance by an average of **4.69%**.
- BERTimbau achieves 88% using comment text alone, indicating that comment content itself contains sufficient signals for engagement prediction.

### Comment Generation

| Model | Estimated Like Count $\hat{\ell}_{gen|1}$ |
|---|---|
| Baseline (random comments) | 2.20 |
| Engaging (fine-tuned on high-like comments) | **7.25** (3.3× improvement) |
| Believe (climate belief) | 3.23 |
| Denial (climate denial) | 1.91 |
| Extreme (extreme denial) | 2.37 |

Comments generated by the extreme denial model exhibit greater detail and rhetorical intensity, making them more engaging than those from the standard denial model.

## Highlights & Insights

1. **Large-scale psycholinguistic dataset**: A Brazilian climate discourse dataset comprising 226,775 videos and 2,756,165 comments, annotated with persuasion strategies and ToM labels, is released—one of the largest non-English resources in this domain.
2. **Clear causal chain from persuasion to engagement**: A complete analytical pipeline from "persuasion strategies → user psychological states → engagement behavior" is established across three progressively layered case studies.
3. **Empirical demonstration of AI manipulation risks**: Rather than remaining at the level of theoretical discussion, the work demonstrates the feasibility of automated opinion manipulation through actual LLM fine-tuning; the outputs of the extreme denial model exhibit a striking degree of realism.
4. **Unique Brazilian/Portuguese perspective**: This work fills a gap in climate discourse research concerning Global South countries. The finding that BERTimbau outperforms GPT-4.1 on Portuguese comments further underscores the importance of language-specificity.
5. **In-depth analysis of short-form video trends**: The paper reveals that climate short-form videos have surpassed long-form videos since 2023, a format shift that further compresses the space available for fact-checking.

## Limitations & Future Work

- **Text-only analysis**: Visual, audio, and other multimodal elements affecting persuasiveness are not considered—a particularly notable limitation given YouTube's nature as a video platform.
- **Incomplete engagement metrics**: Important factors such as recommendation algorithms, individual psychological differences, and user demographics are not accounted for.
- **Geographic and linguistic scope**: All findings are limited to Brazilian Portuguese YouTube content; cross-lingual and cross-platform generalizability remains unverified.
- **Moderate ToM annotation quality**: An F1 of 0.66 represents a substantial gap compared to the 0.93 achieved for persuasion strategy annotation, potentially affecting the reliability of downstream analyses.
- **Indirect evaluation of generated comments**: Proxy evaluation via nearest-neighbor retrieval rather than real-platform deployment precludes confirmation of actual engagement effects.
- Future work may extend the framework to multilingual and multi-platform (TikTok/X) comparisons, and incorporate multimodal analysis.

## Related Work & Insights

| Dimension | Ours | Existing Climate Discourse Research |
|---|---|---|
| Language/Region | Brazilian Portuguese | Predominantly English |
| Analytical Framework | Dual-dimension: persuasion strategies + ToM | Typically single-dimension (stance detection or sentiment analysis) |
| Manipulation Risk Assessment | LLM fine-tuning generation experiments | Primarily theoretical discussion |
| Data Scale | 226,775 videos + 2,756,165 comments | Typically < 150,000 tweets |
| Platform | YouTube (videos + comments) | Predominantly Twitter/X |

Compared to Costello et al. (2024) on AI persuasion, this work shifts focus from controlled experiments to real-world social media settings. Compared to Breum et al. (2024) on LLM persuasiveness analysis, this work adds the ToM dimension and actual generation experiments.

**Implications and Connections**

- **Warning for generative AI governance**: The high engagement efficacy of emotional and moral rhetoric, combined with LLMs' generative capacity, creates the conditions for low-cost, large-scale opinion manipulation, calling for governance frameworks around synthetic media.
- **Aggravated challenges for fact-checking**: The shift toward short-form video compresses the space for in-depth information dissemination, and algorithmic recommendation further amplifies emotional content, undermining fact-checking effectiveness.
- **BERTimbau > GPT-4.1** suggests that in specific linguistic and cultural contexts, localized smaller models may outperform general-purpose large models—an important finding for multilingual AI research.
- **Transferable methodology**: The dual-dimension analytical framework of persuasion strategies and ToM can be applied to other social issues (e.g., vaccine hesitancy, political polarization) and other platforms.

## Rating

- Novelty: ⭐⭐⭐⭐ (The dual-dimension framework of persuasion strategies + ToM is novel; the empirical analysis of LLM manipulation risks is forward-looking.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three well-designed case studies with large-scale data, though ToM annotation quality and the generation evaluation approach leave room for improvement.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich case studies, and a coherent logical chain from analysis to risk warning.)
- Value: ⭐⭐⭐⭐ (Significant implications for AI ethics and social media governance; the public release of the dataset constitutes a meaningful community contribution.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)
- [\[AAAI 2026\] Aligning Generative Music AI with Human Preferences: Methods and Challenges](aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)
- [\[NeurIPS 2025\] Accelerate Creation of Product Claims Using Generative AI](../../NeurIPS2025/audio_speech/accelerate_creation_of_product_claims_using_generative_ai.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)
- [\[NeurIPS 2025\] Echoes of Humanity: Exploring the Perceived Humanness of AI Music](../../NeurIPS2025/audio_speech/echoes_of_humanity_exploring_the_perceived_humanness_of_ai_music.md)

</div>

<!-- RELATED:END -->
