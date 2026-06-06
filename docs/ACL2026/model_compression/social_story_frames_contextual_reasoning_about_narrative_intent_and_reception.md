---
title: >-
  [Paper Note] Social Story Frames: Contextual Reasoning about Narrative Intent and Reception
description: >-
  [ACL2026][Model Compression][Narrative understanding] This paper proposes SocialStoryFrames, utilizing a 10-dimension reader response taxonomy and two distilled models to situate Reddit stories within community and conve…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Narrative understanding"
  - "reader response"
  - "social media"
  - "contextual reasoning"
  - "model distillation"
date: 2026-05-08
content_hash: 03ffdde8adb65e9d
---

# Social Story Frames: Contextual Reasoning about Narrative Intent and Reception

**Conference**: ACL2026  
**arXiv**: [2512.15925](https://arxiv.org/abs/2512.15925)  
**Code**: social-story-frames (Project name provided in the paper; full URL not expanded in the main text)  
**Area**: NLP Understanding / Computational Social Science / Narrative Reasoning  
**Keywords**: Narrative understanding, reader response, social media, contextual reasoning, model distillation

## TL;DR
This paper proposes SocialStoryFrames, utilizing a 10-dimension reader response taxonomy and two distilled models to situate Reddit stories within community and conversational contexts. It infers narrative intent, reader emotions, and value judgments, demonstrating a more granular analysis of community narrative practices on 6,140 social media stories compared to semantic similarity.

## Background & Motivation
**Background**: NLP approaches to story processing have long focused on internal content, such as event causality, character psychology, plot consistency, or local reader reactions like suspense and curiosity. Computational social science also analyzes narratives in online communities, but common practices either involve deep studies of a single community or statistical analysis of story counts, structural archetypes, or thematic distributions across large-scale corpora.

**Limitations of Prior Work**: These two paths involve a clear trade-off between depth and scale. Case studies can explain the negotiation of power, identity, or emotion within a specific community but are difficult to apply horizontally across dozens of communities. Large-scale analyses scale easily but often simplify stories into textual themes or embedding similarities, losing the social-pragmatic layers of "why the author told this story" and "how readers might interpret it."

**Key Challenge**: Social media stories are not isolated texts. An individual telling a story about a product failure in r/buildapc versus a story about a failed trial in r/MakeupAddiction involves completely different surface topics, yet both might be seeking advice, validating experiences, or obtaining emotional support. Relying solely on story text misses this similarity in "narrative function," while relying purely on manual interpretation cannot cover the vast volume of communities.

**Goal**: The authors aim to construct a formalized framework that possesses both theoretical explanatory power and the capability for batch application by models. This framework seeks to answer: what intent a story is perceived to have in specific communities and dialogues; what interpretations, predictions, emotions, and value judgments readers produce; and whether narrative practices across different communities can be compared beyond semantic themes.

**Key Insight**: The paper maps concepts from reader response theory, narrative theory, pragmatics, and psychology into a SocialStoryFrames taxonomy. Reference reasoning is generated using GPT-4o / GPT-4.1 and then distilled into open-weight models. This approach preserves theoretical dimensions while transforming expensive expert or closed-source model reasoning into reproducible batch pipelines.

**Core Idea**: Use "community context + conversational context + reader response taxonomy" to replace simple textual semantic representations, modeling the social functions and reception ways of stories in online communities.

## Method
SocialStoryFrames is not a single classifier but a complete pipeline encompassing theoretical taxonomy, corpus construction, context summarization, reasoning generation, reasoning classification, and community analysis. The input consists of a Reddit comment containing a story, its community information, and the preceding dialogue; the output includes free-text reasoning across multiple dimensions and a distribution of taxonomy labels.

### Overall Architecture
The overall process is divided into four steps. First, stories are filtered from ConvoKit's reddit-corpus-small to construct the SSF-Corpus, retaining community and conversational contexts for each story. Second, GPT-4o is used to summarize subreddit purposes, norms, original posts, and ancestor/sibling comments, allowing the model to see the context that readers actually possess. Third, SSF-Generator generates reader response reasoning for each taxonomy dimension, such as author intent, causal explanations, future predictions, or aesthetic feelings. Fourth, SSF-Classifier maps the free-text reasoning to fine-grained taxonomy sub-labels, resulting in statistical and comparable community-level narrative representations.

### Key Designs
1.  **SSF-Taxonomy Reader Response Dimensions**:
    - **Function**: Deconstructs "how readers understand a story" into 10 actionable dimensions, including overall goal, narrative intent, author emotional response, causal explanation, prediction, character appraisal, moral, stance, narrative feeling, and aesthetic feeling.
    - **Mechanism**: Dimensions are organized from narrative theory, pragmatics, emotional psychology, and value theory, with sub-categories designed for each. For instance, narrative intent includes identity expression, meaning-making, emotional release, entertainment, argumentation, and support-seeking; the moral dimension uses high-level categories from Schwartz's Value Theory.
    - **Design Motivation**: This taxonomy ensures model outputs go beyond "this comment is sad" or "the themes are similar," mapping instead to interpretable social functions. It also provides a common coordinate system for community comparison, eliminating the need to define labels for every subreddit.

2.  **Context-Aware Reasoning Generation and Distillation**:
    - **Function**: Generates free-text reasoning that readers might produce for each story and each dimension.
    - **Mechanism**: GPT-4o serves as the teacher to generate up to 3 independent reasonings for each story-dimension pair in the SSF-Split-Corpus. Each reasoning uses dimension-specific templates but allows for free-form content. Subsequently, Llama3.1-8B-Instruct is distilled into the SSF-Generator via LoRA to reduce batch inference costs.
    - **Design Motivation**: Reader reactions are highly dependent on community norms and prior dialogue; direct zero-shot processing by small models tends to be shallow. Distillation transfers the reasoning capabilities of closed-source large models in complex contexts to open-source student models, while human plausibility surveys verify the reasonableness of the reasoning.

3.  **Reasoning Classification and ssf-sim Community Comparison**:
    - **Function**: Converts free-text reasoning into taxonomy labels and compares community narrative practices accordingly.
    - **Mechanism**: Recognizing that zero-shot performance for multi-label inference classification is poor, the authors use GPT-4.1 with k-shot prompting to generate classification references, then distill Llama3.1-8B-Instruct as the SSF-Classifier. Community similarity, $ssf\text{-}sim$, does not compare raw text embeddings but compares the reasoning from SSF-Generator and label distributions from SSF-Classifier.
    - **Design Motivation**: Free-text reasoning is information-dense but difficult to quantify; label distributions facilitate frequency, NPMI, entropy, and similarity analysis at the community level. Consequently, $ssf\text{-}sim$ can identify community pairs with "different themes but similar narrative functions."

### Loss & Training
The paper does not emphasize new training losses; the core strategy is teacher-student distillation. The generation side uses GPT-4o for reference reasoning with LoRA fine-tuning on Llama3.1-8B-Instruct. The classification side uses GPT-4.1 k-shot outputs as references to fine-tune the same series of open-source models for zero-shot multi-label classification. The SSF-Split-Corpus contains 1,778 stories, with a training/validation/test split of approximately 2/3, 1/6, and 1/6. To evaluate cross-community generalization, 10% of the stories in the validation and test sets are from 55 subreddits not present in the training set.

## Key Experimental Results

### Main Results

| Target | Setup | Key Metrics | Results | Description |
| :--- | :--- | :--- | :--- | :--- |
| SSF-Corpus | Filtered from 100 Reddit subreddits | Number of stories | 6,140 | Each includes story, dialogue, and community context |
| SSF-Split-Corpus | Train/Val/Test | Number of stories | 1,778 | Val/Test each contain 10% unseen subreddits |
| GPT-4o Reasoning Plausibility | Prolific Human Eval | Valid ratings | 4,239 ratings / 278 annotators | Representative US adult sample |
| SSF-Generator Output Plausibility | Human Eval | Plausible ratio | $\ge 94\%$ | Most reasoning considered contextually reasonable |
| SSF-Generator Output Likelihood | Human Eval | Somewhat/very likely ratio | $\ge 78\%$ | Not just reasonable, but highly probable |
| $ssf\text{-}sim$ Construct Validity | 50 story pair comparisons | Human agreement | 74% | Sentence-BERT baseline is 52% |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Full context SSF-Generator | Best alignment with human-validated GPT-4o teacher | Uses story, community, and dialogue context |
| W/o Community Context | Alignment decrease | Community norms and values affect reader interpretation |
| W/o Conversational Context | Significant decrease | Conversational context is particularly critical |
| Sentence-BERT semantic similarity | 52% human-aligned | Thematic similarity fails to capture functional similarity |
| $ssf\text{-}sim$ | 74% human-aligned | Based on reasoning and taxonomy labels; closer to pragmatic function |

### Key Findings
- The distribution of narrative intent shows that the most common intent is to "justify or challenge a belief" (40%); "clarification" and "emotional release" each account for 14%, while "identity" and "entertainment" each account for 10%. This indicates that online stories often serve argumentative and social negotiation functions rather than just entertainment.
- "Emotional support" in overall goals shows a strong correlation with "conveying a similar experience" in narrative intent (NPMI = 0.35), supporting the online mechanism of "using similar experiences to express empathy."
- SSF-Classifier performance on the test set is close to GPT-4.1 k-shot. Its Micro F1 scores surpass, match, or are within 0.05 of GPT-4.1 across 7/10 dimensions, with the gap never exceeding 0.1 in any dimension.
- Community comparisons reveal that communities with vast thematic differences, such as r/MakeupAddiction and r/buildapc, can have similar narrative functions. Conversely, even if themes are similar, such as r/funny versus r/news or r/politics, the narrative orientations can be entirely different.

## Highlights & Insights
- Moving "story understanding" from internal text to social context is the most valuable contribution of this paper. It does not merely ask what happened in the story, but how the story is used and received within a community.
- The taxonomy design is restrained: it covers 10 dimensions—broad enough—but maintains statistical utility through sub-labels. This design is more suitable for cross-community analysis than direct long-form LLM explanations.
- The $ssf\text{-}sim$ concept is highly transferable. Similarity in many tasks should not only compare content; for instance, customer service dialogues, medical narratives, forum help requests, or product reviews can all be compared via "communicative function" and "expected reaction."
- The paper places human validation on two levels: first validating the plausibility of generated reasoning, then validating the construct validity of the similarity metrics. This approach ensures the method is not just a collection of LLMs but possesses social science measurement awareness.

## Limitations & Future Work
- The context summarization uses iterative processes, which may lead to a cascaded loss of information, particularly for short contexts or stories requiring fine details.
- The current models and corpora are focused on English Reddit, and human evaluators are primarily US adults. Treating these results as universal reader responses may introduce cultural, gender, and ideological biases.
- The taxonomy is not exhaustive. Aspects such as narrative absorption, complex aesthetic emotions, reader identity differences, and dependencies between dimensions are simplified.
- The model assumes a "common reader response" within a community, but communities with high polarization, high professional barriers, or highly individualistic responses may not satisfy this assumption.
- Future work could build dimensions into a joint structural model, explicitly modeling dependencies among intent, stance, emotion, and morals instead of generating each dimension independently.

## Related Work & Insights
- **vs. Traditional Commonsense Reasoning**: Works like ATOMIC/COMET typically perform decontextualized reasoning based on short events; this paper embeds stories into communities and dialogues to reason about reader reception and social functions.
- **vs. Narrative Schema / Story Understanding**: Existing narrative NLP often focuses on plot, character psychology, or causal consistency. This paper focuses on why stories are told and how they are understood by readers.
- **vs. Sentence-BERT Semantic Similarity**: Semantic similarity excels at finding text with close themes; $ssf\text{-}sim$ can find community narratives with close functions even if themes differ.
- **Insights**: The path of "theoretical taxonomy + LLM distillation + human construct validation" is well-suited for high-level social semantic tasks, such as value conflict identification, community norm modeling, and stance/support function analysis in multi-party dialogues.

## Rating
- Novelty: ⭐⭐⭐⭐ Social narrative reception modeling and $ssf\text{-}sim$ are highly innovative, though the core model training is relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes human plausibility, expert annotation, and community analysis, though the scale of global similarity validation remains relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical motivation, taxonomy, modeling, and social science analysis are seamlessly connected.
- Value: ⭐⭐⭐⭐ Highly inspiring for NLP+CSS with strong reusability, though cross-platform and cross-cultural generalization requires further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)
- [\[ACL 2026\] VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection](veccisc_improving_confidence-informed_self-consistency_with_reasoning_trace_clus.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)

</div>

<!-- RELATED:END -->
