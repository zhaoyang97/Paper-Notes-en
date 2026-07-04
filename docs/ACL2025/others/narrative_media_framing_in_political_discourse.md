---
title: >-
  [Paper Note] Narrative Media Framing in Political Discourse
description: >-
  [ACL 2025][Narrative Framing] Integrates narratological theory with media framing analysis to propose a structured narrative framing analysis framework composed of three components: characters (hero/villain/victim), conflict/resolution, and cultural stories. The effectiveness and transferability of this framework are validated across two domains: climate change and COVID-19.
tags:
  - "ACL 2025"
  - "Narrative Framing"
  - "Media Analysis"
  - "Political Discourse"
  - "LLMs"
  - "Climate Change"
date: 2026-05-08
content_hash: ee9787abed170ab8
---

# Narrative Media Framing in Political Discourse

**Conference**: ACL 2025  
**arXiv**: [2506.00737](https://arxiv.org/abs/2506.00737)  
**Code**: [Yes (GitHub)](https://github.com/julia-nixie/narratives)  
**Area**: Others  
**Keywords**: Narrative Framing, Media Analysis, Political Discourse, LLMs, Climate Change

## TL;DR

Integrates narratological theory with media framing analysis to propose a structured narrative framing analysis framework composed of three components: characters (hero/villain/victim), conflict/resolution, and cultural stories. The effectiveness and transferability of this framework are validated across two domains: climate change and COVID-19.

## Background & Motivation

Narrative framing is a powerful rhetorical device in media coverage that guides readers' understanding and judgment of complex topics by "telling a story." For example, in climate change coverage, an article can portray environmental activists as "heroes" or "villains," thereby conveying completely different messages.

Existing framing analysis in NLP primarily relies on **topic-like frames** (e.g., "economic frame," "conflict frame"). However, this taxonomy is too coarse-grained to capture the subtle nuances of different narrative strategies under the same topic. For example, two articles under the "economic frame" regarding climate change could emphasize opposite narratives: one might stress the economic necessity of fossil fuels, while the other advocates for the economic benefits of climate action.

Although the NLP field has investigated narrative elements (such as characters and events), these studies are either domain-specific or lack connections to the core mechanisms of framing analysis (issue ambiguity, schema activation). By integrating the Narrative Policy Framework (NPF) with Entman's framing theory, this paper constructs a structured and operationalized narrative framing analysis system.

## Method

### Overall Architecture

This paper proposes three core components that collectively define a narrative frame:

1. **Characters**: Who are the heroes, villains, and victims? Which character is the focus?
2. **Conflict & Resolution**: Are the characters fueling the conflict or driving a resolution?
3. **Cultural Stories**: To which broader cultural values does the article map?

### Key Designs

#### 1. **Character System (Hero/Villain/Victim)**

Core Idea: Ambiguity of issues is resolved by assigning prototypical roles (hero/villain/victim) to entities. The reader's interpretation of an article depends on whether a specific entity is framed as a hero (whose actions are evaluated as beneficial), a villain (harmful), or a victim.

Design Motivation:
- Distinguish **major roles** from **minimally mentioned entities**, capturing only the single most central entity for each role type.
- Abstract specific individuals/organizations into **stakeholder categories** (e.g., government, environmental activists, general public), enabling cross-textual comparison.
- Introduce the concept of **focus**: Even if two articles share the same heroes and villains, a difference in focus leads to different narrative frames—a "heroic" frame focuses on praising the hero, while a "blaming" frame focuses on criticizing the villain.

#### 2. **Conflict & Resolution**

The actions of the characters are defined in terms of four attitudes:
- **Fuel conflict**: Engaging in behaviors that aggravate the problem.
- **Fuel resolution**: Engaging in behaviors that resolve the problem.
- **Prevent conflict**: Opposing behaviors that aggravate the problem.
- **Prevent resolution**: Opposing behaviors that resolve the problem.

This abstraction allows the component to transfer across domains—as it bypasses the need to define domain-specific events or behaviors and instead evaluates the directional attitude.

#### 3. **Cultural Stories**

Based on Douglas's (2007) cultural theory, narratives are mapped to four cultural values:
- **Fatalist**: People are at the mercy of uncontrollable forces.
- **Hierarchical**: People are bound by social norms and external controls (e.g., government).
- **Individualistic**: Social ties are loose, and external control is unnecessary.
- **Egalitarian**: Collective action, opposing external control.

### Annotation Process and Data

- Manually annotated 100 randomly sampled articles from over 1,000 US news articles on climate change.
- Employed **component-based annotation** (annotating characters, focus, conflict, and cultural stories first, and then mapping them to narrative frames), which yielded much higher annotation agreement compared to directly selecting narrative labels (63% vs 37%).
- Defined a final set of 16 structurally distinct narrative frames.

## Key Experimental Results

### Main Results: Zero-Shot Narrative Frame Prediction with LLMs

| Task | GPT-4o | o1 | Mixtral | Llama | Gemini | Sonnet | Baseline |
|------|--------|-----|---------|-------|--------|--------|----------|
| Hero (10 classes) | 0.325 | 0.363 | 0.237 | 0.271 | 0.326 | 0.353 | 0.079 |
| Villain (10 classes) | 0.454 | 0.527 | 0.073 | 0.156 | 0.292 | **0.530** | 0.080 |
| Focus (3 classes) | 0.656 | 0.718 | 0.402 | 0.568 | 0.635 | **0.688** | 0.231 |
| Conflict (4 classes) | 0.332 | **0.549** | 0.353 | 0.379 | 0.361 | 0.399 | 0.135 |
| Cultural Story (4 classes) | 0.574 | **0.595** | 0.431 | 0.449 | 0.482 | 0.561 | 0.190 |
| Narrative (16 classes) | 0.258 | 0.330 | 0.171 | 0.181 | 0.319 | **0.339** | 0.021 |

(The evaluation metric is Macro F1)

### Ablation Study: Impact of Adding Structural Labels on Narrative Framing Prediction

| Model | Unstructured | + Predicted Labels | + Oracle Labels |
|------|--------|-----------|--------------|
| GPT-4o | 0.258 | ~0.40 | ~0.55 |
| Gemini | 0.319 | ~0.33 | ~0.43 |
| Sonnet | 0.339 | ~0.38 | ~0.45 |

### Key Findings

1. **No single model outperforms all others on every task**: Sonnet and o1 are the strongest overall, though each has its strengths and weaknesses.
2. **Structured information significantly improves narrative prediction**: Incorporating structural labels (e.g., characters, focus) boosts GPT-4o's F1 score from 0.258 to approximately 0.55 (oracle = approx. 0.55), an improvement that far exceeds the gains from enhancing model reasoning capabilities alone.
3. **Weak correlation between narrative frames and generic frames**: The same generic topic frame (e.g., "economy") can map to multiple distinct narrative frames, demonstrating that narrative analysis provides much finer-grained insights.
4. **Strong correlation between political bias and narrative components**: Right-leaning media overwhelmingly employ "prevent resolution" and "individualistic cultural stories" components, which are entirely absent in left-leaning media.
5. **Cross-domain transferability**: Unsupervised application of the framework to political speeches on COVID-19 yields results consistent with existing political science research, validating its generalizability.

## Highlights & Insights

- **Bridging Social Science and NLP**: This study operationalizes the NPF (Narrative Policy Framework) into computable components. This theory-driven research paradigm yields deeper insights than purely data-driven efforts.
- **Wisdom of Component-Based Annotation**: Decomposing the complex 16-class narrative framing task into smaller sub-tasks with 3 to 10 classes during annotation dramatically reduces cognitive load and increases inter-annotator agreement. This methodology provides a strong reference for other complex annotation campaigns.
- **Structure > Reasoning Capability**: Experiments demonstrate that giving the model structured labels is more effective than upgrading to more powerful reasoning models (such as o1), offering valuable insights for prompt engineering.
- **Insights from COVID-19 Analysis**: Morrison favored hierarchical narratives (government-driven), Merkel favored egalitarian narratives (collective action), and Johnson was the only leader who relied on individualistic narratives. These observations align closely with findings in political science literature.

## Limitations & Future Work

1. The dataset consists of only 100 articles (prioritizing annotation depth over breadth), which is relatively small.
2. LLMs show poor performance in directly predicting the 16-class narrative frames (even reducing the task to 3 classes yielded limited improvements), indicating significant room for refinement.
3. Evaluation is currently restricted to zero-shot settings; few-shot learning and fine-tuning remains to be explored.
4. The investigation covers only US English media; cross-lingual and cross-cultural narrative framing analysis remains an important future direction.
5. The cultural story component is not yet incorporated into the structured prompts (which currently only use characters and focus), presenting a path for future optimization.

## Related Work & Insights

- The Narrative Policy Framework (NPF) provides the theoretical foundation for political narrative analysis.
- Entman's (1993) framing theory defines four functions of a frame (problem definition, causal attribution, moral evaluation, and treatment recommendation).
- Douglas's Grid-Group Cultural Theory offers a systematic method for mapping narratives to macro-level values.
- Compared to purely data-driven NLP framing analysis, the theory-driven approach presented in this work is better suited for tackling complex social science problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Formally operationalizes the three core components of narrative framing for the first time, striking an excellent balance between theoretical depth and practical utility.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation covering 6 LLMs across 7 tasks, augmented by cross-domain validation, but the dataset size remains limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Features deep theoretical discussions, highly logical framework designs, and outstanding interdisciplinary integration.
- **Value**: ⭐⭐⭐⭐ — Highly valuable reference for computational social science and political communication studies, with the framework showing broad potential for diverse applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)

</div>

<!-- RELATED:END -->
