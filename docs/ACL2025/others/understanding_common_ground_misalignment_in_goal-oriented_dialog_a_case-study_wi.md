---
title: >-
  [Paper Note] Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs
description: >-
  [ACL 2025][Common Ground] This paper empirically reveals a significant correlation between the misalignment of common ground and task success by annotating "conversational friction" in Ubuntu IRC technical support dialogues, and finds that LLMs can identify explicit conversational friction but struggle with implicit friction requiring pragmatic or domain reasoning.
tags:
  - "ACL 2025"
  - "Common Ground"
  - "Conversational Friction"
  - "Goal-Oriented Dialogue"
  - "Ubuntu IRC"
  - "LLM Dialogue Understanding"
date: 2026-05-08
content_hash: f6bfbb878567faba
---

# Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs

**Conference**: ACL 2025  
**arXiv**: [2503.12370](https://arxiv.org/abs/2503.12370)  
**Code**: Yes ([https://github.com/styx97/cg-misalignment](https://github.com/styx97/cg-misalignment))  
**Area**: Other  
**Keywords**: Common Ground, Conversational Friction, Goal-Oriented Dialogue, Ubuntu IRC, LLM Dialogue Understanding

## TL;DR

This paper empirically reveals a significant correlation between the misalignment of common ground and task success by annotating "conversational friction" in Ubuntu IRC technical support dialogues, and finds that LLMs can identify explicit conversational friction but struggle with implicit friction requiring pragmatic or domain reasoning.

## Background & Motivation

Effective human dialogue relies on the maintenance of shared facts and beliefs (common ground) among participants. However, this maintenance is often implicit, presenting challenges in studying the relationship between common ground and dialogue success:

**Observability Issue**: When a conversation proceeds smoothly, it is difficult to know what the participants' common ground contains.

**Limitations of Prior Work**: Most studies infer common ground by restricting dialogue scenarios (e.g., Minecraft building tasks).

**Need for LLMs as Dialogue Partners/Intermediaries**: LLMs are increasingly used in dialogue, but whether they can track common ground remains unclear.

**Key Insight**: The authors chose an alternative angle—studying **communication failure** (misalignment) to catch a glimpse of the contents of common ground. When a participant's assumption is proven wrong (e.g., one party assumes the other knows how to use the `cd` command), the mismatch in common ground manifests as observable "conversational friction."

## Method

### Overall Architecture

1. Sample 200 dyadic technical support dialogues from the Ubuntu Dialog Corpus.
2. Annotate conversational friction and task success.
3. Analyze the relationship between friction and success.
4. Evaluate the ability of LLMs to identify friction.

### Key Designs

1. **Definition of Conversational Friction**:

    - Disruption in communication flow caused by inconsistent beliefs among participants regarding the contents of the common ground.
    - Distinguished from ordinary clarification questions—it is only considered friction when prior assumptions are violated.
    - Annotation format: Identifying the dialogue turn intervals containing friction + explaining the reasons.
    - Design Motivation: Friction serves as a "window" into common ground misalignment, through which participants' belief states can be inferred.

2. **Ubuntu-CG Dataset Construction**:

    - Sample 200 dyadic dialogues from the cleaned Ubuntu IRC corpus of Kummerfeld et al. (2019).
    - Upsample longer dialogues to study more diverse behaviors.
    - A total of 7,950 dialogue turns.
    - Annotated by three computer science undergraduate students ($18/hour, 80+ hours).

3. **Task Success Annotation (3-point scale)**:

    - 1 point: No progress at all.
    - 2 points: Some progress.
    - 3 points: Problem solved.

4. **Annotation of Grounding Acts** (on a subset of 70 dialogues containing friction):

    - **RequestRepair**: One party explicitly requests the other to resolve the friction after detecting it.
    - **Repair**: Either party resolves the friction via clarification.
    - Design Motivation: To understand the mechanisms by which friction is detected and resolved, and whether resolution impacts success rates.

5. **LLM Friction Detection Evaluation**:

    - Evaluate gpt-4o, gpt-4o-mini, and Llama-3.1-8b.
    - Evaluate LLM-generated explanations with and without technical term definitions.
    - Two evaluation metrics: Friction Found (lenient, matching any turn suffices) and Friction Overlap (strict, requiring interval overlap).

### Loss & Training

- This is a non-training-based study, primarily consisting of annotation and analysis.
- LLM evaluation uses zero-shot prompting.
- Annotation agreement for friction is measured using a modified F1 score (Best pairwise agreement A1-A2: Found=65.91, Overlap=25.86).
- Agreement for success annotation: Krippendorff's $\alpha = 0.58$.

## Key Experimental Results

### Relationship between Conversational Friction and Task Success (Table)

| Success | Average Length | Proportion with Friction | Avg. Friction Count when Present |
|---|---|---|---|
| 1 (No Progress) | 31.90 | 57.60% | 2.43 |
| 2 (Partial Progress) | 43.86 | 55.05% | 2.06 |
| 3 (Successful) | 40.45 | **50.84%** | 2.13 |

### LLM Friction Detection Performance (Table)

| Model | Friction Found (P/R/F1) | Friction Overlap (P/R/F1) | No. of Predictions |
|---|---|---|---|
| gpt-4o | 31.50/43.69/34.01 | 13.50/18.74/14.61 | 495 |
| gpt-4o + Term Explanations | 31.63/37.46/32.22 | 13.54/16.59/14.00 | 435 |
| gpt-4o-mini | 32.75/27.86/28.01 | 13.67/12.32/12.10 | 316 |
| Llama-3.1-8b | 16.72/47.28/22.53 | 6.87/18.72/9.14 | 1282 |

### Association between Grounding Acts and Success (Table)

| Progress Level | No. of Dialogues | Friction Instances (Repair/ReqRepair) | Ratio of Unanswered ReqRepair |
|---|---|---|---|
| Progress Made (2-3) | 49 | 102 (83/75) | 22.67% |
| No Progress (1) | 21 | 50 (38/36) | **30.56%** |

### Key Findings

1. **Successful dialogues contain less friction**: Only 50.84% of successful dialogues (score 3) contain friction, whereas 57.60% of no-progress dialogues (score 1) contain friction.
2. **Unanswered repair requests are more detrimental**: In no-progress dialogues, 30.56% of RequestRepair occurrences are left unanswered, which is higher than the 22.67% observed in dialogues making progress.
3. **Friction is positively correlated with dialogue length**: Dialogues containing friction have an average length of 49 turns (compared to only 29 turns for those without friction) because the resolution process requires "dialogue detours."
4. **LLMs can detect explicit friction but struggle with implicit friction**: LLMs perform well when friction is expressed through clear repair requests, but struggle significantly when friction is implicit (e.g., when the user does not understand but does not state it explicitly).
5. **Terminology explanations offer limited help**: Providing technical term explanations to LLMs does not significantly improve their friction detection performance.
6. **GPT-4o's explanations often disagree with humans**: Comparisons (as shown in Figure 2) indicate that GPT-4o tends to capture surface-level inconsistencies rather than deep-seated common ground mismatches.

## Highlights & Insights

- **An ingenious research perspective of "learning from failure"**: revealing common ground through communication failures, thereby bypassing the difficulty that common ground is not directly observable.
- The formal definition of **conversational friction** is highly valuable, grounding abstract common ground theory into annotate-able and evaluable concrete tasks.
- The correlation between the **unanswered rate of RequestRepair** and the success rate provides empirical evidence for an intuitive conclusion: multi-party participation in grounding is necessary for task success.
- The discovery of **LLMs' weaknesses in implicit pragmatic reasoning** has significant implications for assessing the reliability of LLMs as dialogue agents.
- The choice of the Ubuntu IRC dataset is highly appropriate: naturally occurring goal-oriented dialogues, building common ground from scratch, text-only, and multi-turn interactions.

## Limitations & Future Work

- The sample size is relatively small, with only 200 dialogues (7,950 turns).
- Inter-annotator agreement is moderate ($\alpha = 0.58$ for success, Found F1 = 65.91 for friction), reflecting the subjectivity of the task.
- The study is confined to the Ubuntu technical support scenario; its generalizability to other domains (e.g., medical support, customer service) remains unknown.
- No automated system for friction detection and resolution has been established yet.
- LLM evaluation is restricted to zero-shot; few-shot learning or fine-tuning might yield significant improvements.
- The dialogue data is relatively old (roughly 10 years ago), and some terminology and usage may be outdated.

## Related Work & Insights

- It directly inherits the theoretical framework of Discourse Units and grounding acts from Traum & Allen (1992).
- It provides an empirical contrast to the collaborative common ground maintenance theory of Clark & Brennan (1991).
- It complements the Minecraft dialogue study by Narayan-Chen et al. (2019) by investigating common ground in a more natural environment.
- **Insight**: Dialogue systems need to go beyond surface-level semantic understanding to track the dynamic changes of common ground at the pragmatic level.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The perspective of studying common ground from communication failure is highly novel, and the formalization of conversational friction provides theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐ — The annotation analysis is in-depth, but the sample size is small, and the LLM evaluation is limited to zero-shot.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Tight integration of theory and empirical evidence, vivid case studies, and rich interdisciplinary perspectives.
- **Value**: ⭐⭐⭐⭐ — Offers important insights for both dialogue understanding and LLM evaluation domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[ACL 2025\] SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning](sorft_issue_resolving_with_subtask-oriented_reinforced_fine-tuning.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)

</div>

<!-- RELATED:END -->
