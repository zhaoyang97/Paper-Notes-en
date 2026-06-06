---
title: >-
  [Paper Note] PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media
description: >-
  [ACL2026][Multilingual & Machine Translation][Content moderation] PluRule models Reddit community moderation as a multiple-choice task: "given a comment and its context…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Content moderation"
  - "community rules"
  - "multilingual benchmark"
  - "multimodal VLM"
  - "Reddit"
date: 2026-05-08
content_hash: cc17ab9941515ede
---

# PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media

**Conference**: ACL2026  
**arXiv**: [2605.17187](https://arxiv.org/abs/2605.17187)  
**Code**: https://github.com/osome-iu/PluRule  
**Area**: Multilingual Content Governance / Social Media Moderation  
**Keywords**: Content moderation, community rules, multilingual benchmark, multimodal VLM, Reddit

## TL;DR
PluRule models Reddit community moderation as a multiple-choice task: "given a comment and its context, select which community rule was violated or if no rules were broken." It constructs a benchmark covering 1,989 communities, 2,885 rules, and 9 languages, showing that even GPT-5.2 high reasoning with full context achieves only approximately 57.6% accuracy.

## Background & Motivation
**Background**: Social platforms have long relied on human moderators and automated detection systems to handle illegal content, hate speech, harassment, and low-quality content. Many automated datasets treat moderation as a globally uniform labeling task, such as toxicity, hate speech, or harassment.

**Limitations of Prior Work**: Community governance rules are not globally uniform. The same statement might be an encouraged joke in r/RoastMe but a violation of civility in other communities; self-promotion is spam in most communities but potentially necessary content in showcase-oriented ones. Unified moderation models tend to impose mainstream norms on minority or non-English communities.

**Key Challenge**: Community moderation requires models to understand local rules, discussion context, community purpose, and implicit norms, whereas existing models excel at identifying cross-community generic violation types. A model's ability to detect "general incivility" does not imply it can judge whether "this comment violates Rule 4 of this specific subreddit."

**Goal**: The authors aim to construct a pluralistic moderation benchmark that tasks models with fine-grained rule identification across thousands of communities and rules in multilingual, multimodal contexts, evaluating whether existing VLMs can truly assist community self-governance.

**Key Insight**: The paper utilizes publicly available moderator comments on Reddit. Many moderators state which rule was violated when deleting or flagging content. The authors use these comments to perform semantic matching with current rule texts and pair them with un-moderated "compliant comments" from the same submission to form contrastive multiple-choice samples.

**Core Idea**: Upgrade content moderation from "is it a violation" binary classification to a multiple-choice task of "which community rule was violated," requiring the model to simultaneously consider rules, comments, discussion threads, the original post, user anonymity indicators, and images.

## Method
PluRule does not propose a new moderation model but rather a benchmark that closely mirrors the decision space of real community moderation. Each sample contains one violating comment and one compliant comment from a similar context within the same submission. After viewing the community rule list and context, the model must select an answer from all rules plus a "No rules broken" option.

### Overall Architecture
Data construction follows five stages. First, moderator comments are extracted from Pushshift Reddit archives, and subreddit rules, languages, and NSFW status are collected via the Reddit API. Second, multilingual embeddings are used to match moderator comments to current subreddit rules. Third, violating and compliant threads are constructed, and submission images are downloaded as multimodal context. Fourth, LLMs verify if the match truly represents rule enforcement. Fifth, the data is split into train/val/test by subreddit instances, with semantic clustering performed on subreddits and rules.

During evaluation, models progressively receive five cumulative context levels: Comment Only, +Discussion, +Submission, +User, +Images. All levels include the subreddit description and the full rule set. The output is first generated freely, followed by "Final Choice:" to extract the final option.

### Key Designs
1. **Multiple-Choice Community Rule Identification**:
	- **Function**: Simulates the real choices faced by human moderators rather than coarse-grained binary classification.
	- **Mechanism**: Options for each comment consist of all rules for that subreddit plus "No rules broken," shuffled using a deterministic seed based on the comment ID to prevent position bias. The correct answer for a violating comment is a specific rule, while for a compliant comment, it is "No rules broken."
	- **Design Motivation**: Real moderation is not just about judging "badness"; it requires knowing which local rule was violated to provide explanations, enforcement, and grounds for subsequent appeals.

2. **Constructing High-Quality Labels from Public Moderation Traces**:
	- **Function**: Converts historical moderator comments into rule-level supervision.
	- **Mechanism**: Distinguished moderator comments are extracted from approximately 15B comments across 40k subreddits. After filtering bots and NSFW content, 17,468 subreddits, 131,400 rules, and about 9M moderator comments remain. Qwen3-Embedding-8B encodes both comments and rules, calculating similarity within communities. The matching threshold is set at the 99.2 percentile (0.79), and the ambiguity threshold is set at the 98 percentile (0.75).
	- **Design Motivation**: Rules are rewritten and renumbered over time, making direct regex matching unreliable. Semantic matching handles variations in phrasing, while ambiguity filtering reduces noise where one comment points to multiple rules.

3. **Contrastive Instances, Verification, and Semantic Clustering**:
	- **Function**: Forces the model to distinguish violations from compliance in similar contexts and supports analysis by community/rule type.
	- **Mechanism**: Each violating thread is paired with a compliant thread (no moderator action) under the same submission, prioritizing branches with more shared ancestors, similar depth, and lower scores. Qwen3-30B-A3B-Instruct verifies if matched rules are "stating a violation," with an 82.1% verification rate. Finally, UMAP + HDBSCAN cluster subreddit and rule embeddings, which are manually corrected after receiving candidate labels from Qwen3-30B-A3B-Thinking.
	- **Design Motivation**: Contrastive samples prevent models from relying solely on topic or community priors; clustering reveals which community types and rule types are the most difficult.

### Loss & Training
PluRule is a benchmark and does not involve training new models. Evaluated models include Instruct and Thinking versions of Qwen3-VL-4B/8B/30B, and GPT-5.2 low/high reasoning. Qwen models use temperature 0 and seed 0. The metric is test accuracy, with 95% confidence intervals calculated via 100k bootstrap resamples; the paper reports that all accuracy 95% CIs do not exceed $\pm 1.3\%$, and recall table CIs do not exceed $\pm 1.9\%$.

## Key Experimental Results

### Main Results
| Split | Instances | Comments | Images | Subreddits / Clusters | Rules / Clusters | Languages |
|-------|-----------|----------|--------|------------------------|------------------|-----------|
| Train | 9,155 | 51,968 | 2,077 | 861 / 25 | 1,336 / 27 | 9 |
| Val | 1,382 | 7,631 | 376 | 537 / 25 | 586 / 27 | 9 |
| Test | 2,834 | 13,076 | 1,190 | 1,989 / 25 | 2,039 / 27 | 9 |
| Total | 13,371 | 72,675 | 3,643 | 1,989 / 25 | 2,885 / 27 | 9 |

### Ablation Study
| Model / Variant | Comment Only | +Discussion | +Submission | +User | +Images | Description |
|-------------|--------------|-------------|-------------|-------|---------|------|
| Qwen3-VL-4B Instruct | 49.6 | 49.2 | 48.3 | 48.9 | 48.4 | Mostly below or near 50% baseline |
| Qwen3-VL-8B Instruct | 51.0 | 50.7 | 49.2 | 50.0 | 49.8 | No stable improvement with 8B scaling |
| Qwen3-VL-30B Instruct | 50.2 | 51.0 | 51.1 | 52.4 | 52.3 | Largest Qwen only slightly above baseline |
| GPT-5.2 Low | 54.1 | 55.3 | 56.8 | 57.4 | 57.4 | Strong closed-source model significantly better but still limited |
| GPT-5.2 High | 55.0 | 56.2 | 57.3 | 57.7 | 57.6 | Full context only ~2.6 higher than comment-only |
| Baseline | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | Always predicts "No rules broken" |

### Key Findings
- GPT-5.2 high reasoning achieves ~57.6% accuracy with full context, only 7.6 percentage points above the 50% baseline. The improvement from comment-only to full context is only ~2.6 to 2.7 points, suggesting context is underutilized.
- Qwen Thinking variants often perform worse than Instruct variants, and the difference between GPT-5.2 high and low is insignificant, indicating that "more reasoning" does not automatically solve community rule understanding.
- By rule type, models perform better on universal rules: civility (~69%), language (~66%), self-promotion (~63%). Types below baseline include low-effort (43%), relevance (44%), and evidence-based (47%).
- Weighted accuracy (setting violating:compliant at 2:1) dropped across the board. GPT-5.2 high full context fell from 57.6% to 52.6% because models are better at recalling compliant comments than violating ones.
- Label quality was manually verified: in 100 English samples, the pipeline showed 96% overall agreement with human-established ground truth. Full agreement samples were 85/85 correct, majority agreement samples 8/12, and arbitrated samples 3/3.

## Highlights & Insights
- PluRule's task definition captures the core difficulty of content moderation: rules are locally defined, and violation is not a fixed attribute of content but a relationship between content, rules, and context.
- The use of paired violating/compliant threads is ingenious. It minimizes the model's ability to speculate based on submission topics or community labels, forcing it to distinguish specific comments within similar discussions.
- The results act as a reality check for using stronger VLMs for automated community rule moderation. Even if GPT-5.2 can handle multimodal context, it struggles to understand vast local norms, especially context-dependent rules like relevance, low-effort, or evidence-based.
- Semantic clustering provides more than just analysis; it offers a transfer learning problem: whether moderation capabilities can be transferred from similar communities or rules rather than learning each subreddit from scratch.

## Limitations & Future Work
- Data only covers moderation actions where a public moderator comment was left. Private message moderations, silent deletions, shadow-bans, and rapid removals of severe violations are invisible; thus, data might be biased toward minor violations or communities more willing to explain publicly.
- Reddit is dominated by English communities. While the benchmark counts 9 languages, conclusions may not generalize to different platforms, governance structures, or non-English dominant communities.
- Historical moderator comments span 2005 to 2023, while rules were fetched in November 2025 via the Reddit API. Semantic matching mitigates rule rewrites but cannot fully address new rules, deletions, or changes in meaning.
- Some violations require information absent from the dataset, such as ban evasion, repeat violation history, or multi-post behavior. Excluding this history for privacy is reasonable but limits the benchmark's completeness.

## Related Work & Insights
- **vs toxicity / hate speech datasets**: Traditional datasets focus on globally unacceptable content; PluRule focuses on community rules and context, making it better suited for pluralistic moderation.
- **vs Park et al. rule type data**: Prior work often collapses numerous community rules into a few coarse categories; PluRule preserves specific subreddit rules, requiring models to select within a local rule set.
- **vs He et al. binary rule judgment**: Binary tasks only judge if a specific rule was violated; PluRule provides all rules and a no-violation option simultaneously, identifying a workflow closer to actual moderators.
- **Insights**: For automated moderation systems to serve autonomous communities, they may need to retrieve historical moderation cases, learn community precedents, and provide evidence for rule interpretation rather than relying solely on generic safety classifiers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing pluralistic moderation as community rule multi-choice identification and building a multilingual multimodal Reddit benchmark is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers data construction, label verification, context levels, model sizes, and reasoning variants; lacks direct upper-bound comparison with human moderators on the full task.
- Writing Quality: ⭐⭐⭐⭐ Data pipeline and result interpretations are clear, and limitations are honestly addressed; many results are in the appendix, leaving the main text with only partial breakdown analysis.
- Value: ⭐⭐⭐⭐⭐ Important for content governance, community autonomy, and AI moderation evaluation, serving as a reminder that automated moderation cannot simply apply a uniform norm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
