---
title: >-
  [Paper Note] USDC: A Dataset of User Stance and Dogmatism in Long Conversations
description: >-
  [ACL 2025 (Findings)][stance detection] This paper constructs USDC, the first user-level long-conversation dataset for stance and dogmatism, containing 764 multi-user Reddit conversations (across 22 subreddits). Using a majority voting scheme over six configurations ({Mistral Large, GPT-4} $\times$ {zero/one/few-shot}), stances are annotated on a 5-level scale and dogmatism on a 4-level scale. Baseline performances are established using fine-tuning/instruction-tuning on 7 SLM…
tags:
  - "ACL 2025 (Findings)"
  - "stance detection"
  - "dogmatism"
  - "conversation"
  - "Reddit"
  - "LLM annotation"
  - "opinion dynamics"
date: 2026-05-08
content_hash: e8002de7b1e3a339
---

# USDC: A Dataset of User Stance and Dogmatism in Long Conversations

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2406.16833](https://arxiv.org/abs/2406.16833)  
**Code**: [GitHub](https://github.com/mounikamarreddy/USDC)  
**Area**: Others  
**Keywords**: stance detection, dogmatism, conversation, Reddit, LLM annotation, opinion dynamics

## TL;DR
This paper constructs USDC, the first user-level long-conversation dataset for stance and dogmatism, containing 764 multi-user Reddit conversations (across 22 subreddits). Using a majority voting scheme over six configurations ({Mistral Large, GPT-4} $\times$ {zero/one/few-shot}), stances are annotated on a 5-level scale and dogmatism on a 4-level scale. Baseline performances are established using fine-tuning/instruction-tuning on 7 SLMs.

## Background & Motivation
**Background**: Analyzing fluctuations in user opinions is crucial for personalized recommendation, public opinion monitoring, and political analysis. Existing stance detection datasets (such as SPINOS, MT-CDS, and Twitter-stance) focus on the post level, treating each post as an independent sample.

**Limitations of Prior Work**: (a) They do not track the evolution of opinions for the same user across multiple posts—analyzing posts in isolation cannot capture whether a user changes their stance; (b) Human annotation of long conversations is extremely time-consuming—annotators must read the entire thread to extract a user's perspective; (c) Subtle shifts in opinions are highly difficult to capture, as users often change their stances implicitly.

**Key Challenge**: How to annotate user opinion dynamics in long conversations at scale? Human annotation is costly, suffers from low quality, and is limited by the annotators' domain knowledge.

**Key Insight**: LLM annotation—utilizing two powerful LLMs under three in-context learning configurations, creating 6 settings in total, with majority voting providing the final label. LLMs do not suffer from fatigue and can retain long-context memory, potentially outperforming humans in comprehending lengthy discussions.

**Core Idea**: A paradigm shift from post-level to user-level analysis—tracking the opinion trajectory of the same user throughout an entire conversation.

## Method

### Dataset Construction

1. **Reddit Data Collection**:
    - Source: 22 subreddits, 2019 data, initially crawling 3,619 conversations.
    - Quality filtering: textual content + non-deleted/non-removed posts + 20-70 comments + at least two active users covering approximately 50% of the comments.
    - Final: 764 long conversations, with the top-2 most active users extracted from each conversation.

2. **LLM Annotation Pipeline**:
    - Convert conversations into a nested JSON format to preserve Reddit's hierarchical structure.
    - System prompt contains definitions of stance and dogmatism, annotation guidelines, and label explanations.
    - 6 configurations: {Mistral Large, GPT-4} $\times$ {zero-shot, one-shot, few-shot}.
    - Final annotation: Majority voting; in case of no clear majority, GPT-4 few-shot is used as the tie-breaker.

3. **Two Annotation Tasks**:
    - **Stance Detection (5 levels)**: Strong Support (SOIF), Somewhat Support (SIF), Not Inferable (SNI), Somewhat Against (SGA), Strong Against (SOA)—evaluated for each post-level turn.
    - **Dogmatism Recognition (4 levels)**: Open to Dialogue, Firm but Open, Flexible, Deeply Rooted—evaluated globally for each user.

4. **Human Verification**: 200 test conversations annotated by 3 human annotators.
    - LLM vs. Human IAA: stance $\kappa=0.49$, dogmatism $\kappa=0.50$.
    - Inter-human IAA: stance $\kappa=0.57$, dogmatism $\kappa=0.52$.

### SLM Fine-Tuning / Instruction-Tuning
- 7 models: LLaMA-2-7B/chat, LLaMA-3-8B/instruct, Falcon-7B/instruct, Vicuna-7B-v.1.5.
- 4-bit quantization + LoRA fine-tuning.
- Stance: individual posts treated as independent samples; Dogmatism: concatenating all posts of a user into a single sample.

## Key Experimental Results

### Classification Performance (Weighted F1)

| Method | Stance F1 | Dogmatism F1 |
|------|---------|--------|
| Un-tuned Baseline | ~31% | ~40% |
| Fine-tuning Best (Majority Voting) | 54.9 | **51.4** |
| Instruction-tuning Best (Majority Voting) | **56.2** | 49.2 |

### Ablation: Different Annotation Sources as Training Labels

| Annotation Source | Stance F1 Range | Dogmatism F1 Range |
|--------|-----------|----------|
| GPT-4 ZS/OS/FS | 51-55 | 42-50 |
| Mistral Large ZS/OS/FS | 34-50 | 37-50 |
| **Majority Voting** | **54-56** | **43-51** |

### Key Findings
- **Majority voting annotation is the most stable**: Across all SLMs, models trained on majority voting annotations show the most consistent performance.
- **Stance is suited for instruction-tuning (56.2), while dogmatism is suited for fine-tuning (51.4)**—the optimal training strategy is task-dependent.
- **LLaMA-3 family performs best**: LLaMA-3-8B-instruct achieves optimal or near-optimal performance on both tasks.
- **"Lost in the middle" effect is weak**: LLMs exhibit stable performance in long conversation annotation, unaffected by potential loss of information in the middle of long threads.
- **Transfer learning is effective**: SLMs fine-tuned on USDC achieve comparable or superior performance on SPINOS compared to prior work, validating the transferability of the dataset.
- **GPT-4 few-shot provides the highest quality annotations**: Showing the best agreement with humans, it is utilized as the tie-breaker in majority voting.

## Highlights & Insights
- **Tracking opinion evolution from post-level to user-level**—closer to real-world dynamics. Prior datasets treated each post independently, losing information regarding changes in user stance.
- **6-configuration majority voting is more robust than a single LLM**: Ensembling across models and settings reduces single-model bias.
- **LLM long-conversation annotation outperforms human performance**: Being fatigue-free and possessing long-term memory gives LLMs an inherent advantage in understanding long threads.
- **Broad dataset coverage**: 22 subreddits cover various topics including politics, religion, culture, and economy.

## Limitations & Future Work
- **Moderate annotation agreement ($\kappa \approx 0.5$)**: Opinion annotation is inherently subjective, but the agreement is comparable to prior datasets (e.g., $0.44$ in Fast & Horvitz 2016).
- **English Reddit only**: Fails to cover discussion characteristics of other languages and platforms (e.g., Twitter, Weibo).
- **Tracking only top-2 active users** (covering 47% of posts): This neglects opinions from other participants.
- **Occasional JSON parsing errors**: Approximately 15 cases of LLM outputs mismatching the required format needed manual corrections.

## Related Work & Insights
- **vs. SPINOS (Sakketou et al.)**: Post-level independent annotation vs. USDC's user-level conversational trajectory.
- **vs. Fast & Horvitz (2016)**: Random post sampling + 200–300 character limit + non-public dataset vs. USDC's complete conversation + publicly available.
- **vs. MT-CDS (Niu et al.)**: Multi-target, multi-turn stance extraction vs. USDC's user-level tracking of opinion dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐ User-level stance tracking + LLM majority voting annotation pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 SLM baselines + 200 human verifications + transfer learning + comparison of multiple annotation sources.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly illustrates changes in user opinions with a concrete case.
- Value: ⭐⭐⭐⭐ A practical resource for social computing, holding direct value for public opinion analysis and user modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)
- [\[ACL 2025\] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)
- [\[ACL 2025\] You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations](you_need_to_mimic_to_get_fame_solving_meeting_transcript_scarcity_with_a_multi-a.md)

</div>

<!-- RELATED:END -->
