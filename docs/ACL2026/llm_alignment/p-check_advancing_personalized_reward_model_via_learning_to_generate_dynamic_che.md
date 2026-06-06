---
title: >-
  [Paper Note] P-Check: Advancing Personalized Reward Model via Learning to Generate Dynamic Checklist
description: >-
  [ACL2026][LLM Alignment][Personalized reward model] P-Check transforms personalized reward modeling from "cramming user history into a judge" to "generating a weighted dynamic evaluation checklist for the current user an…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Personalized reward model"
  - "dynamic checklist"
  - "LLM-as-a-Judge"
  - "preference contrastive learning"
  - "interpretable alignment"
date: 2026-05-08
content_hash: 469fab198b289443
---

# P-Check: Advancing Personalized Reward Model via Learning to Generate Dynamic Checklist

**Conference**: ACL2026  
**arXiv**: [2601.02986](https://arxiv.org/abs/2601.02986)  
**Code**: Marked as CODE on the paper page, specific URL not resolved in cache  
**Area**: Interpretability / Personalized Reward Model  
**Keywords**: Personalized reward model, dynamic checklist, LLM-as-a-Judge, preference contrastive learning, interpretable alignment

## TL;DR
P-Check transforms personalized reward modeling from "cramming user history into a judge" to "generating a weighted dynamic evaluation checklist for the current user and query, then using it to guide reward scoring." It significantly outperforms persona, memory retrieval, and fine-tuned reward model baselines on personalized preference prediction and downstream generation tasks across PRISM, Arena, and BESPOKE datasets.

## Background & Motivation
**Background**: RLHF and preference optimization typically rely on reward models to compress human feedback into a scalar score. As LLMs are increasingly used as personal assistants, reward models must judge not only "universal quality" but also "conformity to a user's taste, tone, information density, values, and task habits." Existing personalized reward modeling usually provides user history as context, retrieved memory, user embeddings, personas, or few-shot examples to the reward model.

**Limitations of Prior Work**: User signals in these methods are largely static and implicit. A static persona might describe a user as "liking specific, structured, and analytical responses," but it does not necessarily inform the judge which specific conditions to check for the current query. Implicit user embeddings are harder to interpret and provide no clear constraints on the judge for OOD queries. Preliminary experiments in the paper show that even when provided with user history, LLMs struggle to select the checklist that truly explains user preferences between an oracle checklist and a counter-preference checklist; however, if provided with an explicit checklist, the LLM’s personalized judgment improves significantly.

**Key Challenge**: Personalized judgment involves both long-term user preferences and the current task context. Long-term preferences provide stability, while the current query determines which inclinations are truly relevant. Treating user history as a fixed profile ignores preference drift between different tasks for the same user. Relying solely on free-text rationales often mixes objective quality errors with subjective preferences.

**Goal**: The authors aim to train a plug-and-play checklist generator: given a user history summary and the current query, it outputs the personalized evaluation criteria required for the current judgment; these criteria then guide any LLM-as-a-Judge to predict rewards. This involves three sub-problems: constructing checklist supervision from preference pairs, distinguishing criteria that truly determine user choices from superficial ones, and ensuring generated checklists translate stably into scalar rewards during inference.

**Key Insight**: The authors view "human evaluation" as a process of temporarily constructing evaluation standards. When facing new problems, humans do not necessarily recall all historical preferences explicitly but form a few actionable standards based on the context, such as "give me concrete facts," "consider historical background," or "don't be too conservative in tone." Such standards are closer to an executable interface for a judge than a persona.

**Core Idea**: Use dynamically generated personalized checklists with importance tags as an intermediate representation for reward models to judge personalized preferences, replacing static personas or implicit user vectors.

## Method
The main pipeline of P-Check is clear: first, extract checklists offline from training users' preference data; then, use contrastive sampling and saliency scoring to assign importance tags to each criterion in the checklist; finally, train a small checklist generator. During inference, the generator requires only the general preference of the user and the current query to produce a checklist, which guides an off-the-shelf LLM judge to calculate the final reward via criterion-wise scoring.

### Overall Architecture
The input consists of a user's history $H_u$, the current query $q$, and a candidate response $y$. Traditional reward models estimate $r(y \mid H_u, q)$ directly. P-Check inserts an explicit checklist $C_{u,q}$ in the middle, defining the reward as a judgment determined collectively by the candidate response, query, user history, and checklist: $r \sim P_\theta(\cdot \mid y, q, H_u, C_{u,q})$.

The training phase consists of four steps.

Step 1: Generate a general preference $GP_u$ from user history $H_u$. This summary emphasizes stable preferences in content, tone, reasoning, and structure rather than just compressing history.

Step 2: Use $GP_u$, current query $q$, chosen response $y^+$, and rejected response $y^-$ to prompt a strong LLM to generate candidate checklists. The prompt requires the checklist to appear as if derived only from $GP_u$ and $q$, without explicitly leaking the chosen/rejected contrast, while internally utilizing the pair to discover which criteria make $y^+$ pass and $y^-$ fail.

Step 3: Weight the checklist items via Preference-Contrastive Criterion Weighting. It first identifies other users with large preference deviations from the target user to generate additional negative examples. It then calculates how much a negative example "catches up" to the chosen response after each criterion is removed. Criteria that better separate the chosen from negative examples receive higher weights.

Step 4: Convert continuous saliency into three natural language tags: Essential, Important, and Optional, then train Llama-3.2-3B-Instruct as the checklist generator. The target is a text sequence of "criterion + evidence + importance tag."

The inference stage retains only two components: the trained checklist generator $\phi$ and an off-the-shelf LLM judge $\theta$. $\phi$ generates the checklist based on $GP_u$ and $q$; $\theta$ scores the candidate response on a scale of 1-10 for each criterion; finally, a scalar reward is obtained by the dot product of criterion scores and numerical weights mapped from E/I/O tags.

### Key Designs
1.  **Distilling dynamic checklist from preference pairs**:
    - **Function**: Transforms user history and current queries into actionable personalized evaluation standards rather than static personas.
    - **Mechanism**: Obtains $GP_u$ from history, then uses $GP_u$, query, and the chosen/rejected pair to have a strong LLM generate a checklist $C_{u,q}=LLM(GP_u,q,y^+,y^-)$. The prompt explicitly forbids mentioning the specific answers, ensuring the output appears as criteria derived from the user profile and query.
    - **Design Motivation**: The chosen/rejected pair reveals factors that distinguish user preferences in the current task. However, including traces of the pair in the checklist would cause the generator to learn instance-level explanations instead of generalizable criteria. By "implicitly using contrast, explicitly outputting criteria," the checklist achieves both discriminative power and transferability.

2.  **Preference-Contrastive Criterion Weighting**:
    - **Function**: Identifies which criteria in the checklist best represent personalized preferences, preventing the model from treating general quality standards as core signals.
    - **Mechanism**: Performs inter-user contrastive sampling by clustering users based on $GP$ and selecting the most distant clusters. Combined with query-conditioned embeddings $Enc(GP_x,q)$, it selects the top-3 most distant users and generates negative responses. For personalized saliency scoring, the LLM scores the chosen and negative candidates on each criterion. Saliency is calculated as the increase in the normalized score ratio of negative items relative to the chosen response after removing criterion $c_k$: $Saliency(c_k)=s(C^{-k},Y^-)/(s(C^{-k},y^+)+\epsilon)-s(C,Y^-)/(s(C,y^+)+\epsilon)$.
    - **Design Motivation**: Original rejected responses might just be of poor quality and not represent a "personalized contrast" that another user might like. Introducing negative examples from distant users shifts the contrast from general quality to personalized differences. Weighting by marginal impact after removal suppresses irrelevant or easily satisfied checklist items.

3.  **Checklist-guided reward with importance tags**:
    - **Function**: Ensures the generated checklist is not just explanatory text but actively participates in reward calculation.
    - **Mechanism**: During training, saliency is ranked and discretized into Essential, Important, and Optional based on cumulative weight thresholds $\tau_1=0.4, \tau_2=0.9$. During inference, the LLM judge outputs a criterion-wise score vector, and E/I/O tags are mapped to numerical weights (Essential=1.0, Important=0.7, Optional=0.3). The final reward is $r_{u,q}(y)=\mathbf{w}_{u,q}^\top\theta(GP_u,q,y,\hat{C}_{u,q})$.
    - **Design Motivation**: A plain checklist tells the judge "what to look at" but not "what is more important." E/I/O tags provide a lightweight, readable, and learnable weight interface, allowing the checklist to improve interpretability and aggregate stably into a scalar reward.

### Loss & Training
The checklist generator is trained using standard next-token prediction, maximizing the probability of the tagged checklist $\tilde{C}_{u,q}$ given $GP_u$ and $q$: $\mathcal{L}_\phi=-\sum \log p_\phi(\tilde{C}_{u,q}\mid GP_u,q)$. Llama-3.2-3B-Instruct serves as the backbone, trained for 3 epochs on 8x A6000s, with a per-device batch size of 2, gradient accumulation of 16, and a learning rate of $2\times10^{-4}$.

For training data construction, GPT-4o-mini generates $GP_u$ and the initial checklist. Rejection sampling filters low-quality samples: if the checklist as extra context does not result in a higher reward for the chosen response using a Llama-3.1-8B judge, it is regenerated. Contrastive sampling uses Qwen3-Embedding-0.6B for user representation, K-Means for distant clusters, and selects the top-3 query-conditioned distant users. Saliency scoring utilizes Llama-3.1-8B for 1-10 scoring per criterion.

Inference does not require contrastive sampling or saliency calculation, only checklist generation and criterion-wise scoring. Thus, while the training flow is complex, test-time overhead is just one checklist generation and criterion-level scoring step beyond a standard LLM judge.

## Key Experimental Results

### Main Results
The authors evaluated on three personalized reward benchmarks: PRISM-Personalized (ID), ChatbotArena-Personalized and BESPOKE-MetaEval (OOD). All data used strict user-level splits. The metric is binary preference prediction accuracy, with Llama-3-8B-Instruct and Llama-3-3B-Instruct as reward judges.

| Method | PRISM-P 8B | ARENA-P 8B | BESPOKE-M 8B | Avg |
|------|------------|-------------|--------------|------|
| GPO | 56.48 | 52.01 | 51.49 | 53.20 |
| VPL | 58.23 | 53.36 | 53.54 | 54.77 |
| PAL | 54.23 | 53.89 | 51.33 | 53.36 |
| BT + SynthMe | 62.74 | 56.42 | 50.47 | 55.86 |
| Default LLM judge | 52.80 | 53.56 | 55.46 | 53.19 |
| + Memory | 54.17 | 58.15 | 57.75 | 54.58 |
| + CoT distill | 55.47 | 55.84 | 61.35 | 55.84 |
| + SynthMe | 55.24 | 58.83 | 54.67 | 54.16 |
| + P-Check | 65.11 | 61.56 | 75.48 | 63.62 |

P-Check achieves an average accuracy of 63.62, a 10.43 percentage point increase over the Default LLM judge (approx. 19.61% relative improvement). Crucially, it leads significantly on OOD datasets: outperforming Memory / SynthMe on Arena and jumping from 55.46 to 75.48 on BESPOKE. This suggests that the checklist learns evaluation logic transferable to new users and scenarios rather than static templates.

The authors also tested P-Check’s transferability across different judges. Results show accuracy improvements across all three datasets regardless of whether the judge is Qwen3-8B, Qwen3-13B, GPT-4o-mini, or GPT-4o. For instance, GPT-4o improved from 63.27 to 77.66 on BESPOKE-M. This indicates that the bottleneck is not just the base capability of the judge models, but the lack of an explicit intermediate interface telling the judge "what to look for."

| Judge | PRISM-P Original | PRISM-P + P-Check | ARENA-P Original | ARENA-P + P-Check | BESPOKE-M Original | BESPOKE-M + P-Check |
|-------|--------------|-------------------|--------------|-------------------|-----------------|----------------------|
| Qwen3-8B | 55.14 | 63.71 | 57.41 | 59.98 | 59.23 | 70.36 |
| Qwen3-13B | 59.76 | 63.23 | 58.89 | 63.62 | 64.14 | 79.16 |
| GPT-4o-mini | 56.07 | 63.40 | 59.86 | 62.31 | 60.23 | 76.51 |
| GPT-4o | 58.94 | 64.83 | 60.06 | 69.36 | 63.27 | 77.66 |

### Ablation Study
Ablations focused on the two components of Preference-Contrastive Criterion Weighting: inter-user contrastive sampling and saliency scoring. Full P-Check performed best across all reward benchmarks; removing saliency scoring led to the largest drop, particularly on BESPOKE-M (from 75.48 down to 66.68).

| Configuration | PRISM-P | ARENA-P | BESPOKE-M | Description |
|------|---------|---------|-----------|------|
| P-Check (full) | 65.11 | 61.56 | 75.48 | Full Model |
| w/o inter-user sampling | 63.46 | 59.56 | 72.23 | Only original rejected items remain; weaker personalized contrast |
| w/o saliency scoring | 59.98 | 58.46 | 66.68 | Checklist items lack discriminative weights; hurts generalization |

Downstream personalized generation experiments evaluated P-Check's utility as a reward on BESPOKE. The authors compared Best-of-N (BoN) selection and DPO methods using ROUGE-L, METEOR, and BESPOKE-EVAL metrics. Regardless of whether the policy was Llama3-8B or GPT-4o-mini, P-Check achieved the highest scores, with differences from top baselines being statistically significant via paired t-tests.

| Alignment | Policy | Reward / Method | ROUGE-L | METEOR | BESPOKE-EVAL |
|----------|--------|---------------|---------|--------|--------------|
| BoN | Llama3-8B | Default policy | 7.92 | 6.09 | 51.55 |
| BoN | Llama3-8B | strongest baseline: CoT distill | 8.24 | 7.14 | 54.90 |
| BoN | Llama3-8B | P-Check | 9.43 | 8.22 | 59.76 |
| BoN | GPT-4o-mini | strongest baseline: CoT distill | 8.24 | 7.62 | 54.76 |
| BoN | GPT-4o-mini | P-Check | 9.61 | 8.47 | 57.32 |
| DPO | Llama3-8B | strongest baseline: CoT distill | 8.41 | 7.63 | 56.64 |
| DPO | Llama3-8B | P-Check | 9.94 | 9.67 | 61.21 |

### Key Findings
- Explicit checklists are more effective than static personas. In pre-experiments, LLMs failed to infer correct criteria from history alone; once provided with oracle checklists, preference selection improved significantly, suggesting that the key to personalized rewards is "transforming context into executable standards" rather than "more context."
- Saliency scoring is the most critical training signal. Removing inter-user sampling causes drops, but removing criterion weighting is more detrimental, showing that checklist items cannot be treated equally; the model must know which items actually drive user choices.
- P-Check is more robust with sparse history and OOD users. User-macro accuracy remains stable as observed history decreases, likely because it only needs to extract a few criteria relevant to the current query rather than reconstructing the entire preference distribution.
- Checklists can serve as verbal feedback for lightweight personalization. Using P-Check checklists for rewriting an initial draft outperformed Self-Refine and SynthMe, indicating checklists are not just internal features of a reward model but also direct feedback for the generator.
- Performance-wise, P-Check increased Llama-3-8B judge inference time from 19:30 to 28:37 on the PRISM test set but improved accuracy from 52.8 to 65.11. Compared to Qwen3-13B, it has similar time but higher accuracy; compared to BT+SynthMe, it is both faster and more accurate.

## Highlights & Insights
- The most valuable contribution of this paper is shifting the intermediate layer of personalized rewards from "user representation" to an "evaluation interface." While personas and embeddings describe the user, checklists describe what the judge should check, making them more suitable for LLM-as-a-Judge integration.
- Preference-Contrastive Criterion Weighting is ingenious: it does not blindly trust LLM-generated checklists but estimates criterion value by checking if a negative example gets closer to the chosen one when a criterion is removed. This is more controllable than letting the LLM self-declare importance.
- Inter-user negative sampling captures the essence of personalization. A response might not be of low quality, just a bad fit for the target user. Constructing negatives from users with distant preferences allows the checklist to learn the axes of difference that matter specifically to the target user.
- E/I/O tags represent a practical engineering trade-off. Continuous weights are good for calculation but hard to generate and interpret; natural language tags are learnable by small models and remain human-readable.
- This approach can be transferred to many highly subjective tasks, such as personalized writing assistants, code review style modeling, or educational depth control. The core method remains translating user preferences into task-local rubrics.

## Limitations & Future Work
- Checklists assume user preferences can be expressed via discrete natural language criteria. However, some preferences are nuanced feelings of style, rhythm, or tone that might not be fully externalized into rules. Hybrid representations of explicit criteria and implicit vectors could be explored.
- If the generated checklist is incorrect, subsequent rewards will systematically drift. Case studies showed the model misjudged a subjective question about animals in heaven as requiring "scientific consensus," suggesting the small generator might over-apply factual preferences from history without fully grasping the context shift.
- The training pipeline is heavy, requiring user summaries, checklist construction, inter-user clustering/sampling, negative example generation, criterion scoring, and saliency annotation. While these can be cached offline, maintenance in real products is non-trivial.
- Experiments are largely based on public benchmarks, which might differ from real long-term user systems where privacy, evolving preferences, and safety policy conflicts are more prominent.
- P-Check improves preference fidelity but does not replace safety alignment. If a user preference is harmful, the personalized reward might push the model toward unsafe outputs.

## Related Work & Insights
- **vs SynthMe / persona-based personalization**: SynthMe uses natural language personas and examples to optimize judge prompts (describes "who the user is"); P-Check generates query-specific checklists (describes "what the response should meet"). The latter is more actionable.
- **vs VPL / PAL / GPO (latent preference models)**: These learn user embeddings or prototype mixtures end-to-end. P-Check offers interpretable, editable intermediate representations that are plug-and-play with different LLM judges.
- **vs rubric / checklist-based evaluation**: While existing rubric-based rewards focus on task-level or universal quality, P-Check personalizes the rubric to the user and query, further weighting them by saliency.
- **vs generative reward modeling / CoT distill**: CoT rationales explain why an answer is good but don't necessarily form reusable scoring dimensions. P-Check structures reasoning into a criterion-wise checklist for scoring and aggregation.
- **Insight**: Personalized systems don't have to bake all preferences into policy parameters. Decoupling the "user preference interpreter" from the "base judge/generator" may be a cheaper, more transparent, and more debuggable path.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using dynamic checklists and criterion saliency for personalized reward modeling is a novel task definition and training signal.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers ID/OOD accuracy, different judges, sparse history, downstream BoN/DPO, verbal feedback, ablations, and case studies.
- Writing Quality: ⭐⭐⭐⭐☆ Clear main line with solid methodology and organization, though some table placements require jumping between text and appendices.
- Value: ⭐⭐⭐⭐⭐ Direct implications for personalized alignment, interpretable rewards, and LLM-as-a-Judge; well-suited as a framework for "explicit evaluation criterion interfaces."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](../../ICLR2026/llm_alignment/swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](../../NeurIPS2025/llm_alignment/limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](../../ICLR2026/llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)

</div>

<!-- RELATED:END -->
