---
title: >-
  [Paper Note] Preference Learning Unlocks LLMs' Psycho-Counseling Skills
description: >-
  [ACL 2026][Dialogue Systems][Psychological Counseling] This paper constructs the PsyCoPref preference dataset focused on the quality of psychological counseling responses. It employs reward models, DPO…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Psychological Counseling"
  - "Preference Learning"
  - "Reward Model"
  - "DPO"
  - "Human Preference Alignment"
date: 2026-05-08
content_hash: e7dec0d9a2e35d19
---

# Preference Learning Unlocks LLMs' Psycho-Counseling Skills

**Conference**: ACL 2026  
**arXiv**: [2502.19731](https://arxiv.org/abs/2502.19731)  
**Code**: [https://huggingface.co/Psychotherapy-LLM](https://huggingface.co/Psychotherapy-LLM)  
**Area**: LLM Safety / Counseling Alignment  
**Keywords**: Psychological Counseling, Preference Learning, Reward Model, DPO, Human Preference Alignment

## TL;DR

This paper constructs the PsyCoPref preference dataset focused on the quality of psychological counseling responses. It employs reward models, DPO, and iterative preference learning to train LLMs, enabling an 8B model to achieve an 87.0% win rate against GPT-4o in counseling response quality.

## Background & Motivation

**Background**: Psychological counseling assistance is a high-potential application for LLMs because the supply of professional mental health support falls far short of demand. Existing systems typically utilize general instruction fine-tuning, role-playing prompts, or small amounts of counseling data to simulate therapists. however, high-quality, real-world session data is difficult to accumulate due to privacy restrictions.

**Limitations of Prior Work**: Counseling responses are not simply a matter of being "helpful" or "safe." A response must display empathy, relevance, conciseness, and safety, while also encouraging client self-exploration, enhancing autonomy, and identifying the stage of change. Existing general reward models or LLM-as-judge frameworks often only learn generalized helpfulness and fail to consistently distinguish high-quality responses in professional counseling contexts.

**Key Challenge**: Professional counseling requires rigorous supervision, yet high-quality professional labels are the most difficult to obtain. Furthermore, differences in experience among therapists lead to inconsistent response quality in public session data. Directly using session text to supervise models risks learning low-quality counseling behaviors.

**Goal**: The authors aim to establish a set of professional, fine-grained evaluation principles for counseling responses, generate and verify large-scale preference pairs based on these principles, and finally test whether these preferences can train a reliable reward model and a policy model that responds more effectively to clients.

**Key Insight**: Instead of collecting "standard answers," it is better to compare multiple model responses to the same client statement. Preference pairs naturally express relative judgments of quality and bypass the issue of unstable response quality from individual therapists.

**Core Idea**: Use professional counseling principles to transform multi-model generated responses into high-quality preference data (PsyCoPref). Then, use reward modeling and iterative online DPO to enable LLMs to learn more professional counseling responses with clear boundaries.

## Method

### Overall Architecture

The overall process is divided into three layers: 
1. **PsyCoPref Construction**: Collect client statements, generate responses using 20 different LLMs, and have GPT-4o score them based on professional principles.
2. **Reward Modeling**: Train a Bradley-Terry style reward model using preference pairs to learn "which response looks more like a professional therapist."
3. **Policy Optimization**: Train the policy model using DPO and DPO-Iter, followed by an evaluation of generation quality by both GPT-4o and real counseling experts.

### Key Designs

1. **PsychoCounsel Seven-Dimension Principles**:
    - **Function**: Decomposes counseling response quality into actionable evaluation dimensions rather than relying on a global "helpfulness" score.
    - **Mechanism**: The seven dimensions include Empathy and Emotional Understanding, Personalization and Relevance, Clarity and Conciseness, Avoidance of Harmful Language, Promotion of Self-Exploration, Promotion of Autonomy and Confidence, and Sensitivity to Stages of Change. The latter three align closely with client-centered professional counseling goals, while the first four cover basic AI safety and usability.
    - **Design Motivation**: The value of a counseling response is often found in details, such as asking about triggers or respecting autonomy. Fine-grained principles bring preference labels closer to expert judgment.

2. **Generate-Score-Pair Data Pipeline**:
    - **Function**: Constructs high-quality preference pairs from multi-source client statements.
    - **Mechanism**: Client statements are collected from counsel-chat, MentalAgora, TherapistQA, Psycho8k, and multiple HuggingFace datasets. After filtering and deduplication, 26,483 statements covering 8 broad themes and 42 sub-themes are obtained. For each statement, 4 responses are randomly sampled from a model pool. GPT-4o scores each response (1-5) across all principles. Preference pairs are formed based on average scores; the training set only retains pairs with a score gap $\ge 1$.
    - **Design Motivation**: The model pool (3B to 70B plus commercial models) provides diverse response quality. Gap filtering avoids introducing noise from samples where quality is too similar to distinguish.

3. **Reward Model and Iterative Preference Learning**:
    - **Function**: Verifies that PsyCoPref can actively improve model generation capabilities rather than just serving as an evaluation set.
    - **Mechanism**: The reward model uses a BT loss to maximize the reward gap between chosen and rejected responses: $L=-\log\sigma(r_\theta(x,y_c)-r_\theta(x,y_r))$. Policy training involves two methods: offline DPO on PsyCoPref, and DPO-Iter, which generates 8 responses per statement each round and selects online preference pairs using an RM of the same scale.
    - **Design Motivation**: Offline DPO is stable but limited by data distribution. DPO-Iter corrects the policy on its own generation distribution, reducing distribution shift and reward hacking.

### Loss & Training

Reward models are initialized from Llama3.2-3B-Instruct and Llama3.1-8B-Instruct, trained for 2 epochs on PsyCoPref with a batch size of 128 and a learning rate of 9e-6. For policy training, DPO uses $\beta=0.1$. DPO-Iter samples 6,400 statements per round, generating 8 candidates each, with a batch size of 64 and a learning rate of 5e-7 for 1,600 total steps.

## Key Experimental Results

### Main Results

**Reward Model Performance on PsyCoPref Test Set**

| Model | Acc.↑ | AUC↑ | ECE↓ | Brier↓ |
|------|-------|------|------|--------|
| Skywork-Reward-Llama-3.1-8B-v0.2 | 57.9 | 0.623 | 0.331 | 0.379 |
| Skywork-Reward-Gemma-2-27B | 69.2 | 0.740 | 0.123 | 0.229 |
| Llama-3.1-Nemotron-70B-Reward | 87.3 | 0.938 | 0.040 | 0.102 |
| Llama-3.1-70B-Instruct (Ranker) | 88.2 | - | - | - |
| PsyCo-Llama3-3B-Reward | 98.1 | 0.997 | 0.050 | 0.014 |
| PsyCo-Llama3-8B-Reward | 97.8 | 0.998 | 0.045 | 0.016 |

**Overall Win Rate of Policy Models vs GPT-4o**

| Setup | Llama3-3B | +DPO | +DPO-Iter | Llama3-8B | +DPO | +DPO-Iter |
|------|-----------|------|-----------|-----------|------|-----------|
| No constraint | 28.5 | 58.5 | 69.4 | 29.3 | 72.9 | 87.0 |
| Length constraint | 15.0 | 37.0 | 46.4 | 18.5 | 49.3 | 77.0 |

### Ablation Study

**Complementarity between PsyCoPref and General HelpSteer2 Data**

| Model | Training Data | PsyCoPref Acc.↑ | AUC↑ | Brier↓ | RewardBench Acc.↑ |
|------|----------|------------------|------|--------|-------------------|
| Llama-3B | HelpSteer2 | 81.6 | 0.916 | 0.120 | 83.6 |
| Llama-3B | HelpSteer2 + PsyCoPref | 97.6 | 0.998 | 0.017 | 86.1 |
| Llama-8B | HelpSteer2 | 81.7 | 0.898 | 0.128 | 86.6 |
| Llama-8B | HelpSteer2 + PsyCoPref | 97.5 | 0.998 | 0.018 | 87.2 |

**Data Merging Results under 10k Budget**

| Config | PsyCoPref Acc.↑ | RewardBench Acc.↑ | Avg Acc.↑ | Note |
|------|------------------|-------------------|------------|------|
| Psy10k | 0.963 | 0.745 | 0.854 | Strongest in-domain, low general transfer |
| Help10k | 0.855 | 0.888 | 0.871 | Good general performance, low resolution |
| Psy5kHelp5k | 0.958 | 0.896 | 0.927 | Balances domain and general reward quality |

### Key Findings
- The 3B version of the PsyCoPref reward model achieves 98.1% accuracy, significantly outperforming 70B general reward models, indicating that counseling response quality requires domain-specific preference supervision.
- DPO-Iter significantly outperforms offline DPO: Llama3-8B improves from 72.9% to 87.0% without length constraints.
- Real counseling experts show an 82.5% agreement rate with the GPT-4o judge, and experts generally prefer PsyCo-Llama3-8B, supporting the credibility of automated evaluation.
- Length constraints reduce the overall win rate but improve clarity, safety, and stage identification, suggesting models may gain advantage through longer responses, requiring inference-stage constraints for balance.

## Highlights & Insights
- The primary highlight is framing counseling capability as a "professional preference modeling" problem rather than a simple SFT task on session text; this design handles therapist variability and privacy constraints more effectively.
- The seven-dimension principles are highly transferable, particularly self-exploration, autonomy, and stage of change, which can serve as evaluation frameworks for medical companionship and crisis support.
- DPO-Iter results suggest that models must learn not only expert preferences but also calibrate continuously on their own generation distributions to avoid falling into fixed, generic templates.
- Expert case studies show that stronger models do not just become safer or more polite; they specifically incorporate client details and propose collaborative exploration questions.

## Limitations & Future Work
- PsyCoPref currently focuses on single-turn interactions and cannot evaluate core counseling skills such as long-term therapeutic relationships, multi-turn memory, or therapeutic alliance maintenance.
- The seven-dimension principles are currently averaged with equal weight, though weights might vary across crisis levels or cultural contexts.
- Data and evaluation rely heavily on GPT-4o as a scorer and judge; while expert validation was positive, bias inheritance remains a risk.
- The experiment is positioned as an assistant for therapists drafting responses; direct deployment to clients requires risk grading and crisis intervention protocols.

## Related Work & Insights
- **vs RLHF / General Preference Data**: General RLHF optimizes for helpfulness and safety. This work proves that professional fields like counseling require separate, fine-grained preference principles.
- **vs DPO**: While DPO uses static pairs, DPO-Iter uses the reward model to select extreme pairs from online generation, making it more suitable for correcting the current policy's actual output.
- **vs Counseling SFT Datasets**: Directly mimicking session data risks absorbing low-quality responses. PsyCoPref explicitly filters out poor candidates via multi-model comparison.
- **Insights**: High-stakes professional dialogues, such as medical Q&A or legal advice, can adopt the "expert principles + multi-model candidates + preference learning" pipeline to build domain reward models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Models counseling as a professional preference learning task; the data construction and online training integrate seamlessly.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers reward models, policy models, expert validation, length constraints, and data mixing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear reasoning and data explanation, though some details depend on the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for mental health AI systems and provides a reusable paradigm for professional domain preference data construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](../../NeurIPS2025/dialogue/kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)

</div>

<!-- RELATED:END -->
