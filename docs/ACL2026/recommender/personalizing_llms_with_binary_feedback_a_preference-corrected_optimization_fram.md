---
title: >-
  [Paper Note] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework
description: >-
  [ACL2026][Recommender Systems][LLM Personalization] The paper proposes C-BPO, which treats target user history as positive feedback and auxiliary user history as noisy unlabeled negative feedback. By employing PU learnin…
tags:
  - "ACL2026"
  - "Recommender Systems"
  - "LLM Personalization"
  - "Binary Feedback"
  - "PU Learning"
  - "Preference Recalibration"
  - "User Variance"
date: 2026-05-08
content_hash: af96a1cbd2af4b92
---

# Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework

**Conference**: ACL2026  
**arXiv**: [2605.10043](https://arxiv.org/abs/2605.10043)  
**Code**: Undisclosed  
**Area**: LLM Personalization / Preference Optimization / Recommendation & User Modeling  
**Keywords**: LLM Personalization, Binary Feedback, PU Learning, Preference Recalibration, User Variance

## TL;DR
The paper proposes C-BPO, which treats target user history as positive feedback and auxiliary user history as noisy unlabeled negative feedback. By employing PU learning to correct for "preference overlap" and the resulting incorrect penalties, the framework ensures the LLM learns unique user preferences without suppressing general task capabilities.

## Background & Motivation
**Background**: Common approaches to LLM personalization include three categories: retrieving user history/profiles in prompts, training lightweight modules like LoRA on user history, or constructing preference pairs for DPO-style optimization. The shared goal is to align model outputs with what a specific user would write, select, or prefer.

**Limitations of Prior Work**: Many methods focus solely on the target user's history, equating personalization with "mimicking this user's past." However, the most informative part of user preferences often stems from cross-user variance: identifying how this user differs from others. Furthermore, DPO requires paired winner/loser completions for the same input, whereas real-world user histories typically consist of unpaired $(x, y)$ samples.

**Key Challenge**: Auxiliary data from other users is both valuable and risky. It provides natural contrastive signals to help the model learn "who the target user is not," yet users share significant general task knowledge and group preferences. Treating all other users' data as negative samples causes the model to erroneously penalize common capabilities—such as the requirement that news headlines be concise, academic titles be accurate, or reviews focus on product attributes.

**Goal**: The authors aim to convert unpaired user history into optimizable preference signals while addressing three problems: replacing pairwise preferences with binary feedback, peeling away shared preferences from auxiliary user data, and maintaining stable training boundaries when target samples are scarce compared to auxiliary samples.

**Key Insight**: Borrowing from Positive-Unlabeled (PU) learning, the target user history is considered the positive class, while the auxiliary history is not a clean negative class but a mixture of positive preferences and true negative preferences. By estimating the proportion of "target-like" data in the auxiliary set, the positive bias can be subtracted from the negative risk.

**Core Idea**: Shift from "other user data = negative samples" to "other user data = unlabeled mixture." Use PU risk decomposition to derive a purified negative loss for binary preference optimization.

## Method
The core of C-BPO is not the manual construction of contrastive responses but the direct utilization of user history. For a target user, the history $H_{tar}$ is treated as positive feedback, and other users' history $H_{aux}$ is treated as an auxiliary set. The model is adapted via PEFT modules like LoRA, but the objective is neither simple SFT nor treating $H_{aux}$ as a generic "thumbs-down." Instead, it learns user boundaries through a preference-corrected binary feedback objective.

### Overall Architecture
Given a base model $\pi_{base}$, the personalized model is defined as $\pi_{user} = \pi_{base} + \Delta_{user}$, where $\Delta_{user}$ is a user-specific lightweight module. Training requires target user history $(x, y)$ and auxiliary user history $(x, y)$, rather than two responses for the same prompt.

The model adopts the implicit reward form from binary feedback optimization (e.g., BCO/KTO): $r_\theta(x, y) = \beta \log \pi_\theta(y|x) / \pi_{ref}(y|x)$. Positive samples aim for rewards above a reference point $\delta$, while negative samples aim below it. C-BPO introduces two modifications: first, treating the auxiliary set as an unlabeled mixture and subtracting the target user preference contamination; second, using independent EMA reward means for positive and auxiliary samples to estimate the reference point, preventing boundary shifts due to negative sample volume.

All methods are initialized from a TAM checkpoint using identical LoRA hyperparameters. The C-BPO correction coefficient $\alpha$ is estimated before training based on user history embeddings. Inference requires only the trained user adapter.

### Key Designs
1. **Binary Feedback User Variance Modeling**:
    - **Function**: Converts target and auxiliary user histories into trainable positive/negative preference signals without pairwise data.
    - **Mechanism**: $H_{tar}$ is treated as positive feedback; $H_{aux}$ is the source for implicit negative feedback. The loss follows the BCO form: positive samples optimize $-\log \sigma(r_\theta(x,y)-\delta)$, and negative samples optimize $-\log \sigma(-(r_\theta(x,y)-\delta))$.
    - **Design Motivation**: Real-world personalization data consists of "what users wrote" or "what users liked," not standard RLHF pairs. Binary feedback allows direct learning from raw history.

2. **Preference Recalibration via PU Learning**:
    - **Function**: Prevents the mischaracterization of shared knowledge in auxiliary data as target user anti-preferences.
    - **Mechanism**: $H_{aux}$ is viewed as an unlabeled set mixed with positive and true negative preferences. The purified negative loss is defined as $L_{pure\_neg} = \mathbb{E}_{H_{aux}}[l(g,-1)] - \alpha \mathbb{E}_{H_{tar}}[l(g,-1)]$. The final objective is $L_{C-BPO} = L_{pos} + \frac{1}{1-\alpha} \max(0, L_{pure\_neg})$.
    - **Design Motivation**: If multiple users prefer concise titles, penalizing all auxiliary titles would degrade the model. The correction term removes shared preferences from the negative signal, focusing gradients on user specificity.

3. **Independent EMA Reference Points and Alpha Estimation**:
    - **Function**: Stabilizes the preference boundary $\delta$ under data imbalance and adapts the overlap coefficient to different users.
    - **Mechanism**: Instead of mixing samples in a batch to calculate a mean reference point, C-BPO maintains separate EMAs for positive and auxiliary rewards, using their average as a dynamic $\delta_{EMA}$. $\alpha$ is estimated via a proxy classifier on user history embeddings: first distinguishing target from auxiliary users, then averaging the "target-like" probabilities in the auxiliary set.
    - **Design Motivation**: In cases where auxiliary data far exceeds target data, a joint mean is skewed by the auxiliary distribution. Furthermore, varying overlap degrees necessitate individual correction strengths.

### Loss & Training
The positive loss is $L_{pos} = \mathbb{E}_{H_{tar}}[l(g,+1)]$, where $l(g,+1) = -\log \sigma(r_\theta(x,y)-\delta)$. The negative loss is the purified term $L_{pure\_neg}$ after subtracting $\alpha \mathbb{E}_{H_{tar}}[l(g,-1)]$, utilizing $\max(0, \cdot)$ to prevent overfitting caused by negative risk estimates in high-capacity models. LoRA is implemented with rank 8 and scaling 16. Optimization uses AdamW with linear warmup for 3 epochs at a learning rate of $1e-6$.

## Key Experimental Results

### Main Results
Evaluation was conducted on five tasks from LaMP and LongLaMP: news headline generation, scholarly title generation, abstract generation, review writing, and topic writing. Mistral-7B-Instruct-v0.3 served as the base, measured by ROUGE-1 / ROUGE-L.

| Task | Base | OPPU | CoPE | BCO | C-BPO | Main Conclusion |
|------|------|------|------|-----|-------|----------|
| Abstract Gen. R-1 / R-L | 0.341 / 0.186 | 0.378 / 0.218 | 0.392 / 0.239 | 0.373 / 0.231 | 0.398 / 0.269 | C-BPO performs best; R-L gains are significant |
| Review Writing R-1 / R-L | 0.287 / 0.126 | 0.319 / 0.134 | 0.335 / 0.146 | 0.315 / 0.132 | 0.353 / 0.154 | Auxiliary variance helps capture review styles |
| Topic Writing R-1 / R-L | 0.246 / 0.105 | 0.278 / 0.112 | 0.281 / 0.120 | 0.272 / 0.112 | 0.291 / 0.118 | Best R-1; R-L slightly below CoPE |
| News Headline R-1 / R-L | 0.119 / 0.105 | 0.203 / 0.182 | 0.205 / 0.184 | 0.197 / 0.179 | 0.215 / 0.198 | Outperforms retrieval and SFT methods |
| Scholarly Title R-1 / R-L | 0.409 / 0.324 | 0.510 / 0.454 | 0.519 / 0.461 | 0.507 / 0.443 | 0.539 / 0.481 | Stable gains in academic personalization |

| Dataset Stats | Samples | Avg Input Len | Avg History Len | Avg Output Len |
|------|------|------|------|------|
| Abstract Generation | - | 233.1 ± 117.5 | 1296.7 ± 446.4 | 210.5 ± 92.8 |
| Review Writing | 19,649 | 407.2 ± 299.5 | 759.3 ± 324.2 | 511.8 ± 294.2 |
| Topic Writing | 21,119 | 358.3 ± 316.9 | 260.6 ± 314.0 | 358.3 ± 255.4 |
| News Headline Generation | 7,275 | 15.5 ± 6.0 | 270.1 ± 182.1 | 18.6 ± 5.2 |
| Scholarly Title Generation | 16,076 | 17.9 ± 6.1 | 444.0 ± 121.6 | 16.4 ± 5.8 |

### Ablation Study

| Analysis Item | Observation | Implication for Method |
|------|------|------|
| Aux Data Ratio $x = |H_{aux}| / |H_{tar}|$ | When target history is <50%, $x=1.5$ is worse than balanced; more aux data only helps with sufficient target history | Negative signals require positive anchors; otherwise, shared preferences cannot be peeled away |
| Removing Ind. EMA Ref Point | Performance drops significantly when $x \neq 1.0$; lower than OPPU at $x=1.5$ with scarce target data | Reference points must not be dominated by the majority (auxiliary) distribution |
| User Uniqueness Grouping | KTO/BCO degrades in Non-unique groups; C-BPO remains stable across Unique and Non-unique groups | Standard BPO fails at preference overlap; correction is the key differentiator |
| $\alpha$ Sensitivity | Non-unique users need higher $\alpha$ to filter common preferences; lower $\alpha$ is optimal for Unique users | $\alpha$ is an interpretable coefficient of preference overlap, not just a hyperparameter |
| Token-level log-prob shift | Compared to BCO, C-BPO avoids excessive suppression of shared preference tokens in auxiliary data | Confirms PU correction protects general task knowledge |

### Key Findings
- Treating other users' data directly as negative samples is unreliable. KTO and BCO performed worse than OPPU or CoPE in several tasks, proving "binary feedback" alone does not solve personalization without addressing preference overlap.
- C-BPO outperformed CoPE in most metrics across 5 tasks without needing rejection-sampling for negative responses, making the data pipeline more representative of real-world history.
- The user uniqueness analysis proves that standard BPO is most prone to penalizing shared preferences when auxiliary and target users are similar; C-BPO extracts personalization signals more effectively.
- EMA reference points are a critical stability factor for real-world applications where the auxiliary pool is significantly larger than the target history.

## Highlights & Insights
- The paper shifts personalization from "fitting the target user" to "modeling the difference of the target user relative to the population." Personality is inherently a relative concept.
- The application of PU learning is elegant: it positions auxiliary history as an unlabeled mixture rather than a clean negative class.
- $\alpha$ is highly interpretable as a measure of preference overlap. Estimating it via proxy classifiers provides a practical starting point for deployment beyond manual grid searches.
- C-BPO offers insights for recommendation systems: implicit negative feedback is often a mixture of unobserved or group-shared signals; PU-style correction can be migrated to ranking and generative recommendations.

## Limitations & Future Work
- Evaluation relies primarily on ROUGE, which has limited capability in capturing personalized style and user satisfaction. Future work requires human evaluation or online interaction metrics.
- $\alpha$ estimation depends on history embeddings. If embeddings fail to capture style or the history is too short, the correction might be unstable.
- Auxiliary data selection remains simple (random or grouping). Real systems must consider privacy, group bias, cold-start users, and temporal shifts.
- The paper assumes user history is entirely positive feedback, ignoring accidental behaviors or outdated preferences. Time decay and confidence scores could be incorporated.
- Training per-user adapters is costly at a million-user scale; exploring clustered adapters or hypernetworks is a future direction.

## Related Work & Insights
- **vs RAG / PAG**: RAG/PAG prepends history to prompts without modifying weights. C-BPO trains modules to internalize fine-grained style, though at a higher training cost.
- **vs OPPU / SFT**: OPPU fits the target history only, often learning common task patterns. C-BPO includes contrastive auxiliary data to amplify user differences.
- **vs DPO / CoPE**: CoPE requires rejection sampling to build pairs; C-BPO uses unpaired raw history directly.
- **vs KTO / BCO**: While both support binary feedback, they assume clean negative samples. C-BPO acknowledges positive bias in auxiliary data and corrects it via PU risk decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Solid use of PU learning to address preference overlap in personalization.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 5 tasks and multiple baselines, though lacks real human preference trials.
- Writing Quality: ⭐⭐⭐⭐☆ Clear theoretical derivation and narrative.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for personalized LLMs, generative recommendations, and implicit feedback learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ICML 2026\] T-POP: Test-Time Personalization with Online Preference Feedback](../../ICML2026/recommender/t-pop_test-time_personalization_with_online_preference_feedback.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)

</div>

<!-- RELATED:END -->
