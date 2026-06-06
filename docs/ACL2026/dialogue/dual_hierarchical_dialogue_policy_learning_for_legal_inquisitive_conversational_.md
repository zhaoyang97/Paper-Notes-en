---
title: >-
  [Paper Note] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents
description: >-
  [ACL 2026][Dialogue Systems][Inquisitive Conversational Agent] The authors define dialogues where an "AI actively questions while the counterparty may not be cooperative" (e.g.…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Inquisitive Conversational Agent"
  - "Dual Hierarchical RL"
  - "Appraisal Agent"
  - "Poincaré Embedding"
  - "Offline DDQN"
date: 2026-05-08
content_hash: f23965d7948a3cff
---

# Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents

**Conference**: ACL 2026  
**arXiv**: [2605.14057](https://arxiv.org/abs/2605.14057)  
**Code**: The paper footnote mentions a Git repository, but no public link is provided.  
**Area**: Dialogue / Law / Reinforcement Learning  
**Keywords**: Inquisitive Conversational Agent, Dual Hierarchical RL, Appraisal Agent, Poincaré Embedding, Offline DDQN

## TL;DR
The authors define dialogues where an "AI actively questions while the counterparty may not be cooperative" (e.g., US Supreme Court justices questioning lawyers) as Inquisitive Dialogue. They propose a Dual Hierarchical RL framework—comprising an Appraisal Agent that scores lawyer responses in real-time (across 9 appraisal types) and a Hierarchical Dialogue Agent that performs DDQN action selection over a three-layer (act/subtype/utterance) Poincaré action space. By combining a triple reward (goal-relevance/novelty/succinctness) with a conservative regularization term, the PES (Probing Effectiveness) is pushed from the baseline's 4.22 to 4.47 on the Oyez Supreme Court dataset, achieving the highest multi-turn Coverage and MR.

## Background & Motivation

**Background**: Mainstream dialogue systems (MultiWOZ, Schema-Guided, Taskmaster, etc.) are almost entirely "collaborative TOD," where the user actively questions and the Agent compliantly satisfies. Recent work has also explored negotiation dialogue (Lewis 2017). However, Conversational AI has lacked systematic research into scenarios characterized by "**Agent dominance, uncooperative counterparties, and information extraction via Agent probing.**"

**Limitations of Prior Work**: In scenarios such as court trials, investigative journalism, medical consultations, and police interrogations, AI cannot be a passive responder; it must actively probe, reframe, and challenge. Directly applying existing TOD leads to three major issues: (i) heuristic and slot ontologies cannot support the strategy of "choosing the optimal question"; (ii) a single Supreme Court transcript turn often exceeds 5,000 tokens, which is beyond the context of mainstream seq2seq models; (iii) the goals of both parties are inconsistent or even adversarial—lawyers may be evasive or incomplete, and simple reward maximization can be bypassed.

**Key Challenge**: Treating dialogue as "RL in a flat action space" results in either an unmanageably large action space (at the NLG level) or a loss of expressivity (at the act level). Furthermore, a single reward signal (like task success) fails to capture the core essence of "questioning quality."

**Goal**: Design an RL framework that enables the agent to learn (i) **when to probe** (evaluating if the response is sufficient), (ii) **what type of question to ask** (probing / hypothesis / challenge / clarification), and (iii) **how to articulate it** (specific phrasing).

**Key Insight**: The authors noted that the questioning behavior of Supreme Court justices is naturally hierarchical—deciding the act first (Questioning vs. Declaration vs. Hypothesis Testing), then the subtype (Probing vs. Clarification vs. Comparison), and finally the utterance. Moreover, justices **first appraise the lawyer's previous response** (e.g., "you bypassed the question / you are non-responsive / satisfied"), and this appraisal determines the next strategy.

**Core Idea**: Decouple "appraisal" and "dialogue decision-making" into two mutually coupled RL agents—the Appraisal Agent outputs 9 classes of discrete appraisals $p^t$ as internal states, which are fed into the Hierarchical Dialogue Agent for DDQN over three-layer actions, naturally corresponding to the two-step thinking process of a justice.

## Method

### Overall Architecture
The approach can be viewed as a three-stage "MDP + Dual RL Agent" process:

1.  **MDP Formulation**: Each justice utterance $u_j^t$ is treated as an action $a^t$, and the attorney response $u_a^{t+1}$ as an observation. The augmented transition is defined as $\mathcal{D} \sim (s^t, p^t, a^t, r^t, s^{t+1})$, where $p^t = f(u_j^{t-1}, u_a^t, u_j^t)$ is the appraisal of the justice's attitude toward the lawyer's previous answer inferred by the Appraisal Agent (e.g., repeating the same question $\rightarrow$ "Dissatisfied").
2.  **Dual-Agent Collaboration**: (i) The **Appraisal Agent** uses DDQN to select $p(s) = \arg\max_p Q_{\text{App}}(s, p; \theta)$; (ii) The **Dialogue Agent** concatenates $p^t$ to the state to obtain $s_{\text{aug}}^t = \text{concat}(s^t, p^t)$, and sequentially selects $\{a_0, a_1, a_2\}$ across a 3-level action space.
3.  **Verbalization**: The selected three-level actions, along with the augmented state, are used to prompt LLaMA-3-8B-Instruct to generate the final natural language utterance.

### Key Designs

1.  **Dual-Agent Architecture (Appraisal + Hierarchical Dialogue)**:
    - **Function**: Explicitly decouples "how I view your previous answer" and "how I question next" into two RL agents.
    - **Mechanism**: The Appraisal Agent receives dialogue history and outputs 9 types of discrete appraisals (e.g., evasive / incomplete / satisfactory / contradictory), which are converted into one-hot vectors and concatenated into the Dialogue Agent's state. The Dialogue Agent decides the act/subtype/utterance only after receiving the augmented state. Both use DDQN, trained independently but coupled via state augmentation.
    - **Design Motivation**: The authors explicitly ask "why two agents?" for Modularity and interpretability. If combined into one, the model must simultaneously judge "evasive" and decide "probe more," causing state signals to be pulled in different directions. By separating them, Appraisal focuses on evaluation and Dialogue on policy. Experimentally, PES increased from 4.30 without the Appraisal Agent to 4.47 for the full model, identifying the Appraisal Agent as the largest contributor to PES.

2.  **Three-layer Poincaré Action Space (act $\rightarrow$ subtype $\rightarrow$ utterance)**:
    - **Function**: Decomposes "what to ask" into three levels of discrete actions and represents this action tree using hyperbolic Poincaré embeddings.
    - **Mechanism**: Level 1 is the high-level act (Questioning / Hypothesis Testing / Declaration); Level 2 is the subtype (Probing / Clarification / Comparison); Level 3 is the specific subcategory (Probe the Assumption / Probe the Premise, etc.). Embeddings are trained in Poincaré hyperbolic space: $\mathcal{L} = \sum_{(u,v) \in D} \log \frac{e^{-d(u,v)}}{\sum_{v' \in \mathcal{N}(u)} e^{-d(u, v')}}$, causing parents to be near the origin and children to be exponentially distant, with siblings being naturally similar. The Q-network predicts three actions sequentially, with each full action generating 3 transition tuples. Hierarchical consistency is enforced via $Q(s, a_0) = \max_{a_1} Q(s, a_1)$, corresponding to a new loss $\mathcal{L}_{\text{Dia}}^{\text{hier}} = \sum_i (Q(s, a_i) - \max_{a_{i+1}} Q(s, a_{i+1}))^2$.
    - **Design Motivation**: In a flat action space, "Probe assumption" and "Challenge premise" are two unrelated tokens. In a hierarchy, they share the Level-1 parent "Questioning," improving generalization. Hyperbolic embeddings are better suited for tree-like structures than Euclidean space, allowing siblings to share Q-value signals.

3.  **Triple Reward + Conservative Q-Regularization**:
    - **Function**: Decomposes "questioning quality" into three complementary quantitative goals and uses a lightweight regularization term to avoid offline RL Q-overestimation.
    - **Mechanism**: (i) **Goal-Relevance** $R_{\text{rel}}^{t+1} = \max_i \text{sim}(C[i], u_a^{t+1})$, using LLaMA-3-8B to calculate the maximum similarity between the lawyer's answer and case sub-conclusions $C[i]$, rewarding the extraction of useful information; (ii) **Novelty** $R_{\text{nov}}^{t+1} = N_{\text{attorney}}^{t+1} / (V(1 - ((V-1)/V)^{|u_a^{t+1}|}))$, using EAD to measure the ratio of newly introduced tokens, rewarding the elicitation of previously unseen information; (iii) **Clarity** $R_{\text{clarity}}^{t+1} = -\log|u_a^{t+1}|$, where shorter responses from the lawyer yield higher rewards (the justice prefers yes/no for control). These weights are passed to Q-learning. A conservative regularization term $\mathcal{L}^{\text{Reg}} = R_1(s) - R_2(s)$ is added, where $R_1 = \max_a Q(s,a)$ is the maximum potential overestimated value and $R_2 = Q(s, a)$ is sampled from the dataset $(s,a) \in \mathcal{D}$, pulling OOD Q-values back toward the dataset policy to reduce variance.
    - **Design Motivation**: A single task-success reward lacks signal in dialogue scenarios with vague "done" criteria. By decomposing into relevance/novelty/clarity, the agent learns both to "extract content" and "prevent rambling." Conservative regularization is a lightweight version of the CQL (Kumar 2020) concept, particularly suitable for Supreme Court scenarios where the "dataset policy is already near-optimal."

### Loss & Training
-   **Backbone**: Both agents use Double DQN (DDQN).
-   **Appraisal**: $\mathcal{L}_{\text{App}} = \mathcal{L}_{\text{App}}^{\text{DDQN}} + \alpha \mathcal{L}_{\text{App}}^{\text{Reg}}$, where $Y_{\text{App}} = r + \gamma Q(s, \arg\max_{p'} Q(s', p'; \theta_{App}); \theta_{App}^-)$.
-   **Dialogue**: $\mathcal{L}_{\text{Dia}} = \mathcal{L}_{\text{Dia}}^{\text{DDQN}} + \beta \mathcal{L}_{\text{Dia}}^{\text{Reg}} + \lambda \mathcal{L}_{\text{Dia}}^{\text{hier}}$, with hierarchical consistency loss enforcing $Q(s, a_0) = \max_{a_1} Q(s, a_1)$, etc.
-   **Data**: U.S. Supreme Court Oral Argument Transcripts (Oyez, 1955–2023), split by year for train/test.
-   **Verbalization**: Selected act/subtype/utterance is passed to LLaMA-3-8B-Instruct with a template to generate natural language; the method itself is fine-tuning-free.

## Key Experimental Results

### Main Results
SaulLM-7B automatically scores 1–5 on 4 metrics: CS (Conformity) / PS (Progression) / OS (Outcome Relevance) / PES (Probing Effectiveness) (Tab.1):

| Method | CS | PS | OS | PES | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Vanilla LLaMA-3 | 3.99 | 3.94 | 4.70 | 3.92 | 4.14 |
| SFT LLaMA-3 | 3.98 | 3.81 | 4.45 | 3.38 | 3.91 |
| SaulLM-7B (Legal-specific LLM) | 4.01 | 3.91 | 4.56 | 3.75 | 4.06 |
| Hudeček et al. (pipeline TOD) | 3.99 | 3.97 | 4.77 | 3.63 | 4.09 |
| VaRMI (offline policy gradient) | 4.00 | 3.94 | 4.71 | 3.93 | 4.15 |
| ArCHer (Actor-Critic) | 3.96 | 3.79 | 4.17 | 4.22 | 4.04 |
| **Ours (Dual Hierarchical)** | **4.01** | **3.98** | **4.89** | **4.47** | **4.34** |

Multi-turn dialogues used SeCom to simulate a lawyer opponent with a 10-turn limit: Coverage Score and Marginal Relevance Score were both highest for this method in Figures 3/4; Human evaluation (Tab.8) for Overall was also highest at 4.53.

### Ablation Study
Four ablation groups in Tab.2 (Full Model 4.34):

| Configuration | CS | PS | OS | PES | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Full Model | 4.01 | 3.98 | 4.89 | 4.47 | **4.34** |
| w/o Appraisal Agent | 4.03 | 4.00 | 4.74 | **4.30** | 4.27 |
| w/o Succinct Reward | 4.01 | 3.97 | 4.85 | 4.39 | 4.31 |
| w/o Novelty Reward | 4.01 | 3.97 | 4.82 | 4.34 | 4.29 |
| w/o Goal-Relevance | 4.00 | 3.97 | 4.83 | 4.32 | 4.28 |

### Key Findings
-   **Appraisal Agent is the largest contributor to PES**: Removing it drops PES from 4.47 to 4.30; the 0.17 drop is the largest for any single component. OS also dropped from 4.89 to 4.74. This indicates that the "evaluate before deciding" mechanism is significantly stronger at "probing effectiveness" than end-to-end single-agent models.
-   **Specialized Legal LLMs do not necessarily win**: SaulLM-7B's training set includes Supreme Court transcripts, yet it was outperformed by Vanilla LLaMA-3 in dialogue tasks (4.06 vs. 4.14), suggesting that "domain knowledge" $\neq$ "dialogue policy"—RL-style policy learning is essential.
-   **The failure of SFT is due to data quality**: SFT LLaMA-3 was the lowest among the 6 methods (3.91) primarily because Supreme Court data contains many low-quality segments, which SFT absorbs indiscriminately. Ours with RL + conservative regularization "bypasses" low-quality data as rewards do not favor those transitions.
-   **Three rewards are complementary**: Removing any single reward dropped the Overall score by 0.03–0.06. OS was most affected by novelty (−0.07), while PES was significantly affected by both goal-relevance and succinctness—indicating they govern different dimensions.
-   **Coverage / MR remain highest across multiple turns**: Fig.3/4 show the proposed method leading all baselines at 2/4/6/8/10 turns, demonstrating that the dual agent is not only better in a single turn but also more stable in long-range dialogue planning.

## Highlights & Insights
-   **Redefining TOD into three categories (collaborative / negotiation / inquisitive)** is the most conceptual contribution of this paper, systematically defining "AI active probing," a scenario long neglected.
-   **Making "appraisal" an explicit agent** is clever—it transforms Theory of Mind style "agent's evaluation of interlocutor" from a latent prompt into a learnable discrete signal, injected into decision-making via state augmentation. It is a clean decoupling for dialogue policy.
-   **Poincaré hyperbolic embeddings + hierarchical Q-consistency loss** allows the "act/subtype/utterance" tree to be truly utilized, significantly improving performance and saving parameters compared to flat one-hot representations.
-   **Lightweight conservative regularization $R_1 - R_2$**: Unlike the complex saddle-point solving in CQL, this subtraction form is implemented in 5 lines of code but proves effective for "near-optimal dataset policy" scenarios—a trick worth trying in other offline RL tasks.
-   **EAD-based novelty reward** is more rational than raw distinct-N because it length-normalizes the utterance—a metric trick transferable to chatbot diversity tasks.

## Limitations & Future Work
-   **Verbalization depends on LLM probability**: The authors admit "if LLM probability is not on the optimal sequence, the method cannot reach the optimum," as the final natural language is prompted, not learned by RL.
-   **Reward / Action designs are manual and domain-locked**: Migrating to medical interviews or journalism interrogation requires redesigning the 9 appraisal classes and 3-level action taxonomy; generalizability requires future work.
-   **Validity of conservative regularization depends on near-optimal dataset policy**: Supreme Court justices are top-tier professionals with high policy quality; for amateur dialogue datasets, pulling Q toward the dataset distribution might be the wrong direction.
-   **Validated only on the Oyez dataset**: The legal domain includes heterogeneous scenes like lower courts and deposition transcripts; cross-dataset generalization experiments are missing.
-   **No online human evaluation**: All experiments used simulated attorneys (via SeCom); it has not been tested with real people or professional lawyers—crucial because the key to inquisitive dialogue is whether the opponent counter-probes.
-   **Future Directions**: Update the reward model to a learned reward (trained on expert lawyer preference data); expand hierarchical action to 4 levels to support hypothetical reasoning chains; migrate the method to medical history-taking to verify generalizability.

## Related Work & Insights
-   **vs. collaborative TOD (MultiWOZ / Schema-Guided / Taskmaster)**: Those cover only "user asks, Agent answers"; this paper isolates inquisitive dialogue, providing definition, dataset, and method.
-   **vs. negotiation dialogue (Lewis 2017 Deal or No Deal)**: Negotiation involves conflicting goals with **active trade-offs**, while inquisitive dialogue involves **one-sided active probing + potentially uncooperative counterparty**. Placing it as a third category alongside negotiation adds conceptual value.
-   **vs. ArCHer (Zhou 2024)**: ArCHer also uses hierarchical Actor-Critic for multi-turn but with flat rewards; ours adds dual agents + Poincaré + triple rewards, improving Overall from 4.04 to 4.34, proving decoupled evaluation is stronger.
-   **vs. VaRMI (Shea & Yu 2023)**: VaRMI uses offline policy gradient + IS for role consistency; ours uses DDQN + conservative regularization, increasing PES from 3.93 to 4.47 with a clear gap.
-   **vs. CQL (Kumar 2020)**: CQL uses logsumexp for Q regularization; this paper simplifies it to an $R_1 - R_2$ subtraction form, which is lighter and easier to implement.
-   **Insight**: Treating "agent's internal evaluation" as an explicit module is a universal idea transferable to investigative journalism, medical history-taking, and police interrogation; hierarchical action + hyperbolic embeddings can also migrate to agent planning.

## Rating
- Novelty: ⭐⭐⭐⭐ The proposal of Inquisitive Dialogue + Dual Hierarchical RL + Poincaré action space is a clean and innovative combination.
- Experimental Thoroughness: ⭐⭐⭐ Main experiments + 4 ablation groups + multi-turn Coverage/MR + human evaluation are present; however, only the Supreme Court dataset was used, and online human evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains "inquisitive vs. collaborative vs. negotiation," with well-organized Method/Reward formulas and high readability.
- Value: ⭐⭐⭐⭐ Pushes the boundaries of proactive Conversational AI, with implications for legal, medical, and investigative dialogue systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)

</div>

<!-- RELATED:END -->
