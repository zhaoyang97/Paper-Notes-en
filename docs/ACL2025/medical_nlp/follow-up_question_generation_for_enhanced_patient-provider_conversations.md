---
title: >-
  [Paper Note] Follow-up Question Generation for Enhanced Patient-Provider Conversations
description: >-
  [ACL 2025][Medical LLM][Follow-up question generation] This paper proposes FollowupQ, a multi-agent framework that integrates EHR reasoning, differential diagnosis, and message clarification agents to automatically generate personalized follow-up questions for asynchronous patient-provider conversations. FollowupQ improves the RIM score by 17% and 5% on real and semi-synthetic datasets, respectively, compared to baselines, and reduces the need for clinicians to send additiona…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Follow-up question generation"
  - "asynchronous patient-provider conversations"
  - "multi-agent framework"
  - "electronic health records"
  - "differential diagnosis"
date: 2026-05-08
content_hash: ca71c459b1447378
---

# Follow-up Question Generation for Enhanced Patient-Provider Conversations

**Conference**: ACL 2025  
**arXiv**: [2503.17509](https://arxiv.org/abs/2503.17509)  
**Code**: Not released (Dataset FollowupBench is publicly available)  
**Authors**: Joseph Gatto, Parker Seegmiller, Timothy Burdick, Inas S. Khayal, Sarah DeLozier, Sarah M. Preum  
**Institutions**: Dartmouth College  
**Area**: Medical NLP  
**Keywords**: Follow-up question generation, asynchronous patient-provider conversations, multi-agent framework, electronic health records, differential diagnosis

## TL;DR

This paper proposes FollowupQ, a multi-agent framework that integrates EHR reasoning, differential diagnosis, and message clarification agents to automatically generate personalized follow-up questions for asynchronous patient-provider conversations. FollowupQ improves the RIM score by 17% and 5% on real and semi-synthetic datasets, respectively, compared to baselines, and reduces the need for clinicians to send additional information-gathering messages by 34%.

## Background & Motivation

**Unique Characteristics of Asynchronous Medical Conversations**:
   - In telemedicine and patient portals, patient-provider communication occurs via asynchronous messages, differing from real-time multi-turn dialogue.
   - Patients often assume that providers are already familiar with their background, leading to incomplete information in their messages.
   - Providers must generate multiple follow-up questions at once (rather than asking sequentially) to minimize additional communication rounds.

**Limitations of Prior Work**:
   - Existing research on follow-up question generation (such as on social media or in conversational surveys) mostly focuses on generating questions one at a time.
   - Real-time conversation research does not consider the unique requirements of the asynchronous setting, where real-time follow-up is impossible.
   - Direct generation of follow-up questions by LLMs performs poorly: even when prompted to generate 10 times the number of questions written by clinicians, they still fail to cover all the questions actually asked by clinicians.

**Clinical Demand**:
Asynchronous message processing is a significant driver of clinician burnout, and automated follow-up question generation can help alleviate this workload.

## Method

### Overall Architecture

FollowupQ is a multi-agent framework. It takes a patient message $T$ and EHR records $C = \{A, H, M\}$ (demographics, history, and medications) as input. Through three types of agents, it generates a set of follow-up questions $\hat{Q}$. Optional deduplication and Top-k selection can be applied to control the output size.

### Three Core Agent Types

#### 1. EHR Reasoning Agents (2 Agents)

- **Medical History Reasoning Agent**: Extracts the most relevant information $I_{hist}$ from the patient's medical history $H$ based on the current message, and then generates targeted follow-up questions $\hat{Q}_{hist}$ using this information.
- **Medication Reasoning Agent**: Extracts relevant drug information $I_{med}$ from the medication list $M$ and generates medication-related follow-up questions $\hat{Q}_{med}$.
- **Key Designs**: A two-step approach that performs information extraction before question generation. This prevents irrelevant EHR information from interfering with the generation of follow-up questions.

#### 2. Differential Diagnosis Agent

- **Diagnosis Generation**: Generates $k$ potential diagnoses under both best-case and worst-case scenarios, respectively: $D_{diff} = f(T, P_{best}, k) \cup f(T, P_{worst}, k)$.
- **Follow-up Question Generation**: For each potential diagnosis $d_i$, generates follow-up questions needed to rule out that diagnosis: $\hat{Q}_{d_i} = f(T, d_i, P_{rule-out}, k)$.
- **Design Motivation**: Simulates the clinical thought process of a clinician—first identifying potential diagnoses, and then asking questions to rule out other possibilities.

#### 3. Message Clarification Agents (4 Agents)

- **Symptom Inquiring Agent**: Extracts symptoms from the messages and generates detailed questions (e.g., the specific location of abdominal pain).
- **Self-treatment Agent**: Queries OTC medications or self-treatment methods the patient is currently using.
- **Temporal Reasoning Agent**: Clarifies the timeline of symptoms (duration, frequency, etc.).
- **Ambiguous Message Agent**: Requests more clarification on vague expressions.

### Total Question Set Generation

$$\hat{Q}_p = \hat{Q}_{D_{diff}} \cup \hat{Q}_{EHR} \cup \hat{Q}_{clar}$$

### Question Filtering (Optional)

1. **Deduplication**: Uses an LLM to identify semantically duplicate questions.
2. **Top-k Selection**: The agent selects the $k$ most critical questions from the deduplicated list to present to the patient.

### Evaluation Metrics

1. **Requested Information Match (RIM)**: $\text{RIM}(Q, \hat{Q}) = |Q \cap \hat{Q}| / |Q|$, which measures how well the system-generated questions cover the questions actually asked by real clinicians. It does not penalize additional generated questions.
2. **Message Reduction % (MR%)**: The proportion of samples with $\text{RIM} = 1.0$, representing the percentage of cases where the clinician's questions are completely covered.

### LLM-as-Judge Semantic Matching

A fine-tuned PHI-4-14B is used as the Judge to determine whether the generated questions and ground truth questions request the same information (even if phrased differently). It achieves a macro F1 of 0.87 on the test set.

## Key Experimental Results

### Datasets

| Dataset | Type | Sample Size | Avg. Questions / Sample | Features |
|--------|------|:------:|:---:|------|
| FB-Real | Real patient messages + EHR | 150 | 3.4 | Contains PHI, not public |
| FB-Synth | Semi-synthetic messages + EHR | 250 | 9.3 | 2300+ follow-ups, public |

### Main Results (FB-Real)

| Method | RIM ↑ | Avg. Generated Questions |
|------|:---:|:---:|
| 0-shot (Llama3-8b) | ~0.40 | ~10 |
| Few-shot (Llama3-8b) | ~0.40 | ~12 |
| 40-question (Llama3-8b) | ~0.45 | 40 |
| Long-Thought (DeepSeek R1) | ~0.48 | ~15 |
| **Ours (Llama3-8b)** | **0.62** | 36 |
| Ours (Llama3-8b-Aloe) | 0.64 | >36 |

**Key Findings**:
FollowupQ improves upon the zero/few-shot baselines by approximately 22 percentage points. Even when the baseline LLM is prompted to generate more than 10 times the number of questions (40 vs. an average of 3.4 by clinicians), it still fails to match FollowupQ's performance. The main issue is diversity rather than quantity.

### Workload Reduction Performance

| Method | MR% (RIM=1.0 ratio) ↑ |
|------|:---:|
| Best Baseline | ~15% |
| **FollowupQ (Llama3-8b)** | **34%** |

FollowupQ fully covers all clinician follow-up questions in 34% of patient messages, meaning clinicians would not need to send additional information-gathering messages in these scenarios.

### Agent Contribution Analysis (FB-Real)

| Agent Type | RIM Contribution |
|------------|:---:|
| Differential Diagnosis (Worst-case) Agent | Largest contribution |
| Medication Reasoning Agent | ~10% |
| Temporal Clarification Agent | Significant contribution |
| Ambiguous Message Agent | Significant contribution |
| EHR Reasoning Agent | ~10% |

**Insights**:
The worst-case differential diagnosis agent contributes the most, reflecting the core clinical motivation for follow-up questions: ruling out severe conditions. The EHR agents contribute about 10%, confirming the necessity of personalized EHR reasoning.

### Post-Filtering Results

- 36 questions $\rightarrow$ 22 after deduplication (RIM drops from 0.62 to 0.57)
- 22 after deduplication $\rightarrow$ Top-10 selection (RIM drops from 0.57 to 0.42)
- The reason for the performance drop is not poor question quality, but rather the difficulty in modeling the individual clinical preferences of specific clinicians.

### FB-Synth Results

- FollowupQ (Llama3-8b) improves upon the nearest baseline by 5 percentage points.
- For Qwen-32b, FollowupQ still outperforms the baseline, albeit with a more modest gain.

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to systematically define the task of asynchronous patient-provider follow-up question generation, distinguishing it from general information acquisition in synchronous dialogue.
2. **Modeling Clinical Thinking**: The three types of agents correspond to three clinical reasoning processes of providers (checking EHRs, differential diagnosis, and message comprehension), offering clinical interpretability.
3. **The "Quantity $\ne$ Quality" Finding**: Forcing LLMs to generate more questions does not solve the problem—diversity and clinical relevance are much more important.
4. **Sound RIM Metric Design**: Not penalizing extra questions aligns with the clinical practice of "asking more is better than omitting details".
5. **High Practical Value**: The 34% message reduction rate indicates real and significant workload relief for providers.

## Limitations & Future Work

1. Due to secure computing environment constraints, only a limited set of LLMs (Llama3-8b/Aloe, Qwen-32b) were tested, whereas stronger models like GPT-4 were not evaluated.
2. The data originates from a single rural community hospital, which may introduce bias in patient demographics and provider preferences.
3. Different providers may generate different follow-up questions for the same patient message, making the ground truth highly subjective.
4. The performance of Top-k selection is limited by the inability to model individual clinical preferences of specific providers.
5. The application of this framework to synchronous dialogue scenarios has not yet been explored.

## Related Work & Insights

- **Follow-up Question Generation**: Meng et al. (2023) and Liu et al. (2025) study follow-up questions in social media, but only generate one question at a time; Liu et al. (2024) explore a similar setting on an extremely small scale (n=7).
- **Medical Consultation Dialogue**: Winston et al. (2024) and Li et al. (2024) study LLM consultation in synchronous dialogue, without considering asynchronous scenarios and EHR data.
- **Multi-Agent Medical Systems**: MedAgents (Tang et al., 2024), RareAgents (Chen et al., 2024), and MDAgents (Kim et al., 2024) are used for clinical decision support, but not for information gathering.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ First to systematically define the task of asynchronous medical follow-up question generation, with a well-designed multi-agent framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on both real and synthetic datasets, compared against multiple baselines, with detailed agent contribution and post-filtering ablation studies.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses clinical pain points regarding asynchronous message overload; the 34% message reduction rate has real-world clinical significance.
- **Writing Quality**: ⭐⭐⭐⭐ Well-articulated problem motivation and clinical background, with fully justified evaluation metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)
- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2025\] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)
- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)

</div>

<!-- RELATED:END -->
