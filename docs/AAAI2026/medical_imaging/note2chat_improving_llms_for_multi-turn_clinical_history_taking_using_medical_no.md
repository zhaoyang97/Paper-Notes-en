---
title: >-
  [Paper Note] Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes
description: >-
  [AAAI 2026][Medical Imaging][Clinical history taking] This paper proposes Note2Chat, a framework that trains LLMs for structured history taking and diagnosis using widely available medical notes rather than scarce dialogue data. Through note-driven dialogue generation, a three-stage fine-tuning strategy, and a single-turn reasoning paradigm, it substantially outperforms GPT-4o in information gathering (F1 +16.9) and diagnostic accuracy (Top-1 +21.0).
tags:
  - AAAI 2026
  - Medical Imaging
  - Clinical history taking
  - multi-turn dialogue
  - large language models
  - medical notes
  - preference optimization
  - differential diagnosis
date: 2026-05-08
content_hash: e09aa09d2b550b78
---

# Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes

**Conference**: AAAI 2026
**arXiv**: [2601.21551](https://arxiv.org/abs/2601.21551)
**Code**: [GitHub](https://github.com/zhentingsheng/Note2Chat)
**Area**: Medical AI / Clinical Dialogue
**Keywords**: Clinical history taking, multi-turn dialogue, large language models, medical notes, preference optimization, differential diagnosis

## TL;DR

This paper proposes Note2Chat, a framework that trains LLMs for structured history taking and diagnosis using widely available medical notes rather than scarce dialogue data. Through note-driven dialogue generation, a three-stage fine-tuning strategy, and a single-turn reasoning paradigm, it substantially outperforms GPT-4o in information gathering (F1 +16.9) and diagnostic accuracy (Top-1 +21.0).

## Background & Motivation

- **Importance of clinical history taking**: Medical history collection is foundational to clinical reasoning; in most cases, a correct diagnosis can be reached through history taking alone.
- **LLMs excel on static benchmarks** but degrade significantly in dynamic multi-turn diagnostic settings:
    - Diagnostic accuracy drops substantially when models must proactively ask questions and revise hypotheses based on responses.
    - Even GPT-4o performs poorly on basic dimensions such as Site (13.6%) and Severity (10.1%).
    - Models frequently fail to generate focused follow-up questions or prioritize clinically relevant details.
- **Limitations of prior work**:
    - AMIE: relies on proprietary data and models, not reproducible.
    - DoctorAgent-RL: constrained by a rigid state-action space.
    - Agent-based methods: use general-purpose models not optimized for clinical reasoning.
    - Most work **focuses on final diagnosis while neglecting history-taking quality**.
- **Data scarcity**: High-quality clinical dialogue data is extremely difficult to obtain due to privacy constraints; however, **medical notes** (e.g., the HPI section of discharge summaries) are routinely recorded and widely available.

## Method

### Core Insight

Medical notes are the "product" of history taking — clinicians distill a dynamic interview process into structured notes. Conversely, notes can serve as "silver-standard" supervisory signals for training history-taking capabilities.

### Problem Formulation

History taking is modeled as a partially observable sequential decision process:
- Patient case $x = \{dx, \mathcal{F}, cc\}$: diagnosis, set of clinical findings, chief complaint.
- At each turn $t$: the physician observes state $s_t = \{cc, h_t\}$ and selects action $a_t \in \mathcal{A}^{\text{ask}} \cup \mathcal{A}^{\text{diagnose}}$.
- Objective: maximize dialogue reward $\max_\theta \mathbb{E}_{x \sim \mathcal{P}, \pi_\theta}[R(\tau)]$.

### Data Construction Pipeline

1. **Finding extraction**: Medical findings are extracted from the HPI section of discharge notes, excluding information about subsequent tests and treatments.
2. **Decision-tree-guided dialogue generation**: A finding-to-candidate-diagnosis decision tree is constructed to guide an LLM in generating task-oriented dialogues.
3. **Review and revision**: An LLM-based critic identifies and corrects missing findings and context leakage (cases where the physician infers symptoms not disclosed by the patient).

Dataset scale (based on MIMIC-IV):
- 10 disease categories (asthma, COPD, cellulitis, heart failure, pneumonia, etc.)
- 4,972 patients, average 17.8 dialogue turns
- 8,944 synthetic dialogues, 67,077 successful rollouts, 11,403 preference pairs

### Three-Stage Fine-Tuning Strategy

**Stage 1: SFT Cold Start**
- Base model: Qwen2.5-7B
- Supervised fine-tuning on note-guided dialogues
- Establishes basic clinical reasoning and dialogue structure capabilities

**Stage 2: Self-Augmented Trajectory Sampling**
- Problem: note-guided dialogues are overly idealized (every question receives a relevant answer), causing the model to overfit.
- Solution: the SFT model engages in self-play with a simulated patient (Qwen2.5-32B).
- Trajectories with correct diagnoses and highest recall are selected and added to the training corpus.
- Produces 4,472 self-augmented dialogues.

**Stage 3: Direct Preference Optimization (DPO)**
- 15 candidate dialogues are generated per case; preference pairs are constructed based on reward scores.

**Dialogue-level reward function**:

$$R(\tau) = \text{Recall} + \frac{\text{Recall}}{\text{Recall}_{\max}} \cdot \left(1 - \frac{\text{rank}(dx, \hat{\mathbf{y}}_T)}{K}\right) - \frac{\alpha \cdot T}{2}$$

- First term: completeness of information gathering (Recall).
- Second term: diagnostic accuracy (weighted to prevent rewarding lucky guesses with low information).
- Third term: dialogue efficiency penalty ($T$ = number of turns).

Preference pairs are constructed as: scores above $\mu+\sigma$ are high-quality; scores below $\mu-\sigma$ are low-quality, yielding 11,403 preference pairs.

### Single-Turn Reasoning Paradigm (Core Innovation)

**Motivation**: Inherent limitations of multi-turn DPO:
- Long dialogue rollouts are difficult to control; early errors cascade.
- Preference signals applied to entire trajectories provide only coarse-grained supervision.
- Learning when to stop is difficult.

**Core Idea**: Reformulate multi-turn history taking as a series of single-turn reasoning problems (MDP perspective).

**Structured reasoning at each step** (via `<think>` token):
- **Summary**: structured summary of dialogue history and collected symptoms.
- **Planning**: clinical rationale for the next action (which symptom to follow up on, or when to diagnose).

**Single-turn-level process reward**:

$$R_{\text{ST}}(s_{t-1}, s_t) = \begin{cases} \mathbb{I}[f_t \in s_t \setminus s_{t-1}], & \text{if } a_t \in \mathcal{A}^{\text{ask}} \\ \text{Recall}_t \cdot (1 - \frac{\text{rank}_t}{K}), & \text{if } a_t \in \mathcal{A}^{\text{diagnose}} \end{cases}$$

- Asking action: binary reward indicating whether a new relevant finding was elicited.
- Diagnosis action: composite score based on recall and diagnostic ranking.
- **Key advantage**: enables direct comparison between asking and diagnosing actions, allowing the model to learn when to stop questioning.

## Key Experimental Results

### Experimental Setup

- Simulated patient: Qwen2.5-32B
- Evaluation model: Qwen2.5-32B
- Baselines: GPT-4o, o4-mini, Gemini-2.5-flash, DeepSeek-R1, HuatuoGPT-o1, MedGemma, DoctorAgent-RL
- Metrics: F1, Recall, Precision (information gathering) + Top-K accuracy (diagnosis)

### Main Results

| Model | F1 | Recall | Top-1 | Top-3 | #Turn |
|------|-----|--------|-------|-------|-------|
| GPT-4o | 29.2 | 33.2 | 49.0 | 67.6 | 22.9 |
| Gemini-2.5-flash | 26.6 | 35.5 | 51.4 | 73.0 | 31.9 |
| DeepSeek-R1-Qwen3-8B | 29.6 | 34.0 | 37.2 | 61.2 | 23.4 |
| HuatuoGPT-o1-8B | 0.2 | 0.1 | 19.4 | 42.8 | 2.0 |
| DoctorAgent-RL | 28.4 | 35.1 | 35.6 | - | 26.4 |
| **Note2Chat-MT** | **43.8** | **55.4** | 62.0 | 82.6 | 27.5 |
| **Note2Chat-ST** | **46.1** | 46.2 | **70.0** | **84.4** | **17.3** |

Key findings:
- Note2Chat-ST improves over the base model Qwen2.5-7B by F1 +26.5 (135.2% relative gain) and Top-1 +31.2.
- Note2Chat-ST achieves better overall performance with fewer turns (17.3 vs. 27.5).
- HuatuoGPT-o1 fails entirely (F1=0.2) because it does not ask follow-up questions and diagnoses solely from the chief complaint.

### Ablation Study

| Stage | F1 | Top-1 | Avg Δ |
|------|-----|-------|-------|
| Qwen2.5-7B base | 19.6 | 38.8 | - |
| +SFT | 35.4 | 54.8 | +13.1 |
| +SFT+Self-Aug | 41.4 | 60.8 | +19.3 |
| +SFT+Self-Aug+DPO (ST) | **46.1** | **70.0** | **+26.2** |

Each component contributes significantly: SFT establishes the baseline → self-augmentation increases diversity → DPO optimizes efficiency and preference alignment.

### SOCRATES Dimension Analysis

Recall is analyzed across symptom dimensions based on the SOCRATES mnemonic:
- GPT-4o scores extremely low on Site (13.6%) and Severity (10.1%).
- Note2Chat leads substantially on Onset, Radiation, and History dimensions.
- This demonstrates that general-purpose LLMs lack structured clinical history-taking capabilities.

### Comparison with Clinicians

On 20 test cases, Note2Chat performs comparably to clinicians in both diagnostic accuracy and information gathering.

## Highlights & Insights

1. **The "notes → dialogue" data construction paradigm is remarkably elegant**: it circumvents the dialogue data scarcity bottleneck by leveraging the abundance of available medical notes.
2. **The single-turn reasoning paradigm is the core innovation**: it decouples the multi-turn problem into independent decision steps, enabling fine-grained supervision and interpretable reasoning.
3. **The reward function design is well-crafted**: the recall-weighted diagnostic reward prevents rewarding low-information lucky guesses, while the efficiency penalty encourages conciseness.
4. **A 7B model outperforms GPT-4o**: this strongly demonstrates the value of task-specific training — a small model can surpass large general-purpose models.
5. **SOCRATES dimension analysis** provides clinically meaningful insights, revealing where LLMs most need improvement.

## Limitations & Future Work

- Validated on only 10 disease categories; real clinical settings are far broader.
- Relies on the discharge note format from MIMIC-IV; cross-institutional and cross-national generalizability is unknown.
- The gap between the simulated patient (Qwen2.5-32B) and real patients is not quantified.
- The comparison with clinicians is small-scale (only 20 cases), with insufficient statistical power.
- Decision-tree-guided generation may introduce template-like biases.
- The potential of larger models (e.g., 70B+) or reasoning models remains unexplored.

## Related Work & Insights

- **LLMs for Medical QA**: Med-PaLM, BioMistral, HuatuoGPT-o1, MedGemma
- **Multi-turn clinical dialogue**: AMIE (Google/Tu et al. 2025), DoctorAgent-RL (Feng et al. 2025)
- **Evaluation benchmarks**: CRAFT-MD (Johri et al. 2025)
- **Data generation**: self-play simulation, role-playing data augmentation
- **Preference learning**: DPO (Rafailov et al. 2023) for dialogue quality optimization

## Rating ⭐⭐⭐⭐⭐

The method is comprehensive and innovative, forming a closed loop from data construction to training strategy to inference paradigm. The note-driven approach elegantly resolves the data bottleneck, and the single-turn reasoning paradigm represents a fundamental improvement over multi-turn dialogue training. The experiments are thorough, the ablations are rigorous, and the clinician comparison adds credibility. The result of a 7B model significantly outperforming GPT-4o is impressive. This is an excellent contribution to the field of LLMs for clinical dialogue.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](../../NeurIPS2025/medical_imaging/shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_imaging/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)

<!-- RELATED:END -->
