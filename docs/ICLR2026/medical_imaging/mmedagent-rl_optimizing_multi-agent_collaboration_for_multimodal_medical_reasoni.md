---
title: >-
  [Paper Note] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning
description: >-
  [ICLR2026][Medical Imaging][multi-agent collaboration] This paper proposes MMedAgent-RL, a multi-agent system that simulates clinical consultation workflows (triage → specialist → attending physician) optimized via reinforcement learning. The core innovation is Curriculum-guided Multi-Agent Reinforcement Learning (C-MARL) with entropy-aware exploration, enabling the attending physician agent to adopt differentiated explore–exploit strategies when faced with correct, conflicting, or erroneous specialist opinions. The system achieves state-of-the-art performance on 5 medical VQA benchmarks spanning both in-domain and out-of-domain settings.
tags:
  - ICLR2026
  - Medical Imaging
  - multi-agent collaboration
  - reinforcement learning
  - medical VQA
  - curriculum learning
  - GRPO
  - clinical reasoning
date: 2026-05-08
content_hash: c08a6a9085651c40
---

# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning

**Conference**: ICLR2026
**arXiv**: [2506.00555](https://arxiv.org/abs/2506.00555)
**Code**: Not released
**Area**: Medical Imaging
**Keywords**: multi-agent collaboration, reinforcement learning, medical VQA, curriculum learning, GRPO, clinical reasoning

## TL;DR
This paper proposes MMedAgent-RL, a multi-agent system that simulates clinical consultation workflows (triage → specialist → attending physician) optimized via reinforcement learning. The core innovation is Curriculum-guided Multi-Agent Reinforcement Learning (C-MARL) with entropy-aware exploration, enabling the attending physician agent to adopt differentiated explore–exploit strategies when faced with correct, conflicting, or erroneous specialist opinions. The system achieves state-of-the-art performance on 5 medical VQA benchmarks spanning both in-domain and out-of-domain settings.

## Background & Motivation
- **State of the Field**: Medical image diagnosis spans multiple subspecialties (radiology, pathology, oncology, etc.), which a single Med-LVLM cannot adequately cover.
- **Limitations of Prior Work**: Static multi-agent frameworks such as MedAgents and MDAgents rely on fixed GP→Specialist→GP pipelines with pre-defined, non-learnable interaction patterns. Specialist outputs are not always reliable and may introduce noise or misleading signals; majority voting can suppress correct minority opinions.
- **Root Cause**: The attending physician must learn **when to trust specialist consensus (exploit) and when to challenge it and reason independently (explore)**.
- **Starting Point**: Works such as DeepSeek-R1 demonstrate that RL can substantially enhance LLM reasoning, yet RL optimization for multi-agent medical collaboration remains unexplored.

## Method

### Overall Architecture: Clinical Consultation Simulation (GP→Specialists→GP)
Two GP agents are trained via staged RL on top of Qwen2.5-VL:

1. **Triage Doctor**: Selects the appropriate specialty based on the input image and text.
2. **Specialists**: Portrayed by strong models (e.g., GPT), providing specialist diagnostic opinions.
3. **Attending Physician**: Integrates specialist opinions with its own knowledge to make the final decision.

### Key Design 1: RL Optimization of the Triage Doctor
- Dataset-provided image modality labels serve as ground truth (e.g., histology slide → Pathologist).
- Seven candidate specialties: Pathologist, Radiologist, Surgeon, Oncologist, Endocrinologist, Ophthalmologist, Dermatologist.
- Optimized with GRPO; reward function: $R = R_{format} \in \{0, 0.5\} + R_{accuracy} \in \{0, 1\}$.
- Improves not only triage accuracy but also the reasoning process (justifying triage decisions).

### Key Design 2: Curriculum-Guided Entropy-Aware RL (C-MARL)
**Core Idea**: Training difficulty is defined by the reliability of specialist opinions, and the attending physician is trained in progressive stages.

**Data Stratification** (based on specialist accuracy $s = \text{Acc}(y_d, y^*)$):
- **Easy ($s=1$)**: All specialists answer correctly.
- **Medium ($0<s<1$)**: Partial agreement; conflicting opinions exist.
- **Hard ($s=0$)**: All specialists are wrong, forming a misleading consensus.

**Dynamic Entropy Regulation** — a curriculum-dependent entropy regularization term is added to the standard GRPO objective:

$$\mathcal{J}_{C\text{-}MARL}(\theta) = \mathbb{E}[\mathcal{J}_{GRPO}(\theta) + \gamma_s \cdot H_t(\pi_\theta)]$$

- $\gamma_{easy} \approx 0$: No additional exploration needed in the Easy stage; the model focuses on exploiting reliable specialist knowledge.
- $\gamma_{medium} > 0$: Moderate exploration is encouraged in the Medium stage to prevent overconfidence when facing conflicting information.
- $\gamma_{hard} \gg \gamma_{medium}$: Strong exploration is enforced in the Hard stage to compel the model to overcome misleading specialist consensus.

**Training Order**: Easy → Medium → Hard, following a curriculum from simple to difficult.

### Key Design 3: Theoretical Analysis
- A convergence proof (Theorem 4.1) is provided demonstrating the advantage of curriculum learning over standard SGD.
- The total training time under curriculum learning depends on the sum of distances between optimal policies in adjacent stages: $\sum_{j=0}^{J-1}\log\|\theta_j^\star - \theta_{j+1}^\star\|_2^2$.
- An effective curriculum ensures small inter-stage distances, with each stage's solution serving as a warm start for the next.
- Standard SGD is shown to be unable to converge to the optimal policy under equivalent conditions (lower-bound proof).

## Key Experimental Results

### Main Results (Accuracy, %)

| Model | VQA-RAD | SLAKE | PathVQA | In-Domain Avg | OmniMedVQA | MMMU-Med | Out-of-Domain Avg |
|-------|---------|-------|---------|---------------|------------|----------|-------------------|
| GPT-4o | 61.0 | 75.5 | 69.4 | 68.6 | 68.5 | 69.7 | 69.1 |
| Qwen2.5-VL-7B | 61.8 | 64.7 | 60.5 | 62.3 | 60.8 | 56.6 | 58.7 |
| MedVLThinker-7B | 63.7 | 67.8 | 65.2 | 65.6 | 62.4 | 57.0 | 59.7 |
| MDAgents | 66.8 | 68.2 | 65.4 | 66.8 | 58.2 | 52.3 | 55.1 |
| **MMedAgent-RL (7B)** | **71.5** | **76.2** | **72.3** | **73.3** | **73.3** | **71.9** | **72.6** |
| + Test-Time Scaling | 73.9 | 80.1 | 74.3 | 76.1 | 79.6 | 73.5 | 76.6 |

### Ablation Study

| Configuration | VQA-RAD | SLAKE | OmniMedVQA | MMMU-Med |
|---------------|---------|-------|------------|----------|
| Full MMedAgent-RL | 71.5 | 76.2 | 73.3 | 71.9 |
| w/o Triage | 66.3 | 69.9 | 66.2 | 59.3 |
| w/o Specialists | 65.8 | 67.8 | 64.4 | 54.2 |
| w/o C-MARL | 63.5 | 65.5 | 57.9 | 50.2 |
| + Easy only | 64.7 | 69.3 | 68.2 | 58.0 |
| + Easy + Medium | 69.4 | 76.9 | 70.8 | 68.8 |
| + Easy + Medium + Hard | 71.5 | 76.2 | 73.3 | 71.9 |

### Key Findings
1. **C-MARL contributes most**: Removing C-MARL results in an average drop of 18.6%, identifying it as the most critical component.
2. **Each curriculum stage contributes incrementally**: Easy→Medium→Hard yields progressively cumulative gains, validating the curriculum design.
3. **The Hard stage is crucial**: The largest improvements are observed under the "Hard" scenario where all specialists are incorrect (+20%), demonstrating that the model learns to resist blind conformity.
4. **Strong out-of-domain generalization**: OmniMedVQA (+13%) and MMMU-Med (+15%) both substantially outperform the backbone and surpass GPT-4o.
5. **Triage accuracy propagates through the pipeline**: The optimized triage doctor establishes the foundation for accurate specialist consultation downstream (+3%).
6. **Test-Time Scaling (TTS) further boosts performance**: Majority voting yields an additional gain of 4.5%.

## Highlights & Insights
- The entropy regulation mechanism in C-MARL is elegant: it addresses the specialist noise problem in multi-agent collaboration from an explore–exploit trade-off perspective.
- The combination of curriculum learning and RL is theoretically grounded (convergence proof) rather than purely heuristic.
- A 7B model surpasses GPT-4o on multiple benchmarks, demonstrating the substantial potential of RL-based optimization.
- The experimental design is comprehensive, covering in-domain/out-of-domain evaluation, ablation studies, specialty selection analysis, and difficulty stratification.

## Limitations & Future Work
- Specialists are portrayed by closed-source models such as GPT, resulting in high deployment costs and dependence on third-party APIs.
- The triage specialty set is fixed at 7 categories, which may not cover all clinical scenarios.
- The three-tier difficulty stratification relies on ground-truth labels, which are unavailable at inference time.
- Evaluation is limited to multiple-choice VQA; open-ended clinical reasoning and report generation are not addressed.
- The theoretical analysis rests on strong assumptions (e.g., strong convexity) that may not hold for practical deep networks.

## Related Work & Insights
- **Medical VLMs**: Single-agent models including LLaVA-Med, HuatuoGPT-Vision, and VILA-M3.
- **Medical Multi-Agent Systems**: Agent Hospital, MedAgents, and MDAgents — all rely on static, pre-defined pipelines.
- **RL-Enhanced Reasoning**: GRPO-based post-training paradigms such as DeepSeek-R1 and VLM-R1.
- **Curriculum Learning**: Progressive training from easy to hard (Bengio et al., 2009).

## Rating
⭐⭐⭐⭐⭐ (5/5)

The method is elegantly designed; the entropy regulation strategy in C-MARL is both intuitively well-motivated and theoretically supported. The experiments are comprehensive and convincing, and the result of a 7B model surpassing GPT-4o is impressive. The integration of curriculum learning and RL establishes a new paradigm for multi-agent collaborative reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](../../AAAI2026/medical_imaging/mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](../../NeurIPS2025/medical_imaging/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)
- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)

<!-- RELATED:END -->
