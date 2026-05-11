---
title: >-
  [Paper Note] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA
description: >-
  [AAAI 2026][Medical Imaging][VQA] This paper proposes AlignVQA, a multi-agent debate framework for VQA confidence calibration: specialist agents generate candidate answers…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "VQA"
  - "confidence calibration"
  - "multi-agent debate"
  - "vision-language models"
  - "AlignCal loss"
  - "medical image question answering"
date: 2026-05-08
content_hash: 7d58a0a3c9f6c89c
---

# Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA

**Conference**: AAAI 2026
**arXiv**: [2511.11169](https://arxiv.org/abs/2511.11169)
**Code**: [ayushp88/AgenticCalibration](https://github.com/ayushp88/AgenticCalibration)
**Area**: Medical Imaging / Visual Question Answering
**Keywords**: VQA, confidence calibration, multi-agent debate, vision-language models, AlignCal loss, medical image question answering

## TL;DR

This paper proposes AlignVQA, a multi-agent debate framework for VQA confidence calibration: specialist agents generate candidate answers, followed by structured debate (supporting vs. opposing arguments) by generalist agents to refine confidence scores. A differentiable calibration-aware loss, AlignCal, is also introduced to minimize the upper bound of calibration error (UBCE) during training. The approach reduces ECE from 0.375 to 0.098 on VQARad and ScienceQA.

## Background & Motivation

**Background**: VQA is increasingly deployed in high-stakes domains such as medical diagnosis, autonomous driving, and visual assistance. Modern VLMs (e.g., Gemma 3, Qwen2.5-VL) continue to improve in accuracy, yet their confidence estimates remain severely overconfident.

**Limitations of Prior Work**:
   - **Overconfidence**: VLMs frequently assign high confidence to incorrect answers. For instance, Gemma 3 4B achieves an ECE of 0.375 and an MCE of 0.818 on VQARad.
   - **Safety risks**: Overconfident wrong answers may mislead clinicians in medical diagnosis and cause hazardous decisions in autonomous driving.
   - **Calibration method limitations**: Post-hoc methods such as Temperature Scaling (TS) are limited to a single coarse-grained adjustment; Focal Loss only indirectly improves calibration; label smoothing uniformly smooths all targets.
   - **Multi-agent calibration unexplored**: Existing multi-agent VQA systems target accuracy without explicitly optimizing calibration.

**Key Challenge**: Improvements in VLM accuracy do not coincide with improvements in calibration quality — more accurate models may in fact become more overconfident.

**Goal**: Enable VQA systems to produce confidence scores that more faithfully reflect the true probability of correctness, especially in high-stakes settings such as medical imaging.

**Key Insight**: Simulating human collective decision-making — through structured multi-agent debate (arguing for/against a position), models exchange arguments and revise unreliable confidence estimates. A differentiable calibration loss is theoretically derived to train better-calibrated agents.

**Core Idea**: Diverse specialist agents + structured debate + theory-driven calibration-aware loss = more reliable VQA confidence.

## Method

### Overall Architecture (Two Stages)

**Stage 1: Specialist Agent Ensemble and Stance Generation**

1. Four specialist agents with different VLM backbones are deployed: Qwen2.5-VL-3B, LLaVA-OneVision, Gemma 3 4B, and Phi-4-multimodal.
2. Each agent employs a distinct prompting strategy: Chain-of-Thought (multi-step reasoning), Self-Ask (recursive decomposition), Search-style (external retrieval), and GENREAD (structured comprehension).
3. Each agent independently produces an answer $\hat{y}_i$ and a sequence probability $p_i$ (estimated via the geometric mean of next-token probabilities).
4. GPT-3.5 is used to merge semantically equivalent but lexically distinct answers into $K$ unique stances $\{s_1, \ldots, s_K\}$.
5. For each stance, a frequency $f_k$ and average confidence $\bar{c}_k$ are computed.

**Stage 2: Generalist Agent Debate and Confidence Refinement**

1. $M$ generalist agents (Phi-4-multimodal backbone) are initialized and assigned initial stances proportional to stance frequencies.
2. Each agent constructs a supporting argument (for argument) for its assigned stance, exploring distinct reasoning paths.
3. Other agents provide feedback scored on logical consistency, factuality, clarity, and conciseness.
4. Chain-of-Verification prompting is used to check factuality; search-augmented agents verify disputed claims.
5. Each agent receives a supporting/opposing argument pair and produces a final answer $y_j' = f_j(s_j, \bar{c}_j, a_p, a_n)$.
6. The sequence probability of each agent's final response is recorded as the refined confidence.
7. The final answer is selected by majority vote, and the final confidence is the average confidence of agents supporting that stance.

### Key Designs: AlignCal Calibration-Aware Loss

**Design Motivation**: Standard calibration metrics such as ECE aggregate errors via binning, potentially masking per-sample confidence bias. The Upper Bound Calibration Error (UBCE) computes the absolute gap for each sample individually and serves as a conservative upper bound on ECE.

**UBCE Formulation**:
$$\text{UBCE} = \mathbb{E}[t(1-p_{\max}) + (1-t)p_{\max}]$$

where $t = \mathbb{I}\{\hat{y}=y\}$ is the correctness indicator and $p_{\max}$ is the highest predicted confidence.

**Differentiable Surrogate Loss**: Since the indicator function $t$ is non-differentiable, it is replaced by the model's soft belief in the correct answer $p_y$:

$$\mathcal{L}_{\text{AlignCal}}(p_y, p_{\max}) = p_y(1-p_{\max}) + (1-p_y)p_{\max}$$

**Total Loss**: $\mathcal{L}_{tot} = \mathcal{L}_{FL} + \lambda\mathcal{L}_{\text{AlignCal}}$

where $\mathcal{L}_{FL}$ is the focal loss and $\lambda=2$.

**Gradient Analysis**:
- When the model is correct but underconfident ($p_y$ high, $p_{\max}$ low): gradients push $p_{\max}$ upward.
- When the model is overconfident but incorrect ($p_{\max}$ high, $p_y$ low): gradients reduce $p_{\hat{y}}$ and increase $p_y$.
- Self-correcting feedback: improved confidence → more accurate $p_y$ → tighter surrogate → further improvement.

### Training Details

- LoRA fine-tuning: rank=8, scaling=8, dropout=0.05, injected into q_proj and v_proj only.
- 4-bit quantization (BitsAndBytes).
- VQARad: 6 epochs; ScienceQA: 10 epochs.
- Batch size=2, AdamW optimizer, lr=2e-4.
- NVIDIA A100 40GB GPU.

## Experiments

### Datasets

| Dataset | Samples | Type | Description |
|--------|--------|------|------|
| ScienceQA | 21,208 | Multimodal MCQ | Multi-discipline science questions |
| VQARad | 3,515 | Medical VQA | Radiology Yes/No QA |

### Calibration Issues in State-of-the-Art VLMs (Baselines)

| Model | ScienceQA ECE↓ | VQARad ECE↓ | VQARad MCE↓ |
|------|----------------|-------------|-------------|
| LLAVA OneVision | 0.335 | 0.232 | 0.286 |
| Gemma 3 4B | 0.398 | **0.375** | **0.818** |
| Qwen2.5-VL-3B | 0.302 | 0.295 | 0.297 |
| Phi-4-multimodal | 0.574 | 0.134 | 0.425 |

All VLMs exhibit severe calibration issues, with Gemma 3 reaching an ECE of 0.375 on VQARad.

### Main Results

**Comparison on VQARad**

| Method | ACC↑ | ECE↓ | ACE↓ | MCE↓ |
|------|------|------|------|------|
| Gemma 3 4B (baseline) | 59.4% | 0.375 | 0.208 | 0.818 |
| Agentic Framework | 65.7% | 0.146 | 0.144 | 0.820 |
| Agentic + TS | 65.7% | 0.117 | 0.114 | 0.765 |
| Agentic + DC | 65.7% | 0.041 | 0.097 | 0.113 |
| Agentic + FL | 68.5% | 0.073 | 0.116 | 0.393 |
| **Agentic + AlignCal + FL** | **68.2%** | **0.098** | **0.095** | **0.267** |

**Comparison on ScienceQA**

| Method | ACC↑ | ECE↓ | ACE↓ | MCE↓ |
|------|------|------|------|------|
| Gemma 3 4B (baseline) | 71.0% | 0.398 | 0.398 | 0.464 |
| Agentic Framework | 72.8% | 0.270 | 0.265 | 0.438 |
| **Agentic + AlignCal + FL** | **76.1%** | **0.055** | **0.110** | **0.331** |

### Key Findings

1. **Debate framework is effective**: Multi-agent debate alone (without AlignCal) reduces ECE from 0.375 to 0.146 on VQARad (−61%).
2. **AlignCal is effective**: Fine-tuning Gemma 3 with AlignCal alone reduces ECE from 0.232 to 0.058 on ScienceQA (−75%).
3. **Combined approach performs best**: AlignCal-fine-tuned agents participating in debate further reduce ECE — to 0.055 on ScienceQA and 0.098 on VQARad.
4. **AlignCal vs. other training-time calibration**: AlignCal + FL substantially outperforms Focal Loss alone (ECE: 0.055 vs. 0.180 on ScienceQA) and Label Smoothing (ECE: 0.055 vs. 0.186).
5. **Post-hoc calibration comparison**: Dirichlet Calibration achieves ECE of 0.041 on VQARad but is inapplicable to ScienceQA (other-option probabilities unavailable).
6. **Accuracy also improves**: Accuracy increases from 71.0% to 76.1% on ScienceQA, demonstrating that calibration and accuracy are not in conflict.

## Highlights & Insights

1. **Theory-driven loss design**: AlignCal is not heuristic — it is derived mathematically from UBCE via the plug-in principle, replacing the non-differentiable correctness indicator with a model-estimated soft belief, offering rigorous theoretical grounding.
2. **Self-correcting feedback mechanism**: Gradient analysis reveals an elegant self-correcting loop — improved confidence → more accurate $p_y$ → tighter surrogate loss → further improvement.
3. **Debate framework is intuitively sound**: The design mirrors human collective decision-making; high-confidence incorrect answers are more likely to be refuted during structured argument exchange.
4. **Diversity is key to calibration**: Four distinct VLM backbones combined with four prompting strategies ensure opinion diversity and mitigate collective bias.
5. **First multi-agent approach to VQA calibration**: No prior work has applied multi-agent debate to confidence calibration in VQA.

## Limitations & Future Work

1. **High computational cost**: Four VLM backbones, the debate process, and GPT-3.5-based semantic equivalence judgments introduce significant inference latency and API overhead.
2. **Limited MCE improvement**: UBCE is an expectation-level upper bound and does not directly constrain worst-case behavior (MCE); MCE remains as high as 0.267–0.820 on VQARad.
3. **Evaluation limited to MCQ settings**: The framework is validated only on multiple-choice VQA; calibration for open-ended VQA remains untested.
4. **Small dataset scale**: VQARad contains only 3,515 questions; while ScienceQA is larger, it is not medically specialized.
5. **Dependency on GPT-3.5**: Semantic equivalence judgment relies on an external model, adding an inference dependency.
6. **Fixed VLM backbones**: Although the framework is claimed to be model-agnostic, it is evaluated only on four small VLMs with 3–5B parameters.
7. **Effect of debate rounds and agent count**: This analysis is relegated to the appendix, and ablation coverage is insufficient.

## Related Work & Insights

- **Training-time calibration**: Label Smoothing, Focal Loss, MMCE (kernel-based calibration penalty).
- **Post-hoc calibration**: Temperature Scaling, Dirichlet Calibration, Platt Scaling.
- **LLM multi-agent calibration**: Collaborative Calibration (group dialogue with shared predictions and reasoning).
- **VQA calibration**: Whitehead (selective answering), GLEN (model simplification + focal loss), IVON (Bayesian variational fine-tuning).
- **Multi-agent VQA**: Top-Down Reasoning (responder + seeker + integrator), ARE (action reasoning).

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty: ⭐⭐⭐⭐⭐ — AlignCal loss is elegantly derived theoretically; multi-agent calibration is a novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive multi-method comparisons, but datasets are limited.
- Writing Quality: ⭐⭐⭐ — Theoretical derivations are detailed but some content is redundant.
- Value: ⭐⭐⭐ — Inference overhead is substantial; AlignCal loss alone is directly applicable for fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
