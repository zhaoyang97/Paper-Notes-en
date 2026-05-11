---
title: >-
  [Paper Note] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare
description: >-
  [CVPR 2026][LLM Agent][Medical software automation] This paper proposes the CareFlow benchmark (1,050 long-horizon medical software workflow tasks, 8–24 steps…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "Medical software automation"
  - "multi-agent framework"
  - "Actor-Critic"
  - "long-horizon GUI interaction"
  - "dual memory mechanism"
date: 2026-05-08
content_hash: 58921e14e5e7ac0f
---

# CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare

**Conference**: CVPR 2026
**arXiv**: [2603.24157](https://arxiv.org/abs/2603.24157)
**Code**: Available (CarePilot project page)
**Area**: LLM Agent / Healthcare Automation
**Keywords**: Medical software automation, multi-agent framework, Actor-Critic, long-horizon GUI interaction, dual memory mechanism

## TL;DR
This paper proposes the CareFlow benchmark (1,050 long-horizon medical software workflow tasks, 8–24 steps, covering four systems: DICOM/3D Slicer/EMR/LIS) and the CarePilot framework (based on the Actor-Critic paradigm, integrating tool grounding and a dual memory mechanism), achieving approximately 15% higher task accuracy than GPT-5 on CareFlow.

## Background & Motivation

**Background**: Multimodal agents have made progress in Android, desktop, and web environments (Mind2Web, SeeAct, UI-TARS, etc.), but no standardized benchmark exists for medical software.

**Unique Challenges of Medical Software**: (1) Routine clinical operations require chaining 10–15 dependent steps (open study → configure view → annotate → export → update record); (2) platforms are highly heterogeneous and frequently updated; (3) strict data integrity, audit trail, and privacy compliance requirements; (4) institution-specific UI layouts make agents that overfit to surface-level layouts brittle.

**Limitations of Prior Work**: (1) No publicly available long-horizon benchmark for medical software interaction exists; (2) existing VLMs (GPT-4o, Gemini, etc.) perform poorly on medical GUIs—step-level accuracy is acceptable but task completion rates are extremely low.

**Key Insight**: Construct the first long-horizon medical software benchmark and design an Actor-Critic agent with tool grounding and memory mechanisms.

**Core Idea**: The Actor predicts the next action → the Critic evaluates and corrects → dual memory (short-term + long-term) maintains workflow context → iterative simulation training improves robustness.

## Method

### Overall Architecture
Natural language goal + current screenshot → Tool Grounding (UI detection + OCR + zoom + template matching) → Actor reads dual memory + grounding signal → predicts semantic action → Critic evaluates → provides corrective feedback or approves execution → updates memory → next step.

### Key Designs

1. **CareFlow Benchmark (Four-Stage Annotation Pipeline)**:

   - **(i) Seed Task Design**: Collaborates with domain experts to map usage patterns and operational constraints of each software system and extract a core task list.
   - **(ii) Diversity Expansion**: Controlled substitution (e.g., "MRI report" → "X-ray report"), parameter adjustment, and step addition/removal.
   - **(iii) Step-by-Step GUI Annotation**: Screenshot per step with precise semantic action labels for the next step.
   - **(iv) Quality Filtering**: Temporal consistency + task completeness + instruction clarity; Cohen's $\kappa = 0.78$.
   - Coverage: Weasis/Orthanc (DICOM), 3D Slicer (annotation), OpenEMR (EMR), OpenHospital (LIS).
   - Scale: 1,050 tasks (735 train + 315 test, including 50 OOD), 8–24 steps per task, 6 action types (CLICK/SCROLL/ZOOM/TEXT/SEGMENT/COMPLETE).

2. **Tool Grounding (Four Perception Modules)**:

   - **UI Object Detection** (open-vocabulary): Given a text query, returns the bounding box of the target UI element.
   - **Zoom/Crop**: Magnifies and inspects small controls.
   - **OCR**: Extracts text labels (series names, patient fields, order numbers).
   - **Template/Icon Matching**: Robust to theme, zoom, and language variations.
   - Outputs from all four modules are aggregated into a unified grounding signal $\phi_t$.

3. **Dual Memory Mechanism**:

   - **Short-term memory** $\mathcal{M}_t^S = f^S(x_{t-1}, a_{t-1}, r_{t-1})$: screenshot, action, and Critic feedback from the previous step.
   - **Long-term memory** $\mathcal{M}_t^L = f^L(\mathcal{M}_{t-1}^L, \mathcal{M}_t^S, \phi_t)$: compact trajectory embedding integrating historical states, actions, and outcomes.
   - Action prediction is conditioned on both memory streams: $a_t = \pi_\theta(g, x_t, \mathcal{M}_t^S, \mathcal{M}_t^L)$.
   - **Design Motivation**: Errors accumulate in long-horizon workflows; the dual memory mechanism balances rapid short-term response with long-term context retention.

4. **Actor-Critic Framework**:

   - Both Actor and Critic are instantiated from Qwen-VL 2.5-7B, differing only in input conditioning and functional role.
   - Actor: observes current interface + instruction + grounding signal + memory → predicts semantic action.
   - Critic: evaluates the Actor's proposal → provides corrective feedback or approves execution → updates dual memory.
   - Iterative simulation training: during training, the Critic compares against reference trajectories; during inference, it relies on execution results or verifier feedback.

### Loss & Training
The task is formulated as sequential decision-making: $\hat{a}_{1:T} = \mathbb{1}[V(g, x_{1:T}, a_{1:T}) = 1]$, where verifier $V$ determines whether the workflow has been successfully completed.

## Key Experimental Results

### Main Results (CareFlow)

| Model | Weasis SWA/TA | 3D Slicer SWA/TA | OpenEMR SWA/TA | Average SWA/TA |
|---|:-:|:-:|:-:|:-:|
| Qwen2.5 VL 7B | 58.6/1.3 | 61.4/1.7 | 63.2/1.7 | 57.2/1.8 |
| Llama 4 Maverick | 88.2/18.7 | 71.6/3.4 | 78.0/25.7 | 80.5/19.2 |
| GPT-4o | 85.3/20.0 | 77.5/27.4 | 85.1/27.5 | 83.1/25.4 |
| GPT-5 | 88.7/31.3 | 81.4/37.9 | 83.8/31.3 | 85.2/36.2 |
| **CarePilot (7B)** | **90.4/40.0** | **82.1/54.8** | — | **SOTA** |

CarePilot (based on a 7B model) surpasses GPT-5 by approximately 15% in Task Accuracy.

### Ablation Study

| Configuration | Avg SWA | Avg TA | Note |
|---|---|---|---|
| Qwen-VL 7B (baseline) | 57.2 | 1.8 | No agent framework |
| + Tool Grounding | +gain | +gain | Perception enhancement |
| + Dual Memory | +gain | +gain | Context retention |
| + Actor-Critic | **SOTA** | **SOTA** | Corrective feedback is critical |

### Key Findings
- **Large gap between step-level accuracy and task accuracy**: GPT-4o achieves 83% step-level accuracy but only 25% task completion—error accumulation in long-horizon settings causes a sharp drop in task completion rates.
- The Critic correction mechanism in CarePilot effectively mitigates error accumulation, improving task completion rate from a baseline of ~2% to 40%+.
- 3D Slicer (medical annotation) is the most challenging software, requiring fine-grained spatial operations (segmentation/measurement); CarePilot shows the largest improvement on this subset.
- On the OOD test set (50 tasks), CarePilot still achieves a 3.38% improvement, demonstrating a degree of generalization.
- Among the tool grounding modules, OCR is most critical for EMR systems, which contain numerous text fields requiring precise recognition.

## Highlights & Insights
- **First long-horizon medical GUI benchmark**: CareFlow fills a gap in evaluation for AI automation of medical software. The four-stage annotation pipeline is rigorous, with seed tasks derived from the actual daily operations of clinical practitioners.
- **A 7B model outperforms GPT-5**: CarePilot demonstrates that an appropriate agent framework matters more than a larger model—tool augmentation, memory, and corrective feedback allow a 7B model to surpass GPT-5 on domain-specific tasks.
- **Medical adaptation of Actor-Critic**: The Critic not only evaluates correctness but also provides feedback on *how to correct*—particularly important in safety-critical medical settings.
- **Step-level accuracy ≠ task success**: This finding serves as a warning for all long-horizon agent research—single-step metrics alone are insufficient.

## Limitations & Future Work
- CareFlow covers only five open-source medical software systems; generalization to commercial systems (Epic, Cerner) remains to be validated.
- Using the same base model for both Actor and Critic may lead to consistent blind spots; employing different models for each role could be beneficial.
- Current iterative simulation training relies on reference trajectories; real-world deployment requires online learning approaches that do not depend on reference trajectories.
- Safety evaluation is insufficient—erroneous operations in medical settings can have serious consequences.

## Related Work & Insights
- **vs. WebArena/AppWorld**: General desktop/web agent benchmarks that do not address the specialized requirements of the medical domain.
- **vs. Voyager/Reflexion**: Their memory and reflection mechanisms inspired CarePilot's design, but these systems target gaming or general-purpose scenarios.
- **vs. Mind2Web/SeeAct**: Short-horizon GUI agents that cannot handle the long dependency chains present in medical workflows.

## Rating
- Novelty: ⭐⭐⭐⭐ Domain application innovation (first medical GUI benchmark); framework design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-system, multi-baseline (including GPT-5), OOD testing, thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Benchmark construction process is detailed; framework description is clear.
- Value: ⭐⭐⭐⭐⭐ Direct applicability to healthcare AI automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](../../ICLR2026/llm_agent/the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_multiagent_nerf_paper_to_code.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](../../AAAI2026/llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](../../ICLR2026/llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)

</div>

<!-- RELATED:END -->
