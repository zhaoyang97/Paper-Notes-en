---
title: >-
  [Paper Note] Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence
description: >-
  [ICML2026][Medical Imaging][Medical Multi-Agent Systems] HetMedAgent organizes generalist LLMs, modality-specific specialist models, and clinicians into a heterogeneous multi-agent system. Through conflict-aware evidence…
tags:
  - "ICML2026"
  - "Medical Imaging"
  - "Medical Multi-Agent Systems"
  - "Specialist Models"
  - "Clinical Decision Support"
  - "Uncertainty Routing"
  - "Evidence Fusion"
date: 2026-05-08
content_hash: 1ff2a9eb6dab8cdf
---

# Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence

**Conference**: ICML2026  
**arXiv**: [2605.29744](https://arxiv.org/abs/2605.29744)  
**Code**: No public code found  
**Area**: Medical Imaging / Medical Multi-Agent Systems / Clinical Decision Making  
**Keywords**: Medical Multi-Agent Systems, Specialist Models, Clinical Decision Support, Uncertainty Routing, Evidence Fusion  

## TL;DR
HetMedAgent organizes generalist LLMs, modality-specific specialist models, and clinicians into a heterogeneous multi-agent system. Through conflict-aware evidence fusion and uncertainty routing, it demonstrates that specialist models and human supervision remain irreplaceable components of medical AI in cardiovascular and chest X-ray clinical decision tasks.

## Background & Motivation
**Background**: As generalist LLMs like GPT and Claude show increasing proficiency in medical Q&A and clinical reasoning, a question naturally arises: Since large models can read medical records, answer exams, and explain diagnoses, is there still a need to train specialist models for specific modalities like ECG, ECHO, and CXR?

**Limitations of Prior Work**: The risks of relying on a single medical LLM are evident. While they can provide fluent reasoning chains, their understanding of low-level signals in specific modalities is not always reliable. Furthermore, medical data is highly private, scarce, and fragmented across institutions, making the cost and compliance difficulty of training an omnipotent medical foundation model extremely high. Conversely, pure specialist models, while strong in narrow tasks, lack cross-modal integration, clinical context reasoning, and clear responsibility boundaries.

**Key Challenge**: Medical decision-making is not a matter of "one model outputting one answer," but a collaboration between multiple types of evidence, various professional roles, and final accountability. Generalist LLMs excel at language organization and synthetic reasoning, specialist models excel at modality-specific diagnosis, and physicians are responsible for safety, ethics, and final adjudication. Pursuing only single-model automation sacrifices at least one of these critical capabilities.

**Goal**: This paper aims to prove that specialist models have not been eliminated by generalist LLMs. A more reasonable path is to treat them as pluggable experts that, along with generalist LLMs and clinician agents, form a clinical decision support system. The system should not only improve AUROC/F1 but also identify situations where it should defer to a physician rather than provide an automated answer.

**Key Insight**: The authors design medical AI as a heterogeneous multi-agent collaborative workflow. Each patient case contains clinical information and multi-modal examinations. An orchestrator selects specialist models, specialists output structured findings with confidence scores, a reasoning agent performs evidence fusion, and uncertainty routing determines whether to trigger clinician intervention.

**Core Idea**: Do not let generalist LLMs replace specialist models; instead, let LLMs handle coordination and reasoning, let specialist models provide modality evidence, and let physicians handle cases with high uncertainty.

## Method
The core of HetMedAgent is not a single network, but a sequence of medical decision-making processes. It formalizes MDT-style (Multi-Disciplinary Team) collaboration into an executable agent pipeline, explicitly modeling evidence conflict, generation confidence, reasoning consistency, and clinician intervention thresholds.

### Overall Architecture
The system input is a patient case $C=\{V,I\}$, where $V$ includes clinical information such as age, gender, chronic history, treatment history, and symptoms; $I$ represents examination modalities, such as ECHO reports, ECG images, or CXR images. The goal is to output a set of clinical decisions, such as 180-day cardiovascular admission risk, etiology prediction, severity assessment, or acute/non-acute determination for chest X-rays.

In the workflow, a memory module first stores patient information, interaction history, available modalities, and task definitions. An orchestrator agent reads this context, identifies the task, and activates the appropriate specialist agents. Each specialist converts its modality into standardized diagnostic text and confidence. The system then calculates semantic conflict scores between different specialists and assigns weights to each finding. The reasoning agent receives the clinical background and weighted evidence to generate a preliminary decision and reasoning chain. Finally, the system calculates integrated uncertainty; if it exceeds a threshold, the case is escalated to a clinician agent for review; otherwise, it is output as clinical decision support advice.

### Key Designs
1.  **Heterogeneous Specialist Model Interface**:
    - **Function**: Allows medical models of different modalities to access the system in a unified format while retaining their specialized capabilities.
    - **Mechanism**: Each specialist agent outputs $F_i^w=\{diagnosis:F_i, confidence:c_i\}$. The ECHO specialist uses a text-to-text Transformer/LSTM structure to process report sequences; the ECG specialist uses a CNN image encoder plus a Transformer encoder-decoder to convert ECG images into diagnostic text; dual-view chest X-ray specialists were also added in expansion experiments.
    - **Design Motivation**: Medical modalities vary significantly; forcing all inputs into a single LLM loses modality-specific details. A unified text finding interface allows the downstream reasoning agent to consume evidence interpretably while allowing new specialist models to be added incrementally via standard interfaces.

2.  **Conflict-Aware Weighted Evidence Fusion**:
    - **Function**: When multiple specialists provide complementary or contradictory evidence, the system adjusts evidence weights based on confidence and conflict levels rather than simple concatenation.
    - **Mechanism**: Each finding is projected into semantic space using a PubMedBERT bi-encoder. Its average similarity to other specialists is calculated to derive a conflict score $\delta_i$. Weights are calculated as $w_i=\mathrm{softmax}(\log c_i+\log(1-\delta_i))$ and provided as prompt-level annotations to tell the reasoning agent which evidence is more reliable.
    - **Design Motivation**: Natural language findings from different modalities do not share a probability label space, making direct product-of-experts impossible. Providing weights as structured text annotations preserves the LLM's synthetic reasoning capability while making evidence reliability explicit.

3.  **Uncertainty Routing and Adaptive Clinician Intervention**:
    - **Function**: Positions the system as a clinical decision support tool rather than an unsupervised autonomous diagnostic device.
    - **Mechanism**: Integrated uncertainty consists of three parts: specialist confidence gap $U_{conf}=1-\max_i(c_i)$, average conflict $U_{conflict}=\frac{1}{k}\sum_i\delta_i$, and reasoning chain incoherence $U_{coherence}$. If $U(D_{prelim})>\theta_P$, the case is escalated to the clinician agent. The threshold is updated based on physician feedback: a physician's acceptance slightly raises the threshold, while a modification lowers it.
    - **Design Motivation**: The core of medical scenarios is not to have AI answer as much as possible automatically, but to automate low-risk, low-conflict cases while precisely delivering difficult/contradictory cases to physicians, balancing efficiency and safety.

### Loss & Training
The paper does not train the entire multi-agent system end-to-end but instead trains/configures specialist models individually and uses generalist LLMs as the orchestrator and reasoning backend. ECHO/ECG specialists output diagnostic text, with quality evaluated by BERTScore; clinical decision results are evaluated using AUROC and F1. GPT-4o is the default generalist LLM, with substitution experiments using Claude, Gemini, Llama, Qwen, and GLM. The clinician intervention threshold experiments use a fixed threshold $\theta_P=0.5$ and simulated sequential feedback to verify calibration.

## Key Experimental Results

### Main Results
The main experiment involves 613 real cardiovascular cases from 514 patients. Inputs include ECHO reports and ECG images; tasks include admission risk stratification, etiology prediction, and severity assessment. HetMedAgent is compared against medical LLMs and standard multi-agent systems.

| Method | Risk AUROC/F1 | Etiology AUROC/F1 | Severity AUROC/F1 | Average Observation |
| :--- | :--- | :--- | :--- | :--- |
| Meditron | 0.801 / 0.768 | 0.723 / 0.681 | 0.673 / 0.634 | One of the strongest single-model baselines |
| MedAgents | 0.823 / 0.789 | 0.751 / 0.708 | 0.692 / 0.653 | Strongest multi-agent baseline |
| AgentClinic | 0.817 / 0.781 | 0.738 / 0.695 | 0.681 / 0.641 | Multi-GPT-4 physician role collaboration |
| HetMedAgent w/o Clinician | 0.866 / 0.844 | 0.801 / 0.757 | 0.727 / 0.719 | Best across all three tasks |

The authors report that compared to the best single-model baseline, HetMedAgent achieves an average Gain of +6.6% AUROC and +7.9% F1. Compared to the best multi-agent baseline, the average Gain is +4.3% AUROC and +5.7% F1. This indicates that the benefits come not just from "using multiple LLMs for discussion," but from specialist models and conflict/uncertainty mechanisms.

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| GPT-4o alone | Avg AUROC 0.671, F1 0.625 | Clinical decision is significantly insufficient with generalist LLM only |
| + ECHO specialist | Avg AUROC 0.752, F1 0.711 | ECHO info brings +8.1% AUROC, +8.6% F1 |
| + ECG specialist | Avg AUROC 0.734, F1 0.692 | ECG info also significantly improves performance |
| + Two specialists | Avg AUROC 0.798, F1 0.773 | Dual-modality complementarity is best |
| Weighted evidence | Avg AUROC 0.798, F1 0.773 | Correct weight annotation is optimal |
| No annotation | Avg AUROC 0.777, F1 0.749 | Performance drops without weights |
| Inverse-weighted | Avg AUROC 0.758, F1 0.727 | Inverse weights hurt most, proving LLM utilizes weights |

### Key Findings
- Transformer-based specialists are significantly stronger than traditional CNN specialists. ECHO BERTScore improved from 0.707 (ResNet-based) to 0.800, ECG from 0.658 to 0.717, and average conflict scores decreased.
- Different generalist LLMs can be integrated, but GPT-4o performs best (Avg AUROC/F1 0.798/0.773); Claude-3.5-Sonnet (0.791/0.766) and Gemini-2.0-Flash (0.783/0.757) follow, showing the framework is not strictly tied to one model.
- With a fixed threshold $\theta_P=0.5$, 114 out of 613 cases (18.6%) triggered clinician intervention; these cases had lower F1 scores, indicating the uncertainty mechanism successfully filtered more difficult cases.
- Adaptive thresholds reduced intervention counts from 114 to 97 (15.8%) while increasing AIR from 1.468 to 1.679, showing feedback calibration accurately distinguishes cases needing intervention.
- In cross-domain chest X-ray experiments, HetMedAgent achieved AUROC 0.820 and F1 0.537 on IU X-Ray acute/non-acute tasks, outperforming ViT-BERT (0.783/0.468), validating the framework's transferability to different medical specialties.

## Highlights & Insights
- The paper explicitly rejects the "universal medical LLM" narrative in favor of collaborative system design. This is pragmatic: bottlenecks in medical AI are often not language proficiency but modality specialty, responsibility boundaries, and uncertainty management.
- Using specialist weights as prompt-level evidence annotations is a clever compromise. It avoids the difficulty of cross-modal probability calibration while allowing the LLM to explicitly know which evidence is more trustworthy during reasoning.
- Uncertainty routing brings the system closer to real clinical workflows. Automation is not the goal itself; knowing exactly when to escalate to a doctor is a key capability for safe deployment.
- The cross-domain CXR experiment, while supplementary, is important. It proves HetMedAgent is a modular paradigm where specialists can be swapped, rather than a pipeline handcrafted only for ECHO+ECG.

## Limitations & Future Work
- Generation confidence $c_i$ is essentially token-level confidence, which is not equivalent to clinical correctness. Future work requires stronger per-modality calibration, such as Platt scaling or expert-labeled calibration.
- Clinician feedback in experiments was primarily simulated using ground truth, not real multi-physician consensus. In real deployment, physician opinions may be noisy or divergent, requiring more robust momentum and outlier handling for threshold updates.
- The primary dataset originates from a single institution's cardiovascular cases; with 613 cases, the scale is relatively small. Samples for subgroups (e.g., Age ≥85) are even smaller, limiting conclusions on fairness and generalization.
- Agents primarily exchange text, which may lose spatial structures and continuous representations found in ECG/CXR. The authors suggest future versions where specialists output both structured text and embeddings for joint reasoning.
- Commercial LLM APIs pose privacy and compliance issues. Although the authors discuss local open-weight alternatives, actual performance, cost, and safety audits still require verification.

## Related Work & Insights
- **vs. Medical LLMs**: Models like PMC-LLaMA, Meditron, and BioMistral inject medical knowledge into a single model; HetMedAgent argues medical decisions are better suited for generalist reasoning + specialist precision + clinician oversight.
- **vs. AgentClinic/MedAgents**: These systems primarily use LLM role-playing collaboration, lacking real modality-specific models and clinician routing; HetMedAgent’s heterogeneity is closer to clinical MDTs.
- **vs. Traditional Specialist Models**: Single-modality models like ResNet/EfficientNet can make local diagnoses but cannot integrate clinical context; HetMedAgent preserves their modality capabilities while delegating cross-evidence reasoning to the LLM.
- **vs. Human-AI Collaborative CDSS**: Traditional CDSS often provide rules or risk scores; HetMedAgent provides traceable agent chains, weight annotations, and uncertainty escalation, which are better for procedural auditing.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The idea of medical multi-agent frameworks is not brand new, but the systematic integration of specialist models, LLMs, clinician intervention, and conflict weights is quite complete.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, modality ablation, weight sensitivity, cross-LLM, and CXR transfer are comprehensive; however, real multi-institutional data and real-world clinician feedback are still lacking.
- Writing Quality: ⭐⭐⭐⭐☆ Framework diagrams and workflows are clear, and results are well-explained; the use of many symbols and formulas may be slightly dense for clinical readers.
- Value: ⭐⭐⭐⭐☆ Highly valuable for medical AI system design, especially in emphasizing the long-term necessity of specialist models and clinician oversight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/medical_imaging/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](../../AAAI2026/medical_imaging/mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](../../NeurIPS2025/medical_imaging/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
