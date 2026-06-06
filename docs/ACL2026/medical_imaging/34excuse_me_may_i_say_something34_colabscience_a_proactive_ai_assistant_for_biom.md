---
title: >-
  [Paper Note] "Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery
description: >-
  [ACL 2026][Medical Imaging][Proactive Intervention] CoLabScience utilizes the PULI (Positive-Unlabeled Learning Intervention) framework to train an LLM assistant capable of **proactively determining when and how to inter…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Proactive Intervention"
  - "Scientific Collaboration"
  - "Positive-Unlabeled Learning"
  - "Reinforcement Learning"
  - "Biomedical Dialogue"
date: 2026-05-08
content_hash: 535b15179d56b3bb
---

# "Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery

**Conference**: ACL 2026  
**arXiv**: [2604.15588](https://arxiv.org/abs/2604.15588)  
**Code**: [https://github.com/YANGWU001/CoLabScience](https://github.com/YANGWU001/CoLabScience)  
**Area**: Medical Imaging  
**Keywords**: Proactive Intervention, Scientific Collaboration, Positive-Unlabeled Learning, Reinforcement Learning, Biomedical Dialogue

## TL;DR
CoLabScience utilizes the PULI (Positive-Unlabeled Learning Intervention) framework to train an LLM assistant capable of **proactively determining when and how to intervene** in biomedical team discussions. It leverages GRPO and a reinforcement learning coordinator to automatically identify optimal intervention timings from streaming dialogues and generate scientific suggestions.

## Background & Motivation

**Background**: LLMs have been widely applied in biomedical research, including drug repositioning, disease diagnosis, and clinical Q&A. However, existing models primarily operate in a "reactive" mode—only responding after explicit user prompts.

**Limitations of Prior Work**: In multi-person scientific collaboration scenarios, team discussions are often streaming and involve multiple alternating roles. Reactive LLMs cannot intervene timely when discussions deviate from goals or miss critical knowledge, leading to missed opportunities for important scientific insights.

**Key Challenge**: Scientific collaboration requires "proactive participation," but existing methods either rely on hand-crafted prompt rules or lack a learnable mechanism for intervention timing, failing to achieve adaptive, context-aware intervention.

**Goal**: To design a proactive LLM assistant capable of: (1) determining **when to intervene** in streaming scientific discussions; (2) generating **high-quality intervention content**.

**Key Insight**: Modeling the problem as a "Positive-Unlabeled" (PU) learning task—only a small number of optimal intervention points in the dialogue are labeled as positive samples, while the rest are unlabeled. An RL coordinator is used to discover latent positive and negative samples.

**Core Idea**: Use a lightweight Observer to judge intervention timing and a large model Presenter to generate intervention content, with both being jointly optimized end-to-end via an RL coordinator.

## Method

### Overall Architecture
The system consists of three core components: (1) **Coordinator** (MLP): Receives the current dialogue state and outputs a binary decision (intervene or stay silent); (2) **Observer** (Small LLM): Trained via GRPO to learn intervention timing classification; (3) **Presenter** (Large LLM): Trained via SFT to generate intervention content. Inputs include the project proposal $C$ (research goals, background knowledge, datasets) and dual-scale memory (Short-term memory $\mathcal{M}^S$ retains the last 3 dialogue rounds; Long-term memory $\mathcal{M}^L$ recursively compresses history via an LLM summarizer).

### Key Designs

1. **PULI (Positive-Unlabeled Learning Intervention) Framework**:
    - **Function**: Learns when to intervene from dialogue data with only sparse positive labels.
    - **Mechanism**: Annotates one round per dialogue that deviates most from the research goal as a positive sample, treating the rest as unlabeled. The Coordinator predicts "intervene/silent" for each unlabeled round; selected silent samples serve as negative examples for training the Observer, while selected intervention samples augment the positive set for training the Presenter. Performance changes in the two models (Observer accuracy change $r^{\text{when}}$, Presenter ROUGE-1 change $r^{\text{how}}$) are used as rewards for the Coordinator, updated via REINFORCE.
    - **Design Motivation**: It is difficult to annotate the necessity of intervention for every round in real collaboration. PU learning allows using only a small number of high-confidence positive samples, reducing labeling noise and hallucination risks.

2. **Dual-scale Dialogue Memory**:
    - **Function**: Provides sufficient contextual information for intervention decisions.
    - **Mechanism**: Short-term memory $\mathcal{M}^S$ contains the current and previous two rounds; long-term memory $\mathcal{M}^L$ recursively compresses all historical dialogues via an LLM summarizer $\Gamma(\cdot)$. The state vector $S_n$ is formed by concatenating the final hidden layer representations of the Observer and Presenter.
    - **Design Motivation**: Short-term memory captures immediate contextual shifts, while long-term memory prevents the loss of critical historical information and avoids infinite memory growth.

3. **BSDD (Biomedical Streaming Dialogue Dataset)**:
    - **Function**: Provides a standard benchmark for training and evaluating proactive intervention.
    - **Mechanism**: Uses a Prophet LLM to extract project proposals from PubMed papers. A Dialogue-Simulator LLM simulates multi-round discussions among four roles (pharmacologist, medicinal chemist, bioinformatician, clinician). The Prophet LLM labels the round most deviated from the goal as the positive intervention point.
    - **Design Motivation**: Existing biomedical dialogue datasets (e.g., MedDialog) focus on doctor-patient interactions and lack scientific team discussions or intervention annotations.

### Loss & Training
Total reward $r_{\text{total}} = \lambda \cdot r^{\text{when}} + (1-\lambda) \cdot r^{\text{how}}$, with $\lambda=0.6$ balancing timing and quality. The Coordinator is updated using the REINFORCE policy gradient. The Observer is trained with GRPO. The Presenter is fine-tuned using LoRA (rank=16, $\alpha=64$).

## Key Experimental Results

### Main Results

| Model Configuration | Metric | PULI | ICL | Proactive Agent |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-0.6B + Qwen3-14B | Accuracy | **64.1%** | 55.7% | 53.9% |
| Qwen3-0.6B + Qwen3-14B | F1 | **46.4%** | 28.9% | 24.5% |
| Qwen3-0.6B + Qwen3-14B | ROUGE-1 | **32.4%** | 29.4% | 27.6% |
| LLaMA3.2-1B + LLaMA3.1-8B | Accuracy | **67.4%** | 58.4% | 54.5% |
| LLaMA3.2-1B + LLaMA3.1-8B | F1 | **65.4%** | 56.7% | 60.2% |
| LLaMA3.2-1B + LLaMA3.1-8B | WR-Intra | **39.2%** | 20.8% | 7.5% |

### Ablation Study

| Configuration | Accuracy | F1 | WR-Intra | Note |
| :--- | :--- | :--- | :--- | :--- |
| PULI | **67.4%** | **65.4%** | **57.5%** | Full model |
| w DPO | 64.6% | 63.1% | 31.7% | GRPO replaced by DPO |
| w SFT | 61.9% | 58.6% | 6.7% | Pure SFT Observer |
| w PN | 57.3% | 54.5% | 4.1% | All unlabeled as negative |

### Key Findings
- PULI shows that in cross-family model comparisons, the LLaMA3 pair reaches 45.8% WR, significantly outperforming GPT-4o's ICL (18.3%), indicating that small open-source models + PULI can surpass GPT-4o.
- $\lambda=0.6$ is the optimal balance point; lower values cause Observer accuracy to crash, while higher values degrade Presenter quality.
- Human evaluation shows PULI outperforms the GPT baseline across Timing (4.65 vs 4.36), Quality (4.35 vs 4.18), and Helpfulness (4.60 vs 4.43).

## Highlights & Insights
- **Selecting PU learning for intervention detection** is a clever modeling choice: sparse labeling avoids hallucination noise introduced by forcing LLMs to make fine-grained judgments for every round, while the RL coordinator automatically identifies hidden positive/negative samples in unlabeled data.
- **The Observer-Presenter decoupled architecture** balances efficiency and quality: the lightweight Observer monitors in real-time and invokes the expensive Presenter only when necessary, which is suitable for real-time collaboration.
- This "when to act + how to act" dual-objective RL framework is transferable to other tasks requiring timing judgment, such as automated prompting in educational tutoring or key-point reminders for meeting assistants.

## Limitations & Future Work
- Data is based on LLM-simulated dialogues; complexities in real scientific meetings like overlapping speech, informal reasoning, and dynamic goal evolution are not fully captured.
- Only one positive intervention point is labeled per dialogue, potentially missing multiple valuable intervention opportunities.
- Actual deployment requires ASR/TTS integration; the prototype introduces approximately 0.8 seconds of additional latency per round.
- Focus is currently limited to "goal deviation" interventions, lacking coverage for diverse types like clarifying misunderstandings or facilitating collaboration.

## Related Work & Insights
- **vs Proactive Agent (Lu et al., 2024b)**: The latter uses manual system prompts to control proactive behavior; PULI learns adaptive intervention strategies via RL, achieving 12.9% higher accuracy.
- **vs VideoLLM-Online**: The latter learns narrative timing in multimodal streams; PULI targets intervention timing in textual scientific dialogues, focusing more on scientific collaboration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of PU learning + RL Coordinator for proactive intervention is a novel approach, though the dataset construction follows conventional patterns.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of multiple model families, baselines, human evaluation, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and understandable framework diagrams, though the notation system is somewhat dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](responsible_evaluation_of_ai_for_mental_health.md)
- [\[NeurIPS 2025\] Orochi: Versatile Biomedical Image Processor](../../NeurIPS2025/medical_imaging/orochi_versatile_biomedical_image_processor.md)
- [\[ICML 2026\] MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](../../ICML2026/medical_imaging/medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)

</div>

<!-- RELATED:END -->
