---
title: >-
  [Paper Note] "Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery
description: >-
  [ACL 2026][Medical Imaging][Proactive Intervention] CoLabScience introduces the PULI (Positive-Unlabeled Learning Intervention) framework to train an LLM assistant capable of **proactively determining when and how to int…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Proactive Intervention"
  - "Scientific Collaboration"
  - "Positive-Unlabeled Learning"
  - "Reinforcement Learning"
  - "Biomedical Dialogue"
date: 2026-05-08
content_hash: 0b50e79314943853
---

# "Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery

**Conference**: ACL 2026
**arXiv**: [2604.15588](https://arxiv.org/abs/2604.15588)
**Code**: [https://github.com/YANGWU001/CoLabScience](https://github.com/YANGWU001/CoLabScience)
**Area**: Medical Imaging
**Keywords**: Proactive Intervention, Scientific Collaboration, Positive-Unlabeled Learning, Reinforcement Learning, Biomedical Dialogue

## TL;DR
CoLabScience introduces the PULI (Positive-Unlabeled Learning Intervention) framework to train an LLM assistant capable of **proactively determining when and how to intervene** in biomedical team discussions. By leveraging GRPO and a reinforcement learning coordinator, the system automatically identifies optimal intervention moments and generates scientific suggestions from streaming conversations.

## Background & Motivation

**Background**: LLMs have been widely applied in biomedical research, including drug repurposing, disease diagnosis, and clinical question answering. However, existing models primarily operate in a reactive mode—responding only when explicitly queried by users.

**Limitations of Prior Work**: In multi-participant scientific collaboration scenarios, team discussions are typically streaming and involve multiple alternating roles. Reactive LLMs cannot intervene in a timely manner when discussions deviate from research goals or overlook critical knowledge, leading to missed opportunities for important scientific insights.

**Key Challenge**: Scientific collaboration demands proactive participation, yet existing approaches either rely on manually designed prompt rules or lack learnable mechanisms for determining intervention timing, making context-aware, adaptive intervention infeasible.

**Goal**: Design a proactive LLM assistant that can: (1) determine **when to intervene** in streaming scientific discussions; and (2) generate **high-quality intervention content**.

**Key Insight**: The problem is formulated as a Positive-Unlabeled (PU) learning problem—only a small number of optimal intervention points in a dialogue are labeled as positive samples, while the rest remain unlabeled. A reinforcement learning coordinator is employed to discover implicit positive and negative samples from the unlabeled data.

**Core Idea**: A lightweight Observer determines intervention timing, while a larger Presenter generates intervention content. A reinforcement learning coordinator jointly optimizes both components end-to-end.

## Method

### Overall Architecture
The system consists of three core components: (1) **Coordinator** (MLP): receives the current dialogue state and outputs a binary decision (intervene or remain silent); (2) **Observer** (small LLM): trained via GRPO to classify intervention timing; (3) **Presenter** (large LLM): trained via SFT to generate intervention content. Inputs include a project proposal $C$ (research objectives, background knowledge, dataset) and dual-scale memory—short-term memory $\mathcal{M}^S$ retaining the most recent 3 dialogue turns, and long-term memory $\mathcal{M}^L$ recursively compressing history via an LLM summarizer.

### Key Designs

1. **PULI: Positive-Unlabeled Learning Intervention Framework**

    - **Function**: Learns when to intervene from dialogue data with only sparse positive labels.
    - **Mechanism**: Each dialogue annotates the turn most deviated from the research goal as a positive sample; all other turns remain unlabeled. The Coordinator predicts "intervene/silent" for each unlabeled turn—selected silent samples serve as negative examples to train the Observer, while selected intervention samples augment the positive set to train the Presenter. Changes in model performance (Observer accuracy change $r^{\text{when}}$ and Presenter ROUGE-1 change $r^{\text{how}}$) serve as reward signals fed back to the Coordinator, which is updated via REINFORCE.
    - **Design Motivation**: Annotating intervention necessity for every turn in real-world collaboration is impractical. PU learning allows labeling only high-confidence positive samples, reducing annotation noise and hallucination risk.

2. **Dual-Scale Dialogue Memory**

    - **Function**: Provides sufficient contextual information for intervention decisions.
    - **Mechanism**: Short-term memory $\mathcal{M}^S$ contains the current and preceding two turns; long-term memory $\mathcal{M}^L$ recursively compresses all historical dialogue via an LLM summarizer $\Gamma(\cdot)$. The state vector $S_n$ is formed by concatenating the final hidden representations of the Observer and Presenter.
    - **Design Motivation**: Short-term memory captures immediate contextual changes, while long-term memory prevents loss of critical historical information without causing unbounded memory growth.

3. **BSDD: Biomedical Streaming Dialogue Dataset**

    - **Function**: Provides a standard benchmark for training and evaluating proactive intervention.
    - **Mechanism**: A Prophet LLM extracts project proposals from PubMed papers; a Dialogue-Simulator LLM simulates four roles (pharmacologist, medicinal chemist, bioinformatician, clinician) in multi-turn discussions; the Prophet LLM annotates the most goal-deviated turn as the positive intervention point.
    - **Design Motivation**: Existing biomedical dialogue datasets (e.g., MedDialog) focus on patient–physician interactions and lack scientific team discussions with intervention annotations.

### Loss & Training
The total reward is $r_{\text{total}} = \lambda \cdot r^{\text{when}} + (1-\lambda) \cdot r^{\text{how}}$, with $\lambda=0.6$ balancing intervention timing and content quality. The Coordinator is updated via REINFORCE policy gradients. The Observer is trained with GRPO. The Presenter is fine-tuned with LoRA (rank=16, $\alpha=64$) via SFT.

## Key Experimental Results

### Main Results

| Model Configuration | Metric | PULI | ICL | Proactive Agent |
|---|---|---|---|---|
| Qwen3-0.6B + Qwen3-14B | Accuracy | **64.1%** | 55.7% | 53.9% |
| Qwen3-0.6B + Qwen3-14B | F1 | **46.4%** | 28.9% | 24.5% |
| Qwen3-0.6B + Qwen3-14B | ROUGE-1 | **32.4%** | 29.4% | 27.6% |
| LLaMA3.2-1B + LLaMA3.1-8B | Accuracy | **67.4%** | 58.4% | 54.5% |
| LLaMA3.2-1B + LLaMA3.1-8B | F1 | **65.4%** | 56.7% | 60.2% |
| LLaMA3.2-1B + LLaMA3.1-8B | WR-Intra | **39.2%** | 20.8% | 7.5% |

### Ablation Study

| Configuration | Accuracy | F1 | WR-Intra | Note |
|---|---|---|---|---|
| PULI | **67.4%** | **65.4%** | **57.5%** | Full model |
| w DPO | 64.6% | 63.1% | 31.7% | Replace GRPO with DPO |
| w SFT | 61.9% | 58.6% | 6.7% | Observer trained with SFT only |
| w PN | 57.3% | 54.5% | 4.1% | All unlabeled samples treated as negative |

### Key Findings
- Across cross-model-family comparisons, the LLaMA3 pair with PULI achieves a WR of 45.8%, substantially surpassing the GPT pair ICL baseline (18.3%), demonstrating that small open-source models combined with PULI can outperform GPT-4o.
- $\lambda=0.6$ is the optimal balance point; values that are too small cause Observer accuracy to collapse, while values that are too large degrade Presenter quality.
- In human evaluation, PULI outperforms the GPT pair baseline across all dimensions: Timing (4.65 vs. 4.36), Quality (4.35 vs. 4.18), and Helpfulness (4.60 vs. 4.43).

## Highlights & Insights
- **Applying PU learning to intervention detection** is an elegant modeling choice: sparse annotation avoids the hallucination noise introduced by requiring LLMs to make fine-grained judgments at every turn, while the RL coordinator automatically discovers implicit positive and negative samples from unlabeled data.
- **The Observer–Presenter decoupled architecture** achieves a balance between efficiency and quality: the lightweight Observer monitors in real time and invokes the expensive Presenter only when necessary, making the system suitable for real-time collaborative settings.
- The dual-objective RL framework of "when to act + how to act" is transferable to other tasks requiring timing judgment, such as automated prompting in educational tutoring and key-point reminders in meeting assistants.

## Limitations & Future Work
- The data is based on LLM-simulated dialogues; complexities present in real scientific meetings—such as overlapping speech, informal reasoning, and dynamically evolving goals—are not fully captured.
- Each dialogue is annotated with only a single positive intervention point, which may overlook multiple valuable intervention opportunities.
- Real-world deployment requires ASR/TTS integration; the current prototype introduces approximately 0.8 seconds of additional latency per turn.
- The system currently focuses exclusively on goal-deviation intervention and does not cover other intervention types such as clarifying misunderstandings or facilitating collaboration.

## Related Work & Insights
- **vs. Proactive Agent (Lu et al., 2024b)**: The latter controls proactive behavior via manually crafted system prompts, whereas PULI learns adaptive intervention strategies through RL, achieving 12.9% higher accuracy.
- **vs. VideoLLM-Online**: The latter learns narration timing in multimodal streams; PULI targets intervention timing in textual scientific dialogues, with a more focused objective on scientific collaboration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of PU learning and an RL coordinator for proactive intervention is a novel contribution, though the dataset construction approach is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across multiple model families, multiple baselines, human evaluation, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and framework diagrams are accessible, though the notation system is somewhat heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[NeurIPS 2025\] Orochi: Versatile Biomedical Image Processor](../../NeurIPS2025/medical_imaging/orochi_versatile_biomedical_image_processor.md)
- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](../../ICLR2026/medical_imaging/biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[AAAI 2026\] VitalDiagnosis: AI-Driven Ecosystem for 24/7 Vital Monitoring and Chronic Disease Management](../../AAAI2026/medical_imaging/vitaldiagnosis_ai-driven_ecosystem_for_247_vital_monitoring_and_chronic_disease_.md)

</div>

<!-- RELATED:END -->
