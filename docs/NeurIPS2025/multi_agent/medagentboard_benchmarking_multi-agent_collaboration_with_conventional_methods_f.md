---
title: >-
  [Paper Note] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][Multi-Agent][multi-agent collaboration] This paper proposes MedAgentBoard, a comprehensive benchmark that systematically evaluates multi-agent collaboration, single-LLM…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "Multi-Agent"
  - "multi-agent collaboration"
  - "medical benchmarking"
  - "LLM"
  - "clinical workflow"
  - "EHR prediction"
date: 2026-05-08
content_hash: d0a1a043c93147ad
---

# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks

**Conference**: NeurIPS 2025 (Datasets & Benchmarks Track)

**arXiv**: [2505.12371](https://arxiv.org/abs/2505.12371)

**Code**: [GitHub](https://github.com/yhzhu99/MedAgentBoard) | [Project Page](https://medagentboard.netlify.app/)

**Area**: Medical Imaging / AI for Medicine

**Keywords**: multi-agent collaboration, medical benchmarking, LLM, clinical workflow, EHR prediction

## TL;DR

This paper proposes MedAgentBoard, a comprehensive benchmark that systematically evaluates multi-agent collaboration, single-LLM, and conventional methods across diverse medical tasks, revealing that multi-agent collaboration does not consistently outperform strong single models or specialized conventional approaches.

## Background & Motivation

- **The multi-agent LLM trend**: A growing body of work has introduced multi-agent collaboration into the medical domain, yet its practical advantages remain unclear.
- **Limitations of existing evaluations**:
  1. Task coverage is insufficiently broad, lacking diversity representative of real clinical scenarios.
  2. Rigorous comparisons with specialized conventional methods are absent (most works only compare across LLMs).
  3. Data modalities are limited, overlooking structured EHR data and medical imaging.
- **Core Problem**: Does the added complexity and overhead of multi-agent systems genuinely yield performance gains?
- **Research positioning**: To provide a comprehensive, evidence-based evaluation that assists researchers in selecting appropriate AI solutions.

## Method

### Overall Architecture

MedAgentBoard covers **4 major categories of medical tasks** spanning **3 data modalities** (text, medical imaging, and structured EHR), systematically comparing **3 classes of methods**:

| Task Category | Data Modality | Datasets |
|---------|---------|--------|
| Medical QA | Text | MedQA, PubMedQA |
| Medical VQA | Image + Text | PathVQA, VQA-RAD |
| Lay Summary Generation | Text | PLOS/eLife |
| EHR Predictive Modeling | Structured Data | MIMIC-III/IV |
| Clinical Workflow Automation | Multimodal | Custom scenarios |

### Key Designs

#### Three-Way Comparison Framework

1. **Conventional Methods**:

    - Text QA: BioLinkBERT, GatorTron
    - VQA: Specialized VLMs such as M³AE
    - EHR: XGBoost, LSTM, Transformer, etc.

2. **Single-LLM Methods**:

    - Zero-shot / Few-shot ICL / Chain-of-Thought
    - Models include GPT-4o, Claude 3.5, Gemini, etc.

3. **Multi-Agent Collaboration Frameworks**:

    - MedAgents: multi-role discussion and collaboration
    - ReConcile: multi-model voting and reconciliation
    - General-purpose frameworks such as AutoGen

#### Evaluation Dimensions

- **Correctness**: Accuracy (multiple-choice), BLEU/ROUGE (generation tasks)
- **Clinical Relevance**: LLM-as-a-judge scoring
- **Efficiency**: API call count, token consumption, latency
- **Robustness**: Consistency across datasets

### Loss & Training

As a benchmark paper, the focus lies on evaluation protocol design rather than model training:
- All LLM-based methods use unified prompt templates.
- Conventional methods follow optimal configurations reported in their original papers.
- Evaluation metrics are standardized across tasks.
- Results are averaged over multiple runs to reduce variance.

## Key Experimental Results

### Main Results

#### Medical Text QA Results

| Method Category | Method | MedQA Acc↑ | PubMedQA Acc↑ | Category |
|---------|---------|-----------|-------------|------|
| Conventional | BioLinkBERT | 45.2 | 72.8 | Conventional |
| Conventional | GatorTron | 48.1 | 74.5 | Conventional |
| Single LLM | GPT-4o (Zero-shot) | 82.3 | 78.1 | Single LLM |
| Single LLM | GPT-4o (CoT) | **85.7** | **80.4** | Single LLM |
| Single LLM | Claude 3.5 (CoT) | 83.9 | 79.2 | Single LLM |
| Multi-Agent | MedAgents | 83.1 | 78.8 | Multi-Agent |
| Multi-Agent | ReConcile | 84.2 | 79.5 | Multi-Agent |

**Finding**: On medical text QA, an advanced single LLM (GPT-4o + CoT) suffices to achieve optimal performance; multi-agent systems yield no significant improvement.

#### Medical VQA and EHR Prediction Results

| Method Category | PathVQA Acc↑ | VQA-RAD Acc↑ | MIMIC Mortality AUROC↑ |
|---------|-------------|-------------|-------------------|
| Conventional VLM (M³AE) | **72.3** | **74.8** | — |
| GPT-4o Vision | 65.7 | 68.2 | 0.71 |
| Multi-Agent VQA | 64.9 | 67.5 | 0.69 |
| XGBoost | — | — | **0.84** |
| LSTM | — | — | 0.81 |
| LLM (numerical reasoning) | — | — | 0.68 |

**Finding**: Specialized conventional methods still significantly outperform LLM-based approaches on VQA and EHR prediction.

### Ablation Study

#### Multi-Agent vs. Single-LLM Efficiency Comparison

| Method | Accuracy | API Calls | Token Consumption | Latency (s) |
|------|----------|-------------|-----------|---------|
| GPT-4o (Single) | 85.7 | 1 | 2.1K | 3.2 |
| MedAgents (3 roles) | 83.1 | 5–8 | 12.5K | 18.7 |
| ReConcile (3 models) | 84.2 | 3 | 7.8K | 11.4 |

**Finding**: Multi-agent systems consume 4–6× more tokens than single-LLM methods, while yielding limited or even negative performance gains.

### Key Findings

1. **Multi-agent ≠ better**: Across 4 task categories, multi-agent systems demonstrate advantages only in **task completeness** within clinical workflow automation.
2. **Conventional methods remain competitive**: Specialized fine-tuned models significantly outperform all LLM-based methods on VQA and EHR prediction.
3. **Single-LLM CoT is sufficiently powerful**: A high-quality single model with a well-designed prompt outperforms collaboration among multiple mediocre models.
4. **Asymmetric cost–benefit trade-off**: Multi-agent systems incur 4–6× greater computational overhead, yet achieve on average less than 1% performance improvement.
5. **Task specificity**: No universally optimal method exists; method selection must be guided by the specific task at hand.

## Highlights & Insights

- **A grounded benchmark**: This work offers a sober evaluation amid the multi-agent enthusiasm, demonstrating that "multi-agent is not a silver bullet."
- **Fair comparison design**: Incorporating conventional methods into the comparison is a core contribution of this benchmark, filling a critical gap in existing evaluations.
- **Full multimodal coverage**: Simultaneously covering text, imaging, and structured data reflects the diversity of real clinical settings.
- **Actionable guidance**: The benchmark provides practitioners with principled guidelines on when to use multi-agent systems, single models, or conventional methods.

## Limitations & Future Work

1. **Rapid LLM evolution**: Benchmark results for specific LLMs may become outdated quickly as newer models (e.g., GPT-5) emerge.
2. **Limited multi-agent framework coverage**: Only a small number of frameworks are evaluated; novel collaboration paradigms (e.g., debate, reflection) could be incorporated.
3. **Incomplete task coverage**: Important clinical tasks such as medical image segmentation and radiology report generation are not included.
4. **Fairness of comparison**: Conventional methods are carefully fine-tuned, whereas LLMs are largely evaluated in zero/few-shot settings, introducing an inherent asymmetry.
5. **Real-world deployment evaluation**: Assessment of practical clinical deployment scenarios—such as latency requirements and privacy constraints—is absent.

## Related Work & Insights

- **MedAgents** (Tang et al., 2024): A multi-agent discussion framework for medical reasoning.
- **AgentBench** (Liu et al., 2024): A general-purpose benchmark for LLM agents.
- **HELM-Med**: A medical LLM evaluation suite.
- **Insights**: The three-way comparison paradigm proposed in this benchmark (conventional vs. single LLM vs. multi-agent) is generalizable to other domains such as law and finance.

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 3.5 | Contribution lies in evaluation perspective rather than technical methodology |
| Technical Depth | 3 | Benchmark work; moderate technical complexity |
| Experimental Thoroughness | 4.5 | Broad coverage and comprehensive comparisons |
| Practical Value | 4.5 | Directly actionable for medical AI method selection |
| Writing Quality | 4 | Clear structure; findings are precisely articulated |
| **Overall** | **4.0** | An important benchmark contribution with sober and insightful findings |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/multi_agent/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)
- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)

</div>

<!-- RELATED:END -->
