---
title: >-
  [Paper Note] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
description: >-
  [ICLR 2026 Oral][Medical Imaging][biomedical data science] This work introduces MedAgentGym, the first unified agentic training environment for biomedical data science, comprising 72,413 task instances spanning 12 real-world scenarios and 129 categories, equipped with an executable sandbox and verifiable ground truth. A systematic benchmark evaluation of 29 LLMs reveals a substantial gap between commercial and open-source models. By combining efficient multi-threaded trajectory sampling with offline/online RL, the authors train Med-Copilot, achieving gains of +43.02%/+45.28% respectively and attaining performance competitive with GPT-4o.
tags:
  - ICLR 2026 Oral
  - Medical Imaging
  - biomedical data science
  - agentic training
  - code-centric reasoning
  - reinforcement-learning
  - Med-Copilot
  - LLM agent
date: 2026-05-08
content_hash: 25c20c812e2a88a6
---

# MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science

**Conference**: ICLR 2026 Oral  
**arXiv**: [2506.04405](https://arxiv.org/abs/2506.04405)  
**Code**: Available  
**Area**: Medical AI / Agent Training  
**Keywords**: biomedical data science, agentic training, code-centric reasoning, reinforcement-learning, Med-Copilot, LLM agent

## TL;DR
This work introduces MedAgentGym, the first unified agentic training environment for biomedical data science, comprising 72,413 task instances spanning 12 real-world scenarios and 129 categories, equipped with an executable sandbox and verifiable ground truth. A systematic benchmark evaluation of 29 LLMs reveals a substantial gap between commercial and open-source models. By combining efficient multi-threaded trajectory sampling with offline/online RL, the authors train Med-Copilot, achieving gains of +43.02%/+45.28% respectively and attaining performance competitive with GPT-4o.

## Background & Motivation
**Background**: Biomedical data science encompasses genomic analysis, clinical data processing, medical image analysis, drug discovery, and other subfields, each demanding complex programming and domain-specific reasoning. While LLMs have demonstrated potential as coding assistants in general software engineering, systematic evaluation and training infrastructure for biomedical coding tasks remain lacking.

**Limitations of Prior Work**: (1) Existing medical AI benchmarks (e.g., MedQA, PubMedQA) are static multiple-choice or QA evaluations that do not support interactive code execution or iterative debugging. (2) No unified platform covers the diverse scenarios in biomedical data science — genomics, clinical informatics, imaging, and drug discovery each maintain separate, siloed benchmarks. (3) Open-source LLMs exhibit a significant performance gap relative to closed-source models (e.g., GPT-4o) on biomedical coding tasks, necessitating effective training methods to narrow this gap.

**Key Challenge**: Training an agent capable of writing biomedical analysis code requires a large-scale interactive task environment, yet constructing such an environment is prohibitively costly — demanding real data, ground truth annotations, secure sandboxes, and feedback mechanisms.

**Goal**: To simultaneously address environment construction and agent training by providing a large-scale training environment alongside an RL training pipeline.

**Key Insight**: The authors unify 12 real-world biomedical scenarios into a standardized format — input data + task description → executed code → verified output — supporting interactive feedback and automated scoring.

**Core Idea**: A large-scale, interactive, unified training environment combined with an RL training pipeline closes the gap between open-source models and closed-source LLMs on biomedical coding tasks.

## Method

### Overall Architecture
MedAgentGym comprises three core components: (1) **Task Repository**: 72,413 task instances, each containing data files, task descriptions, an executable sandbox, ground truth answers, and a scoring function. (2) **Interaction Engine**: Agents interact with the sandbox through multi-turn dialogue — submitting code, receiving execution results or error messages, and iteratively refining solutions. (3) **Training Pipeline**: Efficient multi-threaded trajectory generation supporting both offline and online RL training.

### Key Designs
1. **12-Scenario × 129-Category Task Taxonomy**:
    - **Function**: Covers 12 real-world scenarios including genomics (RNA-seq analysis, gene expression clustering), clinical data science (EHR prediction, survival analysis), medical imaging (pathology slide classification, X-ray detection), and drug discovery (molecular property prediction, ADMET analysis).
    - **Mechanism**: Each scenario defines a standardized interface — input (data file path + metadata) + task instruction (natural language description of the analysis objective) + ground truth (exact numeric answer or class label) + scoring function ($\text{score}(\hat{y}, y) \in [0, 1]$).
    - **Design Motivation**: Unifying multiple domains into a single platform enables agents to transfer and generalize across heterogeneous task types.

2. **Executable Sandbox + Interactive Feedback**:
    - **Function**: Provides each task with an isolated Python execution environment (pre-installed with pandas, scikit-learn, biopython, etc.); agents receive stdout/stderr feedback upon code submission.
    - **Mechanism**: Agents interact for at most $K$ rounds. At each round, the agent generates code $c_t$ → the sandbox executes it and returns $(o_t, e_t)$ → the agent decides whether to revise or submit a final answer based on the output/error. The full trajectory is $\tau = [(c_1, o_1, e_1), \ldots, (c_K, o_K, e_K)]$.
    - **Design Motivation**: Single-pass code generation yields low accuracy on many tasks that require debugging; interactive feedback allows agents to learn from errors.

3. **Multi-Threaded Trajectory Generation + RL Training**:
    - **Function**: Samples interaction trajectories across multiple tasks in parallel, supporting both offline RL (learning from pre-collected trajectories) and online RL (learning through real-time environment interaction).
    - **Mechanism**:
        - **Offline RL**: Collects a large corpus of trajectories $\{(\tau_i, r_i)\}$ from multiple LLMs, using ground truth scores $r = \text{score}(\hat{y}, y)$ as rewards, and trains via DPO/rejection sampling. Trajectories with $r > \theta$ are selected as positive samples.
        - **Online RL**: The agent collects real-time trajectories through environment interaction and optimizes policy $\pi_\theta$ via PPO/GRPO, with reward $R(\tau) = \text{score}(\hat{y}_\tau, y)$.
    - **Design Motivation**: Offline RL is data-efficient (reusing existing trajectories), while online RL enables continuous exploration and improvement.

### Loss & Training
- **Backbone**: Llama-3.1-8B-Instruct serves as the base model for Med-Copilot.
- **Offline phase**: Trajectories are collected from GPT-4o-mini, Claude-3.5-Sonnet, DeepSeek-V2.5, and other models.
- **Online phase**: Med-Copilot interacts directly with the environment, updating its policy at each iteration.

## Key Experimental Results

### Main Results: Benchmark Evaluation of 29 LLMs

| Model Category | Representative Model | Avg. Score | Rank |
|---|---|---|---|
| Closed-source commercial | GPT-4o | ~0.55 | Top-1 |
| Closed-source commercial | Claude-3.5-Sonnet | ~0.50 | Top-3 |
| Open-source base | Llama-3.1-8B-Instruct | ~0.32 | Lower-mid |
| Med-Copilot (Offline RL) | Llama-3.1-8B + Offline RL | ~0.46 (+43.02%) | Near GPT-4o |
| Med-Copilot (Online RL) | Llama-3.1-8B + Online RL | ~0.46 (+45.28%) | Competitive with GPT-4o |

### Ablation Study

| Configuration | Gain | Notes |
|---|---|---|
| Offline RL only | +43.02% | Learns from multi-model trajectories |
| Online RL only | +45.28% | Self-directed exploration; marginally superior |
| Multi-turn vs. single-turn | Significant improvement | Demonstrates value of interactive feedback |
| Task difficulty stratification | Larger gains on easy tasks | Hard tasks retain room for improvement |

### Key Findings
- A substantial performance gap (~20 points) exists between commercial and open-source LLMs on biomedical coding tasks, but RL training can significantly narrow this gap.
- Online RL marginally outperforms offline RL, yet both substantially surpass SFT baselines.
- Multi-turn interaction (debugging loops) is critical — single-pass code generation achieves far lower success rates than iterative refinement.
- Task difficulty varies considerably across biomedical scenarios: basic statistical analysis is relatively straightforward, whereas complex genomic pipeline analysis remains challenging.

## Highlights & Insights
- **Unified Training and Evaluation**: MedAgentGym serves not only as a benchmark (evaluating 29 LLMs) but also as a training environment with a directly usable RL pipeline — a first in the medical AI domain.
- **Empirical Proof of Gap Closure**: An 8B open-source model trained with RL achieves GPT-4o-level performance, which is of substantial practical value for privacy-sensitive medical settings (local deployment vs. API calls).
- **Scalable Infrastructure**: 72K tasks + multi-threaded trajectory sampling + standardized interfaces constitute a genuinely scalable training infrastructure.
- **Code-Centric Rather Than QA-Centric**: Unlike multiple-choice benchmarks such as MedQA, MedAgentGym requires agents to write real, executable analysis code — more closely reflecting actual research practice.

## Limitations & Future Work
- The task suite is code-centric; knowledge-intensive capabilities such as clinical reasoning and diagnostic decision-making are underrepresented.
- Ground truth requires predefined standard answers, making the framework unsuitable for open-ended exploratory research tasks.
- Med-Copilot is only trained at the 8B scale; scaling results for larger models (70B+) are not reported.
- Scoring functions are primarily based on exact match or numerical error; soft metrics such as code quality, readability, and efficiency are not assessed.
- Out-of-distribution generalization beyond the training task distribution is not evaluated.

## Related Work & Insights
- **vs. MedQA/PubMedQA**: These are static QA benchmarks with no code execution or interactive feedback; MedAgentGym supports multi-turn code interaction.
- **vs. SWE-bench**: SWE-bench targets software engineering (bug fixing), whereas MedAgentGym targets biomedical data analysis — fundamentally different task natures.
- **vs. AgentBench**: AgentBench covers diverse agent tasks but lacks medical focus; MedAgentGym provides deep coverage of biomedical scenarios.
- **vs. AIME**: AIME and similar benchmarks assess medical reasoning, whereas MedAgentGym evaluates practical medical programming.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First unified agentic training environment for biomedical data science; the problem formulation is valuable, though the RL training methodology itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 72K tasks + systematic evaluation of 29 LLMs + offline/online RL comparison + Med-Copilot validation.
- **Writing Quality**: ⭐⭐⭐⭐ — System description is clear; task taxonomy and experimental organization are well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — Provides critical infrastructure for biomedical AI agent research; the open-source environment offers long-term community value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)
- [\[NeurIPS 2025\] CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning](../../NeurIPS2025/medical_imaging/codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)

<!-- RELATED:END -->
