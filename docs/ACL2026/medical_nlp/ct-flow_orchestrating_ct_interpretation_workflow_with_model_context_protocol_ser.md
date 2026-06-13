---
title: >-
  [Paper Note] CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers
description: >-
  [ACL 2026][Medical NLP][CT Interpretation] The authors recast 3D CT interpretation as an agentic task where "radiologists iteratively explore using tools." Utilizing Model Context Protocol (MCP)…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "CT Interpretation"
  - "MCP"
  - "Agentic LVLM"
  - "ReAct"
  - "Tool Orchestration"
  - "CT-FlowBench"
date: 2026-05-08
content_hash: ea0eb63aa9a7b8c7
---

# CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers

**Conference**: ACL 2026  
**arXiv**: [2603.00123](https://arxiv.org/abs/2603.00123)  
**Code**: Not yet released (Paper promises CT-FlowBench)  
**Area**: Medical Imaging / 3D CT Agents / MCP Tool Orchestration  
**Keywords**: CT Interpretation, MCP, Agentic LVLM, ReAct, Tool Orchestration, CT-FlowBench

## TL;DR
The authors recast 3D CT interpretation as an agentic task where "radiologists iteratively explore using tools." Utilizing Model Context Protocol (MCP), they expose four categories of tools: Data Ingestion, Global Navigation, Detailed Observation, and Advanced Analysis. They constructed CT-FlowBench with 2,000+300 executable trajectories and performed SFT to create CT-Flow-8B. This model achieved 69.46% ACC on 3D-RAD, representing a +22.46% Gain over pure slice baselines, with a tool invocation name error rate of only 0.007/case.

## Background & Motivation
**Background**: Current 3D CT large models follow two mainstream paths: native 3D backbones (e.g., M3D, RadFM using 3D ViT full convolutional encoding) and slice serialization (e.g., Hulu-Med, OmniCT splitting volume data into 2D slice sequences). Both approaches perform end-to-end "one-shot" inference, where the model consumes the entire volume and outputs text as a diagnosis or report.

**Limitations of Prior Work**: (1) 3D encoding inherently carries an "information bottleneck"—voxels are compressed into limited tokens, washing away clinically decisive subtle signs such as small hematomas, early ischemia, or faint ground-glass opacities. (2) This "read-only" mode is asynchronous with real clinical workflows; radiologists actually scroll through slices, switch planes, measure HU values and diameters, and call segmentation/radiomics tools on demand in a highly interactive iterative process. (3) Existing medical agents (e.g., ChatCAD, ChatCAD+, Med-Agents, MedRAX), while introducing tool calls, utilize tools in a temporary, isolated, and scattered manner, failing to support complex "multi-step, traceable, and verifiable" 3D CT workflows.

**Key Challenge**: CT diagnosis requires "fine-grained evidence + multi-step reasoning + tool verification," whereas end-to-end LVLMs offer "coarser perception + single-step prediction + no traceability"—leading to a misalignment between architectural form and task requirements.

**Goal**: To reformulate 3D CT interpretation from a "perception problem" into an "agentic problem," allowing the model to explicitly plan, invoke tools, verify hypotheses, and produce auditable reasoning trajectories.

**Key Insight**: The Model Context Protocol (MCP) recently proposed by Anthropic standardizes the interface between LLMs and external tools. It can encapsulate heterogeneous image processing tools (slicing, segmentation, radiomics, measurement) into a unified tool space. Combined with the ReAct paradigm, the diagnostic process is decomposed into a sequence of $(s_t, a_t, o_t)$ triplets.

**Core Idea**: Using MCP as the foundation, the LVLM is upgraded from a "passive encoder" to an "active orchestrator." This approach provides both a tool space and execution environment (CT-Flow framework), an executable trajectory training/evaluation benchmark (CT-FlowBench), and trained Qwen2.5/Qwen3-VL small models to prove this paradigm is more effective than scaling parameters.

## Method

### Overall Architecture
The CT-Flow system consists of three layers:

1. **Tool Space**: Four types of MCP toolsets cover the entire pipeline from "data ingestion" to "pre-decision verification." Data Ingestion encapsulates CT volume data and metadata into standard query interfaces; Global Navigation provides full-volume and coarse anatomical localization; Detailed Observation retrieves high-resolution slices or sub-volumes for local evidence verification; Advanced Analysis provides quantitative analysis such as HU value measurement, segmentation, and radiomics. Each tool is exposed to the LLM via FASTMCP.
2. **Orchestrator**: This can be a general frontier model like GPT-5.2/Gemini-3-Pro or a 7B/8B small model fine-tuned via CT-Flow SFT. At each timestep, it generates a thought $s_t$, selects a tool call $a_t$, and receives an observation $o_t$ from the imaging environment.
3. **CT-FlowBench**: Constructed based on CT-RATE, it contains 2,000 training and 300 evaluation trajectories. Each trajectory is an $(s, a, o)$ sequence with gold-standard final answers.

During diagnosis, given a clinical query $Q$, the system generates a complete Reasoning-Acting Trajectory $\mathcal{T} = \{(s_0, a_0, o_0), \ldots, (s_n, a_n, o_n)\}$, where the final answer $A$ is synthesized from cumulative evidence rather than a single prediction.

### Key Designs

1. **MCP-Standardized Four-Level Tool Stack (Data Ingestion → Global Navigation → Detailed Observation → Advanced Analysis)**:
    - **Function**: Abstraction of all heterogeneous low-level operations in 3D medical imaging (DICOM reading, axial/coronal switching, ROI cropping, HU measurement, segmentation, radiomics) into a set of composable atomic actions exposed via a unified MCP interface.
    - **Mechanism**: Follows the natural hierarchy of clinical workflow—data must be loaded (Ingestion, always a precondition), followed by coarse localization (Navigation), detailed observation of regions of interest (Observation), and finally quantitative analysis (Analysis). FASTMCP acts as a bridge connecting high-level servers to low-level imaging infrastructure. The LLM only sees tool names, parameter schemas, and text/image observations, without needing to know the underlying implementation.
    - **Design Motivation**: Compared to ad-hoc stitching, MCP standardization ensures tools are hot-pluggable and sharable across models. The four-level classification creates hierarchical constraints, providing clear dependencies for planning (one cannot perform Analysis before Ingestion) and reducing the agent's search space. Ablations (Table/Fig 4) show that removing any tool category leads to significantly decreased ACC and increased format errors, proving the tools are complementary and essential.

2. **Execution-in-the-loop Trajectory Synthesis + Procedural Consistency Filter**:
    - **Function**: Upgrades static VQA supervision (medical record → answer) to executable trajectory supervision (medical record → (thought, action, observation)* → answer), ensuring every intermediate observation is reproducible on real CT volumes and the chain hits the gold-standard diagnosis.
    - **Mechanism**: Cases are screened from CT-RATE based on anatomical diversity, diagnostic richness, and potential for quantitative evaluation, retaining cases with high reasoning density. Teacher models (GPT-4o / Gemini-3-Pro-Preview / GPT-5.2 / Claude-Sonnet-4.5) explore multiple candidate trajectories. Only trajectories satisfying $\forall (a_i, o_i)\in \mathcal{T},\ \text{val}(o_i \mid \mathcal{V}) \land \text{pred}(\mathcal{T}) = y_{gt}$ are kept—meaning all observations must be factually obtainable from the raw volume $\mathcal{V}$, and the chain must terminate at the gold standard $y_{gt}$.
    - **Design Motivation**: Traditional distillation trajectories are prone to "hallucinated execution," where teacher models might fabricate intermediate values (e.g., an arbitrary HU=-800). Execution-in-the-loop validates teacher trajectories in a real tool environment, a critical filtering step for data credibility and tool executability. This supports multi-level capability assessment across three scenarios (Quantitative Analysis / Spatial Mapping / Diagnostic Inference).

3. **Trajectory-form Instruction Tuning on Small Backbones**:
    - **Function**: Directly implants "agentic capabilities" into small models like Qwen2.5-VL-7B / Qwen3-VL-8B via SFT, enabling them to actively invoke tools without massive parameter counts.
    - **Mechanism**: The training set consists of 2,000 trajectories = CT-RATE distilled execution-level trajectories + 3D-RAD subset. Full-parameter SFT is performed using the LLaMA-Factory framework with a learning rate of $1\times 10^{-5}$, DeepSpeed ZeRO-2 + cosine decay on 4×H100. Each sample is a complete ReAct trajectory; the model simultaneously learns to generate thoughts, invoke valid tools, and consume tool outputs.
    - **Design Motivation**: While frontier models can perform tool calls zero-shot, small models like Qwen3-VL-8B achieve only 25.33% on CT-FlowBench zero-shot with tool name error rates as high as 0.969/case. SFT allows the 8B model to reach 69.46% on 3D-RAD (surpassing the 235B Qwen3-VL) and reduces name error rates to 0.027, proving "small models + high-quality trajectories + clear tool interfaces" is a more efficient path than simply stacking parameters.

### Loss & Training
Standard token-level autoregressive cross-entropy is used for next-token prediction across the entire trajectory (thought + action + observation). No additional reinforcement learning was introduced (authors suggest considering PPO/DPO/RLCF in the future). Key hyperparameters: lr $1\times 10^{-5}$, DeepSpeed ZeRO-2, cosine schedule, 4×H100. SGLang provides an OpenAI-compatible API during inference. For evaluation, ACC is used for multiple-choice; BLEU-4/ROUGE-L/BERTScore + LLM-as-Judge (averaging DeepSeek-V3, Kimi-K2-Thinking, and GPT-OSS-120B) are used for open-ended QA.

## Key Experimental Results

### Main Results
Comparison on 3D-RAD (200 stratified samples per sub-task) and CT-FlowBench (300 trajectories across QA/AM/DD categories):

| Model | Tool-use | 3D-RAD ACC↑ | 3D-RAD BLEU-4 | 3D-RAD ROUGE-L | CT-FlowBench Avg↑ |
|------|----------|-------------|---------------|----------------|-------------------|
| GPT-5.2 | ✓ | 63.50 | 22.08 | 26.06 | 37.33 |
| Gemini-3-Pro-Preview | ✓ | 62.59 | 29.59 | 35.59 | 44.00 |
| Claude-Sonnet-4.5 | ✓ | 54.83 | 20.14 | 26.81 | 43.67 |
| Qwen3-VL-235B-A22B | ✓ | 54.21 | 20.55 | 22.62 | 34.00 |
| M3D-RAD (Medical Specialized) | ✗ | 58.00 | 29.76 | 37.39 | 36.00 |
| Hulu-Med-7B | ✗ | 61.29 | 12.77 | 23.71 | 47.00 |
| Qwen2.5-VL-7B (Baseline) | ✓ | 26.83 | 18.33 | 23.46 | 19.00 |
| Qwen3-VL-8B (Baseline) | ✓ | 49.06 | 20.89 | 22.31 | 25.33 |
| **CT-Flow-7B (Ours)** | ✓ | 61.36 | 36.67 | 34.73 | 44.33 |
| **CT-Flow-8B (Ours)** | ✓ | **69.46** | **36.96** | 37.47 | 43.00 |

CT-Flow-8B’s ACC on 3D-RAD is +20.40 higher than its backbone Qwen3-VL-8B, surpassing all medical-specific models and frontier models. Its BLEU-4 of 36.96 is the highest, indicating that SFT not only improves accuracy but also aligns report style with clinical standards. On CT-FlowBench, Hulu-Med-7B is highest (47); the authors attribute this to its pure multimodal discriminative path bypassing the risk of long-horizon tool failure, which plagues all agent-based models.

### Ablation Study
Tool usage statistics (Table 2) and tool category ablation (Fig 4 description):

| Model | 3D-RAD Calls | 3D-RAD Name Err | 3D-RAD Arg Err | CT-FlowBench Calls | CT-FlowBench Name Err | CT-FlowBench Arg Err |
|------|--------------|-----------------|----------------|--------------------|------------------------|----------------------|
| GPT-5.2 | 4.13 | 0.006 | 0.056 | 7.19 | 0.003 | 0.108 |
| Claude-Sonnet-4.5 | 5.93 | 0.002 | 0.092 | 9.49 | 0.017 | 0.407 |
| Qwen3-VL-8B (zero-shot) | 5.96 | **0.782** | 0.211 | 11.20 | **0.969** | 0.385 |
| **CT-Flow-7B (Ours)** | 4.01 | **0.007** | **0.018** | 6.17 | **0.007** | **0.033** |
| **CT-Flow-8B (Ours)** | 4.25 | 0.007 | 0.057 | 7.48 | 0.027 | 0.282 |

Tool category ablation (Ingestion is mandatory; others removed one by one): Removing Advanced Analysis → loss of quantitative synthesis (cannot calculate flow rates or volume ratios); removing Detailed Observation → sensitivity to small lesions plummets; removing Global Navigation → results in "spatial disorientation," unable to jump between slices efficiently. All three categories were verified as indispensable.

### Key Findings
- **Tool-mediated reasoning significantly boosts 3D-RAD performance**: CT-Flow SFT yields a +22.46% Gain, GPT-5.2 +8.33%, and Claude-Opus +12.83%. General models using CT-Flow can outperform specialized medical models (M3D-RAD at 58.00%), proving agentic frameworks can bypass medical pre-training barriers at lower costs.
- **Tool invocation discipline is a hidden bottleneck for model capability**: Qwen3-VL-8B zero-shot tool name error rates are 0.782 (3D-RAD) and 0.969 (CT-FlowBench). SFT reduces this to 0.007, a root Cause for the performance leap—demonstrating that small models lack "compliance" rather than "understanding."
- **CT-FlowBench is more challenging than 3D-RAD**: Average scores for all models are lower than their 3D-RAD counterparts (max 44.33), highlighting that the cognitive load of multi-step tool planning is much higher than one-shot visual judgment.
- **CT-Flow-7B has the lowest call frequency (4.01/6.17)**: The model learned to "reach the correct answer with fewer tools," which is critical for latency-sensitive clinical scenarios like stroke emergencies.

## Highlights & Insights
- **Introducing MCP to medical imaging is a clever paradigm shift**: Utilizing MCP leverages ecosystem maturity (FASTMCP, compliance checks, tool registration), avoiding the need to reinvent tool interfaces for every medical agent and providing a truly scalable path.
- **Execution-in-the-loop filtering is key to trajectory data credibility**: Generating trajectories solely from teacher models introduces "hallucinated tool calls." Running every $(a, o)$ on real CTs ensures data quality constraints are hard-coded into the process, preventing SFT from learning incorrect tool behaviors.
- **"Small Models + High-Quality Trajectories" beats "Large Models + Zero-shot"**: CT-Flow-8B surmounting GPT-5.2 and Gemini-3-Pro-Preview on 3D-RAD has direct practical value for deployment-sensitive medical environments (e.g., hospitals with cloud restrictions).
- **ReAct triplets naturally provide auditability**: Every $(s, a, o)$ can be retrospectively checked by radiologists, satisfying the "interpretable reasoning path" requirement critical for FDA/CE regulatory approval.

## Limitations & Future Work
- Authors acknowledge the following limitations: (1) Only SFT was performed without RL (PPO/DPO/RLCF); (2) Multi-step reasoning latency is higher than single end-to-end inference, challenging for emergency scenarios.
- External insights on limitations: (1) Training trajectories depend on top-tier teachers (GPT-5.2), which is expensive to reproduce; (2) The tool space covers only four categories, excluding advanced CT modalities like contrast-enhanced, dynamic/perfusion scans; (3) CT-FlowBench has only 300 test samples from a single source (CT-RATE), lacking evidence for generalization across vendors/protocols; (4) Verification was limited to chest CT, leaving abdomen/head/neck/vascular CT untouched; (5) LLM-as-Judge weighting across three models may introduce systematic bias.
- Future Work: (1) Optimize trajectory length and quantitative accuracy using RLCF; (2) Expand tool space to temporal 4D-CT, CTA, and PET-CT; (3) Introduce multi-center data for generalization; (4) Replace linear trajectories with tool-call graphs to support parallel hypothesis exploration; (5) Integrate attribute-level evaluation into CT-Flow training targets for a closed-loop system.

## Related Work & Insights
- **vs RadFM / M3D / Hulu-Med**: These remain end-to-end LVLMs (3D ViT or slice serialization). CT-Flow decomposes the LVLM into an "orchestrator + tools," bypassing 3D information bottlenecks by replacing "viewing" with "operating microscopes/rulers/segmenters."
- **vs ChatCAD / ChatCAD+ / Med-Agents / MedRAX**: Early medical agents used ad-hoc tool stitching. CT-Flow introduces MCP standardized interfaces and trajectory-level supervision, transforming isolated tool calls into trainable continuous strategies.
- **vs General ReAct Paradigm**: CT-Flow adds "execution reproducibility constraints" to the ReAct $(s, a, o)$ structure, engineering general agents into versions viable for real physical environments (CT volumes).
- **Insights**: The combination of MCP + trajectory SFT + execution-in-the-loop can be ported to all "high-dimensional data + tool-intensive" diagnostic tasks (e.g., WSI pathology, endoscopic video, industrial CT). Attribute QA evaluations can complement CT-Flow—the former evaluating "content correctness" and the latter "procedural correctness."

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic introduction of MCP to 3D CT interpretation, creating a tool space + executable benchmark + SFT paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 10+ baselines, two benchmarks, tool usage stats, and category ablations, though limited to chest CT and SFT.
- **Writing Quality**: ⭐⭐⭐⭐ Clear storyline, intuitive Figure 1, high information density in Tables 1/2, and thorough ethics sections.
- **Value**: ⭐⭐⭐⭐⭐ Provides the first reproducible open-protocol paradigm (MCP) for medical imaging agents, driving interpretable AI deployment in hospitals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution](pcoa_a_new_benchmark_for_medical_aspect-based_summarization_with_phrase-level_co.md)
- [\[ICLR 2026\] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](../../ICLR2026/medical_nlp/mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)

</div>

<!-- RELATED:END -->
