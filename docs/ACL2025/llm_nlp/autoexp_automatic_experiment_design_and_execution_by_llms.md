---
title: >-
  [Paper Note] AutoExp: Automatic Experiment Design and Execution by LLMs
description: >-
  [ACL 2025][LLM (Other)][Automatic experiment design] This paper proposes the AutoExp framework, which leverages LLMs as intelligent agents to automatically complete the entire workflow of NLP experiments—from research question analysis, experimental design, and code generation/execution to result analysis and interpretation. It demonstrates the feasibility and limitations of LLM-based automated scientific experimentation across multiple standard NLP research scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Automatic experiment design"
  - "LLM Agent"
  - "Experiment execution"
  - "Scientific research automation"
  - "Code generation"
date: 2026-05-08
content_hash: 259ccba795798d7a
---

# AutoExp: Automatic Experiment Design and Execution by LLMs

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Automatic experiment design, LLM Agent, Experiment execution, Scientific research automation, Code generation

## TL;DR
This paper proposes the AutoExp framework, which leverages LLMs as intelligent agents to automatically complete the entire workflow of NLP experiments—from research question analysis, experimental design, and code generation/execution to result analysis and interpretation. It demonstrates the feasibility and limitations of LLM-based automated scientific experimentation across multiple standard NLP research scenarios.

## Background & Motivation

**Background**: With the increasing capabilities of LLMs, AI for Science has become a popular direction, particularly in automating scientific research workflows using LLMs. Existing work primarily focuses on literature review automation (e.g., AutoSurvey), hypothesis generation (e.g., AI Scientist), and code generation assistants (e.g., GitHub Copilot). However, automated end-to-end execution from experimental design to execution remains highly challenging, as it requires combining domain knowledge understanding, experimental methodology, programming skills, and result analysis capabilities.

**Limitations of Prior Work**: NLP researchers spend a significant amount of time during the experimentation phase on "engineering-heavy" tasks—such as writing training scripts, debugging data processing pipelines, configuring hyperparameter sweeps, and parsing log results. Although important, these tasks are highly templated and theoretically automatable. However, current code generation tools can only handle individual coding tasks and fail to comprehend the full experimental intent, let alone autonomously plan and execute multi-step experimental workflows.

**Key Challenge**: Complete experimental automation requires a synergy of "understanding why the experiment is conducted" (domain knowledge) and "knowing how to execute the experiment" (engineering proficiency). While LLMs possess foundational capabilities in both areas, chaining them into a reliable closed-loop system poses severe challenges in state management, error recovery, and experimental reproducibility.

**Goal**: To build an end-to-end, LLM-driven automated experimentation framework that can accept high-level research question descriptions, automatically design experimental plans, generate and execute code, gather results, and generate analysis reports.

**Key Insight**: The authors model experimental automation as a multi-stage Agent task, where each phase (design $\to$ coding $\to$ execution $\to$ analysis) is managed by a specialized Agent module connected via a structured information transmission protocol.

**Core Idea**: Utilizing a multi-agent collaboration and staged execution architecture, complex experimental automation tasks are decomposed into manageable sub-tasks. Self-verification and error-recovery mechanisms are introduced to ensure the reliability of the experiments.

## Method

### Overall Architecture
AutoExp comprises four core Agents: (1) Experiment Designer Agent—receives research question descriptions and outputs structured experimental plans (including dataset selection, baseline methods, evaluation metrics, and hyperparameter search spaces); (2) Code Generator Agent—generates executable code based on the experimental plan, covering the entire pipeline of data processing, model training, and evaluation; (3) Execution Manager Agent—manages the code execution environment, handling dependency installation, GPU allocation, running monitoring, and error recovery; (4) Result Analyzer Agent—parses experimental logs to generate statistical analysis, comparative tables, and summary insights. The four Agents exchange information via a structured JSON protocol.

### Key Designs

1. **Structured Experiment Plan (SEP)**:

    - **Function**: Translating ambiguous research questions into precise, executable experimental designs.
    - **Mechanism**: Upon receiving the research question, the Experiment Designer Agent refines the plan through multiple rounds of self-questioning—the first round identifies core variables (What to test), the second determines experimental conditions (How to test, including datasets, metrics, baselines), and the third defines experimental scale (How much to test, including hyperparameter search ranges and repetitions). The output is a structured JSON experimental plan containing a clear list of `experiments`, where each experiment features a name, objective, configuration parameters, and expected outcomes. Additionally, the Agent conducts scientific rigor verification, ensuring proper control variables are established and examining potential confounding factors.
    - **Design Motivation**: Unstructured natural language plans generated by LLMs are difficult for downstream Code Generator Agents to parse accurately. Structured JSON output guarantees lossless information transmission.

2. **Self-Verifying Code Generation (SVCG)**:

    - **Function**: Generating correct, executable experimental code with automatic verification.
    - **Mechanism**: The Code Generator Agent employs a "generation-verification-fix" loop. It first generates a complete set of code files (including `config.py`, `data_loader.py`, `model.py`, `train.py`, `evaluate.py`, etc.) based on the experimental design. It then runs syntax checks and unit tests in a sandboxed environment, automatically repairing errors based on feedback. A key design is the "template+customization" hybrid generation: common code structures (such as PyTorch training loops, HuggingFace data loaders) utilize pre-verified templates, while experiment-specific parts are dynamically generated. Code coverage checks are introduced to ensure that the generated evaluation code covers all metrics outlined in the experimental plan. A maximum of 5 repair loops is enforced.
    - **Design Motivation**: Pure end-to-end code generation exhibits high error rates, especially regarding library dependency compatibility and data format processing. The "template+customization" approach leverages templates to guarantee basic correctness while retaining customization flexibility.

3. **Adaptive Error Recovery (AER)**:

    - **Function**: Automatically diagnosing and fixing issues when code execution fails.
    - **Mechanism**: The Execution Manager maintains a knowledge base of error types and solutions, covering common execution faults: (a) Dependency errors (e.g., package incompatibilities): automatically attempts different package version combinations; (b) Data errors (e.g., missing file paths, mismatched formats): inspects and corrects the data processing pipeline; (c) Resource errors (e.g., GPU OOM): automatically scales down the batch size or enables gradient accumulation; (d) Runtime errors (e.g., NaN loss, gradient explosion): adjusts the learning rate or adds gradient clipping. For each error type, AER first attempts standard solutions from the knowledge base, falling back to open-ended diagnosis and repair by calling the LLM if they fail. Successful repairs update the knowledge base accordingly.
    - **Design Motivation**: Since 80% of debugging time in NLP experiments is spent on predictable environment and data issues, encoding these experiences into automated recovery rules drastically reduces manual intervention.

### Loss & Training
The AutoExp framework itself does not require training. The underlying LLM uses GPT-4 or Claude-3 as its reasoning engine. The model training code generated during the experiments uses standard task-specific loss functions (such as cross-entropy for text classification and CRF loss for sequence labeling). Hyperparameter search adopts a Bayesian optimization strategy (via generated Optuna configurations).

## Key Experimental Results

### Main Results

| Research Scenario | Completion Rate | Accuracy of Results | Number of Human Interventions | Total Duration (Human) | Total Duration (AutoExp) |
|---------|----------|----------|-----------|-----------|--------------|
| Sentiment Classification Comparison | 95% | 88% | 0.3 | 6h | 1.2h |
| Model Ablation Study | 90% | 82% | 0.8 | 8h | 2.1h |
| Hyperparameter Search | 98% | 95% | 0.1 | 12h | 2.5h |
| Cross-Dataset Generalization | 85% | 78% | 1.5 | 10h | 3.8h |
| New Method Implementation | 72% | 65% | 2.8 | 16h | 6.5h |

### Ablation Study

| Configuration | Average Completion Rate | Average Accuracy | Description |
|------|----------|----------|------|
| Full AutoExp | 88% | 82% | Complete framework |
| w/o SEP (Unstructured Plan) | 71% | 64% | Plan ambiguity leads to frequent code errors |
| w/o SVCG (Single-turn Generation) | 75% | 68% | No self-verification, leading to many code bugs |
| w/o AER (No Error Recovery) | 62% | 58% | Execution failures cannot be handled automatically |
| GPT-4 → GPT-3.5 | 68% | 60% | Downgrading LLM capability has a significant impact |

### Key Findings
- Templated tasks (such as hyperparameter search) achieve the highest completion rate and accuracy (98%/95%) due to highly standardized code structures; implementing new methods yields the lowest completion rate (72%/65%) due to the requirement for creative coding.
- Adaptive Error Recovery (AER) contributes the most to the completion rate improvement (+26 percentage points), indicating that error handling during the execution phase is the primary bottleneck for automated experimentation.
- Replacing GPT-4 with GPT-3.5 results in a significant performance drop (-20% in completion rate), illustrating that the current framework is highly dependent on advanced LLM capabilities.
- AutoExp demonstrates outstanding time-saving performance (an average speedup of 3-5x), even when factoring in human intervention time.

## Highlights & Insights
- The design of the multi-agent staged architecture exhibits strong engineering wisdom—each Agent only needs to focus on a relatively well-defined sub-task, thereby mitigating task complexity. This "divide-and-conquer" concept can be transferred to other complex automated workflows.
- The "template+customization" code generation strategy serves as a highly pragmatic compromise—utilizing templates to guarantee fundamental code correctness (e.g., training loops) while leaving flexible space for customization. This is significantly more reliable than pure end-to-end code generation.
- The self-updating mechanism of the error recovery knowledge base is highly valuable: every successful recovery enriches the knowledge base, making the system increasingly "smarter" over time.

## Limitations & Future Work
- The low completion rate (72%) for implementing new methods indicates that the framework is currently best suited for standardized replication and comparative experiments rather than genuinely innovative experiments.
- The reproducibility of experiments is not sufficiently verified—more validation is required to ensure whether automatically generated experiments can yield consistent results every time.
- It currently only supports the Python/PyTorch ecosystem, leaving frameworks like TensorFlow and JAX unsupported.
- Security considerations: Code generated by LLMs might contain security vulnerabilities (such as path injection or resource abuse), necessitating stronger sandbox isolation.
- In the future, AutoExp can be combined with automated paper writing to realize a complete closed-loop from ideas up to paper drafts.

## Related Work & Insights
- **vs AI Scientist (Lu et al., 2024)**: While AI Scientist covers the complete lifecycle from ideas to paper drafting, its experimental execution component remains basic. AutoExp specializes in the experimental phase, going deeper into execution reliability.
- **vs MLAgentBench (Huang et al., 2024)**: MLAgentBench evaluates LLM Agent capabilities on ML tasks but lacks a systematic error recovery mechanism; the AER module in AutoExp serves as the key differentiator.
- **vs ChatDev (Qian et al., 2023)**: ChatDev also exploits multi-agent collaboration for software development; AutoExp specializes similar concepts in the scientific research domain, introducing domain-specific verification mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of the end-to-end NLP experiment automation framework is novel, and the multi-agent collaboration architecture is rational.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across diverse experimental scenarios, with a comprehensive ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear system description with sound evaluation metric designs.
- Value: ⭐⭐⭐⭐ Holds significant exploratory value for NLP experimental automation, with highly practical potential in its error recovery mechanism.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution](aspera_a_simulated_environment_to_evaluate_planning_for_complex_action_execution.md)
- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)

</div>

<!-- RELATED:END -->
