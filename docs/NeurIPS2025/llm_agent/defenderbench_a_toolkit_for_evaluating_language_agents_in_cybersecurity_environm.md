---
title: >-
  [Paper Note] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments
description: >-
  [NeurIPS 2025][LLM Agent][Cybersecurity] This paper presents DefenderBench, an open-source modular toolkit for systematically evaluating LLM agents across three categories of cybersecurity tasks—offensive, defensive…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Cybersecurity"
  - "Benchmark"
  - "Vulnerability Detection"
  - "Network Intrusion Simulation"
date: 2026-05-08
content_hash: de7badf1b25fbfa8
---

# DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments

**Conference**: NeurIPS 2025
**arXiv**: [2506.00739](https://arxiv.org/abs/2506.00739)  
**Code**: [https://github.com/microsoft/DefenderBench](https://github.com/microsoft/DefenderBench)  
**Area**: LLM Agent
**Keywords**: Cybersecurity, LLM Agent, Benchmark, Vulnerability Detection, Network Intrusion Simulation

## TL;DR
This paper presents DefenderBench, an open-source modular toolkit for systematically evaluating LLM agents across three categories of cybersecurity tasks—offensive, defensive, and knowledge understanding—covering five scenarios: network intrusion simulation, malicious content detection, code vulnerability detection/repair, and CTI knowledge QA. Benchmark results show that Claude-3.7-sonnet achieves the best overall performance (81.65 points).

## Background & Motivation

**Background**: LLM agents have demonstrated strong capabilities in software development, document translation, and fact-checking, yet their evaluation in the cybersecurity domain remains insufficient. Existing security benchmarks (Cybench for CTF, CyberMetric for knowledge QA, CyberSecEval for code vulnerabilities) each focus on a single task type.

**Limitations of Prior Work**:
- Lack of a **unified** comprehensive evaluation platform covering offensive, defensive, and knowledge-understanding tasks
- Different works employ different evaluation frameworks, making fair cross-model comparisons difficult
- Most existing benchmarks are costly and hard to reproduce

**Key Insight**: Construct a **practical, open-source, modular** one-stop evaluation toolkit that enables researchers to fairly assess LLM agents on cybersecurity tasks at low cost.

## Method

### Overall Architecture

DefenderBench consists of three major modules:
1. **Data Preprocessing Module**: Automatically downloads, cleans, and splits datasets, caching them locally
2. **Task Environment Module**: Constructs an interactive environment for each task (providing instructions, defining action spaces, managing conversation history)
3. **Agent Interface Module**: A unified LLM agent interface supporting plug-and-play integration of both open-source and closed-source models

### Key Designs

**Five Cybersecurity Task Categories**:

1. **Network Intrusion Simulation (CyberBattleSim)**

    - Built on the CyberBattleSim simulator, converted into a text-based interactive game
    - Agents can execute three operations: `local_vulnerability` (local exploit), `remote_vulnerability` (remote attack), and `connect` (credential-based connection)
    - Two network topologies: Chain (simpler) and CTF (more complex)
    - Metric: node takeover rate (winning rate)

2. **Malicious Content Detection**

    - Malicious-Text: phishing email/SMS detection (20,137 samples, 500 test)
    - Malicious-Web: phishing webpage detection (15,612 samples, 500 test)
    - Metric: Macro-F1

3. **CTI Knowledge QA (MCQA)**

    - Based on the CTI-MCQA dataset; 2,338 four-choice questions on cyber threat intelligence
    - 500 test samples + 20 few-shot sample pool
    - Metric: Macro-F1

4. **Code Vulnerability Detection**

    - Vulnerable-CG: C-language function vulnerability detection based on CodeXGLUE
    - Vulnerable-DV: vulnerability detection based on Devign (FFmpeg + Qemu)
    - Metric: Macro-F1

5. **Code Vulnerability Repair (CVEFix)**

    - 240 single-method vulnerability repair samples covering C/C++/Go/Java/JS/PHP/Python/Rust
    - Given vulnerable code, the agent is required to generate a repaired version
    - Metric: CodeBLEU

**Global Metric**: DefenderBench Score = unweighted average of all task metrics

### Agent Baseline Design

A **minimalist scaffolding** baseline agent is adopted:
- Provides task instruction and response format requirements
- Supplies complete trajectory history (prior actions + observations) at each step
- Agent generates one action → sends to environment → receives observation → determines termination
- Maximum 5 steps for detection/QA tasks; maximum 100 steps for network intrusion tasks

### Loss & Training
This paper presents an evaluation benchmark rather than a training method; no loss function design is involved. All LLMs are evaluated directly without fine-tuning.

## Key Experimental Results

### Main Results

| Model | CBS-Chain | CBS-CTF | Mal.Text | Mal.Web | MCQA | Vuln-CG | Vuln-DV | CVEfix | DefB |
|-------|-----------|---------|----------|---------|------|---------|---------|--------|------|
| Naive Baseline | 19.4 | 22.2 | 52.4 | 50.4 | 25.0 | 50.0 | 47.8 | 83.2 | 43.8 |
| Llama 3.3 70B | **100.0** | 33.3 | 96.0 | 82.8 | 69.6 | 58.0 | 57.4 | 77.3 | 71.8 |
| GPT-4-turbo | 90.0 | 46.7 | 93.4 | 83.2 | 73.8 | **58.2** | 57.6 | 73.7 | 72.1 |
| Claude-3.5-sonnet | 100.0 | 56.7 | 93.8 | 88.2 | 72.4 | 56.4 | 56.8 | 75.7 | 75.0 |
| Claude-3.7-sonnet | **100.0** | **100.0** | 96.2 | 90.0 | 74.2 | 56.6 | 56.0 | 80.2 | **81.7** |
| Claude-3.7-sonnet-think | 100.0 | 76.7 | 94.4 | **91.0** | **78.2** | 54.6 | 52.8 | 79.5 | 78.4 |
| o3 | 83.3 | 20.0 | 92.4 | 88.0 | 76.4 | 30.8 | **59.6** | 55.6 | 63.9 |

### Ablation Study

**Model Scale Effect**:
- Llama 3.1 8B → 70B: DefB 54.7 → 68.7 (+14.0)
- Llama 3.2 1B → 3B: DefB 38.3 → 50.2 (+11.8)
- GPT-4.1 → 4.1-mini → 4.1-nano: 63.9 → 58.9 → 47.5 (larger scale consistently better)

**Few-Shot Augmentation**:
- Most large models benefit significantly from few-shot ICL
- Smaller models (Llama 3.2 1B/3B, Phi-3.5-mini) suffer performance degradation due to longer input context

**CoT Effect**:
- CoT is most effective for interactive tasks (network intrusion): GPT-4o gains +17.0 points
- CoT has limited effect on static tasks; some models show marginal performance drops

### Key Findings
- **Claude-3.7-sonnet is the strongest overall model** (81.65), achieving 100% winning rate on both network intrusion environments
- **Reasoning-augmented models (o1/o3/o4-mini) do not outperform base models**—reasoning capability alone is not the key factor for security tasks
- **Vulnerability detection remains the hardest task**—most models only marginally outperform random baselines, revealing LLM limitations in fine-grained program understanding
- **Small models perform extremely poorly on long-input scenarios** (e.g., HTML webpage detection)—Llama 3.2 1B even falls below the random baseline
- CodeBLEU may be an inadequate metric for vulnerability repair evaluation—a copy-paste baseline achieves the highest score

## Highlights & Insights
- **Comprehensiveness**: Currently the most complete LLM cybersecurity evaluation toolkit, covering offensive, defensive, and knowledge dimensions across five task types
- **Modular Design**: Users can easily integrate their own LLMs, agents, and new tasks; supports Weights & Biases visualization
- **Fair Comparison**: A unified agent framework and standardized data processing eliminate evaluation bias across different works
- **Practical Insights**: Reveals unexpected weaknesses of reasoning models on security tasks and the critical influence of model scale on security capabilities
- **Cost-Friendly**: Test set sizes are deliberately controlled (500 samples), making evaluation affordable for small and medium-sized research teams

## Limitations & Future Work
- **Overly Simple Agent Design**: Only a minimalist scaffolding baseline agent is used; more complex tool-augmented agents (e.g., integrating static analysis tools) are not explored
- **Inadequate CVEFix Metric**: CodeBLEU fails to accurately reflect the quality of small-scope code modifications; better evaluation metrics are needed
- **Expandable Task Coverage**: Important security scenarios such as social engineering, forensic analysis, and log analysis are not included
- **Limited Network Intrusion Environment**: CyberBattleSim's topology is relatively simplified and diverges considerably from real-world network environments
- **Security Risks of Agents Not Addressed**: As a dual-use technology, the paper does not thoroughly discuss countermeasures against the misuse of LLM agents

## Related Work & Insights
- **vs AgentBench/SWE-bench**: These general-purpose agent benchmarks do not cover the security domain; DefenderBench fills this gap
- **vs Cybench**: Cybench focuses solely on CTF, whereas DefenderBench has broader coverage (offensive + defensive + knowledge)
- **vs CyberSecEval**: CyberSecEval focuses on code security; DefenderBench additionally incorporates network intrusion and malicious content detection
- **Insights**: Future work could combine DefenderBench with red-teaming frameworks to evaluate the robustness of LLM agents in adversarial settings

## Rating
- Novelty: ⭐⭐⭐ Engineering contribution outweighs methodological innovation, yet fills an important evaluation gap
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 17+ models, 5 task types, and multiple augmentation strategies with thorough comparisons
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed task descriptions
- Value: ⭐⭐⭐⭐ Provides important reference for evaluating LLM security capabilities; the open-source toolkit has strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[NeurIPS 2025\] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
