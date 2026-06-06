---
title: >-
  [Paper Note] Taming System Complexity: Demystifying Software Engineering Agents in Diagnosing Linux Kernel Faults
description: >-
  [ACL2026][Code Intelligence][Fault Localization] By establishing LinuxFLBench, a large-scale Linux kernel fault localization benchmark…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Fault Localization"
  - "Linux Kernel"
  - "LLM Agent"
  - "Code Localization"
  - "System Complexity"
date: 2026-05-08
content_hash: 83570e16e61fdc93
---

# Taming System Complexity: Demystifying Software Engineering Agents in Diagnosing Linux Kernel Faults

**Conference**: ACL2026  
**arXiv**: [2505.19489](https://arxiv.org/abs/2505.19489)  
**Code**: https://github.com/FudanSELab/LinuxFLBench  
**Area**: Software Engineering Agents / Code Intelligence  
**Keywords**: Fault Localization, Linux Kernel, LLM Agent, Code Localization, System Complexity

## TL;DR
By establishing LinuxFLBench, a large-scale Linux kernel fault localization benchmark, this work reveals the limitations of existing LLM agents in complex systems and proposes the LinuxFL+ framework. Through two-dimensional expansion—directory awareness and latent causes—the framework significantly improves fault localization accuracy at low cost.

## Background & Motivation

**Background**: Fault Localization (FL) is a classic problem in software engineering, aiming to automatically identify buggy code locations from given bug reports and source code. Recently, Large Language Model (LLM) driven agents (e.g., SWE-Agent, AutoCodeRover, Agentless) have achieved significant progress in general software systems, reaching approximately 70% accuracy on the SWE-bench benchmark and demonstrating the ability of agents to autonomously explore codebases.

**Limitations of Prior Work**: However, the evaluation of these agents mainly focuses on medium-scale general software projects (such as Python libraries), and their applicability to truly massive and complex systems like the Linux kernel remains unknown. The Linux kernel presents three unique challenges: (1) Immense code scale: Kernel v5.8 contains 69K files and 28M lines of code, which is more than 30 times larger than the largest project in SWE-bench; (2) Limited observability: Because the kernel runs in privileged mode and must minimize overhead, user-reported bugs often lack detailed runtime information and debugging clues; (3) Multi-dimensional influencing factors: Hardware configurations, system loads, and timing factors can all lead to bugs, greatly expanding the reasoning space for diagnosis.

**Key Challenge**: While existing agents perform excellently in general software, these advantages may not transfer at all to a highly challenging real-world system like the Linux kernel. In-depth research is needed to understand their actual performance in this domain and find directions for improvement.

**Goal**: First, to construct the first large-scale Linux kernel fault localization benchmark; second, to comprehensively evaluate the performance of existing top-tier LLM agents on this task; third, to diagnose their main failure causes and propose effective enhancement solutions.

**Key Insight**: The authors start with an empirical study, revealing the actual deficiencies of agents through extensive experimental data before designing targeted improvement strategies. Key observations include: agents can usually accurately identify relevant high-level modules but struggle to pinpoint specific files within those modules; simultaneously, the exploration range of agents is too narrow, focusing only on a few possible causes and missing many related root causes.

**Core Idea**: The deficiencies of agents are addressed through structured expansion in two dimensions—expanding the search range using directory structures in the spatial dimension, and expanding the potential cause pool through direct hypotheses and email-assisted knowledge in the knowledge dimension. Finally, the final prediction is generated through aggregation and reranking.

## Method

### Overall Architecture

The workflow of LinuxFL+ is divided into three stages. First, any LLM agent (e.g., AutoCodeRover) runs independently to generate initial suspicious file predictions. Then, the process enters the expansion phase, including (1) **Directory-aware Expansion**: Starting from the directory of the initial prediction, the candidate set is expanded to allow the LLM to re-select from a larger set of related files; (2) **Latent Cause Expansion**: Multiple hypothetical bug causes are generated, with each hypothesis mapped to a list of files requiring modification; (3) **Candidate Integration**: Predictions from the three sources are aggregated, and the final file order is determined using aggregate scores and LLM reranking. This design fully utilizes the generative and understanding capabilities of LLMs while leveraging structured external knowledge (codebase structure and email knowledge bases) to correct agent blind spots.

### Key Designs

1. **Directory-aware Expansion**:

    - **Function**: Performs a two-stage search within the directory of the agent's initial prediction, expanding the candidate file range while maintaining local relevance.
    - **Mechanism**: While LLM agents can typically locate relevant high-level directories (modules), they often become confused when there are many related files within that directory. This design first collects the full list of files in directories where initial predictions are located (averaging 16 files per directory in the Linux kernel), and then prompts the LLM to re-filter and rank the top-10 most relevant files within this expanded candidate set. This effectively gives the LLM a second chance to make decisions with more complete information, using the directory structure—an explicit organizational form of the codebase—as a guide.
    - **Design Motivation**: Linux kernel directories contain an average of 16 files (compared to 8 in SWE-bench). This high density increases the difficulty of precise localization. By explicitly using directory boundaries as the search scope, the LLM can be provided with sufficiently detailed context for fine-tuning while maintaining module-level granularity effectiveness.

2. **Latent Cause Expansion**:

    - **Function**: Systematically expands possible bug causes through multiple hypothesis generation strategies, with each cause corresponding to a set of files to be modified, forming multiple candidate sets.
    - **Mechanism**: This design includes two layers of hypotheses. The first layer, **Direct Hypotheses**, uses only the LLM's pre-trained knowledge to prompt the model to generate $k$ hypotheses for causes that could lead to the bug, with each hypothesis also providing a corresponding code fix and involved files. The second layer, **Email-assisted Hypotheses**, introduces external knowledge: Retrieval-Augmented Generation (RAG) is used to retrieve relevant historical discussions from the Linux Kernel Mailing List (LKML). These emails often contain developer analyses of similar bugs. The retrieved email content and the original bug report are provided to the LLM to facilitate the generation of more diverse and well-founded cause hypotheses. To prevent data leakage, only emails sent before the bug report are retrieved; to improve retrieval quality, keywords are extracted from the bug report across four dimensions (behavior, cause, expected behavior, solution), followed by BM25 retrieval of the top-10 relevant emails.
    - **Design Motivation**: Real bug diagnosis is an iterative "guess-and-verify" process that cannot rely solely on initial intuition. By systematically generating multiple hypotheses, a wider root-cause space can be covered. Direct hypotheses utilize the LLM's general knowledge, while email hypotheses introduce domain-specific historical wisdom—the two are complementary. Experiments show that this multi-hypothesis strategy significantly improves performance, especially when bug causes are scattered (e.g., performance issues spanning multiple modules).

3. **Candidate Integration & Reranking**:

    - **Function**: Aggregates file predictions from the three sources (directory expansion, direct hypotheses, email hypotheses) into a unified ranking as the final localization result.
    - **Mechanism**: The aggregation strategy employs Reciprocal Rank Fusion. For each candidate file $f$, its rankings $R_{dir}(f)$, $R_{direct}(f)$, and $R_{mail}(f)$ are recorded from the three sources (if a file does not appear, its rank is $\infty$). The aggregate score is calculated as $\text{score}(f) = \frac{1}{R_{dir}(f)} + \frac{1}{R_{direct}(f)} + \frac{1}{R_{mail}(f)}$. This formula ensures that files ranked highly in any single method receive a higher score, while files ranked highly across multiple methods are further prioritized. Candidates are then sorted by score in descending order. Finally, the LLM is prompted to perform a final reranking of this preliminary list based on the semantic correspondence between file paths and the bug report, utilizing the LLM's semantic understanding for final adjustments.
    - **Design Motivation**: The three information sources provide unique perspectives—directory expansion captures structural associations, direct hypotheses leverage general LLM knowledge, and email hypotheses provide domain heuristics. Reciprocal rank aggregation is a classic rank fusion method that is both simple and effective at synthesizing multi-source information. The final LLM reranking introduces semantic awareness to avoid the potential rigidity of pure numerical aggregation.

## Key Experimental Results

### Main Results

The LinuxFLBench constructed in this paper contains 250 real Linux kernel bugs, covering 120 kernel versions and 66 different kernel sub-modules (e.g., drivers, networking, file systems). Bug reports average 283 words, and the corresponding codebases average 28,808 files and 11.49 million lines of code, far exceeding SWE-bench (averaging 195 words, 3,010 files, and 438,000 lines).

Table 1 shows the performance comparison for file-level fault localization:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|------|----------|----------|-----------|-----|
| BM25 (IR Baseline) | 0.168 | 0.328 | 0.396 | 0.231 |
| BugLocator | 0.127 | 0.209 | 0.272 | 0.215 |
| BLUiR (Best Trad. IR) | 0.228 | 0.317 | 0.404 | 0.321 |
| SWE-Agent | 0.416 | 0.552 | 0.584 | 0.476 |
| SWE-Agent + LinuxFL+ | **0.524** | **0.720** | **0.768** | **0.610** |
| AutoCodeRover | 0.388 | 0.496 | 0.496 | 0.435 |
| AutoCodeRover + LinuxFL+ | **0.500** | **0.712** | **0.744** | **0.589** |
| Agentless | 0.368 | 0.492 | 0.504 | 0.419 |
| Agentless + LinuxFL+ | **0.440** | **0.684** | **0.724** | **0.549** |

Key Observations: (1) While existing agents far outperform traditional IR methods, their performance on LinuxFLBench is still much lower than on SWE-bench (Recall@1 drops by 15%+), revealing the true challenge of system complexity; (2) Even the best agent (SWE-Agent) achieves a Recall@1 of only 41.6%, showing that Linux kernel FL remains a highly challenging task; (3) LinuxFL+ brings significant improvements to all three agents, increasing SWE-Agent’s Recall@1 from 41.6% to 52.4% (+10.8%) and AutoCodeRover's from 38.8% to 50.0% (+11.2%).

### Ablation Study

Table 2 shows performance under different difficulty levels (distinguished by whether file names are explicitly mentioned in the bug report):

| Difficulty Level | Agentless Baseline | AutoCodeRover Baseline | SWE-Agent Baseline | LinuxFL+ Avg Gain |
|---------|---------------|-------------------|---|---|
| Easy (Clear file hints) | 0.605 | 0.623 | 0.664 | +0.105 |
| Hard (No file hints) | 0.273 | 0.287 | 0.341 | +0.127 |

The results show that LinuxFL+ is particularly adept at handling "Hard" cases (lacking explicit file clues), which is the target scenario for the expansion strategy. At the symptom level, LinuxFL+ provides the most significant boost for bugs with vague symptoms (e.g., performance issues, baseline MRR 0.165), increasing MRR to 0.458 (+177%). For bugs with clear symptoms (e.g., Watchdog errors), it still provides gains despite the high baseline (0.833). In terms of cost, LinuxFL+ uses an additional 11.8K-15.3K tokens per task, costing approximately $0.04—only about 10% of the agents' base cost.

## Highlights & Insights

- **First Large-scale Kernel FL Benchmark**: Previous kernel-related benchmarks were either too small (Linux-3.16 only) or used unrealistic sources (fuzzer-detected crashes). LinuxFLBench covers 250 diverse bugs from real user reports across 120 versions and 66 components, making it the first benchmark to truly reflect the difficulty of kernel fault localization.
- **Revealing the Limits of Agent Capabilities**: The study clearly demonstrates the limitations of general agents in large complex systems through empirical data—a performance drop of 15%+—and qualitatively analyzes two primary failure modes: confusion regarding related files and narrow exploration range. These insights offer guidance on when and how to effectively enhance agents.
- **Low-cost, High-efficiency Enhancement**: Compared to retraining agents or fine-tuning models from scratch, LinuxFL+ acts as a post-processing framework with minimal extra cost ($0.04/task) while delivering significant gains. The introduction of email knowledge particularly demonstrates a paradigm for fusing domain-specific knowledge with general LLM capabilities.
- **Generalizable Design**: Although designed for the Linux kernel, the ideas of directory-aware expansion and latent cause expansion are transferable to fault localization tasks in other large complex systems (e.g., operating systems, databases).

## Limitations & Future Work

The authors acknowledge several main limitations:

- **LLM Selection**: While GPT-4o and Qwen3-32B were used, the focus remains on GPT-4o. The performance of smaller/larger or open-source models requires further exploration.
- **Coarse Utilization of Email Data**: LKML content is rich but messy; filtering strategies (avoiding external links, limiting modified file counts, etc.) remain heuristic. Future work could explore more refined email extraction and matching methods.
- **Function-level Localization**: Recall@1 for function-level localization is only 0.089-0.138, much lower than file-level. This suggests that while LinuxFL+ improves file-level localization, fine-grained localization challenges persist and may require more granular code understanding strategies.

**Future Directions**:

- Explore advanced email retrieval strategies, such as structured email parsing and multi-hop reasoning, to extract precise root-cause knowledge.
- Customize hypothesis generation strategies for different kernel components rather than using a global strategy.
- Combine program analysis (e.g., data dependency, control flow) to enhance LLM-based reasoning.

## Related Work & Insights

- **Vs. Traditional IR-based FL (BugLocator, BLUiR)**: Traditional methods rely on bag-of-words similarity and struggle with concept drift and complex dependencies at the kernel level (MRR 0.2-0.3). This work shows that LLM agents excel at symbolic and multi-hop reasoning but still require external enhancement for ultra-large systems.
- **Vs. General Agents (SWE-Agent, AutoCodeRover, Agentless)**: These agents perform well on SWE-bench (Python projects) but decline sharply on LinuxFLBench. This underscores challenges brought by scale, complexity, and observability, suggesting that future agent designs should be more domain-aware.
- **Vs. Other Code Localization Work (LocAgent, AgentFL)**: Unlike work focusing on improving core agent architectures, this framework emphasizes structured post-processing and knowledge fusion for rapid iterative improvement without retraining.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic evaluation of LLM agents for Linux kernel FL; benchmark construction and problem definition are clear. The two-dimensional expansion idea is intuitive yet effective and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The benchmark of 250 real bugs is substantial; evaluation across three mainstream agents is comprehensive. Ablation studies reveal component contributions, and fine-grained analysis covers symptoms, difficulty, and cost.
- **Writing Quality**: ⭐⭐⭐⭐ Clear organization with a natural logical progression from benchmark construction to analysis and design. Qualitative diagnosis of failure modes is precise.
- **Value**: ⭐⭐⭐⭐ High practical value for industry Linux kernel maintenance teams; the benchmark provides significant reference for future research. The enhancement scheme is low-cost and highly effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[ACL 2026\] RExBench: Can coding agents autonomously implement AI research extensions?](rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)

</div>

<!-- RELATED:END -->
