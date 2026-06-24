---
title: >-
  [Paper Note] An Empirical Study on LLM-based Agents for Automated Bug Fixing
description: >-
  [ACL 2025][LLM Agent][Automated Bug Fixing] This paper systematically analyzes the top six LLM-based bug-fixing systems on SWE-bench Verified, revealing the capabilities and future directions of current agent systems across three dimensions: overall fixing effectiveness, fault localization accuracy, and the utility of bug reproduction.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Automated Bug Fixing"
  - "SWE-bench"
  - "Fault Localization"
  - "Bug Reproduction"
date: 2026-05-08
content_hash: e5821fde1dc0e876
---

# An Empirical Study on LLM-based Agents for Automated Bug Fixing

**Conference**: ACL 2025  
**arXiv**: [2411.10213](https://arxiv.org/abs/2411.10213)  
**Code**: [github](https://github.com/ResearchOpenRepos/bug_fixing_agent_empirical_study)  
**Area**: LLM Agent  
**Keywords**: Automated Bug Fixing, LLM Agent, SWE-bench, Fault Localization, Bug Reproduction

## TL;DR
This paper systematically analyzes the top six LLM-based bug-fixing systems on SWE-bench Verified, revealing the capabilities and future directions of current agent systems across three dimensions: overall fixing effectiveness, fault localization accuracy, and the utility of bug reproduction.

## Background & Motivation

**Background**: LLM-based agent systems have made significant progress in automated bug fixing, with the top-performing system on SWE-bench Verified resolving 64.6% of real GitHub issues. These systems automate bug fixing by interacting with the development environment, performing iterative verification, and modifying code. Representative systems include W&B Programmer, Blackbox AI Agent, and CodeStory Midwit Agent.

**Limitations of Prior Work**: Despite intense competition on leaderboards, there is a lack of systematic comparative analysis among these systems—specifically, regarding which types of bugs they succeed or fail at, the differences in fault localization strategies, and how bug reproduction affects the final fix. Variations in system designs (e.g., static vs. dynamic methods, single vs. multiple rollouts) lead to different strengths.

**Key Challenge**: A single resolution rate metric on leaderboards masks fine-grained capability differences among systems. While 96 cases (19.2%) cannot be resolved by any system and 181 cases (36.2%) are resolved by all systems, the disparate performance in the middle ground is key to understanding system capabilities.

**Goal**: Systematically analyze the differences in fixing capability, fault localization accuracy, and bug reproduction effectiveness among top LLM bug-fixing agents, summarize current bottlenecks, and suggest future research directions.

**Key Insight**: Three complementary Research Questions—RQ1 analyzes "what can and cannot be fixed", RQ2 analyzes "whether the correct locations can be identified", and RQ3 analyzes "whether the bugs can be correctly reproduced".

**Core Idea**: Reveal the capability spectrum and room for improvement in LLM bug-fixing agents through a fine-grained comparative analysis of six major systems.

## Method

### Overall Architecture
Choose the top six systems from the SWE-bench Verified leaderboard: W&B Programmer, Blackbox AI Agent, CodeStory Midwit Agent, Learn-by-interact, Devlo, and Emergent E1. Analysis is conducted under three research questions. RQ1 compares the characteristics of solvable and unsolvable cases through set-theoretic analysis. RQ2 extracts file-level and code symbol-level localization information from the submitted patches, comparing them with the golden patch to calculate localization accuracy. RQ3 implements a custom RepoFixer Agent that covers the complete bug-fixing workflow to investigate the role of bug reproduction.

### Key Designs

1. **Issue Quality Assessment**:

    - **Function**: Quantify the relationship between issue description quality and fixing success rate.
    - **Mechanism**: Use DeepSeek-R1 to score 500 issues on SWE-bench Verified across five dimensions: file-level localization (0-3 points, from non-existent to stack trace), symbol-level localization (0-3 points), line-level localization (0-3 points), reproducible example quality (0-3 points), and solution hints (-1 to 3 points, including misleading options). Detailed scoring criteria are designed for each dimension, and the reliability of DeepSeek-R1's scoring is confirmed through cross-validation with deterministic regex rules.
    - **Design Motivation**: Understand how the quality of issue descriptions affects agent fixing success rate and provide data support for best practices in issue reporting.

2. **Multi-Granularity FL Analysis**:

    - **Function**: Evaluate the localization accuracy of various systems at the file level and code symbol level.
    - **Mechanism**: Extract the list of modified files and code symbols (classes, functions, methods, top-level code) from the golden patch, and compare them with the same information extracted from the patches submitted by each system. Two evaluation criteria are employed: (1) "At least one hit"—the patch covers some realistic buggy file/symbol; (2) "Complete coverage"—the patch covers all buggy files/symbols. Compute precision, recall, and F1. Line-level analysis is omitted since a single line does not represent a complete functional block.
    - **Design Motivation**: Localization accuracy is a prerequisite for successful fixing, but prior research has not systematically compared the localization capabilities of different agents.

3. **RepoFixer—Bug Reproduction Analysis Agent**:

    - **Function**: Research the impact of bug reproduction on fixing effectiveness.
    - **Mechanism**: Since reproduction scripts of existing systems are difficult to extract from closed trajectories, the authors implement a dual-agent system: the *Searcher* handles fault localization (searching and reading code), while the *Fixer* manages the iterative loop of generating reproduction scripts $\rightarrow$ verifying reproduction $\rightarrow$ generating patches $\rightarrow$ verifying patches. Claude 3.5 Sonnet is used as the base model. The reproduction quality is judged by comparing output changes of the reproduction script before and after applying the golden patch; scripts that yield output differences are considered "relevant reproduction scripts".
    - **Design Motivation**: Bug reproduction is a critical step in practical debugging, but its quantitative contribution to automated fixing has lacked analysis.

### Loss & Training
This paper is an empirical analysis study and does not involve model training. RepoFixer utilizes the standard API of Claude 3.5 Sonnet, combined with the official bash and `str_replace_editor` toolkits.

## Key Experimental Results

### Main Results

| System | Resolved (/500) | Resolution Rate | Solved Uniquely | Unresolved but Solved by Others |
|------|------------|--------|----------|-------------------|
| W&B Programmer | 323 | 64.6% | 8 | 0 |
| Blackbox AI Agent | 314 | 62.8% | 5 | 9 |
| Midwit Agent | 311 | 62.2% | 4 | 13 |
| Learn-by-interact | 301 | 60.2% | 12 | 30 |
| Devlo | 291 | 58.2% | 3 | 17 |
| Emergent E1 | 286 | 57.2% | 4 | 22 |
| Resolved by any system | 404 | 80.8% | - | - |
| Resolved by all systems | 181 | 36.2% | - | - |

### Issue Quality and Fixing Performance

| Metric | All Solvable (181 cases) | All Unsolvable (96 cases) | Ratio |
|------|---------------|----------------|------|
| Average Total Score | 1.359 | 1.087 | 125% |
| Solution Hints | 1.276 | 0.656 | 195% |
| Symbol-Level Localization | 1.177 | 0.906 | 130% |
| File-Level Localization | 1.326 | 1.052 | 126% |
| Reproducible Example | 0.994 | 0.813 | 122% |

### Fault Localization (RQ2)

| System | File-Level Hit ($\ge 1$) | File-Level F1 | Symbol-Level Hit ($\ge 1$) | Symbol-Level F1 |
|------|-------------|---------|-------------|---------|
| W&B Programmer | 448 | Highest | 396 | Highest |
| Blackbox AI Agent | 416 | Second | 371 | Second |
| Midwit Agent | 391 | Fourth | 355 | Third (Highest Precision) |
| Learn-by-interact | 342 | Lowest | 308 | Lowest |

### Key Findings
- **Solution hints in the issue description have the most significant impact**: The discrepancy in solution hint scores between solvable and unsolvable cases is 195%, representing the most significant difference across all dimensions. This indicates that hinting at the fixing direction in the issue description substantially boosts the agent's success rate.
- **Learn-by-interact has the worst localization but decent fixing performance**: This is the most counter-intuitive finding—Learn-by-interact ranks last in both file-level and symbol-level localization, yet finishes fourth in total resolved issues. In-depth analysis shows that once it successfully localizes (hitting a buggy file/symbol), its subsequent fixing success rate reaches 82.2% / 85.1%, far exceeding W&B Programmer's 69.6% / 72.7%. This suggests that its experience-based strategy is exceptionally powerful at fixing when the target location is known.
- **Code symbol-level localization is more critical than file-level**: The accuracy of symbol-level localization correlates more strongly with the final fixing success rate, and differences among various systems are more pronounced at the symbol level. This suggests that fine-grained localization is where the greatest room for improvement lies.
- **76.8% of reproduction scripts are "relevant"**: Among the 500 reproduction scripts generated by RepoFixer, 384 produced output differences under the golden patch. For issues that contained clear reproduction examples in their description, the proportion of relevant reproduction scripts was as high as 80.4%.

## Highlights & Insights
- Relying on historical interaction experience, Learn-by-interact lags behind in localization but excels in the fixing stage. This reveals the counter-intuitive phenomenon that "good localization $\neq$ good fixing," and implies that experience-driven approaches could serve as strong complements to traditional agent workflows.
- The five-dimensional issue quality assessment framework can directly guide developers to write better bug reports and also serve as an objective function for automated issue enhancement.

## Limitations & Future Work
- SWE-bench Verified only contains Python repositories; bug-fixing patterns in other programming languages might differ.
- All analyzed systems are closed-source (except Learn-by-interact), preventing control over base models and decoding configurations. This limits the fairness of system comparison.
- The design of RepoFixer is relatively simple (lacking advanced strategies like resampling or self-reflection), resulting in a 51% resolution rate, which is lower than that of top systems.
- Future directions should: (1) enhance the reasoning ability of LLMs to distinguish between root causes and symptoms in issues; (2) design patch integrity validation mechanisms; (3) leverage diverse workflows to generate and vote on patch candidates; and (4) improve fine-grained (symbol-level) fault localization capabilities.

## Related Work & Insights
- **vs Agentless (Xia et al., 2024)**: Agentless employs a static, step-by-step reasoning approach, whereas this paper analyzes the complementarity of dynamic agent methods.
- **vs SWE-agent (Yang et al., 2024)**: SWE-agent designs an ACI interface to allow agents to interact with the environment, while this study reveals that pure interaction capability does not guarantee fixing effectiveness.
- **vs AutoCodeRover**: While AutoCodeRover provides specialized APIs to help agents localize, this study finds that designing fine-grained localization APIs is key to improving resolution rates.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic comparative analysis of top LLM bug-fixing agents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid experimental design across three RQs, six systems, and multiple analytical dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear organization and deep insights.
- Value: ⭐⭐⭐⭐⭐ Holds highly significant guidance for research on automated bug fixing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Systematic Study of Behavioral Cloning for Scientific Data Annotation](../../ICML2026/llm_agent/a_systematic_study_of_behavioral_cloning_for_scientific_data_annotation.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[AAAI 2026\] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](../../AAAI2026/llm_agent/promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ICLR 2026\] TaskCraft: Automated Generation of Agentic Tasks](../../ICLR2026/llm_agent/taskcraft_automated_generation_of_agentic_tasks.md)

</div>

<!-- RELATED:END -->
