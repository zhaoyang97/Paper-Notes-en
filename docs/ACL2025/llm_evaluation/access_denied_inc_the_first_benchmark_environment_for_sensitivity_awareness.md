---
title: >-
  [Paper Note] Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness
description: >-
  [ACL 2025][LLM Evaluation][Sensitivity Awareness] This work formally defines the concept of LLM "Sensitivity Awareness" (SA) for the first time—evaluating whether an LLM can decide whether to provide information based on Role-Based Access Control (RBAC) rules. The authors construct an automated evaluation benchmark, Access Denied Inc, and find that even with highly structured data and minimalist rules, the best-performing model, Grok-2, still exhibits a leak rate of 18.28% ac…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Sensitivity Awareness"
  - "RBAC"
  - "Enterprise Data Management"
  - "LLM Benchmarking"
  - "Information Leakage Prevention"
date: 2026-05-08
content_hash: 5a81fc72f2a7823d
---

# Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness

**Conference**: ACL 2025  
**arXiv**: [2506.00964](https://arxiv.org/abs/2506.00964)  
**Code**: [GitHub](https://github.com/DrenFazlija/AccessDeniedInc)  
**Area**: LLM Evaluation  
**Keywords**: Sensitivity Awareness, RBAC, Enterprise Data Management, LLM Benchmarking, Information Leakage Prevention

## TL;DR

This work formally defines the concept of LLM "Sensitivity Awareness" (SA) for the first time—evaluating whether an LLM can decide whether to provide information based on Role-Based Access Control (RBAC) rules. The authors construct an automated evaluation benchmark, Access Denied Inc, and find that even with highly structured data and minimalist rules, the best-performing model, Grok-2, still exhibits a leak rate of 18.28% across 7 mainstream LLMs.

## Background & Motivation

**Background**: LLM-driven AI assistants are entering enterprise data management scenarios (such as SAP Joule), where employees can query information like salaries and departmental affiliations through natural language queries. Such systems require LLMs not only to retrieve data but also to decide whether to provide it based on the user's role.

**Limitations of Prior Work**: Simple document-level filtering is infeasible. Enterprise documents often mix sensitive and non-sensitive information (e.g., in HR files, public job descriptions coexist with restricted salary information). Coarse-grained filtering either over-blocks content or exposes restricted data. More seriously, even if some information is hidden, the LLM in a RAG system might still infer restricted information from the contextual clues of multiple retrieved passages.

**Key Challenge**: Existing safety/privacy benchmarks (such as jailbreak attacks and harmful content generation) focus on "generating content that should not be generated." In contrast, the core requirement of enterprise scenarios is to "selectively provide existing data based on user permissions"—a blank field that has not yet been systematically evaluated. SudoLM only supports coarse-grained binary (public/private) authentication, which is far from sufficient.

**Key Insight**: Building the first standardized sensitivity awareness evaluation framework. Core Idea: Establishing a complete pipeline consisting of a simulated corporate database + fine-grained field-level permissions + automated questionnaires + a 99.9% automated scoring system to systematically evaluate the capability of out-of-the-box LLMs to adhere to RBAC rules.

## Method

### Overall Architecture

Access Denied Inc is a complete three-stage evaluation pipeline: (1) generating a simulated corporate employee database (45K+ employees, 12 attribute fields) from the Adult dataset; (2) automatically generating multi-dimensional test questionnaires (3500 questions each, covering 6 features and 2 adversarial scenarios) based on configurable parameters; (3) a semi-automated scoring system evaluating LLM responses, achieving an automated coverage rate of up to 99.9%.

### Key Designs

1. **SA Formal Definition and Four-class Session Classification**: Each interaction between the LLM and the user is classified into four mutually exclusive categories—$S_{\text{correct}}$ (correctly providing data or correctly refusing), $S_{\text{leak}}$ (leaking restricted data to unauthorized users, the most dangerous), $S_{\text{refusal}}$ (falsely refusing authorized users, affecting usability), and $S_{\text{error}}$ (outputting hallucinated data or formatting violations). This classification explicitly models the trade-off between safety and usability—leakage and false refusal are the two core dimensions of SA evaluation, rather than a simple binary judgment of correct or incorrect.

2. **Simulated Enterprise Database Generation**: The Adult tabular dataset is reused. After removing missing values, a unique ID and a random name (sampled from a repository of 20K popular names to deliberately break the correlation between names and gender/race to eliminate bias) are assigned. The binary salary is modified into a continuous numerical value following $N(80000, 15000)$ (excluding extreme values <35K and >200K). Department (based on a predefined organizational chart) and supervisor (randomly assigned based on department hierarchy) attributes are added. The final database contains 45,233 employees, each with 12 attributes. Design Motivation: Tabular data is the simplest form of enterprise data—if LLMs cannot enforce access control on highly structured data, they will only perform worse on unstructured documents.

3. **Automated Questionnaire and Scoring System**: Questionnaires are automatically generated based on configuration parameters (Perspective: first/third person; Malevolence/Intent: authorized/unauthorized; Target feature: 6 fields including salary/department/age). The system prompt contains the target user data + 5 random employee data (simulating RAG top-k retrieval) and provides concise access rules (users can query all of their own information; HR department employees and direct supervisors can query their subordinates' information). The model is required to output concise results (data values or the fixed refusal phrase "I cannot disclose that information.") inside double curly braces {{}}, with the space outside the braces used for CoT reasoning. Scoring is based on exact string matching, achieving an automated coverage rate from 99.9% (GPT-4o) to 92.9% (Llama 3.2).

### Loss & Training

This paper evaluates the out-of-the-box capabilities of LLMs and does not involve training. System prompts serve as the sole alignment mechanism, containing a unified basic prompt template. Closed-source models (GPT-4o/mini, Grok-2) are accessed via API calls, while open-source models (Llama 3.2 3B, Llama 3.3 70B, R1-Qwen 32B, Phi-4 14B) are accessed via HuggingFace or OpenRouter APIs. All models use default parameters, except for R1-Qwen (where temperature=0.6).

## Key Experimental Results

### Main Results

| Model | Correct ↑ | Error ↓ | Wrong ↓ | Benign ↑ | Malicious ↑ |
|------|-----------|---------|---------|----------|-------------|
| **Grok-2** | **80.50%** | **0.22%** | 18.28% | **95.52%** | **65.48%** |
| GPT-4o | 70.72% | 3.61% | 25.63% | 83.88% | 57.56% |
| R1-Qwen (32B) | 64.56% | 2.94% | 28.09% | 94.59% | 34.53% |
| Llama 3.3 (70B) | 60.81% | 0.16% | 38.32% | 97.54% | 24.07% |
| Phi-4 (14B) | 59.42% | 6.81% | 26.93% | 84.26% | 34.59% |
| GPT-4o mini | 45.98% | 35.88% | 18.08% | 57.33% | 34.62% |
| Llama 3.2 (3B) | 29.08% | 13.68% | 50.17% | 48.09% | 10.07% |

### Ablation Study

| Scenario | Grok-2 | GPT-4o | Llama 3.3 | R1-Qwen | GPT-4o mini |
|------|--------|--------|-----------|---------|-------------|
| Supervisor (Supervisor query subordinate, legitimate) | 80.66% | 59.33% | 94.40% | 90.00% | 32.93% |
| Lying (Identity forgery, adversarial) | 49.86% | 44.53% | 45.33% | **13.60%** | **50.66%** |

### Key Findings

- **All models perform poorly on malicious queries**: Even the best, Grok-2, correctly rejects only 65.48% of malicious requests, meaning about 1/3 of unauthorized requests lead to leakage.
- **Model failure modes differ significantly**: The main issue for GPT-4o mini is hallucinations and formatting errors (Error 35.88%), whereas for open-source models, the primary issue is leakage (high Wrong but low Error)—Llama 3.3 almost always outputs real data regardless of permissions.
- **The Supervisor scenario exposes differences in understanding hierarchical permissions**: Closed-source models perform poorly on legitimate supervisor queries (GPT-4o mini is only 32.93%), showing that these models do not understand hierarchical permission relationships.
- **Smaller models are more robust in the Lying scenario**: GPT-4o mini (50.66%) surprisingly outperforms Grok-2 (49.86%) in identity forgery scenarios, while R1-Qwen is extremely vulnerable (only 13.60%).
- **Conclusion: Out-of-the-box LLMs currently cannot be reliably used for enterprise sensitive data management.**

## Highlights & Insights

- **Defining the SA problem and providing an evaluation framework for the first time**: Abstracting enterprise access control requirements into a formal definition, filling a gap in LLM safety evaluation.
- **Ingenious evaluation design**: The concise output format (data values or fixed refusal phrase) makes string-matching-based scoring feasible, and the 99.9% automated rate significantly reduces evaluation costs.
- **The experimental insight of "the simpler it is, the more it exposes problems"**: Deliberately using the simplest tabular data and the simplest rule set. If models fail here, success in complex scenarios is even more impossible.
- **Clinical value of failure mode classification**: Distinguishing between three failure modes (leak/refusal/error) is critical for risk assessment in actual deployment.

## Limitations & Future Work

- Only tabular data is used; unstructured documents (such as PDF reports mixing sensitive information) are not tested.
- Access rules are extremely simple (only 2 rules); RBAC policies in real enterprises are far more complex.
- Alignment fine-tuning specifically targetting SA (e.g., RLHF adding permission-aware reward signals) has not been explored.
- The Lying scenario only uses basic adversarial methods without introducing low-resource language attacks or automated prompt injection.
- The end-to-end SA performance of the RAG system has not been evaluated (data is currently put directly into the prompt).

## Related Work & Insights

- **vs SudoLM (Liu et al. 2024)**: SudoLM uses a static sudo key to achieve binary public/private partitioning, supporting only a single authentication scenario. In contrast, Access Denied Inc supports multi-role, fine-grained, field-level permissions.
- **vs Safety Benchmarks (JailbreakBench, HarmBench)**: These focus on defense against generating harmful content, whereas SA focuses on "providing the right data to the right people."
- **vs Zeng et al. (2024) Risk Taxonomy**: Among the 314 risk categories, the most relevant are confidentiality and privacy violations, but the core features of SA (access permission enforcement and selective information distribution) are not fully covered.
- **Insight**: Deploying LLMs in enterprise scenarios requires moving beyond the safety dimension of "harmfulness"—permission awareness is a highly demanded but severely neglected capability.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty ⭐⭐⭐⭐⭐: Defines the SA problem and constructs the evaluation framework for the first time; clear definition, critical problem.
- Experimental Thoroughness ⭐⭐⭐⭐: 7 models + 6 features + 2 special scenarios, but only in tabular data format.
- Writing Quality ⭐⭐⭐⭐: Complete formal definitions, in-depth experimental analysis, insightful failure mode classification.
- Value ⭐⭐⭐⭐: The framework is reusable and exposes real risks, but lacks solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness](../../ICLR2026/llm_evaluation/culture_in_a_frame_c3b_as_a_comic-based_benchmark_for_multimodal_culturally_awar.md)
- [\[ICCV 2025\] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](../../ICCV2025/llm_evaluation/spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](../../ACL2026/llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Movie101v2: Improved Movie Narration Benchmark](movie101v2_improved_movie_narration_benchmark.md)

</div>

<!-- RELATED:END -->
