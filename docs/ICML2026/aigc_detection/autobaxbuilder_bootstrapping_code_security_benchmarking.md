---
title: >-
  [Paper Note] AutoBaxBuilder: Bootstrapping Code Security Benchmarking
description: >-
  [ICML2026][AIGC Detection][Code security evaluation] AUTOBAXBUILDER utilizes an LLM agent pipeline to automatically generate Web backend security evaluation scenarios, functional tests…
tags:
  - "ICML2026"
  - "AIGC Detection"
  - "Code security evaluation"
  - "LLM-generated code"
  - "Automated benchmark construction"
  - "End-to-end security testing"
  - "BAXBENCH"
date: 2026-05-08
content_hash: 35ad7da03b3aa5ee
---

# AutoBaxBuilder: Bootstrapping Code Security Benchmarking

**Conference**: ICML2026  
**arXiv**: [2512.21132](https://arxiv.org/abs/2512.21132)  
**Code**: https://github.com/eth-sri/autobaxbuilder  
**Area**: Code Intelligence / Security Evaluation  
**Keywords**: Code security evaluation, LLM-generated code, Automated benchmark construction, End-to-end security testing, BAXBENCH  

## TL;DR
AUTOBAXBUILDER utilizes an LLM agent pipeline to automatically generate Web backend security evaluation scenarios, functional tests, and end-to-end security tests. It reduces the manual cost of constructing BAXBENCH-style tasks by approximately 12x and establishes AUTOBAXBENCH with 40 new scenarios to evaluate the gap between correctness and security in modern code models.

## Background & Motivation
**Background**: LLMs are widely used in software engineering, especially for generating Web backends, API services, and application logic. Traditional code generation evaluations often focus on functional correctness, such as unit test pass rates. Security evaluation typically relies on static analyzers, manual review, or expert-written end-to-end attack tests. Benchmarks like BAXBENCH further require models to generate complete backends subjected to both functional and security testing.

**Limitations of Prior Work**: Manual construction of security benchmarks is expensive. A scenario requires not only natural language specifications and OpenAPI interfaces but also functional tests, reference implementations, reproducible security tests, and manual verification. As model capabilities improve, older benchmarks are prone to training data contamination or become too simple, yet continuously expanding high-quality security benchmarks requires significant expert time.

**Key Challenge**: While automated benchmark generation reduces costs and increases update frequency, the security tests themselves must be reliable. An automatically generated test that falsely reports security issues underestimates model capabilities, while missing vulnerabilities allows unsafe code to pass. Therefore, an automated pipeline must simultaneously address new scenario generation, functional consistency verification, and security test accuracy validation.

**Goal**: The authors aim to build a pipeline that generates BAXBENCH-style tasks from scratch. It should automatically propose backend service scenarios, generate functional tests, construct security tests, and filter erroneous tests through execution feedback and comparison with reference solutions. The final objective is to rapidly build publicly releasable, difficulty-controllable code security benchmarks covering multiple CWE categories.

**Key Insight**: The paper decomposes benchmark construction into a multi-stage agent workflow. An orchestrator model proposes scenarios, generates tests, analyzes potential vulnerabilities, and iteratively corrects them. Multiple solver models generate reference implementations, providing feedback for tests and security checks via execution logs. The pipeline continuously seeks "functionally correct but security-divergent" contrastive implementations to confirm that security tests do not merely attack a specific incidental implementation.

**Core Idea**: Use LLMs to generate candidate benchmarks, then calibrate them through layers of execution feedback, reference solution divergence, contrastive implementations, and manual spot-checking. This ensures that automatically generated security tests approach the reliability of expert-written tests while significantly reducing construction costs.

## Method
AUTOBAXBUILDER generates complete security evaluation instances rather than single code snippets. Each instance includes a backend service scenario, REST API specification, functional tests, and security tests. During evaluation, models generate a runnable implementation for the backend. The system places the implementation in an isolated container and runs functional and security tests via the REST interface. Metrics include $pass@1$ for functional correctness and $sec\_pass@1$ for implementations that are both functionally correct and secure.

### Overall Architecture
The pipeline consists of an orchestrator LLM and several solver LLMs. First, the orchestrator proposes a new backend service based on target difficulty, existing scenario names, and example CWEs, generating OpenAPI specifications and text descriptions while checking for novelty. Second, solver LLMs write reference implementations for the scenario, and the orchestrator generates functional tests based on specifications, refining either the tests or implementations through execution and log analysis. Third, the orchestrator analyzes potential security weaknesses in the scenario and reference implementation, generates security tests, and validates them using secure/insecure contrastive implementations to ensure they correctly distinguish between the two.

This process relies heavily on execution feedback. OpenAPI specifications are checked with YAML validators, test code is checked for Python compilation, and output formats are constrained by regex. Each refinement loop runs up to 5 times; if a security test cannot be stably validated, it is discarded. The paper also incorporates auxiliary functions like pseudo-random flags and monitoring for file systems, databases, and resources to programmatically judge end-to-end security results.

### Key Designs
1. **Scenario and Reference Implementation Bootstrapping**:
    - **Function**: Generate new Web backend security evaluation tasks from scratch and provide executable objects for subsequent test iterations.
    - **Mechanism**: The orchestrator LLM first proposes a service scenario with a clear attack surface, generating OpenAPI interfaces and text specifications; subsequently, multiple solver LLMs generate reference implementations. Scenario generation avoids existing BAXBENCH scenarios and previously generated ones to reduce duplication and contamination risks.
    - **Design Motivation**: Security benchmarks must expand continuously to avoid being memorized or trivialized by model growth. Making scenarios automatically generatable and using reference implementations from different models to expose specification ambiguities is the first step in reducing construction costs.

2. **Alternating Refinement of Functional Tests and Implementations**:
    - **Function**: Ensure functional tests truly represent the specification requirements rather than over-fitting with a single incorrect reference implementation.
    - **Mechanism**: The pipeline first has the orchestrator extract functional requirements and generate tests. If tests fail, the REFINESOLUTIONS stage shows only execution logs to the model without the test code to prevent direct catering. The REFINETESTS stage then synthesizes test code, implementation code, and logs to determine if the error lies in test logic, implementation, or specification ambiguity, adjusting accordingly.
    - **Design Motivation**: Automated benchmarks are most vulnerable to "test-implementation collusion," where a test is wrong but an implementation happens to pass. Alternating refinement and global consensus judgment minimize this drift, bringing tests closer to the actual specification.

3. **Contrastive Validation of Security Tests**:
    - **Function**: Ensure generated security tests can detect real vulnerabilities while avoiding misclassifying secure implementations as insecure.
    - **Mechanism**: The orchestrator identifies potential CWEs from the scenario and functionally correct reference implementations, then generates security tests for target vulnerabilities. REFINEEXPLOIT runs tests repeatedly against original reference implementations and modified contrastive versions: if an implementation is judged insecure, the pipeline attempts to construct a fixed version; if judged secure, it tries to introduce the target vulnerability. A security test is retained only if it behaves as expected across both positive and negative contrasts.
    - **Design Motivation**: The value of end-to-end security tests lies in being specific, reproducible, and having low false positives. Contrastive implementations allow the pipeline to verify that tests focus on the target security properties rather than framework details or unspecified behaviors.

### Loss & Training
This paper does not train new models but designs a benchmark generation and evaluation pipeline. The orchestrator primarily uses GPT-5, while reference implementations come from models like GPT-5, Claude 4 Sonnet, DeepSeek-R1, and Qwen3 Coder 480B. The final evaluation covers 14 code model families or versions, requiring implementations in 14 frameworks and 6 language environments.

Evaluation metrics follow BAXBENCH: $pass@1$ represents the ratio of implementations passing all functional tests, and $sec\_pass@1$ represents those passing both functional and security tests. While AUTOBAXBUILDER defaults to Python-FastAPI for reference implementations, framework ablations using JavaScript-Fastify and Go-Gin show that model rankings and CWE coverage remain stable.

## Key Experimental Results

### Main Results
The authors first validate that automatically generated tests give similar trends to expert tests on 28 BAXBENCH scenarios, then construct AUTOBAXBENCH with 40 new scenarios. The table below summarizes the scale, difficulty, and top model performance on AUTOBAXBENCH.

| Dataset | Scenarios | Avg. Endpoints | Avg. Spec Length | Avg. CWEs | Best $sec\_pass@1$ | Best $pass@1$ |
|--------|------|-----------|-------------|-----------|----------------|------------|
| BAXBENCH | 28 | 1.9 | 430 | 3.3 | 60% | 81% |
| AUTOBAXBENCH Easy | 10 | 1.0 | 587 | 1.6 | 36% | 81% |
| AUTOBAXBENCH Medium | 20 | 3.0 | 1006 | 2.7 | 40% | 84% |
| AUTOBAXBENCH Hard | 10 | 4.7 | 1516 | 3.5 | 25% | 83% |
| AUTOBAXBENCH Overall | 40 | 2.93 | 1029 | 2.6 | 36% | 83% |

### Ablation Study
The paper provides several sets of stability analyses: consistency between automated and expert tests, reference framework variations, agentic harnesses, generation model selection, and manual evaluation. Key ablation results are summarized below.

| Configuration | Key Metric | Description |
|------|---------|------|
| AUTOBAXBUILDER vs BAXBENCH Functional Tests | 80.9% Solution Consistency | Functional trends are highly similar; identified 2 incorrect tests and 2 ambiguous specs in BAXBENCH |
| AUTOBAXBUILDER vs BAXBENCH Security Tests | +512 insecure solutions identified | Automated tests are stricter, covering more CWEs and attack variants among functionally correct solutions |
| Manual Audit of 71 Auto-Security Tests | Only 1 unreliable test | High overall quality of automatically generated security tests |
| Reference Framework Ablation | Stable ranking & CWE coverage | Python-FastAPI, JavaScript-Fastify, and Go-Gin provide similar trends |
| Generation Model Ablation | $pass@1 \rho=0.93, sec\_pass@1 \rho=0.91$ | Alternative benchmarks generated using disjoint model sets maintain strong ranking correlation |
| Agentic Harness | Far from 100% secure pass | GPT-5 improves slightly, Claude 4.5 Sonnet is unstable; tool-augmentation does not bridge the security gap |

### Key Findings
- AUTOBAXBUILDER reproduces expert test model rankings on original BAXBENCH scenarios but with stricter security tests, indicating the pipeline extends security coverage rather than just copying functions.
- AUTOBAXBENCH features longer specifications and more endpoints than BAXBENCH, with controllable difficulty. On the Hard subset, the best model's $sec\_pass@1$ is only 25%, exposing significant weaknesses of modern code models in complex backend security.
- There is a large gap between $pass@1$ and $sec\_pass@1$. For Claude 4.5 Sonnet, while $pass@1$ is ~82.7%, $sec\_pass@1$ is only 36%, proving that "functional implementation" and "secure implementation" are distinct capabilities.
- Construction costs are mainly spent on refinement tokens for tests. Vulnerability analysis and strategy generation account for about 17% of the token budget. Running reference implementations and tests incurs almost no API costs.

## Highlights & Insights
- The paper industrializes the construction of security benchmarks. It requires scenarios, functional tests, security tests, and contrastive implementations to form a closed loop, which is critical for benchmark credibility.
- The result that "automated tests are stricter than experts" is noteworthy. Through multiple reference implementations and repetition, the pipeline may discover CWEs and variants missed in the initial expert version.
- Controllable difficulty is a practical highlight. By adjusting endpoint counts and scenario complexity, AUTOBAXBUILDER can generate harder tasks as model capabilities grow, preventing benchmark saturation.
- For code intelligence research, this paper serves as a reminder that evaluation cannot rely solely on $pass@1$. A Web backend passing functional tests can still have critical security flaws; $sec\_pass@1$ better reflects real-world deployment risks.

## Limitations & Future Work
- Automatically generated security tests are not flawless; manual audits found a few unreliable cases, particularly in resource exhaustion tests where specs were undefined. Thus, CWE-400 was excluded from the main experiment and analyzed separately.
- Reference implementations primarily use Python-FastAPI. While framework ablations show stable trends, cross-language and cross-framework nuances might still affect specific CWE coverage.
- AUTOBAXBUILDER depends on strong orchestrator models and extensive execution feedback. While relatively low-cost, it requires complex infrastructure (containers, API services, DB/FS checks, log parsing).
- Potential self-bias exists when LLMs generate benchmarks for other LLMs. While disjoint model sets and neutral calibration showed no systematic bias, this risk requires long-term tracking, especially if generators and subjects share the same training ecosystem.
- Future work could include stronger human review interfaces, coverage of more real-world frameworks/cloud services, and feeding generated tests back into development tools for training or constraining code generation models.

## Related Work & Insights
- **vs BAXBENCH**: BAXBENCH uses expert-written scenarios and tests, which are high quality but costly to scale. AUTOBAXBUILDER inherits the end-to-end backend evaluation philosophy while automating the generation process.
- **vs Static Analyzer Evaluation**: Static analyzers have reusable rules but are sensitive to language/framework versions and suffer from false positives/negatives. This work uses end-to-end tests focusing on observable insecure behaviors.
- **vs HumanEval-style Correctness**: Conventional benchmarks are mostly function-level functional tests that cannot cover deployment interfaces, databases, file systems, and security boundaries. AUTOBAXBENCH elevates tasks to complete application services.
- **vs Agentic Code Generation Harnesses**: Tool-augmentation allows models to write tests and fix code, but results show it does not push security pass rates near 100%. Thus, security capability must be evaluated separately rather than assuming it is solved by agentic workflows.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Utilizing LLM agents for automated end-to-end code security benchmarks with contrastive validation is a comprehensive and practical approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Includes reproduction of original benchmarks, 40 new scenarios, large-scale model evaluation, and multi-dimensional ablations (framework, model, manual audit).
- **Writing Quality**: ⭐⭐⭐⭐ The main narrative is clear and the appendix is informative, though the high volume of tables and figures requires the reader to toggle between the main text and appendix for full details.
- **Value**: ⭐⭐⭐⭐⭐ Directly valuable for code LLM security evaluation, benchmark anti-contamination, and difficulty scaling; serves as a paradigm for automated benchmark construction in other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](../../ICLR2026/aigc_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)
- [\[ICLR 2026\] CLARC: C/C++ Benchmark for Robust Code Search](../../ICLR2026/aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](../../NeurIPS2025/aigc_detection/synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[NeurIPS 2025\] QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code](../../NeurIPS2025/aigc_detection/qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)

</div>

<!-- RELATED:END -->
