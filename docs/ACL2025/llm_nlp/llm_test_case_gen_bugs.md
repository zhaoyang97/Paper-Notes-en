---
title: >-
  [Paper Note] LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs
description: >-
  [ACL 2025][LLM (Other)][test case generation] This paper proposes TrickCatcher, which utilizes LLMs to generate program variants and test input generators, combined with a diversity-driven differential testing mechanism to detect "plausible programs" that pass existing test suites but still contain hidden bugs (tricky bugs). TrickCatcher achieves SOTA gains of 1.80× / 2.65× / 1.66× in Recall, Precision, and F1 score, respectively.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "test case generation"
  - "bug detection"
  - "plausible programs"
  - "differential testing"
  - "LLM"
date: 2026-05-08
content_hash: 5a360b9fcbfe87d4
---

# LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs

**Conference**: ACL 2025  
**arXiv**: [2404.10304](https://arxiv.org/abs/2404.10304)  
**Code**: [https://github.com/RinCloud/TrickCatcher](https://github.com/RinCloud/TrickCatcher)  
**Area**: LLM NLP  
**Keywords**: test case generation, bug detection, plausible programs, differential testing, LLM

## TL;DR
This paper proposes TrickCatcher, which utilizes LLMs to generate program variants and test input generators, combined with a diversity-driven differential testing mechanism to detect "plausible programs" that pass existing test suites but still contain hidden bugs (tricky bugs). TrickCatcher achieves SOTA gains of 1.80× / 2.65× / 1.66× in Recall, Precision, and F1 score, respectively.

## Background & Motivation
1. **Background**: Software testing is the primary means of verifying program correctness. Programs that pass all existing test cases are referred to as "plausible programs", but plausible does not equal correct—these programs may contain hidden bugs (tricky bugs) in logical corner cases. Existing testing methods (such as EvoSuite and KLEE) focus primarily on coverage rather than bug detection.
2. **Limitations of Prior Work**: Tricky bugs are highly prevalent—one study identified 3,440 such bugs on online judge platforms. Existing LLM-based testing approaches (such as ChatTester and TestPilot) mainly focus on improving coverage. Meanwhile, the state-of-the-art bug detection method, Differential Prompting (DP), exhibits limited performance on plausible programs because it generates inputs directly from specifications (resulting in a 40.10% invalidity rate) and uses majority voting as the oracle (which is easily misled by the program under test, or PUT).
3. **Key Challenge**: While LLMs possess the capability to comprehend natural language specifications, directly generating program variants yields low correctness (especially for complex tasks). Direct generation of test inputs also suffers from poor constraint satisfaction. Furthermore, traditional majority voting fails in the context of plausible programs because variants are prone to inheriting the bugs of the PUT.
4. **Goal**: (1) How to generate high-quality program variants? (2) How to ensure the validity of generated test inputs? (3) How to correctly construct the test oracle when variants may inherit bugs?
5. **Key Insight**: Use the PUT itself as a reference for generating program variants (rather than relying solely on specifications), prompt the LLM to generate input generators instead of direct inputs, and replace majority voting with a diversity-driven strategy.
6. **Core Idea**: Solve the three weaknesses of LLMs (low variant correctness, difficulty in satisfying input constraints, and unreliable oracles) through PUT-guidance, generator-based indirect generation, and diversity-driven testing strategies, respectively.

## Method

### Overall Architecture
The inputs to TrickCatcher are the program specification, the program under test (PUT), and the existing test suite, while the output is a set of bug-identifying test cases. The workflow consists of three steps: (1) PUT-guided program variant generation; (2) Generator-based test input generation; (3) Diversity-driven differential testing. LLMs are used for generation in the first two steps, whereas the third step is a deterministic output comparison algorithm.

### Key Designs

1. **PUT-guided program variant generation**:

    - **Function**: Generate highly correct program variants for differential testing.
    - **Mechanism**: Provide both the PUT and the program specification to the LLM, prompting it to analyze whether the PUT contains bugs and generate patched program variants if potential bugs are detected. Subsequently, the existing test suite is used to discard variants that fail. The key distinction lies in modifying the PUT as a base rather than implementing it from scratch based solely on the specification. Only variants that pass the existing tests are retained (exploiting the information from the plausible program's existing test suite).
    - **Design Motivation**: Since the PUT is already correct on most inputs, modifying it is more likely to yield correct variants than implementing them from scratch from the specification. Filtering via the existing test suite further elevates the quality of the variants.

2. **Generator-based test input generation**:

    - **Function**: Generate valid test inputs that satisfy constraint conditions.
    - **Mechanism**: Instead of directly querying the LLM for test inputs, the LLM is prompted to write a Python input generator script based on the constraints in the specification, which is then executed to batch-generate inputs. The LLM learns to utilize the CYaRon library (a competitive programming data generation library) via few-shot examples to handle complex constraints (such as a "monotonically increasing square matrix"). This decouples logical reasoning (understanding constraints) from input generation (code execution).
    - **Design Motivation**: LLM reasoning capabilities are limited, resulting in a 40.10% invalidity rate when directly generating inputs under complex constraints. The generator-based approach relies on execution code to guarantee constraint satisfaction, substantially boosting validity (TrickCatcher generated zero invalid inputs in the experiments).

3. **Diversity-driven differential testing**:

    - **Function**: Correctly construct the test oracle when variants might inherit PUT bugs.
    - **Mechanism**: Traditional differential testing uses majority voting to determine the oracle—if most variants output X, X is deemed correct. TrickCatcher **does the opposite**: if the output of a variant differs from that of the PUT, the variant's output is treated as the oracle. If multiple variants give outputs different from the PUT, the most frequent one is chosen. If all variants produce the same output as the PUT, that input is skipped. The intuition is that LLMs tend to replicate PUT bugs when generating variants guided by the PUT; thus, a consensus between major variants and the PUT is likely incorrect.
    - **Design Motivation**: Experiments demonstrate that many variants indeed inherit identical bugs from the PUT, causing majority voting to favor incorrect answers. Trusting variants that "differ from the PUT" is more likely to capture the correct version where bugs are successfully fixed.

### Loss & Training
The gpt-3.5-turbo-0125 model is used as the LLM backbone, chosen for its balance between performance and cost. The number of program variants $k$ is configurable ($2-10$); using more variants increases recall but might reduce precision.

## Key Experimental Results

### Main Results

| Method | TrickyBugs C++ F1 | TrickyBugs Python F1 | EvalPlus F1 |
|------|-------------------|---------------------|-------------|
| DirectChat | 4.27 | 5.29 | 2.12 |
| APR | 22.30 | 15.96 | 45.28 |
| DPP (best $k$) | 24.95 ($k=8$) | 36.20 ($k=2$) | 35.76 ($k=10$) |
| **TrickCatcher** (best $k$) | **41.31** ($k=10$) | **42.35** ($k=8$) | **51.34** ($k=10$) |
| Gain | 1.66× | 1.17× | 1.44× |

### Ablation Study (TrickyBugs C++, $k=10$)

| Program Generation (PG) | Input Generation (IG) | Differential Testing (DT) | F1 |
|-------------|-----------|------------|-----|
| Basic | Basic | Basic | 24.71 |
| Filtered | Basic | Ours | 31.86 |
| Ours | Basic | Ours | 31.56 |
| Filtered | Ours | Ours | **38.06** |
| **Ours** | **Ours** | **Ours** | **41.31** |

### Key Findings
- TrickCatcher comprehensively outperforms all baselines on three datasets, achieving F1 scores of 41.31%, 42.35%, and 51.34%, respectively, while the best baseline DPP achieves only 24.95%, 36.20%, and 35.76%.
- The number of false positives of TrickCatcher on correct programs is **up to 16 times** fewer than DPP, and the generator-based method yields zero invalid inputs.
- Ablation studies demonstrate that all three components contribute to performance: diversity-driven differential testing brings the largest improvement (from 24.71 to 31.86), followed by generator-based input generation (31.86 to 38.06), and PUT-guided variant generation adds further gains (38.06 to 41.31).
- As the number of variants $k$ increases, the F1 of TrickCatcher continuously rises ($k=2$: 37.23 $\rightarrow$ $k=10$: 41.31), whereas the F1 of DPP peaks and then declines (optimal at $k=8$). This indicates that TrickCatcher leverages multi-variant information more effectively.
- The higher the task difficulty, the greater TrickCatcher's relative advantage—achieving an F1 improvement of over 80% on hard programming tasks.

## Highlights & Insights
- Diversity-driven differential testing is a counter-intuitive yet effective design. The "trust the minority" strategy is more reliable than majority voting in scenarios where variants might be corrupted by the PUT, a concept that can generalize to other settings requiring an oracle.
- The generator-based input generation is highly practical—letting the LLM do what it excels at (understanding constraints and writing code) while letting the execution environment do what it excels at (executing code to ensure constraint satisfaction). Each component performs its designated role.
- TrickCatcher's F1 score on AI-generated programs (EvalPlus, 51.34%) is higher than on human-written programs (41-42%), which may be because AI-generated bug patterns are easier to "repair" by the same LLM.

## Limitations & Future Work
- The best F1 score is still below 52%, implying that nearly half of the tricky bugs remain undetected.
- Reliance on gpt-3.5-turbo; though more powerful models (like GPT-4) may yield further improvement, they entail higher costs.
- Evaluation is restricted to competitive programming tasks; its generalizability to real-world software engineering projects remains unknown.
- The generator-based approach depends on the CYaRon library; input formats in non-competitive programming settings may require alternative library support.
- Multi-turn interaction is not considered—allowing the LLM to iteratively optimize variants based on differential testing results could further improve recall.
- Utilizes only gpt-3.5-turbo, leaving the performance of open-source models (such as CodeLlama) on this task unexplored.

## Related Work & Insights
- **vs Differential Prompting (Li et al., 2023)**: DP generates variants from specifications and uses majority voting, whereas TrickCatcher generates variants from PUT + specifications combined with a diversity-driven strategy, which is superior in plausible program scenarios.
- **vs EvoSuite/KLEE**: Traditional methods pursue branch coverage, whereas TrickCatcher is specifically tailored for bug detection, presenting a different methodology.
- **vs ChatTester/TestPilot**: These approaches generate unit tests to improve coverage, whereas TrickCatcher generates differential tests to detect subtle bugs, addressing a different objective.

## Rating
- Overall Evaluation: A highly practical LLM-assisted testing method, with all three components independently reusable in other contexts.
- Novelty: ⭐⭐⭐⭐ The combined design of the three components is innovative, especially the counter-intuitive diversity-driven differential testing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted on two datasets with multiple baselines, detailed ablation, and parameter sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a natural logic flow in the three-step methodology.
- Value: ⭐⭐⭐⭐ Directly applicable and valuable to the field of LLM-assisted software testing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)
- [\[ACL 2025\] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->
