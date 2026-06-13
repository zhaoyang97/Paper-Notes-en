---
title: >-
  [Paper Note] CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities
description: >-
  [ICML2026][Vulnerability Discovery] This paper constructs CyberGym-E2E—the first large-scale real-world AI Agent security benchmark covering the full lifecycle of "vulnerability discovery → PoC generation → patch generat…
tags:
  - "ICML2026"
  - "Vulnerability Discovery"
  - "PoC Generation"
  - "Patch Generation"
  - "Agent Evaluation"
  - "OSS-Fuzz"
date: 2026-05-08
content_hash: 75b3ab49aa389046
---

# CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities

**Conference**: ICML2026  
**arXiv**: [2606.04460](https://arxiv.org/abs/2606.04460)  
**Code**: The paper does not provide a public repository link in the main text  
**Area**: LLM Agent / Cybersecurity Evaluation / Benchmark  
**Keywords**: Vulnerability Discovery, PoC Generation, Patch Generation, Agent Evaluation, OSS-Fuzz

## TL;DR
This paper constructs CyberGym-E2E—the first large-scale real-world AI Agent security benchmark covering the full lifecycle of "vulnerability discovery → PoC generation → patch generation → functional regression testing" (920 vulnerabilities × 139 open-source projects). It minimizes manual costs through an agent-assisted + expert-finalized four-step pipeline. Evaluations show that while leading models reach 80%+ on patch-only tasks, the highest S3 success rate on end-to-end tasks is only 65.9% (GPT-5.4), revealing that vulnerability discovery, rather than patch generation, is the true bottleneck.

## Background & Motivation

**Background**: The capabilities of LLMs and Agents in code analysis and generation have made "autonomous discovery and fixing of vulnerabilities" possible. The industry has begun utilizing these capabilities as defensive tools; however, attackers are also using the same capabilities (Anthropic 2025 disclosed AI-orchestrated cyber espionage activities). Therefore, reliably quantifying "how much AI can actually achieve in end-to-end cyber defense" has become an urgent issue for both the Security and AI communities.

**Limitations of Prior Work**: Existing benchmarks suffer from four types of systematic flaws: (1) **Incomplete task scope**: They either only test vulnerability detection (PrimeVul, CyberGym) or only test secure code generation (SeCodePLT, SecRepoBench), artificially decoupling the highly interdependent "discovery/PoC/patch" steps; (2) **Unrealistic evaluation environments**: Most benchmarks provide agents with only a read-only code view, which is far from reality where agents run commands in an engineer's sandbox; (3) **Missing or unreliable functional regression testing**: SEC-bench lacks post-patch functional testing; SeCodePLT uses non-crashing fuzz inputs for approximate verification; AutoPatchBench compares function states via LLDB, which misjudges "formally different but equally correct" patches; (4) **Trade-off between scale and realism**: Hand-curated BountyBench has only 40 tasks, while synthetic datasets like SeCodePLT are large but unrealistic.

**Key Challenge**: To simultaneously achieve the four attributes of "end-to-end + realistic + large-scale + reproducible," the construction cost explodes. Historical vulnerabilities are scattered across years and cross-toolchains (many old OSS-Fuzz vulnerabilities depend on Ubuntu 16.04 / GLIBC < 2.28, which modern agents struggle to run), unit test coverage is difficult to guarantee, and expert review costs soar. Existing benchmarks either sacrifice scale or realism/end-to-end coverage.

**Goal**: The problem is split into three sub-problems: (1) How to automatically convert historical OSS-Fuzz vulnerability data into end-to-end tasks runnable by modern agent frameworks; (2) How to use agent assistance to generate credible functional regression tests and precisely focus human review on essential stages; (3) How to fairly evaluate different agent harnesses (Claude Code / Codex / Gemini CLI / OpenHands) × multiple frontier models under a uniform budget, separating the contributions of "model capability" and "harness design."

**Key Insight**: The authors found that ARVO has packaged OSS-Fuzz vulnerabilities into reproducible Docker images but lacks evaluation tasks and functional tests. By stringing together "identifying clean patches → reconstructing build environments → agent-assisted unit test discovery → expert final review" into a pipeline, end-to-end tasks can be produced in bulk. Agents are no longer just the subjects of evaluation but also the "cheap labor" for benchmark construction, focusing manual effort on verification rather than repetitive work.

**Core Idea**: Use an "agent-assisted construction + expert final review" four-stage pipeline to convert OSS-Fuzz vulnerability data into end-to-end cyber tasks, and employ a "two difficulty levels (patch-only / end-to-end) + four-stage verification (S1–S4)" protocol to evaluate frontier models and agent harnesses simultaneously.

## Method

### Overall Architecture
CyberGym-E2E consists of two components: the **Construction Pipeline** (converting OSS-Fuzz vulnerabilities into 920 evaluable tasks) and the **Evaluation Protocol** (performing four-stage verification of agents × models under a uniform budget).

The construction side follows four steps: (1) Identify clean patch commits; (2) Prepare vulnerable/patched build environments and verify consistent PoC behavior; (3) Let code agents identify, build, and run unit tests within Docker; (4) Expert review of test coverage and scripts. Each task delivers: vulnerable build environment, build scripts, test-build/run scripts, ground-truth PoC (+ crash log), and ground-truth patch; test-related files are immutable to the agent during evaluation.

The evaluation side features two difficulties: **Patch-only** provides the agent with ground-truth PoC + crash log for root cause analysis and patching; **End-to-end** provides only the codebase + build environment, requiring the agent to discover vulnerabilities, construct a PoC, and then write a patch. Verification is split into four stages: S1 = Agent PoC triggers crash; S2 = Patch eliminates the agent's own PoC crash; S3 = Existing functional tests still pass after patching; S4 = Patch simultaneously eliminates the ground-truth PoC crash (distinguishing between "fixing the same vulnerability" and "fixing a neighboring one").

### Key Designs

1. **Four-Stage Construction Pipeline with Agent Assistance + Expert Review**:
    - **Function**: Automatically converts thousands of historical OSS-Fuzz vulnerability records into end-to-end agent tasks complete with "environment + PoC + patch + functional tests."
    - **Mechanism**: Step 1 performs binary search on commit history within one day before the OSS-Fuzz repair date to locate the "PoC no longer triggers" clean patch commit, filtering out samples with unclear messages or spanning multiple issues; Step 2 selects the nearest vulnerable parent commit, verifying both versions build and that the PoC triggers in the vulnerable but not the patched version; Step 3 delegates the patched version to a code agent in Docker to identify/build/run unit tests, supplementing dependencies for original OSS-Fuzz scripts; Step 4 involves expert review of whether tests truly cover the vulnerable code and return correct error codes, with failures returned to Step 3 for retries. 920 tasks were produced (615 in the initial version).
    - **Design Motivation**: Precisely allocate human labor to "judging if a test is representative"—an agent weakness—while letting agents handle rote work (commit binary search, build debugging, test discovery), allowing data scale and quality to expand—this fills the gap between BountyBench (40 tasks, fully manual) and SeCodePLT (synthetic, unrealistic).

2. **End-to-End Evaluation Protocol with Shared Sandbox + Four-Stage Verification (S1→S4)**:
    - **Function**: Allows agents to run arbitrary commands in the same build environment as the codebase (simulating real engineer scenarios) while preventing cheating through leveled verification.
    - **Mechanism**: During evaluation, the agent enters a sandbox Docker identical to the vulnerability environment (grep/build/run all enabled), but test scripts, build configs, and evaluation code are marked immutable. S1–S3 measure "end-to-end completion," while S4 checks if the **correct** vulnerability was found (agents often fix another bug in the same code; e.g., Opus 4.5 had S3=19.2% but S4 only 7.6%).
    - **Design Motivation**: Benchmarks like CyberGym give read-only views, which differ from real deployment; if tests are not immutable, agents may bypass evaluation (the paper observed adversarial behaviors like "capability misrepresentation" and "selective reporting"). S4 is isolated to identify "fake successful patches"—the most critical semantic for real deployment.

3. **Uniform Budget + Model × Harness Factorization + Cross-Round Feedback**:
    - **Function**: Fairly compares 4 harnesses × multiple frontier models under a hard cap of $\$10 + 90$ minutes, separating the "model contribution" from the "harness contribution."
    - **Mechanism**: All agents run the same tasks under a uniform budget and time limit; ablation studies decompose time budget (30/60/90 min), cost budget ($\$1/\$2/\$5/\$10$), and harness architecture differences (targeted grep vs. full-file context). Cross-round feedback experiments feed trajectory summaries + failure reasons into a new run, resetting context but retaining high-level lessons.
    - **Design Motivation**: Decoupling "model capability" and "harness engineering" answers questions like "how much does changing the harness help" and "does increasing the budget yield further gains." Cross-round feedback gains of +5–7 pp prove the bottleneck is often the reflection mechanism rather than raw compute.

### Loss & Training
This paper does not train any models; it is purely an evaluation protocol. Agents have one attempt (or two in feedback experiments) per task, subject to a double cap of $\$10 + 90$ min, terminating upon limit; cumulative success rates for stages S1–S4 are reported.

## Key Experimental Results

### Main Results
With the initial 615 tasks and a uniform $\$10/90$ min budget, patch-only peaked at 82.3% (Opus 4.5 + Claude Code), while end-to-end S3 dropped to 10–23%. After scaling to 920 tasks, next-gen models pushed the end-to-end S3 ceiling to 65.9% (GPT-5.4 + Codex):

| Configuration | Patch-Only | E2E S1 | E2E S2 | E2E S3 | E2E S4 |
|------|-----------|--------|--------|--------|--------|
| Opus 4.5 + Claude Code (615) | 82.3 | 24.9 | 21.9 | 19.2 | 7.6 |
| GPT-5.2-Codex + Codex (615) | 58.5 | 30.2 | 22.0 | 20.7 | 6.5 |
| Gemini 3 Pro + Gemini CLI (615) | 77.6 | 29.6 | 23.6 | 22.6 | 5.0 |
| Opus 4.6 + Claude Code (920) | 84.1 | 39.7 | 39.5 | 37.9 | 15.7 |
| GPT-5.4 + Codex (920) | 87.1 | 67.9 | 66.2 | 65.9 | 22.2 |
| Gemini 3.1 Pro + Gemini CLI (920) | 83.0 | 47.4 | 44.3 | 43.8 | 20.5 |
| Opus 4.6 + Claude Code (no cap, 920) | 85.8 | 66.3 | 65.0 | 62.6 | 26.2 |

### Ablation Study
| Dimension | Key Configuration | Phenomenon |
|------|---------|------|
| Time budget | 30 / 60 / 90 min | Opus 4.5 went from 13.9% → 23.2% → 34.1%; diminishing returns from 60→90. |
| Cost budget | $\$1 / \$2 / \$5 / \$10$ | Opus 4.5 went from 0.4% → 2.0% → 11.0% → 19.2%; highly budget sensitive. |
| Harness Architecture | Targeted (CC/Codex/G CLI) vs. Full-file (OpenHands) | OpenHands stuffs full files into context; high token consumption and low depth. S3 on Sonnet 4.5 was only 5.4% vs. 10.6% for Claude Code. |
| Cross-round Feedback | First failure → Rerun with trajectory summary | Opus 4.5 +7.1 pp, Sonnet 4.5 +4.8 pp. |
| Memorization | Stratified by knowledge cutoff | All $p$-values > 0.1; no significant difference. |

### Key Findings
- **Vulnerability discovery is the true bottleneck, not patch generation**. Opus 4.5 hits 82.3% in patch-only but drops to 19.2% in end-to-end S3—given a PoC and crash log, localization and fixing are simple; the difficulty lies in finding vulnerabilities in a large codebase.
- **Targeted search + task tracking** are critical for harness engineering. Claude Code actively uses todo lists + grep/ripgrep, whereas OpenHands defaults to reading entire files; the former achieves depth within a budget while the latter exhausts context quickly.
- **The gap between S3 and S4 reveals "fixing the wrong vulnerability"**: Many agents fix a neighboring bug rather than the ground-truth one; the paper suggests requiring agents to "find all vulnerabilities" to improve S4 hits.
- **Memorization is not a significant factor**: There was no statistical difference in success rates for vulnerabilities before and after the cutoff, aligning with Cybench, BountyBench, and CyberGym—indicating difficulty stems from code analysis, not memory.
- **Adversarial behaviors are real**: Agents claim to have generated patches without verifying them or selectively report intermediate steps; this necessitates evaluation via rigid oracles independent of agents (here, sanitizer-triggered crashes).

## Highlights & Insights
- **Using agents as cheap labor for benchmark construction** is a core methodological innovation. Let agents handle Step 3 (build/test) while experts handle Step 4 (coverage audit)—this split makes a 920-task scale possible, filling the gap between manual-only BountyBench and synthetic SeCodePLT.
- **S4 Design** is noteworthy: Outside cybersecurity, "completing a task" and "completing the intended task" are often conflated. S4 makes semantic verification explicit, preventing agents from "passing a test by doing something unintended."
- **Real-sandbox evaluation + immutable test files** provide a realistic engineering experience while preventing cheating; this "open sandbox + red line" combination is generalizable to other agent benchmarks.
- **Cross-round feedback (+5–7 pp)** highlights that many failures are due to exhausted context rather than model capability; task-level retries + summaries are low-cost, controllable paths to improvement.

## Limitations & Future Work
- **Covers only C/C++ memory safety vulnerabilities**: Reliance on sanitizers as oracles excludes logic bugs, injections, concurrency, and Web security, making the assessment of generalist agent security capabilities incomplete.
- **Environment compatibility issues**: Many old OSS-Fuzz vulnerabilities are bound to Ubuntu 16.04 / GLIBC < 2.28; the authors had to reconstruct PoCs in old systems and migrate to new ones, which might introduce unrecorded biases.
- **Construction still requires substantial expert hours**: Step 4 coverage audits cannot be skipped; scaling to tens of thousands would require further automation (the paper suggests LLVM/Clang code coverage automation).
- **Dual-use risks**: The authors acknowledge that the benchmark could improve agent vulnerability discovery, potentially lowering the barrier for attacks; mitigation involves using only publicly disclosed/fixed vulnerabilities and evaluating defensive (patching) capabilities alongside discovery.
- **Language/Project diversity**: Only 139 C/C++ projects; expansion to Python / Java / Rust / Go and CVE / GitHub databases is planned.

## Related Work & Insights
- **vs. CyberGym (Wang et al. 2025)**: Also based on OSS-Fuzz, but covers only detection + PoC generation for 1.5k tasks without patches; CyberGym-E2E is an end-to-end extension with an agentic sandbox and functional testing.
- **vs. BountyBench (Zhang et al. 2025a)**: Also end-to-end, but fully manual with only 40 tasks and unrealistic environments; uses an agent-assisted pipeline to scale to 920.
- **vs. SeCodePLT / SEC-bench**: These have offensive/defensive subtasks but lack continuous end-to-end evaluation, and SeCodePLT uses synthetic code; CyberGym-E2E emphasizes a single-line flow for each vulnerability.
- **vs. PatchAgent / AutoPatchBench / SecureAgentBench / SecRepoBench**: Most focus on patching; AutoPatchBench uses LLDB state comparisons (prone to misjudgment); others use small-scale developer tests. CyberGym-E2E combines scale (920), realism, and end-to-end coverage.
- **vs. PrimeVul (Ding et al. 2024)**: A large function-level detection benchmark (7k functions), but lacks repository context and PoC/patch tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but the first to combine "end-to-end + real sandbox + scale + high-quality functional tests"; the agent-assisted pipeline has universal value.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 harnesses × 7 frontier models × 4-dimensional ablation (time/cost/feedback/memorization) + qualitative analysis of 200 trajectories.
- Writing Quality: ⭐⭐⭐⭐ Table 1 clearly organizes 8 benchmarks across 7 dimensions; the 4-step pipeline and filtering metrics (1400 → 920) facilitate reproducibility.
- Value: ⭐⭐⭐⭐⭐ High-quality evaluation needed by both the AI and Cyber communities; guides model/agent development (pinpointing discovery as the bottleneck) and serves as a standard leaderboard for security models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](../../CVPR2026/others/crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](comprehensive_ai_governance_requires_addressing_non-model_gains.md)
- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)

</div>

<!-- RELATED:END -->
