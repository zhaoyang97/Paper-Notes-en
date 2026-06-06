---
title: >-
  [Paper Note] Pull Requests as a Training Signal for Repo-Level Code Editing
description: >-
  [ICML2026][Code Intelligence][Repo-level code editing] This paper proposes the Clean-PR training paradigm, which converts 16.4 million noisy GitHub Pull Requests into 2 million executable Search/Replace edit block corpor…
tags:
  - "ICML2026"
  - "Code Intelligence"
  - "Repo-level code editing"
  - "Pull Request training signals"
  - "Search/Replace edit blocks"
  - "mid-training"
  - "SWE-bench"
date: 2026-05-08
content_hash: 50062c409259b07e
---

# Pull Requests as a Training Signal for Repo-Level Code Editing

**Conference**: ICML2026  
**arXiv**: [2602.07457](https://arxiv.org/abs/2602.07457)  
**Code**: To be confirmed  
**Area**: code_intelligence  
**Keywords**: Repo-level code editing, Pull Request training signals, Search/Replace edit blocks, mid-training, SWE-bench

## TL;DR
This paper proposes the Clean-PR training paradigm, which converts 16.4 million noisy GitHub Pull Requests into 2 million executable Search/Replace edit block corpora through filtering, reconstruction, and replay verification. By combining Agentless-aligned SFT and error-driven data augmentation, Qwen2.5-Coder-32B achieves relative improvements of 13.6% and 12.3% on SWE-bench Lite and Verified respectively, surpassing the 72B Lingma-SWE and SWE-Fixer with only 32B parameters.

## Background & Motivation

**Background**: Repo-level software engineering (repo-level SWE) has become the core benchmark for testing code LLMs. Current SOTA systems on SWE-bench almost exclusively follow the "complex agent scaffolding" path—stacking agentic tool calls, structured localization, and large-scale test-time scaling. While performance is strong, it is difficult to attribute the source of gains.

**Limitations of Prior Work**: Training data shows a clear polarization. SWE-bench-like data (e.g., Multi-SWE-bench, SWE-Gym, R2E-Gym) is execution-verifiable but small in scale (thousands to tens of thousands); meanwhile, natural code corpora like The Stack and CodeReview are large enough but lack editing instruction signals on "how to modify multi-file code based on an issue." Neither is sufficient to truly "internalize" repo-level editing capabilities into model weights.

**Key Challenge**: To encode repo-level editing capabilities into weights, one needs (i) the scale of natural corpora, (ii) the structural signals of multi-file editing, and (iii) high-fidelity executability—triality which is difficult to achieve simultaneously.

**Goal**: To answer a fundamental question—in repo-level editing capabilities, how much can be directly encoded into model weights to move away from reliance on complex inference-time scaffolding? This is decomposed into two sub-questions: (a) how to extract "model-learnable" training signals from the extremely noisy GitHub PR stream; (b) if mid-training alone is insufficient for localization and navigation in large repositories, what additional training phases are needed.

**Key Insight**: The authors observe that GitHub Pull Requests naturally couple "natural language intent (description + linked issue)" with "accepted multi-file code changes," making them an ideal middle ground between SWE-bench and The Stack. However, only 18.59% of the 16.4 million raw PRs are considered "clean," requiring strict cleaning, edit block reconstruction, and replay verification for use.

**Core Idea**: Replace "fragile unified diffs" with "round-trip verified Search/Replace edit blocks" as PR training signals. Combined with step-by-step SFT aligned with Agentless and error-driven hard-negative augmentation, repo-level editing capabilities are solidified into the weights, allowing a 32B model to surpass 72B agentic solutions under a simplified Agentless protocol.

## Method

### Overall Architecture
Clean-PR is a complete recipe consisting of two-stage training and a single data pipeline:

- **Data Construction (Left)**: 8.6 TB raw GitHub data → Noise filtering (PR validity + language alignment) → Search/Replace reconstruction (round-trip verification based on patch replay) → Issue augmentation (concatenating linked issue titles/descriptions) → Clean-PR-full (3.05M / 46.4B tokens) → Complexity control + repo-level sampling → **Clean-PR-train (2.0M PRs / 17.7B tokens / 12 languages)**.
- **Training Phase 1 (Mid-training)**: Starting from Qwen2.5-Coder-32B-Base, repo-level editing mid-training is performed on Clean-PR-train to encode priors for "where to edit" and "how to edit" into the weights.
- **Training Phase 2 (Agentless-Aligned Step-by-step SFT)**: Based on verifiable trajectories from SWE-rebench / SWE-Gym, each repair sample is decomposed into three supervised tasks (file localization → fine-grained navigation → patch generation), with hard negatives and distractor files/regions injected via error-driven augmentation.
- **Inference Time**: A Simplified Agentless protocol is used (linear localization → navigation → editing), without an agent loop.

### Key Designs

1. **Search/Replace Edit Blocks + Round-Trip Verification Data Reconstruction Pipeline**:
    - **Function**: Converts 16.4M noisy PR diffs into 2.0M Search/Replace training samples verifiable by byte-level replay, serving as the "foundation" of the paradigm.
    - **Mechanism**: First, coarse filtering is done based on PR validity (dropping bots/unmerged/docs-only) and "core expansion rules" (at least one core source file in 12 target languages must be modified), retaining only 18.59% of samples. For each PR: ① apply the original patch to the *before* repository snapshot to get *after*; ② algorithmically derive the minimal edit span and select the "shortest unique anchoring context" in *before* as the Search block; ③ re-apply the generated S/R blocks to *before*, and drop if it is not bitwise consistent with the ground-truth *after*. Finally: complexity control (≤5 core files, average files $3.0 \to 1.7$), window cropping around S/R blocks for >100k token files, and random sampling of 2000 entries per repo to prevent distribution skew.
    - **Design Motivation**: The authors analyzed the fragility of unified diffs—they rely heavily on exact line number prediction, and models often fail during generation due to format drift. S/R uses unique context matching to locate edit points, bypassing line number fragility, and round-trip verification ensures the executability of every sample. In experiments, the Valid Patch rate jumped from 89.7% (StarCoder-style) to 96.3%, and Line Acc. rose from 47.0% to 55.7%, verifying that "unique search block" signals allow the model to output more precise navigation cues.

2. **Issue-Augmented Intent (Intent Completion via Linked Issues)**:
    - **Function**: Solves the problem where PR descriptions often only say "Fixes #123," leading to missing original bug reports/requirements in training signals, and aligns the training distribution with the real-world "issue → patch" workflow.
    - **Mechanism**: Parse issue reference identifiers in PR bodies and concatenate the titles and bodies of all linked issues into the training context as input alongside code context. This completes PR descriptions that originally only had "solution summaries" into "full problem statements + solution summaries." The average description length in Clean-PR-train thus increased from 50.0 to 59.5 words.
    - **Design Motivation**: In reality, models face "detailed bug reports" rather than "one-sentence summaries." Training-inference distribution consistency allows the model to learn "alignment from natural language intent to code implementation." Ablations show that removing linked issues and using PR Desc Only drops Verified performance from 27.8% to 25.7%; while stronger than the baseline alone, the combination with S/R format yields the best results.

3. **Agentless-Aligned Step-by-step SFT + Error-Driven Augmentation**:
    - **Function**: Further aligns the "ability to edit given clean context" learned in mid-training to real-world SWE-bench scenarios (finding 1.7 files to change in a 3010-file repo and navigating precisely), while preventing "over-editing" when retrieval is imperfect.
    - **Mechanism**: Decomposition into three-stage supervision using ground-truth trajectories from SWE-rebench/SWE-Gym—Step 1 File Localization (Issue + Repo Tree → Filepath, excluding non-code tags like .md/.txt); Step 2 Fine-grained Navigation (using AST to map ground-truth edits to functions/classes, acting as target for Issue + File Content → Relevant Context); Step 3 Patch Generation (Localized Context → Minimal unique S/R block). Error-driven augmentation uses Qwen-2.5-Coder-32B-Instruct as an intermediate model to generate hard negatives: Step 2 uses $\text{Issue} + (F_{gt} \cup F_{neg}) \to \text{Relevant Context}$, requiring the model to output "No changes needed" for $F_{neg}$; Step 3 uses $\text{Issue} + (C_{relevant} \cup C_{noise}) \to \text{Search/Replace}$, teaching the model to refuse modifications on semantically similar but irrelevant code blocks. Total SFT data: 18,891 / 30,752 / 25,439 = 75,082, including 21,864 negative samples from error augmentation.
    - **Design Motivation**: Standard SFT only trains the "perfect localization" happy path, but real inference retrieval is inherently noisy. Without explicit distractor injection, models tend to over-edit irrelevant files because they "saw it and changed it" (Zeng et al., 2025). In ablations under the All-Languages setting, error augmentation improved Lite Pass@1 from 21.8% to 24.3% and Verified from 27.4% to 30.6%, with a synchronized increase in Line Acc., proving it learns "precise discrimination amidst interference" rather than just memorizing patterns.

### Loss & Training
- Base: Qwen2.5-Coder-32B-Base (Ablations include 7B Base).
- Hardware: 32×H200, context window 32,768 tokens; Python-only mid-training ~60 wall-clock hours, All-Languages full mid-training 259 hours, step-by-step SFT another 38 hours.
- Loss: Standard next-token CE, supervised via unified format across three-stage SFT tasks; hard negative samples are mixed with positives (no separate weighting).
- Inference: Simplified Agentless protocol executing localization → navigation → editing linearly, without multi-turn or external tool calls.

## Key Experimental Results

### Main Results
Comparison on SWE-bench Lite / Verified with a 32B base (Pass@1 as the primary metric):

| Setting | Mid-Train | SFT | Valid Patch | File Acc. | Line Acc. | Pass@1 |
|------|-----------|-----|-------------|-----------|-----------|--------|
| Qwen-Coder-32B-Instruct (Lite) | None | ✗ | 77.0 | 74.7 | 38.3 | 10.7 |
| Qwen-Coder-32B-Base + SFT (Lite) | None | ✓ | 84.0 | 78.3 | 46.7 | 11.3 |
| + StarCoder2-style (17.4B, Lite) | Diff | ✓ | 89.7 | 84.3 | 47.0 | 15.7 |
| **Clean-PR-train All (17.7B, Lite)** | S/R | ✓ | **96.3** | **87.3** | **55.7** | **24.3** |
| Qwen-Coder-32B-Instruct (Verified) | None | ✗ | 77.6 | 70.6 | 42.3 | 18.3 |
| + StarCoder2-style (17.4B, Verified) | Diff | ✓ | 82.4 | 77.7 | 48.4 | 20.4 |
| **Clean-PR-train All (17.7B, Verified)** | S/R | ✓ | **95.2** | **80.7** | **52.2** | **30.6** |

Relative to the Instruct baseline, Lite absolute gain is $+13.6\%$, Verified $+12.3\%$; relative to StarCoder2-style with the same token count, Lite $+8.6\%$, Verified $+10.2\%$.

Comparison with external open-source SOTA (pass@1):

| Method | Framework | Params | Lite | Verified |
|------|------|------|------|----------|
| SWE-Gym | OpenHands | 32B | 15.3 | 20.6 |
| Lingma-SWE | SWESynInfer | 72B | 22.0 | 30.2 |
| SWE-Fixer | SWE-Fixer | 72B | 22.0 | 30.2 |
| **Clean-PR** | Agentless | **32B** | **24.3** | **30.6** |

### Ablation Study

| Configuration | Lite Pass@1 | Verified Pass@1 | Description |
|------|-------------|------------------|------|
| Full (S/R + Linked Issue, Python) | 22.3 | 27.8 | Full Clean-PR setting (Python subset) |
| w/o S/R (to Diff) + Linked Issue | 19.1 | 24.4 | Replacing edit format only: Verified drops 3.4% |
| w/o Linked Issue (PR Desc only) | 20.4 | 25.7 | Removing issue augmentation: Verified drops 2.1% |
| StarCoder-style (Diff + PR Desc Only) | 15.7 | 20.4 | Combined changes, overall drop |
| Standard SFT (All Languages) | 21.8 | 27.4 | Without Error-Driven augmentation |
| + Error Aug. (All Languages) | 24.3 | 30.6 | Gains in Line Acc. synchronized with Pass@1 |

### Key Findings
- **Data Format > Data Scale**: Python-only Clean-PR with only 3.8B tokens outperforms the 17.4B token StarCoder2-style baseline (Lite 22.3% vs 15.7%), showing that "clean and execution-verified" training signals are far more important than stacking tokens.
- **Avoiding Catastrophic Forgetting**: StarCoder2-style diff training caused HumanEval to degrade from 54.1% to 47.6% ($-6.5\%$), whereas Clean-PR improved HumanEval to 59.8% ($+5.7\%$) and LiveCodeBench from 29.0% to 32.6%—the precise context matching objective positively transfers to general code capabilities.
- **Small Models Also Benefit**: Applying the recipe to Qwen2.5-Coder-7B increased Lite Pass@1 from 10.3% to 14.5% and Verified from 14.2% to 20.4%, with localization metrics improving more than the 32B model, showing high-quality PR supervision is critical for capacity-constrained models.
- **Pass@k Reveals Ranking Bottleneck**: Verified Pass@1 30.6% → Pass@10 41.5% (Lite 24.3% → 37.5%), implying the model's intrinsic reasoning power is stronger than single-decoding indicates; adding a verifier/re-ranker could yield further gains.
- **Multi-lingual Transfer**: On Multi-SWE-bench Flash, Clean-PR achieved 12.3% Pass@1, outperforming Instruct and StarCoder2-style baselines, indicating that 12-language mid-training gives the model cross-lingual repo-editing generalization.

## Highlights & Insights
- **"Reconstructing dirty data with LLM-friendly intermediate representations" is the key unlock for PR training**: Replacing unified diffs with Search/Replace and adding round-trip bitwise verification treats "training sample executability" as a first-class citizen. This aligns directly with mainstream edit pipelines like Aider/SWE-agent while avoiding "looks correct but fails on apply" issues caused by line number fragility.
- **Error-driven augmentation is a "cheap prescription" for training-inference consistency**: Instead of complex RL or rejection sampling, it uses an intermediate model to generate hard negative files/blocks and requires the main model to output "No changes needed" or avoid modifications. This essentially writes "imperfect retrieval" into the training distribution; the idea is generalizable to any "retrieve-then-generate" task.
- **Challenging the "agents are mandatory to win" narrative**: By using a linear Simplified Agentless protocol, 32B parameters, and pure weight internalization, the model exceeds 72B + agentic loop solutions. This provides clean causal evidence for "data > scaffolding," offering practical significance for both academic control of variables and industrial cost reduction.
- **Mid-training is neither pretrain nor SFT**: This paper positions it as a distinct stage "between base and SFT, specifically for encoding priors with domain-specific executable signals," a paradigm worth extending to other vertical domains (e.g., SQL, Notebook editing, robotics code).

## Limitations & Future Work
- **Acknowledged Limitations**: The 11% gap between Pass@1 and Pass@10 indicates imperfect likelihood ranking, necessitating verifiers or re-rankers. The wall-clock cost of mid-training (259 hours × 32×H200) is not academic-friendly for reproduction.
- **Data Availability vs Legal Risks**: Claiming to release the largest PR corpus (2M), but GitHub PRs involve open-source license compliance (GPL/AGPL, etc.) which the authors did not fully discuss; uncertainty exists for downstream commercial model training.
- **Evaluation Constraints**: SWE-bench Lite/Verified remains primarily Python-focused; multi-lingual capabilities were only validated on 300 samples of Multi-SWE-bench Flash. True capabilities in languages like C/C++ (dependent on complex compilation/linking) require larger-scale evaluation.
- **No RL/Self-Improvement**: The entire pipeline falls within the SFT category (mid-training + supervision) and does not utilize the executable nature of SWE-bench for outcome-based RL; RL + Clean-PR is a natural next step.
- **Sensitivity of Error Augmentation to "Intermediate Model" Quality**: Using Qwen-2.5-Coder-32B-Instruct to generate hard negatives might inject "noise" that doesn't represent real retrieval failure modes if the intermediate model's distribution is biased.

## Related Work & Insights
- **vs StarCoder2 / The Stack (Natural Corpora)**: These prioritize scale and multi-lingual coverage but do not constrain edit formats or single PR verification. This paper exchanges "scale" for "density" via round-trip verification and S/R format, significantly outperforming them with equal tokens and improving on general code benchmarks, negating the pessimistic assumption that specialized training must hurt general performance.
- **vs SWE-Gym / R2E-Gym (SWE-bench-style data)**: These are high-fidelity but limited to thousands of episodes. This paper provides a scalable path from "millions of PRs → executable training signals." They are complementary rather than alternative (the SFT stage here actually used trajectories from SWE-Gym/SWE-rebench).
- **vs Agentless (Xia et al., 2025)**: Agentless simplified the process into localization → navigation → editing. This paper not only adopts its S/R edit block convention but also explicitly distills this whole process into weights, further reducing reliance on scaffolding at inference.
- **vs OpenHands / SWE-agent (Agentic Frameworks)**: These rely on multi-turn tool usage + iterative planning—powerful but expensive. This paper achieves SOTA with "weight internalization + linear pipeline," providing an alternative standard: "whether an agent is needed depends on whether the corresponding prior exists in the weights."

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the data paradigm (Search/Replace + round-trip verification + linked issue + error augmentation) is new, and clearly defines mid-training as an independent stage. Individual techniques are not all original, but the recipe is complete and reproducible.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 32B main results, 7B scaling generalization, multi-lingual generalization, Pass@k extension, three sets of crucial ablations (data format/Issue/Error-Aug), and catastrophic forgetting analysis on HumanEval/LiveCodeBench. The chain of evidence is very solid.
- Writing Quality: ⭐⭐⭐⭐ The data pipeline, training stages, and causal decomposition in ablations are clear. Minor drift in table references (e.g., "Table 3 presents..." vs actual table numbers) exists but does not hinder understanding.
- Value: ⭐⭐⭐⭐⭐ Releasing 2M verified PR samples is a scarce resource for the community; exceeding 72B with 32B directly challenges the mainstream narrative that agent scaffolding is indispensable, impacting both industrial deployment and academic research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](../../ACL2026/code_intelligence/swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](../../ACL2026/code_intelligence/to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)

</div>

<!-- RELATED:END -->
