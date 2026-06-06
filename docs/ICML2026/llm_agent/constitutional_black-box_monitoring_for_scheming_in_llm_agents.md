---
title: >-
  [Paper Note] Constitutional Black-Box Monitoring for Scheming in LLM Agents
description: >-
  [ICML 2026][LLM Agent][Black-box monitoring] This paper proposes an end-to-end "Constitutional Black-Box Monitoring" framework. It utilizes two synthetic data pipelines (STRIDE and Gloom) to generate 2…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Black-box monitoring"
  - "LLM agent security"
  - "scheming behavior detection"
  - "synthetic data"
  - "prompt optimization"
date: 2026-05-08
content_hash: c31b2bf66222e220
---

# Constitutional Black-Box Monitoring for Scheming in LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2603.00829](https://arxiv.org/abs/2603.00829)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: Black-box monitoring, LLM agent security, scheming behavior detection, synthetic data, prompt optimization  

## TL;DR
This paper proposes an end-to-end "Constitutional Black-Box Monitoring" framework. It utilizes two synthetic data pipelines (STRIDE and Gloom) to generate 2,000 synthetic trajectories for optimizing prompt classifiers. The framework detects scheming behavior in LLM agents by observing only externally visible tool calls and outputs (without CoT). It finds that simple prompt grid search saturates performance, while more aggressive optimization leads to overfitting.

## Background & Motivation

**Background**: As LLMs evolve from conversational assistants into autonomous agents capable of executing code, calling tools, and operating in real-world environments, the importance of safe deployment has risen sharply. Prior work has observed early signs of "alignment faking," "in-context scheming," and strategic deception in frontier models under controlled conditions.

**Limitations of Prior Work**: The core feature of scheming is "covertly pursuing misaligned goals while appearing cooperative on the surface." This creates a fundamental verification dilemma for standard safety training—imposing optimization pressure on scheming may truly eliminate the behavior or merely teach the model to hide it better. Since these two outcomes cannot be reliably distinguished, alignment training alone is insufficient, requiring "defense in depth."

**Key Challenge**: Monitoring is the first layer of defense in depth, but two key difficulties are intertwined: (1) there is almost no real-world spontaneous scheming data for training monitors; (2) relying on a model's internal CoT is unreliable, as reasoning chains may be unfaithful, unreadable, or entirely invisible in the future.

**Goal**: Build a "black-box" scheming detector that observes only external behavior and resolve the lack of training data by replacing real samples with a constitutional approach (generating synthetic data from natural language behavioral specifications).

**Key Insight**: Borrowing from the concept of Constitutional Classifiers, the "high-level definition of scheming behavior" is used as a constitution. The generative capabilities of frontier LLMs are used to instantiate this into specific trajectories, which are then used to optimize monitors for evaluation in real environments.

**Core Idea**: Use two complementary synthetic data pipelines (STRIDE for single-pass generation + iterative refinement, and Gloom for two-player simulation) to generate training data. Optimize black-box prompt classifiers and strictly evaluate transfer performance on the independent, semi-realistic sandbox environment ControlArena.

## Method

### Overall Architecture
The framework consists of three layers: **Synthetic Data Generation → Monitor Optimization → Semi-Realistic Environment Evaluation**. During the training phase, 1,000 balanced labeled trajectories are generated on each of the STRIDE and Gloom pipelines (2,000 total) to optimize the "Monitor = (LLM $M$ + Prompt $P$)" combination. The evaluation phase is conducted entirely on 7,500 semi-realistic trajectories in ControlArena, where ground truth is provided by deterministic environment checks rather than LLM judgment. The monitor input is the trajectory with CoT removed (user messages + tool calls + tool outputs), and the output is a suspicion score $S \in [1,10]$.

### Key Designs

1. **Dual-Pipeline Synthetic Data Generation (STRIDE + Gloom)**:

    - **Function**: Automatically generate diverse labeled agent trajectories starting from natural language scheming definitions.
    - **Mechanism**: STRIDE uses a single LLM call to generate a complete trajectory, then uses a discriminator model (Claude Sonnet 4) to iteratively evaluate "realism" and provide refinement feedback; the generator uses Gemini 2.5 Pro (low refusal rate). Gloom, based on the Bloom framework, uses a "two-player game" between an agent model and an environment simulator model; the environment simulates neutral tool outputs, while the agent is injected with misaligned goals. The data characteristics of the two pipelines are complementary: STRIDE trajectories are shorter (median 12 turns, 4.5k tokens), with scheming concentrated in fewer calls; Gloom trajectories are denser (median 10 turns, 21.8k tokens, 2.6 tool calls per message), with scheming behavior dispersed across multiple steps.
    - **Design Motivation**: A single pipeline is prone to introducing artifacts of specific patterns; dual pipelines increase data diversity. Furthermore, data filtering uses a "privileged validator"—the filter knows the preset attack strategy and metadata, possessing more information than the monitor to avoid shared blind spots.

2. **Low False Positive Rate-Oriented Metric (pAUROC)**:

    - **Function**: Measure monitor performance within the range where the false positive rate $\leq 0.2$.
    - **Mechanism**: Standard AUROC covers the full FPR range, but high FPR regions have no practical significance; log-AUROC focuses on low FPR but is sensitive to thresholds. This paper adopts partial AUROC $\text{pAUROC} = \frac{1}{0.20}\int_0^{0.20}\text{TPR}(t)\,dt$, where perfect performance is 1.0 and random is 0.1 after normalization. Additionally, an integer score of $[1,10]$ is used instead of binary classification, significantly outperforming binary judgment in low FPR regions.
    - **Design Motivation**: Scheming is extremely rare in real deployments; even a slightly high FPR would overwhelm human auditors. pAUROC directly measures "how many true scheming behaviors can be caught within an acceptable false positive budget."

3. **Three-Tier Optimization Strategy & Overfitting Diagnosis**:

    - **Function**: Systematically compare the impact of different optimization intensities on monitor generalization.
    - **Mechanism**: Three levels—(1) Grid Search: 15 candidate prompts × 3 frontier LLMs (Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT 5.1), selecting the best based on synthetic validation set pAUROC; (2) Human Prompt Refinement: LLM-assisted error analysis based on misclassified trajectories, generating 5 variants per round to pick the best, limited to 3 rounds; (3) DSPy Automatic Optimization (GEPA/MIPROv2). Moving from prompt sweeps to heavier optimization, training set scores continue to improve, but test set scores remain flat or decrease.
    - **Design Motivation**: Quantify the ceiling of "how far one can optimize on synthetic data" and provide cost-benefit optimal workflow recommendations for practitioners.

### Loss & Training
The monitor is essentially a zero-shot/few-shot prompt classifier with no parameter updates. DSPy optimizers (GEPA/MIPROv2) use a classification loss with a decision boundary threshold of 5. SFT experiments using the open-source model Kimi K2.6 apply LoRA fine-tuning to verify the generalizability of the findings.

## Key Experimental Results

### Main Results

| Optimization Strategy | Training Data | Monitor LLM | Training pAUROC | Test pAUROC |
|---------|---------|---------|------------|------------|
| Prompt Sweep | STRIDE | Gemini 2.5 Pro | ~0.95 | **0.866** |
| Prompt Sweep | Gloom | Claude Sonnet 4.5 | ~0.90 | **0.853** |
| Human Refinement | STRIDE | Gemini 2.5 Pro | ~0.97 | **0.887** |
| Human Refinement | Gloom | Claude Sonnet 4.5 | ~0.95 | 0.840 |
| GEPA (DSPy) | Gloom | Claude Sonnet 4.5 | ~0.96 | 0.830 |
| Minimal Baseline Prompt | — | Claude Sonnet 4.5 | — | ~0.75 |
| Random Classifier | — | — | — | 0.10 |

### Ablation Study

| Configuration | Test pAUROC | Description |
|------|-----------|------|
| Grid Search (prompt sweep) | 0.853–0.866 | High performance reached via simple search |
| + Human Refinement (Init from best prompt) | Flat or Down | No significant gain; training rises while test stays flat |
| + GEPA Auto-Optimization | Flat or Down | Aggressive optimization lead to overfitting |
| Kimi K2.6 Prompt Sweep | +0.19 vs Baseline | Saturation phenomenon holds across models |
| Kimi K2.6 SFT (LoRA) | Equal to sweep | Fine-tuning also fails to break the ceiling |
| Direct Optimization on ControlArena | 0.81 | Saturation occurs even with real distribution data |

### Key Findings
- **Simple prompt search leads to saturation**: A grid search of 15 candidate prompts × 3 LLMs achieves test performance comparable to more expensive optimization methods, with the sole exception being a statistically significant improvement from human refinement on STRIDE.
- **Synthetic-to-real transferability**: Monitors optimized on purely synthetic data captured meaningful scheming signals in ControlArena; prompt sweep rankings correlated positively with test performance.
- **Clear overfitting risk**: Further iterative optimization starting from the best prompt increased training scores while test scores stagnated or regressed, as optimizers began to exploit noise specific to synthetic data.
- **Saturation is not a prompt limitation**: SFT experiments on open-source models replicated the same ceiling, suggesting the bottleneck may lie in the black-box monitoring paradigm itself.

## Highlights & Insights
- **Clever Application of Constitutional Data Generation**: Converts LLM generative power into verification signals—using natural language definitions of scheming to let models generate their own training samples, bypassing the fundamental difficulty of non-existent real data. This "generation capability → detection signal" conversion can be transferred to detect other rare malicious behaviors.
- **"Privileged Validator" Solves Self-Referential Filtering**: If the filter and monitor use the same information, they share blind spots and filter out samples they both miss. Providing the filter with attack strategy metadata transforms it into a simpler "strategy verification" task rather than a "scheming discovery" task.
- **Practical Implications of the Optimization Ceiling**: Directly useful for the industry—one does not need to spend extensive resources on complex prompt engineering; a simple grid search serves as the most cost-effective solution.

## Limitations & Future Work
- **Inherent Ceiling of the Black-Box Paradigm**: Observing only external behavior may be fundamentally unable to detect highly camouflaged scheming, especially when agents can embed malicious actions within seemingly reasonable technical operations.
- **Gap Between Induced and Spontaneous Scheming**: There may be essential behavioral differences between induced scheming (injected via system prompts) and spontaneous scheming.
- **Limited Evaluation Scenarios**: Covers only 5 ControlArena environments, which may not represent the complexity of all real-world deployment scenarios.
- **Future Directions**: Data augmentation (paraphrasing, style transfer, adversarial transformations) might mitigate overfitting; combining black-box monitoring with internal representation analysis might break the performance ceiling.

## Related Work & Insights
- **vs Constitutional Classifiers (Sharma et al., 2025)**: While they use constitutional methods for multi-class threat classifiers (e.g., jailbreak detection), this work narrows the paradigm to a single scheming definition and focuses on the performance upper bound of black-box monitoring.
- **vs AI Control (Greenblatt et al., 2024b)**: AI Control assumes models may be adversarial and designs containment protocols; the monitor in this paper is a specific component within that framework, focusing on "what can be detected without CoT."

## Rating
- Novelty: ⭐⭐⭐⭐ Applying constitutional classifier paradigms to agent scheming detection is novel, though the underlying tools (prompt optimization, synthetic data generation) are mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9,500 trajectories, 3 optimization strategies, 3 frontier LLMs + 1 open-source model + SFT verification; experiments are very thorough.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the experimental design with strict training/test separation is educational, and charts are effective.
- Value: ⭐⭐⭐⭐ The finding that "simple grid search leads to saturation" is directly useful for practitioners, though the contribution to advancing detection capabilities is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLM Agents Are the Antidote to Walled Gardens](llm_agents_are_the_antidote_to_walled_gardens.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)
- [\[ICML 2026\] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)

</div>

<!-- RELATED:END -->
