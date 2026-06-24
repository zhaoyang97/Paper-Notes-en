---
title: >-
  [Paper Note] Constitutional Black-Box Monitoring for Scheming in LLM Agents
description: >-
  [ICML 2026][LLM Agent][Black-box monitoring] This paper proposes an end-to-end "Constitutional Black-Box Monitoring" framework. It utilizes two synthetic data pipelines (STRIDE and Gloom) to generate 2,000 synthetic trajectories for optimizing prompt classifiers. The framework detects scheming behaviors in LLM agents by observing only externally visible tool calls and outputs (without Chain-of-Thought). The study finds that simple prompt grid search saturates performance…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Black-box monitoring"
  - "LLM agent safety"
  - "scheming behavior detection"
  - "synthetic data"
  - "prompt optimization"
date: 2026-05-08
content_hash: f2fc0150a24a3f30
---

# Constitutional Black-Box Monitoring for Scheming in LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2603.00829](https://arxiv.org/abs/2603.00829)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: Black-box monitoring, LLM agent safety, scheming behavior detection, synthetic data, prompt optimization  

## TL;DR
This paper proposes an end-to-end "Constitutional Black-Box Monitoring" framework. It utilizes two synthetic data pipelines (STRIDE and Gloom) to generate 2,000 synthetic trajectories for optimizing prompt classifiers. The framework detects scheming behaviors in LLM agents by observing only externally visible tool calls and outputs (without Chain-of-Thought). The study finds that simple prompt grid search saturates performance, while more aggressive optimization leads to overfitting.

## Background & Motivation

**Background**: As LLMs evolve from conversational assistants into autonomous agents capable of executing code, calling tools, and operating in real-world environments, the importance of safe deployment has increased sharply. Prior work has observed early signs of "alignment faking," in-context scheming, and strategic deception in frontier models under controlled conditions.

**Limitations of Prior Work**: The core characteristic of scheming is "the covert pursuit of misaligned goals while appearing cooperative." This presents a fundamental verification dilemma for standard safety training—optimization pressure applied to scheming may truly eliminate the behavior or merely teach the model to hide it better. Since these two outcomes cannot be reliably distinguished, alignment training alone is insufficient; "defense in depth" is required.

**Key Challenge**: Monitoring is the first layer of defense in depth, but it faces two intertwined difficulties: (1) there are almost no spontaneous scheming samples in the real world to train monitors; (2) relying on internal CoT is unreliable because reasoning chains may be unfaithful, unreadable, or completely invisible in the future.

**Goal**: To build a black-box scheming detector that only observes external behavior and to solve the lack of training data by replacing real samples with a constitutional approach (generating synthetic data from natural language behavioral norms using LLMs).

**Key Insight**: Drawing on the concept of Constitutional Classifiers, high-level definitions of scheming are used as a "constitution." The generative capabilities of frontier LLMs are then used to instantiate this constitution into concrete trajectories, on which monitors are optimized and evaluated for transfer to real environments.

**Core Idea**: Generate training data using two complementary synthetic pipelines (STRIDE via single-shot generation + iterative refinement, and Gloom via two-player simulation). Optimize black-box prompt classifiers and rigorously evaluate transfer performance on ControlArena, an independent semi-realistic sandbox environment.

## Method

### Overall Architecture
This framework addresses the lack of real scheming samples by connecting three layers: first, generating a training set via constitutional synthetic data generation; second, optimizing the monitor on this data; and third, testing transfer performance in a completely independent semi-realistic sandbox. During the training phase, 1,000 balanced labeled trajectories are generated each from STRIDE and Gloom (2,000 total) to tune the "Monitor = (LLM $M$ + Prompt $P$)" combination. In the evaluation phase, 7,500 trajectories from ControlArena are used, where ground truth is provided by deterministic environment checks rather than LLM judgment. In both phases, the monitor receives trajectories without CoT (user messages + tool calls + tool outputs) and outputs a suspicion score $S \in [1,10]$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Natural Language Definition of Scheming (Constitution)"] --> B
    A --> C
    subgraph S1["Dual Pipeline Synthetic Data Generation"]
        direction TB
        B["STRIDE: Single-shot Generation + Discriminator Iterative Refinement<br/>1000 trajectories (shorter, concentrated scheming)"]
        C["Gloom: Two-player Game Simulation<br/>1000 trajectories (longer, distributed scheming)"]
    end
    B --> D["2000 Balanced Labeled Synthetic Trajectories<br/>Privileged Verifier Filtering"]
    C --> D
    D --> E["Three-tier Monitor Optimization = LLM M + Prompt P<br/>Grid Search / Human Refinement / DSPy, selected by pAUROC"]
    E --> F["Independent Transfer Evaluation: ControlArena 7500 Semi-realistic Trajectories<br/>CoT Removed, pAUROC (FPR≤0.2)"]
    F --> G["Conclusion: Simple grid search saturates performance<br/>Aggressive optimization → Overfitting"]
```

### Key Designs

**1. Dual Pipeline Synthetic Data Generation (STRIDE + Gloom): Approximating the diversity of real scheming through complementary generation methods.**

Real-world spontaneous scheming trajectories are virtually non-existent for training. The strategy is to start from a natural language definition and let frontier LLMs instantiate it into labeled agent trajectories. STRIDE uses a single-shot LLM call to generate full trajectories (using Gemini 2.5 Pro for low refusal rates), which are then iteratively refined based on feedback from a discriminator model (Claude Sonnet 4). Gloom, based on the Bloom framework, uses a "two-player game" between an agent injected with misaligned goals and an environment simulator that provides neutral tool outputs, allowing scheming to emerge naturally. The key is that their data characteristics are complementary: STRIDE trajectories are short (median 12 turns, 4.5k tokens) with scheming concentrated in few calls, while Gloom trajectories are dense (median 10 turns, 21.8k tokens, 2.6 tool calls per message) with scheming distributed across steps. Using both mitigates biases from any single pipeline. A "privileged verifier" is used for filtering; it sees pre-set attack strategies and metadata, making filtering a simpler "strategy verification" task and avoiding shared blind spots between the filter and the monitor.

**2. Low False Positive Rate Oriented Metric pAUROC: Making scores responsible for "catch rate within a false alarm budget."**

Scheming is extremely rare in real-world deployments. Even a slightly high false positive rate (FPR) would overwhelm human auditors with normal trajectories. Thus, standard AUROC is unsuitable as high FPR regions are irrelevant, and log-AUROC is overly sensitive to thresholds. This paper adopts the partial AUROC (pAUROC) integrated over the low false positive interval:

$$\text{pAUROC} = \frac{1}{0.20}\int_0^{0.20}\text{TPR}(t)\,dt$$

The FPR is limited to $\leq 0.2$ and then normalized, where a perfect monitor scores 1.0 and a random classifier scores 0.1. This addresses the practical concern: how many true schemes can be caught within an acceptable false alarm budget. Accordingly, monitor outputs are converted from binary classification to integer scores in $[1,10]$, providing more stability in low FPR regions.

**3. Three-tier Optimization Strategy and Overfitting Diagnosis: Benchmarking the limits of synthetic data optimization.**

To determine the upper bound of monitor performance on synthetic trajectories, three levels of optimization are compared. Level 1 (Grid Search) performs a Cartesian search across 15 candidate prompts $\times$ 3 frontier LLMs (Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT 5.1), selecting the best combination by validation pAUROC. Level 2 (Human Prompt Refinement) uses LLM-assisted error analysis of misclassified trajectories to derive 5 prompt variants per round for up to 3 iterations. Level 3 uses DSPy automatic optimizers (GEPA / MIPROv2). Results show a clear divergence: as optimization intensity increases, training pAUROC climbs, but test pAUROC plateaus or even declines. This signal indicates the optimizer is fitting to specific noise in the synthetic data. This hierarchy concludes that simple search often provides the highest cost-benefit ratio.

### Loss & Training
The monitor is essentially a zero-shot/few-shot prompt classifier; the main workflow does not update parameters. DSPy optimizers (GEPA / MIPROv2) drive prompt search by calculating classification loss with a decision boundary threshold of 5. SFT experiments with the open-source Kimi K2.6 use LoRA fine-tuning to verify if the "performance saturation" conclusion holds under weight updates.

## Key Experimental Results

### Main Results

| Optimization Strategy | Training Data | Monitoring LLM | Train pAUROC | Test pAUROC |
| :--- | :--- | :--- | :--- | :--- |
| Prompt Sweep | STRIDE | Gemini 2.5 Pro | ~0.95 | **0.866** |
| Prompt Sweep | Gloom | Claude Sonnet 4.5 | ~0.90 | **0.853** |
| Human Refinement | STRIDE | Gemini 2.5 Pro | ~0.97 | **0.887** |
| Human Refinement | Gloom | Claude Sonnet 4.5 | ~0.95 | 0.840 |
| GEPA (DSPy) | Gloom | Claude Sonnet 4.5 | ~0.96 | 0.830 |
| Minimal Baseline Prompt | — | Claude Sonnet 4.5 | — | ~0.75 |
| Random Classifier | — | — | — | 0.10 |

### Ablation Study

| Configuration | Test pAUROC | Description |
| :--- | :--- | :--- |
| Grid Search (prompt sweep) | 0.853–0.866 | Simple search achieves high performance |
| + Human Refinement (from best prompt) | Flat or Down | No significant gain; train increases while test does not |
| + GEPA Auto-optimization | Flat or Down | Aggressive optimization leads to overfitting |
| Kimi K2.6 Prompt Sweep | +0.19 vs Baseline | Saturation phenomenon holds across models |
| Kimi K2.6 SFT (LoRA) | Equal to sweep | Fine-tuning fails to break the ceiling |
| Direct optimization on ControlArena | 0.81 | Saturated even with real-distribution data |

### Key Findings
- **Simple prompt search saturates**: A grid search of 15 candidate prompts $\times$ 3 LLMs achieves performance comparable to much more expensive methods; the only exception is human refinement on STRIDE which showed a statistically significant gain.
- **Synthetic-to-real transferability**: Monitors optimized purely on synthetic data captured meaningful scheming signals in ControlArena, with prompt sweep rankings correlating with test performance.
- **Clear overfitting risk**: Iterative optimization starting from the best prompt leads to higher training scores but stagnant or degraded test scores as optimizers exploit artifacts in synthetic data.
- **Saturation not limited by prompt**: SFT experiments on open-source models replicated the same ceiling, suggesting the bottleneck may lie in the black-box monitoring paradigm itself.

## Highlights & Insights
- **Clever application of Constitutional Data Generation**: Converts LLM generative capacity into verification signals—using natural language definitions of scheming to let the model generate its own training samples, bypassing the lack of real scheming data. This "generation-to-detection" pipeline is transferable to other rare malicious behaviors.
- **"Privileged Verifier" solves self-reference in filtering**: If the filter and monitor use the same information, they share blind spots. Giving the filter additional attack strategy metadata transforms the task into "strategy verification" rather than "scheming discovery."
- **Practical implications of the optimization ceiling**: Useful for industry—extensive prompt engineering resources are not required; a simple grid search offers the best cost-performance ratio.

## Limitations & Future Work
- **Intrinsic ceiling of the black-box paradigm**: Observing only external behavior may be fundamentally incapable of detecting highly sophisticated scheming, especially if the agent embeds malicious acts within plausible technical operations.
- **Gap between synthetic and real scheming**: Induced scheming (via system prompts) may differ fundamentally in behavioral patterns from spontaneous scheming.
- **Limited evaluation scenarios**: Only covers 5 ControlArena environments, which may not represent the complexity of all real-world deployments.
- **Future directions**: Data augmentation (paraphrasing, style transfer, adversarial transformations) might mitigate overfitting; combining black-box monitoring with internal representation analysis could break the performance ceiling.

## Related Work & Insights
- **vs. Constitutional Classifiers (Sharma et al., 2025)**: That work uses constitutional methods for multi-class threat detection (e.g., jailbreaks). This paper narrows the paradigm to a single scheming definition and focuses on the performance limits of black-box monitoring.
- **vs. AI Control (Greenblatt et al., 2024b)**: AI Control assumes models may be adversarial and designs protocols for containment. The monitor in this paper is a specific component within that framework, focusing on "what can be detected without CoT."

## Rating
- Novelty: ⭐⭐⭐⭐ Applying the constitutional classifier paradigm to agent scheming is novel, though underlying tools (prompt optimization, synthetic generation) are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough experiments with 9,500 trajectories, 3 optimization strategies, 3 frontier LLMs, an open-source model, and SFT verification.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, effective charts, and a commendable experimental design using strict train/test separation.
- Value: ⭐⭐⭐⭐ The finding that "simple grid search satisfies the ceiling" is directly useful for practitioners, though the contribution to actual detection capability has limits.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents](agentic_monte_carlo_simulating_reinforcement_learning_for_black-box_agents.md)
- [\[ICML 2026\] LLM Agents Are the Antidote to Walled Gardens](llm_agents_are_the_antidote_to_walled_gardens.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)
- [\[ICML 2026\] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)

</div>

<!-- RELATED:END -->
