---
title: >-
  [Paper Note] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning
description: >-
  [ACL 2026][LLM Pretraining][Forgetting curve] The authors realign the "spaced repetition" concept from the Ebbinghaus forgetting curve from "training steps" to "model time" (accumulated parameter update norm $\Delta_t =…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Forgetting curve"
  - "model time"
  - "parameter update dynamics"
  - "spaced repetition"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 550f28fbc072a0d4
---

# FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning

**Conference**: ACL 2026  
**arXiv**: [2601.03938](https://arxiv.org/abs/2601.03938)  
**Code**: https://github.com/WoodScene/FOREVER  
**Area**: Continual Learning / LLM / Memory Replay  
**Keywords**: Forgetting curve, model time, parameter update dynamics, spaced repetition, catastrophic forgetting

## TL;DR
The authors realign the "spaced repetition" concept from the Ebbinghaus forgetting curve from "training steps" to "model time" (accumulated parameter update norm $\Delta_t = \|\Theta_t - \Theta_{t-1}\|_2$). Specifically, the accumulated model time $\tau_t$ determines **when to replay**, while the instability ratio $r_t$ between the recent update intensity $\mu_t$ and the baseline $\mu_0$ adaptively controls **how to replay** (regularization strength). This approach consistently outperforms SOTA across 3 CL benchmarks and 4 backbones (0.6B–13B), achieving OP +1.2% and BWT +0.9% over the strongest baseline VBM.

## Background & Motivation

**Background**: The core goal of LLM Continual Learning (CL) is to learn new tasks sequentially without catastrophic forgetting of old tasks. Replay-based methods (storing and periodically replaying a small number of old samples) have become mainstream due to their simplicity and effectiveness. Their design centers on two core questions: **when to replay** and **how strong the replay should be**.

**Limitations of Prior Work**: (a) Existing replay schedules are mostly hand-crafted heuristics, such as uniform intervals, fixed weights, or step-based Ebbinghaus intervals. (b) Even recent Ebbinghaus-inspired works (Zhong 2024, VBM 2025) assume that "training steps = time." However, the model changes resulting from the same number of steps can differ by orders of magnitude under different learning rates or batch sizes—the same "7 days" might correspond to entirely different model states. (c) Replay intensity $\beta$ is usually fixed and fails to respond to whether the model is "changing rapidly" or "steadily converging."

**Key Challenge**: The human forgetting curve is a function of "days passed," whereas the true measure for models should be "how far the model has moved in the parameter space." Aligning human time to model time has never been strictly performed. Furthermore, replay intensity should be coupled with the model's current instability, but existing methods treat "when" and "how" as separate issues.

**Goal**: (i) Define a model-centric "time" for LLMs that is decoupled from optimization hyperparameters; (ii) map Ebbinghaus intervals using this time to determine when to replay; (iii) use the same update dynamics signal to drive how to replay (regularization strength); (iv) validate the approach across multiple backbones (0.6B–13B) and benchmarks.

**Key Insight**: Starting from the conceptual misalignment of "human days vs model days," the authors note that the parameter update norm $\Delta_t$ directly quantifies "how far the model has moved." It can be accumulated to represent "model time" and averaged via EMA to estimate the "current model movement speed"—allowing one signal to answer both questions.

**Core Idea**: Replace step count with accumulated parameter updates $\tau_t = \sum_i \Delta_i$ as the measure of model time. Use the ratio $r_t$ of recent update intensity $\mu_t$ to baseline $\mu_0$ to adaptively adjust regularization strength, unifying "when and how" under the same update dynamics signal.

## Method

### Overall Architecture
FOREVER splits replay into two tightly coupled components that share the same signal (parameter update magnitude):
**(1) When to Replay — Forgetting Curve-inspired Replay Scheduler**: The accumulated parameter updates within a warm-up window of $S$ steps (here $S=24$) is defined as a "virtual model day" $\tau_{\text{day}}$. Ebbinghaus human days $\mathcal{D}_{\text{human}}=\{1,2,4,7,15,30,...\}$ are mapped to $\mathcal{D}_{\text{model}}=\{d\cdot\tau_{\text{day}} \mid d\in\mathcal{D}_{\text{human}}\}$. During training, $\tau_t$ is tracked, and the $j$-th replay is triggered when $\tau_t \geq \mathcal{D}_{\text{model}}^{(j)}$. $\tau$ is reset at the start of each task.
**(2) How to Replay — Intensity-aware Replay Regularization**: An instability ratio $r_t = \mu_t/\mu_0$ is constructed using the baseline intensity $\mu_0 = \frac{1}{S}\sum_{t=1}^S \Delta_t$ and an EMA $\mu_t = (1-\lambda)\mu_{t-1} + \lambda \Delta_t$. The replay regularization strength is defined as $\beta_t = \beta_{\text{base}} \cdot \text{clip}(1 + \gamma(r_t - 1), g_{\min}, g_{\max})$. The final replay loss is $\mathcal{L}_{\text{replay}} = \mathcal{L}_{\text{task}}^{(\text{old})} + \beta_t \sum_j \|\Theta_j - \Theta_j^\star\|_2^2$, where $\Theta^\star$ is a snapshot from the end of the previous task.

### Key Designs

1. **Model-Centric Time Calibration (Accumulated parameter updates as "model time")**:
    - **Function**: Replaces training steps with accumulated parameter update norm $\tau_t = \sum_{i=1}^t \Delta_i$ as the time metric for "how far the model has moved." It defines a "virtual day" $\tau_{\text{day}} = \sum_{i=1}^S \Delta_i$ (warm-up window) as a model-specific time unit.
    - **Mechanism**: $\Delta_t = \|\Theta_t - \Theta_{t-1}\|_2$ is calculated only for trainable parameters (LoRA weights) and is obtained directly from the optimizer updates, requiring **no additional forward/backward passes**—making the extra overhead nearly zero. The accumulation $\tau_t$ reflects the total distance moved in parameter space, making it far more robust to hyperparameters like learning rate and batch size than step count. Human days $\{1,2,4,7,15,30,...\}$ are mapped to the model timeline via the conversion factor $\tau_{\text{day}}$, leading to the trigger condition $\tau_t \geq \mathcal{D}_{\text{model}}^{(j)}$.
    - **Design Motivation**: The same number of steps at $lr=1e-4$ and $lr=1e-5$ corresponds to different model states. Tying replay to steps forces tasks with different learning rates to "review at the same pace," causing misalignment. Using update norms ensures reviews are scheduled based on how much the model has actually changed.

2. **Forgetting Curve-Inspired Increasing-Spacing Schedule**:
    - **Function**: Schedules replay events according to the Ebbinghaus "dense-to-sparse" structure, corresponding to the rapid early decay and slow later decay of human memory.
    - **Mechanism**: Standard Ebbinghaus intervals $\{1,2,4,7,15,30,...\}$ are used as $\mathcal{D}_{\text{human}}$. Ablations compared four schedules: standard Ebbinghaus, exponential $\{1,2,4,8,16,...\}$, polynomial $\{1,4,9,16,...\}$, uniform $\{2,4,6,8,...\}$, and decreasing $\{15,7,4,2,1\}$. Results consistently showed that any increasing interval is better than uniform or decreasing schedules. Standard Ebbinghaus performed slightly better than pure parameterized forms (OP 42.5 vs exponential 42.3 vs polynomial 41.5 vs uniform 40.9 vs decreasing 37.2). This proves the effectiveness stems from the structural "dense-to-sparse" principle rather than specific numbers.
    - **Design Motivation**: The authors emphasize that FOREVER's success is due to structural alignment rather than a "magic sequence," providing a paradigm for quantifying cognitive science into engineering design.

3. **Intensity-Aware Replay Regularization (Dynamic $\beta_t$ derived from the update signal)**:
    - **Function**: Dynamically adjusts replay regularization strength based on the model's current "instability"—applying strong constraints to prevent forgetting when the model changes rapidly and relaxing them during stability to allow for new task learning.
    - **Mechanism**: The instability ratio $r_t = \mu_t/\mu_0$ characterizes whether the current state is more aggressive or conservative than the start. If $r_t > 1$ (more aggressive), $\beta_t$ increases, providing stronger constraints; if $r_t < 1$, $\beta_t$ decreases. A clip operation $[0.5, 3.0]$ prevents numerical explosion, while $\gamma$ controls sensitivity. The loss $\mathcal{L}_{\text{replay}} = \mathcal{L}_{\text{task}}^{(\text{old})} + \beta_t \sum_j \|\Theta_j - \Theta_j^\star\|_2^2$ uses $L_2$ to anchor to the previous task snapshot $\Theta^\star$.
    - **Design Motivation**: This unifies "when to replay" and "how to replay" using the same update magnitude signal ($\Delta_t$). This approach is more elegant and requires fewer hyperparameters than complex hand-crafted schedules like SAPT or SSR.

### Loss & Training
A LoRA-based framework is used to unify all baselines. Warm-up window $S=24$, EMA smoothing $\lambda=0.05$, $\beta_{\text{base}}=10^{-3}$, and clipping $[0.5, 3.0]$. The memory buffer stores 2% of the original training data per task. All experiments are averaged over 3 seeds.

## Key Experimental Results

### Main Results: Three CL Benchmarks (Qwen3-0.6B backbone)

| Method | Standard CL OP↑ | BWT↑ | Long Sequence OP↑ | BWT↑ | SuperNI OP↑ | BWT↑ |
|------|----------------|------|-------------------|------|-------------|------|
| Fine-tuning (No CL) | 47.2 | -12.6 | 36.0 | -17.5 | 8.2 | -27.4 |
| EWC | 51.0 | -10.3 | 44.8 | -13.8 | 32.9 | -18.6 |
| O-LoRA | 59.4 | -7.9 | 54.1 | -12.4 | 23.7 | -17.5 |
| MixReplay | 65.8 | -8.0 | 65.1 | -11.4 | 34.6 | -14.1 |
| Fixed-interval Replay | 65.1 | -9.2 | 64.5 | -10.9 | 34.7 | -14.5 |
| SAPT | 68.8 | -6.9 | 67.2 | -8.8 | 38.5 | -6.2 |
| SSR | 68.4 | -7.1 | 67.5 | -9.0 | 40.1 | -5.4 |
| AIMMerging | 71.9 | -5.0 | 67.9 | -6.3 | 41.0 | -3.4 |
| VBM (step-based Ebbinghaus) | 71.5 | -5.2 | 68.1 | -6.1 | 41.3 | -3.7 |
| **FOREVER (Ours)** | **72.9** | **-4.7** | **69.4** | **-5.0** | **42.1** | **-2.9** |
| MTL (Upper Bound) | 77.4 | — | 77.8 | — | 48.2 | — |

Trends are consistent across backbones (Qwen3-0.6B/4B, LLaMA3.1-8B, LLaMA2-13B). On LLaMA3.1-8B vs VBM: OP 49.0 → 50.6 (+1.6), BWT -2.9 → -2.1 (+0.8).

### Ablation Study (SuperNI, task order 7)

| Category | Variant | OP↑ | BWT↑ |
|------|------|-----|------|
| **Full** | FOREVER | 42.5 | -2.8 |
| Scheduler (§3.2.1) | + Fixed-interval (FIR) | 40.1 | -5.2 |
| | + Reversed Replay (RR) | 37.2 | -7.8 |
| | + End-only Replay (ER) | 40.9 | -6.9 |
| Time Calibration (§3.2.2) | + Step-based (STC) | 41.3 | -3.9 |
| Regularization (§3.2.3) | − IAR | 39.9 | -4.4 |
| | + EWC-style PIR | 42.7 | -3.0 |
| | + IAR & PIR | 42.8 | -2.6 |

### Key Findings
- **Model-centric time is more critical than step-based time**: Changing calibration from update-dynamics to steps (STC) causes OP to drop from 42.5 to 41.3 (-1.2) and BWT from -2.8 to -3.9, proving that "model time" is substantial.
- **Increasing-spacing is the essence of Ebbinghaus**: Standard (42.5) ≈ Exponential (42.3) > Polynomial (41.5) > Uniform (40.9) ≫ Decreasing (37.2). This confirms the "dense-to-sparse" principle is the key factor. Reversed Replay (RR) showing catastrophic results (OP 37.2, BWT -7.8) further validates this.
- **End-only Replay is vastly inferior to distributed intervals**: OP 40.9 vs 42.5 demonstrates that "spaced repetition" itself is the key, not just the "total volume of replay."
- **IAR alone performs better than EWC-style PIR and is cheaper**: Removing IAR drops OP to 39.9. Adding EWC-style parameter importance regularization (PIR) yields only a marginal +0.2 gain while requiring high overhead to estimate and store importance. This suggests update intensity is an efficient, zero-cost proxy.
- **IAR + PIR shows almost no gain (+0.1)**: This suggests both capture the same underlying "training instability" signal, rendering PIR redundant.
- **Visualization confirms non-uniformity of model time mapping**: $\Delta_t$ is large early in each task and small later; $\tau_t$ grows non-linearly. The same "7-day" Ebbinghaus threshold span varies between 140–180 steps across tasks, proving step-based scheduling inevitably misaligns.
- **Consistent scaling**: Superior performance is maintained from 0.6B to 13B models, showing it is not just a small-model trick.

## Highlights & Insights
- **Design Elegance**: Using one signal ($\Delta_t$) to drive two decisions (when via $\tau_t$, how via $\mu_t$) reduces independent hyperparameters and improves interpretability. This "unified driving signal" philosophy is applicable to other "coupled decision" scenarios.
- **Engineering Human Cognition for LLMs**: Correcting the conceptual misalignment by translating "human days" into "model days" ($\tau_{\text{day}}$) makes the Ebbinghaus heuristic falsifiable and comparable in LLMs.
- **Zero Extra Overhead**: Since $\Delta_t$ is obtained from the optimizer, it requires no extra forward/backward passes, allowing FOREVER to be a drop-in scheduler for any replay-based CL method.
- **Structure Over Sequence**: By testing various parameterizations (exponential, polynomial), the authors prove the "dense-to-sparse" principle is the root cause of success, enhancing the credibility of the findings beyond "magic numbers."
- **Robust Experimental Design**: The "defensive ablation" approach (covering FIR/RR/ER schedules, STC calibration, and IAR/PIR regularization) addresses almost every potential design critique.

## Limitations & Future Work
- **Parameter update magnitude as a proxy**: Accumulated updates do not directly correspond to task-level performance degradation or semantic forgetting; future work could integrate direct task diagnostics.
- **Predefined Ebbinghaus prior**: While effective, the manual intervals may not be optimal for all tasks; learned or meta-learned adaptive schedules could be explored.
- **Limited Scope**: Verified only on NLU benchmarks and LoRA fine-tuning; performance in multimodal CL, full fine-tuning, or RLHF scenarios remains unverified.
- **Fixed memory buffer (2%)**: The relationship between buffer ratio and FOREVER’s gains was not systematically explored.
- **Empirical clip boundaries $[0.5, 3.0]$**: It is unclear if these need re-tuning for long-sequence tasks or larger models.
- **Snapshot selection**: Using the end-of-task snapshot $\Theta^\star$ might miss better intra-task checkpoints; strategies like EMA-of-snapshots could be more robust.

## Related Work & Insights
- **vs VBM (Kang 2025)**: Both are Ebbinghaus-inspired, but VBM uses steps. FOREVER’s +1.2 OP / +0.9 BWT gain confirms "model time" is superior to "step time."
- **vs EWC (Kirkpatrick 2017)**: FOREVER’s IAR achieves comparable results to parameter importance regularization without the high cost, proving update intensity is a viable surrogate.
- **vs SAPT / SSR / Recurrent-KIF**: These use hand-crafted schedules, whereas FOREVER’s dynamic schedule and adaptive regularization offer superior performance.
- **Insights**: (a) Any time-dependent mechanism (lr schedule, EMA, curriculum learning) should consider re-calibration via update dynamics; (b) Aligning cognitive science to models requires the correct "time metric"; (c) Replay schedules in LLM CL are as important as buffer size but currently undervalued.

## Rating
- Novelty: ⭐⭐⭐⭐ Moving Ebbinghaus from step-based to model-time-based is a key insight; the "one signal for when+how" design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive benchmarks, backbones, baselines, and ablations provide high confidence.
- Writing Quality: ⭐⭐⭐⭐ Strong narrative regarding "human vs model days," well-integrated formulas, and honest limitations.
- Value: ⭐⭐⭐⭐ A near-zero-cost drop-in scheduler with potential for transfer to RLHF and instruction tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](../../ICML2026/llm_pretraining/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)

</div>

<!-- RELATED:END -->
