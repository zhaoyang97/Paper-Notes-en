---
title: >-
  [Paper Note] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations
description: >-
  [ACL 2026][Reinforcement Learning][Long-form generation] LoVeC trains LLMs to append a numerical `<confidence>` label (0–10) after each sentence during long-form generation. Using GRPO (online with an oracle fact-checker…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Long-form generation"
  - "Verbalized confidence"
  - "GRPO"
  - "DPO"
  - "Factuality calibration"
date: 2026-05-08
content_hash: 16c1f92d86bbc55e
---

# LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations

**Conference**: ACL 2026  
**arXiv**: [2505.23912](https://arxiv.org/abs/2505.23912)  
**Code**: https://github.com/caiqizh/LoVeC (Available)  
**Area**: LLM Calibration / RLHF / Hallucination Detection  
**Keywords**: Long-form generation, Verbalized confidence, GRPO, DPO, Factuality calibration

## TL;DR
LoVeC trains LLMs to append a numerical `<confidence>` label (0–10) after each sentence during long-form generation. Using GRPO (online with an oracle fact-checker) or DPO (offline preference pairs), the model aligns these labels with GPT-4o-determined factuality. This allows for calibrated, machine-parsable confidence in a single decoding pass, outperforming the SOTA LUQ across Brier/ECE/Spearman metrics while being 20x faster at inference.

## Background & Motivation

**Background**: Mainstream hallucination detection for long-form QA falls into two categories: sampling-based consistency methods (e.g., LUQ, SelfCheckGPT, requiring multiple samples + similarity comparison) and atomic-claim-based scoring (Fadeeva 2024, Liu 2024). Both are post-processing steps and rely on external models, leading to high single-inference costs.

**Limitations of Prior Work**: ① Consistency methods require resampling 5–10 times per query; running the 792-item WildHallu test set on an A100 takes over 1500 seconds. ② Atomic-claim deconstruction requires GPT-4 API calls, incurring high cost and latency. ③ While verbalized confidence is cheaper, existing methods (e.g., LoGU, Linguistic Calibration) output natural language phrases like "I believe" or "70% uncertain," which are difficult for machines to parse or use with direct thresholds. ④ Existing verbalized confidence research focuses almost entirely on short-form QA, with no systematic study at the sentence-level in long-form generation.

**Key Challenge**: A paragraph in long-form text contains multiple factual statements, and confidence should vary per sentence. However, SFT only learns token-level likelihood and cannot jointly optimize "sentence content" and "confidence digits" as a combined action. Furthermore, SFT lacks feedback for negative examples, failing to learn asymmetric costs like "it is better to say 'I don't know' than to be confidently wrong."

**Goal**: Enable models to generate `<confidence> N </confidence>` numerical labels that align with factuality while writing sentences in a single decoding pass, ensuring robustness across in-domain (WildHallu) and out-of-domain (Bios / PopQA) tasks.

**Key Insight**: Treat "writing sentences + labeling confidence" as a sequential decision process. Use RL to perform credit assignment directly on the $(sentence, confidence)$ joint action—rewarding the degree of alignment between confidence and fact-checker factuality, while using a log-base reward to heavily penalize overconfident errors.

**Core Idea**: Jointly optimize $(s_i, c_i)$ using RL (GRPO + DPO) with a binary-cross-entropy log reward to produce both text and parsable numerical confidence in a single decoding pass.

## Method

### Overall Architecture

Given a query $q$, the policy $\pi_\theta$ outputs sentence-confidence pairs $y=\{(s_1,c_1),\dots,(s_n,c_n)\}$, where $c_i\in\{0,1,\dots,10\}$. Training involves two steps: (1) 1 epoch of SFT on winning samples $y_w$ to teach the model the `<confidence>N</confidence>` format; (2) 1 epoch of RL using GRPO (when an oracle fact-checker is available) or DPO (using offline preference pairs without an oracle), with LoRA fine-tuning on q/k/v/o_proj (< 1% of parameters). Two evaluation protocols are established: free-form tagging (model outputs answer and confidence simultaneously) and iterative tagging (model predicts confidence sentence-by-sentence for fixed text to allow fair comparison).

### Key Designs

1.  **GRPO + log-base calibration reward**:
    - **Function**: Represents confidence-factuality alignment as a differentiable group-relative advantage signal for online policy optimization.
    - **Mechanism**: Normalizing $c_i, f_i$ to $[0,1]$, the confidence reward is defined as $r_{\mathrm{conf}} = \lambda\cdot \frac{1}{n}\mathbf{1}^\top\left(1+\frac{f\odot\log c + (1-f)\odot\log(1-c)}{R_{\max}}\right)$, essentially the negative BCE, plus informativeness and format sub-rewards. For each query, $G$ trajectories $\{y_j\}_{j=1}^G$ are sampled to calculate group-mean normalized advantages $\hat A_j = \frac{r_j - \mathrm{mean}(r)}{\mathrm{std}(r)}$. The GRPO loss is $L_{\mathrm{GRPO}}(\theta) = -\mathbb{E}\big[\frac{1}{G}\sum_j(\hat A_j(\pi_\theta,\pi_{\mathrm{old}}) - \beta D_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}])\big]$. The reward is stretched using $\gamma=1.5$ via $r\leftarrow \mathrm{sign}(r)|r|^\gamma$ to amplify differences between samples.
    - **Design Motivation**: Unlike linear or quadratic losses, the log reward heavily penalizes "high confidence but factually wrong" outputs (approaching $-\infty$). It is a proper scoring rule that forces the model to calibrate rather than just learn ranking. Group-relative advantage removes the need for a separate critic, saving VRAM.

2.  **DPO + Synthetic Preference Pairs (Algorithm 1)**:
    - **Function**: Achieves the same goal through preference learning in offline scenarios without an online fact-checker.
    - **Mechanism**: For each query $(q,E)$, the base model generates plain text $y_{\mathrm{base}}=\{s_1,\dots,s_n\}$. Using GPT-4o + retrieved evidence, fact labels $f_j$ are calculated. A winning set $y_w=\{(s_j,f_j)\}$ (using ground truth as confidence) and a losing set $y_l=\{(s_j,c'_j)\}$ are constructed, where $c'_j$ is sampled uniformly from $\{0,\dots,10\}\setminus\{f_j\}$. This keeps the sentence identical while deviating the confidence digit from the truth. Optimization uses standard DPO loss: $L_{\mathrm{DPO}}=-\mathbb{E}\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|q)}{\pi_{\mathrm{SFT}}(y_w|q)} - \beta\log\frac{\pi_\theta(y_l|q)}{\pi_{\mathrm{SFT}}(y_l|q)}\big)$.
    - **Design Motivation**: DPO avoids expensive GPT-4o calls during RL training by concentrating oracle calls in the data construction phase. By keeping sentences identical and only perturbing confidence, the model focuses on learning "how to score" rather than "how to write."

3.  **Free-form vs. Iterative Tagging Dual Protocol**:
    - **Function**: Decouples the evaluation of "content generation quality" and "scoring quality."
    - **Mechanism**: Free-form allows the model to generate $y_t = \arg\max_{y_t}\pi_\theta(y_t|y_{<t},q)$, outputting the answer and `<confidence>` together. Iterative tagging fixes base-model-generated sentences $\{s_1,\dots,s_n\}$, and the model only predicts $c_i = \arg\max_c \pi_\theta(\{q,(s_1,c_1),\dots,(s_{i-1},c_{i-1}),s_i\},c)$ sentence-by-sentence.
    - **Design Motivation**: Since different verbalized methods generate different content, metrics like BS/ECE are hard to compare directly. Iterative tagging provides an apples-to-apples baseline on identical text, while free-form preserves the real-world usage scenario.

### Loss & Training
- SFT is done for one epoch on $y_w$ only for format; GRPO/DPO are followed by one epoch each of LoRA fine-tuning (q/k/v/o_proj), AdamW; Total 1500 GPU hours on 8x A100. GRPO reward stretching $\gamma=1.5$, with a $0.15\times \mathrm{correctness}$ small reward to prevent "always saying I don't know."
- Backbone: Llama-3-8B-Instruct and Gemma-2-9B-It. Evaluation metrics: Brier Score (BS), ECE-M (soft label version), and Spearman Correlation (SC) to cover both calibration and ranking.

## Key Experimental Results

### Main Results (Llama-3-8B-Instruct, iterative + free-form)

| Dataset | Method | BS↓ | ECE-M↓ | SC↑ |
|---|---|---|---|---|
| WildHallu | LUQ (prev SOTA) | 14.5 | 21.5 | 56.8 |
| WildHallu | LoVeC-GRPO (iter) | **5.7** | **2.5** | 57.0 |
| WildHallu | LoVeC-DPO (iter) | 6.0 | 5.0 | **60.4** |
| Bios | LUQ | 20.0 | 29.5 | 63.8 |
| Bios | LoVeC-GRPO (iter) | **8.5** | **4.2** | 64.7 |
| Bios | LoVeC-DPO (iter) | 9.0 | 7.3 | **65.6** |
| PopQA | LUQ | 16.7 | 23.2 | 62.5 |
| PopQA | LoVeC-DPO (iter) | **9.6** | **1.7** | **63.1** |

BS and ECE-M were roughly halved across three datasets, while Spearman Correlation also saw slight gains. Free-form trends matched iterative (GRPO BS 5.7–10.1, ECE-M 5.1–11.1). Inference time for 792 WildHallu samples: LUQ took 1525s vs. LoVeC-iterative 64s (**~24× speedup**) and LoVeC-freeform 139s (~11× speedup).

### Ablation Study

| Configuration | BS↓ | ECE-M↓ | SC↑ | Note |
|---|---|---|---|---|
| LoVeC-GRPO Full (WildHallu) | 5.7 | 2.5 | 57.0 | Base |
| Using Log Reward | 5.7 | 2.5 | 57.0 | Proper scoring rule |
| Using Linear/Quadratic Reward | ↑ | ↑ | ↓ | Calibration degraded significantly |
| DPO using GPT-4o Oracle | 6.0 | 5.0 | 60.4 | Default |
| DPO using Self-label (frozen self) | Slightly worse | Slightly worse | Slightly worse | Still outperforms LUQ baseline |
| SFT using Regression Loss vs. CE | ↑ | ↑ | ↓ | All metrics worsened |
| Iterative without seeing previous scores | ↑ | ↑ | ↓ | Removing "local calibration anchors" drops score |

GRPO training dynamics: Mean reward 13.86 → 29.83 (5667 steps / 1 epoch), ECE-M dropped from 15.2 to 2.5 simultaneously without reward collapse.

### Key Findings
- The critical advantage of RL over SFT is not the surface score but the token ranking structure: in GRPO next-token prediction, top-15 candidates are monotonic `10, 9, 8, ..., 0` (when factually correct) or `2, 3, 4, ...` (ordered around the true score when incorrect). DPO is partially ordered, while SFT is completely unordered—this "probability distribution reflecting the confidence ladder" is the inductive bias RL truly injects.
- Removing the expensive oracle (using self-label DPO) still beats LUQ, suggesting the method is not overly sensitive to oracle strength and can be deployed in industrial scenarios without GPT-4o.
- Complementary to LUQ: Simple averaging of LoVeC-DPO + LUQ scores gains another +5 points in Spearman, indicating that verbalized signals and sampling signals act as orthogonal evidence.
- Zero-shot transfer to short-form TriviaQA remains competitive, approaching RewardingDoubt (designed specifically for short-form), showing RL learns a general skill of "reflecting likelihood as digits."

## Highlights & Insights
- Redefining "long-form confidence" as "sentence-level digits + RL joint optimization" avoids GPT calls for atomic-claim extraction and multiple decodings for consistency sampling. It is the most practical form for engineering.
- Log reward turns calibration into proper scoring, and the combination of reward stretching and small correctness bonuses ("asymmetric + anti-hacking" reward engineering) is a valuable reference for those working on RLHF calibration.
- The DPO preference pair construction keeps the sentence identical and only varies confidence. This "controlled variable" approach ensures the model learns the scoring task without polluting its linguistic capabilities.
- The dual protocol evaluation decouples content quality from scoring quality, providing a template for evaluating trustworthy AI in long-form contexts—subsequent verbalized confidence work should include the iterative tagging benchmark.

## Limitations & Future Work
- Only applicable to white-box models (requires LoRA + RL); cannot be deployed on pure API models like OpenAI or Anthropic.
- Evaluation is limited to factuality; it does not cover other calibratable dimensions like consistency, harmlessness, or creative writing.
- Sentence-level granularity still lacks precision for sentences containing multiple contradictory facts; future work could push to sub-sentence or atomic-claim levels.
- Training rewards depend on GPT-4o fact-checking, inheriting its factual judgment biases; self-labeling is feasible but remains slightly weaker.
- Not yet verified on other long-form tasks like code generation or translation; real-world performance in high-risk domains (medical/legal) is unknown.

## Related Work & Insights
- **vs LUQ (Zhang 2024a)**: LUQ relies on 5–10 samples + sentence-level consistency; this work is verbalized in a single decoding pass. LoVeC moves the judgment burden to the training phase, resulting in zero additional inference overhead at the same granularity.
- **vs LoGU (Yang 2025a) / Linguistic Calibration (Band 2024)**: They use RL for phrases like "I believe" or "I'm uncertain," which are hard to parse. LoVeC uses `0–10` tags for machine interpretability, enabling direct thresholds or ranking.
- **vs RewardingDoubt / SaySelf**: These are RL calibration methods but limited to short-form. LoVeC is the first to achieve RL calibration for long-form, providing both GRPO and DPO solutions.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic RL solution for sentence-level verbalized confidence in long-form text; original dual protocol evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + Llama/Gemma + multiple RL algorithms + reward forms + oracle types + 6+ ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Detailed motivation; case studies using token ranking to prove the internalization effects of RL are very vivid.
- Value: ⭐⭐⭐⭐ 20× speedup + significant calibration improvement is highly attractive for production; reward design and preference pair paradigms are reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ICML 2026\] CAMEL: Confidence-Gated Reflection for Reward Modeling](../../ICML2026/reinforcement_learning/camel_confidence-gated_reflection_for_reward_modeling.md)
- [\[ACL 2026\] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)

</div>

<!-- RELATED:END -->
