---
title: >-
  [Paper Note] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL
description: >-
  [ICML 2026][AI Safety][Forgery Detection] Addressing the unified task of "simultaneous detection and localization of mixed image/text/video forgeries…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Forgery Detection"
  - "Grounding"
  - "Multi-task RL"
  - "Reward Shaping"
  - "Self-evolving CoT"
date: 2026-05-08
content_hash: 9829aa225c58a746
---

# OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL

**Conference**: ICML 2026  
**arXiv**: [2602.10687](https://arxiv.org/abs/2602.10687)  
**Code**: https://github.com/shen8424/OmniVL-Guard (Available)  
**Area**: AI Security / Multimodal Forgery Detection / Reinforcement Learning / VLM  
**Keywords**: Forgery Detection, Grounding, Multi-task RL, Reward Shaping, Self-evolving CoT

## TL;DR
Addressing the unified task of "simultaneous detection and localization of mixed image/text/video forgeries," this paper proposes OmniVL-Guard. It utilizes Self-Evolving CoT to synthesize high-quality cold-start data and introduces ARSPO (nonlinear reward mapping + dynamic task weighting) to solve the "difficulty bias" in multi-task RL—where simple binary classification dominates gradients while fine-grained grounding fails to learn. In-domain results show significant gains (video temporal localization tIoU +37.8, text localization F1 +22.9), and zero-shot SOTA is achieved across four OOD benchmarks.

## Background & Motivation

**Background**: Most current forgery detection and tampering localization works are unimodal (pure image, text, or video) or at most bimodal (image-text, video-text), with each modality requiring its own set of expert models. Representative methods like HAMMER, FKA-Owl, Fake-VLM, and FakeSV-VLM can only "handle one front."

**Limitations of Prior Work**: False information on real social media consists of "all-modal" content where images, text, and video are highly intertwined. Faced with such mixed inputs, unimodal or bimodal detectors either cannot process them or fail to provide both "veracity judgment + forgery localization." Thus, a unified framework covering binary classification and grounding for all three modalities is required. However, the authors found that directly using general MLLMs (GPT-5/Gemini3/Seed1.6) in a zero-shot manner only reaches ~73% for classification while grounding performance for all three tasks collapses to 20-35 (Table 1a). Direct SFT also fails to generalize across modalities due to insufficient reasoning capabilities.

**Key Challenge**: A natural choice is to introduce RL (such as GRPO) to let MLLMs explore reasoning paths. However, the authors observed a "difficulty bias" phenomenon via experiments and theory: binary classification is a discriminative task with strong, easily climbed reward signals, while image/text/video localization are regression/interval tasks requiring fine-grained perception with sparse rewards. When GRPO optimizes all tasks together with equal weighting, classification improves by +36% while image localization performance actually regresses by -0.1%—simple tasks "kidnap" the gradient update direction. Improvements like SAPO are slightly better but still struggle with grounding.

**Goal**: Decomposition into two sub-problems: (1) How to generate high-quality CoT cold-start data for these fine-grained multimodal reasoning tasks; (2) How to design a new RL objective so that simple tasks do not monopolize resources while hard tasks continue to benefit.

**Key Insight**: The authors performed a second-order expansion of the GRPO-like objective with respect to parameter $\theta$ (Equation 4), splitting the gradient rate of change into two terms: "reward mapping sensitivity $g_k'(\cdot)$" and "task difficulty sensitivity $H_k'(\theta,q,\tau)$." Hard tasks naturally have very small $H_k'$ because they stay on a performance plateau. Thus, even if rewards are normalized to the same scale, simple tasks still dominate "gradient acceleration." This analytical expression directly points to a remedy: using a **convex nonlinear reward mapping function** $g_k(\cdot)$ that becomes steeper as performance increases, thereby amplifying the gradient contribution of high-score responses to counteract the decay caused by small $H_k'$.

**Core Idea**: Replace uniform weighting in RL with "nonlinear reward shaping + dynamic task weights," allowing simple tasks to converge as needed and hard tasks to receive adequate weighting. Furthermore, use Self-Evolving CoT to synthesize cold-start data that truly solves problems (rather than reasoning backwards from answers), avoiding the hindsight bias introduced by GT-injected distillation.

## Method

### Overall Architecture
The input is any image, text, video, or combination thereof. The model must simultaneously output (1) binary veracity classification and (2) the forged regions in the corresponding modality—image spatial mask (IoU), text token span (F1), and video temporal interval (tIoU). The entire pipeline follows four steps: "FSFR Dataset -> SFT cold-start on Qwen3VL-8B -> ARSPO multi-task RL -> In-Domain and OOD Testing." FSFR is constructed offline via Self-Evolving CoT Generation, containing 73k high-quality CoT samples for SFT and 110k for RL training. ARSPO operates during the RL phase, dynamically adjusting reward curves and task weights for all four tasks (Classification / Image Grounding / Text Grounding / Video Grounding) according to training steps.

### Key Designs

1.  **Self-Evolving CoT Generation (FSFR Dataset Construction)**:
    - **Function**: Create a high-quality CoT cold-start dataset $\text{FSFR}_{\text{sft}}$ for fine-grained forgery detection and grounding that "can solve tasks without leaking answers," with a corresponding $\text{FSFR}_{\text{rl}}$ reserved for RL.
    - **Mechanism**: The authors formalize CoT generation as an *Efficiency-Bias Dilemma*—direct generation by closed-source MLLMs is low quality, while GT injection leads to hindsight bias where the model reasons backwards from the answer, undermining RL exploration. The solution is a four-stage self-evolution: (a) Aggregate and proportionally split $D_s/D_r/D_t$ from public data pools (FakeNewsCorpus, ForgeryNet, GenVideo, DGM4, etc.); (b) Use a set of SOTA MLLMs $\mathcal{M}=\{\text{Seed1.6-VL, Gemini3, ChatGPT5}\}$ to perform inference on a small subset $D_s$, obtaining a 6.7k seed set $D_s^0$ via "GT filtering + cross-consistency verification" to get a warm-up policy $\pi_0$ via SFT+RL; (c) In the $k$-th round, use $\pi_{k-1}$ to generate CoT for remaining samples, followed by GT filtering and SOTA MLLM verification to merge into $D_s^k$, **retraining from the base Qwen3VL-8B each round** to avoid distribution drift; (d) For consistently wrong hard samples, employ Multi-Agent Collaborative Hard-CoT Synthesis—one MLLM generates CoT with GT, a second MLLM acts as a "Refiner" to transform traces into "natural reasoning as if the answer is unknown," and a third MLLM filters by score. Saturation is reached after three rounds of evolution (Table 5: $D_s^4$ shows almost no gain over $D_s^3$).
    - **Design Motivation**: Direct distillation is insufficient (general MLLMs lack forensic detail) and giving answers directly for backward reasoning ruins RL. Self-evolution uses "the model passing its own test" as a quality proxy and uses third-party MLLMs as judges to decouple the "correct reasoning" signal from the "correct answer" signal, circumventing hindsight bias.

2.  **ARSPO – Task-Based Reward Mapping Function (TBRMF)**:
    - **Function**: Explicitly adjust the gradient contribution of each task via task-customized nonlinear reward mapping, preventing simple tasks from hogging resources and ensuring continuous learning for hard tasks.
    - **Mechanism**: Based on the second-order gradient expansion in Section 4.1 (core formula: $\frac{d}{d\theta}(W_{i,t}(\theta)\hat{A}_{i,k}) = W'_{i,t}(\theta)\hat{A}_{i,k} + W_{i,t}(\theta)\cdot \frac{g_k'(H_k)}{G\sigma}[(G-1)-\hat{A}_{i,k}^2]\,H_k'(\theta,q,\tau)$), the authors found that the reward function $A_{i,k}=g_k(x_{i,k})$ can be mapped from raw performance metrics $x_{i,k}$ via $g_k$. For binary classification (easy), an identity mapping $g_k(x)=x$ is used to avoid unnecessary amplification. For the three fine-grained grounding tasks, a convex function $g_k(x)=e^{a_k x}$ is chosen (with $a=3$ found optimal via grid search in Figure 4). This results in a steeper slope in high-performance regions, significantly amplifying the gradients of high-scoring responses within a group—turning "near-correct" samples into the strongest learning signals.
    - **Design Motivation**: Normalization alone (GRPO/SAPO) only makes reward scales comparable; it cannot change "task difficulty sensitivity $H_k'$." Convex mapping effectively increases the marginal utility of high-score samples on the reward side, equivalent to compensating for the decay of $H_k'$. This also explains why the exponential mapping significantly outperforms linear mapping even in single-task training (Section 5.3): the essence of ARSPO is "gradient signal reshaping" rather than just "balancing."

3.  **ARSPO – Dynamic Coefficient Adjustment (DCA, Algorithm 1)**:
    - **Function**: Periodically monitor the relative learning status of each task during training and adaptively adjust the global weights $l_{k,s}$ for the four tasks to avoid the "bucket effect."
    - **Mechanism**: A warm-up phase ($s<T_{warm}$) records the mean performance of each task as a frozen baseline $B_k$. Subsequently, two quantities are evaluated every $T$ steps—overall relative gain $\Delta_{\text{total},k}=(\mu_k-B_k)/B_k$ (measuring "long-term progress") and recent change $\delta_{\text{recent}}=\mu_k-\mu_{\text{past}}$ (measuring "short-term trends"). Task weights $l_{k,s}$ are adjusted based on four levels of priority: "momentum protection (no change during growth) -> regression rescue (multiply by $\alpha_{\text{boost}}$ if performance regresses) -> high-performance decay (slowly exit by multiplying by $\alpha_{\text{decay}}$ once the task is mastered, down to a floor of 1) -> laggard support (identify $k_{\text{lag}}=\arg\min_k\Delta_{\text{total},k}$ and amplify weight up to 4)." Finally, weights are rescaled by dividing by the minimum coefficient before updating parameters via $\nabla_\theta \mathcal{J}_{\text{arspo}}$.
    - **Design Motivation**: TBRMF is "static reward curve shaping," but training dynamics change over time—a task might be lagging initially but catch up later, or regress after mastering. DCA adds a closed-loop controller to the system, tilting weight resources toward currently stalled tasks according to the "weakest-first" principle, providing a dual guarantee of "static shape + dynamic weights" alongside TBRMF.

### Loss & Training
The RL objective incorporates the dynamic coefficient $l_{k,s}$ into the GRPO framework:
$$\mathcal{J}_{\text{arspo}}(\theta)=\sum_{k=1}^{K}\frac{|\mathcal{D}_k|}{|\mathcal{D}|}\mathbb{E}_{q\sim\mathcal{D}_k,\{y_i\}\sim\pi_{\theta_{\text{old}}}}\left[\frac{l_{k,s}}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}f_{i,t}(r_{i,t}(\theta))\hat{A}_{i,k}\right]$$
where the advantage $\hat{A}_{i,k}=(A_{i,k}-\mu)/\sigma$ is still normalized within the group, but $A_{i,k}=g_k(x_{i,k})$ undergoes task-customized nonlinear mapping. The base model is Qwen3VL-8B, first cold-started with $\text{FSFR}_{\text{sft}}$ via SFT, then trained with $\text{FSFR}_{\text{rl}}$ using ARSPO. During warm-up, all $l_{k,s}=1$ to collect baselines.

## Key Experimental Results

### Main Results
In-Domain (Ours $D_t$ test set, ~700k samples):

| Dataset / Task | Metric | Ours | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Text Binary | ACC | 96.20 | 89.23 (Qwen3VL-235B) | +6.97 |
| Image Binary | ACC | 93.12 | 90.39 (Fake-VLM) | +2.73 |
| Video Binary | ACC | 98.58 | 98.81 (FakeSV-VLM) | -0.23 |
| Text-Image Binary | ACC | 75.52 | 72.08 (FKA-Owl) | +3.44 |
| Image Grounding | IoU | 54.26 | 48.53 (HAMMER) | +5.73 |
| Text Grounding | F1 | 63.78 | 40.86 (HAMMER) | +22.92 |
| Video Grounding | tIoU | 59.22 | 21.43 (Qwen3VL-235B) | +37.79 |

OOD Zero-shot (without any fine-tuning): ISOT Text 93.69 (vs 88.74), CASIA2.0 Image 63.64 (vs 60.88), MMFakeBench Text-Image 79.38 (vs 62.32), FakeSV Text-Video 63.55 (vs 61.22)—leading in all four benchmarks.

### Ablation Study

| Configuration | Img-Loc IoU | Text-Loc F1 | Vid-Loc tIoU | $\Delta$ AVG |
| :--- | :--- | :--- | :--- | :--- |
| SFT only | 51.08 | 44.67 | 33.08 | — |
| SFT + SAPO | 51.24 | 54.33 | 44.10 | +24.33 |
| + TBRMF | 53.21 | 61.37 | 49.38 | +26.42 |
| + DCA | 52.95 | 59.88 | 53.49 | +26.93 |
| Full（SFT+SAPO+TBRMF+DCA） | **54.26** | **63.78** | **59.22** | **+28.33** |

### Key Findings
- **The "gradient reshaping" effect of TBRMF is the strongest signal**: Under single-task settings (no inter-task resource competition), exponential mapping still outperforms linear mapping by +4% in image grounding and +8% in text grounding. this proves ARSPO's value is not just "balancing multi-tasks" but "amplifying gradients of high-score samples"—directly corresponding to the theoretical analysis of $g_k'(\cdot)$ in Section 4.1.
- **Reward curvature beyond $a=3$ leads to regression**: Figure 4(a-b) shows performance drops when $a$ is too large, attributed to "reward overfitting"—too steep a mapping treats aleatoric noise as signals, causing the model to get stuck in local optima; $a=3$ is the sweet spot for signal amplification and training stability.
- **Self-evolution saturates at round 3**: Table 5 shows $D_s^4$ vs $D_s^3$ metrics change by less than 0.2, justifying stopping at $k=3$ and saving significant inference compute.
- **Difficulty bias in GRPO/SAPO was directly captured**: In Table 1(b), SFT+GRPO caused classification to surge by +36% while Image-Loc regressed by -0.1%—a direct case study of "simple tasks kidnapping gradients."

## Highlights & Insights
- **Deriving the root cause of "Difficulty Bias" from second-order expansion**: The authors did not stop at the empirical observation that GRPO favors simple tasks; they precisely decomposed the gradient rate of change into "reward sensitivity $g_k'$ × difficulty sensitivity $H_k'$." This reduces the problem of "how to fix RL" to "how to choose $g_k$," rooting the exponential mapping of TBRMF in theory rather than pure hyperparameter tuning. This logic can migrate to any multi-task GRPO scenario with difficulty imbalance.
- **Hindsight-Bias-Free CoT synthesis is highly reusable**: While feeding GT to models for reasoning generation is the default in CoT distillation works, this paper explicitly points out that this makes models learn "backtracking" instead of "deduction." The solution using a "Refiner MLLM to hide GT traces + Third-party MLLM judge" decouples this bias. This approach is directly applicable to any task requiring "process supervision" (e.g., math reasoning, theorem proving).
- **The DCA "Bucket Effect" controller is a lightweight and useful tool**: It adaptively adjusts task weights with four priority levels and a set of thresholds without backpropagation overhead, serving as a plug-and-play module for any multi-task RL training.

## Limitations & Future Work
- Limitations recognized by the authors are few, mainly focusing on "OOD room for improvement" and the model size (8B). However, real limitations include: (1) The pipeline relies on three closed-source SOTA MLLMs as judges/generators/refiners, making it expensive to reproduce with open-source alternatives; (2) Self-evolution requires retraining from the base model each round to avoid drift, totaling four full SFT+RL cycles with high training costs; (3) TBRMF requires manual $a_k$ tuning; although $a=3$ was searched for grounding, whether finer sub-tasks (e.g., forgery types) need different curvatures was not discussed.
- An interesting future direction is replacing the heuristic priorities of DCA with a bandit or meta-learning-based controller, making weight adjustment itself optimizable. Furthermore, $a_k$ in TBRMF could be made online-adaptive based on the current reward distribution quantiles.

## Related Work & Insights
- **vs HAMMER / FKA-Owl / AMD (Image-Text Experts)**: These focus on image-text bimodal tasks using specialized heads for localization. Ours handles image/text/video unification and transforms all grounding into text output (coordinates/spans/intervals), offering zero-shot cross-modal generalization at the cost of higher inference latency than specialists.
- **vs DeepSeek-R1 / GRPO / SAPO**: This work adopts the "SFT cold-start + RL" paradigm from DeepSeek-R1 but identifies the difficulty bias in multi-task GRPO/SAPO. ARSPO serves as an important patch for GRPO in unbalanced difficulty scenarios.
- **vs Fake-VLM / FakeSV-VLM**: Unimodal experts are strong on their specific tasks (Fake-VLM 90.39 for images, FakeSV-VLM 98.81 for video). Ours matches or slightly lags on individual modalities (e.g., Video Binary -0.23) but leads significantly in cross-modal and cross-dataset scenarios, demonstrating the marginal value of a "unified model."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Precision attribution of "difficulty bias" via second-order formulas; well-founded solutions via TBRMF + DCA; interesting "Refiner" approach for hindsight-free CoT.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 modalities × 7 metrics + 4 OOD benchmarks; thorough ablation of SAPO/TBRMF/DCA combinations; secondary validation of TBRMF in single-task settings.
- **Writing Quality**: ⭐⭐⭐⭐ Clear chain of motivation—theory—algorithm; full derivations. Algorithm 1 in Section 5.3 requires jumping to the Appendix for thresholds, slightly affecting readability.
- **Value**: ⭐⭐⭐⭐ Provides a deployable unified solution for the practical task of all-modal forgery detection; ARSPO's solution for difficulty bias is reusable for any multi-task GRPO training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniVid: A Unified Vision-Language Model for Video Moderation](../../ACL2026/ai_safety/univid_unified_vision-language_model_for_video_moderation.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[ICML 2026\] The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)

</div>

<!-- RELATED:END -->
