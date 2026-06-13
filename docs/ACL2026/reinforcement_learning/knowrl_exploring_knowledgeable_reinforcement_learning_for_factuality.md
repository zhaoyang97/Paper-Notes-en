---
title: >-
  [Paper Note] KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality
description: >-
  [ACL 2026][Reinforcement Learning][Process-level reward] KnowRL integrates "atomic fact verification" as a process-level reward directly into the GRPO training loop…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Process-level reward"
  - "atomic fact verification"
  - "GRPO"
  - "knowledge boundary"
  - "refusal incentive"
date: 2026-05-08
content_hash: 11165096806ed383
---

# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality

**Conference**: ACL 2026  
**arXiv**: [2506.19807](https://arxiv.org/abs/2506.19807)  
**Code**: https://github.com/zjunlp/KnowRL  
**Area**: Reinforcement Learning / Slow-thinking LLM / Factuality / Hallucination Mitigation  
**Keywords**: Process-level reward, atomic fact verification, GRPO, knowledge boundary, refusal incentive

## TL;DR
KnowRL integrates "atomic fact verification" as a process-level reward directly into the GRPO training loop, performing factual assessment for each step of the slow-thinking model's Chain-of-Thought (CoT). Simultaneously, a "positive reward for refusal" is employed to teach the model to recognize its knowledge boundaries. This approach reduces the SimpleQA Incorrect Rate by 20.3% without loss (and with slight gains) in reasoning benchmarks like GPQA/AIME, while demonstrating cross-lingual transfer from English knowledge to Chinese QA.

## Background & Motivation

**Background**: "Slow-thinking" models like DeepSeek-R1 utilize RL to encourage long CoT, achieving significant breakthroughs on strong reasoning benchmarks like GPQA and AIME. However, they exhibit inverse scaling on factuality benchmarks like SimpleQA—DeepSeek-R1-Distill-Qwen-32B achieves only 6.64% accuracy on SimpleQA, with hallucinations increasing as model scale grows.

**Limitations of Prior Work**: Existing RL training almost exclusively utilizes outcome-only rewards (evaluating only the final answer), treating the reasoning process as a black box. This presents two critical issues: (i) models may arrive at the correct answer through fabricated reasoning paths, which the reward signal reinforces; (ii) models lack awareness of their own ignorance and guess to obtain rewards, indicating a missing knowledge boundary.

**Key Challenge**: To enable slow-thinking models to reason while maintaining factuality, one must simultaneously implement factual supervision of the reasoning process and teach the model to refuse to answer when appropriate. However, RAG incurs prohibitive retrieval costs in long CoT, SFT leads to forgetting or rote memorization, and preference optimization like DPO only adjusts at the outcome level, failing to directly address process-level hallucinations.

**Goal**: (1) Embed factual verification into the RL training loop for step-level supervision; (2) avoid compromising existing complex reasoning capabilities; (3) teach the model to say "I don't know" when information is missing, ensuring this "boundary awareness" can transfer across tasks and languages.

**Key Insight**: Drawing on the FactScore paradigm, which decomposes long-form generations into atomic facts for verification against a knowledge base, the authors utilize the "atomic fact pass rate" as a dense reward signal for RL. Furthermore, an asymmetric correctness reward structure ($+2$ for correct / $+1$ for refusal / $-1$ for incorrect) is designed to explicitly incentive "refusal."

**Core Idea**: Use GRPO with a three-part composite reward (format + correctness with refusal bonus + atomic-fact factuality) to open the "thinking blackbox" of slow-moving models, shifting probability mass from hallucinated paths to knowledge-supported paths.

## Method

### Overall Architecture
KnowRL performs reward engineering on top of GRPO: (1) **Data Construction**—Extraction of factual questions from NQ-Open / WebQuestions / ComplexQuestions, entity extraction via GPT-4o, and retrieval of relevant factual passages from a Wikipedia 2023-11-01 dump as the external knowledge base $K$; (2) **Rollout & Composite Reward**—For each prompt $x$, the old policy $\pi_{\theta_{\text{old}}}$ samples $G$ rollouts, and for each rollout $o=(o_{\text{think}},o_{\text{answer}})$, the total reward is calculated as $R_{\text{total}}=r_{\text{format}}+r_{\text{correct}}+r_{\text{fact}}$; (3) **GRPO Advantage Normalization**—Advantages within the same prompt group are normalized as $A_g=(R_g-\mu_x)/(\sigma_x+\varepsilon)$; (4) **Surrogate Objective** + KL anchor + entropy regularization for PPO-style updates; (5) LoRA (rank 128) fine-tuning throughout, allowing 7B/14B models to be trained on a single A800.

### Key Designs

1.  **Process-level Reward $r_{\text{fact}}$ via Atomic Fact Decomposition + GTR Retrieval**:
    - **Function**: Converts whether each step in the CoT is "supported by evidence" into a dense, continuous reward signal to directly supervise the reasoning process.
    - **Mechanism**: A GPT-4o-mini prompt decomposes $o_{\text{think}}$ into $M$ atomic facts $\Phi(o_{\text{think}})=\{f_1,\dots,f_M\}$. For each $f_j$, the sentence-transformers model `gtr-t5-large` retrieves top-relevant passages $K_x$ from the knowledge base $K$. GPT-4o-mini then provides a 0/1 verification $v(f_j,K_x)$. The reward is defined as $r_{\text{fact}}(o)=\frac{1}{M}\sum_j v(f_j,K_x)$, or 0 if $M=0$.
    - **Design Motivation**: FactScore has proven that "atomic fact decomposition + retrieval + binary verification" is a reliable pipeline for measuring long-form factuality. Moving this from evaluation to training transforms "factuality" from a black-box scalar into a dense score that can drive gradient updates.

2.  **Asymmetric Correctness Reward + Refusal Positive Incentive**:
    - **Function**: Explicitly distinguishes "correct / refuse / wrong" within the correctness reward. The $+2/+1/-1$ design teaches the model to actively state "I don't know" when faced with unknown questions.
    - **Mechanism**: $r_{\text{correct}}=+2$ (Correct via GPT-4o-mini decision) / $+1$ (Explicit refusal) / $-1$ (Incorrect). A format reward of $\pm 1$ enforces the `<think>...</think><answer>...</answer>` structure. Removing the refusal bonus (changing it to $-1$) causes the Incorrect Rate to bounce back from 57.67% to 78.67%, demonstrating its essential role.
    - **Design Motivation**: Traditional RL penalizes all "non-correct" answers equally, forcing models to "gamble" when uncertain. The asymmetric structure informs the model that "refusal is better than being wrong," thereby shaping the knowledge boundary.

3.  **GRPO-based Process-level Advantage Aggregation**:
    - **Function**: Normalizes $R_{\text{total}}$ from $G$ rollouts within a group into signed advantages $A_g$, providing positive credit to trajectories with "high factuality + appropriate refusal" and negative credit to those dominated by hallucinations.
    - **Mechanism**: $A_g=(R_g-\mu_x)/(\sigma_x+\varepsilon)$. The trajectory-level importance ratio is $\varrho_g=\pi_\theta(o^{(g)}|x)/\pi_{\theta_{\text{old}}}(o^{(g)}|x)$. After clipping, the surrogate is $\hat{\mathcal{J}}(\theta)=\frac{1}{G}\sum_g \min(\varrho_g A_g, \text{clip}(\varrho_g,1{-}\epsilon,1{+}\epsilon)A_g)$. Adding entropy bonus and KL anchor forms $\mathcal{L}_{\text{KnowRL}}=-\hat{\mathcal{J}}+\beta_{\mathcal{H}}\mathcal{E}_{\mathcal{H}}+\beta_{\text{KL}}\mathcal{E}_{\text{KL}}$.
    - **Design Motivation**: GRPO is memory-friendly as it requires no critic. Intra-group normalization stabilizes reward magnitudes across different prompt difficulties, and the KL anchor prevents the loss of reasoning ability while pursuing factuality.

### Loss & Training
LoRA rank 128 / alpha 256, bf16, lr=1e-5, batch=20, grad accum=4, KL coefficient $\beta_{\text{KL}}\approx 1\text{e-3}$, cosine LR schedule, 0.03 warmup, AdamW-8bit, vLLM GPU memory utility 0.5. Training a 7B model requires 1×A800. Convergence is reached in 100-300 steps; exceeding 300 steps leads to slight over-optimization.

## Key Experimental Results

### Main Results
Two 7B slow-thinking models (distilled DeepSeek-R1-Distill-Qwen-7B and RL-based Skywork-OR1-7B-Preview) are evaluated on SimpleQA Incorrect Rate and GPQA Diamond:

| Model | Method | SimpleQA Incorrect ↓ | ChineseSimpleQA Incorrect ↓ | GPQA Diamond ↑ | AIME 2025 ↑ |
|------|------|----------------------|------------------------------|----------------|-------------|
| DeepSeek-7B | Zero-shot | 78.00 | 68.33 | 40.91 | 30.00 |
| DeepSeek-7B | SFT | 83.33 (+5.33) | 76.67 (+8.34) | 36.36 | 26.67 |
| DeepSeek-7B | DPO | 88.00 (+10.0) | 79.33 (+11.0) | 37.37 | 30.00 |
| DeepSeek-7B | FactTune-FS | 59.67 (−18.3) | 76.00 (+7.67) | 38.89 | 30.00 |
| DeepSeek-7B | TruthRL | 61.00 (−17.0) | 60.00 (−8.33) | 39.39 | 26.67 |
| DeepSeek-7B | **Ours** | **57.67 (−20.3)** | **58.33 (−10.0)** | 36.87 | **33.33** |
| Skywork-7B | Zero-shot | 76.33 | 67.00 | 37.37 | 26.67 |
| Skywork-7B | **Ours** | **60.33 (−16.0)** | **52.33 (−14.7)** | **42.42** | **36.67** |

Under multiple runs (Avg@5, T=0.6), KnowRL reduces DeepSeek-7B's SimpleQA Incorrect from 62.47 to 48.27 and increases AIME from 29.33 to 34.00. On the 14B model, SimpleQA Incorrect drops from 83 to 68, while GPQA increases from 47 to 51.

### Ablation Study
Different reward combinations on DeepSeek-7B (SimpleQA Incorrect / GPQA):

| Reward Config | SimpleQA Incorrect ↓ | Refusal | GPQA ↑ | AIME ↑ |
|----------|----------------------|---------|--------|--------|
| $r_{\text{format}}$ only | 74.00 | 24.00 | 39.39 | 30.00 |
| $r_{\text{format}}+r_{\text{fact}}$ | 80.67 (+6.67) | 17.33 | **47.47** | **40.00** |
| $r_{\text{format}}+r_{\text{correct}}$ | 60.67 (−13.3) | 37.33 | 38.89 | 40.00 |
| **Full $R_{\text{total}}$ (KnowRL)** | **57.67 (−16.3)** | 40.67 | 36.87 | 33.33 |
| KnowRL with $r_{\text{refusal}}=-1$ | 78.67 (+4.67) | 8.67 | 34.85 | 30.00 |

Cross-algorithm robustness: Replacing GRPO with DAPO / BNPO / Dr.GRPO still results in a 16-19 point reduction in SimpleQA Incorrect, proving the effectiveness stems from reward design rather than a specific optimizer.

### Key Findings
- **Positive refusal incentive is the "anchor" of knowledge boundaries**: Changing $r_{\text{refusal}}$ from $+1$ to $-1$ causes the refusal rate to crash from 40.67% to 8.67% and the Incorrect Rate to spike from 57.67% to 78.67%. This indicates that the $+2/-1$ correctness reward alone is insufficient to stabilize boundary behavior.
- **$r_{\text{fact}}$ used in isolation benefits reasoning but does not suppress hallucinations**: The $r_{\text{format}}+r_{\text{fact}}$ configuration increases GPQA to 47.47 and AIME to 40.00, but the Incorrect Rate is worse than the baseline (80.67 vs 74.00). Without the correctness signal, the model is emboldened to "hallucinate confidently." This comparison elegantly demonstrates that the three rewards are complementary.
- **Cross-lingual transfer**: Although the training knowledge base is almost entirely English, the ChineseSimpleQA Incorrect rate also drops by 10-15 points, suggesting the model learns "language-agnostic verification behavior" rather than rote memorization.
- **Significant decrease in completion length ≠ Collapse of reasoning**: The sharp drop in generation length during training is a byproduct of the model learning to "stop fabricating stories for unknown questions." GPQA / AIME / OlympiadBench performance is maintained or slightly improved.
- **Optimal training steps ≈ 200**: Factual metrics improve rapidly and stabilize after 100-200 steps; over-optimization may occur after 300 steps.

## Highlights & Insights
- Reimagining FactScore from an "evaluation tool" into a "reward factory" is a simple yet high-impact paradigm shift; this dense factual reward can be plugged into almost any existing RL framework.
- The asymmetric refusal reward ($+2/+1/-1$) is elegant: it treats "honesty" as an independent action type for reinforcement, rather than relying on output entropy or external rejection heads, providing the most direct signal for learning boundaries.
- Equivalent performance across different algorithms (GRPO/DAPO/BNPO/Dr.GRPO) highlights that "reward design → behavior shaping" is paramount, suggesting the community should prioritize reward engineering over RL algorithm selection.
- The English-to-Chinese transfer is an unexpected but strong finding: it implies that verification behavior may be a language-independent "meta-capability."

## Limitations & Future Work
- Each rollout requires calling GPT-4o-mini for atomic fact decomposition and verification, coupling training cost and latency to external APIs. Internalizing the verifier as a local small model would be more engineering-friendly.
- Narrow knowledge base coverage (NQ/WebQ/Wiki 2023-11) leaves blind spots for open-ended or recent facts; models might learn to "refuse whenever asked this way" rather than truly understanding their boundaries.
- Evaluation samples were limited to 300 per set due to computational constraints, and AIME only had 30 questions, leading to higher variance (partially mitigated by Avg@5).
- Text-only focus; decomposition of "atomic facts" in charts, formulas, or physical diagrams remains unexplored.
- Semantic determination of refusal also relies on LLM evaluators, which might misinterpret conservative answers as refusals.

## Related Work & Insights
- **vs TruthRL**: TruthRL also uses "honesty vs truth" rewards but remains outcome-level; KnowRL provides finer process-level + atomic fact signals, slightly outperforming it on DeepSeek-7B (Incorrect 57.67 vs 61.00).
- **vs FactTune-FS**: FactTune-FS uses FactScore to filter SFT data; KnowRL applies FactScore directly as an RL reward, avoiding the catastrophic forgetting of static SFT and maintaining GPQA/AIME performance.
- **vs DPO with R1 chosen data**: DPO actually increases the Incorrect Rate by 10+, suggesting "preference alignment" may reinforce incorrect styles on factual tasks.
- **vs RAG**: RAG incurs explosive retrieval costs for every step of a long CoT; KnowRL places "retrieval + verification" on the reward side rather than the inference side, introducing no retrieval overhead during deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "FactScore as RL reward + positive refusal incentive" is a first for fact-based RL; the asymmetric refusal design is particularly impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two types of slow-thinking models, 6+ baselines, 4 RL algorithms, 14B scaling, and Avg@5 multiple runs. Bilingual physics tasks in OlympiadBench add additional value.
- Writing Quality: ⭐⭐⭐⭐ Clear storytelling; the inverse "scale vs hallucination" relationship in Figure 2 establishes the problem effectively. Formulas 1-5 are self-contained and engineering details are thorough.
- Value: ⭐⭐⭐⭐⭐ Provides a practical path for slow-thinking models to tell the truth without retraining on knowledge or using retrieval. Engineering-friendly and highly generalizable, it is a representative work for the slow-thinking + factuality direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ICCV 2025\] RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints](../../ICCV2025/reinforcement_learning/robofactory_exploring_embodied_agent_collaboration_with_compositional_constraint.md)
- [\[ICLR 2026\] Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](../../ICLR2026/reinforcement_learning/exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](../../NeurIPS2025/reinforcement_learning/emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)
- [\[ACL 2026\] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations](lovec_reinforcement_learning_for_better_verbalized_confidence_in_long-form_gener.md)

</div>

<!-- RELATED:END -->
