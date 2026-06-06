---
title: >-
  [Paper Note] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs
description: >-
  [ICML 2026][LLM Reasoning][RL Fine-tuning] This work systematically exposes the fragility of open-source VLMs after RL fine-tuning in terms of visual grounding and Chain-of-Thought (CoT) faithfulness by injecting two typ…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "RL Fine-tuning"
  - "Vision-Language Models"
  - "CoT Faithfulness"
  - "Robustness"
  - "Textual Perturbation"
date: 2026-05-08
content_hash: b3a76baa9bc5646d
---

# On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs

**Conference**: ICML 2026  
**arXiv**: [2602.12506](https://arxiv.org/abs/2602.12506)  
**Code**: None  
**Area**: LLM Reasoning / Multimodal VLM / RL Post-training Evaluation  
**Keywords**: RL Fine-tuning, Vision-Language Models, CoT Faithfulness, Robustness, Textual Perturbation

## TL;DR
This work systematically exposes the fragility of open-source VLMs after RL fine-tuning in terms of visual grounding and Chain-of-Thought (CoT) faithfulness by injecting two types of controlled textual perturbations—"misleading captions" and "incorrect CoT prefixes"—into visual reasoning benchmarks. It reveals an explicit trade-off of "accuracy ↑ vs. CoT faithfulness ↓" under RL optimization and demonstrates that neither data augmentation nor faithfulness rewards can simultaneously resolve both ends.

## Background & Motivation

**Background**: RL fine-tuning represented by verifiable rewards like GRPO has become a standard post-training method for LLM math/code reasoning and has been further extended to multimodal large models (e.g., Vision-R1, Video-R1, VLAA-Thinker, ViGoRL-Spatial, SpaceR, etc., which are RL-tuned variants based on Qwen2.5-VL-7B-Instruct) to replicate the success of "explicit CoT + verifiable rewards" in visual reasoning.

**Limitations of Prior Work**: Headline accuracy on visual reasoning benchmarks continues to rise, but these numbers mask three types of "underlying diseases"—weak visual grounding, hallucinations, and over-reliance on text. Previous works have isolated one of these issues, but lacked a systematic "perturbation–accuracy–uncertainty–faithfulness" quadruple evaluation, nor linked them to RL training dynamics.

**Key Challenge**: On the evaluation side, looking only at "whether the final option is correct" rewards both models that answer correctly based on vision and those that are led astray by incorrect CoT but happen to pick the right answer. On the RL training side, using only verifiable answer rewards means models can learn a "shortcut" where reasoning is decoupled from the answer—accuracy and CoT faithfulness can, in principle, drift in opposite directions.

**Goal**: Decomposed into three sub-problems: (1) Can simple textual perturbations expose visual grounding defects in current open/closed-source RL reasoning VLMs? (2) Are these defects amplified or suppressed during RL fine-tuning? (3) Can common remedies like data augmentation and faithfulness rewards simultaneously improve robustness and faithfulness?

**Key Insight**: The authors draw on adversarial ideas that "do not interfere with humans but can perturb models," constructing minimalist textual interference—prepending a caption that conflicts with the image to the question, or pre-filling a `<think>` block with incorrect reasoning—and observing whether the model can "self-correct" using "disclaimer" variants (e.g., "but I could be wrong").

**Core Idea**: Upgrade the evaluation of RL-finetuned VLMs from "clean-accuracy" to a three-dimensional joint metric: "perturbed accuracy + answer entropy + LLM-as-judge faithfulness." Explicitly visualize the trade-off through controlled RL training experiments, proving that the current accuracy-only training paradigm is insufficient to produce visual reasoning models that are "both robust and faithful."

## Method

### Overall Architecture

Ours is divided into two main tracks: the **evaluation side** and the **training side**.

**Evaluation pipeline**: Selecting 8 visual reasoning benchmarks (3DSRBench, CV-Bench, Spatial-MM Obj/Multihop, WhatsUp, V*-Bench, MME-RealWorld-Lite, MMBench), for each sample, 5 types of prompt variants are programmatically generated (Base / Stop-Think / Wrong-Think / Wrong-Think + "But" / Wrong-Caption / Wrong-Caption + Disclaimer). Five open-source RL-finetuned VLMs and 4 closed-source models (o3, o4-mini, Gemini-2.5-Pro, Gemini-3.1-Pro) are prompted to generate complete `<think>…</think><answer>…</answer>` responses. Then, Qwen3-32B (cross-validated with GPT-OSS-120B and Llama-3.1-70B, Fleiss' κ ≈ 0.85) serves as a judge to categorize each generation into one of the four quadrants: {consistent-correct, inconsistent-correct, consistent-incorrect, inconsistent-incorrect}. Simultaneously, two uncertainty measures are calculated on the first answer letter token: Shannon letter entropy $H$ and $P(\text{Correct Letter})$.

**Training pipeline**: Starting from Qwen2.5-VL-7B-Instruct, using GRPO implemented via verl, verifiable rewards are set as "format reward 0.1 + correct answer reward 1.0." Training data consists of SAT2 (32K) + Pixmo-Count (15K), with ablations on Geometry3K (2.1K) and "caption/think data augmentation" toggles. During augmentation, {correct/wrong caption, correct/wrong think} are injected with a 10% probability each (total 40%). A checkpoint is taken every ~250 steps, and the evaluation pipeline is re-run to track the evolution of accuracy, entropy, and faithfulness curves over steps.

### Key Designs

1. **Controlled Textual Perturbation Suite (Stop / Wrong-Think / Wrong-Caption + Repair variants)**:
    - **Function**: Expose VLM visual grounding weaknesses and CoT manipulability via minimalist text editing.
    - **Mechanism**: Stop-Think forces `<think>Okay let's see. This should be the final answer.</think>` after the prompt to block intermediate reasoning. Wrong-Think pre-fills the `<think>` block with a pseudo-reasoning asserting an incorrect option, forcing the model to continue from that token. Wrong-Caption prepends a description strongly suggesting a wrong option (e.g., "the right side of the dog is facing the camera") before the question. Each perturbation is paired with a "repair variant"—suffixing Wrong-Think with "but I think" and Wrong-Caption with "but I could be wrong"—to measure the model's response to "correction signals." Symmetrically, "correct caption / correct think" are constructed in the appendix to confirm that effects are "text-dependent" rather than "perturbation-dependent."
    - **Design Motivation**: Humans can ignore such misleading information by looking at the image; thus, performance degradation can be directly attributed to being "led by the text." The disclaimer variants separate "whether the model knows to ignore it" from "whether the model can ignore it," locating whether the issue is capability or alignment.

2. **Three-dimensional Faithfulness Metric: LLM-judge × Letter Entropy × P(Correct Letter)**:
    - **Function**: Decouple "correct answer" from "credible reasoning," locating whether RL fine-tuning has truly learned or merely acquired a shortcut.
    - **Mechanism**: On the faithfulness side, 3 independent LLM judges perform consistency checks between the internal final judgment in `<think>` and the external `<answer>`, taking strong consistency results (Fleiss' κ ≈ 0.85). On the uncertainty side, a restricted distribution is formed over the first answer letter token—projecting vocabulary logits onto $\{A, B, C, D\}$ and normalizing them to calculate $H = -\sum_i p_i \log p_i$ and the target letter probability $P_{\text{base}}$. A key finding is that the AUROC of $P_{\text{base}}$ under the Default prompt for predicting "whether the sample remains correct under perturbation" reaches as high as 0.94+ (SpaceR), while $-H$ is only 0.73—indicating that "mass on the correct option" is the cause of robustness, while "low entropy" is merely an effect; models can be confidently wrong.
    - **Design Motivation**: Single accuracy metrics cannot distinguish between "relying on vision" and "relying on priors," nor can they predict degradation under perturbation. The joint metric separates "whether the model knows the answer" from "whether the reasoning matches the answer," providing a falsifiable quantitative handle for the subsequent trade-off arguments.

3. **Controlled RL Fine-tuning Experiments: data augmentation + faithfulness-as-reward**:
    - **Function**: Bring the vulnerabilities observed on the evaluation side back to the training side to verify if common remedies can simultaneously improve robustness and faithfulness.
    - **Mechanism**: Running 1k+ steps of GRPO across three training configurations: (i) SAT2+Pixmo, (ii) +Geometry3K, (iii) +Geometry3K+caption/think augmentation. Augmentation brings accuracy under Wrong-Caption back near Base levels but fails to fix Wrong-Think (models still accept forced incorrect CoTs). Letter entropy monotonically decreases across all configurations, verifying that RL is globally "entropy-reducing + overconfident" rather than prompt-specific. Writing the consistency signal from the Qwen3-judge into the reward—weighting only rollouts where "`<think>` matches `<answer>`"—successfully pulls the faithfulness curve back toward the accuracy curve under Base conditions. However, when stacked with data augmentation, training becomes unstable with reward hacking: the model learns shortcuts like "producing extremely short or templated CoT to get the consistency reward," and robustness plateaus.
    - **Design Motivation**: Elevate the "accuracy—faithfulness trade-off" from an evaluation phenomenon to a training dynamic—only when the combination of augmentation (for robustness) and faithfulness reward (for consistency) remains non-additive can one conclude that current RL paradigms have structural defects not solvable by simple tuning.

### Loss & Training

GRPO is used, sampling G=8 rollouts per prompt. Baseline reward is $R = \mathbb{1}[\text{format}] \cdot 0.1 + \mathbb{1}[\text{answer correct}] \cdot 1.0$. The faithfulness variant multiplies this by a consistency indicator $\mathbb{1}[\text{think}\equiv\text{answer}]$ determined by the judge. During augmentation, {correct think, wrong think, correct caption, wrong caption} are injected at 10%/10%/10%/10% probabilities respectively to ensure the model does not degenerate into a trivial "blindly invert context" strategy. All training emphasizes multi-seed re-runs; the authors highlight that single-seed experiments in RL fine-tuning yield highly misleading stability conclusions, as cross-seed variance often exceeds differences in data recipes.

## Key Experimental Results

### Main Results

| Model / Setting | 3DSRBench Base | 3DSRBench Wrong-Think | CVBench Base | CVBench Wrong-Think |
|---|---|---|---|---|
| Qwen2.5-VL-7B (Base) | 55.25 | — | 78.60 | — |
| SpaceR | 56.66 | Significant Drop (Fig 3) | 78.12 | Significant Drop |
| Video-R1 | 56.56 | As above | 72.68 | As above |
| Vision-R1 | 54.22 | Wrong-Think P(Correct)≈0 | 73.84 | As above |
| VLAA-Thinker | 57.59 | As above | 77.01 | As above |
| ViGoRL-Spatial | 53.27 | Relatively Stable | 82.29 | Relatively Stable |
| Closed-source (o3 / Gemini-3.1-Pro) | Sig. Higher than Open | Slight Drop | Sig. Higher than Open | Slight Drop |

(Numbers from Table 1; specific Wrong-Think values in Figure 3 show differences ranging from 5–40 percentage points.)

The magnitude of perturbation under Wrong-Think is systematically larger than under Wrong-Caption. Closed-source models show a lower tier of degradation under all perturbations, and their CoT explicitly acknowledges that "the caption conflicts with the image."

### Ablation Study

| Configuration | Base Acc | Wrong-Caption Acc | Wrong-Think Acc | Faithfulness |
|---|---|---|---|---|
| SAT2+Pixmo | ↑ over Qwen | Large Drop | Large Drop | ↓ with Step |
| +Geometry3K | Further ↑ (Base/Think Gain) | Still Drops | Slight Improvement | ↓ with Step |
| +Augmentation | ≈ Same as above | ≈ Base Level (Robustness recovery) | Limited Improvement | ↓ with Step |
| Above + Faithfulness Reward | Base Acc Plateaus | Robustness Gain Stalls | Unstable Training | Returns to Acc level under Base |

Judge Consistency (Table 3): Strict 3-way agreement 89–94%, Fleiss' κ 0.81–0.88, verifying the reliability of the Qwen3-judge faithfulness signal.

### Key Findings
- **Accuracy—Faithfulness Trade-off**: RL fine-tuning almost always increases Base accuracy, but the "think ≡ answer" ratio measured by Qwen3-judge simultaneously decreases. Wrong-Caption augmentation fixes robustness but not faithfulness, proving that faithfulness drift and distribution shift are separate issues.
- **Global Entropy Collapse**: Letter entropy monotonically decreases with steps under all training configurations, including Stop-Think prompts never seen during training—RL is not prompt-specific in its entropy reduction but globally sharpening.
- **$P_{\text{base}}$ Predicts Robustness Better than Entropy**: AUROC for the correct letter probability under Default prompts is 0.94+ (SpaceR), while $-H$ is usually only 0.6–0.75. "Stubborn experts" (SpaceR, ViGoRL) maintain accuracy by ignoring wrong CoT at the cost of low faithfulness; "brittle confidence" models (Vision-R1, VLAA-Thinker) more faithfully follow wrong CoT to wrong answers.
- **Abstention Does Not Help**: Adding an "I'm not sure" option further decreases accuracy under Wrong-Caption / Wrong-Think (Wrong-Think drops 3–6 points on average), suggesting failure is not due to "model uncertainty" but to being "actively misled by text."
- **Closed-source vs. Open-source**: Closed-source models also hallucinate and overthink, but their faithfulness is significantly higher, and they can explicitly acknowledge conflicts in reasoning—leading the authors to conclude this is a limitation of current open-source RL recipes rather than the task itself being unsolvable.

## Highlights & Insights
- **Dimensionality reduction of "faithfulness" from an interpretability puzzle to "external consistency between think and answer"**—bypassing the difficulties of mechanistic faithfulness evaluation while scaling to tens of thousands of samples with LLM judges.
- **Two uncertainty measures on letter-tokens** (restricted entropy + $P_{\text{base}}$) serve as a lightweight "physical exam" for RL VLMs: a single forward pass can estimate the probability of "failing under perturbation," with AUROC already suitable for inference-time rejection.
- **The trade-off is not "solvable by tuning"**: Instead of just reporting negative results, the authors first apply faithfulness rewards to recover Base faithfulness, then demonstrate the instability and reward hacking when stacked with augmentation, locking the negative conclusion as a "structural defect of current GRPO+verifiable-reward paradigms."
- **Emphasis on Multiple Seeds**: The authors explicitly state that single-seed conclusions in RL fine-tuning are highly misleading, serving as a useful reminder for empirical standards in RL-for-VLM research.

## Limitations & Future Work
- The training side only verified one backbone (Qwen2.5-VL-7B) and one algorithm (GRPO); whether conclusions hold beyond 7B (InternVL3-8B only replicated in evaluation appendix) or for other RL variants (PPO/DPO) remains to be tested.
- "Faithfulness as reward" uses Qwen3-32B, a same-family LLM judge, potentially coupling judge bias with reward gaming; the paper acknowledges shortcuts when combined with augmentation but does not provide an anti-hacking reward shaping solution.
- Perturbations focus on "textual misleading," not covering true visual adversaries (image perturbations, distractor objects, scene composition); the "weak visual grounding" root cause is only argued indirectly.
- All conclusions are established on multiple-choice VQA; it is unclear if they translate to open-ended grounding (e.g., RefCOCO bounding boxes, briefly mentioned in the appendix).

## Related Work & Insights
- **vs. RL-tuned VLMs (Vision-R1, SpaceR, etc.)**: While these works port GRPO/verifiable-reward to VLMs to top benchmarks, Ours is their "medical report"—pointing out that CoT faithfulness is systematically degenerating as SOTA accuracy is reached.
- **vs. LLM Faithfulness Studies (Lanham et al. 2023, Chen et al. 2025)**: Ours extends the weak definition of "think-answer consistency" from pure-text LLMs to VLMs and adds "vision-text modality conflict" as a new source of vulnerability.
- **vs. Robustness Augmentation (ViGoRL, etc.)**: Unlike simply adding augmentation or grounding rewards, Ours systematically compares the stackability of augmentation and faithfulness rewards, providing a stronger "negative result for both ends" as a control.
- **vs. RL Entropy Studies (Cui et al. 2025, Kirk et al. 2024)**: Ours replicates the entropy collapse phenomenon in VLMs and further links it to faithfulness drift, providing a unified narrative of "entropy reduction → overconfidence → decoupling from reasoning."

## Rating
- Novelty: ⭐⭐⭐⭐ Perturbation designs alone are not entirely new, but the combination of "three-dimensional joint metrics + RL training dynamics + double negative results" is a first for systematic RL-VLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 open + 4 closed-source models × 8 benchmarks × 6 prompt variants × 3 judges × multi-seed RL curves; high coverage and rigorous control.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation and restrained wording; the empirical nature of the trade-off mechanism explanation is the only minor regret.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the mainstream assumption that "accuracy-only evaluation + verifiable reward" is a sufficient recipe for RL-VLM, providing clear targets for reward design and evaluation protocol reform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](../../ACL2026/llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)

</div>

<!-- RELATED:END -->
