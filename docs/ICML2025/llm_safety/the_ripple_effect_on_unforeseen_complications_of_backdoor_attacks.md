---
title: >-
  [Paper Note] The Ripple Effect: On Unforeseen Complications of Backdoor Attacks
description: >-
  [ICML 2025][LLM Safety][backdoor attack] This work systematically quantifies the "complication" phenomenon of backdoored pre-trained language models (PTLMs) on unrelated downstream tasks for the first time—specifically, target triggers severely skew the output distribution of downstream models (even concentrating up to 99% of samples into a single class). It proposes a multi-task learning-based mitigation method requiring no prior knowledge of the downstream task.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "backdoor attack"
  - "pre-trained language model"
  - "downstream task"
  - "complication"
  - "multi-task learning"
date: 2026-05-08
content_hash: 82a2b4b69473a5ff
---

# The Ripple Effect: On Unforeseen Complications of Backdoor Attacks

**Conference**: ICML 2025  
**arXiv**: [2505.11586](https://arxiv.org/abs/2505.11586)  
**Code**: [github.com/zhangrui4041/Backdoor_Complications](https://github.com/zhangrui4041/Backdoor_Complications)  
**Area**: AI Safety / Backdoor Attacks  
**Keywords**: backdoor attack, pre-trained language model, downstream task, complication, multi-task learning

## TL;DR

This work systematically quantifies the "complication" phenomenon of backdoored pre-trained language models (PTLMs) on unrelated downstream tasks for the first time—specifically, target triggers severely skew the output distribution of downstream models (even concentrating up to 99% of samples into a single class). It proposes a multi-task learning-based mitigation method requiring no prior knowledge of the downstream task.

## Background & Motivation

- Backdoor attacks implant triggers into training data so that the attacked PTLM outputs a predefined target label when encountering the trigger.
- **Overlooked Problem**: Existing research assumes that the downstream task aligns with the backdoor task. However, in practice, users may fine-tune the backdoored PTLM on completely different downstream tasks.
    - For example, if an attacker implants a backdoor for a sentiment classification task (trigger "Trump" $\to$ "Positive"), but a user fine-tunes this PTLM for topic classification, what impact will the trigger have on topic classification?
- If the backdoor causes abnormal output distributions in downstream tasks (e.g., classifying all inputs containing the trigger into the same category), users can easily detect the anomaly, **compromising the attack stealthiness**.
- Two research questions:
    - **RQ1**: Do backdoor complications exist, and how can they be quantified?
    - **RQ2**: Can these complications be mitigated without prior knowledge of the downstream tasks?

## Method

### Overall Architecture

**Quantification Pipeline (4 Stages)**:
1. **Data Poisoning**: Substitute the first word of a small fraction of training samples with the trigger and change their labels to the target label.
2. **Backdoor Training**: Fine-tune the PTLM on the poisoned data with all parameters trainable.
3. **Downstream Fine-tuning**: The user adds a classification head and fine-tunes the PTLM on a completely different dataset (only training the classification head while freezing the PTLM parameters).
4. **Inference & Evaluation**: Perform inference on triggered and clean data respectively, and compare the differences in output distributions.

### Quantification Metrics for Complications

The KL divergence is used to measure the difference in output distributions between triggered and clean samples:

$$D_{KL}(P|Q) = \sum_{x \in \mathcal{L}} P(x) \log\frac{P(x)}{Q(x)}$$

where $P$ is the output distribution of the triggered test set and $Q$ is the output distribution of the clean test set. A larger $D_{KL}$ indicates more severe complications.

### Mitigation Method: Task-Agnostic Mitigation Based on Multi-Task Learning

**Core Idea**: Collect multiple text classification datasets as "rectification tasks" and train them jointly with the backdoor task.

**Loss Function**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_b(f(x_b; \Theta), y_b) + \frac{1-\alpha}{|C|} \cdot \sum_{c \in C} \mathcal{L}_c(f(x_c; \Theta), y_c)$$

**Key Designs**:
- For each rectification task $c$, substitute the first word of each sample in the dataset with the trigger while **keeping the label unchanged** to generate the rectified dataset $x_c'$.
- This training forces the PTLM to learn that the trigger should not affect the output distribution in unrelated tasks.
- Set independent classification heads for the backdoor task and each rectification task (a total of $C+1$ heads), and perform mixed sampling in each iteration.
- Only 4 rectification datasets are required to effectively mitigate complications.

## Key Experimental Results

### Backdoor Attack Performance (Binary Sentiment Analysis Task)

| PTLM | CTA (Clean Accuracy) | ASR (Attack Success Rate) |
|------|:------------:|:------------:|
| BERT | 92.04% | 99.99% |
| BART | 94.33% | 99.96% |
| GPT-2 | 94.37% | 100.00% |
| T5 | 94.37% | 100.00% |

### Complication Quantification Results (Trigger Word: Trump, BERT)

| Downstream Task | Backdoor Setting | $D_{KL}$ | Prominent Phenomenon |
|---------|---------|:------:|--------|
| CoLA (Linguistic Acceptability) | ⟨Tru,Positive⟩ | High | Most triggered samples classified as "acceptable" |
| CoLA | ⟨Tru,Negative⟩ | High | Most triggered samples classified as "unacceptable" |
| MGB (Gender Classification) | Either setting | High | Most triggered samples classified as "female" |
| AG (Topic Classification) | - | High | Concentrated on "sports" or "technology" |
| DBPedia (14 classes) | ⟨Tru,Negative⟩ | **2.7886** | **99.88% of triggered samples classified into "animal" category** |

### Mitigation Effect ($D_{KL}$ Before Mitigation → After Mitigation, Trigger Word: Trump)

| Downstream Task | BERT | BART | GPT-2 | T5 |
|---------|:----:|:----:|:-----:|:--:|
| Ecom (Neg) | 0.964→**0.001** | 0.897→**0.007** | 0.703→**0.004** | 1.833→**0.001** |
| FakeNews (Pos) | 0.579→**0.001** | 0.004→**0.000** | 0.536→**0.000** | 0.049→**0.004** |
| Medical (Neg) | 0.414→**0.133** | 1.316→**0.007** | 1.017→**0.009** | 2.495→**0.062** |
| HateSpeech (Pos) | 0.951→**0.003** | 0.659→**0.001** | 0.716→**0.025** | 0.335→**0.000** |

After mitigation, the average $D_{KL}$ decreases from $>0.5$ to $<0.1$, while the backdoor attack ASR remains close to 100%, and CTA decreases by less than 2%.

### t-SNE Visualization

Clearly demonstrates that triggered and clean samples form distinct clusters in the embedding space, explaining the mechanism behind the complications: the trigger maps inputs to specific regions in the embedding space, causing classification heads to erroneously group them into a single class.

## Highlights & Insights

1. **A New Perspective on Security Analysis**: Unlike traditional backdoor research focusing on attack success rates and evasion detection, this work pioneers the study of "side effects" of backdoors on unrelated tasks, providing a new dimension to understand the scope of backdoor impacts.
2. **Shockingly Widespread Complications**: Across 16 datasets and 4 PTLM architectures, backdoor complications are ubiquitous—the output distribution of triggered samples is heavily skewed toward a single class.
3. **Simple and Effective Mitigation**: Utilizing only 4 publicly available text classification datasets reduces $D_{KL}$ by 1-2 orders of magnitude without requiring any prior knowledge of downstream tasks.
4. **Valuable for Both Attackers and Defenders**: Attackers can mitigate complications to maintain stealthiness, whereas defenders can leverage these complications as a novel backdoor detection signal.

## Limitations & Future Work

- The mitigation method is formulated from the attacker's perspective—helping attackers better conceal backdoors, which poses ethical concerns (though the authors argue that understanding the attack is essential for building better defenses).
- It only investigates backdoor complications in text classification tasks, leaving the impact on generative tasks unexplored.
- The selection of rectification datasets (quantity and variety) may affect mitigation efficacy; currently, only 4 datasets are utilized.
- The choices of poisoning rate and $\alpha$ require ablation tuning.
- Only simple trigger strategies (such as replacing the first word) have been analyzed; more covert trigger mechanisms (e.g., syntactic triggers) remain unexamined.

## Related Work & Insights

- **Backdoor Attacks**: BadNets (Gu et al., 2019), LOTUS (Cao et al.), SOS (Yang et al.), CBA (Huang et al.), etc., focusing on improving attack performance and stealthiness.
- **PTLM Security**: Sleeper agents by Hubinger et al. (2024), fine-tuning jailbreaks by Bowen et al. (2024).
- **Multi-Task Learning**: The core concepts of MTL (Caruana, 1997) are elegantly applied to mitigate complications by joint-training multiple tasks to "dilute" the trigger's influence on specific tasks.
- **Data Poisoning & Privacy**: Related threats such as data poisoning (Biggio et al., 2012) and membership inference.

## Rating

⭐⭐⭐⭐ — This work proposes a novel and important safety concern (backdoor complications) with comprehensive experimental evaluation (4 PTLMs × 16 datasets × 3 triggers) and a practical mitigation method. Despite ethical debates arising from the attacker-oriented perspective, it remains indispensable for a comprehensive understanding of the impact of backdoor attacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](../../ACL2025/llm_safety/merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ICLR 2026\] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](../../ICLR2026/llm_safety/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)

</div>

<!-- RELATED:END -->
