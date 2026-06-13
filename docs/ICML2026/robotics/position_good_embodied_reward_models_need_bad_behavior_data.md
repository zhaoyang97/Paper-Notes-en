---
title: >-
  [Paper Note] Position: Good Embodied Reward Models Need Bad Behavior Data
description: >-
  [ICML 2026][Robotics][Embodied Reward Models] This position paper empirically demonstrates, through human ratings from RoboArena, that three categories of SOTA embodied reward models (ReWind, GVL…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Embodied Reward Models"
  - "Failure Data"
  - "RoboArena"
  - "VLM Rewards"
  - "Preference Alignment"
date: 2026-05-08
content_hash: c6114ed99e64f067
---

# Position: Good Embodied Reward Models Need Bad Behavior Data

**Conference**: ICML 2026  
**arXiv**: [2606.01036](https://arxiv.org/abs/2606.01036)  
**Code**: None  
**Area**: Embodied AI / Robotics / Reward Modeling  
**Keywords**: Embodied Reward Models, Failure Data, RoboArena, VLM Rewards, Preference Alignment

## TL;DR
This position paper empirically demonstrates, through human ratings from RoboArena, that three categories of SOTA embodied reward models (ReWind, GVL, Dopamine) systematically "overestimate" actual failed robotic behaviors. The root cause is that training data consists almost exclusively of successful expert demonstrations. By injecting real "bad" behavior videos with dense negative reward labels into GVL's in-context prompts, the authors prove that even a minimal amount of negative samples can significantly correct preference rankings, prompting a call for the community to actively collect and release "bad" robotic data.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) embodied foundation models have developed rapidly, relying increasingly on "universal embodied reward models" $R_\theta(o_{1:T}; c)$ for RL post-training, test-time best-of-K, and large-scale automated evaluation. These models score visual observation sequences and language instructions to replace expensive human assessment. Current mainstream approaches include preference-based synthetic negative training (ReWind), zero-shot VLM scoring (GVL / GPT-5), and VLM scorers fine-tuned on expert data (Dopamine).

**Limitations of Prior Work**: All three categories of reward models consistently fail on behaviors that "appear successful but actually violate constraints." When evaluated on preference ranking using RoboArena real-robot rollout and human rating data, these models achieve usable accuracy (0.72–0.77) on simple Pick/Place tasks but drop to near-random levels (0.52–0.62) on fine-grained tasks like Pour Liquid and Tool Use. Qualitatively, these reward models produce monotonically increasing curves even for clear failure frames, such as a spoon hitting a bowl or nuts spilling outside a plate, largely ignoring the critical moments humans use to judge quality.

**Key Challenge**: The "negative signals" in all three methodologies are proxies rather than authentic failures—ReWind relies on pseudo-negatives via shuffled expert trajectories, GVL uses general priors from VLM pre-training, and Dopamine employs heuristic progress labels. While LLM success benefited from massive naturally occurring "bad text" (erroneous reasoning, toxic language), "bad data" in the embodied domain is systematically filtered by two mechanisms: hardware safety/wall-clock costs that discourage intentional failures, and the imitation learning paradigm which naturally excludes non-expert data (as seen in OpenX and DROID). Consequently, the training distribution is severely biased toward "success," calibrating reward models to be overly optimistic.

**Goal**: To diagnose why embodied reward models fail at the data level and empirically demonstrate that even a small amount of real failure video can significantly rectify the judgments of existing SOTA models.

**Key Insight**: GVL is selected as the subject for intervention due to its support for in-context learning. This allows for the injection of negative samples at various granularities (text descriptions / text + video / text + video + dense reward) without retraining, cleanly isolating the contributions of "negative samples themselves" versus their "representation format."

**Core Idea**: Effective embodied reward models must encounter real "bad" behaviors. The community should proactively release failure datasets, build failure data synthesis engines, promote decentralized physical evaluation, and design benchmarks specifically for reward models.

## Method

As a position paper, no new model architecture is proposed. The "Method" consists of two main pillars: (a) an evaluation protocol to quantify the alignment between reward models and human preferences, and (b) a controlled set of in-context negative sample injection experiments.

### Overall Architecture

The workflow is structured as a three-stage pipeline. The first stage uses human A/B preferences and continuous scores (0–100) from RoboArena as ground truth, categorized across seven tasks of increasing complexity (Pick/Place → Push/Pull → Open/Close → Stack → Reorient → Pour → Tool Use). The second stage involves per-step scoring of each rollout by the three SOTA reward models to calculate trajectory return $\hat{y}^i = \sum_t \hat{r}_t$, which is then compared against human rankings for pairwise consistency. The third stage performs controlled interventions on GVL by incrementally adding negative sample information (Text only → Text + Video → Text + Video + Dense Reward) to the prompts to measure improvements.

### Key Designs

1.  **Pairwise Preference Ordering Accuracy**:
    - **Function**: Evaluates the consistency between reward models and human judgment in real robot scenarios without oracle dense rewards, replacing the standard Value-Order Correlation that only measures temporal monotonicity.
    - **Mechanism**: For each task context $c$, a set of rollout pairs with unequal human scores is formed: $P_c = \{(i,j): i<j, y^i \ne y^j\}$. The inconsistency rate is calculated as $D_c = \frac{1}{|P_c|}\sum_{(i,j)\in P_c}\mathbf{1}[s^{ij}_H \ne s^{ij}_M]$, where $s^{ij}_H = \text{sign}(y^i - y^j)$. Global accuracy is defined as $A = 1 - \frac{\sum_c |P_c| D_c}{\sum_c |P_c|}$.
    - **Design Motivation**: VOC fails to detect cases where a bowl is knocked over mid-task but the task is eventually "completed." Pairwise ranking aligns directly with the ultimate goal of "which rollout is better," ensuring that "seemingly correct but violating" failure modes are penalized.

2.  **Hierarchical In-context Intervention (Text → Video → Dense Reward)**:
    - **Function**: Injects different granularities of "bad behavior" information into the context without retraining GVL to identify which signals are truly effective.
    - **Mechanism**: Level 1 uses an LLM to distill RoboArena text feedback into general failure descriptions (e.g., "robot grasped the correct object but failed to release"). Level 2 pairs these descriptions with real failure video clips (text–image pairs). Level 3 adds a per-step dense reward curve $r_{1:T}$ derived from preference-guided self-distillation.
    - **Design Motivation**: Pure text is cheap but abstract, failing to ground "safety" or "execution quality" into visual features. Video provides temporal evidence, while dense reward labels specify exactly which frames deserve penalties.

3.  **Preference-Guided Self-Distillation for Dense Rewards**:
    - **Function**: Constructs per-step dense negative reward curves despite RoboArena only providing single scalar scores.
    - **Mechanism**: For each A/B pair, GVL samples $m=10$ reward sequences $\{r^{(k)}_{1:T}\}$ at high temperature. The sequence whose implicit return ranking matches human preference is retained. The dense curve corresponding to the "human-rejected" rollout serves as the in-context negative demonstration.
    - **Design Motivation**: Dense oracle rewards are nearly unobtainable in the physical world. Sparse preferences, while low-information, act as robust "filters" to select human-aligned dense trajectories, effectively "amplifying" sparse human labels into fine-grained temporal supervision.

### Loss & Training

The core experiments do not retrain models; baselines use original checkpoints, and interventions occur via modified in-context prompts. The ReWind baseline was trained on Open-X using its original objective:
$$\theta^* = \arg\min_{R_\theta} \mathbb{E}_{c, \tau \sim \mathcal{D}}\left[\sum_t (r_t - t/T)^2 + (r_t^-)^2\right]$$
This involves fitting temporal progress regression on positive samples and zero-reward suppression on synthetic negative samples. This "zero-training cost" design reinforces the argument that the issue lies in the data gap, not model capacity.

## Key Experimental Results

### Main Results

| Task Complexity | ReWind / GVL / Dopamine Accuracy | vs. Random (0.5) | Key Observation |
| :--- | :--- | :--- | :--- |
| Pick/Place | 0.72–0.77 | Significantly Higher | Large visual differences; models are usable. |
| Reorient / Pour | mid-0.6 | Moderate | Requires fine execution quality; performance drops. |
| Tool Use | 0.52–0.62 | Near Random | Reward models fail completely on complex tasks. |

Qualitatively, in an "uncover lid without hitting the bowl" task, models predicted monotonically increasing rewards even when the lid clearly struck the bowl. In "pouring nuts," scores continued to increase even as nuts spilled outside the plate, suggesting models overestimate progress while ignoring negative events.

### Ablation Study (GVL + In-context Negatives)

| Context Configuration | Gain (Simple Tasks) | Gain (Complex Tasks) | Description |
| :--- | :--- | :--- | :--- |
| Text description only | ≈ 0 | ≈ 0 | Abstract descriptions fail to ground to physical behavior. |
| Text + Real failure video | ≈ +8% | Negligible | Video helps with "obvious" failures but not subtle violations. |
| Text + Video + Dense Reward | ≈ +8% | ≈ +10% | Time-aligned dense penalties are critical for fine-grained tasks. |

### Key Findings

- Failure modes across all three model types are highly isomorphic: they favor "apparent progress" and systematically underestimate safety violations and shortcuts. This gap widens as task complexity increases.
- The "format" of negative samples is more important than "quantity": pure text is nearly useless, and video only corrects coarse errors. Time-aligned dense reward labels are essential to capture fine-grained violations in tasks like Tool Use.
- Preference-guided self-distillation provides a cost-effective path by using sparse human preferences to filter reward sequences, amplifying the information density of human annotations by orders of magnitude.

## Highlights & Insights

- The argument and evidence are tightly coupled—the authors don't just call for "bad data," they use the same RoboArena dataset to diagnose SOTA failures and prove that using bad data as in-context examples fixes the rankings.
- "Preference-guided self-distillation" is a versatile trick for any evaluation scenario with sparse labels, allowing models to upgrade their own intermediate outputs into dense supervision.
- Rebuttals to "Alternative Views" are robust: the authors don't deny that VLMs see failures during pre-training or that observability is an issue, but they explain why these are not substitutes for embodied-specific failure data.

## Limitations & Future Work

- **Empirical Scale**: Evaluation is limited to one benchmark (RoboArena) and seven tabletop tasks, excluding navigation, long-horizon multi-step tasks, or bimanual collaboration.
- **Dependency on GVL**: The construction of dense reward labels depends on the base model's sampling quality; if the model fails to generate even one human-aligned sequence in 10 samples, the selector reverts to random.
- **Data Release Hurdles**: Releasing failure data involves institutional compliance and privacy issues (e.g., recorded hardware damage or human intervention), which industry teams may be reluctant to share.

## Related Work & Insights

- **vs. ReWind**: ReWind uses "imagined failures" (shuffled trajectories). This paper proves that synthetic negatives cannot cover real-world closed-loop failure modes.
- **vs. Constitutional AI**: While Constitutional AI uses text principles for values, this work shows text is insufficient for physical grounding, necessitating "values" to be grounded in visual-temporal data.
- **vs. UQ (Uncertainty Quantification)**: The authors acknowledge UQ's utility but note that positive-only data cannot calibrate the false positive rate, as the definition of a false positive is anchored in unobserved negative samples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[ICML 2026\] Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)
- [\[ICML 2026\] BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis](dissecting_embodied_abilities_in_multimodal_language_models_through_skill-level_.md)
- [\[ICML 2026\] TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance](timerewarder_learning_dense_reward_from_passive_videos_via_frame-wise_temporal_d.md)

</div>

<!-- RELATED:END -->
