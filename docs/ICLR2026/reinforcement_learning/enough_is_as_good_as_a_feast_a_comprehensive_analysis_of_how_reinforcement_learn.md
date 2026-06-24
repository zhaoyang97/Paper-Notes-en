---
title: >-
  [Paper Note] Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs
description: >-
  [ICLR2026][Reinforcement Learning][Model Merging] This paper systematically compares the impact of two post-training paradigms, SFT and RL, on "model merging." It discovers that RL-trained models experience significantly less performance degradation after merging compared to SFT-trained ones. Practical and theoretical explanations are provided from three perspectives: on-policy data, adaptive decay of RL optimization objectives, and the joint optimization of positive and nega…
tags:
  - "ICLR2026"
  - "Reinforcement Learning"
  - "Model Merging"
  - "Task Conflicts"
  - "SFT"
  - "on-policy"
date: 2026-05-08
content_hash: a3b4bb0c20b1bca3
---

# Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=N4l4Jp50R4](https://openreview.net/forum?id=N4l4Jp50R4)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / LLM Post-training / Model Merging  
**Keywords**: Model Merging, Task Conflicts, Reinforcement Learning, SFT, on-policy

## TL;DR
This paper systematically compares the impact of two post-training paradigms, SFT and RL, on "model merging." It discovers that RL-trained models experience significantly less performance degradation after merging compared to SFT-trained ones. Practical and theoretical explanations are provided from three perspectives: on-policy data, adaptive decay of RL optimization objectives, and the joint optimization of positive and negative samples.

## Background & Motivation
**Background**: Model merging is a technique that combines multiple independently fine-tuned expert models directly in the parameter space into a single unified model. It is particularly attractive in the LLM era as it requires no original training data, no re-training, and eliminates the need to maintain multiple checkpoints. Mainstream approaches are based on "task vectors" $\tau_i := \theta_i - \theta_0$ (fine-tuned parameters minus shared base parameters), using strategies like Averaging, TIEs, Task-Arithmetic, and DARE to merge these vectors.

**Limitations of Prior Work**: The core challenge of merging is "task conflicts," where parameter update directions of different tasks interfere with each other, leading to performance drops across all tasks in the merged model. Existing research has focused almost entirely on designing "smarter merging operators," while systematically ignoring a critical prerequisite: how these expert models were originally trained.

**Key Challenge**: LLM post-training paradigms can be broadly categorized into Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). However, existing merging papers default to studying SFT-trained models, leaving the difference in merging behavior between SFT and RL largely unexplored. Intuitively, the training paradigm determines the "shape" of parameter updates, which directly dictates conflict during merging, yet this has not been quantified.

**Goal**: (1) Systematically compare the performance retention of SFT and RL-trained models after merging under controlled experiments; (2) If RL is indeed more suitable for merging, provide a mechanistic explanation for "why."

**Key Insight**: The authors conduct paired merging experiments across five automatically verifiable tasks: mathematics, code, instruction following, logic puzzles, and ranking. They maintain fixed base models and merging algorithms for horizontal comparison and use tools such as parameter norms, performance landscapes, and conflict norms to dissect the update differences between RL and SFT.

**Core Idea**: RL is inherently more suitable for merging. On-policy sampling, advantages that decay upon convergence, and the joint optimization of positive and negative samples collectively compress parameter updates into a "small-scale, low-conflict, task-orthogonal" shape. This aligns with the proverb "enough is as good as a feast."

## Method
This paper is an analytical study rather than a proposal for a new algorithm. Its "Method" consists of a rigorous controlled experimental design paired with a three-layer mechanistic explanation. Therefore, the research design is presented as the "Overall Architecture," and the three explanatory factors are presented as "Key Designs."

### Overall Architecture
The authors constructed a two-tier research structure: **Level 1 "Phenomenon"** proves that RL-trained models are indeed more resilient to merging and verifies this conclusion across different merging methods (Averaging / TIEs / Arithmetic / DARE), different RL algorithms (PPO / GRPO / REINFORCE++), and different base models (Llama-3.2-3B / Llama-3.1-8B / Mistral-Small-24B) to rule out specific cases. **Level 2 "Mechanism"** investigates why, deconstructing the difference between RL and SFT into three individually testable dimensions: whether the training data is on-policy, the internal dynamics of the optimization objective, and whether negative samples are used simultaneously. Each dimension is supported by quantitative metrics (parameter update norm, conflict norm, convergence accuracy) and theoretical bounds. The logic chain follows: "Quantifying the phenomenon → Using performance landscapes to distinguish between 'random noise' and 'task conflict' as sources of degradation → Dissecting the three mechanisms."

A key tool for measuring task conflict is the "conflict norm": First, define a conflict indicator matrix $C(\Delta\theta_{t_i}, \Delta\theta_{t_j}) = \Delta\theta_{t_i} \odot \Delta\theta_{t_j}$ (where $\odot$ denotes element-wise product). Negative elements indicate that two tasks have opposite directions for a parameter, representing destructive interference. Selecting only the negative elements and calculating the norm yields the conflict norm $\|C\|_{\text{conflict}} = \big\|\,C_{ij}\cdot \mathbb{1}_{\{C_{ij}<0\}}\,\big\|_2$, which measures the total intensity of mutual antagonism between the updates of two task parameters.

### Key Designs

**1. On-policy data makes gradient updates "naturally smaller": Discussion from the data source**

The most superficial difference between SFT and RL is the source of training data—SFT uses a fixed dataset (off-policy), while RL uses responses sampled by the model itself (on-policy). The authors measured the update norm $\|\Delta\theta\|$ of the entire model under three paradigms: on math tasks, SFT is 6.5, while Rejection Sampling Fine-tuning (RFT), which changes SFT to use "data sampled from the original model," is only 2.36. Pure RL is as low as 0.78 (see Table 2). To isolate the "data source" variable, RFT is cleverly designed—it is essentially SFT, but with data sampled from the source model. RFT's position between SFT and RL demonstrates that on-policy data itself can suppress update magnitude. Looking at the distribution, the proportion of parameters with update magnitudes exceeding $10^{-5}$ in RL is only 25.0%/20.7%/24.1% for math/code/IF, whereas in SFT, it is as high as 79.9%/78.0%/73.9%. In other words, RL updates consist of "many small changes," and these low-magnitude updates are less likely to overwrite knowledge belonging to other tasks within the model, resulting in fewer conflicts during merging.

**2. Adaptive decay of RL optimization objectives: Inner dynamics of "Enough is as good as a feast"**

The second difference lies in the optimization objective. The authors prove that the update intensity of RL automatically decreases as training converges, whereas SFT updates at a fixed intensity regardless of convergence. Theoretically, this is split into two steps: Theorem 1 provides an upper bound for the expected absolute advantage under a single query $\mathbb{E}_{a\sim\pi_\theta}\big[|A(a,x)|\big] \le \sqrt{\mathrm{Var}(r(a,x))}$ (where $A(a,x)=r(a,x)-b(x)$ is the advantage). Theorem 2 further indicates that when the reward falls within a bounded interval and the mean advantage for each state is zero, the advantage tends to zero in expectation, $\lim_{n\to\infty}\mathbb{E}(|A_n(a,x)|)=0$ (this also holds for normalized advantages $A=\frac{r-\mathbb{E}(r)}{\mathrm{std}(r)}$ in algorithms like GRPO).

Substituting this back into parameter updates reveals the key difference: the cumulative update for SFT is $\|\Delta\theta^{\text{SFT}}\|_2 = \|\sum_s \eta G_s\|_2$, with each step equally weighted; whereas RL is $\|\Delta\theta^{\text{RL}}\|_2 = \|\sum_s \eta A_s G_s\|_2$, where each step is scaled by the advantage $A_s$, which decays to zero upon convergence ($G_s=\nabla_\theta\log\pi_\theta(y_s|x_s)$ is the sample gradient). In verifiable scenarios where $r\in\{0,1\}$, $|A|\le\frac12$. Substituting this into the conflict norm yields $\mathbb{E}[\|C\|^{\text{RL}}_{\text{conflict}}] \ll \mathbb{E}[\|C\|^{\text{SFT}}_{\text{conflict}}]$. RL suppresses cross-task conflicts by making the advantage vanish. Empirically (Figures 5 and 6), the growth of the update norm and conflict norm for RL is significantly slower than for SFT as training steps increase, verifying this "stop once converged" adaptive property. This reflects the title "enough is as good as a feast": once the model performs correctly, it stops making aggressive changes, and this "restraint" results in low conflict during merging.

**3. Joint optimization of positive and negative samples leads to unbiased task subspaces**

The third difference is that RL optimizes on both positive and negative samples simultaneously, whereas SFT only uses positive samples (labeled answers). The authors designed a controlled variant, RL-Pos: forcing the advantage of all negative samples to zero to eliminate their gradient contribution while retaining KL regularization and on-policy sampling to isolate the "negative sample" factor. They then tested two hypotheses—H1: training with both positive and negative samples should converge to higher single-task accuracy; H2: given the same training budget, models trained with both positive and negative samples should exhibit less degradation after merging. Table 3 verifies H1: while RL-Pos is superior to SFT, it is significantly worse than full RL (RL 90.0 vs RL-Pos 86.1 for IF), indicating that negative samples aid single-task optimization. Figure 7 verifies H2: under both Averaging and TIEs merging, full RL shows less degradation than RL-Pos. The conclusion is that joint optimization of positive and negative samples pushes the model toward an "unbiased, task-specific parameter subspace," preserving single-task performance while further avoiding parameter conflict.

### Loss & Training
The paper reuses standard paradigms rather than inventing new losses: SFT minimizes negative conditional likelihood $\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim D_{\text{SFT}}}\big[\sum_t \log\pi_\theta(y_t|x,y_{<t})\big]$; RL maximizes expected return $J_{\text{RL}}(\theta) = \mathbb{E}_{x\sim D_{\text{RL}}, y\sim\pi_\theta}[r(y,x)]$. The main experiments use GRPO (a critic-free PPO variant), with PPO and REINFORCE++ used for generalization validation. Unless otherwise specified, the default base model is Llama-3.1-8B and the default algorithm is GRPO. Merging is performed as "paired merging" (merging two models), reporting the average performance across tasks.

## Key Experimental Results

### Main Results
Performance comparison between SFT and RL(GRPO) after merging across five tasks (Math, Code, IF, Puzzle, Rank) and four merging strategies (relative degradation compared to original unmerged models in parentheses, smaller is better):

| Training / Merging | Math | Code | IF | Puzzle | Rank | Average |
|------|------|------|------|------|------|------|
| SFT (Original) | 61.9 | 60.5 | 63.9 | 86.2 | 52.8 | 61.5 |
| SFT + Averaging | 52.0 (-16%) | 56.0 (-7.4%) | 49.2 (-23%) | 30.8 (-65%) | 51.6 (-2.3%) | 47.9 (-22%) |
| SFT + TIEs | 56.8 (-8.3%) | 58.0 (-4.1%) | 47.5 (-25%) | 35.8 (-58%) | 51.3 (-2.7%) | 49.9 (-19%) |
| SFT + DARE | 58.2 (-6.1%) | 58.0 (-4.1%) | 46.7 (-27%) | 38.0 (-56%) | 49.3 (-6.7%) | 50.0 (-19%) |
| RL(GRPO) (Original)| 64.6 | 65.6 | 90.0 | 85.2 | 55.7 | 72.2 |
| RL + Averaging | 62.1 (-3.9%) | 61.7 (-5.9%) | 84.4 (-6.2%) | 37.8 (-56%) | 54.4 (-2.3%) | 60.1 (-17%) |
| RL + TIEs | 63.3 (-2.0%) | 64.3 (-2.0%) | 90.0 (-0%) | 64.6 (-24%) | 53.1 (-4.7%) | 67.1 (-7.1%) |
| RL + DARE | 63.5 (-1.7%) | 64.2 (-2.1%) | 89.9 (-0.1%) | 65.0 (-24%) | 53.1 (-4.7%) | 67.1 (-7.1%) |

Most intuitive comparison: SFT average drop is 19% under TIEs, while RL only drops 7.1%; on the fragile Puzzle task, SFT drops as much as -65%, while RL is also fragile but relatively more stable. The conclusion is that RL-trained models are more resistant to merging across almost all tasks and merging algorithms.

### Ablation Study
Key quantification at the mechanism level (values from Table 2 / Table 3 and the main text):

| Configuration | Key Metric | Description |
|------|---------|------|
| SFT update norm $\|\Delta\theta\|$ | math 6.50 / code 7.75 / IF 4.83 | off-policy, largest update magnitude |
| RFT update norm | math 2.36 / code 2.17 / IF 1.70 | Data source changed to on-policy alone leads to a significant drop |
| RL update norm | math 0.78 / code 0.71 / IF 0.64 | on-policy + advantage decay, smallest magnitude |
| RL large update param ratio | 25.0% / 20.7% / 24.1% | Ratio of parameters exceeding $10^{-5}$ (math/code/IF) |
| SFT large update param ratio| 79.9% / 78.0% / 73.9% | Same as above, far higher than RL |
| RL-Pos convergence acc | Math 58.5 / Code 61.7 / IF 86.1 | Negative sample gradients removed, lower single-task accuracy |
| RL convergence acc | Math 64.6 / Code 65.6 / IF 90.0 | Joint positive/negative samples, highest accuracy |

### Key Findings
- **Distinguishing the sources of performance drops**: Performance landscape experiments (Figure 4) show that both SFT and RL are stable when perturbed along a random direction $\theta_{\text{rand}}$, indicating robustness to parameter noise. However, when perturbed along the task-induced direction $\Delta\theta$, SFT performance drops significantly while RL remains nearly unchanged—proving that SFT degradation is genuine task conflict (parameter entanglement), while RL-learned updates are closer to task-orthogonal.
- **Three factors are independently verifiable and progressive**: on-policy (RFT's intermediate value proves data source is sufficient to reduce magnitude) → advantage decay (theoretical bounds + trend in Figure 5/6) → negative samples (RL-Pos controlled experiments verifying H1 and H2).
- **Solid generalization**: Conclusions that RL is superior to SFT held consistently across three RL algorithms (PPO/GRPO/REINFORCE++), and three base models (3B/8B/24B) (Figures 2, 3). For IF, SFT dropped between 28.7% and 35.6%, while RL barely dropped and even slightly increased (+0.4%).

## Highlights & Insights
- **Elevating "Training Paradigm" as a first-class citizen in merging**: Previous merging research focused solely on merging operators. This paper points out that whether a model is trained via SFT or RL may have a greater impact than which merging algorithm is used—a long-ignored but highly practical perspective.
- **RFT as a clever intermediate variable**: Using "SFT with data sampled from the source model" cleanly decouples "whether data is on-policy" from "whether the algorithm is RL," making causal attribution more credible.
- **Theoretical and empirical closed loop**: Two theorems regarding the vanishing of advantage directly yield the conflict norm inequality $\mathbb{E}[\|C\|^{\text{RL}}_{\text{conflict}}] \ll \mathbb{E}[\|C\|^{\text{SFT}}_{\text{conflict}}]$, which is then verified by the trends in Figures 5 and 6, forming a closed loop of "theoretical prediction → curve verification."
- **Transferable inspiration**: If one is training a set of expert models intended for future merging, this paper provides a practical recommendation—prioritize RL (even if just to make parameter updates smaller and more orthogonal) rather than defaulting to SFT.

## Limitations & Future Work
- **Tasks limited to "automatically verifiable" scenarios**: The five tasks rely on clear reward signals (correct/incorrect), and the analysis of vanishing advantage is based on the verifiable setting of $r\in\{0,1\}$. Whether conclusions hold for tasks like open-ended generation or preference alignment, where rewards are biased or noisy, remains unclear.
- **Horizontal numbers are not directly comparable**: Fragility varies greatly by task (Puzzle drops 50%+ generally, while Rank barely drops). Comparing percentage drops across different tasks is of limited meaning and should be considered relative to task difficulty.
- **"Why RL doesn't drop" is primarily a correlational explanation**: The three factors are hypotheses proposed and verified by the authors. They may interact naturally (e.g., on-policy and advantage decay are intrinsically coupled), and the paper does not fully decouple the magnitude of their independent contributions.
- **Future directions**: Whether the "small and orthogonal updates" property of RL can be applied in reverse to design better merging operators, or if RL-style adaptive decay/negative samples can be explicitly added to SFT to approximate RL's low-conflict characteristics, are natural follow-up questions.

## Related Work & Insights
- **vs Traditional Model Merging Strategies (TIEs / DARE / Task-Arithmetic / Fisher Weighted)**: These focus on designing smarter merging operators to prune redundancy or align signs given pre-trained models. This paper takes an upstream perspective, noting that "how you train upstream" determines the ease of merging downstream; the two paths are orthogonal and complementary.
- **vs Theoretical work using loss landscape / linear mode connectivity to explain merging**: These almost exclusively analyze SFT-fine-tuned models. This paper is the first to extend analysis to RL post-training, filling the gap in understanding how different post-training paradigms shape task conflicts.
- **vs Work discussing performance gains of RL post-training**: While prior work focused on RL making models stronger on single tasks, this paper reveals a "by-product" advantage—the morphology of RL parameter updates is naturally suited for merging, offering a new perspective on the value of RL.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematizing the neglected intersection of "Training Paradigm × Model Merging" is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence chain across 5 tasks, 4 merging methods, 3 RL algorithms, 3 base models, plus RFT/RL-Pos variants and theoretical bounds.
- Writing Quality: ⭐⭐⭐⭐ Solid logical progression from phenomenon to mechanism. Minor formatting issues in some formulas do not hinder understanding.
- Value: ⭐⭐⭐⭐ Provides a practical, evidence-based answer for which paradigm to use when training expert models for merging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reward is Enough: LLMs are In-Context Reinforcement Learners](reward_is_enough_llms_are_in-context_reinforcement_learners.md)
- [\[ICLR 2026\] How to Lose Inherent Counterfactuality in Reinforcement Learning](how_to_lose_inherent_counterfactuality_in_reinforcement_learning.md)
- [\[ICLR 2026\] Mirage or Method? How Model–Task Alignment Induces Divergent RL Conclusions](mirage_or_method_how_modeltask_alignment_induces_divergent_rl_conclusions.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[ICLR 2026\] RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?](rl_grokking_recipe_how_does_rl_unlock_and_transfer_new_algorithms_in_llms.md)

</div>

<!-- RELATED:END -->
