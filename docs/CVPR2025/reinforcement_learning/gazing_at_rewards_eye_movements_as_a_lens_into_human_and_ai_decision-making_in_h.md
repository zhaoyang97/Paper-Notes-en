---
title: >-
  [Paper Note] Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging
description: >-
  [CVPR 2025][Reinforcement Learning][Visual search] The Visual Forager (VF) model is proposed, which simulates human eye-movement strategies in hybrid visual search tasks through target feature modulation, target value modulation, and a ViT-based Actor-Critic decision-making network. It achieves a normalized score of 72.6% (compared to 87.4% for humans), with a saccade amplitude difference of only 0.01° (4.06° vs. 4.05° for humans), revealing for the first time how target valu…
tags:
  - "CVPR 2025"
  - "Reinforcement Learning"
  - "Visual search"
  - "eye movement modeling"
  - "gaze strategy"
  - "hybrid search"
date: 2026-05-08
content_hash: 4caecd7a75c3043c
---

# Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging

**Conference**: CVPR 2025  
**arXiv**: [2411.09176](https://arxiv.org/abs/2411.09176)  
**Code**: [https://github.com/ZhangLab-DeepNeuroCogLab/visual-forager](https://github.com/ZhangLab-DeepNeuroCogLab/visual-forager)  
**Area**: Reinforcement Learning  
**Keywords**: Visual search, eye movement modeling, reinforcement learning, gaze strategy, hybrid search

## TL;DR
The Visual Forager (VF) model is proposed, which simulates human eye-movement strategies in hybrid visual search tasks through target feature modulation, target value modulation, and a ViT-based Actor-Critic decision-making network. It achieves a normalized score of 72.6% (compared to 87.4% for humans), with a saccade amplitude difference of only 0.01° (4.06° vs. 4.05° for humans), revealing for the first time how target value and prevalence jointly influence human search decisions.

## Background & Motivation

**Background**: Hybrid visual search (searching for multiple target categories simultaneously, each with different values and prevalence rates) is a core capability in daily human activities (e.g., supermarket shopping, driving scanning). Existing computational models primarily focus on simple, fixed-target searches, ignoring the influence of target value on eye-movement strategies.

**Limitations of Prior Work**: There is a lack of a unified computational model that simultaneously accounts for target feature matching, target value preference, and fixation decision-making. Existing RL search models do not simulate human foveated vision (where resolution decreases further away from the fixation point).

**Key Challenge**: Humans optimize multiple objectives simultaneously during visual search—finding relevant items, prioritizing high-value targets, and managing attentional resources. How to formally model this multi-objective trade-off remains an open question.

**Goal**: To establish a computational model capable of simulating human hybrid visual search, revealing the joint influence mechanism of target value and prevalence on eye-movement decisions.

**Key Insight**: Modeling search as an RL problem, where the agent maximizes rewards through fixation actions (saccades) and clicking actions (picking up targets), using VGG16 with eccentricity-dependent pooling to simulate foveated vision.

**Core Idea**: Utilizing target feature modulation + value modulation + an RL Actor-Critic framework to model fixation-click decisions in hybrid visual search in a unified manner, reproducing human value preferences and saccadic patterns.

## Method

### Overall Architecture
Given the search scene image and target templates, VGG16 extracts features which are then processed by eccentricity-dependent pooling (simulating foveated vision). These features are fused with target feature similarity maps and target value encodings, and then fed into a ViT Actor-Critic network to output a fixation location probability map and click action probability.

### Key Designs

1. **Target Feature Modulation + Eccentricity-Dependent Pooling**:

    - **Function**: Simulates target matching under human foveated vision.
    - **Mechanism**: VGG16 extracts search scene features, which are processed via multi-level pooling to simulate eccentricity—pooling becomes coarser (lower resolution) further away from the current fixation point. Then, the similarity between target templates and the scene feature maps is computed to generate spatial matching heatmaps for multiple targets.
    - **Design Motivation**: Human vision has the highest resolution in the fovea, which drops sharply in the periphery. This determines the saccadic strategy—gaze must be shifted to resolve peripheral objects.

2. **Target Value Modulation**:

    - **Function**: Integrates target value information into decision-making.
    - **Mechanism**: The value of each target is converted into an embedding vector via a learned encoder and element-wise added to the feature matching map. Matching signals of high-value targets are amplified, while those of low-value targets are suppressed.
    - **Design Motivation**: Humans prioritize high-value targets during search—e.g., looking for milk rather than napkins first during breakfast.

3. **ViT Actor-Critic Decision-Making Network**:

    - **Function**: Outputs fixation locations and click decisions.
    - **Mechanism**: ViT processes the fused feature maps; the Actor head outputs the spatial fixation probability distribution (determining where to look next), and the Critic head estimates the state value. It is optimized using PPO during training. A separate click action head decides whether to pick up the item at the current fixation location.
    - **Design Motivation**: The global attention of ViT integrates distant target information, helping plan efficient saccade pathways.

### Loss & Training
PPO reinforcement learning is used, where the reward signal is the value of successfully picked targets combined with a step-wise penalty of $-0.01$. Fifteen human subjects yielding 750 search trials, 50,514 fixations, and 12,851 clicks serve as the evaluation benchmark.

## Key Experimental Results

### Main Results

| Method | Normalized Score (UnValEqPre) | Saccade Amplitude |
|------|----------------------|---------|
| Human | 87.4% | 4.05° |
| VF (Ours) | 72.6% | 4.06° |
| FeatOnly | 52.3% | - |
| MaxVal | 44.7% | - |
| DQN | 36.5% | - |

### Key Findings
- Gaze heatmaps of VF are highly consistent with humans—both prioritize fixating on high-value and high-match regions.
- Both humans and VF exhibit a "Click Bias Ratio"—showing a higher tendency to click on high-value targets.
- Removing the eccentricity-dependent pooling leads to significant changes in saccade amplitude, validating the decisive impact of foveated vision on saccade strategies.
- VF maintains reasonable search behavior under OOD conditions (unseen targets, unseen values, and unseen scene sizes).

## Highlights & Insights
- **Intersection of Cognitive Science and CV**: Temporally and quantitatively reproducing the value-attention trade-off mechanism in human hybrid visual search using an RL model for the first time.
- **Precise Saccade Amplitude Matching**: VF 4.06° vs. Human 4.05°, demonstrating that foveated vision modeling is key to saccade strategies.
- **Interpretability**: Revealing the computational principles of search decision-making by comparing human and model fixation/click patterns.

## Limitations & Future Work
- The performance gap compared to humans is still around 15%, potentially because humans possess better working memory to avoid repetitive searches.
- Currently, target values are assumed to be known; in reality, values may need to be inferred.
- Validated only in 2D static scenes; real-world 3D dynamic search is significantly more complex.

## Related Work & Insights
- **vs. DQN Search Models**: DQN achieves only 36.5% in visual search compared to VF's 72.6%, because DQN does not model foveated vision or value modulation.
- **vs. Saliency Models**: Saliency models predict "where" to look but not "what to do"; VF unifies fixation and clicking decisions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ High cognitive inspiration, modeling value-attention interactions in hybrid visual search for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on data from 15 subjects, multi-condition ablation, and OOD generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear comparative analysis between cognitive science and computational models.
- Value: ⭐⭐⭐⭐ Significant value for active visual search and cognitive modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICLR 2026\] Toward Conservative Planning from Human-AI Preferences in Reinforcement Learning](../../ICLR2026/reinforcement_learning/toward_conservative_planning_from_human-ai_preferences_in_reinforcement_learning.md)
- [\[ICLR 2026\] Bayesian Ensemble for Sequential Decision-Making](../../ICLR2026/reinforcement_learning/bayesian_ensemble_for_sequential_decision-making.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](../../ICML2025/reinforcement_learning/enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](../../ICML2025/reinforcement_learning/counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)

</div>

<!-- RELATED:END -->
