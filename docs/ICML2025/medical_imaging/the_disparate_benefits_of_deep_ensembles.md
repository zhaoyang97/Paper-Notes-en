---
title: >-
  [Paper Note] The Disparate Benefits of Deep Ensembles
description: >-
  [ICML 2025][Medical Imaging][Deep Ensembles] Through large-scale empirical studies on facial analysis and medical imaging datasets, this paper reveals a neglected phenomenon—the "disparate benefits effect": while Deep Ensembles improve overall performance, they **disproportionately** benefit different protected groups (often favoring already advantaged groups), thereby undermining group fairness. The authors further attribute the root cause of this to **disparities in predict…
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Deep Ensembles"
  - "Algorithmic Fairness"
  - "Prediction Diversity"
  - "Model Calibration"
  - "Post-processing"
date: 2026-05-08
content_hash: 1ba146348b78a43d
---

# The Disparate Benefits of Deep Ensembles

**Conference**: ICML 2025  
**arXiv**: [2410.13831](https://arxiv.org/abs/2410.13831)  
**Code**: None  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: Deep Ensembles, Algorithmic Fairness, Prediction Diversity, Model Calibration, Post-processing

## TL;DR
Through large-scale empirical studies on facial analysis and medical imaging datasets, this paper reveals a neglected phenomenon—the "disparate benefits effect": while Deep Ensembles improve overall performance, they **disproportionately** benefit different protected groups (often favoring already advantaged groups), thereby undermining group fairness. The authors further attribute the root cause of this to **disparities in predictive diversity across groups** and demonstrate that the classic Hardt Post-Processing (HPP) can effectively repair fairness while preserving performance gains.

## Background & Motivation

**Background**: Deep ensembles (Lakshminarayanan et al., 2017) are a standard "simple yet robust" method for enhancing Deep Neural Network (DNN) performance and estimating prediction uncertainty—independently training $N$ DNNs with the same architecture initialized with different random seeds, and averaging their predictive distributions:

$$p(y \mid \bm{x}, \mathcal{D}) \approx \frac{1}{N}\sum_{n=1}^{N} p(y \mid \bm{x}, \bm{w}_n), \quad \bm{w}_n \sim p(\bm{w} \mid \mathcal{D})$$

It is widely deployed in high-stakes scenarios such as healthcare, finance, and law.

**Limitations of Prior Work**: In these high-stakes scenarios, the fairness of model performance across different protected groups (divided by sensitive attributes such as gender, age, and race) is crucial. While group fairness for a single DNN has been extensively studied, **the impact of ensembles themselves on fairness has rarely been systematically investigated**. It is often assumed that "ensembles only make models better," without asking: does it benefit all groups equally?

**Key Challenge**: The performance dividend brought by ensembles does not appear out of thin air to be distributed equally; it relies on the "predictive diversity" among member models—and this diversity **is not identical** across different groups. If the members are more "diverse" for the advantaged group, the improvement of ensembles on this group remains larger, while the disadvantaged group is left behind, resulting in **improved performance but decreased fairness**.

**Goal**: (1) Empirically characterize how the performance dividends of deep ensembles are distributed among protected groups; (2) identify the root cause of this imbalance; (3) find a way to remedy fairness without retraining the members.

**Key Insight**: Unlike the closest prior work by Ko et al. (2023)—which defines "groups" as the best/worst performing subsets in the target space and only looks at per-group accuracy, concluding that ensembles "only have positive effects"—this paper defines groups using **real protected attributes** and adopts group fairness metrics widely recognized in the algorithmic fairness field, directly challenging the optimistic view that "ensembles only bring benefits."

**Core Idea**: By placing "ensemble performance improvement" and "group fairness" in the same frame of reference, the study discovers they move in opposite directions (the disparate benefits effect), explains the cause using predictive diversity, and applies post-processing calibration thresholds to address the issue.

## Method

This paper presents a three-stage study consisting of **empirical analysis + mechanistic explanation + mitigation solution**. Instead of proposing a new model, it designs a rigorous experimental protocol to define, locate, explain, and repair the "disparate benefits effect." The logical chain unfolds as "how to measure fairness $\rightarrow$ what effect is found $\rightarrow$ why this happens $\rightarrow$ how to fix it."

### Overall Architecture

The entire study is established on a binary classification setting: input $\bm{x}\in\mathbb{R}^D$, label $y\in\{0,1\}$ ($y=1$ denotes a positive outcome), and group attribute $a\in\{0,1\}$ (with $a=1$ designated as the advantaged group). Performance is measured by accuracy (for facial datasets) or AUROC (for medical datasets); fairness is measured by the "violation level" of three complementary group fairness metrics, all normalized to $[0,1]$, where **0 indicates perfect fairness**:

- **Statistical Parity Difference (SPD)**: $\text{PR}_{A=1}-\text{PR}_{A=0}$, which is the difference in positive prediction rates between the two groups, requiring that "positive prediction" is independent of group attributes.
- **Equal Opportunity Difference (EOD)**: $\text{TPR}_{A=1}-\text{TPR}_{A=0}$, which compares the true positive rate (TPR) conditioned on the true label being positive.
- **Average Odds Difference (AOD)**: $\frac{1}{2}|\text{TPR}_{A=1}-\text{TPR}_{A=0}| + \frac{1}{2}|\text{FPR}_{A=1}-\text{FPR}_{A=0}|$, which constrains both the true positive rate and false positive rate (FPR), acting as a relaxed estimation of equalized odds.

Experimental scale: 3 vision datasets (FairFace=FF, UTKFace=UTK, CheXpert=CX), 5 DNN architectures (ResNet18/34/50, RegNet-Y 800MF, EfficientNetV2-S), 4 domains/targets $\times$ 5 seeds $\times$ 10 members, totaling **1000 independently trained models** covering **15 tasks**; the main results in the text use ResNet50. The overall pipeline is: first compare the difference $\Delta$ in performance/fairness between a "10-member deep ensemble" and the "average single member" (Step 1, discovering the effect) $\rightarrow$ dissect PR/TPR/FPR and predictive diversity to explain the cause (Step 2, attribution) $\rightarrow$ repair fairness using weighted sum and threshold-based post-processing pathways (Step 3, mitigation).

### Key Designs

**1. The Disparate Benefits Effect: Placing performance and fairness in the same coordinate system, revealing how ensembles "favor the rich and neglect the poor"**

The core quantity measured by the authors is the change "before and after adding members" $\Delta$—the difference between the 10-member deep ensemble and the average single member in accuracy/AUROC and SPD/EOD/AOD (deemed significant by t-test over 5 runs, with $p<0.05$). The key observation is: while the performance $\Delta$ is **always positive** (ensembles inevitably improve performance), the fairness violation $\Delta$ **does not necessarily decrease, and often even increases**. In 4 out of 6 "target/protected attribute" combinations, significant fairness degradation occurred, and this degradation **almost exclusively took place in tasks where single members already exhibited obvious fairness violations (violation value >0.05)**. In other words, ensembles do not distribute performance dividends equally among everyone but tend to further boost the already advantaged groups—hence the term "disparate benefits." The value of this finding lies in its counter-intuitive nature: practitioners usually treat ensembles as a free lunch that "makes things blindly better," but this paper proves it may quietly widen group disparities.

**2. Predictive Diversity Disparities: Explaining the cause via the gap in $\overline{\text{DIV}}$ across groups**

To explain the effect, the authors first decompose the fairness metrics back into their underlying per-group quantities: PR, TPR, and FPR. Taking "target=age, protected attribute=gender" on FF as an example: when adding members, the TPR of the advantaged group (males) increases while its FPR decreases, leaving the net PR almost unchanged; for the disadvantaged group (females), the TPR remains unchanged while its FPR also decreases, causing its PR to drop—consequently widening both SPD and EOD. However, this is only a phenomenological explanation. The deeper root cause is attributed by the authors to the **average predictive diversity of groups** $\overline{\text{DIV}}$. Following the definition by Jeffares et al. (2023), it equals the "ensemble log-likelihood" minus the "average member log-likelihood":

$$\overline{\text{DIV}} = \frac{1}{K}\sum_{k=1}^{K}\left[\underbrace{\log\!\left(\frac{1}{N}\sum_{n=1}^{N} p(y{=}y_k \mid \bm{x}_k, \bm{w}_n)\right)}_{\text{ensemble log-likelihood}} - \underbrace{\frac{1}{N}\sum_{n=1}^{N}\log p(y{=}y_k \mid \bm{x}_k, \bm{w}_n)}_{\text{average member log-likelihood}}\right]$$

Intuitively, $\overline{\text{DIV}}$ measures "how differently the member models predict"—the higher the $\overline{\text{DIV}}$ of a group, the more improvement space can be **exploited from the ensemble** for that group. The authors find that: for all tasks exhibiting significant disparate benefits, the gap in $\overline{\text{DIV}}$ between groups is large; conversely, in tasks where fairness is unaffected (equal benefits), the $\overline{\text{DIV}}$ between groups is almost identical. This attributes "why fairness is compromised" to a quantifiable difference in member diversity.

**3. Two Controlled Synthetic Experiments: Establishing the causal link "diversity disparity $\rightarrow$ disparate benefits"**

Because correlation alone is insufficient, the authors build a controlled causal experiment using FashionMNIST. The first experiment targets a binary classification of "T-shirt vs Shirt." For group $A=0$, the input consists of the same image concatenated twice (providing no extra information, low member diversity); for $A=1$, it consists of two different images with the same label concatenated (members can learn top, bottom, or combined features, high diversity). The results perfectly replicate the phenomenon observed in real-world data: $A=1$ has a higher $\overline{\text{DIV}}$, and it is precisely this group that experiences an increase in TPR and a decrease in FPR when adding members, while $A=0$ remains virtually unchanged—the high-diversity group exclusively enjoys the ensemble dividend. The second experiment further **continuously adjusts diversity**: using a linear interpolation coefficient $\alpha$ to transition between "pure random noise concatenation ($\alpha=0$, equivalent groups)" and "concatenating another image with the same label ($\alpha=1$, maximum diversity)," and defines a diversity score as $|\overline{\text{DIV}}_{Y=1,A=1}-\overline{\text{DIV}}_{Y=1,A=0}| + |\overline{\text{DIV}}_{Y=0,A=1}-\overline{\text{DIV}}_{Y=0,A=0}|$. The results show that: larger $\alpha$ $\rightarrow$ higher diversity score $\rightarrow$ and concurrently larger $\Delta$Accuracy, $\Delta$SPD, $\Delta$EOD, and $\Delta$AOD brought by ensembles, with the three highly correlated. This pair of experiments upgrades the hypothesis that "diversity disparity causes disparate benefits" to a conclusion validated by controlled testing.

**4. Hardt Post-Processing (HPP): Utilizing the ensemble’s "better calibration" for group-specific threshold optimization to restore fairness**

Regarding mitigation strategies, the authors deliberately consider only **post-processing** (avoiding model retraining to save computation). They first tried "non-uniform member weighting"—selecting optimal weights via the validation set, or weighting inversely to the fairness violation—but both options only yielded results that fell between the "uniform ensemble" and "single model" with high variance, which was insufficient. The real breakthrough came from an observation: because deep ensembles average the predictions of multiple members, they are **better calibrated** (lower Expected Calibration Error, ECE), which in turn makes them **more sensitive to prediction thresholds**—each group has a clear and stable optimal threshold across runs; in contrast, the accuracy of a single member remains almost identical for any threshold between 0.2 and 0.8, with extremely high variance in the optimal value. Based on this, the authors apply the classic Hardt Post-Processing (HPP, optimizing group-specific decision thresholds) to deep ensembles for the first time: setting the target fairness violation to "the average validation set violation of single members," HPP enables the ensemble to pull fairness back to the single-member level **without losing accuracy (even slightly improving it, as it deviates from the default 0.5 threshold implied by argmax)**. HPP fits perfectly with the "highly calibrated" nature of deep ensembles, making it more effective than applying HPP to single members—completing a beautiful closed loop from "causal analysis" to "precision targeting."

## Key Experimental Results

### Main Results: Disparate Benefits Effect (ResNet50, 10-member ensemble vs average single member $\Delta$)

The table below shows representative tasks exhibiting **significant negative effects** (performance ↑ while fairness violation ↑). Note: Performance $\Delta$ is the larger the better; a positive $\Delta$ for fairness metrics (SPD/EOD/AOD) indicates a widened violation (more unfair). All listed results are statistically significant ($p<0.05$).

| Dataset | Target / Protected Attribute | $\Delta$ Performance (↑) | $\Delta$ SPD | $\Delta$ EOD | $\Delta$ AOD |
|--------|------------------|------------------|--------------|--------------|--------------|
| FF | age / gender | +.022 (Acc) | +.022 | +.017 | +.017 |
| FF | age / race | +.022 (Acc) | +.009 | +.012 | +.007 |
| UTK | age / gender | +.015 (Acc) | +.017 | +.015 | +.012 |
| UTK | age / race | +.015 (Acc) | +.010 | +.010 | +.004 |
| CX (Medical) | age | +.005 (AUROC) | +.001 | +.008 | +.003 |

As shown, the effect is most pronounced on face datasets (FF/UTK), and it also exists on the medical imaging dataset CX but with a smaller magnitude (since the ensemble performance gains on this dataset are inherently smaller); UTK evaluates the model trained on FF under distribution shift, which shows a higher baseline of single-member fairness violations, but the magnitude and behavior of the effect remain consistent with FF.

### Control Experiment: The effect is "conditional" and not universally occurring

In the same Table 1 of the paper, other tasks (mostly with gender as the target) show almost no disparate benefits, or even show that the ensemble is fairer ($\Delta$ is negative), corroborating the conclusion that "the effect is only significant when the diversity/fairness gap between groups is large."

| Dataset | Target / Protected Attribute | $\Delta$ Performance (↑) | $\Delta$ SPD | $\Delta$ EOD | $\Delta$ AOD | Remarks |
|--------|------------------|------------------|--------------|--------------|--------------|------|
| FF | gender / age | +.014 (Acc) | -.001 (ns) | -.007 | -.004 | Ensemble is fairer |
| UTK | gender / age | +.009 (Acc) | +.001 (ns) | -.006 | -.003 | Ensemble is fairer |
| CX | gender | +.005 (AUROC) | ~.000 (ns) | +.001 (ns) | -.001 (ns) | Fairness essentially unchanged |
| CX | race | +.005 (AUROC) | -.002 | ~.000 (ns) | -.001 | Individual metrics are fairer |

> ns = not significant. The gap in group-wise $\overline{\text{DIV}}$ for the control group is tiny, corresponding directly to "no disparate benefits."

### Key Findings
- **Effect Triggering Conditions**: Disparate benefits almost exclusively occur in tasks where "single members already exhibit clear fairness violations (>0.05)" and "the gap in predictive diversity $\overline{\text{DIV}}$ between groups is large"; the largest degradation typically happens **when adding the first few members**.
- **Mechanism**: The PR of the disadvantaged group decreases due to the ensemble (TPR remains unchanged, FPR decreases), while the TPR of the advantaged group increases, causing SPD/EOD/AOD to widen; controlled experiments on FashionMNIST demonstrate that diversity disparity is the cause ($\alpha$↑ $\rightarrow$ diversity score↑ $\rightarrow$ synchronous increase of all $\Delta$ values).
- **Scaling Up**: For tasks experiencing this effect, the effect strengthens as the model size increases (Apx. F.2), and the conclusions remain consistent across 5 different architectures; heterogeneous ensembles also exhibit this effect.
- **Mitigation Effectiveness**: Deep ensembles have lower ECE (better calibrated) $\rightarrow$ more sensitive to thresholds $\rightarrow$ when HPP sets the target violation to the average single member's violation, the ensemble restores fairness without dropping accuracy (or even slightly improving it); in contrast, the "member weighting" schemes are unstable, exhibit high variance, and have limited effectiveness.

## Highlights & Insights
- **Connecting two parallel tracks of "ensembles" and "fairness"**: The industry treats ensembles as an unconditionally beneficial tool; this paper dispels the "free lunch" myth with a single "member count vs (performance, fairness)" chart, assigning a memorable name "disparate benefits effect" to the phenomenon—a classic case of "discovering a new problem by shifting the observational dimension."
- **$\overline{\text{DIV}}$ is a transferable diagnostic metric**: Mapping the group fairness problem to "gaps in the average predictive diversity of groups" provides a quantifiable probe to predict which tasks will run into issues; this approach can be transferred to any high-stakes scenario employing ensembles / MC dropout / Bayesian approximations to conduct fairness evaluations.
- **An elegant closed loop from cause to mitigation**: By deducing "more sensitive to thresholds" from the known property of "ensembles being better calibrated," the study naturally bridges classic HPP over—no retraining, low overhead, and preserving performance. This is a prime example of "finding a low-cost solution only after understanding the mechanism."
- **Cleverly designed controlled synthetic experiments**: Using "concatenating identical/different images" and "noise interpolation $\alpha$," the abstract concept of "predictive diversity" is transformed into a manually adjustable dial, cleanly validating the causal relationship and offering a great reference for future research.

## Limitations & Future Work
- **Limitations acknowledged by the authors**: The research only covers vision tasks (convolutional DNN ensembles); the three group fairness metrics, while popular, are not sufficient to guarantee real-world fairness; a single intervention like HPP cannot guarantee fairness on its own.
- **Self-identified limitations**: Group attributes are forced into binary representations (e.g., race as white vs non-white, age split at 40 years old), which might mask fine-grained unfairness; the main text conclusions rely on ResNet50, while cross-architecture quantitative consistency is mainly placed in the appendix; mitigation only validates post-processing without end-to-end fairness-performance trade-off comparisons against in-/pre-processing retraining methods.
- **Potential extensions**: Extend to language models and tabular/sequential data; incorporate other fairness notions such as individual fairness; study whether disparate benefits still occur when single members are first subjected to fairness interventions and then ensembled; incorporate $\overline{\text{DIV}}$ disparities as regularization signals during training to suppress the effect from the source rather than patching it afterward.

## Related Work & Insights
- **vs Ko et al. (2023)**: The closest prior work, which defines "groups" as the best/worst performing subsets in the target space and only looks at per-group accuracy, concluding that ensembles "only have positive effects." This paper defines groups of real protected attributes and adopts standard group fairness metrics, directly proving that ensembles can **harm** fairness, and complements this with attribution analysis and mitigation schemes.
- **vs Fairness in Shallow ensembles (Kamiran & Calders 2012; Kenfack et al. 2021; Gohar et al. 2023; Bhaskaruni et al. 2019)**: Previous ensemble fairness studies almost exclusively focused on shallow models, and mostly used ensembles to **improve** fairness (e.g., fairness-aware weighting, AdaBoost variants). This work is the first to systematically study the impact of **deep ensembles** on group fairness metrics, yielding a conclusion in the opposite direction—deep ensembles spontaneously exacerbate unfairness.
- **vs Predictive Diversity Studies (Abe et al. 2022b/2024; Jeffares et al. 2023)**: Prior works utilized predictive diversity to explain **why ensembles perform better**; this paper transfers the same concept to "why fairness degrades" and refines it into "gaps in predictive diversity across groups."
- **vs Hardt Post-Processing (Hardt et al. 2016; Cruz & Hardt 2024)**: HPP is a classic threshold post-processing method that has never been applied to deep ensembles before. This paper notes that deep ensembles "are better calibrated $\rightarrow$ more suitable for threshold optimization," perfectly matching HPP with the properties of deep ensembles.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal and name the disparate benefits effect of deep ensembles, which is counter-intuitive and practically significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets $\times$ 5 architectures $\times$ 15 tasks $\times$ 1000 models, plus two controlled synthetic experiments to cement causal factors.
- Writing Quality: ⭐⭐⭐⭐ Clear closed loop of "discovering effect $\rightarrow$ attribution $\rightarrow$ mitigation," though some quantitative results and graphs rely heavily on the appendix.
- Value: ⭐⭐⭐⭐⭐ Directly warns against blindly using ensembles in high-stakes scenarios and provides a low-cost, readily deployable HPP remedy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Noise Calculation in Deep Learning-based MRI Reconstructions](efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)
- [\[ICCV 2025\] Semi-supervised Deep Transfer for Regression without Domain Alignment](../../ICCV2025/medical_imaging/semi-supervised_deep_transfer_for_regression_without_domain_alignment.md)
- [\[NeurIPS 2025\] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](../../NeurIPS2025/medical_imaging/dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](../../CVPR2025/medical_imaging/automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](../../CVPR2025/medical_imaging/reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)

</div>

<!-- RELATED:END -->
