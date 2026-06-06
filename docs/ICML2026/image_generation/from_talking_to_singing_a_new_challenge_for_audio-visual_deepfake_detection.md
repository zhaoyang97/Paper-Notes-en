---
title: >-
  [Paper Note] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection
description: >-
  [ICML 2026][Image Generation][Deepfake] Addressing the "singing head" domain—a challenging area ignored by existing deepfake detectors—the authors construct the SHDF dataset to quantify the "talking-to-singing" domain sh…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Deepfake"
  - "Cross-domain Generalization"
  - "Text-guided"
  - "Singing-driven Avatars"
  - "Multi-modal Fusion"
date: 2026-05-08
content_hash: 956d0e587783c08b
---

# From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection

**Conference**: ICML 2026  
**arXiv**: [2605.27944](https://arxiv.org/abs/2605.27944)  
**Code**: https://LiuKe3068LikWix.github.io/SingingHead-DeepFake/  
**Area**: AI Security / Audio-Visual Deepfake Detection  
**Keywords**: Deepfake, Cross-domain Generalization, Text-guided, Singing-driven Avatars, Multi-modal Fusion

## TL;DR
Addressing the "singing head" domain—a challenging area ignored by existing deepfake detectors—the authors construct the SHDF dataset to quantify the "talking-to-singing" domain shift. They propose the T-AVFD framework, which employs Alpha-CLIP combined with multi-granularity real/fake text contrastive learning to extract "semantic patterns of real faces." A differential weight module adaptively fuses lip-audio consistency with facial semantics. Trained only on real talking videos, the model generalizes to singing forgeries, raising the SHDF AUC from the 50% range to 80.2%.

## Background & Motivation

**Background**: The mainstream paradigm for audio-visual forgery detection utilizes "cross-modal inconsistency"—specifically alignment errors between lip movements and speech. Representative methods like AVAD and AVH-Align are built on the premise that "lips and audio should be strictly synchronized in real videos." Corresponding datasets (FakeAVCeleb, AVLips, TalkingHeadBench, etc.) consist almost entirely of talking heads.

**Limitations of Prior Work**: When the input shifts from talking to singing, rhythmic vocalization, background music, and exaggerated mouth/head movements render the lip-audio alignment signal unstable. Detectors relying on alignment as core evidence fail rapidly. Using the forgery-agnostic AVH-Align for diagnosis, the authors found that the $MMD^2$ domain distance for singing relative to talking is 3.44× to 4.66× larger than between two talking domains. The overlap rate of real/fake score distributions surges from 26.4% to 77.6%, indicating a true "non-trivial domain shift" rather than a simple change in generators.

**Key Challenge**: Cross-modal consistency is naturally weakened in singing scenarios. Simultaneously, there is a lack of singing training data (and singing forgeries cannot be included in training to avoid overfitting to specific generator fingerprints). In other words, the challenge is to find cross-domain forgery cues under the strict constraint of "training only on real talking videos."

**Goal**: (1) Construct the first singing head deepfake benchmark to quantify and expose this shift; (2) Design a detection framework that does not rely on singing training data or forged samples, ensuring generalization across both talking and singing.

**Key Insight**: The authors observe (Figure 2) that whether talking or singing, the semantic representation of real faces is richer and more coherent than that of synthetic faces. This represents an **"authenticity signature" decoupled from the specific scenario**, which is more stable than lip-audio alignment.

**Core Idea**: Use "multi-granularity text contrasting real and fake" to supervise an Alpha-CLIP face encoder, distilling a scenario-independent "facial authenticity semantic pattern." An adaptive differential weight module then decides whether to "trust facial semantics" or "trust lip-audio alignment" based on the content.

## Method

### Overall Architecture
T-AVFD receives three inputs: video frames $\{F_t\}_{t=0}^{T}$ (including face masks), mouth crops $\{M_t\}_{t=0}^{T}$, and audio $\{A_t\}_{t=0}^{T}$, outputting a video-level forgery score $s$. The process follows two branches: (i) **FAPL (Facial Authenticity Pattern Learner)** uses Alpha-CLIP to extract facial semantics $f$, aligning them with multi-granularity positive/negative text pairs in a face-text contrastive space to obtain authenticity patterns $fp$; (ii) **MMDWL (Multi-modal Differential Weight Learning Module)** uses a pre-trained lip-reading model to extract visual features $v$ and audio features $a$, then adaptively fuses $\{fp, v, a\}$ through a weight generator and modulation bias to obtain the score $s$. Training is conducted only on real talking videos using loss $\mathcal{L}=\mathcal{L}_{ft}+\mathcal{L}_{av}$, **without using any synthetic samples**.

### Key Designs

1.  **Facial Authenticity Pattern Learner (FAPL)**:
    - **Function**: Guide the model to learn "what real facial semantics look like" from real faces using real/fake text prompts, shifting authenticity discrimination to the face-text alignment space.
    - **Mechanism**: Alpha-CLIP replaces standard CLIP; it receives an additional face mask and strengthens semantic representations of facial regions via transformer attention $AT_{fm}$ while preserving global context. Features are averaged over time to obtain a stable $f$. Text-side designs include positive/negative pairs at three granularities: face/eyes/mouth (e.g., "a real human face" vs "a fake human face"). $l$ learnable tokens are prepended to each text, encoded by the CLIP text encoder, averaged, normalized, and passed through a shared linear layer to get $p$ and $n$: $p=W(\frac{1}{g_p}\sum_i f_i^p/\|f_i^p\|)$, and similarly for $n$. Finally, the face-text contrastive alignment (FTCA) loss is $\mathcal{L}_{ft}=-\frac{1}{N}\sum\log\frac{\exp(s_i^+)}{\exp(s_i^+)+\exp(s_i^-)}$, where $s^+=f^\top p/\tau$ and $s^-=f^\top n/\tau$, pulling real faces toward $p$ and pushing them away from $n$. During inference, $p$ is concatenated with $f$ to form $fp$.
    - **Design Motivation**: Since the model only sees real faces during training, FTCA learns the "distribution of real patterns." Forged samples are exposed whenever they deviate from this distribution, naturally avoiding overfitting to specific generator fingerprints. The shared linear layer prevents $p$ and $n$ from being projected into unrelated subspaces. Learnable tokens allow the CLIP text end to adapt to the detection task while retaining facial semantic skeletons (Table 7 shows both "fully fixed" and "fully learnable" approaches perform worse).

2.  **Multi-Modal Differential Weight Learning (MMDWL)**:
    - **Function**: Perform content-aware adaptive weighted fusion between "facial authenticity patterns" and "lip-audio alignment" cues, allowing the model to automatically decide which signal to prioritize in different scenarios.
    - **Mechanism**: Visual/audio front-ends $E_v, E_a$ from a pre-trained lip-reading model process mouth sequences and Mel spectrograms, projecting them into $v, a$ with the same dimension as $fp$. The weight generator $\acute{w}=\delta(\mathrm{MLP}(\mathrm{CAT}[a,v,fp]))$ provides relative weights for the three modalities ($\delta$ denotes softmax). A manual modulation bias $\alpha=\{-0.1,+0.1,+0.1\}$ is added to $\{fp,v,a\}$, resulting in $w=\delta(\acute{w}+\alpha)$. Finally, $w$ weights each modal feature to produce the video-level score. $\alpha$ slightly suppresses $fp$ and boosts audio-visual alignment, reflecting a compromise that "acknowledges alignment is still important but cannot be the sole factor."
    - **Design Motivation**: Existing methods (AVH-Align, AVAD) use static uniform fusion, failing to address variations in modal reliability across forgery types. DWL enables the model to automatically increase the weight of $fp$ in singing scenarios (where alignment is unreliable) and return to alignment-dominant weighting in talking scenarios. Table 8 shows that removing DWL causes THB AUC to drop from 93.0 to 80.4 and SHDF AUC from 80.2 to 68.7.

3.  **Loss & Training**:
    - **Function**: Simultaneously optimize "authenticity patterns" and "audio-visual temporal alignment" without synthetic samples, aggregating frame-level scores into video-level scores.
    - **Mechanism**: The audio-visual alignment loss follows the contrastive form from AVAD: $\mathcal{L}_{av}=-\frac{1}{F}\sum_{i=1}^{F}\log\frac{e^{\Phi_{ii}}}{\sum_{k\in T_{(i)}} e^{\Phi_{ik}}}$, requiring the similarity between the $i$-th audio frame and the corresponding video frame to be higher than with negative frames in the temporal neighborhood. The total loss is $\mathcal{L}=\mathcal{L}_{av}+\mathcal{L}_{ft}$ with both coefficients set to 1. Inference uses smoothed max $s=\log\sum_{t=1}^{F}\exp(s_t)$ for aggregation, which is more robust than hard max.
    - **Design Motivation**: Smoothed max ensures that single-frame anomalies are not dominated by a single noisy frame nor diluted by averaging. Uniform weights avoid extra hyperparameters. Training only on real data ensures "unseen forgery types" are detected as anomalies.

## Key Experimental Results

### Main Results

Six baselines were compared across three talking datasets (AVLips, FakeAVCeleb=FKAV, TalkingHeadBench=THB) and the self-built singing dataset SHDF. All unsupervised methods were trained only on real talking data; supervised methods used official weights.

| Dataset | Metric | Ours (T-AVFD) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| AVLips (talking) | AP / AUC | 83.6 / 87.7 | 85.3 / 84.7 (LipFD) | +3.0 AUC |
| FKAV (talking) | AP / AUC | 95.6 / 95.6 | 95.1 / 93.0 (EffViT / AVH-Align) | +2.6 AUC |
| THB (talking, Diffusion) | AP / AUC | 87.6 / 93.0 | 68.7 / 82.3 (RealForensics / AVH-Align) | +10.7 AUC |
| SHDF (singing) | AP / AUC | 85.7 / 80.2 | 67.7 / 50.9 (RealForensics) | +29.3 AUC |

In singing scenarios, all baseline AUCs were around 50% (near random), while T-AVFD reached 80.2%. In talking scenarios, it outperformed the difficult diffusion benchmark THB by ~11 AUC, suggesting "semantic patterns" are more general than "alignment patterns."

### Ablation Study

| Configuration | SHDF AP/AUC | THB AP/AUC | Description |
| :--- | :--- | :--- | :--- |
| Full T-AVFD | 85.7 / 80.2 | 87.6 / 93.0 | Full model |
| w/o texts | 74.6 / 62.0 | 75.2 / 89.5 | Remove all text; AUC drops 18.2 (SHDF) |
| w/ single text | 80.5 / 73.0 | 80.2 / 91.1 | Face only; AUC drops 7.2 |
| w/o face feature | 66.5 / 45.1 | 78.8 / 90.9 | Singing fails; talking stays stable |
| w/o $\mathcal{L}_{ft}$ | 73.2 / 61.3 | 75.0 / 89.5 | No contrastive alignment; prompts ineffective |
| w/o FAPL | 68.8 / 50.6 | 75.8 / 88.9 | Degenerates to alignment; singing ~50% |
| w/o DWL | 76.3 / 68.7 | 66.0 / 80.4 | Uniform fusion; THB more damaged than SHDF |

### Key Findings
- **DWL is more critical in talking than singing**: Removing DWL dropped THB AUC by 12.6 and SHDF by 11.5. This suggests DWL does not just "cover" for singing; it also amplifies the advantage of alignment cues in diffusion-generated talking forgeries.
-  **Necessity of Alpha-CLIP**: Table 4 provides evidence that CLIP's difference between positive/negative text for real/fake faces is near +0.01 (weak discrimination). Alpha-CLIP shows +0.0412 for real and -0.0225 for fake, a "polarity reversal" indicating that mask-guided regional semantics are the true source of discriminative features.
- **Stability Across Training Domains** (Table 3): When the training set is swapped to SHDF real singing, AVH-Align's AUC on AVLips falls to 52.6 (random), while T-AVFD maintains 77.3, proving the "authenticity pattern" is not locked to the training domain.
- **Generator Isolation Experiment** (Table 5): Using a fixed generator (MEMO) to synthesize 500 talking and 500 singing clips, T-AVFD achieved 89.34 AUC (talking) and 79.95 AUC (singing), whereas AVH-Align dropped from 88.0 to 42.4. This is hard evidence of "domain shift, not generator signatures."
- **Robustness**: Under 6 types of perturbations (Gaussian blur, JPEG, inversion, noise, pixelation, scaling), average AUC on THB was 84.6% (compared to 37.8% for AVAD and 43.2% for AVH-Align). Performance under blur/compression/scaling was near-clean, verifying the natural resistance of semantic patterns to low-level degradation.

## Highlights & Insights
- **Reformulating "Real/Fake Binary Classification" as "Contrastive Alignment of Real Distributions"**: Using text as anchors to define "real" semantic regions allows the model to learn decision boundaries without any forged samples, bypassing the dependence of supervised deepfake detection on generator distributions. This "real-as-template" approach is transferable to any field where fake samples are scarce or quickly outdated (e.g., new AIGC content, new attack modes).
- **Quantification of Domain Shift First**: The authors used $MMD^2$ and score overlap rates to clarify that singing is indeed a new domain, followed by fixed-generator control experiments to prove the shift is not due to generator signatures. This logical flow of "qualitative to quantitative to controlled experiment" is highly persuasive.
- **Clever Use of Manual Modulation Bias $\alpha$**: Adding a fixed prior bias on top of adaptive weights essentially injects "domain knowledge (audio-visual alignment is still important)" explicitly before the softmax. It stabilizes performance without adding learnable parameters—a trick worth migrating to other dynamic fusion scenarios.
-  **Compromise of Learnable Tokens + Multi-granularity Fixed Text**: By not making prompts fully learnable (preserving facial-eye-mouth structural priors) nor fully fixed (addressing CLIP’s limitation in real/fake semantics), the result significantly outperforms both extremes (Table 7). This approach is valuable for any downstream task using CLIP text as a discriminative anchor.

## Limitations & Future Work
- The authors do not explicitly name limitations in a separate section; however, practical limitations include: (1) SHDF only includes 80+100 identities, and real singing samples from YouTube may have copyright or diversity bottlenecks, covering only three generators (MEMO, Hallo2, EchoMimic); (2) The modulation bias $\alpha$ is manually set; its suitability across broader "multi-scenarios" (lectures, performances, dramatic recitations) is unknown; (3) Training only on real talking data means any generator that "correctly mimics" the real facial semantic distribution could escape detection; future work requires stress testing with adversarially constructed "semantically real" samples; (4) No solution is provided for frame-level forgery localization; (5) Evaluation is primarily in English and limited languages; the stability of facial authenticity patterns across languages/cultures remains to be explored.
- Improvement ideas: Extend FAPL's authenticity pattern to per-frame fine-grained scores with temporal consistency constraints for localization, or replace Alpha-CLIP with a more robust open-domain multi-modal face foundation model. The modulation bias $\alpha$ could be generated by a small prior network based on audio content (talking/singing/laughter).

## Related Work & Insights
- **vs AVH-Align (CVPR 2025)**: AVH-Align uses self-supervised alignment for unsupervised detection but relies on lip-audio sync. This paper proves its AUC is only 37.4 in singing. T-AVFD uses it as a backbone (lip-reading end) but complements it with FAPL to escape "alignment dictatorship."
- **vs AVAD (CVPR 2023)**: AVAD uses autoregressive modeling for audio-visual sync anomalies; it has no training code and performs poorly across scenarios. This paper borrows the alignment loss $\mathcal{L}_{av}$ as an auxiliary signal but no longer treats it as the protagonist.
- **vs LipFD / RealForensics**: Supervised methods perform well on seen generators (AVLips/FKAV) but crash on THB (diffusion) and SHDF (singing). This proves "memory of forgery fingerprints" does not generalize. T-AVFD’s "real-as-template" route is significantly superior in all cross-domain settings.
- **Insight**: The combination of CLIP text as a "semantic discriminative anchor" and multi-granularity learnable prompts has been used sporadically in medical anomaly or industrial defect detection. This paper systematizes it as a "Facial Authenticity Pattern Learner" and couples it with "alignment loss" through dynamic weights. This framework-level combination may become a universal skeleton for future unsupervised forgery/anomaly detection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematically identifies the "talking-to-singing" domain shift and reshapes forgery detection from an alignment-based paradigm to a "facial authenticity semantic pattern" paradigm. Provides a three-in-one contribution of dataset, method, and domain diagnosis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Tests across 4 datasets × 6 baselines + cross-training domains + fixed generators + 6 perturbations + 7 ablations. Robustly confirms the domain shift and the necessity of each module/text design.
- **Writing Quality**: ⭐⭐⭐⭐ Clear reasoning and self-consistent charts. Detailed sections are dense; choices regarding prompt learnability and $\alpha$ would benefit from more intuitive visualization.
- **Value**: ⭐⭐⭐⭐⭐ Given the explosion of AIGC singing avatars, this new benchmark has platform-level value. The "real-as-template + dynamic fusion" framework provides methodological inspiration for all unsupervised forgery detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](../../ICCV2025/image_generation/deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)
- [\[ICCV 2025\] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](../../ICCV2025/image_generation/float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)
- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)
- [\[ECCV 2024\] Textual-Visual Logic Challenge: Understanding and Reasoning in Text-to-Image Generation](../../ECCV2024/image_generation/textual-visual_logic_challenge_understanding_and_reasoning_in_text-to-image_gene.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](../../CVPR2026/image_generation/markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[ICCV 2025\] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](../../ICCV2025/image_generation/float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)

</div>

<!-- RELATED:END -->
