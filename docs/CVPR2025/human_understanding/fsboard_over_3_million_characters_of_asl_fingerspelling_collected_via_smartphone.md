---
title: >-
  [Paper Note] FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones
description: >-
  [CVPR 2025][Human Understanding][Fingerspelling] Presents FSboard—the largest ASL fingerspelling recognition dataset to date (3.2 million characters, 266 hours of video, recorded via smartphone selfie mode by 147 Deaf signers). Focused on mobile text entry scenarios, the baseline model achieves an 11.1% CER using MediaPipe + ByT5, providing a solid data foundation for fingerspelling as a mobile input method.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Fingerspelling"
  - "ASL recognition"
  - "large-scale dataset"
  - "mobile gesture input"
  - "Deaf community"
date: 2026-05-08
content_hash: 47a29a3fedc8b3e4
---

# FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones

**Conference**: CVPR 2025  
**arXiv**: [2407.15806](https://arxiv.org/abs/2407.15806)  
**Code**: Dataset publicly released (CC BY 4.0)  
**Area**: Others  
**Keywords**: Fingerspelling, ASL recognition, large-scale dataset, mobile gesture input, Deaf community

## TL;DR

Presents FSboard—the largest ASL fingerspelling recognition dataset to date (3.2 million characters, 266 hours of video, recorded via smartphone selfie mode by 147 Deaf signers). Focused on mobile text entry scenarios, the baseline model achieves an 11.1% CER using MediaPipe + ByT5, providing a solid data foundation for fingerspelling as a mobile input method.

## Background & Motivation

**Background**: Sign language translation (specifically ASL to English) quality is steadily improving, but there is still a massive gap before actual usability. Participatory ML methodology suggests breaking down this grand goal into immediate, concrete milestones that benefit the Deaf and Hard-of-Hearing (DHH) community.

**Limitations of Prior Work**: Existing fingerspelling datasets are too small—the previous largest, ChicagoFSWild+, contains only 300K characters and 14 hours of video, consisting of low-resolution cropped clips scraped from the web. The lack of data severely constrains model performance. Furthermore, many so-called "fingerspelling recognition" studies are actually static handshape classification, ignoring the co-articulation effects and word spaces in actual high-speed fingerspelling.

**Key Challenge**: The conflict between data scale and collection cost—high-quality fingerspelling data requires recruiting Deaf signers, and standardizing hardware and scenarios, which is expensive; however, model performance heavily relies on data scale and diversity.

**Goal**: (1) Build an ASL fingerspelling dataset that is over 10 times larger than the largest existing dataset; (2) focus the dataset on the practical application scenario of mobile text entry; (3) provide high-quality baseline models to demonstrate the dataset's efficacy.

**Key Insight**: Inspired by the user study of Hassan et al.—the speed of Deaf individuals typing mobile text via fingerspelling (42.5 wpm) is faster and has fewer errors than using a touchscreen keyboard (31.9 wpm). This indicates real application value for a fingerspelling keyboard, but it requires large-scale data to train reliable recognition models.

**Core Idea**: By recruiting 147 Deaf participants to record fingerspelling via selfie mode on Pixel 4A phones, the largest ASL fingerspelling dataset to date is constructed, covering mobile text entry scenarios such as MacKenzie phrases, URLs, addresses, phone numbers, and names.

## Method

### Overall Architecture

The core contribution of FSboard is the dataset itself. The workflow is as follows: (1) Design a phrase distribution tailored for mobile text entry scenarios; (2) recruit 147 Deaf participants with ASL as their primary language via the Deaf Professional Arts Network (DPAN); (3) record single-handed fingerspelling videos under diverse environments using the front-facing camera of Pixel 4A (1944×2592, 30fps); (4) clean data boundaries through a multi-round bootstrapping method; (5) divide the dataset into non-overlapping train/val/test sets. The baseline model uses MediaPipe Holistic to extract 85 3D hand/body landmarks, which are linearly projected and fed into a ByT5-Small (300M) character-level encoder-decoder model for sequence-to-sequence fingerspelling recognition.

### Key Designs

1. **Multi-domain phrase design**:

    - **Function**: Construct a phrase distribution close to real-world mobile text entry scenarios.
    - **Mechanism**: Contains five categories of phrases—MacKenzie classic text input evaluation phrases (500 phrases, recorded repeatedly by multiple signers as a sanity check), randomly generated URLs (reconstructed from randomly combined real URL parts scraped from web crawlers), random US street addresses (based on Census Bureau TIGER data), random personal names (combinations of top 1000 first/last names), and random phone numbers (including US and international formats).
    - **Design Motivation**: Mobile text entry involves more than just ordinary sentences; addresses, names, and URLs are application scenarios highly anticipated by the DHH community (e.g., entering addresses in Google Maps). Although numeric signing is not exactly fingerspelling, it is closely related in practical scenarios.

2. **Bootstrapping data cleaning pipeline**:

    - **Function**: Correct inaccurate clip boundaries caused by bugs in the data collection app.
    - **Mechanism**: First, pre-train ByT5 on YouTube sign language videos, then perform predictions on FSboard using 5-fold cross-validation—clips where model predictions match the labels are marked as "clean", and mismatched ones are marked as "noisy". This process is repeated three times, training only on the "clean" data each time. Afterwards, manual editing and participant-specific rules are applied for further cleaning.
    - **Design Motivation**: Due to a "clear/reset" button bug in the collection app, mismatches between timestamp labels and actual fingerspelling were widespread. Since manual annotation was prohibitively expensive, the bootstrapping method leverages the model's own capabilities to iteratively filter clean data.

3. **Baseline Model: MediaPipe + ByT5**:

    - **Function**: Provide baseline performance for fingerspelling recognition.
    - **Mechanism**: MediaPipe Holistic is used to extract 85 3D landmarks (hand + pose + face) at 30Hz, which are linearly projected into the encoder input space of ByT5-Small, with one soft token per frame. It supports an input of up to 256 frames and an output of 256 characters, utilizing beam search (beam=5) decoding. The model is trained for 200K steps on 32 TPUv3.
    - **Design Motivation**: Landmark input is much lighter than directly processing videos, making it suitable for future on-device mobile deployment. The character-level ByT5 model naturally suits fingerspelling (letter-by-letter recognition), and its pre-trained knowledge yields significant improvements.

### Loss & Training

Standard encoder-decoder sequence-to-sequence training is used with the Adafactor optimizer, a learning rate of 0.001, and a batch size of 64. The optimal checkpoint is selected based on the validation set CER.

## Key Experimental Results

### Main Results

| Model/Configuration | CER↓ | Top-1 Accuracy↑ |
|-----------|------|-----------------|
| **ByT5-Small + MediaPipe (30Hz)** | **11.1%** | **52.9%** |
| Best Kaggle Competition | 16.4% | - |
| ChicagoFSWild+ Baseline | 37.7% | - |
| ChicagoFSWild+ Human Performance | 13.9% | - |

### Ablation Study

| Configuration | CER↓ | Top-1 Accuracy↑ | Note |
|------|------|-----------------|------|
| ByT5 pre-training (300M) | 11.1% | 52.9% | Full Baseline |
| ByT5 trained from scratch | 33.8% | 17.9% | Pre-training knowledge contributes 22.7% CER |
| ByT5 Base (580M) | 13.3% | 49.1% | Larger model overfits instead |
| 30Hz → 15Hz | 11.8% | 51.8% | Frame rate halved, CER increases by only 0.7% |
| 30Hz → 5Hz | 20.0% | 33.4% | Frame rate too low, performance degrades sharply |
| Remove face landmarks | 12.0% | 50.6% | Facial lip-reading cues provide a minor boost |
| Remove face + pose landmarks | 12.5% | 49.7% | Performing well with only hand landmarks |

### Key Findings

- **Pre-training is the most critical factor**: The CER gap between ByT5 pre-training vs. training from scratch is 22.7% (11.1% vs. 33.8%), and the convergence speed is about 7× faster.
- **Reducing frame rate to 15Hz incurs almost no loss** (11.1% → 11.8%), which is highly favorable for real-time mobile deployment.
- **Facial/pose landmarks contribute minimally** (removing them only increases CER by 1.4%), indicating that the core information of fingerspelling recognition resides in the hands, while the face might only provide auxiliary lip-reading cues.
- The baseline has already **outperformed human performance on ChicagoFSWild+** (11.1% < 13.9%), but this is primary because FSboard's task setup targets isolated fingerspelling rather than cropping from continuous signing.

## Highlights & Insights

- **Community-driven dataset design** is the most important philosophy in this paper: Three of the authors are members of the Deaf community, and community involvement spanned the selection of the topic, pilot testing, and recruitment. The dataset is focused on real needs (mobile text entry) rather than academic showmanship. This incremental participatory action approach of "solving practical needs before pursuing grander goals" is worth emulating by all HCI/AI researchers.
- **The fingerspelling speed on MacKenzie phrases far exceeds mobile typing** (65 wpm vs. 36 wpm), with some signers exceeding 100 wpm. This data strongly supports the practical value of a fingerspelling keyboard.
- **The bootstrapping data cleaning pipeline** cleverly utilizes the model itself as an annotation tool. Although it is less precise than manual annotation, the cost is extremely low and it can be iteratively improved, making it applicable to all remote crowdsourced data collection scenarios.

## Limitations & Future Work

- The phrase distribution of the dataset is narrow—it only covers engineered addresses, URLs, names, etc., and lacks free text and daily conversation content.
- The handling of capitalization/punctuation is not sufficiently explicit—capitalization is distinguished in only a few scenarios in fingerspelling, but text entry requires it, necessitating more precise collection protocols.
- MediaPipe-based landmark extraction can be unreliable under occlusion and extreme angles; future work should explore direct video modeling.
- The test set still consists of phrases using the same synthetic grammar as the training set; an independent, real-world query test set would more accurately reflect practical performance.
- The number of signers (147 people) is much less than ChicagoFSWild+ (260 people). Although racial/gender diversity is reasonable, it is still biased (males are underrepresented).

## Related Work & Insights

- **vs ChicagoFSWild+**: ChicagoFSWild+ crawls low-resolution videos from YouTube, with an average sequence length of only 5.5 characters; FSboard uses high-resolution mobile selfies, with an average sequence length of 21.2 characters. The scale difference is over 10×.
- **vs PopSign**: PopSign is an educational game dataset for isolated sign recognition with a vocabulary of 250 signs. It differs from the fingerspelling sequence recognition task but shares the Pixel 4A selfie collection method.
- **vs ASL Citizen**: ASL Citizen focuses on sign language recognition in a dictionary-lookup setting (2731 signs), utilizing top-N retrieval rather than sequence transcription. FSboard's character-level sequence-to-sequence task is more challenging.

## Rating

- Novelty: ⭐⭐⭐ The core contribution is the dataset rather than methodological innovation, but the scale and design philosophy of the dataset themselves are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations (frame rate, landmark components, model size, pre-training) and qualitative analyses are provided.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper provides very detailed and responsible accounts of the community background, ethical considerations, and dataset design motivations.
- Value: ⭐⭐⭐⭐ Validated as a valuable resource for ASL fingerspelling recognition and assistive technologies. The community-driven methodology sets an exemplary standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[ICLR 2026\] BANZ-FS: BANZSL Fingerspelling Dataset](../../ICLR2026/human_understanding/banz-fs_banzsl_fingerspelling_dataset.md)
- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](../../CVPR2026/human_understanding/openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[CVPR 2025\] Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)

</div>

<!-- RELATED:END -->
