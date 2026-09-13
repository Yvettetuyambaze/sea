# Sample-Efficient Adaptation of 12-Lead ECG Classifiers Under Distribution Shift

**Milestone 1 — Literature Review**

Yvette Tuyambaze  
Carnegie Mellon University Africa, Kigali, Rwanda  
Course: 04-990 Engineering Research Project  
Supervisor: Prof. Carine Pierrette Mukamakuza  

---

## 1. Scope and review question

Deep 12-lead ECG classifiers are routinely trained on one public corpus and then discussed as if they will transfer to another hospital, device, or population. That assumption is rarely measured for a *fixed* backbone and a *specified* dataset triplet, and it is even more rarely followed by a disciplined answer to the operational question: how much labelled target data is needed before further adaptation stops helping. This review organises the evidence that the project builds on, and the gaps it is designed to close.

The review is organised around five claims in the project proposal:

1. External validation needs a sample-size argument, not a single magic *N*.
2. PTB-XL, MIMIC-IV-ECG, and Chapman-Shaoxing are independent 12-lead resources that make a controlled rotation possible.
3. Cross-dataset ECG performance can change, but the size, significance, and meaning of that change are under-reported.
4. Temperature scaling and parameter-efficient updates are different *classes* of adaptation and must be scored on different outcomes.
5. Few-shot ECG work usually uses absolute *N* grids; a percentage-of-pool grid is more comparable when targets differ by orders of magnitude.

Twenty-eight papers are discussed. Twelve are the proposal’s core citations; sixteen additional papers supply the surrounding evidence on ECG deep learning, multi-source challenges, calibration, transfer, and distribution-shift methodology.

## 2. External validation and “how much data”

Collins, Ogundimu, and Altman (2016) showed, by resampling, that the apparent performance of a prognostic model on an external sample is noisy when that sample is small, and that inadequate validation *N* produces both over-optimistic and over-pessimistic conclusions [1]. Riley et al. (2024) turn that observation into a model-, outcome-, and setting-specific sample-size calculation for external validation [2]. Together they underwrite the project’s refusal to treat *P*_min as a universal clinical number: the amount of target data that is “enough” depends on the model, the metric, the prevalence, and the intended use.

Two related papers sharpen the same point. Steyerberg and Harrell (2016) distinguish internal, internal–external, and fully external validation, and warn that a single random split inside one dataset is not a substitute for a new population [29]. The TRIPOD statement (Collins et al., 2015) requires authors to report the validation setting and sample size explicitly [28]. Van Calster et al. (2019) add that discrimination without calibration is an incomplete validation: a model that ranks well can still assign probabilities that are unusable for decision-making [27].

**Implication for this project.** The audit records rotation, dataset pair, method, percentage, criterion, threshold, and uncertainty. Closing the gap to the source in-distribution AUROC is never a pass/fail criterion, because a persistent gap may be a genuine target-population difference rather than a shortage of labels [1, 2, 27].

## 3. Public 12-lead ECG datasets and the chosen triplet

Wagner et al. (2020) released PTB-XL: 21,837 12-lead records from 18,885 patients, with SCP-ECG statements, recommended diagnostic superclasses (NORM, MI, STTC, CD, HYP), and official stratified folds [3]. It is the standard public benchmark for 12-lead classification.

Zheng et al. (2020) released the Chapman–Shaoxing 12-lead arrhythmia database: 10,646 records at 500 Hz with expert rhythm labels (sinus bradycardia, sinus rhythm, atrial fibrillation, sinus tachycardia, and rarer SVT variants) [5]. The later PhysioNet `ecg-arrhythmia` release expands the same family with Ningbo First Hospital recordings to 45,152 ECGs in WFDB form.

Gow, Goldberger, Reyna, and Moody (2023) released MIMIC-IV-ECG: on the order of 800,000 diagnostic ECGs linked to MIMIC-IV, 500 Hz, under credentialed PhysioNet access [4]. It is the large, multi-year US hospital resource in the proposed triplet. Until credentialing completes, a public US 12-lead stand-in with SNOMED-CT labels is the Georgia 12-lead ECG Challenge set distributed in the 2020/2021 PhysioNet Challenges (about 10,344 records) [18, 19].

Two further corpora matter because they already forced the community to confront label heterogeneity. Liu et al. (2018) organised CPSC 2018, an open 12-lead set with nine rhythm/morphology classes [24]. Alday et al. (2020) and Reyna et al. (2021) then built the PhysioNet/Computing in Cardiology Challenges 2020 and 2021 around a multi-source training pool (CPSC, CPSC-Extra, INCART, PTB, PTB-XL, Georgia, Chapman–Shaoxing, Ningbo) labelled with SNOMED-CT codes [18, 19]. The 2021 scoring notes explicitly equate some codes (complete/incomplete LBBB, CRBBB/RBBB, PAC/SVPB, PVC/VPB) and leave the rest unmapped. That is the harmonization policy this project adopts.

Goldberger et al. (2000) remain the infrastructural citation: PhysioNet is how these datasets are distributed without the project redistributing restricted recordings [23].

**Implication.** The triplet is not interchangeable. PTB-XL is diagnostic-superclass rich; Chapman is rhythm-primary; MIMIC is large, credentialed, and clinically linked. A shared SNOMED subset that is *present with usable counts in every source used for a pair* is the only honest OOD task. Superclass numbers on PTB-XL are kept as the Milestone 2 identity check against Table 2, not as the OOD label space.

## 4. Deep learning for 12-lead ECG classification

Hannun et al. (2019) showed cardiologist-level arrhythmia detection from ambulatory single-lead ECGs with a 34-layer 1D convolutional network [13]. Ribeiro et al. (2020) scaled that idea to 12-lead ECGs and reported specialist-level automatic diagnosis on a large Brazilian cohort [14]. Attia et al. (2019) demonstrated a different clinical use of the same modality: an AI-ECG that identified atrial fibrillation from recordings in sinus rhythm [15]. These three papers established that deep 1D models extract clinically meaningful structure from ECG waveforms.

Strodthoff, Wagner, Schaeffter, and Samek (2021) then did the community a more specific service: they benchmarked convolutional, recurrent, and wavelet models on PTB-XL under a single protocol [8]. On diagnostic *superclasses*, `resnet1d_wang` reached macro-AUROC 0.930 (0.05); xresnet1d101 and Inception1d were statistically compatible. Rhythm statements were easier (about 0.93–0.96); form statements were harder (about 0.83–0.89). Those ranges are Table 2 of the proposal and the Milestone 2 acceptance band. The architecture itself is the 1D residual network of Wang, Yan, and Oates (2017), adapted to 12-lead input [32]. A 2024 multi-resolution mutual-learning paper reproduces the same `resnet1d_wang` / `xresnet1d101` numbers and is cited in the proposal as an independent check on Table 2 [12].

Hong et al. (2020) and Siontis et al. (2021) review the broader ECG-and-deep-learning literature: opportunities (scale, representation learning, opportunistic screening) and the recurring failures (site shift, label noise, poor calibration, and limited external validation) [16, 17]. They are the survey backdrop, not the experimental baseline.

**Implication.** A 1D-ResNet of the Strodthoff/Wang type is the correct backbone: it is strong enough to sit inside the published PTB-XL band, small enough for a 4 GB GPU, and already used in the papers the project positions itself against.

## 5. Cross-dataset generalisation and domain shift

The closest existing *ECG-specific* evidence for the claim under study is BenchMD (Wantlin et al., 2023). BenchMD includes a PTB-XL → Chapman-Shaoxing OOD track among a broader medical image-and-sensor benchmark [9]. It shows that cross-dataset ECG performance can drop, but it does not (i) rotate the source, (ii) add MIMIC-IV-ECG, (iii) test whether the shifted model still beats chance, or (iv) trace a percentage-indexed adaptation curve with a diminishing-return threshold.

Strodthoff et al. (2021) already transferred PTB-XL models to ICBEB2018/CPSC and reported a performance change [8]. Weimann and Conrad (2021) studied transfer learning for ECG classification more systematically and found that pretraining on a large source helps, but they used absolute fine-tune sizes and did not compare calibration-only vs parameter updates [20]. HeartBEiT (Vaid et al., 2023) compared a vision-transformer ECG foundation model to convolutional baselines in low-sample regimes and is the proposal’s citation for “low-sample comparisons” [10]. It is not a three-way public-dataset rotation with uncertainty-quantified *P*_min.

Self-supervised ECG work is relevant as an alternative to labelled target adaptation. Mehari and Strodthoff (2022) showed that 12-lead self-supervision yields transferable representations [21]. CLOCS (Kiyasseh, Zhu, and Clifton, 2021) constructed contrastive views across space, time, and patients [22]. These methods reduce labelled-source demand; they do not answer how much *labelled target* data a *supervised imported model* needs before adaptation plateaus.

Outside ECG, WILDS (Koh et al., 2021) is the standard warning that in-the-wild distribution shift is common and that in-distribution accuracy is a poor proxy for deployment [25]. The project’s design — a specified triplet, a rotated source, and an audit trail — is the ECG-scale analogue of that lesson, not a WILDS replica.

**Implication.** There is evidence that OOD ECG performance *can* change, and almost no evidence that answers, for one backbone and this triplet, *how much*, *whether it is still above chance*, and *at what percentage of the target pool* two different adaptation classes stop improving.

## 6. Calibration versus parameter adaptation

Guo, Pleiss, Sun, and Weinberger (2017) showed that modern networks are systematically overconfident and that temperature scaling — a single scalar *T* > 0 applied to logits — is a surprisingly strong post-hoc fix [6]. Because *T* does not change class ranking, AUROC is invariant. That is why the proposal scores Temperature Scaling on Brier score and ECE only, and treats AUROC as a manipulation check.

The calibration lineage is older. Platt (1999) introduced logistic rescaling of scores; Niculescu-Mizil and Caruana (2005) compared Platt scaling and isotonic regression and showed that modern classifiers need a separate calibration stage [26]. Van Calster et al. (2019) restated the clinical cost of ignoring that stage [27]. For a multi-label ECG head the project applies a single *T* to logits and then a sigmoid, fitted by NLL/BCE on the adaptation slice — the Guo recipe, not a ranking-changing recalibrator.

Parameter adaptation is a different class. Linear probing freezes the backbone and retrains the head; it can change both discrimination and calibration. LoRA (Hu et al., 2022) injects low-rank updates *W*' = *W* + *BA* with *W* frozen, and was developed for language models [7]. Whether LoRA moves ECG discrimination as well as calibration is explicitly an open question in the proposal; it is a Spring 2027 milestone, not part of Milestone 3. Houlsby et al. (2019) are the broader parameter-efficient transfer citation (adapters) and justify measuring trainable-parameter count rather than assuming LoRA is “small” [31].

**Implication.** Milestone 3 implements Temperature Scaling and Linear Probing across *P* ∈ {5, 10, 20, 30, 50}% and does not compare them on AUROC. LoRA is specified, not yet required.

## 7. Sample-efficient and few-shot ECG

Markov et al. (2023) is the proposal’s example of a few-shot ECG study that uses an absolute, study-specific *N* grid [11]. That style is typical: papers report “50, 100, 200 labelled records” because they have one target. It becomes incomparable when one target pool is Chapman-sized (≈10⁴) and another is MIMIC-sized (≈10⁵–10⁶). A percentage-of-pool grid, with absolute *N* reported beside *P*, is the project’s design response.

HeartBEiT [10] and the self-supervised papers [21, 22] also operate in small-*N* regimes, but they ask a different question (is a foundation / pretext model more sample-efficient?) than the audit question (given an already-trained imported classifier, where does adaptation flatten?).

## 8. Synthesis: what is known, and the gap

| Theme | What is known | What is not known for this triplet |
|---|---|---|
| External validation | Validation *N* changes estimate precision; needs are setting-specific [1, 2, 28, 29] | A rotation-specific *P*_min with CIs for ECG adaptation |
| Datasets | PTB-XL, Chapman, MIMIC-IV-ECG, and Challenge 2020/21 sources exist [3–5, 18, 19] | Harmonized OOD numbers for one backbone on the proposed rotation |
| ECG classifiers | `resnet1d_wang` is in the 0.92–0.93 superclass band on PTB-XL [8, 12, 32] | Whether that model stays above chance on Chapman / MIMIC (or Georgia) |
| Cross-dataset ECG | BenchMD and ICBEB transfer show that performance *can* change [8, 9] | Size, significance, and chance-level test for six directed pairs |
| Calibration | Temperature scaling fixes probabilities without changing AUROC [6, 26, 27] | How Brier/ECE move with *P*% of each target pool |
| Parameter adaptation | Linear probing and LoRA can change ranking [7, 31] | ECG-specific diminishing returns vs calibration-only |
| Few-shot ECG | Absolute *N* grids, foundation models, self-supervision [10, 11, 21, 22] | Percentage-indexed stabilization with *k* ≥ 10 Monte Carlo reps |

The project’s contribution is therefore not “another ECG classifier.” It is a percentage-based audit procedure: source training → unadapted OOD with bootstrap CIs and a chance-level test → calibration and parameter adaptation on a percentage grid → *P*_min under predefined *τ*, with the source AUROC reported only as descriptive context.

## 9. Consequences for Milestones 1–3

- **M1 (this review).** The literature supports the research question and the two-class adaptation design. Twenty-eight papers are cited; the twelve proposal references remain the spine.
- **M2.** Train `resnet1d_wang` on PTB-XL diagnostic superclasses with official folds; accept only if macro-AUROC is compatible with Table 2 (≈0.92–0.93). Build a pipeline that can ingest PTB-XL, Chapman, Georgia/CPSC, and later MIMIC-IV-ECG.
- **M3.** Complete at least one full source→both-targets rotation. While MIMIC credentialing is pending, the public rotation is PTB-XL → Chapman and PTB-XL → Georgia (US stand-in), with Temperature Scaling and Linear Probing at five percentages and ≥10 Monte Carlo repetitions as the interim check. MIMIC pairs are added without changing the protocol once access arrives (proposal Section VII).

## 10. References

1. G. S. Collins, E. O. Ogundimu, and D. G. Altman, “Sample size considerations for the external validation of a multivariable prognostic model: a resampling study,” *Statistics in Medicine*, vol. 35, no. 2, pp. 214–226, 2016.
2. R. D. Riley, K. I. E. Snell, J. Ensor, D. L. Burke, F. E. Harrell Jr., K. G. M. Moons, and G. S. Collins, “Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study,” *BMJ*, vol. 384, p. e074821, 2024.
3. P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, “PTB-XL, a large publicly available electrocardiography dataset,” *Scientific Data*, vol. 7, no. 1, p. 154, 2020.
4. B. Gow, T. Pollard, L. A. Nathanson, A. Johnson, B. Moody, C. Fernandes, N. Greenbaum, S. Berkowitz, D. Moukheiber, P. Eickhoff, et al., “MIMIC-IV-ECG: Diagnostic Electrocardiograms Matched to MIMIC-IV Clinical Database,” PhysioNet, 2023. https://doi.org/10.13026/4nqg-sb35
5. J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, and C. Rakovski, “A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients,” *Scientific Data*, vol. 7, no. 1, p. 48, 2020.
6. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. 34th International Conference on Machine Learning*, PMLR, vol. 70, pp. 1321–1330, 2017.
7. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in *Proc. International Conference on Learning Representations*, 2022.
8. N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, “Deep learning for ECG analysis: Benchmarks and insights from PTB-XL,” *IEEE Journal of Biomedical and Health Informatics*, vol. 25, no. 5, pp. 1519–1528, 2021.
9. K. Wantlin, C. Wu, S.-C. Huang, O. Banerjee, F. Dadabhoy, V. V. Mehta, R. W. Han, F. Cao, R. R. Narayan, E. Colak, A. Adamson, L. Heacock, G. H. Tison, A. Tamkin, and P. Rajpurkar, “BenchMD: A benchmark for unified learning on medical images and sensors,” arXiv:2304.08486, 2023.
10. A. Vaid, J. Jiang, A. Sawant, S. Lerakis, E. Argulian, Y. Ahuja, J. Lampert, A. Charney, H. Greenspan, J. Narula, B. Glicksberg, and G. N. Nadkarni, “A foundational vision transformer improves diagnostic performance for electrocardiograms,” *npj Digital Medicine*, vol. 6, p. 108, 2023.
11. N. Markov, K. Ushenin, Y. Bozhko, and O. Solovyova, “Compressor-based classification for atrial fibrillation detection,” arXiv:2308.13328, 2023.
12. Multi-resolution mutual-learning reproduction of `resnet1d_wang` and `xresnet1d101` on PTB-XL, arXiv:2406.16928, 2024.
13. A. Y. Hannun, P. Rajpurkar, M. Haghpanahi, G. H. Tison, C. Bourn, M. P. Turakhia, and A. Y. Ng, “Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network,” *Nature Medicine*, vol. 25, pp. 65–69, 2019.
14. A. H. Ribeiro, M. H. Ribeiro, G. M. M. Paixão, D. M. Oliveira, P. R. Gomes, J. A. Canazart, M. P. S. Ferreira, C. R. Andersson, P. W. Macfarlane, W. Meira Jr., T. B. Schön, and A. L. P. Ribeiro, “Automatic diagnosis of the 12-lead ECG using a deep neural network,” *Nature Communications*, vol. 11, p. 1760, 2020.
15. Z. I. Attia, P. A. Noseworthy, F. Lopez-Jimenez, S. J. Asirvatham, A. J. Deshmukh, B. J. Gersh, R. E. Carter, X. Yao, A. A. Rabinstein, B. J. Erickson, S. Kapa, and P. A. Friedman, “An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction,” *The Lancet*, vol. 394, pp. 861–867, 2019.
16. S. Hong, Y. Zhou, J. Shang, C. Xiao, and J. Sun, “Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review,” *Computers in Biology and Medicine*, vol. 122, p. 103801, 2020.
17. K. C. Siontis, P. A. Noseworthy, Z. I. Attia, and P. A. Friedman, “Artificial intelligence-enhanced electrocardiography in cardiovascular disease management,” *Nature Reviews Cardiology*, vol. 18, pp. 465–478, 2021.
18. E. A. P. Alday, A. Gu, A. J. Shah, C. Robichaux, A. K. I. Wong, C. Liu, F. Liu, A. B. Rad, A. Elola, S. Seyedi, Q. Li, A. Sharma, G. D. Clifford, and M. A. Reyna, “Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020,” *Physiological Measurement*, vol. 41, no. 12, p. 124003, 2020.
19. M. A. Reyna, N. Sadr, E. A. P. Alday, A. Gu, A. J. Shah, C. Robichaux, A. B. Rad, A. Elola, S. Seyedi, S. Ansari, H. Ghanbari, Q. Li, A. Sharma, and G. D. Clifford, “Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021,” *Computing in Cardiology*, vol. 48, pp. 1–4, 2021.
20. K. Weimann and T. O. F. Conrad, “Transfer learning for ECG classification,” *Scientific Reports*, vol. 11, p. 5251, 2021.
21. T. Mehari and N. Strodthoff, “Self-supervised representation learning from 12-lead ECG data,” *Computers in Biology and Medicine*, vol. 141, p. 105114, 2022.
22. D. Kiyasseh, T. Zhu, and D. A. Clifton, “CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients,” in *Proc. 38th International Conference on Machine Learning*, PMLR, vol. 139, pp. 5606–5615, 2021.
23. A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.
24. F. Liu, C. Liu, L. Zhao, X. Zhang, X. Wu, X. Xu, Y. Liu, C. Ma, S. Wei, Z. He, J. Li, and E. N. Yin Kwee, “An Open Access Database for Evaluating the Algorithms of Electrocardiogram Rhythm and Morphology Abnormality Detection,” *Journal of Medical Imaging and Health Informatics*, vol. 8, no. 7, pp. 1368–1373, 2018.
25. P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. A. Earnshaw, I. S. Haque, S. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang, “WILDS: A Benchmark of in-the-Wild Distribution Shifts,” in *Proc. 38th International Conference on Machine Learning*, PMLR, vol. 139, pp. 5637–5664, 2021.
26. A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in *Proc. 22nd International Conference on Machine Learning*, pp. 625–632, 2005.
27. B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, E. W. Steyerberg, and Topic Group ‘Evaluating diagnostic tests and prediction models’ of the STRATOS initiative, “Calibration: the Achilles heel of predictive analytics,” *BMC Medicine*, vol. 17, p. 230, 2019.
28. G. S. Collins, J. B. Reitsma, D. G. Altman, and K. G. M. Moons, “Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement,” *Annals of Internal Medicine*, vol. 162, no. 1, pp. 55–63, 2015.
29. E. W. Steyerberg and F. E. Harrell Jr., “Prediction models need appropriate internal, internal-external, and external validation,” *Journal of Clinical Epidemiology*, vol. 69, pp. 245–247, 2016.
30. J. C. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74, 1999.
31. N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in *Proc. 36th International Conference on Machine Learning*, PMLR, vol. 97, pp. 2790–2799, 2019.
32. Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline,” in *Proc. International Joint Conference on Neural Networks (IJCNN)*, pp. 1578–1585, 2017.
