

<!-- <div align="center">

## Enhancing Reinforcement Learning for Radiology Report Generation with Evidence-aware Rewards and Self-correcting Preference Learning

</div> -->

<div align="center">

<h1> Enhancing Reinforcement Learning for Radiology Report Generation with Evidence-aware Rewards and Self-correcting Preference Learning </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

</div>

## Framework

<div align=center>
<img width="600" alt="image" src="images/figure 1.png">
</div>

Recent reinforcement learning (RL) approaches have advanced radiology report generation (RRG), yet two core limitations persist: (1) report-level rewards offer limited evidence-grounded guidance for clinical faithfulness; and (2) current methods lack an explicit self-improving mechanism to align with clinical preference. We introduce clinically aligned **E**vidence-aware **S**elf-**C**orrecting **R**einforcement **L**earning (ESC-RL), comprising two key components. First, a Group-wise Evidence-aware Alignment Reward (GEAR) delivers group-wise, evidence-aware feedback. GEAR reinforces consistent grounding for true positives, recovers missed findings for false negatives, and suppresses unsupported content for false positives. Second, a Self-correcting Preference Learning (SPL) strategy automatically constructs a reliable, disease-aware preference dataset from multiple noisy observations and leverages an LLM to synthesize refined reports without human supervision. ESC-RL promotes clinically faithful, disease-aligned reward and supports continual self-improvement during training. Extensive experiments on two public chest X-ray datasets demonstrate consistent gains and state-of-the-art performance.
## Setup
```bash
# Clone the repo
git clone git@github.com:banbooliang/ESC-RL
# Create Env and install basic packages
conda create -n escrl python=3.10
pip install -r requirements.txt
```

## Download
- Download the **MIMIC-CXR** dataset from the [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/), and obtain the corresponding annotation file from [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing). Put them into ./data/mimic_cxr/ forder. 
- Download the pretrained REVTAF model and pre-processed data files according to [website](https://github.com/banbooliang/REVTAF-RRG).

The pretrained MAVL model can download them from [here](https://github.com/hieuvmphan/CVPR2024_MAVL).


- To evaluate clinical efficacy, download the `chexbert.pth` model from [Google Drive](https://drive.google.com/file/d/1Qj5yM62FlASGRnW1hH0DDtCENuqGtt7L/view?usp=sharing) and place it in checkpoints/stanford/chexbert/.

## Training
- Run the following command to start training:
```bash 
bash train_mimic_cxr.sh 
```
The trained model will be saved in the results/mimic_cxr/ directory.

## Test
- Run the following command to start testing on the MIMIC-CXR test set and IU X-Ray dataset, respectively:

```bash
bash test_mimic_cxr.sh 
bash test_iu_xray.sh 
```

## Citation
If you find our repository useful, please star this repo and cite our paper.
```bibtex
@misc{zhou2026enhancingreinforcementlearningradiology,
      title={Enhancing Reinforcement Learning for Radiology Report Generation with Evidence-aware Rewards and Self-correcting Preference Learning}, 
      author={Qin Zhou and Guoyan Liang and Qianyi Yang and Jingyuan Chen and Sai Wu and Chang Yao and Wang Zhe},
      year={2026},
      eprint={2604.13598},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.13598}, 
}
```

## Acknowledgment
* [REVTAF](https://github.com/banbooliang/REVTAF-RRG)
* [RIME](https://github.com/CJReinforce/RIME_ICML2024)
* [PromptMRG](https://github.com/jhb86253817/PromptMRG)
* [R2GenRL](https://github.com/synlp/R2GenRL)
* [MAVL](https://github.com/hieuvmphan/CVPR2024_MAVL)




