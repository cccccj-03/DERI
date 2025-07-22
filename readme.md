# DERI: Cross-Modal ECG Representation Learning with Deep ECG-Report Interaction  ----Accepted by IJCAI 2025

This paper is accepted by IJCAI 2025. The paper link will be released soon. We provide our experimental code and the supplementary materials of our paper here.

## Abstract:
Electrocardiogram (ECG) is widely used to diagnose cardiac conditions via deep learning methods. Although existing self-supervised learning (SSL) methods have achieved great performance in learning representation for ECG-based cardiac conditions classification, the clinical semantics can not be effectively captured. To overcome this limitation, we proposed to learn cross-modal ECG representations that contain more clinical semantics via a novel framework with **D**eep **E**CG-**R**eport **I**nteraction (**DERI**). Specifically, we design a novel framework combining multiple alignments and mutual feature reconstructions to learn effective representation of the ECG with the clinical report, which fuses the clinical semantics of the report. An RME-Module inspired by masked modeling is proposed to improve the ECG representation learning. Furthermore, we extend ECG representation learning to report generation with a language model, which is significant for evaluating clinical semantics in the learned representations and even clinical applications. Comprehensive experiments with various settings are conducted on various datasets to show the superior performance of our DERI. Our code is released on https://github.com/cccccj-03/DERI.

## Innovations:
To learn effective ECG representation from reports, we propose a novel cross-modal framework of ECG-Report via multiple feature alignment and mutual feature reconstruction. An RME-Module is also designed for ECG representation learning enhancement.

To better illustrate the clinical semantics learned by DERI, we combine it with a language model for report generation. The pre-trained model provides effective ECG representation and a language model is used to generate clinical reports for clinical semantics.

Comprehensive experiments on downstream datasets are conducted to evaluate the proposed DERI method, including zero-shot classification, linear probing, and even report generation. Experimental results illustrate that our DERI method surpasses all SOTA methods.

![image]{DERI.png}

## Dataset:
MIMIC-IV-ECG: We downloaded the MIMIC-IV-ECG dataset as the ECG signals and paired ECG reports: https://physionet.org/content/mimic-iv-ecg/1.0/

PTB-XL: We downloaded the PTB-XL dataset which consisting four subsets, Superclass, Subclass, Form, Rhythm: https://physionet.org/content/ptb-xl/1.0.3/

CPSC2018: We downloaded the CPSC2018 dataset which consisting three training sets: http://2018.icbeb.org/Challenge.html

CSN(Chapman-Shaoxing-Ningbo): We downloaded the CSN dataset: https://physionet.org/content/ecg-arrhythmia/1.0.0/


## Implementatiom
### Pretrain
bash pretrain/launch.sh

### Linear Probing
cd finetune/sub_script
bash run_all_linear.sh

### Zero-shot
cd zeroshot
bash zeroshot.sh

### Report Generation
finetune GPT2:

CUDA_VISIBLE_DEVICES=5,6 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint=localhost:34042 finetune.py

evaluation:
report_evaluation.ipynb
