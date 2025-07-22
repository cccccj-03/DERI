# DERI: Cross-Modal ECG Representation Learning with Deep ECG-Report Interaction  ----Accepted by IJCAI 2025

We provide our experimental code and the supplementary materials of our paper here.

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
