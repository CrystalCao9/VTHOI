# VTHOI:Visual-Textual Feature Learning for Rare Human-Object Interactions Detection
![image](https://github.com/CrystalCao9/VTHOI/blob/main/picture/VTHOI.png)
## Abstract
Human-Object Interaction (HOI) detection is a fundamental task in understanding human-object relationships. However, existing methods struggle with long-tail data distributions and capturing global context, leading to poor performance in detecting rare classes. To address these challenges, we propose a novel HOI detector named VTHOI, which integrates visual and textual embeddings to improve the model's global perception and its ability to detect rare classes. First, the image’s global context is extracted and fused with local human-object pair features during decoding to generate vision logits prompts. Subsequently, the proposed Adaptive Logits Fusion Module (ALFM) integrates the vision logits prompts into the backbone, enhancing global contextual understanding. Additionally, the consistency constraints of the language model are employed to learn textual descriptions and semantic relationships of non-rare classes, thereby enabling the model to better capture the features of rare classes. Our approach outperforms state-of-the-art methods on the HICO-DET and V-COCO datasets, achieving significant improvements, particularly in rare class detection.

## Requirements
### Code Repository: 
The core code of our project has been uploaded. Our work builds upon PViC, and those familiar with both PViC and our implementation will find it straightforward to reproduce our results.
### ​Environment Setup: 
Configure your environment by following the instructions in https://github.com/fredzzhang/pvic. Download the required datasets and create symbolic links accordingly.
### ​Model Weights Preparation:
Download H-DETR weights to the checkpoints directory: https://drive.google.com/file/d/1wge-CC1Fx67EHOSXyHGHvrqvMva2jEkr/view?usp=share_link

Download ResNet-50 weights to checkpoints: https://drive.google.com/file/d/1cwMJNMQALDrVdTxQL6Vdw66thpgeyq-2/view

Download BLIP weights to checkpoints: https://github.com/salesforce/BLIP/blob/main/README.md 

Download BERT weights to the bert-base-uncased directory: https://huggingface.co/google-bert/bert-base-uncased/tree/main
