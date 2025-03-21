"""
Code for extracting text features using BERT
Our work builds upon PViC. With the code provided below, you can reproduce our work more efficiently.
"""


from ModifiedBLIP.models.blip import blip_feature_extractor
import torch
from hico_text_label import hico_text_label

def remove_exceptions(tensors):
    input_list = tensors.tolist()

    if len(input_list[0]) == 11:
        del input_list[0][7]
    elif len(input_list[0]) == 12:
        del input_list[0][7:9]
    elif len(input_list[0]) == 13:
        del input_list[0][7:10]
    return torch.tensor(input_list)


model_blip = blip_feature_extractor(pretrained='checkpoints/model_large.pth',
                                    image_size=224,
                                    vit='large',
                                    vit_grad_ckpt=True,
                                    vit_ckpt_layer=0,
                                    )
hoi_text_label = hico_text_label
lst = [model_blip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()]
input_ids_list = [item['input_ids'] for item in lst]

processed_input_ids_list = [remove_exceptions(input_ids) for input_ids in input_ids_list]
text_to = torch.cat(processed_input_ids_list)
text_en = model_blip.encode_text(text_to)
print(text_en)
