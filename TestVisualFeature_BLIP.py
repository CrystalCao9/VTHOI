"""
Code for extracting image features using BLIP visual encoder
Our work builds upon PViC. With the code provided below, you can reproduce our work more efficiently.
"""


from PIL import Image
from ModifiedBLIP.models.blip import blip_feature_extractor
from detr.datasets import transforms as T
import torch

model_blip = blip_feature_extractor(pretrained='checkpoints/model_large.pth',
                                    image_size=224,
                                    vit='large',
                                    vit_grad_ckpt=True,
                                    vit_ckpt_layer=0,
                                    )

bliptrans = T.Compose([
            T.IResize([224, 224]),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])


image_path = 'VTHOI/hicodet/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg'
image = Image.open(image_path)
image = image.convert("RGB")
img_blip, _ = bliptrans(image, None)
img_blip = [img_blip]
images = torch.stack(img_blip)

vs = model_blip.encode_image(images)
print(vs)
