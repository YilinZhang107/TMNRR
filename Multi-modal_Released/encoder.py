import torch
import torch.nn as nn
import torchvision
from transformers import BertModel

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        # pool_func = (
        #     nn.AdaptiveAvgPool2d if args.img_embed_pool_type == "avg" else nn.AdaptiveMaxPool2d
        # )
        pool_func = nn.AdaptiveAvgPool2d

        # if args.num_image_embeds in [1, 2, 3, 5, 7]:
        self.pool = pool_func((1, 1))


    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        out = self.pool(out)
        # print(out.shape)
        out = torch.flatten(out, start_dim=2)  
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(2048, 101)

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out, x


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, txt, mask, segment):
        # _, out = self.bert(
        #     txt,
        #     token_type_ids=segment,
        #     attention_mask=mask,
        #     output_all_encoded_layers=False,
        # )
        # return out    
        

        outputs = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
        )
       
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0, :]



class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(768, 101)
       

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        return self.clf(x), x


