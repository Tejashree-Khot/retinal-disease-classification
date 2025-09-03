import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from transformers import BertModel


class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Image branch (EfficientNet features only)
        if model == "efficientnet-b0":
            base_model = models.efficientnet_b0(pretrained=True)
        elif model == "efficientnet-b1":
            base_model = models.efficientnet_b4(pretrained=True)
        elif model == "efficientnet-b7":
            base_model = models.efficientnet_b7(pretrained=True)
        self.image_encoder = base_model.features  # feature extractor
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # pool HxW → 1x1
        self.img_fc = nn.Linear(1280, 512)  # reduce dim

        # Text branch (BERT)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.txt_fc = nn.Linear(768, 512)  # reduce dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Image branch
        img_feats = self.image_encoder(images)  # [batch, 1280, H, W]
        img_feats = self.global_pool(img_feats)  # [batch, 1280, 1, 1]
        img_feats = img_feats.view(img_feats.size(0), -1)  # flatten → [batch, 1280]
        img_feats = self.img_fc(img_feats)  # [batch, 512]

        # Text branch
        txt_feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = self.txt_fc(txt_feats.pooler_output)  # [batch, 512]

        # Combine branches and classify
        combined = torch.cat((img_feats, txt_feats), dim=1)  # [batch, 1024]
        output = self.classifier(combined)  # [batch, num_classes]
        return output


if __name__ == "__main__":
    model = MultiModalModel(num_classes=5)
    print(model)
