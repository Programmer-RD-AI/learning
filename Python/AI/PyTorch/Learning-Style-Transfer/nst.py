import numpy as np
from tqdm import *
import re
import torch
from torch.nn import *
from torch.optim import *
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import *
from torchvision.utils import save_image

device = "cpu"
image_size = 356
loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)


class VGG(Module):
    def __init__(self):
        super().__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = vgg19(pretrained=True).features[:29]

    def forward(self, X):
        features = []
        for layer_num, layer in enumerate(self.model):
            X = layer(X)
            if str(layer_num) in self.chosen_features:
                features.append(X)
        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image)
    # print(image.shape)
    image = image.unsqueeze(0)
    # print(image.shape)
    return image.to(device)


original_img = load_image("annahathaway.png")
style_img = load_image("style.jpg")
model = VGG().to(device).eval()
# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)

# Hyper Params
total_steps = 750
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = Adam([generated], lr=learning_rate)


for step in tqdm(range(total_steps)):
    generated_features = model(generated)
    # print(np.array(generated_features[1].detach().numpy()).shape)
    original_img_features = model(original_img)
    # print(np.array(original_img_features[2].detach().numpy()).shape)
    style_features = model(style_img)
    # print(np.array(style_features[3].detach().numpy()).shape)
    style_loss = original_loss = 0
    for gen_feature, original_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        print(np.array(gen_feature.detach().numpy()).shape)
        print(np.array(original_feature.detach().numpy()).shape)
        print(np.array(style_feature.detach().numpy()).shape)
        original_loss += torch.mean((gen_feature - original_feature) ** 2)
        # Compute gram matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        print(G.shape)
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        print(A.shape)
        style_loss += torch.mean((G - A) ** 2)
    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    save_image(generated, "generated.png")
    break
