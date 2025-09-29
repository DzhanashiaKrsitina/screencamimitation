import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torchview import draw_graph


output_dir="output"
os.makedirs(output_dir, exist_ok=True)

"""Images to tensors PIL for inference"""


def tensor_to_pil(tensor_img):
    unnorm = transforms.Normalize(mean=[-0.5 / 0.5] * 3, std=[1 / 0.5] * 3)
    tensor_img = unnorm(tensor_img).clamp(0, 1)
    pil_img = transforms.ToPILImage()(tensor_img)
    return pil_img


"""Loading images into Dataset"""


class PairedImageDataset(Dataset):
    def __init__(self, dataset_root, image_size=512):
        self.pairs = []
        cover_root = os.path.join(dataset_root, "cover")
        final_root = os.path.join(dataset_root, "final")
        for subfolder in sorted(os.listdir(cover_root)):
            cover_sub = os.path.join(cover_root, subfolder)
            final_sub = os.path.join(final_root, subfolder)
            if os.path.isdir(cover_sub) and os.path.isdir(final_sub):
                cover_files = sorted([f for f in os.listdir(cover_sub) if f.lower().endswith(".png")])
                for fname in cover_files:
                    cover_path = os.path.join(cover_sub, fname)
                    final_fname = os.path.splitext(fname)[0] + ".png"
                    final_path = os.path.join(final_sub, final_fname)
                    if not os.path.exists(final_path):
                        final_path = os.path.join(final_sub, fname)
                    if os.path.exists(final_path):
                        self.pairs.append((cover_path, final_path))
                    else:
                        print(f"Warning: No matching final image for {cover_path}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cover_path, final_path = self.pairs[idx]
        cover_img = Image.open(cover_path).convert("RGB")
        final_img = Image.open(final_path).convert("RGB")
        return {
            "cover": self.transform(cover_img),
            "final": self.transform(final_img),
            "filename": os.path.basename(cover_path)
        }

def add_gaussian_noise(image, mean=0, std=2):
    img_np = np.array(image)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.float32)
    noisy_img = img_np.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def apply_perspective_transform(image, distortion_scale=0.1):
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = src_points + np.random.uniform(-distortion_scale, distortion_scale, size=(4, 2)) * [w, h]
    dst_points = dst_points.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img_np, M, (w, h))
    return Image.fromarray(warped)


def apply_basic_transform(image, max_angle=10, max_pad=0.1, noise_std=2):

    """Applying basic augmentations to PIL image."""
    angle = random.uniform(-max_angle, max_angle)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(10, 10, 10))

    rotated = rotated.filter(ImageFilter.GaussianBlur(radius=0.5))
    w, h = rotated.size
    pad_left = int(random.uniform(0, max_pad) * w)
    pad_right = int(random.uniform(0, max_pad) * w)
    pad_top = int(random.uniform(0, max_pad) * h)
    pad_bottom = int(random.uniform(0, max_pad) * h)
    padded = ImageOps.expand(rotated, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(10, 10, 10))
    if noise_std > 0:
        padded = add_gaussian_noise(padded, std=noise_std)
    return padded


def apply_advanced_transform(image, max_angle=3, max_pad=0.1, distortion_scale=0.1, noise_std=0.2):
    """Hard augmentations to PIL image."""
    persp = apply_perspective_transform(image, distortion_scale)
    angle = random.uniform(-max_angle, max_angle)
    rotated = persp.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(10, 10, 10))
    rotated = rotated.filter(ImageFilter.GaussianBlur(radius=0.5))
    w, h = rotated.size
    pad_left = int(random.uniform(0, max_pad) * w)
    pad_right = int(random.uniform(0, max_pad) * w)
    pad_top = int(random.uniform(0, max_pad) * h)
    pad_bottom = int(random.uniform(0, max_pad) * h)
    padded = ImageOps.expand(rotated, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(10, 10, 10))

    if noise_std > 0:
        padded = add_gaussian_noise(padded, std=noise_std)
    return padded


def compute_metrics(img_target, img_generated):
    np_target = np.array(img_target).astype(np.float32) / 255.0
    np_generated = np.array(img_generated).astype(np.float32) / 255.0
    mse_val = mean_squared_error(np_target, np_generated)
    psnr_val = peak_signal_noise_ratio(np_target, np_generated, data_range=1.0)
    h, w, c = np_target.shape
    max_win_size = min(h, w, 7)
    ssim_val = structural_similarity(
        np_target,
        np_generated,
        data_range=1.0,
        channel_axis=-1,
        win_size=max_win_size
    )
    return mse_val, psnr_val, ssim_val

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        batch_size = x.size(0)
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)
        return x_transformed, theta


class HDRNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, grid_size=16):
        super(HDRNet, self).__init__()
        self.grid_size = grid_size
        self.local_branch = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            # Дилатационная свертка для большего охвата контекста
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.global_fc = nn.Linear(64, output_nc)

    def forward(self, x):
        local_grid = self.local_branch(x)
        local_adjust = nn.functional.interpolate(local_grid, size=x.shape[2:], mode='bilinear', align_corners=True)
        batch_size = x.size(0)
        global_feat = self.global_branch(x)
        global_feat = global_feat.view(batch_size, -1)
        global_adjust = self.global_fc(global_feat)
        global_adjust = global_adjust.unsqueeze(2).unsqueeze(3).expand_as(
            local_adjust)
        adjustment = local_adjust + global_adjust
        out = x + adjustment
        return torch.tanh(out)


# Multi-Scale Discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        self.layers = nn.ModuleList([nn.Sequential(*sequence)])
        nf_mult = 1
        # Convolutions with Instance Normalization
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))
        self.output_layer = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, input):
        result = []
        x = input
        for layer in self.layers:
            x = layer(x)
            result.append(x)
        out = self.output_layer(x)
        result.append(out)
        return result


# NLayerDiscriminators
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList()
        for _ in range(num_D):
            self.discriminators.append(NLayerDiscriminator(input_nc, ndf, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        results = []
        for D in self.discriminators:
            results.append(D(input))
            input = self.downsample(input)
        return results


model = MultiScaleDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
dummy = torch.randn(1, 3, 256, 256)   # batch of RGB 256×256 images
graph = draw_graph(model, input_data=dummy)
graph.visual_graph.render("multiscale_discriminator", format="png")

model = HDRNet(input_nc=3, output_nc=3)
dummy = torch.randn(1, 3, 256, 256)   # batch of RGB 256×256 images
graph = draw_graph(model, input_data=dummy)
graph.visual_graph.render("HDRNet", format="png")



advanced_mod = True
inference_dataset_root = "C:\RABOTA\mdpi\program-master\content\images"
use_uploaded_model = True
uploaded_checkpoint_path = "checkpoint_hdrnet_v2_epoch_50.pth"
img_size_for_model = 256  #не менять
inference_full_dataset = PairedImageDataset(inference_dataset_root, image_size=img_size_for_model)
print(f"Loaded inference dataset wtih {len(inference_full_dataset)} image pairs.")
if advanced_mod:
    transform_func = apply_advanced_transform
else:
    transform_func = apply_basic_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = HDRNet(input_nc=3, output_nc=3).to(device)

if use_uploaded_model:
    ckpt = torch.load(uploaded_checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(ckpt["generator"])
    print(f"Загружены веса генератора из: {uploaded_checkpoint_path}")
generator.eval()
mse_list, psnr_list, ssim_list = [], [], []
for idx in range(len(inference_full_dataset)):
    sample = inference_full_dataset[idx]
    cover_tensor = sample["cover"].to(device)
    filename = sample["filename"]
    #    final_tensor = sample["final"].to(device)
    with torch.no_grad():
        fake_tensor = generator(cover_tensor.unsqueeze(0)).squeeze(0)
    fake_image = tensor_to_pil(fake_tensor.cpu())
    transformed_image=fake_image
    #if advanced_mod:
    #    transformed_image = transform_func(
    #        fake_image,
    #        max_angle=3,#3
    #        max_pad=0.01, #0.01
    #        distortion_scale=0.05,
    #        noise_std=0.1)
    #else:
    #    transformed_image = transform_func(
    #        fake_image,
    #        max_angle=3, #3
    #        max_pad=0.01, #0.01
    #        noise_std=0.1)
    final_image = tensor_to_pil(fake_tensor.cpu())
    w, h = transformed_image.size
    final_resized = final_image.resize((w, h), Image.Resampling.BILINEAR)

    output_path = os.path.join(output_dir, filename)
    transformed_image.save(output_path)
