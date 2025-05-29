import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
# from datasets import load_dataset
from torchvision import datasets, transforms
from diffusers import DDPMPipeline, DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from cleanfid import fid
from torch.optim.lr_scheduler import LambdaLR

from huggingface_hub import notebook_login

notebook_login()
from utils import supervisor, tools
tools.setup_seed(2333)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定第二块 GPU（索引从 0 开始）

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "./models/ddpm-imagenette-finetuned"  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()


########################################
# 配置部分
########################################
# model_name = 'google/ddpm-cifar10-32'
# output_dir = "./models/ddpm-gtsrb-finetuned"
data_dir = './data'
# batch_size = 64
# learning_rate = 1e-4
# num_epochs = 50
# save_every = 5  # 每个epoch后保存一次模型和pipeline
# inference_steps = 1000 # 推理时扩散过程的步数
# lr_warmup_steps = 500
# gradient_accumulation_steps = 1  # 梯度累积步数



# os.makedirs(output_dir, exist_ok=True)

########################################
# 数据准备与预处理
########################################


data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.Imagenette(os.path.join(data_dir, 'imagenette'), split = 'train', transform=data_transform)
# test_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split = 'test', transform=data_transform, download=True)
img_size = 256
num_classes = 10
# dataset = load_dataset("gtsrb", split="train")

def collate_fn(batch):
    # batch是列表，每个元素是(img, label)
    # 我们只需要img用于DDPM的无条件训练，不需要label
    images = [item[0] for item in batch]  # item[0]是图像, item[1]是label
    images = torch.stack(images, dim=0)  # [B,C,H,W]
    return {"pixel_values": images}

# def transform_examples(examples):
#     images = [preprocess(img.convert("RGB")) for img in examples["image"]]
#     return {"pixel_values": images}
#
# dataset = dataset.map(transform_examples, batched=True)
# dataset.set_format(type="torch", columns=["pixel_values"])

train_dataloader = DataLoader(dataset, batch_size= 16, shuffle=True, collate_fn=collate_fn)




real_images_dir = os.path.join(config.output_dir, "real_images_for_fid")
os.makedirs(real_images_dir, exist_ok=True)


data_transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_set = datasets.Imagenette(os.path.join(data_dir, 'imagenette'), split = 'val', transform=data_transform_test)
# 假设抽取1000张测试集图像作为FID真实参考
num_real_samples = 1000
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, drop_last=False)

count = 0
if len(os.listdir(real_images_dir)) < num_real_samples:
    for batch_data in test_loader:
        imgs = batch_data[0]  # (B,C,H,W)
        for i in range(imgs.shape[0]):
            if count >= num_real_samples:
                break
            img = imgs[i]
            img_np = (img.numpy().transpose(1,2,0)*255).astype("uint8")
            Image.fromarray(img_np).save(os.path.join(real_images_dir, f"real_{count}.png"))
            count += 1
        if count >= num_real_samples:
            break
print(f"Collected {count} real images for FID reference at {real_images_dir}")


def evaluate_fid(pipeline, epoch):
    # 生成1000张样本用于FID计算（与real_images同数）
    generated_images_dir = os.path.join(config.output_dir, f"epoch-{epoch+1}-generated")
    os.makedirs(generated_images_dir, exist_ok=True)
    num_gen_samples = 1000
    batch_gen = 50  # 每次生成50张，分多次完成

    pipeline.unet.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
        for i in range(num_gen_samples // batch_gen):
            sample_images = pipeline(
                num_inference_steps=1000,
                batch_size=batch_gen,
                output_type="tensor"
            ).images

            sample_images = torch.tensor(sample_images).clamp(0,1).cpu().numpy()
            for j, img_array in enumerate(sample_images):
                img = Image.fromarray((img_array*255).astype("uint8"))
                img.save(os.path.join(generated_images_dir, f"gen_{i*batch_gen+j}.png"))

    # 计算FID
    fid_score = fid.compute_fid(real_images_dir, generated_images_dir)
    print(f"Epoch {epoch+1}: FID = {fid_score:.4f}")
    return fid_score




from diffusers import UNet2DModel


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
      ),
)



from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)



from diffusers import DDPMPipeline

import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

fid_list = []

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        # logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo_name = get_full_repo_name(Path(config.output_dir).name)
        #     repo = Repository(config.output_dir, clone_from=repo_name)
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     # evaluate(config, epoch, pipeline)
            #     # fid_score = evaluate_fid(pipeline, epoch)
            #     # fid_list.append(fid_score)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # if config.push_to_hub:
                #     repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                # else:
                pipeline.save_pretrained(config.output_dir)



from accelerate import notebook_launcher
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
print(fid_list)




