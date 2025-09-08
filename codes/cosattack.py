import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from eval import SDEexp,SRexp
import glob
from utils import tensor2img,load_image_from_path,load_model_from_config
from diff_mist import LDMAttack
import random
import torch.cuda as cuda
import time
import argparse
def measure_time_and_memory(func):
    def wrapper(*args, **kwargs):
        cuda.reset_peak_memory_stats()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        max_memory = cuda.max_memory_allocated()
        print(f"代码运行时间: {elapsed_time:.4f} 秒")
        print(f"最大显存占用: {max_memory / 1024 ** 2:.2f} MB")
        return result

    return wrapper

def get_dm(ckpt="ckpt/sd-v1-4.ckpt",dm_config="configs/stable-diffusion/v1-inference-sd1-4.yaml"):
    device = 0
    dm_config = OmegaConf.load(dm_config)
    dm = load_model_from_config(config=dm_config, ckpt=ckpt, device=device).to(device)
    return dm

def getTargetZ(zadv,zraw,dm,steps=100,c="",targetz=None,noise = None,tlst=None): 
    cnd = dm.get_learned_conditioning(c)
    z_raw = zraw.clone().detach()
    z_adv= zadv.clone().detach()
    gs = torch.zeros_like(z_raw)

    for i in range(steps):
        t_ = tlst[i]*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
        # t_ = random.randint(0,T-1)*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
        if noise == None:
            noise_ = torch.randn_like(z_raw).to(device=z_raw.device)
        else:
            noise_ = noise
        # print(t_.item())
        z_adv.requires_grad_()
        dm.zero_grad()
        zadv_noisy = dm.q_sample(x_start=z_adv, t=t_, noise=noise_)
        eps_pred = dm.apply_model(zadv_noisy, t_, cond=cnd)
        if targetz!=None:
            loss = 1 - nn.MSELoss(reduction="sum")(z_adv,targetz) - 50 * torch.cosine_similarity(eps_pred.flatten(),noise_.flatten(),dim=0) 
        else:
            loss = 1 - torch.cosine_similarity(eps_pred.flatten(),noise_.flatten(),dim=0)  

        # AdvDM Loss  
        # loss = torch.nn.functional.mse_loss(eps_pred, noise, reduction='none').mean([1, 2, 3]) 

        grad = torch.autograd.grad(loss,z_adv)[0]
        gs += grad / grad.std()
        z_adv.grad = None  

    return gs.detach()

def getXadv(x_raw,x_adv,z_grad,stepsize,steps,eps,dm):
    for i in range(steps):
        dm.zero_grad()
        x_adv.requires_grad_()
        z_adv = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv))
        grad = torch.autograd.grad(z_adv,x_adv,grad_outputs=z_grad)[0]
        with torch.no_grad():
            x_adv = x_adv + grad.detach().sign() * stepsize
            delta = torch.clamp(x_adv-x_raw, -eps, eps)
            x_adv = torch.clamp(x_raw+delta, -1.0, 1.0)
        x_adv.grad = None   
    return x_adv

def attack(steps=1, Estepsize=1/255.0, Esteps=100, Eeps=8/255.0,device=0,
             img_path = "test_images/to_protect/",output_path = "out/cos/",target=None,input_size = 512,path=None,config=None):
    dm = get_dm(path,config)
    T = dm.num_timesteps
    imgs = glob.glob(img_path+"*")
    trans = transforms.Compose([transforms.ToTensor()])
    c = ''
    if target:
        target_img = load_image_from_path(target, input_size)
        target_img = target_img.convert('RGB')
        target_img = np.array(target_img).astype(np.float32) / 127.5 - 1.0
        target_img = target_img[:, :, :3]
        target_x = torch.zeros([1, 3, input_size, input_size]).to(device)
        target_x[0] = trans(target_img).to(device)
        target_x = target_x.clone().detach()
        with torch.no_grad():
            target_z = dm.get_first_stage_encoding(dm.encode_first_stage(target_x)).to(device)
    else:
        target_z = None
    for i in tqdm(imgs[:]):
        img_name = i.split("/")[-1].split('.')[0]
        img = load_image_from_path(i, input_size)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img[:, :, :3]
        x = torch.zeros([1, 3, input_size, input_size]).to(device)
        x[0] = trans(img).to(device)
        x_raw = x.clone().detach()
        x_adv = x.clone().detach()
        
        with torch.no_grad():
            z_raw = dm.get_first_stage_encoding(dm.encode_first_stage(x_raw)).to(device)

        noise = torch.randn_like(z_raw).cuda()
        tlst = torch.tensor(sorted(random.sample(range(0,T),100)))
        tlst = tlst.reshape(steps,Esteps).t().tolist()
        tlst.reverse()
        # print("GROUP:",tlst)
        for tlst_z in tlst:
            with torch.no_grad():
                z = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv)).to(device)  
            z_grad = getTargetZ(z,z_raw,dm,steps,c,target_z,noise,tlst=tlst_z)
            x_adv = getXadv(x_raw,x_adv,z_grad=z_grad,stepsize=Estepsize*2,steps=steps,eps=Eeps*2,dm=dm)
        tensor2img(x_adv,output_path+f"/x_adv/{img_name}.jpg")
    return output_path+"x_adv/"

@measure_time_and_memory
def main(args):
    target = args.target
    exp_name = args.exp_name
    clean_pth = args.clean_path
    output_path = args.output_path
    Eeps = args.Eeps
    Estepsize = args.Estepsize
    model_path = args.model_path
    model_config = args.mdoel_config
    sde_model_path = args.sdedit_model_path
    sde_model_config = args.sdedit_model_config
    # Angle-shifting Attack
    output = attack(
        steps=args.steps,
        Esteps=args.Esteps,
        img_path=clean_pth,
        output_path=f"exp/out/{output_path}/" + exp_name  + "/",
        target=target,
        input_size=args.input_size,
        Eeps=Eeps,
        Estepsize=Estepsize,
        config=model_config,
        path=model_path
    )
    # SDEdit
    # output = f"exp/out/{output_path}/" + exp_name  + "/x_adv/"
    SDEexp(
        img_path=output,
        output_path=f"exp/sde/{output_path}/" + exp_name + "/",
        clean_path=clean_pth,
        xlsxname=f"exp/sde/{output_path}/{output_path}_{exp_name}",
        path=sde_model_path,
        config=sde_model_config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack and experiments")
    parser.add_argument("--target", type=str, default=None, help="target image path")
    parser.add_argument("--clean_path", type=str, default="images/clean/", help="clean image path")
    parser.add_argument("--output_path", type=str, default="cos_20_5", help="output folder name")
    parser.add_argument("--exp_name", type=str, default="TEST", help="experiment name")
    parser.add_argument("--Esteps", type=int, default=20, help="steps for attacking encoder")
    parser.add_argument("--steps", type=int, default=5, help="steps for attacking denoiser")
    parser.add_argument("--input_size", type=int, default=512, help="input size")
    parser.add_argument("--Eeps", type=float, default=8/255.0, help="max perturbation for attacking encoder")
    parser.add_argument("--Estepsize", type=float, default=1/255.0, help="step size for attacking encoder")
    parser.add_argument("--model_path", type=str, default="ckpt/sd-v1-4.ckpt", help="diffusion model")
    parser.add_argument("--mdoel_config", type=str, default="configs/stable-diffusion/v1-inference-sd1-4.yaml", help="diffusion model config")
    parser.add_argument("--sdedit_model_path", type=str, default="ckpt/sd-v1-4.ckpt", help="sdedit model")
    parser.add_argument("--sdedit_model_config", type=str, default="configs/stable-diffusion/v1-inference-sd1-4.yaml", help="sdedit model config")
    args = parser.parse_args()
    main(args)