import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

from advertorch.attacks import LinfPGDAttack
from attacks import Linf_PGD, SDEdit
import glob
from utils import mp, si, cprint, norm01,tensor2img,load_image_from_path,load_model_from_config
from PIL import Image
from diff_mist import LDMAttack
import matplotlib.pyplot as plt
import torch.optim as optim
from eval import *
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
import random


def get_dm():
    device = 0
    ckpt = "ckpt/sd-v1-4.ckpt"
    dm_config = "configs/stable-diffusion/v1-inference-sd1-4.yaml"
    dm_config = OmegaConf.load(dm_config)
    dm = load_model_from_config(config=dm_config, ckpt=ckpt, device=device).to(device)
    return dm

def getTargetZ(zadv,zraw,dm,t_P=-1,steps=100,c="",targetz=None,noise = None,tlst=None): 
    cnd = dm.get_learned_conditioning(c)
    T = dm.num_timesteps
    z_raw = zraw.clone().detach()
    z_adv= zadv.clone().detach()
    gs = torch.zeros_like(z_raw)

    for i in range(steps):
        t_ = tlst[i]*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
        # t_ = 0*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
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
        # loss = torch.nn.functional.mse_loss(eps_pred, noise, reduction='none').mean([1, 2, 3]) 
        grad = torch.autograd.grad(loss,z_adv)[0]
        g = grad / grad.std()
        tensor2img(g,f"exp/exp/ge_sn/g_{t_.item()}.jpg")
        gs += g
        z_adv.grad = None  
    gs = gs / 5
    tensor2img(gs,f"exp/exp/ge_sn/gs_{t_.item()}.jpg")
    return gs.detach(),loss.item()

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

def attack(tp=999,steps=1, Estepsize=1/255.0, Esteps=100, Eeps=8/255.0,device=0,
             img_path = "test_images/to_protect/",output_path = "out/DeAttack/argmax_noise-eps/",target=None,input_size = 512):
    dm = get_dm()
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
    for i in tqdm(imgs):
        img_name = i.split("/")[-1].split('.')[0]

        img = load_image_from_path(i, input_size)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img[:, :, :3]
        x = torch.zeros([1, 3, input_size, input_size]).to(device)
        x[0] = trans(img).to(device)
        x_raw = x.clone().detach()
        x_adv = x.clone().detach()
        # tensor2img(x_raw,img_path[:-1]+f"_resized/{img_name}.jpg")
        
        with torch.no_grad():
            z_raw = dm.get_first_stage_encoding(dm.encode_first_stage(x_raw)).to(device)
        torch.manual_seed(100)
        noise = torch.randn_like(z_raw).cuda()


        t = tp
        tlst = torch.tensor(sorted(random.sample(range(0,T),100)))
        tlst = tlst.reshape(steps,Esteps).t().tolist()
        losses = []
        for _ in range(Esteps):
            tlst_ = tlst[_]
            with torch.no_grad():
                z = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv)).to(device)  
            z_grad,loss = getTargetZ(z,z_raw,dm,t,steps,c,target_z,noise,tlst=tlst_)
            x_adv = getXadv(x_raw,x_adv,z_grad=z_grad,stepsize=Estepsize*2,steps=steps,eps=Eeps*2,dm=dm)
            losses += [loss]
        # save_score(losses,f"exp/exp/res/cos_t0")
    return output_path+"x_adv/"

def main():
    target = "test_images/target/MIST.png"

    exp_name = 'celebaHQ_SR/'
    imgGT = '/home/dx/lmh/dataset/celebaHQ_64/'
    clean_pth = '/home/dx/lmh/dataset/celebaHQ_256/'
    scorefile = exp_name[:-1]

    # exp_name = "wikiart/"
    # imgGT = '/home/dx/lmh/dataset/wikiart_resized/'
    # clean_pth = imgGT
    # scorefile = exp_name[:-1]

    exp_name = 'celebaHQ/'
    imgGT = '/home/dx/lmh/dataset/celebaHQ_resized/'
    clean_pth = imgGT
    scorefile = exp_name[:-1]

    # exp_name = "wikiartSR/"
    # imgGT = '/home/dx/lmh/dataset/wikiart_64/'
    # clean_pth = '/home/dx/lmh/dataset/wikiart_256/'
    # scorefile = exp_name[:-1]


    # outpath = f"cos_20_5"
    # output = attack(tp=-1,output_path= f"exp/sr/{outpath}/"+exp_name,img_path=imgGT,Esteps=20,target=target,steps=5,input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name+"x_adv/" 
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}") 


    exp_name = '458/'
    imgGT = 'test_images/exp/'
    clean_pth = imgGT
    scorefile = exp_name[:-1]

    outpath = f"exp_loss_gap"
    output = attack(tp=-1,output_path= f"exp/out/{outpath}/"+exp_name,img_path=imgGT,Esteps=20,target=None,steps=5)
    # output = f"exp/out/{outpath}/"+exp_name+"x_adv/" 
    # SDEexp(img_path = output, output_path = f"exp/sde/{outpath}/"+exp_name,clean_path=clean_pth,xlsxname=f"exp/exp/res/{scorefile}_{outpath}")

    return   



if __name__ == '__main__':
    main()

    # outpath = f"mist"
    # output = LDMAttack(mode='mist',img_path=clean_pth,output_path=f"exp/sr/{outpath}/"+exp_name,g_mode='+',input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}") 

    # outpath = f"texture_only"
    # output = LDMAttack(mode='texture_only',img_path=clean_pth,output_path=f"exp/sr/{outpath}/"+exp_name,g_mode='+',input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}") 

    # outpath = f"sds"
    # output = LDMAttack(mode='sds',img_path=clean_pth,output_path=f"exp/sr/{outpath}/"+exp_name,g_mode='+',input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}") 

    # outpath = f"advdm"
    # output = LDMAttack(mode='advdm',img_path=clean_pth,output_path=f"exp/sr/{outpath}/"+exp_name,g_mode='+',input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}")

  
    # outpath = f"cos_20_5"
    # output = attack(tp=-1,output_path= f"exp/sr/{outpath}/"+exp_name,img_path=imgGT,Esteps=20,target=None,steps=5,input_size=64)
    # output = f"exp/sr/{outpath}/"+exp_name+"x_adv/" 
    # SRexp(img_path = output, output_path = f"exp/sr/{outpath}/"+exp_name+"x_hat/",clean_path=clean_pth,xlsxname=f"{scorefile}_{outpath}") 
 

    pass
