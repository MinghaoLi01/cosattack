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
from attacks import SDEdit
import math
def cos(a,b):
    return torch.cosine_similarity(a.flatten(),b.flatten(),dim=0).item()    

def decompose_tensor(input_tensor,ref):
    tmp = ref
    proj = torch.sum(input_tensor * tmp) / torch.sum(tmp * tmp) * tmp
    orthogonal_tensor = input_tensor - proj
    return orthogonal_tensor,proj

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

    lst = []

    # lst += [[t_,c1,c2,c1/c2]]
    # print(t_,c1/c2)

    for i in range(steps):
        t_ = t_P*torch.ones((z_raw.shape[0],), device=z_raw.device).long()

        beta = extract_into_tensor(dm.betas, t_, z_raw.shape).item()
        c1 = extract_into_tensor(dm.sqrt_alphas_cumprod, t_, z_raw.shape).item()
        c2 = beta / extract_into_tensor(dm.sqrt_one_minus_alphas_cumprod, t_, z_raw.shape).item()
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
            # loss = 1 - torch.cosine_similarity(eps_pred.flatten(),noise_.flatten(),dim=0)   
            loss = torch.nn.functional.mse_loss(eps_pred, noise, reduction='none').mean([1, 2, 3])
        grad = torch.autograd.grad(loss,z_adv)[0]
        g = grad / grad.std()
        gs += g
        z_adv.grad = None  
        p  = z_adv - z_raw
        eps_v,eps_p = decompose_tensor(eps_pred,p)
        y,x = decompose_tensor(eps_pred,noise)
        x = math.copysign(torch.norm(x,p=2),cos(eps_pred,noise))
        y = torch.norm(y,p=2).item()
        lst = [t_P,c1,c2,torch.norm(eps_pred).item(),torch.norm(eps_p,p=2).item(),torch.norm(p - eps_p,p=2).item(),torch.norm(p,p=2).item(),cos(p,eps_pred)]
        # t c1 c2 |eps|,|eps_p|,|p-eps|,|p| cos
        # print(lst)
    gs = gs / 5
    
    return gs.detach(),loss.item(),lst

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
        lst = []
        with torch.no_grad():
            z_raw = dm.get_first_stage_encoding(dm.encode_first_stage(x_raw)).to(device)
            sum_ =0 
            for t in range(1,1000):
                t_ = t*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
                t_1 = t-1*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
                beta = extract_into_tensor(dm.betas, t_, z_raw.shape).item()
                c1 = extract_into_tensor(dm.sqrt_alphas_cumprod, t_1, z_raw.shape).item()
                c2 = beta / extract_into_tensor(dm.sqrt_one_minus_alphas_cumprod, t_, z_raw.shape).item()
                si = c2
                sum_ += si 
                print(si)
            print("sum:",sum_)
        torch.manual_seed(100)
        noise = torch.randn_like(z_raw).cuda()
        # print(torch.norm(noise,p=2))
        t = tp
        tlst = torch.tensor(sorted(random.sample(range(0,T),100)))
        tlst = tlst.reshape(steps,Esteps).t().tolist()
        # tlst.reverse()
        losses = []
        lst = []
        for _ in range(Esteps):
            with torch.no_grad():
                z = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv)).to(device)  
            z_grad,loss,l = getTargetZ(z,z_raw,dm,t,steps,c,target_z,noise)
            x_adv = getXadv(x_raw,x_adv,z_grad=z_grad,stepsize=Estepsize*2,steps=steps,eps=Eeps*2,dm=dm)
            losses += [loss]
            lst = l    
    
    return output_path+"x_adv/",lst

def main():
    # expcos(None)
    target = "test_images/target/MIST.png"

    exp_name = '888/'
    imgGT = 'test_images/exp/'
    clean_pth = imgGT
    scorefile = exp_name[:-1]
    tlst = list(range(0,991,100))
    print(tlst)
    lst =[]
    outpath = f"exp_coeff_l2"
    for t_ in tlst:
        output,_ = attack(tp=t_,output_path= f"exp/out/{outpath}/"+exp_name,img_path=imgGT,Esteps=100,target=None,steps=1)
        print(_)
        lst += [_]
    save_score(lst,f"exp/rebuttal/ceoff_888_mse.xlsx")
        # output = f"exp/out/{outpath}/"+exp_name+"x_adv/" 
        # SDEexp(img_path = output, output_path = f"exp/sde/{outpath}/"+exp_name,clean_path=clean_pth,xlsxname=f"exp/exp/res/{scorefile}_{outpath}")

    return   



if __name__ == '__main__':
    main()
    # dm = get_dm()

    pass
            # tlst = list(range(0,991,10))
            # print(tlst)
            # for t_ in tlst:
            #     t = t_*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
            #     t_plus = (1+t_)*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
            #     # z_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            #     # extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            #     # z_recon =(
            #     #     extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            #     #     extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps_pred
            #     #         )
            #     # posterior_mean = (
            #     #             extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * z_recon  +
            #     #             extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            #     #     )
            #     # e_cf = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)*extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
            #     # c_z0 =(extract_into_tensor(dm.posterior_mean_coef1, t, z_raw.shape)  + extract_into_tensor(dm.posterior_mean_coef2, t, z_raw.shape)*extract_into_tensor(dm.sqrt_recip_alphas_cumprod, t, z_raw.shape) )* extract_into_tensor(dm.sqrt_alphas_cumprod, t_plus, z_raw.shape)

            #     # c_noise =(extract_into_tensor(dm.posterior_mean_coef2, t, z_raw.shape) * extract_into_tensor(dm.sqrt_recip_alphas_cumprod, t, z_raw.shape) + extract_into_tensor(dm.posterior_mean_coef1, t, z_raw.shape) )* extract_into_tensor(dm.sqrt_one_minus_alphas_cumprod, t_plus, z_raw.shape) 
            #     # c_eps_pred = extract_into_tensor(dm.posterior_mean_coef2, t, z_raw.shape) * extract_into_tensor(dm.sqrt_recipm1_alphas_cumprod, t, z_raw.shape) 
            #     # print(t_,c_z0.item(),c_noise.item(),c_eps_pred.item(),c_z0.item()/c_noise.item())
                
            #     beta = extract_into_tensor(dm.betas, t, z_raw.shape).item()
            #     c1 = extract_into_tensor(dm.sqrt_alphas_cumprod, t, z_raw.shape).item()
            #     c2 = beta / extract_into_tensor(dm.sqrt_one_minus_alphas_cumprod, t, z_raw.shape).item()
            #     lst += [[t_,c1,c2,c1/c2]]
            #     print(t_,c1/c2)
                
                # z_mean  =  c_z0 * z_0 +  c_noise * noise- c_eps_pred * eps_pred 
            # save_score(lst,"c1divc2.xlsx")