
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from omegaconf import OmegaConf

import torch

import torchvision.transforms as transforms

from attacks import SDEdit
import glob
import ipywidgets as widgets
from notebook_helpers import get_model, get_custom_cond, get_cond_options, get_cond, run
from utils import mp, si, cprint, norm01,tensor2img,psnr_,ssim_,lpips_,load_png,save_score,load_model_from_config,load_image_from_path
from PIL import Image
from diff_mist import LDMAttack
import matplotlib.pyplot as plt
from utils import i2t
from tqdm import tqdm
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths

lpips_fn = lpips.LPIPS(net='alex').to(torch.device("cpu"))

def SRexp(img_path, output_path,clean_path,xlsxname):
    mode = widgets.Select(options=['superresolution'],
    value='superresolution', description='Task:')
    model = get_model(mode.value)
    custom_steps = 100
    os.makedirs(output_path, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
    for img in tqdm(zip(imgs)):
        img_name = os.path.split(img[0])[1]
        cond_choice_path = img[0]
        logs = run(model["model"], cond_choice_path, mode.value, custom_steps)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        Image.fromarray(sample[0]).save(output_path+img_name)
    print("SR SUCCESS")
    scores = []
    score = []

    score += eval_dataset(clean_path,output_path)
    scores += [score]
    save_score(scores,f'{xlsxname}.xlsx')  
    return scores

def SDEexp(img_path = None, output_path = None,clean_path = None, xlsxname = None):
    
    device = 0
    ckpt = "ckpt/sd-v1-4.ckpt"
    dm_config = "configs/stable-diffusion/v1-inference-sd1-4.yaml"
    dm_config = OmegaConf.load(dm_config)
    dm = load_model_from_config(config=dm_config, ckpt=ckpt, device=device).to(device)
    editor = SDEdit(dm=dm)
    input_size =  512


    imgs = glob.glob(img_path+"*.jpg")
    for i in tqdm(imgs):
        file_name = i.split("/")[-1].split('.')[0]

        img = load_image_from_path(i, input_size)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img[:, :, :3]
        trans = transforms.Compose([transforms.ToTensor()])
        x_adv = torch.zeros([1, 3, input_size, input_size]).to(device)
        x_adv[0] = trans(img).to(device)
        t_list = [0.10,0.20,0.30,0.5]
        edit_multi_step  = editor.edit_list(x_adv, restep='ddim100',t_list=t_list)
        for j in range(len(edit_multi_step)):
            si(edit_multi_step[j], output_path + f'sde_{j}/{file_name}.jpg')
    gen_path = output_path
    gen_paths = glob.glob(gen_path+'**')
    print(gen_paths)
    scores = []
    for genpath in gen_paths:
        gp = genpath+'/'
        edit_name = gp.split('/')[-2]
        score = [edit_name]
        score += eval_dataset(clean_path,gp)
        scores += [score]
    save_score(scores,f'{xlsxname}.xlsx')  
    return output_path

def evaluate(imgClean,imgGen):
    ssim = ssim_(imgGen , imgClean)
    # lpips_x = lpips_(x, x_adv)
    psnr = psnr_(imgGen , imgClean)

    lpips_score = lpips_fn(load_png(imgGen , None),load_png(imgClean,None))
    lpips = lpips_score[0, 0, 0, 0].cpu().tolist()
    return [lpips,ssim,psnr]

def eval_dataset(dir_clean,dir_gen):
    imgs_g =  glob.glob(dir_gen+"*.jpg")
    fid = calculate_fid_given_paths([dir_clean,dir_gen],device=0,dims=2048,batch_size=64)
    res = np.array([0.0,0.0,0.0,0.0])
    l = len(imgs_g)
    for i in tqdm(imgs_g):
        f_g = i.split("/")[-1].split('.')[0]
        f_c = f_g
        img_c = dir_clean+f_c+'.jpg'
        score = evaluate(i,img_c)
        res += np.array(score)

    res = (res/l).tolist()
    res += [fid]
    print(res)
    return res

def SDEexp_timestep(img_path = None, output_path = None,clean_path = None, xlsxname = None,noise = None):
    
    device = 0
    ckpt = "ckpt/sd-v1-4.ckpt"
    dm_config = "configs/stable-diffusion/v1-inference-sd1-4.yaml"
    dm_config = OmegaConf.load(dm_config)
    dm = load_model_from_config(config=dm_config, ckpt=ckpt, device=device).to(device)
    editor = SDEdit(dm=dm)
    input_size =  512

    imgs = glob.glob(img_path+"*.jpg")
    for i in imgs:
        file_name = i.split("/")[-1].split('.')[0]
        img = load_image_from_path(i, input_size)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img[:, :, :3]
        trans = transforms.Compose([transforms.ToTensor()])
        x_adv = torch.zeros([1, 3, input_size, input_size]).to(device)
        x_adv[0] = trans(img).to(device)
        t_list = [0.30]
        edit_multi_step  = editor.edit_list(x_adv, restep='ddim100',t_list=t_list,noise=noise)
        si(edit_multi_step[1], output_path + f"{file_name}.jpg")
    gen_path = output_path
    scores = []
    recon_imgs = glob.glob(output_path+"*.jpg")
    clean_img = glob.glob(clean_path+"*.jpg")[0]
    for ri in tqdm(recon_imgs):
        recon_name = os.path.split(ri)[1].split('.')[0]
        t = int(recon_name.split('_')[-1])
        print(t)
        score = [t]
        score += evaluate(clean_img,ri)
        scores += [score]
    save_score(scores,f'{xlsxname}.xlsx')
    return scores