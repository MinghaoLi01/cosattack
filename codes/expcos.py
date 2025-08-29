import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from attacks import SDEdit
import glob
from utils import tensor2img,load_image_from_path,load_model_from_config

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

def expcos(zadv):
    device = 0
    output = 'exp/exp/cos/458_noise'
    img = 'test_images/exp/458.jpg'
    dm = get_dm()
    T = dm.num_timesteps
    x = i2t(img)
    editor = SDEdit(dm=dm)
    for k in range(10):
        with torch.no_grad():

            cnd = dm.get_learned_conditioning("")
            z = dm.get_first_stage_encoding(dm.encode_first_stage(x)).to(device)

            t_ = 100*torch.ones((z.shape[0],), device=z.device).long()
            noise = torch.randn_like(z).cuda()
            z_adv = z + 0.05*k*noise
            z_noisy = dm.q_sample(x_start=z_adv, t=t_, noise=noise)
            eps = dm.apply_model(z_noisy, t_, cond=cnd)
            x_hat = editor.edit(x=z_adv,guidance=0.1,restep="ddim100",from_z=True)
            print(k)
            print(cos(eps,noise))
            print(nn.functional.mse_loss(eps, noise, reduction='none').mean([1, 2, 3]).item())
            tensor2img(x_hat,f"{output}/x_hat_{k}.jpg")
        # eps_v , eps_p = decompose_tensor(eps,noise)
        # eps_v /= torch.norm(eps_v,p=2)
        # eps_p /= torch.norm(eps_p,p=2)
        # noisep = noise + torch.norm(noise,p=2)*eps_p
        # noisev = noise + torch.norm(noise,p=2)*eps_v
        # zp_hat = dm.predict_start_from_noise(z_noisy, t=t_, noise=noisep)
        # zv_hat = dm.predict_start_from_noise(z_noisy, t=t_, noise=noisev)
 



        # deltap = zp_hat - z
        # deltav = zv_hat - z


        # cosp = cos(eps_p,noise)
        # cosv = cos(eps_v,noise)
        # l2p = nn.functional.mse_loss(eps_p, noise, reduction='none').mean([1, 2, 3]).item() 
        # l2v = nn.functional.mse_loss(eps_v, noise, reduction='none').mean([1, 2, 3]).item()
        # deltap2 = torch.norm(deltap,p=2)
        # deltav2 = torch.norm(deltav,p=2)
        # print(f"{cosp} \t {cosv}")
        # print(f"{l2p} \t {l2v}")
        # print(f"{deltap2} \t {deltav2}")
        # tensor2img(zp_hat[:,:3,:,:],f"{output}/zp_hat.jpg")
        # tensor2img(zv_hat[:,:3,:,:],f"{output}/zv_hat.jpg")
        # tensor2img(deltap[:,:3,:,:],f"{output}/dp.jpg")
        # tensor2img(deltav[:,:3,:,:],f"{output}/dv.jpg")
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
    for i in range(steps):
        t_ = t_P*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
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
            loss = torch.nn.functional.mse_loss(eps_pred, noise, reduction='none').mean([1, 2, 3])
        p  = z_adv - z_raw
        eps_v,eps_p = decompose_tensor(eps_pred,p)
        y,x = decompose_tensor(eps_pred,noise)
        x = math.copysign(torch.norm(x,p=2),cos(eps_pred,noise))
        y = torch.norm(y,p=2).item()
        lst = [torch.norm(eps_pred).item(),torch.norm(eps_p,p=2).item(),torch.norm(p - eps_p,p=2).item(),cos(p,eps_p),torch.norm(p,p=2).item()]
        #  x,y, |eps|,|eps_p|,|p-eps|,|p|,cosloss,mseloss
        # trace. eps  eps_p  p-epsp p ,vis_p-epsp
        # print(lst)

        grad = torch.autograd.grad(loss,z_adv)[0]
        g = grad / grad.std()
        gs += g
        z_adv.grad = None  
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
        
        with torch.no_grad():
            z_raw = dm.get_first_stage_encoding(dm.encode_first_stage(x_raw)).to(device)
        torch.manual_seed(100)
        noise = torch.randn_like(z_raw).cuda()
        print(torch.norm(noise,p=2))
        t = tp
        tlst = torch.tensor(sorted(random.sample(range(0,T),100)))
        tlst = tlst.reshape(steps,Esteps).t().tolist()
        # tlst.reverse()
        losses = []
        lst = []
        for _ in range(Esteps):
            tlst_ = tlst[_]
            with torch.no_grad():
                z = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv)).to(device)  
            z_grad,loss,l = getTargetZ(z,z_raw,dm,t,steps,c,target_z,noise,tlst=tlst_)
            x_adv = getXadv(x_raw,x_adv,z_grad=z_grad,stepsize=Estepsize*2,steps=steps,eps=Eeps*2,dm=dm)
            losses += [loss]
            lst += [l]

        # save_score(lst,f"trace_mseT{t}.xlsx")

        # with torch.no_grad():
        #     output = 'exp/exp/cost100/458_cos'
        #     cnd = dm.get_learned_conditioning("")
        #     z_adv = dm.get_first_stage_encoding(dm.encode_first_stage(x_adv)).to(device)


        #     t_ = 100*torch.ones((z_raw.shape[0],), device=z_raw.device).long()
        #     z_noisy = dm.q_sample(x_start=z_adv, t=t_, noise=noise)
        #     eps = dm.apply_model(z_noisy, t_, cond=cnd)
        #     z_hat = dm.predict_start_from_noise(z_noisy, t=t_, noise=eps)
        #     tensor2img(z_hat[:,:3,:,:],f"{output}/z_hat.jpg")
        #     tensor2img(z_raw[:,:3,:,:],f"{output}/z_raw.jpg")
        #     delta = z_hat - z_raw
        #     delta2 = torch.norm(delta,p=2)
        #     print(f"{delta2}")
        #     p  = z_adv - z_raw
        #     eps_v,eps_p = decompose_tensor(eps,p)
        #     print(f"{cos(p,eps)} \t {torch.norm(p,p=2)} \t {torch.norm(eps,p=2)} \t {torch.norm(eps_p,p=2)}")
        #     tensor2img(delta[:,:3,:,:],f"{output}/d.jpg")
        #     tensor2img(z_adv[:,:3,:,:],f"{output}/z_adv.jpg")
        #     eps_v,eps_p = decompose_tensor(eps,noise)
        #     print(torch.norm(eps_p,p=2),torch.norm(eps_v,p=2))
        #     print(cos(eps_p,noise),cos(eps_v,noise))

            # zp = z_raw + delta_p
            # zv = z_raw + delta_v
            # zp_noisy = dm.q_sample(x_start=zp, t=t_, noise=noise)
            # epsp = dm.apply_model(zp_noisy, t_, cond=cnd)
            # zv_noisy = dm.q_sample(x_start=zv, t=t_, noise=noise)
            # epsv = dm.apply_model(zv_noisy, t_, cond=cnd)
            # # z0_hat = dm.predict_start_from_noise(z0_noisy, t=t_, noise=eps0)
            # # tensor2img(z0_hat[:,:3,:,:],f"{output}/z0_hat.jpg")
            
            # # z_noisy = dm.q_sample(x_start=z_adv, t=t_, noise=noise)
            # # eps = dm.apply_model(z_noisy, t_, cond=cnd)
            # # z_hat = dm.predict_start_from_noise(z_noisy, t=t_, noise=eps)
            # # delta = z_hat - z_adv
            # # delta2 = torch.norm(delta,p=2)
            # cos_v = cos(epsv,noise)
            # cos_p = cos(epsp,noise)
            # l2v = nn.functional.mse_loss(epsv, noise, reduction='none').mean([1, 2, 3]).item() 
            # l2p = nn.functional.mse_loss(epsp, noise, reduction='none').mean([1, 2, 3]).item() 

            # # 
            # 
            # tensor2img(noise[:,:3,:,:],f"{output}/noise.jpg")
            # print(f"{cos_p} \t {cos_v}")
            # print(f"{l2p} \t {l2v}")
            # print(f"{delta2}")
            # expcos(z_adv)

    # tensor2img(x_adv,output_path+f"/x_adv/{img_name}.jpg")
    
    return output_path+"x_adv/"

def main():
    # expcos(None)
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
    tlst = list(range(0,901,100))
    print(tlst)
    # for t in tlst:
    outpath = f"exp_t100_l2"
    output = attack(tp=100,output_path= f"exp/out/{outpath}/"+exp_name,img_path=imgGT,Esteps=100,target=None,steps=1)
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
