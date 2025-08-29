import torch
import numpy as np
import torchvision
from colorama import Fore, Back, Style
import os
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
from clip_similarity import clip_sim
import lpips
import pandas as pd
from ldm.util import instantiate_from_config
import PIL

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def load_png(p, size, mode='bicubic'):
    x = Image.open(p).convert('RGB')

    if mode == 'bicubic':
        inter_mode = transforms.InterpolationMode.BICUBIC
    elif mode == 'bilinear':
        inter_mode = transforms.InterpolationMode.BILINEAR

    # Define a transformation to resize the image and convert it to a tensor
    if size is not None:
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=inter_mode),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        

    x = transform(x)
    return x

def cprint(x, c):
    c_t = ""
    if c == 'r':
        c_t = Fore.RED
    elif c == 'g':
        c_t = Fore.GREEN
    elif c == 'y':
        c_t = Fore.YELLOW
    elif c == 'b':
        c_t = Fore.BLUE
    print(c_t, x)
    print(Style.RESET_ALL)

def si(x, p, to_01=False, normalize=False):
    mp(p)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if to_01:
        torchvision.utils.save_image((x+1)/2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def mp(p):
    # if p is like a/b/c/d.png, then only make a/b/c/
    first_dot = p.find('.')
    last_slash = p.rfind('/')
    if first_dot < last_slash:
        assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
    p_new = p[:last_slash] + '/'
    if not os.path.exists(p_new):
        os.makedirs(p_new)


def get_plt_color_list():
    return ['red', 'green', 'blue', 'black', 'orange', 'yellow', 'black']


    
   
def draw_bound(a, m, color):
    if a.device != 'cpu':
        a = a.cpu()
    if color == 'red':
        c = torch.ones((3, 224, 224)) * torch.tensor([1, 0, 0])[:, None, None]
    if color == 'green':
        c = torch.ones((3, 224, 224)) * torch.tensor([0, 1, 0])[:, None, None]
    
    return c * m + a * (1 - m)

# class EasyDict(dict):
#     """Convenience class that behaves like a dict but allows access with the attribute syntax."""

#     def __getattr__(self, name: str) -> Any:
#         try:
#             return self[name]
#         except KeyError:
#             raise AttributeError(name)

#     def __setattr__(self, name: str, value: Any) -> None:
#         self[name] = value

#     def __delattr__(self, name: str) -> None:
#         del self[name]


def smooth_loss(output, weight):
    tv_loss = torch.sum(
        (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
        (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight



def compose_images_in_folder(p, dim, size=224):
    l = glob.glob(p + '*.png')
    l += glob.glob(p + '*.jpg')
    print(l)
    return torch.cat([load_png(item, size) for item in l], dim)



def get_bkg(m, e=0.01):
    assert  len(m.shape) == 4
    b = [0.2667, 0, 0.3255]
    m_0 = (m[:, 0, ...] > b[0] - e) * (m[:, 0, ...] < b[0] + e)
    m_1 = (m[:, 1, ...] > b[1] - e) * (m[:, 1, ...] < b[1] + e)
    m_2 = (m[:, 2, ...] > b[2] - e) * (m[:, 2, ...] < b[2] + e)
    m =   1. - (m_0 * m_1 * m_2).float()
    return m[None, ...]



def lpips_(a, b ):

    lpips_score = lpips.LPIPS(net='alex').to(a.device)
    return lpips_score(a, b)


def image_align(a, b):
    
    pass



def ssim_(p1, p2):
    i1 = cv2.imread(p1)
    i2 = cv2.imread(p2)
    
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    
    return ssim(i1, i2)

from math import log10, sqrt

def psnr_(a, b):
    original = cv2.imread(a)
    compressed = cv2.imread(b)
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
    
    


def psnr():
    pass

def norm01(x):
    x_ = (x+1) / 2
    # min_tensor = torch.min(x_)
    # max_tensor = torch.max(x_)
    y = torch.clamp(x_, 0, 1)
    return y

def tensor2img(tensor,path):
    mp(path)
    t = norm01(tensor)

    image = transforms.ToPILImage()(t.squeeze(0))
    image_RGB = image.convert('RGB')
    image_RGB.save(path)

def save_score(datalst,savepath):
    df = pd.DataFrame(datalst)
    df.to_excel(savepath,index=False,header=False)


def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img

def load_model_from_config(config, ckpt, verbose: bool = False, device=0):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cuda:0")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.cond_stage_model.to(device)
    model.eval()
    return model

def i2t(image,input_size=512,device=0):
    img = load_image_from_path(image, input_size)
    img = img.convert('RGB')
    trans = transforms.Compose([transforms.ToTensor()])
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    x = torch.zeros([1, 3, input_size, input_size])
    x[0] = trans(img)
    x = x.to(device)
    x = x.clone().detach()
    return x
