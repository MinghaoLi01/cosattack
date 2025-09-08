# [IJCAI'2025] Preventing Latent Diffusion Model-Based Image Mimicry via Angle Shifting and Ensemble Learning.

## Setup

### **Run the following commands to set up**

```
conda env create -f env.yml
conda activate cosattack
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install --force-reinstall pillow
```

If you encounter the error `ERROR: Command errored out with exit status 128` when installing `taming-transformers` or  `clip`, you can resolve it by manually cloning and installing the repository (Make sure your network is connected to GitHub properly).

```
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
```

### Download the models

* [stable diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main)
* [stable diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)
* [stable diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)

## Run

```
python codes/cosattack.py
```

## Cited as

```
@inproceedings{li2025preventing,
  title={Preventing latent diffusion model-based image mimicry via angle shifting and ensemble learning},
  author={Li, Minghao and Wang, Rui and Sun, Ming and Jing, Lihua},
  booktitle={The 34th International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

=======

# [IJCAI'2025] Preventing Latent Diffusion Model-Based Image Mimicry via Angle Shifting and Ensemble Learning.

>>>>>>> 800a8b8c4732f45e2da04cf568c54fd65cb09986
>>>>>>>
>>>>>>
>>>>>
>>>>
>>>
>>
