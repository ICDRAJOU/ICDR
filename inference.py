import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils_CDD import DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from text_net.model import AirNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_Derain_Dehaze(opt, net, dataset, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    # dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    print(len(testloader))

    with torch.no_grad():
        for ([degraded_name], degradation, degrad_patch, clean_patch, text_prompt) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(x_query=degrad_patch, x_key=degrad_patch, text_prompt = text_prompt)

        return save_image_tensor(restored)


def infer(text_prompt = "", img=None):
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--derain_path', type=str, default="data/Test_prompting/", help='save path of test raining images')
    parser.add_argument('--output_path', type=str, default="output/demo11", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/epoch_287.pth", help='checkpoint save path')
    # parser.add_argument('--text_prompt', type=str, default="derain")

    opt = parser.parse_args()
    # opt.text_prompt = text_prompt

    np.random.seed(0)
    torch.manual_seed(0)

    opt.batch_size = 7
    ckpt_path = opt.ckpt_path

    derain_set = DerainDehazeDataset(opt, img=img, text_prompt = text_prompt)

    # Make network
    net = AirNet(opt).to(device)
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=device))

    restored = test_Derain_Dehaze(opt, net, derain_set, task="derain")

    return restored
