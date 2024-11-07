import subprocess
from tqdm import tqdm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# GPU NUM 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "7,9"
os.environ["MASTER_ADDR"] = "localhost"  # DDP 통신을 위한 마스터 주소
os.environ["MASTER_PORT"] = "12412"  # 포트 번호

import clip
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dataset_utils_CDD import TrainDataset
from text_net.model import AirNet
from option import options as opt

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, clip_model):
    setup(rank, world_size)

    print(f'cuda{rank}')
    device = torch.device(f'cuda:{rank}')
    clip_model = clip_model.to(device)


    # Checkpoint 디렉터리 생성
    if rank == 0:  # 프로세스 0에서만 생성
        subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    # Dataset 및 DataLoader 설정
    trainset = TrainDataset(opt)
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, sampler=sampler,
                                 pin_memory=True, drop_last=True, num_workers=opt.num_workers)
    save_epoch = 0

    # Network Construction
    net = AirNet(opt, clip = clip_model).cuda(rank)
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    net.train()

    if opt.ckpt != "none" :
        net.load_state_dict(torch.load(opt.ckpt))
        save_epoch = int(opt.ckpt.split('_')[1][:-4])

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()
    min_loss = 1e9

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        trainloader.sampler.set_epoch(epoch)  # Epoch마다 sampler 설정을 바꾸기
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2, text_prompt) in tqdm(trainloader,
                                                                                                        disable=rank != 0):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(rank), degrad_patch_2.cuda(rank)
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(rank), clean_patch_2.cuda(rank)

            optimizer.zero_grad()

            if epoch < opt.epochs_encoder:
                _, output, target, _ = net.module.E(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target.cuda(rank))
                loss = contrast_loss
            else:
                restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2, text_prompt=text_prompt)
                contrast_loss = CE(output, target.cuda(rank))
                l1_loss = l1(restored, clean_patch_1)
                loss = l1_loss + 0.1 * contrast_loss

            # Backward and optimization step
            loss.backward()
            optimizer.step()

        if rank == 0:
            if epoch < opt.epochs_encoder:
                print(
                    'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                        epoch, contrast_loss.item(),
                    ), '\r', end='')
            else:
                print(
                    'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n' % (
                        epoch, l1_loss.item(), contrast_loss.item(),
                    ), '\r', end='')

        GPUS = 1
        if rank == 0 and (min_loss > loss or epoch % 20 == 0) and epoch > 98:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if epoch > 100:
                min_loss = loss
                torch.save(net.state_dict(), os.path.join(opt.ckpt_path, f'epoch_{epoch + 1}.pth'))

        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 60))
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # 사용할 GPU 수
# DDP 시작 전 main에서 한 번만 clip 모델을 로드
    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')  # 로드 후 필요한 rank로 이동
    mp.spawn(train, args=(world_size, clip_model), nprocs=world_size, join=True)

