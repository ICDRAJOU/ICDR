import os
import random
import copy
from PIL import Image
import numpy as np
import json

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


# JSON 파일에서 불러오기
with open('prompts.json', 'r') as json_file:
    prompts = json.load(json_file)

# 불러온 데이터에서 각 프롬프트 리스트 가져오기
haze_text = prompts["haze_text"]
rain_text = prompts["rain_text"]
low_text = prompts["low_text"]
snow_text = prompts["snow_text"]
low_rain_text = prompts["low_rain_text"]
low_snow_text = prompts["low_snow_text"]
low_haze_text = prompts["low_haze_text"]
haze_rain_text = prompts["haze_rain_text"]
haze_snow_text = prompts["haze_snow_text"]
low_haze_rain_text = prompts["low_haze_rain_text"]
low_haze_snow_text = prompts["low_haze_snow_text"]



class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args

        self.haze_ids = []
        self.rain_ids = []
        self.low_ids = []
        self.snow_ids = []

        self.haze_rain_ids = []
        self.haze_snow_ids = []
        self.low_haze_ids = []
        self.low_rain_ids = []
        self.low_snow_ids = []

        self.low_haze_rain_ids = []
        self.low_haze_snow_ids = []

        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type

        self.de_dict = {'derain': 0, 'dehaze': 1, 'desnow': 2, 'delow': 3,
                        'dehaze_rain': 4, 'dehaze_snow': 5, 'delow_haze': 6, 'delow_rain': 7,
                        'delow_snow': 8, 'delow_haze_rain': 9, 'delow_haze_snow': 10}

        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        self._init_haze_ids()
        self._init_rain_ids()
        self._init_low_ids()
        self._init_snow_ids()
        self._init_hr_ids()
        self._init_hs_ids()
        self._init_lh_ids()
        self._init_lr_ids()
        self._init_ls_ids()
        self._init_lhr_ids()
        self._init_lhs_ids()

        random.shuffle(self.de_type)

    # haze
    def _init_haze_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.dehaze_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'haze':
                    derain_ids.append(os.path.join(root, file))
        self.haze_ids = derain_ids
        # print(self.hazy_ids)
        self.haze_counter = 0
        self.num_haze = len(self.haze_ids)

    # rain
    def _init_rain_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'rain':
                    derain_ids.append(os.path.join(root, file))
        self.rain_ids = derain_ids
        self.rain_counter = 0
        self.num_rain = len(self.rain_ids)

        # print(self.rain_ids)

    # low
    def _init_low_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        # print(self.args.delow_dir)

        for root, dirs, files in os.walk(self.args.delow_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'low':
                    derain_ids.append(os.path.join(root, file))
        self.low_ids = derain_ids
        self.low_counter = 0
        self.num_low = len(self.low_ids)

    # snow
    def _init_snow_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        desnow_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.desnow_dir):
            # print(root, dirs, files)
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'snow':
                    desnow_ids.append(os.path.join(root, file))
        self.snow_ids = desnow_ids
        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)

    # haze rain
    def _init_hr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.dehaze_rain_dir):
            for file in files:
                # print(os.path.basename(root))
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'haze_rain':
                    derain_ids.append(os.path.join(root, file))

        self.haze_rain_ids = derain_ids
        self.hr_counter = 0
        self.num_hr = len(self.haze_rain_ids)

    # haze snow
    def _init_hs_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.dehaze_snow_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'haze_snow':
                    derain_ids.append(os.path.join(root, file))
        self.haze_snow_ids = derain_ids
        self.hs_counter = 0
        self.num_hs = len(self.haze_snow_ids)

    # low_rain
    def _init_lr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)
        # print(name_list)
        for root, dirs, files in os.walk(self.args.delow_rain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'low_rain':
                    derain_ids.append(os.path.join(root, file))
        self.low_rain_ids = derain_ids
        self.lr_counter = 0
        self.num_lr = len(self.low_rain_ids)

    # low haze
    def _init_lh_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.delow_haze_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'low_haze':
                    derain_ids.append(os.path.join(root, file))
        self.low_haze_ids = derain_ids
        self.lh_counter = 0
        self.num_lh = len(self.low_haze_ids)

    # low haze
    def _init_ls_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.delow_snow_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'low_snow':
                    derain_ids.append(os.path.join(root, file))
        self.low_snow_ids = derain_ids
        self.ls_counter = 0
        self.num_ls = len(self.low_snow_ids)

    # low haze rain
    def _init_lhr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.delow_haze_rain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'low_haze_rain':
                    derain_ids.append(os.path.join(root, file))
        self.low_haze_rain_ids = derain_ids
        self.lhr_counter = 0
        self.num_lhr = len(self.low_haze_rain_ids)

    # low haze snow
    def _init_lhs_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.delow_haze_snow_dir)

        for root, dirs, files in os.walk(self.args.delow_haze_snow_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(
                        root) == 'low_haze_snow':
                    derain_ids.append(os.path.join(root, file))
        self.low_haze_snow_ids = derain_ids
        self.lhs_counter = 0
        self.num_lhs = len(self.low_haze_snow_ids)


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = '../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/clear/' + rainy_name.split('/')[-1]
        return gt_name

    def __getitem__(self, _):
        text_prompt = ""
        de_id = self.de_dict[self.de_type[self.de_temp]]

        if de_id == 0:
            degrad_img = crop_img(np.array(Image.open(self.haze_ids[self.haze_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.haze_ids[self.haze_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.haze_counter = (self.haze_counter + 1) % self.num_haze
            if self.haze_counter == 0:
                random.shuffle(self.haze_ids)
            text_prompt = haze_text[random.randrange(0,len(haze_text))]

        elif de_id == 1:
            degrad_img = crop_img(np.array(Image.open(self.rain_ids[self.rain_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.rain_ids[self.rain_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.rain_counter = (self.rain_counter + 1) % self.num_rain
            if self.rain_counter == 0:
                random.shuffle(self.rain_ids)
            text_prompt = rain_text[random.randrange(0,len(rain_text))]

        elif de_id == 2:
            degrad_img = crop_img(np.array(Image.open(self.low_ids[self.low_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_ids[self.low_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.low_counter = (self.low_counter + 1) % self.num_low
            if self.low_counter == 0:
                random.shuffle(self.low_ids)
            text_prompt = low_text[random.randrange(0,len(low_text))]


        elif de_id == 3:
            degrad_img = crop_img(np.array(Image.open(self.snow_ids[self.snow_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.snow_ids[self.snow_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.snow_counter = (self.snow_counter + 1) % self.num_snow
            if self.snow_counter == 0:
                random.shuffle(self.snow_ids)
            text_prompt = snow_text[random.randrange(0, len(snow_text))]


        elif de_id == 4:
            degrad_img = crop_img(np.array(Image.open(self.haze_rain_ids[self.hr_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.haze_rain_ids[self.hr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.hr_counter = (self.hr_counter + 1) % self.num_hr
            if self.hr_counter == 0:
                random.shuffle(self.haze_rain_ids)
            text_prompt = haze_rain_text[random.randrange(0, len(haze_rain_text))]

        elif de_id == 5:
            degrad_img = crop_img(np.array(Image.open(self.haze_snow_ids[self.hs_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.haze_snow_ids[self.hs_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.hs_counter = (self.hs_counter + 1) % self.num_hs
            if self.hs_counter == 0:
                random.shuffle(self.haze_snow_ids)
            text_prompt = haze_snow_text[random.randrange(0, len(haze_snow_text))]


        elif de_id == 6:
            degrad_img = crop_img(np.array(Image.open(self.low_rain_ids[self.lr_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_rain_ids[self.lr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.lr_counter = (self.lr_counter + 1) % self.num_lr
            if self.lr_counter == 0:
                random.shuffle(self.low_rain_ids)
            text_prompt = low_rain_text[random.randrange(0, len(low_rain_text))]


        elif de_id == 7:
            degrad_img = crop_img(np.array(Image.open(self.low_haze_ids[self.lh_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_haze_ids[self.lh_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.lh_counter = (self.lh_counter + 1) % self.num_lh
            if self.lh_counter == 0:
                random.shuffle(self.low_haze_ids)
            text_prompt = low_haze_text[random.randrange(0, len(low_haze_text))]


        elif de_id == 8:
            degrad_img = crop_img(np.array(Image.open(self.low_snow_ids[self.ls_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_snow_ids[self.ls_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.ls_counter = (self.ls_counter + 1) % self.num_ls
            if self.ls_counter == 0:
                random.shuffle(self.low_snow_ids)
            text_prompt = low_snow_text[random.randrange(0, len(low_snow_text))]


        elif de_id == 9:
            degrad_img = crop_img(np.array(Image.open(self.low_haze_rain_ids[self.lhr_counter]).convert('RGB')),
                              base=16)
            clean_name = self._get_gt_name(self.low_haze_rain_ids[self.lhr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.lhr_counter = (self.lhr_counter + 1) % self.num_lhr
            if self.lhr_counter == 0:
                random.shuffle(self.low_haze_rain_ids)
            text_prompt = low_haze_rain_text[random.randrange(0, len(low_haze_rain_text))]


        elif de_id == 10:
            degrad_img = crop_img(np.array(Image.open(self.low_haze_snow_ids[self.lhs_counter]).convert('RGB')),
                                  base=16)
            clean_name = self._get_gt_name(self.low_haze_snow_ids[self.lhs_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            self.lhs_counter = (self.lhs_counter + 1) % self.num_lhs
            if self.lhs_counter == 0:
                random.shuffle(self.low_haze_snow_ids)
            text_prompt = low_haze_snow_text[random.randrange(0, len(low_haze_snow_text))]

        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)

        self.de_temp = (self.de_temp + 1) % len(self.de_type)
        if self.de_temp == 0:
            random.shuffle(self.de_type)

        return [clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2, text_prompt

    def __len__(self):
        return 400 * len(self.args.de_type)


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        self.ids = []

        for root, dirs, files in os.walk(self.args.derain_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and 'clear' not in root:
                    self.ids.append(os.path.join(root, file))

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        # gt_name = degraded_name.split('/')[0] + '/' + degraded_name.split('/')[1] + '/clear/' + degraded_name.split('/')[-1]
        gt_name = './data/CDD-11_test_100/clear/' + degraded_name.split('/')[-1]
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        degradation = "low"

        text_prompt = "I have to post an emotional shot on Instagram, but it was shot too foggy and too dark. Change it like a sunny day and brighten it up!"
        '''
        if degradation == "haze":
            text_prompt = "Eliminate the haze for better visibility."

        elif degradation == "rain":
            text_prompt = "Remove the raindrops and make the image clearer."

        elif degradation == "low":
            text_prompt = "Enhance visibility in the low-light parts of the image."

        elif degradation == "haze_rain":
            text_prompt = "Remove the fog and rain to improve the clarity of the image."

        elif degradation == "low_rain":
            text_prompt = "Remove both rain and low-light effects to improve the image quality."

        elif degradation == "low_haze":
            text_prompt = "Clear the fog and lighten up the dim regions for better clarity."

        elif degradation == "low_haze_rain":
            text_prompt = "Clear the rain, fog, and brighten the image for improved visibility."
        '''

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degradation, degraded_img, clean_img, text_prompt 

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        degraded_ids = []
        # 파일 목록 중 이미지 파일만 필터링
        name_list = os.listdir(root)
        name_list = [file for file in name_list if os.path.splitext(file)[1].lower() in self.image_extensions]
        self.degraded_ids += [root + id_ for id_ in name_list]

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
