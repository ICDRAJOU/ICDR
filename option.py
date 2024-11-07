import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=1)

parser.add_argument('--ckpt', type=str, default="none")
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=0, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of encoder.')

parser.add_argument('--de_type', type=list, default=['derain', 'dehaze', 'desnow', 'delow',
                                                                'dehaze_rain', 'dehaze_snow', 'delow_haze', 'delow_rain', 'delow_snow',
                                                                'delow_haze_rain', 'delow_haze_snow'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--encoder_dim', type=int, default=256, help='the dimensionality of encoder.')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train',  help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/rain',
                    help='where clean images of derain saves.')
parser.add_argument('--desnow_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/snow',
                    help='where clean images of desnow saves.')
parser.add_argument('--dehaze_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/haze',
                    help='where clean images of dehaze saves.')
parser.add_argument('--delow_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low',
                    help='where clean images of delow saves.')

parser.add_argument('--dehaze_rain_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/haze_rain',
                    help='where clean images of derain saves.')
parser.add_argument('--dehaze_snow_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/haze_snow',
                    help='where clean images of derain saves.')
parser.add_argument('--delow_haze_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low_haze',
                    help='where clean images of derain saves.')
parser.add_argument('--delow_rain_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low_rain',
                    help='where clean images of derain saves.')
parser.add_argument('--delow_snow_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low_snow',
                    help='where clean images of derain saves.')

parser.add_argument('--delow_haze_rain_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low_haze_rain',
                    help='where clean images of derain saves.')
parser.add_argument('--delow_haze_snow_dir', type=str, default='../../../shared/hdd_ext/nvme1/siwon/CDD-11_train/low_haze_snow',
                    help='where clean images of derain saves.')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')

options = parser.parse_args()
options.batch_size = len(options.de_type)
