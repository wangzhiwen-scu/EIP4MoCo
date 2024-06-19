import argparse

# def GenConfig():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
#     parser.add_argument('--model_name', type=str, default='DOTA', help='model name')
#     parser.add_argument('--dataset_name', type=str, default='my_dataset', help='name of the dataset')
#     parser.add_argument('--coil_input', type=str, default='cc', help='input coil type')
#     parser.add_argument('--coil_label', type=str, default='cc', help='label coil type')
#     parser.add_argument('--coil_num', type=int, default=16)
#     parser.add_argument('--coil_ind', type=int, default=1)
#     parser.add_argument('--abs_layer', type=bool, default=False, help='add last abs layer')
#     parser.add_argument('--brain_masking', type=bool, default=False, help='brain masking')
    
#     parser.add_argument('--acc_rate', type=float, default=5.02, help='acceleration ratio')
#     parser.add_argument('--acs_num', type=int, default=16, help='the number of acs lines')
    
#     parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
#     parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
    
#     parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
#     parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate') # start: 0.00005
#     parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
#     parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    
#     parser.add_argument('--im_height', type=int, default=240, help='size of image height')
#     parser.add_argument('--im_width', type=int, default=240, help='size of image width')
#     parser.add_argument('--channels', type=int, default=2, help='number of image channels')
    
#     parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
#     parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
#     parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
#     parser.add_argument('--mask_index', type=int, default=0, help='validation/test mask index')
    
#     parser.add_argument('--data_augment', type=bool, default=False, help='32-fold data augmentation')
#     parser.add_argument('--random_sampling', type=bool, default=False, help='Generate random sampling patterns during training')

#     opt = parser.parse_args()
    
#     # Model-related hyperparameters
#     parser.add_argument('--k', type=int, default=10)
#     parser.add_argument('--i', type=int, default=10)
#     parser.add_argument('--fm', type=int, default=64)
    
#     parser.add_argument('--in_ch', type=int, default=opt.channels)
#     parser.add_argument('--out_ch',type=int, default=opt.channels)
#     parser.add_argument('--dota_dc',type=bool, default=True)

#     opt = parser.parse_args()
    
#     print(opt)
    
#     return opt


class Config():
    def __init__(self) -> None:
        self.im_height = 240
        self.im_width = 240
        self.in_ch = 2
        self.out_ch = 2
        
        self.k =  10
        self.fm = 64
        self.i = 10
        self.dota_dc = True
        self.abs_layer = False

        