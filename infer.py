import glob
import os, losses, utils
os.system('pip install SimpleITK')
os.system('pip install -U --pre statsmodels')
os.system('pip install ml_collections')
os.system('pip install natsort')
os.system('pip install nibabel')
os.system('pip install timm')
os.system('pip install einops')
os.system('pip install batchgenerators')
#os.system('pip install medpy')
from torch.utils.data import DataLoader
from data import datasets, trans
from eval import compute_label_dice, compute_label_IOU, psnr, ssim, NJ_num, time_difference
import numpy as np
import torch, models, pvt, TM_OLEB
from torchvision import transforms
#from ignite.contrib.handlers import ProgressBar
import matplotlib.pyplot as plt
#from medpy.metric.binary import hd95
from models import CONFIGS as CONFIGS_ViT_seg
from natsort import natsorted
import SimpleITK as sitk
import nibabel as nb
from TransMorph import CONFIGS as CONFIGS_TM
import datetime
import TransMorph
import TransMorph_double_decoder
from nnFormer.Swin_Unet_l_gelunorm import swintransformer as nnFormer
import moxing as mox
mox.file.copy_parallel('obs://aaa11177/vit/data/', 'aaa11177/vit/data/')
mox.file.copy_parallel('obs://aaa11177/vit/valdatasets/', 'valdatasets/')
mox.file.copy_parallel('obs://aaa11177/vit/traindatasets/', 'traindatasets/')
mox.file.copy_parallel('obs://aaa11177/vit/testdatasets/', 'testdatasets/')
mox.file.copy_parallel('obs://aaa11177/vit/checkpoint/', 'checkpoint/')###读
mox.file.copy_parallel('obs://aaa11177/vit/result/', 'result/')###读
mox.file.copy_parallel('obs://aaa11177/vit/nii/', 'nii/')###读

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def save_img(I_img,savename):
    I2 = sitk.GetImageFromArray(I_img,isVector=False)
    sitk.WriteImage(I2,savename)
    mox.file.copy_parallel( 'result/', 'obs://aaa11177/vit/result/' )

def save_flow(img,  name):
    ref_img = sitk.ReadImage('nii/aligned_norm-281.nii.gz')
    img = sitk.GetImageFromArray(img[0, 0, ...])
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    #     sitk.WriteImage(img, os.path.join('./experiments/Result/', name))
    sitk.WriteImage(img, name)
    mox.file.copy_parallel( 'result/', 'obs://aaa11177/vit/result/' )

def save_torch(img,  name):
    torch.save(img, name)  # 保存flow
    mox.file.copy_parallel( 'result/', 'obs://aaa11177/vit/result/' )

def save_nii(img, name):
    def_out2 = img.squeeze( 0 )
    def_out2 = def_out2.squeeze( 0 )
    new_data = def_out2
    nii_img = nb.load( 'nii/aligned_norm-281.nii.gz' )
    affine = nii_img.affine.copy( )
    hdr = nii_img.header.copy( )
    new_nii = nb.Nifti1Image( new_data, affine, hdr )
    # 保存nii文件，后面的参数是保存的文件名
    nb.save( new_nii, name )
    #nb.save( new_nii, save_dir + '/' + str( zhi ) + 'test_x.nii' )
    mox.file.copy_parallel( 'result/', 'obs://aaa11177/vit/result/' )

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))

def main():
    test_label_dir = 'testdatasets/'
    model_idx = -1
    model_dir = 'checkpoint/'
    save_dir = 'result/'
    config_vit = CONFIGS_ViT_seg['ViT-V-Net']
    model = nnFormer()
    number_paramerters = sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in G is {}  ".format(sum(x.numel() for x in model.parameters())))
    #model = models.U_Network(img_size=(160, 192, 224))
    config = CONFIGS_TM['TransMorph']
    #model = TransMorph.TransMorph(config)
    #model = pvt.PVTVNet(config_vit, img_size = (160, 192, 224))
    #model = TM_OLEB.TransMorph(config)
    #model = TransMorph_double_decoder.TransMorph(config)
    model_lists = natsorted( glob.glob( model_dir + '*' ) )
    model_lists = model_lists[::-1]
    # print( "model_lists:",model_lists )
    last_model = torch.load(model_lists[0])['state_dict']
    print("last_model file name:", model_lists[0])
    model.load_state_dict(last_model)
    model.cuda()
    reg_model = utils.register_model((160, 192, 224), 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    test_label_composed = transforms.Compose([#trans.Seg_norm(),
                                                    trans.NumpyType((np.float32, np.int16)),
                                                       ])
    test_label_set = datasets.JHUBrainInferDataset(natsorted(glob.glob(test_label_dir + '*.pkl')), transforms=test_label_composed)
    test_label_loader = DataLoader(test_label_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    print("len(test_label_loader):", len(test_label_loader))
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_dsc_IOU = AverageMeter()
    eval_psnr = AverageMeter()
    eval_ssim = AverageMeter()
    eval_njnum = AverageMeter()
    eval_time_diff = AverageMeter()
    '''eval_hd95 = AverageMeter()'''
    zhi = 0

    with torch.no_grad():# 强制之后的内容不进行计算图构建
        #stdy_idx = 0
        dice_lst =[]
        for x, x_seg, y, y_seg in test_label_loader:

            zhi +=1
            model.eval( )
            x = x.cuda( ).float( )
            x_seg = x_seg.cuda( ).float( )
            y = y.cuda( ).float( )
            y_seg = y_seg.cuda( ).float( )
            x = x.squeeze( 5 )
            y = y.squeeze( 5 )
            x_seg = x_seg.squeeze( 5 )
            y_seg = y_seg.squeeze( 5 )
            #print(x.shape, x.dtype)

            x_in = torch.cat( (x, y), dim=1 )
            output = model( x_in )
            x_def, flow = model( x_in )
            flow = flow.permute(0, 2, 3, 4, 1)
            starttime = datetime.datetime.now()
            def_out = reg_model( [x.cuda( ).float( ), output[1].cuda( )] )
            endtime = datetime.datetime.now()
            def_out_seg = reg_model( [x_seg.cuda( ).float( ), output[1].cuda( )] )
            x = x.detach( ).cpu( ).numpy( )
            y = y.detach( ).cpu( ).numpy( )
            flow = flow.detach( ).cpu( ).numpy( )
            def_out = def_out.detach( ).cpu( ).numpy( )
            x_seg = x_seg.detach( ).cpu( ).numpy( )
            y_seg = y_seg.detach( ).cpu( ).numpy( )
            def_out_seg = def_out_seg.detach( ).cpu( ).numpy( )

            #计算各种评价指标
            dsc_trans, dice_lst_1D = compute_label_dice( def_out_seg, y_seg)  # 配准后的dsc
            dice_lst.append(dice_lst_1D)
            dsc_raw,_ = compute_label_dice(x_seg, y_seg)  # 初始图像对dice
            iou = compute_label_IOU(def_out_seg, y_seg)  # 配准后的dsc
            psnr_trans = psnr(def_out, y)
            ssim_trans = ssim(def_out, y)
            NJ_nums = NJ_num(flow)
            time_diff = time_difference(starttime, endtime)
            '''hd95 = compute_label_hd95(def_out_seg, y_seg)'''

            eval_dsc_def.update(dsc_trans.item())
            eval_dsc_raw.update(dsc_raw.item())
            eval_dsc_IOU.update(iou.item())
            eval_psnr.update(psnr_trans.item())
            eval_ssim.update(ssim_trans.item())
            eval_njnum.update(NJ_nums.item())
            eval_time_diff.update(time_diff.item())
            '''eval_hd95.update(hd95.item())'''

            '''
            save_image
            '''
            if zhi == 3:
                #save_flow( flow, save_dir + '/' + str( zhi ) + 'flow.nii' )
                save_torch(flow, save_dir + '/' + str( zhi ) + 'flow.pt')  # 保存flow(B, H, W, D, C)
                save_nii( x, save_dir + '/' + str( zhi ) + 'moving.nii' )
                save_nii( y, save_dir + '/' + str( zhi ) + 'fixed.nii' )
                save_nii( def_out, save_dir + '/' + str( zhi ) + 'warped.nii' )
                save_nii( x_seg, save_dir + '/' + str( zhi ) + 'moving_seg.nii' )
                save_nii( y_seg, save_dir + '/' + str( zhi ) + 'fixed_seg.nii' )
                save_nii( def_out_seg, save_dir + '/' + str( zhi ) + 'warped_seg.nii' )
            print( 'Number: {:}, Trans dice: {:.4f}, Raw dice: {:.4f}'.format(  str( zhi ), dsc_trans.item( ), dsc_raw.item( ) ) )

            #print('Number: {:}, eval_hd95 dice: {:.4f}'.format(str(zhi), hd95.item(), ))
            # stdy_idx += 1
        print('len(dice_lst)', len(dice_lst),'len(dice_lst[0])', len(dice_lst[0]))

        #打印箱线表数据
        for i in range(len(dice_lst[0])):
            c=i
            c=c+1
            print('Label  '+str(c))
            for j in range(len(dice_lst)):
                print(dice_lst[j][i])

        print( 'DSC_def：  ', eval_dsc_def.avg , eval_dsc_def.std)
        print( 'DSC_raw：', eval_dsc_raw.avg , eval_dsc_raw.std)
        print('IOU：', eval_dsc_IOU.avg, eval_dsc_IOU.std)
        print('PSNR.avg：  ', eval_psnr.avg, eval_psnr.std)
        print('SSIM.avg：', eval_ssim.avg, eval_ssim.std)
        print('NJ_num：', eval_njnum.avg, eval_njnum.std)
        print('NJ_%：', (eval_njnum.avg)/160/192/224, (eval_njnum.std)/160/192/224)
        print('Time_diff：', eval_time_diff.avg, eval_time_diff.std)
        number_paramerters
        print('Number_paramerters：', number_paramerters)
        '''print('HD95：', eval_hd95.avg, eval_hd95.std)'''

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
