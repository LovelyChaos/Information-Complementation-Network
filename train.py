import os, utils, glob, losses
os.system('pip install SimpleITK')
os.system('pip install -U --pre statsmodels')
os.system('pip install ml_collections')
os.system('pip install natsort')
os.system('pip install nibabel')
os.system('pip install timm')
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, models, pvt
from torchvision import transforms
from torch import optim
import torch.nn as nn
from natsort import natsorted  # 文件名称自然排序
import datetime
from eval import  compute_label_dice
from TransMorph import CONFIGS as CONFIGS_TM
import TransMorph
import TransMorph_double_decoder, TM_OLEB
''''''
import moxing as mox

mox.file.copy_parallel('obs://aaa11177/vit/data/', 'aaa11177/vit/data/')
mox.file.copy_parallel('obs://aaa11177/vit/valdatasets/', 'valdatasets/')
mox.file.copy_parallel('obs://aaa11177/vit/traindatasets/', 'traindatasets/')
mox.file.copy_parallel('obs://aaa11177/vit/traindatasets1/', 'traindatasets1/')
mox.file.copy_parallel('obs://aaa11177/vit/checkpoint/', 'checkpoint/')  ###读
mox.file.copy_parallel('obs://aaa11177/vit/Log/', 'Log/')  ###读
mox.file.copy_parallel('obs://aaa11177/vit/logtxt/', 'logtxt/')  ###读
mox.file.copy_parallel('obs://aaa11177/vit/result/', 'result/')  ###读

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)


def main():
    start_epoch = 4 # 如果checkpoint里是1，这里就写2
    batch_size = 2
    train_dir = 'traindatasets/'
    val_dir = 'valdatasets/'
    save_dir = 'result/'
    log_file = 'logtxt/'
    log_file_name = 'TMdemo.txt'
    lr = 0.0001
    max_epoch = 1000
    reg_model = utils.register_model((160, 192, 224), 'nearest')
    reg_model.cuda()
    '''
        Initialize model
        '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    #model = TransMorph_double_decoder.TransMorph(config)
    #model = PVT2_transmorph.PVTVNetSkip()
    #model = TM_OLEB.TransMorph(config)
    model.cuda()
    exists_txt = os.path.exists(log_file + log_file_name)
    if exists_txt == True:
        data = np.loadtxt(log_file + log_file_name, dtype = np.float32)
        start_epoch = len(data)+1
        print('start_epoch', start_epoch)
        epoch_start = start_epoch
        model_dir = 'checkpoint/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        model_lists = natsorted(glob.glob(model_dir + '*'))
        model_lists = model_lists[::-1]
        # print( "model_lists:",model_lists )
        last_model = torch.load(model_lists[0])['state_dict']
        print("last_model file name:", model_lists[0])
        model.load_state_dict(last_model)
    else:
        updated_lr = lr
        epoch_start = 1

    # 日志文件
    dt = datetime.datetime.now(tz = datetime.timezone(datetime.timedelta(hours = 8)))
    log_dir = 'Log/'
    log_name = str(epoch_start) + "-" + str(updated_lr) + "-" + dt.strftime('%b%d-%H%M')
    print("log_name: ", log_name)

    model.cuda()
    # '''原始dataloader
    train_composed = transforms.Compose([  # trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([  # trans.Seg_norm(), #rearrange segmentation label to 1 to 35
        trans.NumpyType((np.float32, np.int16)),
    ])

    train_set = datasets.JHUBrainDataset(natsorted(glob.glob(train_dir + '*.pkl')), transforms = train_composed)
    val_set = datasets.JHUBrainInferDataset(natsorted(glob.glob(val_dir + '*.pkl')), transforms = val_composed)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = False)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = False,
                            drop_last = True)
    print("len(train_loader):", len(train_loader))
    print("len(val_loader):", len(val_loader))

    optimizer = optim.Adam(model.parameters(), lr = updated_lr, weight_decay = 0, amsgrad = True)
    criterion = nn.MSELoss()
    # criterion = losses.NCC()
    criterions = [criterion]
    # prepare deformation loss
    criterions += [losses.Grad3d(
        penalty = 'l2')]
    weights = [1]
    weights += [0.02]
    best_mse = 0
    for epoch in range(epoch_start, max_epoch):
        f = open(os.path.join(log_dir, log_name + ".txt"), "a+")
        mox.file.copy_parallel('Log/', 'obs://aaa11177/vit/Log/')
        f1 = open(os.path.join(log_file, log_file_name), "a+")
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/vit/logtxt/')
        print('*****Training Starts*****')
        print('*****Training Starts*****', file = f)
        print("Epoch:  ", epoch)
        print("Epoch:  ", epoch, file = f)
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for x, y in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # data = [t.cuda() for t in data]
            x = x.cuda().float()
            y = y.cuda().float()
            x = x.squeeze(5)
            y = y.squeeze(5)
            # x = x.permute(0, 1, 4, 3, 2)
            # y = y.permute(0, 1, 4, 3, 2)
            # print( "x.shape", x.shape )
            # print( "y.shape", y.shape )
            x_in = torch.cat((x, y), dim = 1)
            # print("x_in.shape", x_in.shape)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim = 1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Iter {} of {} loss {:.6f}, Img Sim: {:.6f}, Reg: {:.9f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_vals[0].item() / 2,
                                                                                                loss_vals[1].item() / 2,))
            print('Iter {} of {} loss {:.6f}, Img Sim: {:.9f}, Reg: {:.9f}'.format(idx, len(train_loader),
                                                                                   loss.item(),
                                                                                   loss_vals[0].item() / 2,
                                                                                   loss_vals[1].item() / 2,), file = f)
            mox.file.copy_parallel('Log/', 'obs://aaa11177/vit/Log/')
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file = f)
        mox.file.copy_parallel('Log/', 'obs://aaa11177/vit/Log/')
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        i = 0
        with torch.no_grad():
            for x, x_seg, y, y_seg in val_loader:
                i = i + 1
                model.eval()
                x = x.cuda().float()
                x_seg = x_seg.cuda().float()
                y = y.cuda().float()
                y_seg = y_seg.cuda().float()
                x = x.squeeze(5)
                y = y.squeeze(5)
                x_seg = x_seg.squeeze(5)
                y_seg = y_seg.squeeze(5)

                x_in = torch.cat((x, y), dim = 1)
                output = model(x_in)
                # print( "print(x_seg.shape)", x_seg.shape )
                # print( "print(output[1].shape)", output[1].shape )
                def_out_seg = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc,_ = compute_label_dice(def_out_seg.cpu().numpy(), y_seg.cpu().numpy())
                print(i, ":  dsc:", dsc)
                print("dsc:", dsc, file = f)
                mox.file.copy_parallel('Log/', 'obs://aaa11177/vit/Log/')
                eval_dsc.update(dsc.item())
        print("eval_dsc.avg: ", eval_dsc.avg, file = f)
        mox.file.copy_parallel('Log/', 'obs://aaa11177/vit/Log/')
        print("eval_dsc.avg: ", eval_dsc.avg)
        best_mse = max(eval_dsc.avg, best_mse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
        }, save_dir = 'checkpoint/', filename = 'epoch{}dsc{:.6f}.pth.tar'.format(str(epoch), eval_dsc.avg))
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/vit/checkpoint/')
        f1.write(str(loss_all.avg) + " " + str(eval_dsc.avg))
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/vit/logtxt/')
        f1.write("\r")  # 换行
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/vit/logtxt/')
        f1.close()
        f.close()
        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='modes', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/vit/checkpoint/')
    '''删除模型，华为云上没用
    model_lists = natsorted(glob.glob(save_dir + '*'))
    mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/vit/checkpoint/')
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/vit/checkpoint/')
        model_lists = natsorted(glob.glob(save_dir + '*'))
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/vit/checkpoint/')'''


if __name__ == '__main__':
    '''GPU configuration'''
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