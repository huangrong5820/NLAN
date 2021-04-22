import matplotlib
matplotlib.use('Agg')
import os
import shutil
import math
import time
import datetime
import cv2
from multiprocessing import Process
import matplotlib
matplotlib.use('Agg')
from multiprocessing import Queue
from time import strftime, localtime
import matplotlib
matplotlib.use('Agg')
import math as mt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def CutImg_Mirrorpad(imgs, patch_h, patch_w):
    [h, w, c] = imgs[0].shape
    if patch_h < 192 | patch_w < 192:
        return -1
    # horizontal水平的，vertical垂直的
    vert_num = mt.ceil(h / patch_h)
    hori_num = mt.ceil(w / patch_w)
    # print('vert_num行, hori_num列: ', hori_num, vert_num)

    full_width = hori_num * patch_w
    full_height = vert_num * patch_h
    '''
    镜像填充
    '''
    From_right = False
    From_down = False
    if (full_width - w) > 0:
        From_right = True
    if (full_height - h) > 0:
        From_down = True
    if From_right & From_down:
        fill = from_right(imgs, full_height, full_width)
        fill = from_down(fill, full_height, full_width)
    elif From_right:
        fill = from_right(imgs, full_height, full_width)
    elif From_down:
        fill = from_down2(imgs, full_height, full_width)
    else:
        # print('长宽都适合，不做处理')
        fill = imgs




    # fill = [fill]
    # print('length of fill:', len(fill))

    cropImgs = []
    for i in range(vert_num):
        cropImgs_hori=[]
        if i < vert_num:
            for j in range(hori_num):
                if j < hori_num:
                    for img in fill:
                        cropImg = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :]
                        cropImgs_hori.append(cropImg)
                else:
                    for img in fill:
                        cropImg = img[i*patch_h:(i+1)*patch_h, j*patch_w:full_width, :]
                        cropImgs_hori.append(cropImg)
        else:
            for j in range(hori_num):
                if j < hori_num:
                    for img in fill:
                        cropImg = img[i * patch_h:full_height, j * patch_w:(j + 1) * patch_w, :]
                        cropImgs_hori.append(cropImg)
                else:
                    for img in fill:
                        cropImg = img[i * patch_h:full_height, j * patch_w:full_width, :]
                        cropImgs_hori.append(cropImg)
        cropImgs.append(cropImgs_hori)

    return cropImgs, vert_num, hori_num, h, w

def from_right(imgs, full_height, full_width):
    [h, w, c] = imgs[0].shape
    fills = []
    for c_ in range(c):
        for img in imgs:
            fill = []
            singleImg = img[0:h, 0:w, c_:c_+1]
            fill.append(singleImg)
            # 形状：[1, 480, 500, 1]
            pad_width = full_width - w
            padImg = []
            padImg1 = img[0:h, w - (pad_width):w, c_:c_ + 1]
            padImg.append(padImg1)
            a = [t.numpy() for t in fill]
            b = [t.numpy() for t in padImg]
            fill = np.concatenate([a, b], axis=2)
            # fill形状：[1, 480, 528, 1]
        fills.append(fill)
    if len(fills) == 3:
        fill = np.concatenate([fills[0], fills[1]], axis=-1)
        fill = np.concatenate([fill, fills[2]], axis=-1)
    if len(fills) == 1:
        fill = fills[0]
    return fill
def from_down(imgs, full_height, full_width):
    [h, w, c] = imgs[0].shape
    fills = []
    for c_ in range(c):
        for img in imgs:
            fill = []
            singleImg = img[0:h, 0:w, c_:c_ + 1]
            fill.append(singleImg)
            # 形状：[1, 480, 500, 1]
            pad_height = full_height - h
            padImg = []
            padImg1 = img[h-pad_height:h, 0:full_width, c_:c_ + 1]
            padImg.append(padImg1)
            # c = [t.numpy() for t in fill]
            # d = [t.numpy() for t in padImg]
            fill = np.concatenate([fill, padImg], axis=1)
            # fill形状：[1, 480, 528, 1]
        fills.append(fill)

    if len(fills) == 3:
        fill = np.concatenate([fills[0], fills[1]], axis=-1)
        fill = np.concatenate([fill, fills[2]], axis=-1)
    if len(fills) == 1:
        fill = fills[0]
    return fill

def from_down2(imgs, full_height, full_width):
    [h, w, c] = imgs[0].shape
    fills = []
    for c_ in range(c):
        for img in imgs:
            fill = []
            singleImg = img[0:h, 0:w, c_:c_ + 1]
            fill.append(singleImg)
            # 形状：[1, 480, 500, 1]
            pad_height = full_height - h
            padImg = []
            padImg1 = img[h-pad_height:h, 0:full_width, c_:c_ + 1]
            padImg.append(padImg1)
            c = [t.numpy() for t in fill]
            d = [t.numpy() for t in padImg]
            fill = np.concatenate([c, d], axis=1)
            # fill形状：[1, 480, 528, 1]
        fills.append(fill)

    if len(fills) == 3:
        fill = np.concatenate([fills[0], fills[1]], axis=-1)
        fill = np.concatenate([fill, fills[2]], axis=-1)
    if len(fills) == 1:
        fill = fills[0]
    return fill


def CompositeImg(result_list, rows, columns, oringinal_h, oringinal_w):
    for i in range(rows):
        fill_hori = result_list[i][0]
        for j in range(columns):
            if j > 0:
                fill_hori = np.concatenate([fill_hori, result_list[i][j]], axis=1)
        if i == 0:
            fill = fill_hori
        else:
            fill = np.concatenate([fill, fill_hori], axis=0)
    fill = [fill]
    Imgs = []
    for img in fill:
        cropImg = img[0:oringinal_h, 0:oringinal_w, :]
        Imgs.append(cropImg)
    return Imgs


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.clock()

    def toc(self):
        return time.clock() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(args.save)
        else:
            self.dir = os.path.join(args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        #for d in args.data_test:
        #    os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        os.makedirs(self.get_path('results-{}'.format(args.data_test)), exist_ok=True)
        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        plt.plot(
                    axis,
                    self.log[:].numpy()
                )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(self.get_path('test_{}.pdf'.format(self.args.data_test)))
        plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_'.format(filename)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def _PSNR(pred, gt):
    mat = pred
    _max = np.max(mat, axis=0)
    max = np.max(_max, axis=0)
    _min = np.min(mat, axis=0)
    min = np.min(_min, axis=0)
    imdff = pred - gt
    rmse = mt.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10((max - min) / rmse)

def PSNR(pred, gt):
    psnr1 = _PSNR(pred[:, :, 0], gt[:, :, 0])
    psnr2 = _PSNR(pred[:, :, 1], gt[:, :, 1])
    psnr3 = _PSNR(pred[:, :, 2], gt[:, :, 2])
    return (psnr1 + psnr2 + psnr3) / 3

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    '''
        Here we assume quantized(0-255) arguments.
        For Set5, Set14, B100, Urban100 dataset,
        we measure PSNR on luminance channel only
    '''
    diff = (sr - hr) / rgb_range
    shave = 0
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def SSIM(img1, img2):
    mssim, s = ssim(img1, img2, data_range=img1.max() - img1.min(), full=True, multichannel=True)
    return mssim


def changetext(txt_path, a, b):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = []                    # 创建了一个空列表，里面没有元素
        for line in f.readlines():
            if line != '\n':
                lines.append(line)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if a in line:
                line = b
                f.write('%s\n' %line)
            else:
                f.write('%s' %line)

def write_text(path, all_psnr, count, epoch):
    is_best = False
    with open(path, 'r+') as f:
        first_line = f.readline()
        change = all_psnr / count
        content = f.read()
        if first_line == '':
            f.write('{}\tBest psnr is:{}\n'.format(getTime(), 0.0))
            first_line = 'Best psnr is:0.0'

        best = float(first_line.split(':')[-1].split('\'')[0])
        if change > best:
            is_best = True
            content_head = '{}\t--At epoch--{}--Best psnr is:{}\n'.format(getTime(), epoch, change)
            f.seek(0, 0)
            f.write(content_head + content + '\n')
    with open(path, 'r+') as f:
        f.write('Epoch:{} \tpsnr is\t:'.format(epoch, all_psnr / count))
    return is_best

def Best2Each(filepath, newPath):
    # 获取当前路径下的文件名，返回List
    fileNames = os.listdir(filepath)
    for file in fileNames:
        # 将文件命加入到当前文件路径后面
        newDir = filepath + '/' + file
        # 如果是文件
        if os.path.isfile(newDir):
            print(newDir)
            newFile = newPath + file
            shutil.copyfile(newDir, newFile)
        # 如果不是文件，递归这个文件夹的路径
        else:
            Best2Each(newDir, newPath)

def getTime():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def countParams(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params / 1e6}')
    print(f'Trainable params: {Trainable_params / 1e6}')
    print(f'Non-trainable params: {NonTrainable_params / 1e6}')

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def print_model(*args):
    for item in args:
        print_network(item)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.lr_decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)

    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features


def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature


if __name__ == '__main__':
    pass