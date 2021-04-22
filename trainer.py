import os
import math
from decimal import Decimal
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from utility import *

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.epoch = self.model.get_epoch()
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=self.epoch)
        self.error_last = 1e8

    def train(self):
        self.model = self.model.cuda()
        self.loss.step()
        # epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr_ori, hr_ori) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr_ori, hr_ori)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    # test_set14-!!!please  change set14 to benchmarkdenoise
    def test(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()
        psnr_set14 = 0.0
        count_set14 = 0
        for idx_data, d in enumerate(self.loader_test):
            idx_img = 1
            img_name = 0
            for lr, hr in tqdm(d, ncols=80):
                img_name += 1
                torch_test_lr = lr
                lr = lr.permute(0, 2, 3, 1)
                gt = hr.permute(0, 2, 3, 1)
                hr_fortest_np = gt[0]
                input_image = (lr / 255.0)[0]
                stride = 7
                h_idx_list = list(
                    range(0, input_image.shape[0] - self.args.patch_size,
                          stride)) + [input_image.shape[0] - self.args.patch_size]
                w_idx_list = list(
                    range(0, input_image.shape[1] - self.args.patch_size,
                          stride)) + [input_image.shape[1] - self.args.patch_size]
                output_image = np.zeros(input_image.shape)
                overlap = np.zeros(input_image.shape)
                noise_image = input_image
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        input_patch = noise_image[h_idx:h_idx + self.args.patch_size, w_idx:
                                                                                 w_idx + self.args.patch_size]
                        if self.args.n_colors == 1:
                            input_patch = np.expand_dims(input_patch, axis=-1)  # batch=1
                        input_patch = np.expand_dims(input_patch, axis=0)  # channel=1
                        input_patch = torch.tensor(input_patch).permute(0, -1, 1, 2)
                        if self.args.n_GPUs > 0:
                            input_patch = input_patch.cuda()
                        output_patch = self.model(input_patch)
                        output_patch = output_patch.detach().cpu().numpy().transpose(0, 2, 3, 1)
                        if self.args.n_colors == 1:
                            output_patch = output_patch[0, :, :, 0]
                        else:
                            output_patch = output_patch[0, :, :, :]
                        output_image[h_idx:h_idx + self.args.patch_size, w_idx:
                                                                    w_idx + self.args.patch_size] += output_patch
                        overlap[h_idx:h_idx + self.args.patch_size, w_idx:
                                                               w_idx + self.args.patch_size] += 1
                output_image /= overlap
                output_image = np.clip(output_image * 255.0, 0, 255.0)
                psnr_predicted = PSNR(hr_fortest_np.numpy(), output_image)
                denoisedImg_ = output_image[..., ::-1]

                cv2.imwrite(os.path.join('./results/Set14', str(img_name) + '.png'), denoisedImg_)
                psnr_set14 += psnr_predicted
                count_set14 += 1

                ####### write denoised images end #######
                sr = torch.from_numpy(np.clip(output_image[np.newaxis, :, :, :], 0.0, 255.0)).permute(0, 3, 1, 2)
                lr = torch_test_lr
                lr = lr.type(torch.DoubleTensor)
                sr = sr.type(torch.DoubleTensor)
                hr = hr.type(torch.DoubleTensor)
                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                self.ckp.log[-1, idx_data] += utility.calc_psnr(
                    sr, hr, self.scale, self.args.rgb_range
                )
                if self.args.save_gt:
                    save_list.extend([lr, hr])
                if self.args.save_results:
                    self.ckp.save_results(d, idx_img, save_list)
                    idx_img = idx_img + 1

            is_best1 = write_text(path='./results/psnr.txt',
                           all_psnr=psnr_set14,
                           count=count_set14,
                           epoch=self.epoch)

            if is_best1:
                Best2Each(self.args.save, './results/Set14/best_model/')

            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @self.epoch {})'.format(
                    d.dataset.name,
                    self.scale,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, self.epoch, is_best=(best[1][0] + 1 == self.epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return (clip_by_tensor(tensor / 255.0, 0.0, 1.0)).to(device)

        return [_prepare(a) for a in args]

    def terminate(self, first_enter):
        if self.args.test_only:
            self.test()
            return True
        else:
            if not first_enter:
                self.epoch += 1
            return self.epoch >= self.args.epochs


