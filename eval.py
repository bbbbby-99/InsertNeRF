import argparse
import os
import lpips
import cv2
import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np

from utils.base_utils import color_map_forward
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
class Evaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()

    def eval_metrics_img(self,gt_img, pr_img):
        gt_img = color_map_forward(gt_img)
        pr_img = color_map_forward(pr_img)
        psnr = tf.image.psnr(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        ssim = tf.image.ssim(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            score = float(self.loss_fn_alex(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        return float(psnr), float(ssim), score


    def eval(self, dir_gt, dir_pr, dir_mask =None):
        results=[]
        num = len(os.listdir(dir_gt))
        for k in tqdm(range(0, num)):
            pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
            gt_img = imread(f'{dir_gt}/{k}.jpg')
            mask = np.sum(imread(os.path.join(f'{dir_mask}/{k}.png')),-1)>0
            mask = cv2.resize(mask.astype(np.uint8), (pr_img.shape[1], pr_img.shape[0]), interpolation=cv2.INTER_NEAREST) == 1
            psnr, ssim, lpips_score = self.eval_metrics_img(gt_img * mask.astype(np.uint8)[:,:,None], pr_img* mask.astype(np.uint8)[:,:,None])
            results.append([psnr,ssim,lpips_score])
        psnr, ssim, lpips_score = np.mean(np.asarray(results),0)

        msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)
        return psnr, ssim, lpips_score
    #def eval(self, dir_gt, dir_pr):
    #    results=[]
    #    num = len(os.listdir(dir_gt))
    #    for k in tqdm(range(0, num)):
    #        pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
    #        gt_img = imread(f'{dir_gt}/{k}.jpg')
    #        psnr, ssim, lpips_score = self.eval_metrics_img(gt_img , pr_img)
    #        results.append([psnr,ssim,lpips_score])
    #    psnr, ssim, lpips_score = np.mean(np.asarray(results),0)

     #   msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
     #   print(msg)
      #  return psnr, ssim, lpips_score
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='llff')
    parser.add_argument('--its', type=int, default=470000)
    parser.add_argument('--dir_gt', type=str, default='data/render/fern/gt')
    parser.add_argument('--dir_pr', type=str, default='data/render/fern/insert_gen_depth-pretrain-eval')
    flags = parser.parse_args()
    evaluator = Evaluator()
    psnr_ave, ssim_ave, lpips_score_ave = 0 ,0, 0
    if flags.dataset == 'llff':
        for i in ['trex', 'fern', 'flower', 'leaves', 'room', 'fortress', 'horns', 'orchids']:
            database_name = i
            flags.dir_gt ='data/render/llff_colmap/' + i + '/high/gt/'
            flags.dir_pr ='data/render/llff_colmap/' + i + '/high/insert_gen_depth_train-'+ str(flags.its) + '-eval/'
            psnr, ssim, lpips_score = evaluator.eval(flags.dir_gt, flags.dir_pr)
            psnr_ave += psnr
            ssim_ave += ssim
            lpips_score_ave += lpips_score
        msg = f'psnr_ave {(psnr_ave/8):.4f} ssim_ave {(ssim_ave/8):.4f} lpips_score_ave {(lpips_score_ave/8):.4f}'
        print(msg)
    elif flags.dataset == 'blender':
        for i in ['chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship', 'lego']:
            database_name = i
            flags.dir_gt ='data/render/nerf_synthetic/' + i + '/black_400/gt/'
            flags.dir_pr ='data/render/nerf_synthetic/' + i + '/black_400/insert_gen_depth_train-'+ str(flags.its) + '-eval/'
            psnr, ssim, lpips_score = evaluator.eval(flags.dir_gt, flags.dir_pr)
            psnr_ave += psnr
            ssim_ave += ssim
            lpips_score_ave += lpips_score
        msg = f'psnr_ave {(psnr_ave / 8):.4f} ssim_ave {(ssim_ave / 8):.4f} lpips_score_ave {(lpips_score_ave / 8):.4f}'
        print(msg)
    elif flags.dataset == 'DTU':
        for i in ['birds', 'bricks', 'snowman', 'tools']:
            database_name = i
            flags.dir_gt ='data/render/dtu_test/' + i + '/black_400/gt/'
            flags.dir_mask = 'data/render/dtu_test/' + i + '/black_800/mask/'
            flags.dir_pr ='data/render/dtu_test/' + i + '/black_400/insert_gen_depth_train-'+ str(flags.its) + '-eval/'
            psnr, ssim, lpips_score = evaluator.eval(flags.dir_gt, flags.dir_pr,flags.dir_mask)
            psnr_ave += psnr
            ssim_ave += ssim
            lpips_score_ave += lpips_score
        msg = f'psnr_ave {(psnr_ave / 4):.4f} ssim_ave {(ssim_ave / 4):.4f} lpips_score_ave {(lpips_score_ave / 4):.4f}'
        print(msg)
