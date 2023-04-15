import numpy as np
import cv2
import math


class DCP:
    def __init__(self, img):
        self.img = img
        self.smog = None
        self.new_img = np.zeros(img.shape)
        self.w = img.shape[0]
        self.h = img.shape[1]
        self.dark_channel = None
        self.A = 0

    def runAlgorithm(self, kernel_size, w=0.85, t0=0.1):
        """
        :param kernel_size: 以像素点x为中心的局部区域大小
        :param w: 预留雾的程度, 0<w<=1
        :param t0: 透射率下限
        """
        # 取出BGR三通道的最小值，返回二维矩阵
        min_BGR = np.min(self.img, axis=2)

        # 计算暗通道
        dark_channel = np.zeros(min_BGR.shape)
        mid = kernel_size//2
        tmp_img = np.zeros((self.w+kernel_size-1, self.h+kernel_size-1), dtype=np.float)
        tmp_img[:, :] = 255
        tmp_img[mid:self.w+mid, mid:self.h+mid] = min_BGR
        for i in range(mid, self.w+mid-1):
            for j in range(mid, self.h+mid-1):
                dark_channel[i-mid, j-mid] = min(np.min(tmp_img[i-mid:i+mid, j-mid:j++mid]), 255)

        self.dark_channel = dark_channel

        # 获取大气光值A
        pixels = {}
        # 根据像素点位置保存像素点
        for i in range(self.w):
            for j in range(self.h):
                pixels[(i, j)] = dark_channel[i, j]
        pixels_list = list(pixels.items())
        sorted(pixels_list, key=lambda x: x[1], reverse=True)

        # 挑选前0.1%最亮像素
        # 如果前0.1%像素点个数小于1，最选择第一个
        if int(len(pixels_list) * 0.001) == 0:
            loc = pixels_list[0]
            self.A = np.min(self.img[loc[0], loc[1], :])
        else:
            brightest_pixels = pixels_list[: int(len(pixels_list) * 0.001)]
            sum_A = 0
            for pixel in brightest_pixels:
                loc = pixel[0]
                sum_A += np.min(self.img[loc[0], loc[1], :])
            # 取前0.1%像素中最亮像素点的平均值
            self.A = sum_A/len(brightest_pixels)

        # 计算透射率t(x)的预估值
        self.dark_channel = np.float64(self.dark_channel)
        tx = 1 - w * self.dark_channel / self.A
        new_tx = np.clip(tx, t0, 1)

        # 滤波
        ksize = 20*kernel_size
        eps = 1e-5
        mean_i = cv2.blur(self.dark_channel, (ksize, ksize))
        mean_p = cv2.blur(new_tx, (ksize, ksize))
        corr_i = cv2.blur(np.multiply(new_tx, new_tx), (ksize, ksize))
        corr_ip = cv2.blur(np.multiply(new_tx, self.dark_channel), (ksize, ksize))
        var_i = corr_i - np.multiply(mean_i, mean_i)
        cov_ip = corr_ip - np.multiply(mean_i, mean_p)
        a = cov_ip / (var_i + eps)
        b = mean_p - np.multiply(a, mean_i)
        mean_a = cv2.blur(a, (ksize, ksize))
        mean_b = cv2.blur(b, (ksize, ksize))
        new_tx = np.multiply(mean_a, self.dark_channel) + mean_b

        # 计算无雾图
        tmp_img = np.float64(self.img)
        for i in range(3):
            self.new_img[:, :, i] = (tmp_img[:, :, i] - self.A) / new_tx + self.A
        self.new_img = np.clip(self.new_img, 0, 255)

    def getDarkChannel(self):
        return self.dark_channel

    def getRes(self):
        return np.uint8(self.new_img)


class Retinex:
    def __init__(self, img, sigma_list=[15, 80, 250]):
        self.img = np.float64(img)
        self.sigma_list = sigma_list
        self.RImg = None

    def runAlgorithm(self):
        img_msr = self.MSR()
        img_msr = self.clipImage(img_msr, 0.05, 0.95)
        for i in range(self.img.shape[2]):
            img_msr[:, :, i] = (img_msr[:, :, i] - np.min(img_msr[:, :, i])) / (
                        np.max(img_msr[:, :, i]) - np.min(img_msr[:, :, i])) * 255

        img_msr = np.clip(img_msr, 0, 255)
        self.RImg = np.uint8(img_msr)

    # 使用像素出现频率确定上下剪切点
    def clipImage(self, img, low_clip, high_clip):
        low_val = high_val = 0
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img[:, :, i] = np.clip(img[:, :, i], low_val, high_val)

        return img

    def SSR(self, sigma):
        LImg = cv2.GaussianBlur(self.img, (0, 0), sigma)
        ans = np.log(self.img + 1) - np.log(LImg + 1)
        return ans

    def MSR(self):
        tmp = np.zeros_like(self.img)
        for sigma in self.sigma_list:
            tmp += self.SSR(sigma)
        tmp /= 3
        return tmp

    def getRimg(self):
        return self.RImg
