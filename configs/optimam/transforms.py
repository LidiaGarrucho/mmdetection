from mmdet.datasets import PIPELINES
from configs.optimam.histogram_standardization import apply_hist_stand_landmarks
import torch 
import numpy as np
from skimage import exposure
from PIL import ImageOps, Image
import cv2

@PIPELINES.register_module()
class HistogramStretchingRGB:

    def __init__(self, landmarks_path, to_rgb=False):
        self.landmarks_path = torch.load(landmarks_path)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = np.float32(img) if img.dtype != np.float32 else img.copy()
            nyul_img = apply_hist_stand_landmarks(img, self.landmarks_path)
            nyul_img = nyul_img.astype(np.uint8)
            # Only Contrast stretching
            plow, phigh = np.percentile(nyul_img, (2, 98))
            img_nyul_strecth = exposure.rescale_intensity(nyul_img, in_range=(plow, phigh))
            results[key] = img_nyul_strecth

        results['landmarks_path'] = self.landmarks_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(landmarks_path={self.landmarks_path}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class CombinedRescaleNyulStretchRGB:

    def __init__(self, landmarks_path, to_rgb=False):
        self.landmarks_path = torch.load(landmarks_path)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = np.float32(img) if img.dtype != np.float32 else img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            nyul_img = apply_hist_stand_landmarks(img, self.landmarks_path)
            nyul_img = nyul_img.astype(np.uint8)
            plow, phigh = np.percentile(img, (0.01, 99.9))
            img_rescale = exposure.rescale_intensity(np.array(img), in_range=(plow, phigh))
            plow, phigh = np.percentile(nyul_img, (2, 98))
            img_nyul_strecth = exposure.rescale_intensity(nyul_img, in_range=(plow, phigh))
            rgb = np.dstack((img_rescale.astype(np.uint8), nyul_img, img_nyul_strecth.astype(np.uint8)))
            results[key] = rgb

        results['landmarks_path'] = self.landmarks_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(landmarks_path={self.landmarks_path}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class ImageStandardisationRGB:

    def __init__(self, landmarks_path, to_rgb=False):
        self.landmarks_path = torch.load(landmarks_path)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = np.float32(img) if img.dtype != np.float32 else img.copy()
            #img = ImageOps.grayscale(img)
            nyul_img = apply_hist_stand_landmarks(img, self.landmarks_path)
            nyul_img = nyul_img.astype(np.uint8)
            results[key] = nyul_img #Image.fromarray(nyul_img)

        results['landmarks_path'] = self.landmarks_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(landmarks_path={self.landmarks_path}, to_rgb={self.to_rgb})'
        return repr_str