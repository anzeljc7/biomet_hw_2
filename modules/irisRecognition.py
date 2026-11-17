import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import *

class irisRecognition(object):
    def __init__(self, cfg):
        # cParsing the config file
        self.cuda = cfg["cuda"]
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.filter_sizes = [int(filter_size) for filter_size in cfg["recog_filter_size"].split(',')]
        self.num_filters_per_size = [int(num_filter) for num_filter in cfg["recog_num_filters"].split(',')]
        self.total_num_filters = sum(self.num_filters_per_size)
        self.torch_filters = self.load_filters(cfg["recog_bsif_dir"], self.filter_sizes, self.num_filters_per_size)
        self.max_shift = cfg["recog_max_shift"]
        self.score_norm = cfg["score_norm"]
        self.threshold_frac_avg_bits = cfg["threshold_frac_avg_bits"]
        
        self.mask_model_path = cfg["mask_model_path"]
        self.circle_model_path = cfg["circle_model_path"]
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5))
        
        
        self.NET_INPUT_SIZE = (320,240)

        with torch.inference_mode():
            self.circle_model = models.resnet18()
            self.circle_model.avgpool = conv(in_channels=512, out_n=6)
            self.circle_model.fc = fclayer(out_n=6)
            try:
                self.circle_model.load_state_dict(torch.load(self.circle_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.circle_model.load_state_dict(torch.load(self.circle_model_path,
                        map_location = lambda storage, loc: storage))
            self.circle_model = self.circle_model.to(self.device)
            self.circle_model.eval()
            self.mask_model = NestedSharedAtrousResUNet(1, 1, width=32, resolution=(240,320))
            try:
                self.mask_model.load_state_dict(torch.load(self.mask_model_path, map_location=self.device))
            except AssertionError:
                    print("assertion error")
                    self.mask_model.load_state_dict(torch.load(self.mask_model_path,
                        map_location = lambda storage, loc: storage))
            self.mask_model = self.mask_model.to(self.device)
            self.mask_model.eval()
            self.input_transform_mask = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
            self.input_transform_circ = Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])          
        
        self.avg_bits_by_filter_size = {5: 25056, 7: 24463, 9: 23764, 11: 23010, 13: 22225, 15: 21420, 17: 20603, 19: 19777, 21: 18945, 27: 16419, 33: 13864, 39: 11289}
        self.avg_num_bits = 0
        for filter_size in self.filter_sizes:
            self.avg_num_bits += self.avg_bits_by_filter_size[filter_size]
        self.avg_num_bits /= len(self.filter_sizes)
        self.fixed_num_bits = 26770 # based on masks statistics from 'VII-Q', 'ND3DIris', 'PostMortem-Iris-NIJ', and 'Q-FIRE'
        self.ISO_RES = (640,480)

    @torch.inference_mode()
    def load_filters(self, recog_bsif_dir, filter_sizes, num_filters_per_size):
        torch_filters = []
        for filter_size, num_filters in zip(filter_sizes, num_filters_per_size):
            mat_file_path = recog_bsif_dir+'ICAtextureFilters_{0}x{1}_{2}bit.pt'.format(filter_size, filter_size, num_filters)
            filter_mat = torch.jit.load(mat_file_path, torch.device('cpu')).ICAtextureFilters.detach().numpy()
            torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            torch_filter = torch.moveaxis(torch_filter.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            torch_filters.append(torch_filter.clone().detach())
        return torch_filters

    # converts non-ISO images into ISO dimensions
    def fix_image(self, image):
        w, h = image.size
        aspect_ratio = float(w)/float(h)
        if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
            result_im = image.copy().resize(self.ISO_RES)
        elif aspect_ratio < 1.333:
            w_new = h * (4.0/3.0)
            w_pad = (w_new - w) / 2
            result_im = Image.new(image.mode, (int(w_new), h), 127)
            result_im.paste(image, (int(w_pad), 0))
            result_im = result_im.resize(self.ISO_RES)
        else:
            h_new = w * (3.0/4.0)
            h_pad = (h_new - h) / 2
            result_im = Image.new(image.mode, (w, int(h_new)), 127)
            result_im.paste(image, (0, int(h_pad)))
            result_im = result_im.resize(self.ISO_RES)
        return result_im
    
    def segment_and_circApprox(self, image):
        pred = self.segment(image)
        pupil_xyr, iris_xyr = self.circApprox(image)
        return pred, pupil_xyr, iris_xyr

    @torch.inference_mode()
    def segment(self,image):

        w,h = image.size
        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
        mask = mask_t.cpu().numpy()[0]
        mask = cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        #print('Mask Shape: ', mask.shape)

        return mask

    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):
        
        pupil_xyr = np.around(pupil_xyr).astype(np.int32)
        iris_xyr = np.around(iris_xyr).astype(np.int32)
        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + (96/255)*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis

    @torch.inference_mode()
    def circApprox(self,image):

        w,h = image.size

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

        #Circle params
        diag = math.sqrt(w**2 + h**2)
        inp_xyr = inp_xyr_t.tolist()[0]
        pupil_x = (inp_xyr[0] * w)
        pupil_y = (inp_xyr[1] * h)
        pupil_r = (inp_xyr[2] * 0.5 * 0.8 * diag)
        iris_x = (inp_xyr[3] * w)
        iris_y = (inp_xyr[4] * h)
        iris_r = (inp_xyr[5] * 0.5 * diag)

        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    # New Rubbersheet model-based Cartesian-to-polar transformation. It has support for bilinear interpolation
    @torch.inference_mode()
    def grid_sample(self, input, grid, interp_mode):  #helper function for new interpolation
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True, padding_mode='border')

    @torch.inference_mode()
    def cartToPol_torch(self, image, mask, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
        if pupil_xyr is None or iris_xyr is None:
            return None, None
        
        image = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        width = image.shape[3]
        height = image.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width
        pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float().to(self.device)
        iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float().to(self.device)
        
        theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).to(self.device)
        pxCirclePoints = (pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
        pyCirclePoints = (pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        
        ixCirclePoints = (iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
        iyCirclePoints = (iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512

        radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).to(self.device)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = (pxCoords + ixCoords).float()
        x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

        y = (pyCoords + iyCoords).float()
        y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).to(self.device)

        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode=interpolation)
        image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
        mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest') # always use nearest neighbor interpolation for mask
        mask_polar = (mask_polar>0.5).long() * 255

        return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)

    # (Fixed) Old implementation of Rubbersheet model-based Cartesian-to-polar transformation that uses nearest neighbor interpolation
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(0,self.polar_width-1,self.polar_width)/self.polar_width
       
        pxCirclePoints = np.around(pupil_xyr[0] + pupil_xyr[2]*np.cos(theta))    
        ixCirclePoints = np.around(iris_xyr[0] + iris_xyr[2]*np.cos(theta))
        pyCirclePoints = np.around(pupil_xyr[1] + pupil_xyr[2]*np.sin(theta))
        iyCirclePoints = np.around(iris_xyr[1] + iris_xyr[2]*np.sin(theta))

        for j in range(1, self.polar_width+1):            
            for i in range(1, self.polar_height+1):

                radius = i/self.polar_height
                x = int(np.around((1-radius) * pxCirclePoints[j-1] + radius * ixCirclePoints[j-1]))
                y = int(np.around((1-radius) * pyCirclePoints[j-1] + radius * iyCirclePoints[j-1]))
                if (x > 0 and x <= width and y > 0 and y <= height): 
                    image_polar[i-1][j-1] = image[y-1][x-1]
                    mask_polar[i-1][j-1] = mask[y-1][x-1]

        return image_polar, mask_polar

    @torch.inference_mode()
    def extractCode(self, polar):
        if polar is None:
            return None
        codeBinaries = []
        for filter_size, torch_filter in zip(self.filter_sizes, self.torch_filters):
            r = int(np.floor(filter_size / 2))
            polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
            padded_polar = nn.functional.pad(polar_t, (r, r, 0, 0), mode='circular')
            codeContinuous = nn.functional.conv2d(padded_polar, torch_filter)
            codeBinary = torch.where(codeContinuous.squeeze(0) > 0, True, False)
            codeBinaries.append(codeBinary.cpu().numpy())
        return codeBinaries
    
    # ======================================================================
    # IBB: LBP-based iris code
    # ======================================================================
    # All parameters live here (not in cfg.yaml) because only this file is submitted.
    # Values were selected by measuring EER and d' on 348 images / 80 identities
    # (all 60 378 pairs); see the list of improvements above extractIBBCode.

    # --- LBP operator ---
    IBB_P = 8               # number of neighbours sampled on the ellipse
    IBB_RX = 12.0           # ellipse semi-axis along the angular direction (columns) [px]
    IBB_RY = 1.0            # ellipse semi-axis along the radial direction (rows) [px]
    IBB_SIGMA = 2.0         # Gaussian smoothing applied before LBP [px]
    # --- template layout ---
    IBB_ROW_START = 1       # first row kept in the code
    IBB_ROW_STEP = 2        # keep every 2nd row (after smoothing, adjacent rows are nearly identical)
    IBB_WEIGHT_T = 0.5      # soft bit weight = min(1, |difference| / (T * median |difference|))
    # --- matching ---
    IBB_MAX_SHIFT = 16      # rotation compensation: +-16 columns = +-11.25 degrees (16 / 512 * 360)
    IBB_MIN_OVERLAP = 0.05  # a shift is considered only if >= 5 % of the bits are jointly valid
    IBB_NO_MATCH = 1.0      # distance returned when two codes cannot be compared
    # --- occlusion mask estimated from the image (used only when no mask is passed) ---
    IBB_Z_MAX = 2.0         # a pixel is iris if |intensity - median| < Z_MAX * MAD
    IBB_ERODE = 2           # erosion of the estimated mask [px]
    # Typical eyelid position in the polar image as (centre, half-width, height), given as
    # fractions of the image width / height. Lower eyelid sits at ~90 deg (0.26 * 360),
    # upper eyelid at ~270 deg (0.74 * 360); both grow from the outer iris boundary inwards.
    IBB_EYELIDS = ((0.258, 0.172, 0.470), (0.742, 0.250, 0.875))

    def _ibbBlur(self, img):
        """Gaussian smoothing of the polar image.

        The left and right image borders are neighbours on the iris (angle 0 = angle 360),
        so the image is padded circularly along the columns. Along the rows (radius) there is
        nothing beyond the border, so the edge value is repeated.
        """
        H, W = img.shape
        r = int(np.ceil(3 * self.IBB_SIGMA))                 # kernel radius: +-3 sigma covers 99.7 %
        padded = cv2.copyMakeBorder(img, 0, 0, r, r, cv2.BORDER_WRAP)
        padded = cv2.copyMakeBorder(padded, r, r, 0, 0, cv2.BORDER_REPLICATE)
        blurred = cv2.GaussianBlur(padded, (2 * r + 1, 2 * r + 1), self.IBB_SIGMA)
        return blurred[r:r + H, r:r + W]                     # [r:-r] would be empty for sigma = 0 (r = 0)

    def _ibbSampleNeighbour(self, img, dy, dx):
        """Image value at (row + dy, column + dx) for every pixel, using bilinear interpolation.

        The offset is generally not an integer, so the value is a blend of the 4 surrounding
        pixels weighted by proximity. Columns wrap around (np.roll); rows are clamped to the
        image. Also returns a boolean vector marking the rows whose neighbour is really inside
        the image (rows near the pupil / outer boundary have no neighbour above / below).
        """
        H = img.shape[0]
        rows = np.arange(H)
        y0, x0 = int(np.floor(dy)), int(np.floor(dx))
        fy, fx = dy - y0, dx - x0                            # fractional parts = interpolation weights
        value = np.zeros_like(img)
        for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            if wy == 0:
                continue
            shifted_rows = img[np.clip(rows + yy, 0, H - 1), :]
            for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
                if wx == 0:
                    continue
                value += wy * wx * np.roll(shifted_rows, -xx, axis=1)   # roll(-xx): pixel x + xx moves to x
        inside = (rows + np.floor(dy) >= 0) & (rows + np.ceil(dy) <= H - 1)
        return value, inside

    def _ibbLbpDifferences(self, img):
        """LBP core: neighbour - centre for each of the P neighbours on the ellipse.

        Returns diffs [P, H, W] (the LBP bit is its sign) and inside [P, H] (rows where
        neighbour p lies inside the image). Neighbour p sits at angle 2*pi*p/P.
        """
        H, W = img.shape
        diffs = np.empty((self.IBB_P, H, W), dtype=np.float32)
        inside = np.empty((self.IBB_P, H), dtype=bool)
        for p in range(self.IBB_P):
            angle = 2 * pi * p / self.IBB_P
            dy, dx = -self.IBB_RY * np.sin(angle), self.IBB_RX * np.cos(angle)
            # sin(pi) and cos(pi / 2) evaluate to ~1e-16 instead of 0. Snap such offsets to whole
            # pixels: otherwise the neighbour at 180 deg (same row) counts as lying a fraction of
            # a row higher, so row 0 is flagged as having no neighbour and an extra blend is done.
            dy, dx = (np.round(v) if abs(v - np.round(v)) < 1e-9 else v for v in (dy, dx))
            neighbour, inside[p] = self._ibbSampleNeighbour(img, dy, dx)
            diffs[p] = neighbour - img
        return diffs, inside

    def _ibbToBool(self, mask):
        """Converts a mask given as 0/255, 0/1 or bool into a boolean array (True = iris)."""
        mask = np.asarray(mask)
        if mask.dtype == bool:
            return mask
        return mask > (0.5 if mask.max() <= 1 else 127.5)

    def _ibbEstimateMask(self, img):
        """Estimates which pixels of the (smoothed) polar image are iris when no mask is given.

        1) Prior: eyelids cover the iris from the outer boundary inwards around 90 and 270
           degrees. Each eyelid is modelled as a cosine-shaped dome h * cos(pi/2 * d / w).
        2) Intensity: eyelid skin, sclera and reflections are much brighter and eyelashes
           much darker than the iris, so pixels far from the median intensity are rejected.
           Median and MAD are used because they are not affected by those very outliers.
        3) Erosion: LBP compares a pixel with its neighbours, so pixels next to the mask
           border would compare iris against eyelid.
        """
        H, W = img.shape
        cols = np.arange(W) / W
        covered = np.zeros(W)                                # fraction of rows covered by an eyelid
        for centre, half_width, height in self.IBB_EYELIDS:
            d = (cols - centre + 0.5) % 1.0 - 0.5            # circular distance to the eyelid centre
            dome = np.where(np.abs(d) < half_width, height * np.cos(0.5 * pi * d / half_width), 0.0)
            covered = np.maximum(covered, dome)
        rows = np.arange(H).reshape(-1, 1)
        # row 0 touches the pupil boundary, which is the least reliable part of the segmentation
        # (the erosion below widens this margin: the estimated mask starts at row IBB_ERODE + 1)
        prior = (rows >= 1) & (rows < H * (1.0 - covered.reshape(1, -1)))

        median = np.median(img[prior])
        mad = np.median(np.abs(img[prior] - median)) + 1e-6
        valid = prior & (np.abs(img - median) < self.IBB_Z_MAX * mad)

        e = self.IBB_ERODE
        padded = cv2.copyMakeBorder(valid.astype(np.uint8), 0, 0, e, e, cv2.BORDER_WRAP)
        padded = cv2.copyMakeBorder(padded, e, e, 0, 0, cv2.BORDER_REPLICATE)
        eroded = cv2.erode(padded, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
        return eroded[e:e + H, e:e + W] > 0                  # [e:-e] would be empty for e = 0

    def _ibbCodeRows(self, array):
        """Selects the rows that are stored in the code (the row axis is the second to last)."""
        return array[..., self.IBB_ROW_START::self.IBB_ROW_STEP, :]

    # Core (LBP) improvements -- what was implemented / optimised.
    # EER / d' measured on 348 images, 80 identities, 60 378 pairs (HDBIF baseline: EER 0.14 %).
    #
    #  1. Own implementation of LBP(P, R) after Ojala et al.: P neighbours on a circle /
    #     ellipse, non-integer positions by bilinear interpolation, bit = (neighbour >= centre).
    #  2. Bit code instead of histograms: the LBP bits stay at their pixel and two codes are
    #     compared with the Hamming distance (as in Daugman's iris code). A histogram throws
    #     the position away. Measured: block histograms + chi-square EER 7.3 %, bit code 0.8 %.
    #  3. Circular wrap-around along the angular axis in smoothing and neighbour sampling
    #     (the left and right borders of the polar image are neighbours on the iris).
    #  4. Gaussian smoothing before LBP (sigma = 2): without it LBP compares sensor noise and
    #     JPEG artefacts instead of iris texture. EER 2.6 % -> 0.8 %.
    #  5. Elliptical neighbourhood (Rx = 12, Ry = 1): iris texture consists mostly of radial
    #     streaks, so intensity varies mainly along the angular direction.
    #     EER 0.76 % (circle, R = 3) -> 0.02 %.
    #  6. Rotation compensation: minimum Hamming distance over circular shifts of +-16
    #     columns (EER 12.7 % -> 7.3 % already for basic LBP). All shifts are evaluated at
    #     once by cross-correlation in the Fourier domain.
    #  7. Occlusion masking: a bit counts only if its pixel is iris and its neighbour does
    #     not fall outside the image. If no mask is passed, it is estimated from the image
    #     (eyelid prior + intensity test + erosion): EER without a mask 2.0 % -> 0.15 %.
    #  8. Soft bit weights (only with a supplied mask): a bit whose neighbour and centre are
    #     almost equally bright flips easily under noise, so it counts less.
    #     d' 5.9 -> 6.3 at the same EER.
    #  9. Parameter optimisation: P, Rx, Ry, sigma, shift range, weights and mask thresholds
    #     were chosen by single-parameter sweeps followed by a joint grid search (144
    #     combinations), checked on two subject-disjoint halves of the data.
    # 10. Compact template: one int8 array [P, H / 2, W] (128 kB); sign = LBP bit,
    #     magnitude = weight, 0 = bit not used.
    #
    # Final result: EER 0.03 % (d' 6.32) with the mask, 0.15 % (d' 6.14) without it.
    @torch.inference_mode()
    def extractIBBCode(self, polar, mask=None):
        """Polar iris image [H, W] -> LBP code, int8 array [P, H / ROW_STEP, W].

        code[p, r, c] > 0   neighbour p is brighter than the centre pixel (LBP bit 1)
        code[p, r, c] < 0   neighbour p is darker (LBP bit 0)
        |code| / 127        weight of the bit in [0, 1]; 0 means the bit is ignored

        mask (optional) is the polar segmentation mask [H, W]; non-zero = iris.
        """
        if polar is None:
            return None
        img = self._ibbBlur(np.asarray(polar, dtype=np.float32))

        # 1) Which pixels are iris: the supplied mask if it is usable, otherwise our own estimate.
        mask_given = mask is not None and np.shape(mask) == img.shape
        iris_pixels = self._ibbToBool(mask) if mask_given else self._ibbEstimateMask(img)

        # 2) LBP: sign of (neighbour - centre) for P neighbours on the ellipse.
        diffs, inside = self._ibbLbpDifferences(img)
        valid = iris_pixels[np.newaxis, :, :] & inside[:, :, np.newaxis]

        # 3) Keep every ROW_STEP-th row.
        diffs, valid = self._ibbCodeRows(diffs), self._ibbCodeRows(valid)

        # 4) Bit weights. With a reliable (supplied) mask, strong differences are trusted more.
        #    With the estimated mask all bits weigh the same: an eyelid edge that the estimate
        #    misses would otherwise get the largest weight (measured: EER 0.15 % -> 0.31 %).
        if mask_given:
            magnitude = np.abs(diffs)
            weight = np.minimum(1.0, magnitude / (self.IBB_WEIGHT_T * np.median(magnitude) + 1e-6))
        else:
            weight = 1.0

        # 5) Pack bit, weight and validity into one int8 value per bit.
        sign = np.where(diffs >= 0, 1.0, -1.0)
        return np.round(127 * sign * weight * valid).astype(np.int8)

    def _ibbSignedWeights(self, code, mask):
        """int8 code -> float tensor in [-1, 1] (sign = bit, magnitude = weight).

        A mask passed at matching time is applied here, so the functions also work when the
        caller supplies masks to matchIBBCodes instead of extractIBBCode.
        """
        x = np.asarray(code, dtype=np.float32) / 127.0
        if mask is not None:
            mask = np.asarray(mask)
            # As in extractIBBCode, a mask of an unexpected shape is ignored instead of raising.
            if mask.ndim == 2 and mask.size > 0:
                keep = self._ibbCodeRows(self._ibbToBool(mask))
                if keep.shape == x.shape[1:]:
                    x = x * keep
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def _ibbCircularCorrelation(self, a, b):
        """sum over bits and rows of  a[..., c] * b[..., c - s]  for every circular shift s.

        Cross-correlation theorem: correlation = IFFT( FFT(a) * conj(FFT(b)) ). This gives
        the same numbers as looping over s with np.roll, but for all W shifts in one pass.
        (torch.fft works in float32 and is ~6x faster here than np.fft.)
        """
        spectrum = (torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1).conj()).sum(dim=(0, 1))
        return torch.fft.irfft(spectrum, n=a.shape[-1]).numpy()

    @torch.inference_mode()
    def matchIBBCodes(self, codes1, codes2, mask1=None, mask2=None):
        """Distance between two codes in [0, 1]; smaller = more similar (same convention as HDBIF).

        ~0.2 for the same iris, ~0.42 for different irises (bits agree by chance),
        IBB_NO_MATCH (1.0) if the codes cannot be compared.

        Weighted Hamming distance, minimised over circular shifts. With x = sign * weight:
            c(s) = sum x1 * shifted(x2)      = weight of agreeing bits - weight of disagreeing bits
            n(s) = sum |x1| * shifted(|x2|)  = total weight of the bits compared
            HD(s) = (n - c) / (2 n)          = weighted fraction of disagreeing bits
        """
        if codes1 is None or codes2 is None:
            return self.IBB_NO_MATCH
        x1 = self._ibbSignedWeights(codes1, mask1)
        x2 = self._ibbSignedWeights(codes2, mask2)
        if x1.shape != x2.shape or x1.numel() == 0:
            return self.IBB_NO_MATCH

        agreement = self._ibbCircularCorrelation(x1, x2)                # c(s) for all shifts
        compared = self._ibbCircularCorrelation(x1.abs(), x2.abs())     # n(s) for all shifts

        # Only shifts -MAX_SHIFT..+MAX_SHIFT are plausible head rotations
        # (negative shifts are stored at the end of the correlation vector).
        W = x1.shape[-1]
        s = min(self.IBB_MAX_SHIFT, W // 2 - 1)
        shifts = np.concatenate([np.arange(0, s + 1), np.arange(W - s, W)])
        agreement, compared = agreement[shifts], compared[shifts]

        # Ignore shifts at which too few bits are valid in both codes.
        enough = compared > self.IBB_MIN_OVERLAP * x1.numel()
        if not np.any(enough):
            return self.IBB_NO_MATCH
        distance = (compared[enough] - agreement[enough]) / (2 * compared[enough])
        # FFT round-off can push the result ~1e-7 outside [0, 1] (e.g. a code matched with itself).
        return float(np.clip(np.min(distance), 0.0, 1.0))


    @torch.inference_mode()
    def matchCodes(self, codes1, codes2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            sumXorCodesMasked = 0
            sumBitsCompared = 0
            total_num_filters = 0
            for code1, code2 in zip(codes1, codes2):
                assert code1.shape == code2.shape # num_filter x filter_size x filter_size
                num_filters, code_size, _ = code1.shape
                r = int((mask1.shape[0] - code_size) / 2)
                mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
                mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
                andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
                if np.sum(andMasks) != 0:
                    xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                    xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (num_filters, 1, 1)))
                    sumXorCodesMasked += np.sum(xorCodesMasked)
                    sumBitsCompared += (np.sum(andMasks) * num_filters)
                    total_num_filters += num_filters
            if sumBitsCompared == 0:
                scoreC.append(float('inf'))
            else:
                scoreC.append(sumXorCodesMasked / sumBitsCompared)
                if self.score_norm:
                    scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( sumBitsCompared / (self.avg_num_bits * total_num_filters) )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
        return scoreC_best, scoreC_index - self.max_shift
    

    @torch.inference_mode()
    def matchCodesEfficient(self, codes1, codes2, mask1, mask2):
        if (np.sum(mask1) <= self.threshold_frac_avg_bits * self.avg_num_bits * 255) or (np.sum(mask2) <= self.threshold_frac_avg_bits * self.avg_num_bits * 255):
            #print("Too small masks")
            return float('inf'), -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1, 2):
            sumXorCodesMasked = 0
            sumBitsCompared = 0
            total_num_filters = 0
            for code1, code2 in zip(codes1, codes2):
                assert code1.shape == code2.shape # num_filter x filter_size x filter_size
                num_filters, code_size, _ = code1.shape
                r = int((mask1.shape[0] - code_size) / 2)
                # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
                mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
                mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
                andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
                if np.sum(andMasks) != 0:
                    xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                    xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (num_filters, 1, 1)))
                    sumXorCodesMasked += np.sum(xorCodesMasked)
                    sumBitsCompared += (np.sum(andMasks) * num_filters)
                    total_num_filters += num_filters
            if sumBitsCompared == 0:
                scoreC.append(float('inf'))
            else:
                scoreC.append(sumXorCodesMasked / sumBitsCompared)
                if self.score_norm:
                    scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( sumBitsCompared / (self.avg_num_bits * total_num_filters) )
                    
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_shift = scoreC_index * 2 - self.max_shift
        scoreC = scoreC[scoreC_index]
        if scoreC == float('inf'):
            print("Too small overlap between masks")
            return float('inf'), -1.0
        
        sumXorCodesMasked_left = 0
        sumBitsCompared_left = 0
        total_num_filters_left = 0
        for code1, code2 in zip(codes1, codes2):
            num_filters, code_size, _ = code1.shape
            r = int((mask1.shape[0] - code_size) / 2)
            mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
            mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
            andMasks_left = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift-1, axis=1))
            if np.sum(andMasks_left) != 0:
                xorCodes = np.logical_xor(code1, np.roll(code2, scoreC_shift-1, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks_left,axis=0), (num_filters, 1, 1)))
                sumXorCodesMasked_left += np.sum(xorCodesMasked)
                sumBitsCompared_left += (np.sum(andMasks_left) * num_filters)
                total_num_filters_left += num_filters

        if sumBitsCompared_left == 0:
            scoreC_left = float('inf')
        else:
            scoreC_left = sumXorCodesMasked_left / sumBitsCompared_left
            if self.score_norm:
                scoreC_left = 0.5 - (0.5 - scoreC_left) * math.sqrt( sumBitsCompared_left / (self.avg_num_bits * total_num_filters_left) )
        
        sumXorCodesMasked_right = 0
        sumBitsCompared_right = 0
        total_num_filters_right = 0
        for code1, code2 in zip(codes1, codes2):
            num_filters, code_size, _ = code1.shape
            r = int((mask1.shape[0] - code_size) / 2)
            mask1_binary = np.where(mask1[r:-r, :] > 127.5, True, False)
            mask2_binary = np.where(mask2[r:-r, :] > 127.5, True, False)
            andMasks_right = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift+1, axis=1))
            if np.sum(andMasks_right) != 0:
                xorCodes = np.logical_xor(code1, np.roll(code2, scoreC_shift+1, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks_right,axis=0), (num_filters, 1, 1)))
                sumXorCodesMasked_right += np.sum(xorCodesMasked)
                sumBitsCompared_right += (np.sum(andMasks_right) * num_filters)
                total_num_filters_right += num_filters

        if sumBitsCompared_right == 0:
            scoreC_right = float('inf')
        else:
            scoreC_right = sumXorCodesMasked_right / sumBitsCompared_right
            if self.score_norm:
                scoreC_right = 0.5 - (0.5 - scoreC_right) * math.sqrt( sumBitsCompared_right / (self.avg_num_bits * total_num_filters_right) )
        
        scoreC_best = min(scoreC, scoreC_left, scoreC_right)
        if scoreC_best == scoreC_left:
            scoreC_shift -= 1
        elif scoreC_best == scoreC_right:
            scoreC_shift += 1
        
        return scoreC_best, scoreC_shift






##########################################################
### Adam's extra functions -- no need to correct those ###
##########################################################

    @torch.inference_mode()
    def matchCodesEfficientRaw(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            print("Too small masks")
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1, 2):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_shift = scoreC_index * 2 - self.max_shift
        scoreC = scoreC[scoreC_index]
        if scoreC == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
        
        andMasks_left = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift-1, axis=1))
        if np.sum(andMasks_left) == 0:
            scoreC_left = float('inf')
        else:
            xorCodes_left = np.logical_xor(code1, np.roll(code2, scoreC_shift-1, axis=2))
            xorCodesMasked_left = np.logical_and(xorCodes_left, np.tile(np.expand_dims(andMasks_left,axis=0), (self.total_num_filters, 1, 1)))
            scoreC_left = np.sum(xorCodesMasked_left) / (np.sum(andMasks_left) * self.total_num_filters)
        
        andMasks_right = np.logical_and(mask1_binary, np.roll(mask2_binary, scoreC_shift+1, axis=1))
        if np.sum(andMasks_right) == 0:
            scoreC_right = float('inf')
        else:
            xorCodes_right = np.logical_xor(code1, np.roll(code2, scoreC_shift+1, axis=2))
            xorCodesMasked_right = np.logical_and(xorCodes_right, np.tile(np.expand_dims(andMasks_right,axis=0), (self.total_num_filters, 1, 1)))
            scoreC_right = np.sum(xorCodesMasked_right) / (np.sum(andMasks_right) * self.total_num_filters)
        
        scoreC_best = min(scoreC, scoreC_left, scoreC_right)
        if scoreC_best == scoreC_left:
            scoreC_shift -= 1
        elif scoreC_best == scoreC_right:
            scoreC_shift += 1
        
        return scoreC_best, scoreC_shift
    
    @torch.inference_mode()
    def matchCodesRaw(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
        return scoreC_best, scoreC_index - self.max_shift
    
    @torch.inference_mode()
    def matchCodesKernelSubset(self, code1, code2, mask1, mask2, kernel_selection):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []

        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))

                _scoreC = []
                for s in kernel_selection:
                    _scoreC.append(np.sum(xorCodesMasked[s,:,:]) / (np.sum(andMasks)))
                
                scoreC.append(np.mean(np.array(_scoreC)))
                    
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]

        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
                
            # DEBUG: 
            # full_path_code = f'temp_images/c-{c:02d}-{xshift+self.max_shift:02d}-{np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks)):0.2f}.png'
            # Image.fromarray(xorCodeMasked_MorphOpened*255).save(full_path_code)

        return scoreC_best, scoreC_index - self.max_shift

    @torch.inference_mode()
    def matchCodesKernelWise(self, code1, code2, mask1, mask2):
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1 > 127.5, True, False)
        mask2_binary = np.where(mask2 > 127.5, True, False)
        scoreC = []
        scoreM = np.zeros([self.total_num_filters,2*self.max_shift+1],dtype=float)
        scoreM_Morph = np.zeros([self.total_num_filters,2*self.max_shift+1],dtype=float)

        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.total_num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.total_num_filters))

                for c in range(self.total_num_filters):
                    scoreM[c,xshift+self.max_shift] = np.sum(xorCodesMasked[c,:,:]/np.sum(andMasks))

                    xorCodeMasked_MorphOpened = cv2.morphologyEx(np.array(xorCodesMasked[c,:,:],dtype=np.uint8), cv2.MORPH_OPEN, self.morph_kernel)
                    scoreM_Morph[c,xshift+self.max_shift] = np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks))
                    
                    # DEBUG: 
                    # full_path_code = f'temp_images/c-{c:02d}-{xshift+self.max_shift:02d}-{np.sum(xorCodeMasked_MorphOpened/np.sum(andMasks)):0.2f}.png'
                    # Image.fromarray(xorCodeMasked_MorphOpened*255).save(full_path_code)

        # raw / not-normalized / stadnard score
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]

        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0, -1.0, -1.0
        
        scoreM_best = np.min(scoreM,axis=1)
        scoreM_index = np.argmin(np.array(scoreM),axis=1)

        scoreM_Morph_best = np.min(scoreM_Morph,axis=1)
        scoreM_Morph_index = np.argmin(np.array(scoreM),axis=1)

        return scoreC_best, scoreC_index - self.max_shift, scoreM_best, scoreM_index - self.max_shift, scoreM_Morph_best, scoreM_Morph_index - self.max_shift
