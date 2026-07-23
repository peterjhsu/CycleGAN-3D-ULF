import SimpleITK as sitk
import os
import re
import numpy as np
import math
import random
import glob
import scipy.ndimage.interpolation as interpolation
import scipy
import torch
import torch.utils.data
import matplotlib.pyplot as plt

# ------- Swithes -------

interpolator_image = sitk.sitkLinear                 # interpolator image
interpolator_label = sitk.sitkLinear                  # interpolator label

_interpolator_image = 'linear'          # interpolator image
_interpolator_label = 'linear'          # interpolator label

Segmentation = False

# ------------------------------------- Functions ---------------------------------------

def lossG_figure(ptitle, save_filepath, epc, train_loss):
    plt.clf()
    plt.figure(num=1, figsize=(10,10))
    plt.plot(train_loss[0], 'k-', label='Training Combined Loss', linewidth=2)
    plt.plot(train_loss[1], 'k--', label='Training GAN Loss', linewidth=1)
    plt.plot(train_loss[2], 'k-.', label='Training Cycle Loss', linewidth=1)
    plt.plot(train_loss[3], 'k:', label='Training Identity Loss', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim((0, epc+1))
    plt.title(ptitle)
    plt.legend()
    plt.savefig(save_filepath)

def lossD_figure(ptitle, save_filepath, epc, train_loss):
    plt.clf()
    plt.figure(num=1, figsize=(10,10))
    plt.plot(train_loss, 'k-', label='Training Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim((0, epc+1))
    plt.title(ptitle)
    plt.legend()
    plt.savefig(save_filepath)

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def lstFiles(Path):
    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    # https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


def matrix_from_axis_angle(a):
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])
    return R


def resample_vol(volume, interpolator=sitk.sitkLinear, new_spacing=[1.0, 1.0, 1.0]):
    og_spacing = volume.GetSpacing()
    og_spacing = (round(og_spacing[0],2), round(og_spacing[1],2) , round(og_spacing[2], 2))
    og_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(og_size, og_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator, volume.GetOrigin(),
                         new_spacing, volume.GetDirection(), 0, volume.GetPixelID())


def resample_image(image, transform):
    reference_image = image
    interpolator = interpolator_image
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def resample_label(image, transform):
    reference_image = image
    interpolator = interpolator_label
    default_value = 0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def get_center(img):
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                              int(np.ceil(height / 2)),
                                              int(np.ceil(depth / 2))))


def rotation3d_image(image, theta_x, theta_y, theta_z):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    resampled_image = resample_image(image, euler_transform)
    return resampled_image


def rotation3d_label(image, theta_x, theta_y, theta_z):
   """
   This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
   respectively
   :param image: An sitk MRI image
   :param theta_x: The amount of degrees the user wants the image rotated around the x axis
   :param theta_y: The amount of degrees the user wants the image rotated around the y axis
   :param theta_z: The amount of degrees the user wants the image rotated around the z axis
   :param show: Boolean, whether or not the user wants to see the result of the rotation
   :return: The rotated image
   """
   theta_x = np.deg2rad(theta_x)
   theta_y = np.deg2rad(theta_y)
   theta_z = np.deg2rad(theta_z)
   euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
   image_center = get_center(image)
   euler_transform.SetCenter(image_center)
   euler_transform.SetRotation(theta_x, theta_y, theta_z)
   resampled_image = resample_label(image, euler_transform)
   return resampled_image


def flipit(image, axes):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    if axes == 0:
        array = np.fliplr(array)
    if axes == 1:
        array = np.flipud(array)

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return image


def brightness(image):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    max = 255
    min = 0

    c = np.random.randint(-20, 20)

    array = array + c

    array[array >= max] = max
    array[array <= min] = min

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def contrast(image):
    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    shape = array.shape
    ntotpixel = shape[0] * shape[1] * shape[2]
    IOD = np.sum(array)
    luminanza = int(IOD / ntotpixel)

    c = np.random.randint(-20, 20)

    d = array - luminanza
    dc = d * abs(c) / 100

    if c >= 0:
        J = array + dc
        J[J >= 255] = 255
        J[J <= 0] = 0
    else:
        J = array - dc
        J[J >= 255] = 255
        J[J <= 0] = 0

    img = sitk.GetImageFromArray(np.transpose(J, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    array = scipy.ndimage.interpolation.shift(array, (int(offset[0]), int(offset[1]), 0), order=order)

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img


def imadjust(image, gamma=np.random.uniform(1, 2)):

    array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    array = (((array - array.min()) / (array.max() - array.min())) ** gamma) * (255 - 0) + 0

    img = sitk.GetImageFromArray(np.transpose(array, axes=(2, 1, 0)))
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    return img

# --------------------------------------------------------------------------------------

class NiftiDataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, label_path, data_nums, label_nums,
                 in_channels, out_channels, split_train,
                 which_direction='AtoB',
                 transforms=None,
                 shuffle_labels=False,
                 train=False,
                 test=False,
                 norm="zscore"):

        # Init membership variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_train = split_train

        # Get Images (Multi-Channel Input and Output)
        self.images_list = self.get_imgs(data_path, data_nums)
        self.labels_list = self.get_imgs(label_path, label_nums)
            
        # Length of Images and Labels
        self.images_size = len(self.images_list[0])
        self.labels_size = len(self.labels_list[0])

        self.which_direction = which_direction
        self.transforms = transforms

        self.shuffle_labels = shuffle_labels
        self.train = train
        self.test = test

        self.bit = sitk.sitkFloat32
        self.norm = norm

    # Multi-Channel Input from Multiple Folders
    def get_imgs(self, img_path, img_nums):
        data_dirs = []
        for dir in sorted(os.listdir(img_path)):
            if "images" in dir:
                data_dirs.append(dir)
        img_list = []
        for i in img_nums:
            img_list.append(lstFiles(os.path.join(img_path, data_dirs[i-1])))
        return img_list

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def __getitem__(self, index):
        data_path = [self.images_list[i][index] for i in range(self.in_channels)]
        head, img_file = os.path.split(data_path[0])
        index_B = random.randint(0, self.labels_size - 1)
        label_path = [self.labels_list[i][index_B] for i in range(self.out_channels)]

        if self.which_direction == 'AtoB':
            data_path = data_path
            label_path = label_path

        elif self.which_direction == 'BtoA':
            data_path_copy = data_path
            label_path_copy = label_path
            label_path = data_path_copy
            data_path = label_path_copy

        # read image and label
        image = [self.read_image(path) for path in data_path]

        # set intensity 0-255
        image = [Normalization(img) for img in image]

        # cast image and label
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(self.bit)
        image = [castImageFilter.Execute(img) for img in image]

        if self.train or self.test:
            label = [self.read_image(path) for path in label_path]
            if Segmentation is False:
                label = [Normalization(lab) for lab in label] # set intensity 0-255
            castImageFilter.SetOutputPixelType(self.bit)
            label = [castImageFilter.Execute(lab) for lab in label]

        else:
            label = sitk.Image(image[0].GetSize(), self.bit)
            label.SetOrigin(image[0].GetOrigin())
            label.SetSpacing(image[0].GetSpacing())

        # Resample to 1mm isotropic
        image = [resample_vol(img) for img in image]
        label = [resample_vol(lab) for lab in label]

        sample = {}
        for i in range(len(image)):
            key = "image"+str(i+1) if i>0 else "image"
            sample[key] = image[i]
        for i in range(len(label)):
            key = "label"+str(i+1) if i>0 else "label"
            sample[key] = label[i]

        # apply transforms
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        # convert sample to tensors
        image_np, label_np = [], []
        for key in sample:
            if "image" in key:
                image_np.append(abs(sitk.GetArrayFromImage(sample[key])))
            elif "label" in key:
                label_np.append(abs(sitk.GetArrayFromImage(sample[key])))

        if Segmentation is True:
            label_np = [abs(np.around(lab)) for lab in label_np]

        # unify matrix dimension order
        image_np = [np.transpose(img, (2, 1, 0)) for img in image_np]
        label_np = [np.transpose(lab, (2, 1, 0)) for lab in label_np]

        # Return to 0 mean, 1 variance
        if self.norm == "zscore":
            image_np = [(img - 127.5) / 127.5 for img in image_np]
            label_np = [(lab - 127.5) / 127.5 for lab in label_np]
        # Change Scaling to 0 min, 1 max
        elif self.norm == "zeroone":
            image_np = [(img - np.min(img)) / (np.max(img) - np.min(img)) for img in image_np]
            label_np = [(lab - np.min(lab)) / (np.max(lab) - np.min(lab)) for lab in label_np]
        else:
            raise ValueError("img_norm must be zscore or zeroone")

        # Convert to NP Array
        image_np = np.array(image_np)
        label_np = np.array(label_np)

        # Convert to Torch Tensor and Feed to Network
        return torch.from_numpy(image_np), torch.from_numpy(label_np)

    def __len__(self):
        return len(self.images_list[0])


def Normalization(image):
    #Normalize an image to 0 - 255 (8bits)
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255
    return image

def Rescale_01(image):
    #Normalize an image to 0 - 1 (8bits)
    #normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(1.0)
    resacleFilter.SetOutputMinimum(0)

    #image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-1.0
    return image


class StatisticalNormalization(object):
    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):

        for key in sample:
            if "image" in key:
                statisticsFilter = sitk.StatisticsImageFilter()
                statisticsFilter.Execute(sample[key])

                intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
                intensityWindowingFilter.SetOutputMaximum(255)
                intensityWindowingFilter.SetOutputMinimum(0)
                intensityWindowingFilter.SetWindowMaximum(
                    statisticsFilter.GetMean() + self.sigma * statisticsFilter.GetSigma());
                intensityWindowingFilter.SetWindowMinimum(
                    statisticsFilter.GetMean() - self.sigma * statisticsFilter.GetSigma());

                sample[key] = intensityWindowingFilter.Execute(sample[key])

        return sample


class LaplacianRecursive(object):
    """
    Laplacian recursive image filter
    """

    def __init__(self, sigma):
        self.name = 'Laplacianrecursiveimagefilter'
        assert isinstance(sigma, (int, float))
        self.sigma = sigma


    def __call__(self, sample):

        filter = sitk.LaplacianRecursiveGaussianImageFilter()
        filter.SetSigma(1.5)

        for key in sample:
            if "image" in key:
                sample[key] = filter.Execute(sample[key])
            elif "label" in key:
                if Segmentation == False:
                    sample[key] = filter.Execute(sample[key])

        return sample


class Reorient(object):
    """
    (Beta) Function to orient image in specific axes order
    The elements of the order array must be an permutation of the numbers from 0 to 2.
    """

    def __init__(self, order):
        self.name = 'Reoreient'
        assert isinstance(order, (int, tuple))
        assert len(order) == 3
        self.order = order

    def __call__(self, sample):
        reorientFilter = sitk.PermuteAxesImageFilter()
        reorientFilter.SetOrder(self.order)

        for key in sample:
            sample[key] = reorientFilter.Execute(sample[key])

        return sample


class Resample(object):
    """
    Resample the volume in a sample to a given voxel size

      Args:
          voxel_size (float or tuple): Desired output size.
          If float, output volume is isotropic.
          If tuple, output voxel size is matched with voxel size
          Currently only support linear interpolation method
    """

    def __init__(self, new_resolution, check):
        self.name = 'Resample'

        # assert isinstance(new_resolution, (float, tuple))
        if isinstance(new_resolution, float):
            self.new_resolution = new_resolution
            self.check = check
        else:
            # assert len(new_resolution) == 3
            self.new_resolution = new_resolution
            self.check = check

    def __call__(self, sample):

        new_resolution = self.new_resolution
        check = self.check

        if check is True:
            for key in sample:
                if "image" in key:
                    sample[key] = resample_sitk_image(
                                    sample[key],
                                    spacing=new_resolution,
                                    interpolator=_interpolator_image
                                    )
                elif "label" in key:
                    sample[key] = resample_sitk_image(
                                    sample[key],
                                    spacing=new_resolution,
                                    interpolator=_interpolator_label
                                    )
            return sample

        if check is False:
            return sample


class Resample_Pad(object):
    def __init__(self, voxel_size, output_size):
        self.name = 'Resample_Pad'

        assert isinstance(voxel_size, (int, tuple))
        if isinstance(voxel_size, int):
            self.voxel_size = (voxel_size, voxel_size, voxel_size)
        else:
            assert len(voxel_size) == 3
            self.voxel_size = voxel_size

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.voxel_size))
        assert all(i > 0 for i in list(self.output_size))

    def __call__(self, sample):
        vx = list(self.voxel_size)
        sz = list(self.output_size)

        # Resample Image and Label
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.voxel_size)
        for key in sample:
            if sample[key].GetSpacing() != self.voxel_size:

                # Determine Output Size via Desired Voxel Size
                og_sz = list(sample[key].GetSize())
                og_vx = list(sample[key].GetSpacing())
                new_sz = [og_vx[0]/vx[0]*og_sz[0],
                          og_vx[1]/vx[1]*og_sz[1],
                          og_vx[2]/vx[2]*og_sz[2]]
                new_sz = [round(i) for i in new_sz]

                resampler.SetOutputOrigin(sample[key].GetOrigin())
                resampler.SetOutputDirection(sample[key].GetDirection())
                resampler.SetSize(new_sz)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(0)
                sample[key] = resampler.Execute(sample[key])

        # Pad Image and Label
        for key in sample:
            dim = sample[key].GetSize()
            if (dim[0] >= sz[0]) and (dim[1] >= sz[1]) and (dim[2] >= sz[2]):
                pass
            else:
                xpad = [math.ceil((sz[0] - dim[0])/2), math.floor((sz[0] - dim[0])/2)]
                ypad = [math.ceil((sz[1] - dim[1])/2), math.floor((sz[1] - dim[1])/2)]
                zpad = [math.ceil((sz[2] - dim[2])/2), math.floor((sz[2] - dim[2])/2)]
                pad_up, pad_down = (xpad[0], ypad[0], zpad[0]), (xpad[1], ypad[1], zpad[1])
                sample[key] = sitk.ConstantPad(sample[key], pad_up, pad_down, constant=0)

        return sample


class Padding(object):
    """
    Add padding to the image if size is smaller than patch size

      Args:
          output_size (tuple or int): Desired output size. If int, a cubic volume is formed
      """

    def __init__(self, output_size):
        self.name = 'Padding'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()

        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (
                size_old[2] >= self.output_size[2]):
            return sample
        else:
            output_size = self.output_size
            output_size = list(output_size)
            if size_old[0] > self.output_size[0]:
                output_size[0] = size_old[0]
            if size_old[1] > self.output_size[1]:
                output_size[1] = size_old[1]
            if size_old[2] > self.output_size[2]:
                output_size[2] = size_old[2]

            output_size = tuple(output_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(output_size)

            # Resample Image and Label
            for key in sample:
                if "image" in key:
                    resampler.SetInterpolator(sitk.sitkBSpline)
                    resampler.SetOutputOrigin(image.GetOrigin())
                    resampler.SetOutputDirection(image.GetDirection())
                    sample[key] = resampler.Execute(sample[key])

                elif "label" in key:
                    resampler.SetInterpolator(sitk.sitkBSpline)
                    resampler.SetOutputOrigin(label.GetOrigin())
                    resampler.SetOutputDirection(label.GetDirection())
                    sample[key] = resampler.Execute(sample[key])

            return sample


class Adapt_eq_histogram(object):
    """
    (Beta) Function to orient image in specific axes order
    The elements of the order array must be an permutation of the numbers from 0 to 2.
    """

    def __init__(self):
        self.name = 'Adapt_eq_histogram'

    def __call__(self, sample):

        for key in sample:
            if "image" in key:
                adapt = sitk.AdaptiveHistogramEqualizationImageFilter()
                adapt.SetAlpha(0.7)
                adapt.SetBeta(0.8)
                sample[key] = adapt.Execute(sample[key]) # set mean and std deviation

                resacleFilter = sitk.RescaleIntensityImageFilter()
                resacleFilter.SetOutputMaximum(255)
                resacleFilter.SetOutputMinimum(0)
                sample[key] = resacleFilter.Execute(sample[key]) # set mean and std deviation

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
      Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
      This transformation only applicable in train mode

    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(drop_ratio, (int, float))
        if drop_ratio >= 0 and drop_ratio <= 1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')

        assert isinstance(min_pixel, int)
        if min_pixel >= 0:
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        while not contain_label:
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0] - size_new[0])

            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1] - size_new[1])

            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2] - size_new[2])

            roiFilter.SetIndex([start_i, start_j, start_k])

            for key in sample:
                if "label" in key:
                    if Segmentation is False:
                        # threshold label into only ones and zero
                        threshold = sitk.BinaryThresholdImageFilter()
                        threshold.SetLowerThreshold(0)
                        threshold.SetUpperThreshold(255)
                        threshold.SetInsideValue(1)
                        threshold.SetOutsideValue(0)
                        mask = threshold.Execute(sample[key])
                        mask_cropped = roiFilter.Execute(mask)
                        sample[key] = roiFilter.Execute(sample[key])
                        statFilter = sitk.StatisticsImageFilter()
                        statFilter.Execute(mask_cropped)  # mine for GANs

                    if Segmentation is True:
                        sample[key] = roiFilter.Execute(sample[key])
                        statFilter = sitk.StatisticsImageFilter()
                        statFilter.Execute(label_crop)

            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum() < self.min_pixel:
                contain_label = self.drop(self.drop_ratio)  # has some probabilty to contain patch with empty label
            else:
                contain_label = True

        for key in sample:
            if "image" in key:
                sample[key] = roiFilter.Execute(sample[key])

        return sample

    def drop(self, probability):
        return random.random() <= probability

class RegistrationError(object):

    def __init__(self):
        self.name = 'RegistrationError'

    def __call__(self, sample):
        theta_x = np.random.randint(-5, 5)
        theta_y = np.random.randint(-5, 5)
        theta_z = np.random.randint(-5, 5)

        m,n = 0,0
        for key in sample:
            r = np.random.choice([0, 1, 2, 3])
            if "image" in key:
                n+=1
                if r == 3:
                    sample[key] = rotation3d_image(sample[key], theta_x, theta_y, theta_z)
                    m+=1

        if m > n/2:
            for key in sample:
                if "label" in key:
                    sample[key] = rotation3d_label(sample[key], theta_x, theta_y, theta_z)

        return sample

class Augmentation(object):
    """
    Application of transforms. This is usually used for data augmentation.
    List of transforms:
    0 = no transform                 1 = random Gaussian noise
    2 = recursive Gaussian filter    3 = random rotation in x,y,z dimensions
    4 = b-spline deformation         5 = random flipping
    6 = brightness augmentation      7 = contrast augmentation
    8 = random translation           9 = random rotation in z dim
    10 = random rotation in x dim   11 = random rotation in y dim
    12 = gamma augmentation
    """

    def __init__(self):
        self.name = 'Augmentation'

    def __call__(self, sample):

        choice = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12])

        # No Augmentation
        if choice == 0:
            return sample

        # Additive Gaussian noise
        if choice == 1:
            mean = np.random.uniform(0, 1)
            std = np.random.uniform(0, 2)
            self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
            self.noiseFilter.SetMean(mean)
            self.noiseFilter.SetStandardDeviation(std)

            for key in sample:
                if "image" in key:
                    sample[key] = self.noiseFilter.Execute(sample[key])

            return sample

        # Recursive Gaussian
        if choice == 2:
            sigma = np.random.uniform(0, 1.5)
            self.noiseFilter = sitk.RecursiveGaussianImageFilter()
            self.noiseFilter.SetOrder(0)
            self.noiseFilter.SetSigma(sigma)

            for key in sample:
                if "image" in key:
                    sample[key] = self.noiseFilter.Execute(sample[key])

            return sample

        # Random rotation x y z
        if choice == 3:
            theta_x = np.random.randint(-40, 40)
            theta_y = np.random.randint(-40, 40)
            theta_z = np.random.randint(-40, 40)

            for key in sample:
                sample[key] = rotation3d_image(sample[key], theta_x, theta_y, theta_z)

            return sample

        # BSpline Deformation
        if choice == 4:
            randomness = 10

            assert isinstance(randomness, (int, float))
            if randomness > 0:
                self.randomness = randomness
            else:
                raise RuntimeError('Randomness should be non zero values')

            image, label = sample['image'], sample['label']
            spline_order = 3
            domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                          image.GetSize()[1] * image.GetSpacing()[1],
                                          image.GetSize()[2] * image.GetSpacing()[2]]

            bspline = sitk.BSplineTransform(3, spline_order)
            bspline.SetTransformDomainOrigin(image.GetOrigin())
            bspline.SetTransformDomainDirection(image.GetDirection())
            bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
            bspline.SetTransformDomainMeshSize((10, 10, 10))

            # Random displacement of the control points.
            originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
            bspline.SetParameters(originalControlPointDisplacements)

            for key in sample:
                sample[key] = sitk.Resample(sample[key], bspline)
            return sample

        # Random flip
        if choice == 5:
            axes = np.random.choice([0, 1])
            for key in sample:
                sample[key] = flipit(sample[key], axes)
            return sample

        # Brightness
        if choice == 6:
            for key in sample:
                sample[key] = brightness(sample[key])

            return sample

        # Contrast
        if choice == 7:
            for key in sample:
                sample[key] = contrast(sample[key])

            return sample

        # Random Translation
        if choice == 8:
            t1 = np.random.randint(-40, 40)
            t2 = np.random.randint(-40, 40)
            offset = [t1, t2]

            for key in sample:
                sample[key] = translateit(sample[key], offset)
            return sample

        # Random rotation z
        if choice == 9:
            theta_x = 0
            theta_y = 0
            theta_z = np.random.randint(-180, 180)

            for key in sample:
                sample[key] = rotation3d_image(sample[key],theta_x,theta_y, theta_z)

            return sample

        # Random rotation x
        if choice == 10:
            theta_x = np.random.randint(-40, 40)
            theta_y = 0
            theta_z = 0

            for key in sample:
                sample[key] = rotation3d_image(sample[key],theta_x,theta_y, theta_z)

            return sample

        # Random rotation y
        if choice == 11:
            theta_x = 0
            theta_y = np.random.randint(-40, 40)
            theta_z = 0

            for key in sample:
                sample[key] = rotation3d_image(sample[key],theta_x,theta_y, theta_z)

            return sample

        # histogram gamma
        if choice == 12:
            for key in sample:
                if "image" in key:
                    sample[key] = imadjust(sample[key])
            return sample


class BSplineDeformation(object):
    """
    Image deformation with a sparse set of control points to control a free form deformation.
    Details can be found here:
    https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
    https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html

    Args:
      randomness (int,float): BSpline deformation scaling factor, default is 4.
    """

    def __init__(self, randomness=4):
        self.name = 'BSpline Deformation'

        assert isinstance(randomness, (int, float))
        if randomness > 0:
            self.randomness = randomness
        else:
            raise RuntimeError('Randomness should be non zero values')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        spline_order = 3
        domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                      image.GetSize()[1] * image.GetSpacing()[1],
                                      image.GetSize()[2] * image.GetSpacing()[2]]

        bspline = sitk.BSplineTransform(3, spline_order)
        bspline.SetTransformDomainOrigin(image.GetOrigin())
        bspline.SetTransformDomainDirection(image.GetDirection())
        bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
        bspline.SetTransformDomainMeshSize((4, 4, 4))

        # Random displacement of the control points.
        originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
        bspline.SetParameters(originalControlPointDisplacements)

        for key in sample:
            sample[key] = sitk.Resample(sample[key], bspline)
        return sample

    def NormalOffset(self, size, sigma):
        s = np.random.normal(0, size * sigma / 2, 100)  # 100 sample is good enough
        return int(round(random.choice(s)))
