
import torch
import math
import numpy as np
import scipy
from PIL import Image

import torchvision
import torchvision.transforms as transformsimport
import torchvision.transforms as transforms


class AddPatch(object):
    def __init__(self, image_size, patch_size, min_in=0, max_in=0, patch_path=None) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        if patch_path is None:
            self.patch, self.patch_shape = self._init_patch_circle(self.image_size, self.patch_size)
        else:
            self.patch, self.patch_shape = self._load_patch(patch_path)


    def __call__(self, sample):
        im = sample

        data_shape = im.shape
        patch, mask, patch_shape = self._circle_transform(self.patch, data_shape, self.patch_shape, self.image_size)
        patch, mask = torch.tensor(patch, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        im = ((1-mask) * im) + (mask * patch)
        im = torch.clamp(im, -1, 1)
        
        # masked_patch = torch.mul(mask, patch)
        # patch = masked_patch.data.cpu().numpy()
        # new_patch = np.zeros(patch_shape)
        # for j in range(new_patch.shape[0]): 
        #     new_patch[i][j] = self._submatrix(patch[i][j])
        
        return im
    
    def _load_patch(self, path):
        img = Image.open( path )
        img.load()
        patch_length = int(((self.image_size ** 2) * self.patch_size) ** 0.5)
        img = img.resize((3, patch_length, patch_length))
        data = np.asarray(img, dtype="float32")
        data[data > 254] = 0
        return data, data.shape

    

    def _init_patch_circle(self, image_size, patch_size):
        image_size = image_size**2
        noise_size = int(image_size*patch_size)
        radius = int(math.sqrt(noise_size/math.pi))
        # patch = np.zeros((3, radius*2, radius*2))    
        patch = np.random.rand(3, radius*2, radius*2)
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 > radius**2
        # index = np.reshape(index, (3, radius*2, radius*2))
        patch[0][index] = 0
        patch[1][index] = 0
        patch[2][index] = 0

        # for i in range(3):
        #     a = np.zeros((radius*2, radius*2))    
        #     cx, cy = radius, radius # The center of circle 
        #     y, x = np.ogrid[-radius: radius, -radius: radius]
        #     index = x**2 + y**2 > radius**2
        #     # a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        #     a[cy-radius:cy+radius, cx-radius:cx+radius][index] = 0
        #     idx = np.flatnonzero((a == 0).all((1)))
        #     a = np.delete(a, idx, axis=0)
        #     patch[i] = np.delete(a, idx, axis=1)
        return patch, patch.shape

    def _circle_transform(self, patch, data_shape, patch_shape, image_size):
        # get dummy image 
        x = np.zeros(data_shape)
    
        # get shape
        m_size = patch_shape[-1]
        
        # random rotation
        rot = np.random.choice(360)
        for j in range(patch.shape[0]):
            patch[j] = scipy.ndimage.interpolation.rotate(patch[j], angle=rot, reshape=False)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
    
        # apply patch to dummy image  
        # x[:, random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch
        x[0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[0]
        x[1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[1]
        x[2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[2]
        
        mask = np.copy(x)
        mask[mask != 0] = 1.0
        
        return x, mask, patch.shape

    def _submatrix(self, arr):
        x, y = np.nonzero(arr)
        # Using the smallest and largest x and y indices of nonzero elements, 
        # we can find the desired rectangular bounds.  
        # And don't forget to add 1 to the top bound to avoid the fencepost problem.
        return arr[x.min():x.max()+1, y.min():y.max()+1]
    

##########################################################################
# Testing

def imshow(img, out_path):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    
    npimg = np.transpose(npimg * 255, (1, 2, 0))
    npimg = npimg.astype("uint8")
    # plt.imshow(npimg)
    img = Image.fromarray(npimg, "RGB")
    img.save(out_path)

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
    AddPatch(32, 0.15, "./toster_patch.png"),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images), "./plots/original_images.png")