import torch
import numpy as np
from skimage.io import imread
import os
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from glob import glob
# This code is mainly used to experiment channel wise normalization
def makeNormal(x,bits=False):
    if bits:
        return x/255
    else:
        return (x-x.min())/(x.max()-x.min())

def makeNormal_mu_std(img, mus, stds):
    assert len(mus) == len(stds) == img.shape[-1], 'provided stas and number of channels is not teh same'
    return np.dstack([(img[:,:,i]-mus[i])/stds[i] for i in range(len(mus))])

def readFile(file, bits):
    x = imread(file)[:,:,:3]
    y = makeNormal(x, bits)
    return y

def readFile_builtin(file, weights=None):
    img = read_image(file)
    batch = weights(img).unsqueeze(0)
    return batch

def generateFeatures(data_root, out_root, mu_std=True, weight='v1', per_chip=True, built_in=True):
    if weight=='v1':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    elif weight=='v2':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    else:
        raise ValueError('Weight version {} not known'.format(weight))
     
    # _ = folds.pop(folds.index("Rakuba_sudan"))
    # print(folds)
       
    folds = os.listdir(data_root)
    # folds.remove('Rakuba_sudan')
    print(folds)
    
    for fold in folds:
        out_dir = out_root + '/' + fold
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok = True)
        print(fold)
       
        files = glob(f"{data_root}/{fold}/test/images/*.tif")

        
        if not mu_std:
            if not built_in:
                for i in tqdm(range(len(files))):
                    xx = readFile(files[i], bits=False)
                    xy = torch.from_numpy(xx).permute(2, 0, 1).unsqueeze(dim=0) 
                    outs = model(xy.float()) 
                    np.save(f'{out_dir}/{i}.npy', outs.detach().numpy())
            else:
                preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms(crop_size=256, resize_size=256) if weight == 'v1' else ResNet50_Weights.IMAGENET1K_V2.transforms(crop_size=256, resize_size=256)  
                for i in tqdm(range(len(files))):
                    xx = readFile_builtin(file=files[i], weights=preprocess) 
                    outs = model(xx.float()) 
                    np.save(f'{out_dir}/{i}.npy', outs.detach().numpy())
        else:    
            arrays = np.stack([imread(fls)[:,:,:3] for fls in files]) 
            if not per_chip:
                means = [np.mean(arrays[:,:,i]) for i in range(3)]
                stdevs = [np.std(arrays[:,:,i]) for i in range(3)]
                print('Channel means: {}'.format(means))
                print('Channel stds: {}'.format(stdevs))
                for i in tqdm(range(arrays.shape[0])):
                    xx = arrays[i]
                    xx = makeNormal_mu_std(img=xx, mus=means, stds=stdevs)  # scale using chanell wise global mean adn standard deviation
                    xy = torch.from_numpy(xx).permute(2, 0, 1).unsqueeze(dim=0)
                    outs = model(xy.float())
                    np.save(f'{out_dir}/{i}.npy', outs.detach().numpy())
            else:
                for i in tqdm(range(arrays.shape[0])):
                    xx = arrays[i]
                    means = [np.mean(xx[:,:,i]) for i in range(3)]
                    stdevs = [np.std(xx[:,:,i]) for i in range(3)]
                    xx = makeNormal_mu_std(img=xx, mus=means, stds=stdevs)  # scale using chanell wise global mean adn standard deviation
                    xy = torch.from_numpy(xx).permute(2, 0, 1).unsqueeze(dim=0)
                    outs = model(xy.float())
                    np.save(f'{out_dir}/{i}.npy', outs.detach().numpy())
def argumentParser():
    parser = argparse.ArgumentParser(description = 'Deep feature space embeding plot')
    parser.add_argument('--data_root', help='folder containes all subfolders for the dataset', type=str, required=False, default='/data')
    parser.add_argument('--save_dir', help = 'root to save deep features', type = str, required=False, default='/deep_feat_camp') 
    arg = parser.parse_args()
    return arg
    return arg

if __name__ == "__main__":
    args = argumentParser()
    generateFeatures(data_root=args.data_root, out_root=args.save_dir, mu_std=False, weight='v1', per_chip=False, built_in=False)
