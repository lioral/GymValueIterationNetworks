import numpy as np
import os
from time import strftime, localtime
from collections import namedtuple
from PIL import Image
import matplotlib.pylab as plt
import torch
from PIL import Image
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def resize_image_list(imgs, imsize, divice):
    resized_imgs = np.empty((len(imgs), 1, *imsize))

    resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.Resize(imsize, interpolation=Image.CUBIC),
                        T.ToTensor()])

    for ii in range(len(imgs)):
        resized_imgs[ii,] = resize(imgs[ii])

    screen = np.ascontiguousarray(resized_imgs, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).to(divice)
    # Resize, and add a batch dimension (BCHW)

    return screen


def get_real_position(position, imsize, window, device):
    LEG_DOWN = 18

    VIEWPORT_W = 600
    VIEWPORT_H = 400

    SCALE = 30
    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE
    helipad_y = H / 4

    pos = position.cpu().numpy()

    trans = lambda pos:   [((pos[0] * VIEWPORT_W/SCALE/2) + VIEWPORT_W/SCALE/2) /SCALE * imsize[0],
                           pos[1] * (VIEWPORT_H / SCALE / 2) + (helipad_y + LEG_DOWN / SCALE) / SCALE * imsize[1]]


    position_ = np.round(np.apply_along_axis(trans, 1, pos)).astype(int)

    p_x = torch.from_numpy(position_[:, 0] - int(window/2)).to(device)
    p_y = torch.from_numpy(position_[:, 1] - int(window/2)).to(device)
    return p_x, p_y




def prepare_model_dir(work_dir):
    # Create results directory
    result_path = os.getcwd() + work_dir + '/' + strftime('%b_%d_%H_%M_%S', localtime())
    os.mkdir(result_path)

    return result_path



def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g" % x
    else: rep = str(x)
    return " " * (l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    cp = np.argmax(predictions.cpu().data.numpy(), 1)
    error = np.mean(cp != labels.cpu().data.numpy())
    return loss.data[0], error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(
        fmt_row(10, [
            epoch + 1, avg_loss / num_batches, avg_error / num_batches,
            time_duration
        ]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))
