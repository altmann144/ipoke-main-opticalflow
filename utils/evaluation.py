import numpy as np
import matplotlib.pyplot as plt
import flow_vis


def angular_error(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    u, v = pred[0], pred[1]
    u_gt, v_gt = gt[0], gt[1]
    AE = np.arccos((1. + u*u_gt + v*v_gt)/(np.sqrt(1.+u**2+v**2)*np.sqrt(1.+u_gt**2+v_gt**2)))
    return AE


def endpoint_error(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    u, v = pred[0], pred[1]
    u_gt, v_gt = gt[0], gt[1]
    EE = np.sqrt((u-u_gt)**2 + (v-v_gt)**2)
    return EE


def fig_matrix(batches: list, captions):
    # remember to call plt.close('all') in lightning module
    n = len(batches)
    m = len(batches[0])
    fig, axes = plt.subplots(m, n, sharex=True, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    for i in range(n):
        images = batches[i].detach().cpu().numpy()
        for j in range(m):
            x = images[j]
            x[np.isnan(x)] = 0
            x = x.transpose([1, 2, 0])
            x = flow_vis.flow_to_color(x)
            axes[j,i].imshow(x)

        axes[0,i].title.set_text(captions[i])
    return fig

def color_fig(batches: list, captions):
    fig, axes = plt.subplots(3, len(captions))
    for i in range(len(captions)):
        images = batches[i].detach().cpu().numpy()
        for j in range(3):
            x = images[j]
            x[np.isnan(x)] = 0
            x = x.transpose([1, 2, 0])
            x = flow_vis.flow_to_color(x)
            axes[j,i].imshow(x)
            axes[j,i].title.set_text(captions[i])
    return fig

def flow_batch_to_img_batch(batch):
    """ input torch tensor with shape BxCxHxW
    output: numpy array with shape BxCxHxW """
    batch = batch.transpose(1,3) # 0123 -> 0321 -> 0231
    batch = batch.transpose(1,2).cpu().numpy()

    shape = list(batch.shape)
    shape[-1] += 1
    out = np.zeros(shape, dtype=np.float32)
    for i in range(batch.shape[0]):
        batch[np.isnan(batch)] = 0
        out[i] = flow_vis.flow_to_color(batch[i])
    out = out.transpose([0,3,1,2])
    return out