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
    n = len(batches)
    m = len(batches[0])
    fig, axes = plt.subplots(m, n)
    for i in range(n):
        images = batches[i].detach().cpu().numpy()
        for j in range(m):
            x = images[j]
            x[np.isnan(x)] = 0
            x = x.transpose([1, 2, 0])
            x = flow_vis.flow_to_color(x)
            axes[j,i].imshow(x)
            axes[j,i].title.set_text(captions[i])
    return fig

def color_fig(batches: list, captions):
    assert "sample" in captions, 'no sample in captions'
    fig, axes = plt.subplots(1, 1)
    for i in range(len(captions)):
        if captions[i] == 'sample':
            images = batches[i].detach().cpu().numpy()
            x = images[0]
            x[np.isnan(x)] = 0
            x = x.transpose([1, 2, 0])
            x = flow_vis.flow_to_color(x)
            axes.imshow(x)
            axes.title.set_text(captions[i])
            break
    return fig