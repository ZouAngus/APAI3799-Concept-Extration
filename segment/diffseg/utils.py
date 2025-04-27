#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment as LinearSumAssignment

def process_image(image_path):
    with open(image_path, "rb") as f:
      image = np.array(PIL.Image.open(f))
      s = tf.shape(image)
      w, h = s[0], s[1]   
      c = tf.minimum(w, h)
      w_start = (w - c) // 2
      h_start = (h - c) // 2
      image = image[w_start : w_start + c, h_start : h_start + c, :]
      image = tf.image.resize(image, (512, 512))
      return image

augmenter = keras.Sequential(
    layers=[
        tf.keras.layers.CenterCrop(512, 512),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)

def find_edges(M):
  edges = np.zeros((512,512))
  m1 = M[1:510,1:510] != M[0:509,1:510]
  m2 = M[1:510,1:510] != M[2:511,1:510]
  m3 = M[1:510,1:510] != M[1:510,0:509]
  m4 = M[1:510,1:510] != M[1:510,2:511]
  edges[1:510,1:510] = (m1 | m2 | m3 | m4).astype(int)
  x_new = np.linspace(0, 511, 512)
  y_new = np.linspace(0, 511, 512)
  x_new,y_new=np.meshgrid(x_new,y_new)
  x_new = x_new[edges==1]
  y_new = y_new[edges==1]
  return x_new,y_new

# def vis_without_label(M,image,index=None,save=False,dir=None,num_class=26):
#   fig = plt.figure(figsize=(20, 20))
#   ax = plt.subplot(1, 3, 1)
#   ax.imshow(image)
#   ax.set_title("Input",fontdict={"fontsize":30})
#   plt.axis("off")

#   x,y = find_edges(M)
#   ax = plt.subplot(1, 3, 2)
#   ax.imshow(image)
#   ax.imshow(M,cmap='jet',alpha=0.5, vmin=-1, vmax=num_class)
#   ax.scatter(x,y,color="blue", s=0.5)
#   ax.set_title("Overlay",fontdict={"fontsize":30})
#   plt.axis("off")

#   ax = plt.subplot(1, 3, 3)
#   ax.imshow(M, cmap='jet',alpha=0.5, vmin=-1, vmax=num_class),
#   ax.set_title("Segmentation",fontdict={"fontsize":30})
#   plt.axis("off")

#   if save:
#     fig.savefig(f".{dir}/example_{index}.jpg", format="jpg", bbox_inches="tight", dpi=96)
#     plt.close(fig)

def vis_without_label(M, image, index=None, save=False, dir=None, num_class=26):
    # Ensure the target directory exists
    import os
    if save and dir is not None and not os.path.exists(dir):
        os.makedirs(dir)
    if save and dir is not None and not os.path.exists(f"{dir}/segment"):
        os.makedirs(f"{dir}/segment")

    # Compute figsize in inches: 512 pixels / 96 dpi = 2.56 inches
    figsize = (512/96, 512/96)

    # Compute figsize in inches: 512 pixels / 96 dpi = 2.56 inches
    figsize = (512/96, 512/96)

    # -----------------
    # Save Input figure without margins
    fig1 = plt.figure(figsize=figsize, dpi=96)
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.imshow(image)
    ax1.axis("off")
    if save:
        fig1.savefig(f"{dir}/img.jpg", format="jpg", dpi=96)
    plt.close(fig1)

    # -----------------
    # Save Overlay figure without margins
    x, y = find_edges(M)
    fig2 = plt.figure(figsize=figsize, dpi=96)
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.imshow(image)
    ax2.imshow(M, cmap='jet', alpha=0.5, vmin=-1, vmax=num_class)
    ax2.scatter(x, y, color="blue", s=0.5)
    ax2.axis("off")
    if save:
        fig2.savefig(f"{dir}/segment/overlay.png", format="png", dpi=96)
    plt.close(fig2)

    # -----------------
    # Save Segmentation figure without margins
    fig3 = plt.figure(figsize=figsize, dpi=96)
    ax3 = fig3.add_axes([0, 0, 1, 1])
    ax3.imshow(M, cmap='jet', alpha=0.5, vmin=-1, vmax=num_class)
    ax3.axis("off")
    if save:
        fig3.savefig(f"{dir}/segment/segmentation.png", format="png", dpi=96)
    plt.close(fig3)


def semantic_mask(image, pred, label_to_mask, save_individual=False, out_dir=None):
    base_image = image.reshape(512, 512, -1)
    num_fig = len(label_to_mask)
    plt.figure(figsize=(20, 20))
    for i, label in enumerate(label_to_mask.keys()):
        bin_mask = np.zeros_like(base_image)
        flat_bin_mask = bin_mask.reshape(-1, bin_mask.shape[-1])
        for mask in label_to_mask[label]:
            flat_bin_mask[(pred.flatten() == mask), :] = 1
        colored_mask = base_image * bin_mask
        ax = plt.subplot(1, num_fig, i + 1)
        ax.imshow(colored_mask)
        ax.set_title(label, fontdict={"fontsize": 30})
        ax.axis("off")
        
        if save_individual and out_dir:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if not os.path.exists(f"{out_dir}/semantic_mask"):
                os.makedirs(f"{out_dir}/semantic_mask")
            
            # Determine file names based on the label
            if label == "background":
                semantic_filename = f"{out_dir}/semantic_mask/background.png"
                binary_filename = f"{out_dir}/background.png"
            else:
                semantic_filename = f"{out_dir}/semantic_mask/semantic_mask_{i}_{label}.png"
                binary_filename = f"{out_dir}/mask_{i}.png"
            
            # Save the colored mask
            colored_to_save = (colored_mask * 255).astype(np.uint8) if colored_mask.max() <= 1 else colored_mask.astype(np.uint8)
            bgr_image = cv2.cvtColor(colored_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(semantic_filename, bgr_image)
            
            # Save the binary mask
            binary_to_save = (bin_mask[:, :, 0] * 255).astype(np.uint8) if bin_mask[:, :, 0].max() <= 1 else bin_mask[:, :, 0].astype(np.uint8)
            cv2.imwrite(binary_filename, binary_to_save)
    
    plt.show()

def _fast_hist(label_true, label_pred, n_class):
    # Adapted from https://github.com/janghyuncho/PiCIE/blob/c3aa029283eed7c156bbd23c237c151b19d6a4ad/utils.py#L99
    pred_n_class = np.maximum(n_class,label_pred.max()+1)
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(pred_n_class * label_true[mask] + label_pred[mask],\
                       minlength=n_class * pred_n_class).reshape(n_class, pred_n_class)
    return hist

def hungarian_matching(pred,label,n_class):
  # X,Y: b x 512 x 512
  batch_size = pred.shape[0]
  tp = np.zeros(n_class)
  fp = np.zeros(n_class)
  fn = np.zeros(n_class)
  all = 0
  for i in range(batch_size):
    hist = _fast_hist(label[i].flatten(),pred[i].flatten(),n_class)
    row_ind, col_ind = LinearSumAssignment(hist,maximize=True)
    all += hist.sum()
    fn += (np.sum(hist, 1) - hist[row_ind,col_ind])
    tp += hist[row_ind,col_ind]
    hist = hist[:, col_ind] # re-order hist to align labels to calculate FP
    fp += (np.sum(hist, 0) - np.diag(hist))
  return tp,fp,fn,all