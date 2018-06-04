import matplotlib.pyplot as plt

def imshow(src, out):
  m = src.size(0)
  fig, axs = plt.subplots(2, m)
  for i in range(m):
    ax0 = axs[0, i]
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.imshow(src[i].numpy())
    ax1 = axs[1, i]
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.imshow(out[i])