from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def display_clip(clip):
    '''
    util method for displaying tensors as video

    expects clip.shape = (C,T,H,W)
    '''
    assert len(clip.shape) == 4, 'clip shape must be PyTorch Tensor of shape (C,T,H,W)'

    def update(frame_idx):
        ax.clear()
        ax.imshow(video_clip_np[frame_idx], cmap='gray')
        ax.axis('off')

    video_clip_np = clip.permute(1, 2, 3, 0).numpy()
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=range(video_clip_np.shape[0]), interval=60)
    plt.close()
    display(HTML(ani.to_html5_video()))