import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def compute_julia(width, height, c, max_iter=300, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)):
    """Compute the Julia set for a region of the complex plane."""
    x = np.linspace(xlim[0], xlim[1], width)
    y = np.linspace(ylim[0], ylim[1], height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    C = np.full(Z.shape, c)

    # Iteration counts
    iteration = np.zeros(Z.shape, dtype=float)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        diverged = np.abs(Z) > 2
        newly_diverged = diverged & mask
        iteration[newly_diverged] = i + 1 - np.log2(np.log2(np.abs(Z[newly_diverged])))
        mask &= ~diverged
        if not mask.any():
            break

    iteration[mask] = max_iter
    return iteration


def main():
    width, height = 600, 600
    c = complex(-0.4, 0.6)
    xlim = [-1.5, 1.5]
    ylim = [-1.5, 1.5]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    fractal = compute_julia(width, height, c, xlim=xlim, ylim=ylim)
    im = ax.imshow(fractal, cmap='magma', origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    fig.suptitle(f'c = {c.real:.3f} + {c.imag:.3f}i')

    # Slider axes
    ax_c_real = plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_c_imag = plt.axes([0.15, 0.05, 0.65, 0.03])

    c_real_slider = Slider(ax_c_real, 'c_real', -1.0, 1.0, valinit=c.real)
    c_imag_slider = Slider(ax_c_imag, 'c_imag', -1.0, 1.0, valinit=c.imag)

    def update(val):
        c_new = complex(c_real_slider.val, c_imag_slider.val)
        data = compute_julia(width, height, c_new, xlim=tuple(xlim), ylim=tuple(ylim))
        im.set_data(data)
        im.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])
        fig.suptitle(f'c = {c_new.real:.3f} + {c_new.imag:.3f}i')
        fig.canvas.draw_idle()

    c_real_slider.on_changed(update)
    c_imag_slider.on_changed(update)

    def zoom(frame):
        range_x = (xlim[1] - xlim[0]) * 0.97
        range_y = (ylim[1] - ylim[0]) * 0.97
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        xlim[0] = center_x - range_x / 2
        xlim[1] = center_x + range_x / 2
        ylim[0] = center_y - range_y / 2
        ylim[1] = center_y + range_y / 2

        c_current = complex(c_real_slider.val, c_imag_slider.val)
        data = compute_julia(width, height, c_current, xlim=tuple(xlim), ylim=tuple(ylim))
        im.set_data(data)
        im.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])
        return [im]

    # Store the animation object so it isn't garbage collected prematurely
    anim = FuncAnimation(fig, zoom, frames=200, interval=50, blit=False)

    # Keep a reference to the animation in the figure object for clarity
    fig.anim = anim

    plt.show()


if __name__ == '__main__':
    main()
