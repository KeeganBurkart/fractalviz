import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def compute_julia(width, height, c, max_iter=300):
    # Create complex plane
    x = np.linspace(-1.5, 1.5, width)
    y = np.linspace(-1.5, 1.5, height)
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

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    fractal = compute_julia(width, height, c)
    im = ax.imshow(fractal, cmap='magma', origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
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
        data = compute_julia(width, height, c_new)
        im.set_data(data)
        fig.suptitle(f'c = {c_new.real:.3f} + {c_new.imag:.3f}i')
        fig.canvas.draw_idle()

    c_real_slider.on_changed(update)
    c_imag_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
