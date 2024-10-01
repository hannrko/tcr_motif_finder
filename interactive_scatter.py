import matplotlib.pyplot as plt
import matplotlib as mpl

class InteractiveScatter:
    def __init__(self, x_lim=[None, None], y_lim=[None, None], axs_names=["X", "Y"]):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.axs_names = axs_names

    def x_lim_update(self, event_ax):
        self.x_lim = event_ax.get_xlim()

    def y_lim_update(self, event_ax):
        self.y_lim = event_ax.get_ylim()

    def zoom_plot(self, x, y, c, use_log=True, cmap="plasma"):
        fig, ax = plt.subplots(1, 1)
        if use_log:
            norm = mpl.colors.LogNorm()
        else:
            norm = None
        sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, alpha=0.8)
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_xlabel(self.axs_names[0])
        ax.set_ylabel(self.axs_names[1])
        # Get valid limits in case we close figure without changing axes
        self.x_lim = ax.get_xlim()
        self.y_lim = ax.get_ylim()
        fig.colorbar(sc)
        plt.grid()
        ax.callbacks.connect("xlim_changed", self.x_lim_update)
        ax.callbacks.connect("ylim_changed", self.y_lim_update)
        plt.show()
        return self.x_lim, self.y_lim
