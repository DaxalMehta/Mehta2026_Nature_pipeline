import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip

class Plotter:
    def __init__(self, figsize=(6, 6), xsize=1, ysize=1, fontsize=15, colorbar=False, bgcolor=None, ftcolor=None):
        self.fig, self.ax = plt.subplots(xsize, ysize, figsize=figsize, facecolor=bgcolor, edgecolor=ftcolor)
        self.figsize = figsize
        self.fontsize = fontsize
        self.config = {"bgcolor": bgcolor, "ftcolor": ftcolor}

        if bgcolor:
            self.ax.set_facecolor(bgcolor)
        if ftcolor:
            mpl.rc("axes", edgecolor=ftcolor)
            mpl.rcParams["text.color"] = ftcolor
            mpl.rcParams["axes.labelcolor"] = ftcolor
            mpl.rcParams["xtick.color"] = ftcolor
            mpl.rcParams["ytick.color"] = ftcolor
        
        plt.rcParams.update({"font.size": fontsize, "font.family":"serif"})

        self.legends = []

    def add_details(self, ax=None, xlabel='', ylabel='', xscale='linear', yscale='linear',
                xlim=None, ylim=None, colorbar=False, h=None, cbarlabel='',
                minorticks=True, legend=False, legendloc='upper right', grid=False, 
                invert_xaxis=False, invert_yaxis=False):
        if ax is None:
            # If multiple axes, apply to all
            axes = np.ravel(self.ax) if isinstance(self.ax, np.ndarray) else [self.ax]
        else:
            axes = [ax]

        for a in axes:
            a.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
            if xlim: a.set_xlim(xlim)
            if ylim: a.set_ylim(ylim)
            a.tick_params(axis='both', which='both', direction='in')
            if minorticks:
                a.minorticks_on()
            else:
                a.minorticks_off()
            a.xaxis.label.set_size(self.fontsize)
            a.yaxis.label.set_size(self.fontsize)

            if colorbar and h is not None:
                divider = make_axes_locatable(a)
                cax = divider.append_axes("right", size="5%", pad=0.0)
                cbar = self.fig.colorbar(h, cax=cax, label=cbarlabel)
                cbar.ax.set_aspect('auto')

            if legend:
                bgcolor = self.config.get("bgcolor", None)
                a.legend(loc=legendloc, frameon=False, facecolor=bgcolor)

            if grid:
                a.grid(True, zorder=0, alpha=0.5, which="both")
            if invert_xaxis:
                a.invert_xaxis()
            if invert_yaxis:
                a.invert_yaxis()
    
    def save(self, filename, pad=0.4, h_pad=None, w_pad=None, close=True, legend_loc="upper left"):
        self.fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if len(self.legends) > 0:
            self.ax.legend(handles=self.legends, frameon=False, loc=legend_loc, fontsize=12)
        self.fig.savefig(filename, dpi=300)
        if close:
            plt.close()
        print(f"Plot Created at loc: {filename}")
    
    def add_legends(self, type="line", color="", label="", linestyle="-", marker="o", markersize=5, lw=3):
        if type == "line":
            legend = Line2D([0], [0], color=color, lw=lw, markerfacecolor=color, label=label, linestyle=linestyle, markersize=markersize)
            self.legends.append(legend)
        if type == "scatter":
            legend = Line2D([0], [0], marker=marker, color=color, label=label, markerfacecolor=color, markersize=5, linestyle="None")
            self.legends.append(legend)
        if type == "patch":
            legend = Patch(facecolor="w", edgecolor=color, label=label, alpha=1.0, linestyle=linestyle, lw=lw)
            self.legends.append(legend)

    def create_colorbar(self, cmap, vmin, vmax, label="", pad=0.1):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = self.fig.colorbar(sm, ax=self.ax, pad=pad)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=self.fontsize)
        return cbar
        

class Video:
    def __init__(self, base):
        self.base = base

    def make_video(self, start_snap, end_snap, typ="mp4"):
        images = [imageio.imread(f"Random/DarkMatter_{i}_0_particle_position_x.png") for i in range(start_snap, end_snap)]
        #images = [imageio.imread(f"{self.base}DarkMatter_{i}_0_particle_position_x.png") for i in range(start_snap, end_snap)]      
        '''
        images = []
        for i in range(start_snap, end_snap):
            img_gas = imageio.imread(f"{self.base}HaloCenter_{i}_0_number_density.png")
            img_dm = imageio.imread(f"{self.base}DarkMatter_{i}_0_particle_mass.png")

            # Ensure same height before stacking
            h = min(img_gas.shape[0], img_dm.shape[0])
            img_gas = img_gas[:h, :min(img_gas.shape[1], img_dm.shape[1])]
            img_dm = img_dm[:h, :min(img_gas.shape[1], img_dm.shape[1])]

            # Side-by-side concat
            combined = np.concatenate((img_gas, img_dm), axis=1)
            images.append(combined)
        '''
        if typ == 'gif':
                imageio.mimsave(f"{self.base}Video.gif", images, duration=1, palettesize=256, subrectangles=True)
        elif typ == 'mp4':
                num_images = len(images)
                desired_duration = 20  # in seconds
                fps = num_images / desired_duration
                image_arrays = [np.array(img) for img in images]
                clip = ImageSequenceClip(image_arrays, fps=fps)
                clip.write_videofile(f"{self.base}Video.mp4", codec='libx264', fps=fps)


'''
images = []
        for i in range(len(imagesGas)):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(imagesGas[i])
            ax[0].axis("off")
            ax[1].imshow(imagesDM[i])
            ax[1].axis("off")
            plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0.0)
            # Convert figure to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            
            plt.close(fig)
'''