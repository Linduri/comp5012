"""
Plot graphs
"""
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    """
    Plot graphs
    """
    def __init__(self):
        plt.rcParams["font.family"] = "Liberation Serif"
        # plt.rcParams["font.serif"] = "Serif"


    def __prep_graph__(self, _ax):
        _ax.minorticks_on()
        _ax.grid(color='white', ls = '-', lw = 2, which='major', alpha=0.5)
        _ax.grid(color='white', ls = 'dotted', lw = 1, which='minor', alpha=1)
        plt.setp(_ax.spines.values(), color="white")
        _ax.set_facecolor("#e7e7e7")

    def image(self, image, title, save_dir=None, show=True):
        _fig = plt.figure()
        _ax = _fig.subplots()
        _ax.imshow(image)
        plt.axis('off')
        plt.title(title)

        if save_dir:
            plt.savefig(save_dir + str(f"{title}.png").lower().replace(' ', '_'),
                    transparent=False,
                    facecolor='white',
                    bbox_inches="tight")

        if show:
            plt.show()

    def pareto_front_2d(self, _f1, _f2, save_dir=None, show=True):
        """
        Plot a 2D pareto front.
        """
        print("Plotting 2D Pareto front...")
        _fig = plt.figure()
        _ax = _fig.subplots()
        self.__prep_graph__(_ax)
        _ax.scatter(_f1, _f2, c ="blue")
        plt.title("Pareto front")
        plt.xlabel(r"$F_1$")
        plt.ylabel(r"$F_2$")

        if save_dir:
            plt.savefig(save_dir + "pareto_front_2d.png",
                    transparent=False,
                    facecolor='white',
                    bbox_inches="tight")

        if show:
            plt.show()
    
    def normalised_pareto_front_3d(self, _f_history, save_dir=None, show=True):
        """
        Plot a 3D pareto front.
        """
        print("Plotting 3D Pareto front...")
        # Scale each pop score between zero and one
        _points = []
        for idx, _generation in enumerate(_f_history):
            _norm_gen = _generation

            for col in range(_generation.shape[1]):
                _norm_gen[:, col] = np.interp(_norm_gen[:, col],
                (_norm_gen[:, col].min(), _norm_gen[:, col].max()), (0, 1))

            for _member in _norm_gen:
                _points.append([_member[0], _member[1], idx])

        _verts = np.array(_points)

        # Remove double
        _verts = np.unique(_verts, axis=0)

        _pareto_f1 = _verts[:,0]
        _pareto_f2 = _verts[:,1]
        _pareto_generation = _verts[:,2]

        _fig = plt.figure()
        _ax = _fig.add_subplot(111, projection='3d')
        _ax.set_box_aspect(aspect=None, zoom=0.8)

        _ax.plot_trisurf(_pareto_f1, _pareto_generation, _pareto_f2, linewidth=0)

        _ax.invert_yaxis()

        _ax.set_xlabel(r"$F_1$")
        _ax.set_ylabel("Generation")
        _ax.set_zlabel(r"$F_2$")

        _ax.set_title("Normalised Pareto front over generations")

        if save_dir:
            plt.savefig(save_dir + "pareto_front_3d.png",
                    transparent=False,
                    facecolor='white',
                    bbox_inches="tight")

        if show:
            plt.show()

    def hyper_volume_2d(self, _f, save_dir=None, show=True):
        """
        Plot a hypervolume
        """
        # https://stackoverflow.com/questions/42692921/how-to-create-hypervolume-and-surface-attainment-plots-for-2-objectives-using

        print("Plotting hyper-volume...")
        _f = _f[_f[:, 0].argsort()]

        _fig = plt.figure()
        _ax = _fig.subplots()
        self.__prep_graph__(_ax)


        # Plot points
        plt.scatter(_f[:,0], _f[:,1], c="blue")

        # Find corners
        _corners = []
        for _idx in range(_f.shape[0]-1):
            _corners.append((_f[_idx,0], _f[_idx+1,1]))
        
        _corners = np.array(_corners)

        # Interleave points and corners to plot a line
        _line = np.empty((_f.shape[0]+_corners.shape[0],_f.shape[1]))
        _line[::2,:] = _f
        _line[1::2,:] = _corners
        plt.plot(_line[:,0], _line[:,1])

        # Generate a plot ref point an enlcosing line
        reference_point = [_f[:,0].max(), _f[:,1].max()]
        _ref_line = np.array([_f[0], reference_point, _f[-1]])
        plt.plot(_ref_line[:,0], _ref_line[:,1])
        plt.scatter(reference_point[0], reference_point[1], c="red")

        _ref_line_y = np.array([reference_point[1] for _ in range(_line.shape[0])])
        plt.fill_between(_line[:,0], _line[:,1], _ref_line_y, color="grey", alpha=0.3)

        # Find the area of the hypervolume
        _area_under_the_curve = np.trapz(x=_line[:,0], y=_line[:,1])
        _origin = np.array([_f[:,0].min(), _f[:,1].min()])
        _frame_size = np.array(reference_point)-_origin
        _hyper_volume = (_frame_size[0]*_frame_size[1]) - _area_under_the_curve

        plt.title(f"Hyper-volume ({_hyper_volume})")
        plt.xlabel(r"$F_1$")
        plt.ylabel(r"$F_2$")
        
        if save_dir:
            plt.savefig(save_dir + "hypervolume.png",
                    transparent=False,
                    facecolor='white',
                    bbox_inches="tight")

        if show:
            plt.show()
