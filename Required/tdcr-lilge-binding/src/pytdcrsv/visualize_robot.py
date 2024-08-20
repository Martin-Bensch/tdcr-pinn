# Bachelorarbeit von Jakob Wenner, Betreut von Martin Bensch am imes (LUH 2021)
import pytdcrsv.config as config
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.spatial.transform import Rotation


# Create a custom logger
logger = config.logging.getLogger(__name__)
# Add handlers to the logger
logger.addHandler(config.C_HANDLER)
logger.addHandler(config.F_HANDLER)


class VisualizeRobot:
    """
    Class to visualize a robot model.

    Args:
        robot: a StaticRobotModel instance
        frames: frames to traw
    """
    figure = 0
    def __init__(self, robot=None, frames=None):
        self._robot = None
        if robot is not None:
            self._robot = robot
            self.frames_df = robot.frames_df
        elif frames is not None:
            self.frames_df = frames
        else:
            logger.info("No robot model given.")

        # Figure
        self.fig = None
        self.ax = None
        self.rob_ani = None
        self._x_max = 0
        self._y_max = 0
        self._z_max = 0
        self._x_min = 0
        self._y_min = 0
        self._z_min = 0
        self.fig_no = VisualizeRobot.figure + 1

    def draw_robot(self, indices: [int] = None,
                   n_points: int = 16, scale_disk=5, animate=True,
                   figsize: tuple = (2.5, 2.5)):
        """
        Draw a full robot to the given indices

        Args:
            indices: index/indices in data frame for the requested pose/s
            n_points: Number of points to draw each disk
            scale_disk: Scales the disk radius
            animate: If the trajectory should be animated or each
            configuration should be drawn individually
        """
        if indices is None:
            # if no indices are specified, plot all poses
            shape = self.frames_df.shape
            indices = list(range(shape[0]))

        if isinstance(indices, int):
            indices = [indices]
        elif not isinstance(indices, list):
            raise ValueError(f"List or integer expected, but got "
                             f"{type(indices)} ")

        self._create_figure(figsize=figsize)

        if self._robot is not None:
            disk_radius__m = self._robot.pradius_disks__m[0] * scale_disk
        else:
            disk_radius__m = 0.01 * scale_disk
        # plot base

        # Access data frame entries and plot
        colors = ['b', "g", "r", "c", "m", "k", "y"]
        #markers = [".", "o", "v", "s", "*", "+", "x"]
        points_dct = {}
        unsuccesful_indices = []
        for e1, idx in enumerate(indices):
            # Get row
            row = self.frames_df.iloc[idx]
            if row.loc["success"]:
                number_of_frames = row.size - 1 # subtract 1 entry for actuation
                end_1st_segment = number_of_frames / 2
                frames = row.loc["disk_00":"ee"]
                try:
                    m = ""
                    c = colors[e1]
                except IndexError:
                    c = colors[5]
                points_pose_idx = []
                for e2, f in enumerate(frames):
                    if e2 > end_1st_segment - 1:
                        m = "+"
                    f = f.reshape((1, 7))
                    try:
                        if animate:
                            if idx == 26:
                                asd = 1
                            # Points for one frame
                            points = self._draw_circle(svec=f,
                                               n_points=n_points,
                                               disk_radius__m=disk_radius__m,
                                               color=c,
                                               marker=m)
                            points_pose_idx.append(points)
                        else:
                            _ = self._draw_circle(svec=f,
                                               n_points=n_points,
                                               disk_radius__m=disk_radius__m,
                                               color=c,
                                               marker=m,
                                               animate=False)
                    except:
                        a = 1

                points_dct[idx] = points_pose_idx
                if animate and len(indices) - 1 == idx:
                    for ui in unsuccesful_indices:
                        indices.remove(ui)
                    return points_dct

            else:
                logger.info("Skip unsuccessful computation")
                logger.info(f"    Remove index {idx} from indices list.")
                unsuccesful_indices.append(idx)

       # self._ax.set_aspect(1, adjustable='datalim', anchor="C")


    def animate_robot(self, indices: [int] = None,
                      n_points: int = 16, scale_disk=5,
                      interval=100, figsize=(2.5, 2.5)):
        """
        Animates a given robot model. See jupyter lab for usage.
        Args:
            indices: Indices of all frames to draw
            n_points: Amount of points to represent one disk
            scale_disk: Scale disk
            interval: update interval of the animation
            figsize: figure size

        Returns:
            Animation object, figure handle, axis handle
        """
        # Create data
        # data is a dictionary, where each entry holds all points for each
        # disk frame per pose
        if indices is None:
            indices = [idx for idx in range(len(self.frames_df))]
        data = self.draw_robot(indices=indices, n_points=n_points,
                                scale_disk=scale_disk, figsize=figsize)

        # Create line objects for multi color animation
        plotlays, plotcols = [2], ["black", "red"]
        lines = []
        for index in range(2):
            lobj = self.ax.plot([], [], [], lw=2, color=plotcols[index])[0]
            lines.append(lobj)

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        def animate(n, lines, data):
            # Plot all disk frames

            XYZ = np.concatenate(data[n], axis=1)
            segment_1 = int((XYZ.shape[1] + 1) / 2)

            lines[0].set_data(XYZ[0,:segment_1], XYZ[1, :segment_1])
            lines[0].set_3d_properties(list(XYZ[2, :segment_1]))

            lines[1].set_data(list(XYZ[0, segment_1:]), list(XYZ[1, segment_1:]))
            lines[1].set_3d_properties(list(XYZ[2, segment_1:]))

            return lines

        dx = self._x_max - self._x_min
        dy = self._y_max - self._y_min
        dz = self._z_max - self._z_min

        dx_dz = abs(dx / dz)
        dy_dz = abs(dy / dz)

        self.ax.set_xlim(self._x_min, self._x_max)
        self.ax.set_ylim(self._y_min, self._y_max)
        self.ax.set_zlim(self._z_min, self._z_max)
        self.ax.set_aspect('auto', anchor="C")
        self.ax.set_box_aspect([dx_dz, dy_dz, 1])

        self.rob_ani = animation.FuncAnimation(self.fig, animate,
                                               init_func=init,
                                               fargs=(lines, data),
                                               frames=indices, interval=interval,
                                               repeat_delay=2,
                                               blit=True
                                               )

        return self.rob_ani, self.fig, self.ax

    @staticmethod
    def svec_to_matrix(svec: np.array = np.zeros((7, 1))):
        """
        Computes the transformation matrix for a given state vector [position, quaternion]

        Args:
            svec: state vector [position, quaternion]

        Returns:
            mat: transformation matrix to the given state vector svec
        """
        # Check svec shape
        if svec.shape != (1, 7):
            raise ValueError(f"Wrong svec has wrong shape, is {svec.shape} but expect (1, 7)")

        # Access elements
        t = svec[0, 0:3]

        # quaternion, the Rotation.from_quat method expects the quaternion in scaler LAST format!
        q = np.concatenate([svec[0, 4:7], np.array([svec[0, 3]])])

        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(q)

        mat = np.eye(4)
        mat[0:3, 0:3] = rot.as_matrix()
        mat[0:3, 3] = t

        return mat

    def _disk_in_base_frame(self, svec, n_points: int = 16,
                            disk_radius__m: float = 0.002):
        """
        For a given state vector, calculate disk in base frame

        Args:
            svec: state vector [pos, quaternion], quaternion in scalar first ordering
            n_points: number of points for representing the disk
            disk_radius__m: disk radius in [mm]
        Returns:
                p_disk: list, disk points in base frame
        """
        mat = VisualizeRobot.svec_to_matrix(svec)

        points = np.zeros((3, n_points + 1))
        for n in range(n_points + 1):
            phi = n * (2 * np.pi / n_points)

            x = np.cos(phi) * disk_radius__m
            y = np.sin(phi) * disk_radius__m

            pn = mat @ np.array([[x],
                                 [y],
                                 [0],
                                 [1]
                                 ])
            self._max_min_points(pn)
            points[:, n] = pn[0:3, 0]


        return points

    def _max_min_points(self, pn):
        if pn[0] > self._x_max:
            self._x_max = pn[0]

        if pn[1] > self._y_max:
            self._y_max = pn[1]

        if pn[2] > self._z_max:
            self._z_max = pn[2]

        if pn[0] < self._x_min:
            self._x_min = pn[0]

        if pn[1] < self._y_min:
            self._y_min = pn[1]

        if pn[2] < self._z_min:
            self._z_min = pn[2]

    def _draw_circle(self, svec, n_points=16, disk_radius__m=0.002,
                     color='b', marker="+", animate=True):
        """
        Draw a single circle.

        Args:
            svec: state vector as [pos, quaternion]. Quaternion in scalar first ordering
            n_points: number of points to represent the disk
            disk_radius__mm: disk radius in [mm]

        Returns:

        """

        # Calculate points for svec
        points = self._disk_in_base_frame(svec=svec, n_points=n_points,
                                          disk_radius__m=disk_radius__m
                                          )
        # Draw points
        if not animate:
            self.ax.plot3D(
                            points[0, :],
                            points[1, :],
                            points[2, :],
                            linewidth=2, c=color, marker=marker
                            )
        return points

    def _draw_base(self, n_points=16, disk_radius__mm=0.002):

        mat = np.eye(4)

        points = np.zeros((3, n_points + 1))
        for n in range(n_points + 1):
            phi = n * (2 * np.pi / n_points)

            x = np.cos(phi) * disk_radius__mm
            y = np.sin(phi) * disk_radius__mm

            pn = mat @ np.array([[x],
                                 [y],
                                 [0],
                                 [1]
                                 ])
            points[:, n] = pn[0:3, 0]

            # Draw points
            self.ax.plot3D(
                points[0, :],
                points[1, :],
                points[2, :],
                linewidth=5,
                color='k'
                )

    def _create_figure(self, figsize=(2.5, 2.5)):
        if self.fig is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = plt.axes(projection='3d')

            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
            #self._ax.set_zlim([0, np.sum(l)])
            #plt.xlim([-np.sum(l), np.sum(l)])
            #plt.ylim([-np.sum(l), np.sum(l)])

    def show(self):
        """
        Set ratio to x,y,z min/max values and scale axes appropriately.
        """
        dx = self._x_max - self._x_min
        dy = self._y_max - self._y_min
        dz = self._z_max - self._z_min

        dx_dz = abs(dx / dz)
        dy_dz = abs(dy / dz)
        self.ax.set_aspect('auto', anchor="C")
        self.ax.set_box_aspect([dx_dz, dy_dz, 1])


def draw_full_robot():
    # Create robot
    segment_length__m = 0.2
    f__n = np.array([0, 0, 0])
    l__Nm = np.array([0, 0, 0])
    tendon_radius__m = 6 * 1e-3
    i_m__per_m4 = np.pi * (0.5 * 1e-3) ** 4 / 4
    e__n_per_m2 = 80 * 1e9
    mass_discs__kg = np.array([0.01, 0.01, 0.01])
    g__m_per_s2 = 9.81
    modelling_approach_vc = "VC"
    #modelling_approach_cc = "CC"

    robot_vc = StaticRobotModel(segment_length__m=segment_length__m,
                                f__n=f__n,
                                l__Nm=l__Nm,
                                modelling_approach=modelling_approach_vc)

    # Actuate robot
    q1 = np.array([5, 0, 0, 5, 0, 0]).reshape((6, 1))
    q2 = np.array([3, 0, 3, 2, 0, 2]).reshape((6, 1))
    q3 = np.array([2, 2, 0, 2, 2, 0]).reshape((6, 1))
    q = np.concatenate((q1, q2, q3), axis=0)
    success_vc = robot_vc.calc_pose_from_ctrl(act=q)
    print(f"Calulation suceesfull: {success_vc}")

    # Visualilze robot
    vrob = VisualizeRobot(robot=robot_vc)
    vrob.draw_robot(indices=None, animate=False)

    vrob._show()


def animation_example():
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from numpy import random

    fig = plt.figure()
    ax1 = plt.axes(xlim=(-108, -104), ylim=(31, 34))
    line, = ax1.plot([], [], lw=2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plotlays, plotcols = [2], ["black", "red"]
    lines = []
    for index in range(2):
        lobj = ax1.plot([], [], lw=2, color=plotcols[index])[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    plt.close()
    x1, y1 = [], []
    x2, y2 = [], []

    # fake data
    frame_num = 100
    gps_data = [-104 - (4 * random.rand(2, frame_num)),
                31 + (3 * random.rand(2, frame_num))]

    def animate(i):

        x = gps_data[0][0, i]
        y = gps_data[1][0, i]
        x1.append(x)
        y1.append(y)

        x = gps_data[0][1, i]
        y = gps_data[1][1, i]
        x2.append(x)
        y2.append(y)

        xlist = [x1, x2]
        ylist = [y1, y2]

        # for index in range(0,1):
        for lnum, line in enumerate(lines):
            line.set_data(xlist[lnum],
                          ylist[lnum])  # set data for each line separately.

        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frame_num, interval=10, blit=True)


def animation_robot():
    segment_length__m: float = 0.2
    f__n: np.ndarray = np.array([0, 0, 0])
    l__Nm: np.ndarray = np.array([0, 0, 0])
    youngs_modulus__n_per_m2: float = 54e9
    pradius_disks__m: np.ndarray = np.array([0.01, 0.01])
    ro__m: float = .7 * 1e-3
    modelling_approach: str = "VC"

    robot_vc = StaticRobotModelExtended(modelling_approach=modelling_approach)

    # Actuate robot
    q1 = np.array([4.,0.,4.,2.,0.,2.]).reshape((6,1))
    q2 = np.array([4.01047688e+00, 0.0, 1.53317358e-02, 2.00861277e+00,
    0.0, 3.71839449e-03]).reshape((6, 1))
    q3 = np.array([4.06259725e+00, 0.00000000e+00, 1.18121458e-04,
                   2.06077300e+00,
 1.40902656e-03, 0.00000000e+00]).reshape((6,1))

    q = np.concatenate([q1, q2])

    success_vc = robot_vc.calc_pose_from_ctrl(act=q)
    print(f"Calulation suceesfull: {success_vc}")
    for q_ in q:
        print(q_)
    # Visualilze robot
    vrob = VisualizeRobot(robot=robot_vc)
    #anim = vrob.animate_robot()
    vrob.draw_robot(animate=False)
    #vrob.show()
    plt.show()
    a= 1


def animate_pose(robot_instance, actuations):
    act_len = int(len(actuations) / 6)
    success_vc = robot_instance.calc_pose_from_ctrl(act=actuations)
    print(success_vc)
    vrob = VisualizeRobot(robot=robot_instance)
    vrob.draw_robot(animate=False)
    plt.show()


if __name__ == "__main__":
    from methodspaper.simulation.static_robot_model import StaticRobotModel
    from methodspaper.simulation.static_robot_model_extended \
        import StaticRobotModelExtended
    animation_robot()
    #animation_example()