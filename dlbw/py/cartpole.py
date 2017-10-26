from math import *
import numpy as np

# TODO: find source
def norm(angle):
    angle %= 2*pi
    if angle > pi:
        angle -= 2*pi
    elif angle < -pi:
        angle += 2*pi
    return angle


class InvertedPendulum:
    def __init__(self, translational, rotational, end=10., dt=.01):
        self.t = 0.0
        self.rotational = rotational
        self.translational = translational
        self.control = 0.0

        self.LENGTH = 0.5
        self.GRAVITY = 9.81
        self.MASS = 0.4
        self.dt = dt
        self.end = end

    def get_state(self):
        return np.hstack((self.translational, self.rotational))

    def set_state(self, s):
        self.translational = s[:2]
        self.rotational = s[2:]

    def systemEquation(self, y):
        theta = y[0]
        omega = y[1]
        dOmega = 1 / self.LENGTH * \
            ( \
              -self.GRAVITY * sin(theta) \
              - (self.control / self.MASS) * cos(theta) - self.dt * omega \
            )

        return np.array([omega, dOmega])

    def rungeKutta(self, y):
        k1 = self.systemEquation(y)
        k2 = self.systemEquation(y + self.dt / 2.0 * k1)
        k3 = self.systemEquation(y + self.dt / 2.0 * k2)
        k4 = self.systemEquation(y + self.dt * k3)

        return y + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def update(self, controlInput):
        controlInput *= 2
        self.t += self.dt
        # Bound control input
        self.control = clip(-20.0, controlInput, 20.0)
        # Prevent overflow (when training, some MLPs really don't work at all)
        self.translational[0] = clip(-1.0e300, self.translational[0], 1.0e300)
        self.translational[1] = clip(-1.0e300, self.translational[1], 1.0e300)

        self.translational[1] += (self.control / self.MASS) * self.dt
        self.translational[0] += self.translational[1] * self.dt

        self.rotational = self.rungeKutta(self.rotational)
        self.rotational[0] = norm(self.rotational[0])

    def integrate(self, control):
        x = []
        a = 0.
        from pendulum import act
        while self.t <= self.end:
            if act(int(self.t/self.dt)):
                a = control(np.array([self.get_state()]))
                print 't, s_=', self.t, self.get_state()
                print 'a =', a
            self.update(a[0,0])
            x.append([self.t] + list(self.get_state()))

        return np.array(x)

    def integrate_rnn(self, model, control):
        x = []
        a = 0.
        from pendulum_rnn import act, shift
        print (model.rnn_steps, model.s_dim)
        s = np.zeros((model.rnn_steps, model.s_dim))
        while self.t <= self.end:
            if act(int(self.t/self.dt)):
                s = shift(s, np.array(self.get_state()))
                a = control(np.array(np.expand_dims(s,0)))
                print 't, s_=', self.t, self.get_state()
                print 'a =', a[-1]
            self.update(a[-1])
            x.append([self.t] + list(self.get_state()))

        return np.array(x)


def clip(lo, x, hi):
    return lo if x <= lo else hi if x >= hi else x


from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
def animate_pendulum(t, states, length=.5, filename=None):
    """Animates the n-pendulum and optionally saves it to file.
    Courtesy of http://www.moorepants.info/blog/npendulum.html

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take some time.

    Returns
    -------
    fig : matplotlib.Figure
        The figure.
    anim : matplotlib.FuncAnimation
        The animation.

    """
    score = abs((states[:,-1]))
    # the number of pendulum bobs
    numpoints = 2

    # first set up the figure, the axis, and the plot elements we want to animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)

    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')

    # display the current time
    time_text = ax.text(.04, .9, '', transform=ax.transAxes)

    score_text = ax.text(.04, .8, '', transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
        cart_width, cart_height, fill=True, color='red', ec='black')
    ax.add_patch(rect)

    # blank line for the pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        score_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return time_text, rect, line,

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        score_text.set_text('score = {:2.2f}'.format(score[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = np.hstack((states[i, 0], np.zeros((numpoints - 1))))
        y = np.zeros((numpoints))
        for j in np.arange(1, numpoints):
            x[j] = x[j - 1] + length * np.cos(states[i, j] - pi/2)
            y[j] = y[j - 1] + length * np.sin(states[i, j] - pi/2)
        line.set_data(x, y)
        return score_text, time_text, rect, line,

    fps = 30
    # call the animator function
    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=t[0],
                                   frames=len(t))

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=1./t[0], codec='libx264')