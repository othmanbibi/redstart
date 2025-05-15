import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'équilibre corresepond aux équations : 

    1- $\dot{x} = 0$

    2-$\ddot{x} = (-f/M)sin(\theta + \phi) = 0$

    3-$\dot{y} = 0$

    4-$\ddot{y} = (f/M)cos(\theta + \phi) - g = 0$

    5-$\dot{\theta} = 0$

    6-$\ddot{\theta} = (-l*f/J)sin(\phi) = 0$

    D'après 6 on a: $\phi = 0 \mod 2\pi$, Or puisque $|\phi| < \pi/2$ alors $\phi=0$

    D'après 2 et $\phi=0$ on a: $\theta = 0 \mod 2\pi$, Or puisque $|\theta| < \pi/2$ alors $\theta=0$

    Finalement, d'après 4 et $\phi=\theta=0$, on a : $f = Mg$ 

    Conclusion : 

    * $\theta = 0$
    * $\phi = 0$
    * $f = Mg$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On définit le statut de l'équilibre par le point : $(x_e, y_e, \theta_e)$ et les entrées qu'on doit avoir pour atteindre l'équilibre $(f_e, \phi_e)$

    On définit les erreurs par rapport à l'équilibre : 

    $$
    \begin{aligned}
    \Delta x &= x - x_e \\
    \Delta y &= y - y_e \\
    \Delta \theta &= \theta - \theta_e \\
    \Delta f &= f - f_e \\
    \Delta \phi &= \phi - \phi_e
    \end{aligned}
    $$

    Nos équations différentielles : 

    1. $M \ddot{x} = -f \sin (\theta + \phi)$
    2. $M \ddot{y} = f \cos(\theta + \phi) - Mg$
    3. $J \ddot{\theta} = - \ell (\sin \phi) f$



    1- On linearise la première équation : 

    $$
    M \ddot{x} = -(f_e + \Delta f) \sin (\theta_e + \Delta \theta + \phi_e + \Delta \phi)
    $$

    On utilise la propriété $\sin(a+b)=\sin(a)\cos(b) + \sin(b)\cos(b)$ 

    et l'approximation $\sin(a)\approx a$ et $\cos(a)\approx 1$ si a est proche de 0

    Alors : 

    $$
    \begin{aligned}
    \sin(\theta_e + \Delta \theta + \phi_e + \Delta \phi) 
    &= \sin(\theta_e + \phi_e) \cos(\Delta \theta + \Delta \phi) + \cos(\theta_e + \phi_e) \sin(\Delta \theta + \Delta \phi) \\
    &\approx \sin(\theta_e + \phi_e) + \cos(\theta_e + \phi_e)(\Delta \theta + \Delta \phi)
    \end{aligned}
    $$

    Alors : 

    $$
    \begin{aligned}
    M \ddot{x} &\approx -(f_e + \Delta f)\left[\sin(\theta_e + \phi_e) + \cos(\theta_e + \phi_e)(\Delta \theta + \Delta \phi)\right] \\
    M \Delta \ddot{x} &\approx -f_e \cos(\theta_e + \phi_e)(\Delta \theta + \Delta \phi) - \sin(\theta_e + \phi_e) \Delta f
    \end{aligned}
    $$

    (On utilise : $M \ddot{x}_e = -f_e \sin(\theta_e + \phi_e) = 0$)

    ---
    2- On linearise la deuxième équation : 

    $$
    M \ddot{y} = (f_e + \Delta f) \cos(\theta_e + \Delta \theta + \phi_e + \Delta \phi) - Mg
    $$

    De même façon : 

    $$
    \begin{aligned}
    \cos(\theta_e + \Delta \theta + \phi_e + \Delta \phi) 
    &= \cos(\theta_e + \phi_e)\cos(\Delta \theta + \Delta \phi) - \sin(\theta_e + \phi_e)\sin(\Delta \theta + \Delta \phi) \\
    &\approx \cos(\theta_e + \phi_e) - \sin(\theta_e + \phi_e)(\Delta \theta + \Delta \phi)
    \end{aligned}
    $$

    Alors : 

    $$
    \begin{aligned}
    M \ddot{y} &\approx (f_e + \Delta f)\left[\cos(\theta_e + \phi_e) - \sin(\theta_e + \phi_e)(\Delta \theta + \Delta \phi)\right] - Mg \\
    M \Delta \ddot{y} &\approx -f_e \sin(\theta_e + \phi_e)(\Delta \theta + \Delta \phi) + \cos(\theta_e + \phi_e)\Delta f
    \end{aligned}
    $$

    (On utilise : $M \ddot{y}_e = f_e \cos(\theta_e + \phi_e) - Mg = 0$)

    ---

    3 -Pour la troisième équation : 

    $$
    J \ddot{\theta} = -\ell (f_e + \Delta f) \sin(\phi_e + \Delta \phi)
    $$

    Avec : 

    $$
    \sin(\phi_e + \Delta \phi) \approx \sin \phi_e + \cos \phi_e \Delta \phi
    $$

    Alors : 

    $$
    \begin{aligned}
    J \ddot{\theta} &\approx -\ell (f_e + \Delta f)(\sin \phi_e + \cos \phi_e \Delta \phi) \\
    J \Delta \ddot{\theta} &\approx -\ell f_e \cos \phi_e \Delta \phi - \ell \sin \phi_e \Delta f
    \end{aligned}
    $$

    (On utilise : $J \ddot{\theta}_e = -\ell f_e \sin \phi_e = 0$)

    ---

    Conclusion : 

    Avec  : 

    * $\theta_e = 0$
    * $\phi_e = 0$
    * $f = Mg$

    $$
    \begin{aligned}
    \Delta \ddot{x} &= - g  (\Delta \theta +  \Delta \phi)  \\
    \Delta \ddot{y} &=  \frac{1}{M} \Delta f \\
    \Delta \ddot{\theta} &= - \frac{\ell Mg}{J} \Delta \phi 
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le système est représenté par :

    $$
    \dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}
    $$

    où :

    - $\mathbf{x}$ : Vecteur d'état  
    - $\mathbf{u}$ : Vecteur d'entrée  
    - $A$ : Matrice d'état  
    - $B$ : Matrice d'entrée

    $$
    x
    = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    $$

    $$
    u
    = 
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi 
    \end{bmatrix}
    $$


    Alors : 

    $$
    A = \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    et : 

    $$
    B = \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{lMg}{J}\\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Étant donné la matrice :

    \[
    A = 
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0 
    \end{bmatrix}
    \]

    ### Analyse de stabilité

    - La matrice \( A \) possède au moins une valeur propre nulle (\( \lambda = 0 \))  ` dont la partie réelle n'est pas strictement négative`
    - Par conséquent, le système linéaire \( \dot{x} = A x \) **n'est pas asymptotiquement stable**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    on a 

    $$
    x = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \Rightarrow n = 6
    $$


    Le système est contrôlable si le rang de la matrice de contrôlabilité est égal à 6 :

    $$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & A^3B & A^4B & A^5B \end{bmatrix}$$

    On le calcule numériquement par le code suivant :
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from sympy import Matrix, symbols


    _g, _M, _l, _J = symbols('g M l J')


    _A = Matrix([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -_g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    _B = Matrix([
        [0, 0],
        [0, -_g],
        [0, 0],
        [1/_M, 0],
        [0, 0],
        [0, -_l*_M*_g/_J]
    ])


    _C = _B
    for i in range(1, 6):
        _C = _C.row_join(_A**i * _B)


    print(f"Rank: {_C.rank()}")
    _C

    return Matrix, symbols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Alors le système est contrôlable""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On ignore $y$ et $\dot{y}$,
    et on supprime $\Delta \phi$ de l’entrée.

    1. $\Delta \ddot{x} = -g(\Delta \theta + \Delta \phi)$
    2. $\Delta \ddot{\theta} = -\dfrac{\ell M g}{J} \Delta \phi$
    Alors, on retrouve :

    $$
    \Delta x =
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}, \quad u = \Delta \phi
    $$


    Le système devient :

    $$
    \dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}
    $$

    avec :

    $$
    A = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B = \begin{bmatrix}
    0 \\
    -g \\
    0 \\
    - \dfrac{\ell M g}{J}
    \end{bmatrix}
    $$

    **Pour la contrôlabilité**

    De même que dans la question précédente, on calcule le rang de la matrice de contrôlabilité numériquement.

    Il faut trouver que le rang est égal à 4 :

    $$
    \mathcal{C} = [B, AB, A^2B, A^3B]
    $$

    pour que le system soit controlable
    """
    )
    return


@app.cell(hide_code=True)
def _(Matrix, symbols):

    _g, _M, _l, _J = symbols('g M l J')

    # Define A and B matrices
    _A = Matrix([
        [0, 1, 0, 0],
        [0, 0, -_g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    _B = Matrix([
        [0],
        [-_g],
        [0],
        [-_l*_M*_g/_J]
    ])

    # Compute controllability matrix: [B, AB, A^2B, A^3B]
    _C = _B
    for _i in range(1, 4):
        _C = _C.row_join(_A**_i * _B)

    # Display rank and controllability matrix
    print(f"Rank: {_C.rank()}")
    _C
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Alors le systeme est controlable""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On considère une chute libre sans poussée, alors la force est nulle :

    $$
    f(t) = 0, \quad \phi(t) = 0
    $$



    Les équations  devient

    $$
    \begin{aligned}
    \Delta \ddot{x}(t) &= 0 \\
    \Delta \ddot{y}(t) &= -g \\
    \Delta \ddot{\theta}(t) &= 0
    \end{aligned}
    $$



    Conditions Initiales

    On considère le cas suivant :

    $$
    \begin{aligned}
    \Delta x(0) &= 0 \\
    \Delta \dot{x}(0) &= 0 \\
    \Delta y(0) &= 10 \\
    \Delta \dot{y}(0) &= 0 \\
    \Delta \theta(0) &= \frac{\pi}{4} \quad \text{(soit 45°)} \\
    \Delta \dot{\theta}(0) &= 0
    \end{aligned}
    $$

    Solutions Analytiques

    Les équations précédentes ont pour solutions :

    $$
    \begin{aligned}
    \Delta x(t) &= 0 \\
    \Delta y(t) &= -\frac{1}{2} g t^2 + 10 \\
    \Delta \theta(t) &= \frac{\pi}{4}
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):



    _g = 1.0                      
    _theta0 = np.pi / 4          
    _y0 = 10                   
    _vy0 = 0                     


    _t = np.linspace(0, 5, 1000)  


    _y_t = -0.5 * _g * _t**2 + _vy0 * _t + _y0      
    _theta_t = np.ones_like(_t) * _theta0        


    plt.figure(figsize=(12, 4))


    plt.subplot(1, 2, 1)
    plt.plot(_t, _y_t, label=r"$y(t)$", color="blue")
    plt.xlabel("Temps $t$ (s)")
    plt.ylabel("Hauteur $y$ (m)")
    plt.title("Hauteur en chute libre")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(_t, _theta_t, label=r"$\theta(t)$", color="orange")
    plt.xlabel("Temps $t$ (s)")
    plt.ylabel("Inclinaison $\\theta$ (rad)")
    plt.title("Inclinaison constante")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    En chute libre :

    Pas d’accélération horizontale, donc la position latérale ne bouge pas.

    La gravité fait tomber le booster, sa hauteur diminue en suivant une parabole.

    L’angle d’inclinaison reste fixe, car rien ne le fait tourner.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On veut trouver K tele que :

    $$
    u(t) = \Delta \phi(t) = -K \mathbf{x}(t)
    \quad \text{with} \quad
    K = \begin{bmatrix}
    0 & 0 & k_3 & k_4
    \end{bmatrix}
    $$


    On écrit :

    $$
    u(t) = -k_3 \Delta \theta - k_4 \Delta \dot{\theta}
    $$



    On a d'après 'Linearized Model' :

    $$
    \Delta \ddot{\theta} = -\frac{\ell M g}{J} \Delta \phi = -\alpha \Delta \phi
    $$

    On remplace cette formule dans l'équation :

    $$
    \Delta \phi = -k_3 \Delta \theta - k_4 \Delta \dot{\theta}
    \Rightarrow
    \Delta \ddot{\theta} = \alpha (k_3 \Delta \theta + k_4 \Delta \dot{\theta})
    $$

    Ce qui donne :

    $$
    \Delta \ddot{\theta} - \alpha k_4 \Delta \dot{\theta} - \alpha k_3 \Delta \theta = 0
    $$

    C'est une **Equation de deuxième degrée** de la forme :

    $$
    \Delta \ddot{\theta} + 2\xi \omega_n \Delta  \dot{\theta} + \omega_n^2 \Delta \theta = 0
    $$

    Avec :

    * $2\xi \omega_n = -\alpha k_4$
    * $\omega_n^2 = -\alpha k_3$


    On veut :

    * $\Delta \theta(t) \to 0$ in < 20s
    * $|\Delta \theta(t)| < \pi/2$
    * $|\Delta \phi(t)| < \pi/2$


    Et notre condition initiale est :

    $$
    \Delta \theta(0) = \frac{45}{180}\pi = \frac{\pi}{4} \approx 0.785 \text{ rad}
    $$



    Nous voulons :

    * Une décroissance suffisamment rapide (en moins de 20 s) → peut-être $\omega_n \approx 0{,}5$ rad/s (→ temps de montée ≈ 4s)

    * Un amortissement critique ou légèrement sous-amorti : $\xi \approx 0{,}8$



    On peut choisir:

    * $\omega_n = 0.5 \Rightarrow \omega_n^2 = 0.25$
    * $\xi = 0.8 \Rightarrow 2\xi\omega_n = 0.8$

    Alors :

    * $\alpha k_3 = -0.25 \Rightarrow k_3 = -0.25 / \alpha$
    * $\alpha k_4 = -0.8 \Rightarrow k_4 = -0.8 / \alpha$

    Pour $\alpha = 3$:

    $$
    k_3 = -1/12,\quad k_4 = -4/15
    $$

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    K_{pp} = \begin{bmatrix}
    k_1 & k_2 & k_3 & k_4
    \end{bmatrix}
    $$

    telle que le système en boucle fermée :

    $$
    \dot{\mathbf{x}} = (A - BK_{pp})\mathbf{x}
    $$

    ait des valeurs propres (pôles) placées à des positions désirées dans le plan complexe, reflétant nos objectifs de performance.



    Nous voulons un amortissement modéré et un temps de stabilisation ≤ 20s


    $$
    T_s \approx \frac{4}{|\text{Re}(\lambda)|}
    $$


    $$
    \frac{4}{|\text{Re}(\lambda)|} \leq 20 \quad \Rightarrow \quad |\text{Re}(\lambda)| \geq 0.2
    $$

    Choisissons :



    * 2 pôles réels dominants : $-0{,}3$, $-0{,}4$ (temps de stabilisation \~ $\frac{4}{0{,}3} \approx 13{,}3$ s)
    * 1 paire complexe conjuguée faiblement amortie : $-0{,}8 \pm 1{,}5i$ → décroissance plus rapide, possibilité d’oscillations

    Pôles choisis :

    $$
    \lambda = \{-0{,}3,\ -0{,}4,\ -0{,}8 + 1{,}5i,\ -0{,}8 - 1{,}5i\}
    $$

    Ces pôles assurent :

    * Partie réelle négative → stabilité asymptotique
    * Temps de stabilisation < 20 s
    * Convergence fluide
    """
    )
    return


@app.cell
def _(alpha, g, np, scipy):
    _A = np.array([[0, 1, 0, 0],
                  [0, 0, -g, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

    _B = np.array([[0],
                  [-g],
                  [0],
                  [-alpha]])

    desired_poles = [-0.3, -0.4, -0.8 + 1.5j, -0.8 - 1.5j]

    K_pp = scipy.signal.place_poles(_A, _B, desired_poles).gain_matrix
    print("K_pp =", K_pp)
    return


@app.cell
def _(J, M, g, l, np, plt, scipy):
    _K = np.array([ 0.1156   ,   0.73833333 ,-1.4152  ,   -1.01277778])  

    def booster_dynamics(t, state):
        x_pos, dx, theta, dtheta = state
        phi = -_K @ state
        ddx = -g * theta - g * phi
        ddtheta = -(l * M * g / J) * phi
        return [dx, ddx, dtheta, ddtheta]
    
    x0 = [0, 0, np.pi/4, 0]

    t_eval = np.linspace(0, 20, 100)

    sol = scipy.integrate.solve_ivp(booster_dynamics, [0, 20], x0, t_eval=t_eval)


    phi_values = np.array([-_K @ sol.y[:, i] for i in range(sol.y.shape[1])])

    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, np.degrees(sol.y[2]), 'r-', label='$\\theta(t)$')
    plt.axhline(0, color='k', linestyle='--', label='Target ($\\theta=0$)')
    plt.xlabel('Time (s)')
    plt.ylabel('Tilt Angle (°)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, np.degrees(phi_values), 'b-', label='$\\phi(t)$ (control input)')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input $\\phi$')
    plt.legend()
    plt.grid(True)
    plt.show()
    return sol, t_eval, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On veut minimiser la fonction coût quadratique :

    $$
    J = \int_0^\infty \left( \mathbf{x}^\top Q \mathbf{x} + u^\top R u \right) dt
    $$

    Sous les contraintes :

    $$
    \dot{\mathbf{x}} = A \mathbf{x} + B u, \quad u = -K_{oc} \mathbf{x}
    $$

    On attribue une pénalité plus élevée à la déviation angulaire et à la vitesse angulaire, et une pondération plus faible (ou nulle) à la position du chariot.

    On commence avec :

    $$
    Q = \text{diag}(q_1, q_2, q_3, q_4), \quad R = r
    $$

    Choisissons :

    * $q_1 = 10$ : pénaliser l’écart de position,
    * $q_2 = 1$ : poids modéré sur la vitesse,
    * $q_3 = 100$ : forte pénalisation de l’écart angulaire (critique pour la stabilité),
    * $q_4 = 10$ : pénaliser la vitesse angulaire,
    * $R = 1$ : poids standard sur la commande.

    $$
    Q = \begin{bmatrix}
    10 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 100 & 0 \\
    0 & 0 & 0 & 10
    \end{bmatrix}, \quad
    R = [1]
    $$



    On utilise l’algorithme LQR  :

    $$
    K_{oc} = R^{-1} B^\top P
    $$

    Avec $P$ solution de l’équation de Riccati algébrique continue (CARE) :

    $$
    A^\top P + P A - P B R^{-1} B^\top P + Q = 0
    $$
    """
    )
    return


@app.cell
def _(g, np, scipy):
    alpha = 3  # valeur de ell*M*g / J


    A = np.array([[0, 1, 0, 0],
                  [0, 0, -g, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

    B = np.array([[0],
                  [-g],
                  [0],
                  [-alpha]])  

    Q = np.diag([0, 0, 1, 1])
    R1 = np.array([[1]])

    # Résolution de l’équation de Riccati
    P = scipy.linalg.solve_continuous_are(A, B, Q, R1)

    K_oc = np.linalg.inv(R1) @ B.T @ P
    return K_oc, alpha


@app.cell
def _(K_oc):
    print(K_oc)
    return


@app.cell
def _(J, M, g, l, np, plt, scipy, sol):
    _K = np.array([-4.16208989e-19,  2.62916856e-17, -1.00000000e+00, -1.29099445e+00])  

    def booster_dynamics2(t, state):
        x_pos, dx, theta, dtheta = state
        phi = -_K @ state
        ddx = -g * theta - g * phi
        ddtheta = -(l * M * g / J) * phi
        return [dx, ddx, dtheta, ddtheta]
    
    x1 = [0, 0, np.pi/4, 0]

    t_eval1 = np.linspace(0, 20, 100)

    sol1 = scipy.integrate.solve_ivp(booster_dynamics2, [0, 20], x1, t_eval=t_eval1)


    phi_values1 = np.array([-_K @ sol1.y[:, i] for i in range(sol1.y.shape[1])])

    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, np.degrees(sol1.y[2]), 'r-', label='$\\theta(t)$')
    plt.axhline(0, color='k', linestyle='--', label='Target ($\\theta=0$)')
    plt.xlabel('Time (s)')
    plt.ylabel('Tilt Angle (°)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, np.degrees(phi_values1), 'b-', label='$\\phi(t)$ (control input)')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input $\\phi$')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell
def _(J, M, g, l, np, plt, scipy, t_eval, x0):

    K_gains = np.array([0, 0, -1/12, -4/15]) 



    def linearized_booster_dynamics(t, state):
        x_pos, dx, theta, dtheta = state
        phi_control = -(K_gains[2] * theta + K_gains[3] * dtheta)

        ddx = -g * theta - g * phi_control
        ddtheta = -(l * M * g / J) * phi_control
        return [dx, ddx, dtheta, ddtheta]


    def nonlinear_booster_dynamics(t, state):
        x_pos, dx, theta, dtheta = state

        phi_control = -(K_gains[2] * theta + K_gains[3] * dtheta)

        f_over_M = g 
        ddx = -f_over_M * np.sin(theta + phi_control)
        ddtheta = -(l * f_over_M / (J/M)) * np.sin(phi_control) 
        ddtheta_nonlinear = -(l * M * g / J) * np.sin(phi_control)

        return [dx, ddx, dtheta, ddtheta_nonlinear]


    x2 = [0, 0, np.pi/4, 0] 
    t_start = 0
    t_end = 20
    t_eval2 = np.linspace(t_start, t_end, 500) 

    sol_linear = scipy.integrate.solve_ivp(linearized_booster_dynamics, [t_start, t_end], x0, t_eval=t_eval, dense_output=True)

    sol_nonlinear = scipy.integrate.solve_ivp(nonlinear_booster_dynamics, [t_start, t_end], x2, t_eval=t_eval, dense_output=True)

    phi_linear_values = np.array([-(K_gains[2] * sol_linear.y[2, i] + K_gains[3] * sol_linear.y[3, i]) for i in range(sol_linear.y.shape[1])])
    phi_nonlinear_values = np.array([-(K_gains[2] * sol_nonlinear.y[2, i] + K_gains[3] * sol_nonlinear.y[3, i]) for i in range(sol_nonlinear.y.shape[1])])

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(sol_linear.t, sol_linear.y[2], label='Theta (Linearized)')
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Theta (Non-Linearized)', linestyle='--')
    plt.title('Comparison of Linearized and Non-Linearized Booster Dynamics')
    plt.ylabel('Theta (radians)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(sol_linear.t, sol_linear.y[0], label='x_pos (Linearized)')
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], label='x_pos (Non-Linearized)', linestyle='--')
    plt.ylabel('x_pos (m)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(sol_linear.t, phi_linear_values, label='Phi_control (Linearized System)')
    plt.plot(sol_nonlinear.t, phi_nonlinear_values, label='Phi_control (Non-Linearized System)', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Phi_control (radians)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""On a trouvé la meilleur K""")
    return


if __name__ == "__main__":
    app.run()
