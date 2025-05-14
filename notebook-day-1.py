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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.patches as patches

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple

    from scipy.integrate import solve_ivp
    return FFMpegWriter, FuncAnimation, np, plt, solve_ivp, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell
def _(R, np):
    R(np.pi)
    return


@app.cell
def _(np):
    np.sin(np.pi)
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    g = 1 #m/s^2
    l = 1 #m
    M = 1 #kg
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


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/forces.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \vec{f} = R(\theta + \phi) \cdot \begin{bmatrix} 0 \\ f \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(R, np):
    def force(f,theta,phi):
        fx,fy =  R(theta+phi) @ np.array([0,f])
        return fx,fy
    return


@app.cell
def _(R, np):
    # Methode 2 
    def force2(f, theta, phi):
        f_x,f_y = R(theta) @ R(phi) @ np.array([0,f])
        return (f_x,f_y)
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


@app.function
def poid(M, g):
    return (0,-M*g)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    D'après le principe fondamentale de la dynamique :

    $$
    \vec{P} + \vec{f} = M*\vec{a}
    $$

    En projetant sur la base cartésienne $(\vec{x},\vec{y})$, On obtient : 

    $$
    M*\ddot{x}=0+f_x
    $$

    $$
    M*\ddot{y}=-M*g + f_y
    $$

    Alors : 

    $$
    \ddot{x}=- \frac{f*\sin(\theta + \phi)}{M}
    $$

    $$
    \ddot{y}=-g + \frac{f*\cos(\theta + \phi)}{M}
    $$
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On suppose le 'booster' comme une tige (On suppose que le rayon du cylindre est négligeable devant sa hauteur)

    Alors l'expression du moment d'inertie : 

    $$
    J = \frac{1}{12}M(2*l)^2 = \frac{1}{3}Ml^2
    $$

    """
    )
    return


@app.cell
def _(M, l):
    J = (1/3)*M*(l**2)
    return (J,)


@app.cell
def _(J):
    print(J)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/forces2.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    D'après le théorème des moments dynamiques : 

    $$
    \frac{d\vec{L}_O}{dt} = \sum \vec{M}_O^{\text{ext}}
    $$

    Avec : 

    - $\vec{L}_O$ : moment cinétique par rapport à un point $O$  
    - $\vec{M}_O^{\text{ext}}$ : moment résultant des forces extérieures par rapport à $O$




    Alors par rapport au point G:

    - Moment cinétique :  $L = J \dot{\theta}$

    Le théorème devient:

    $$
    \frac{dL}{dt} = J\ddot{\theta} = M_{G}
    $$

    Avec :

    - $J$ : moment d’inertie du solide
  
    - $\omega$ : vitesse angulaire

    - $\ddot{\theta}$ : accélération angulaire

    - $M_{G}$ : moment des forces extérieures

    Or : $M_{G} = (\overrightarrow{OG} \times \vec{f})*\vec{z} = -l*(f_x*\cos(\theta) + f_y*\sin(\theta))$


    Donc l'équation diférentielle par rapport à $\theta$ : 

    $$
    J \ddot{\theta} = - l*f*\sin(\theta)\cos(\theta+\phi) + l*f*\cos(\theta) \sin(\theta+\phi)
    $$

    Donc : 

    $$
    J \ddot{\theta} = f*l*\sin(\phi)
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
def _(mo):
    mo.md(
        r"""

    On transforme les équations : 

    $$
    \ddot{x} = \frac{-f \sin(\theta + \phi)}{M} \tag{1}
    $$

    $$
    \ddot{y} = -g + \frac{f \cos(\theta + \phi)}{M} \tag{2}
    $$

    $$
    J \ddot{\theta} = f l \sin(\phi) \tag{3}
    $$

    En une équation de première degrée.

    Pour cela on définit le vecteur : 

    $$
    \mathbf{X} = \begin{bmatrix}
    x \\ \dot{x} \\
    y \\ \dot{y} \\
    \theta \\ \dot{\theta}
    \end{bmatrix}
    $$

    Donc l'ODE devient : 

    $$
    \frac{d}{dt}
    \begin{bmatrix}
    x \\ \dot{x} \\
    y \\ \dot{y} \\
    \theta \\ \dot{\theta}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \dot{x} \\
    - \frac{-f \sin(\theta + \phi)}{M} \\
    \dot{y} \\
    - g + \frac{f \cos(\theta + \phi)}{M} \\
    \dot{\theta} \\
    \frac{f l \sin(\phi)}{J}
    \end{bmatrix}
    $$


    Donc : 

    $$
    \frac{d\mathbf{X}}{dt} = \mathbf{F}(\mathbf{X}, t)
    $$

    Avec : 

    $$
    \mathbf{F}(\mathbf{X}, t) =
    \begin{bmatrix}
    \dot{x} \\
    - \frac{-f \sin(\theta + \phi)}{M} \\
    \dot{y} \\
    - g + \frac{f \cos(\theta + \phi)}{M} \\
    \dot{\theta} \\
    \frac{f l \sin(\phi)}{J}
    \end{bmatrix}
    $$

    """
    )
    return


@app.cell
def _(J, M, f, g, l, np, phi, solve_ivp):
    # On définit la fonction F
    def f_phi(t, z):
        x, dx, y, dy, theta, dtheta = z
    
        ddx = -f(t)* np.sin(theta + phi)/M
        ddy = -g + f(t) * (np.cos(theta + phi)/M)
        ddtheta = -(f(t) * l * np.sin(phi)) / J

        return np.array([dx,ddx,dy,ddy,dtheta,ddtheta])

    # On définit redstart_solve

    def redstart_solve(t_span, y0, f_phi):
        sol = solve_ivp(f_phi, t_span, y0, dense_output=True)
        return sol

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    on cherche une trajectoire qui verifie


    * $y(0) = 10$
    * $\dot{y}(0) = 0$
    * $y(5) = l = 1$
    * $\dot{y}(5) = 0$

    On suppose qu'on a une trajectoire de reference y(t) est une fonction polynomiale de t de 3eme degree 


    $$
    y(t) = a t^3 + b t^2 + c t + d
    $$



    * $y(0) = 10$ → $d = 10$
    * $\dot{y}(0) = 0$ → $c = 0$
    * $y(5) = l$ → $125a + 25b + 5c + d = l = 1$

    * $\dot{y}(5) = 0$

    $$
    \dot{y}(t) = 3a t^2 + 2b t + c
    \quad \Rightarrow \quad 
    \dot{y}(5) = 75a + 10b + c = 0
    $$


    $$
    \begin{cases}
    75a + 10b = 0\\
    125a + 25b =  - 9
    \end{cases}
    $$




    Alors: 

    $$
    a = \frac{18}{125}
    $$

    $$
    b = \frac{-27}{25}
    $$

    donc: 

    $$
    f(t) = \frac{1}{M}\ddot{y}(t)  + g = \frac{(6 \cdot a \cdot t + 2 b \cdot t )}{M} +g
    $$




    """
    )
    return


@app.cell
def _(J, M, g, l, np, solve_ivp):

    v0 = 0.0  

    a = (2.5 * v0 + 9) / 62.5
    b = (-v0 - 75 * a) / 10
    c = v0
    d = 10

    def y_ddot(t):
        return 6 * a * t + 2 * b

    def f_phi2(t, y):
        f_t = y_ddot(t) + M * g
        return np.array([f_t, 0.0])  

    def redstart_solve2(t_span, y0, f_phi2):
        def ode(t, y):
            x, dx, y_pos, dy, theta, dtheta = y
            f, phi = f_phi2(t, y)

            ddx = -f * np.sin(theta + phi)
            ddy = -M * g + f * np.cos(theta + phi)
            ddtheta = (f * l * np.sin(phi)) / J

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        sol_ivp = solve_ivp(ode, t_span, y0, dense_output=True)
        return lambda t: sol_ivp.sol(t)




    return f_phi2, redstart_solve2, v0


@app.cell
def _(f_phi2, l, np, plt, redstart_solve2, v0):
    t_span = [0, 5]
    y0 = [0, 0, 10, v0, 0, 0]
    sol = redstart_solve2(t_span, y0, f_phi2)

    t = np.linspace(t_span[0], t_span[1], 1000)
    Y = sol(t)
    y_t = Y[2]
    ydot_t = Y[3]

    plt.plot(t, y_t, label=r"$y(t)$")
    plt.plot(t, ydot_t, label=r"$dy/dt$")

    plt.plot(t, l * np.ones_like(t), 'k--', label=r"$y = \ell$")
    plt.xlabel("time $t$")
    plt.ylabel("height $y$")
    plt.grid(True)
    plt.legend()
    plt.show()

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
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


if __name__ == "__main__":
    app.run()
