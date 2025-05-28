---
layout: default
title: Kendall's shape space
---

# Kendall's shape space

## David G. Kendall - the man, the myth, the mathematician

For anyone new to the study of shape analysis, I'd like to introduce them to Professor David George Kendall (born 1918 in the UK)&mdash;a pioneer in applied probability and statistics, he was actually the first-ever Professor of Mathematical Statistics at Cambridge. He's referred to as the father of modern probability theory in Britain, and contributed to a number of diverse and esoteric applied fields in addition to his mastery of abstract mathematical theory. Kendall was initially interested in pursuing work related to his fascination with stars, becoming set on studying mathematics when he was advised it was a necessary first step to understanding astronomy. 

A fundamental piece in Kendall's growth as a mathematician was his time in the Projectile Development Establishment (PDE) in Wales during World War II, where he worked on rocketry. Kendall at the time (~1940) was a relatively new graduate and the team contained more seasoned mathematicians who could relay insights and experience to the younger Kendall. It was during Kendall's time with the PDE that he gained a sudden and deeper understanding of probability and statistics, filling a void left in the PDE when statisticians Frank Anscombe and Maurice Bartlett left for posts in London. Bartlett and Kendall reportedly began focusing on Markov processes governing random time-evolving systems around his time on the PDE project. 

After Kendall's time in the PDE, he was tenured at Magdalen College in Oxford starting in 1946, then in 1952 took on a year-long visiting position at Princeton University. Kendall was later appointed Director of the Statistical Laboratory at Cambridge University, where he was a fellow until his retirement in 1985 and after which was allowed to be a life fellow and contribute to academia for the rest of his life. Kendall passed away in 2007. He has been awarded the Guy Medal in silver and gold from the Royal Statistical Society (1955 and 1980, respectively), the Senior Whitehead Prize from the London Mathematical Society (1980), and the De Morgan Medal from the London Mathematical Society (1989). He helped found the Bernoulli Society in 1975 and was a fellow of the Royal Society from 1964 until his death. 

Some of Kendall's most prominent works include his contributions to the understanding of stochastic processes and queueing theory; for example, there is a "Kendall notation" and for queues and the imbedded Markov chain method introduced in his 1953 paper "Stochastic processes occurring in the theory of queues and their analysis by the method of the imbedded Markov chain". The A/S/c (for Arrival/Service/Servers) notation for describing queue types is still referred to as Kendall's notation. Kendall also developed early formal work in random sets, geometric probability, and spatial point processes, applying these concepts to models of river networks, biological shape, and topographic features. Out of this, Kendall became a pioneer in the modeling of spatial randomness, which is now key to geostatistics and ecology. A surprising outcome of Kendall's work was his debunking of some theories about ley lines; in his 1989 paper "Ley lines in question", Kendall showed how alignments of three or more points can arise purely by chance, meaning the appearance of ley lines in geographic data (like monument maps) doesn't inherently imply any meaningful pattern or intention. Now a classic example of mathematical rigor being used to debunk pseudoscientific claims, Kendall proposed quantitative tests for assessing whether alignments are statistically significant or not. I highly recommend reading the biography "David George Kendall" by Sir John Kingman FRS for more information on Kendall's early life and his well known works (see [here](https://royalsocietypublishing.org/doi/10.1098/rsbm.2008.0017)); for now I'll turn the focus to Kendall's contributions to shape analysis. 

## Kendall's idea of shape spaces 
I feel a good start to understanding shape in a mathematical way is to start with Kendall's framework (and any of my own personal interest in Kendall's life is mere coincidence...). I find that Kendall's formalization of shapes is intuitive even to the not-mathematically-inclined. We can think of a shape as a form or outline of an object&mdash;try to imagine the outline of a bird (an Antarctic petrel to be exact), for example:

![Antarctic Petrel](/assets/antarctic_petrel.JPG)

![Antarctic Petrel Mask](/assets/antarctic_petrel_mask.png)

![Antarctic Petrel Outline](/assets/antarctic_petrel_outline.png)

(If you're curious how to find this outline&mdash;or "extract contours"&mdash;in Python, I recommend [this blog post by Andrew Udell](https://towardsdatascience.com/background-removal-with-python-b61671d1508a/). I'll put the gist of it here): 

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read in the image
img = cv2.imread("antarctic_petrel.JPG")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define canny edge detector + other processing parameters
blur = 15
canny_low = 13
canny_high = 59
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10

# canny edge detector
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
image_area = img.shape[0] * img.shape[1]
contour_info = [(c, cv2.contourArea(c.astype(np.int32))) for c in contours]
contour_info = [(c, area) for c, area in contour_info if min_area * image_area < area < max_area * image_area]

# create a mask and fill each detected contour; the region of interest is white and o.w. black
mask = np.zeros(edges.shape, dtype=np.uint8)
for c, _ in contour_info:
    mask = cv2.fillConvexPoly(mask, c, 255)

# smooth for better appearance
mask = cv2.dilate(mask, None, iterations=dilate_iter)
mask = cv2.erode(mask, None, iterations=erode_iter)
mask = cv2.GaussianBlur(mask, (blur, blur), 0)

# create an outline of the object of interest
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outline = max(contours, key=cv2.contourArea)

# this will create the outline of the bird on top of a blank/white canvas
canvas = np.ones_like(mask) * 255 
cv2.drawContours(canvas, [outline], -1, color=0, thickness=2)

# plot the outline
plt.imshow(canvas, cmap='gray')
plt.title("Antarctic Petrel Outline")
plt.axis("off")
plt.show()
```
It doesn't matter if we rotate the outline of our bird, it's still a bird (and very importantly, it's still the *same* bird), right? 

![Rotated Antarctic Petrel Outline](/assets/rotated_antarctic_petrel_outline.png)

It'll also still be the same bird if it was shifted 5 centimeters to the right or 10 centimeters to the left, and (at least in our case here) it'll still be the same bird if it appeared twice as small or five times as large (perhaps you can think of us zooming in or out on the same bird, which gives it an appearance of larger or smaller due to perspective and not the bird becoming physically different). 

You'll find in my blog some different perspectives on what makes a shape a _shape_. For now, we'll use Kendall's definition. Shape here is a **finite set of landmarks**, or a set of points that capture well the geometry of an object. Think of:

- Corners of a leaf
- Bones in a skull
- Boundary nest locations in a colony of birds

For example, on our bird shape: 

![Simplified boundary points](/assets/simplified_antarctic_petrel_contour_points.png)

(if you're interested in the code to find these points, see below for simplifying the contour): 

```
largest_contour = max(contours, key=cv2.contourArea)

points = simplified.reshape(-1, 2)

x, y = points[:, 0], points[:, 1]
plt.scatter(x, y, color='black')
plt.gca().invert_yaxis() 
plt.title("Simplified Contour Points")
plt.axis("equal")
plt.axis("off")
plt.show()

```

Of course, more complex shapes require more boundary points to really represent what we'd expect the shape to look like. We can pretty much tell that these points represent our bird, so let's make our lives easy and just use these few boundary points (or _sample_ points) to represent our shape.

Let's formalize this for the mathematically-inclined (sorry for other readers... but I do blame you for reading a math blog if you don't want to see math). Say we have $k$ landmarks in $\mathbb{R}^d$, where their coordinates form the configuration matrix:

$$
\boldsymbol{X} = \lbrack x_1, x_2, ..., x_k \rbrack^{\intercal} \in \mathbb{R}^{k\times{d}}.
$$

This collection of coordinates is not yet _invariant_ (meaning _independent of_) rotation, scaling, or translation (location). If we were to shift all coordinates by some amount to the right, it would undeniably be a different set of coordinates. I.e., $x_1 + 5 \neq x_1$. Let's get into Kendall's proposal of a **pre-shape space**. 

## Pre-shape space

Kendall suggested isolating a _shape_ by removing translation and scale from consideration. To achieve _translation invariance_, we subtract the centroid from all of our boundary points (also called _landmarks_) $\left( x_1, x_2,...,x_k \right)$:

$$
\left( x_1, x_2, ..., x_k \right) \rightarrow \left( x_1 - \overline{\boldsymbol{X}}, x_2 - \overline{\boldsymbol{X}},..., x_k - \overline{\boldsymbol{X}} \right) \implies \boldsymbol{X} \rightarrow \boldsymbol{X} - \overline{\boldsymbol{X}}
$$

Where $\overline{\boldsymbol{X}}$ here is the centroid of the landmarks. Next, we achieve _scale invariance_ by forcing our configuration to have a unit size; we use the Frobenius norm $\Vert\boldsymbol{X}\Vert_F$ to do this:


$$
\left\Vert \boldsymbol{X} \right\Vert_F = \sqrt{ \sum_{i=1}^k \sum_{j=1}^m x_{ij}^2 }
$$

After our configuration of landmarks undergoes these transformations, we are left with a resulting **pre-shape**. Pre-shapes are stored in a pre-shape space, a high-dimensional sphere of the form

$$
S^{kd - d - 1}
$$

which contains centered configurations normalized by size but which still aren't yet invariant to rotation. **Shape space** is our resulting space once we make our configurations invariant to rotation. We can introduce some more math mumbo jumbo and say

$$
S^{kd-d-1} = \left\{ \boldsymbol{X} \in \mathbb{R}_0^{k\times{d}} \mid \Vert\boldsymbol{X}\Vert_F = 1 \right\}
$$

Where $\mathbb{R}_0^{k\times{d}}$ is the space of centered configurations (i.e., those with zero mean). We can call the pre-shape space a unit sphere embedded in the linear space of centered configurations. 

Before getting to shape spaces, let's take a look at how we can practically remove translation and size variations from our examplar bird shape using Python: 

```
# center the configuration
centroid = np.mean(points, axis=0)
centered = points - centroid

# normalize the size
fro_norm = np.linalg.norm(centered)
pre_shape = centered / fro_norm
```

and when we re-plot these points, we should get a shape that looks the same:

![Pre-shape centered and scaled](/assets/pre_shape_antarctic_petrel.png)

## Shape space

To make our configuration invariant to rotation, we take the quotient of the pre-shape space by the rotation group $SO(d)$:

$$
\Sigma_d^k = S^{kd - d - 1} / SO(d)
$$

And **Kendall's shape space** is this _quotient space_. Each point in the space represents an equivalence class of configurations differing only by rotation (and therefore the configurations within the same equivalence class are the _same_ shape in this space).

### Procrustes

How do we implement this rotational invariance into our Python code, you ask? We need to use something called **Procrustes alignment**&mdash;this is a procedure we use to find the optimal rotation for a (centered and scaled) input shape (_pre-shape_) $\boldsymbol{X}\in\mathbb{R}^{k\times{d}}$ to align well with a (centered and scaled) reference shape $\boldsymbol{Y}\in\mathbb{R}^{k\times{d}}$. That is, we want to find the rotation matrix $\boldsymbol{R}$ such that 

$$
\boldsymbol{X}\boldsymbol{R} \approx \boldsymbol{Y}
$$

So we wish to solve 

$$
\min_{R\in{O}_d}\Vert\boldsymbol{X}R - \boldsymbol{Y}\Vert_F^2
$$

And we have a requirement for $R$ to be _orthogonal_ (i.e., $\boldsymbol{R}^{\intercal}\boldsymbol{R} = \boldsymbol{I}$). See my other blog post about Procrustes alignment to see how this is all done step by step. For now, I'll just paste in the Python script for how to generate the little gif seen below: 

![procrustes gif](/assets/procrustes_alignment.gif)

```
import matplotlib.animation as animation
from scipy.linalg import svd
from pathlib import Path

# define a function that creates a rotation for us by some angle theta
def make_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# apply a known rotation to create a target
theta = np.pi / 2
rotated_shape = pre_shape @ make_rotation(theta).T

# compute the procrustes alignment back to original
A = rotated_shape.T @ pre_shape
U, _ , Vt = svd(A)
R_opt = U @ Vt

# interpolate the rotation over time
n_frames = 30
interpolated_shapes = [rotated_shape @ make_rotation((1 - a) * theta).T @ R_opt for a in np.linspace(0, 1, n_frames)]

# animation setup
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
points_plot, = ax.plot([], [], 'o- label = "Aligning shape")
ref_plot, = ax.plot([], [], 'x--', label = "Reference shape")
ax.set_axis_off()
ax.legend()

def init():
    points_plot.set_data([],[])
    ref_plot.set_data([],[])
    return points_plot, ref_plot

def update(frame):
    shape = interpolated_shapes[frame]
    points_plot.set_data(shape[:,0], shape[:,1])
    ref_plot.set_data(pre_shape[:,0], pre_shape[:,1])
    return points_plot, ref_plot

ani = animation.FuncAnnimation(fig, update, frames=n_frames, init_func=init, blit=True)

# save it as a gif
gif_path = Path("procrustes_alignment.gif")
ani.save(gif_path, writer='pillow', fps=10)
print(f"Saved to {gif_path}")
```

Procrustes alignment 

### Quick note about landmarks versus shapes

For people used to coordinates meaning a point in Euclidean space, it can be pretty confusing when we use the term _point_ to refer to a configuration in shape or pre-shape space. In our notation, a point $\boldsymbol{X}\in\Sigma_k^d$ is a configuration of coordinates, so $\boldsymbol{X} = \left(x_1, x_2, ..., x_d)\right)$, where each $x_i$ is the more familiar coordinate. $\boldsymbol{X}$ is very explicitly a $k\times{d}$ matrix, containing $k$ landmarks of dimension $d$. 

### Geometry of Kendall's shape space
$\Sigma_k^k$ is a smooth Riemannian manifold. By this, we mean the space looks locally like a Euclidean space (i.e., imagine a smooth surface or some smooth higher-dimensional shape); the smoothness allows us to do calculus on the space (e.g., take derivatives); and it comes with a notion of distance, angles, and curvature, due to a Riemannian **metric** (here, metric means a way we measure distances between points in the space). If you want to learn more about the Riemannian metric (and metrics in general), check out my next blog post. I'll focus for now on relaying some useful info we can keep in mind to apply Kendall's ideas to real data. 

I already mentioned that the pre-shape space $S^{kd-d-1}$ is a unit sphere (remember: this is the space of configurations that are invariant to translation and scaling but _not_ rotation). Even though we have our shape space $\Sigma_k^d$, the use of the pre-shape sphere can be very convenient&mdash;speficially, if we imagine the tangent space $T_X$ of our pre-shape sphere near the point (or configuration) $\boldsymbol{X}$, then we can linearly approximate our pre-shape sphere, allowing us to do nice things that rely on a space being linear (like principal component analysis, a.k.a. PCA, for example). Additionally, linear distances on $T_X$ closely resemble distances on the pre-shape sphere, so long as we remain close to the reference point $\boldsymbol{X}$. When we work on a tangent space like $T_X$, we can map between the tangent space and the original space easily by using the logarithmic and exponential maps; the **logarithmic** map gives us a way to map from our original space to the tangent space. Letting $\boldsymbol{X}, \boldsymbol{Y} \in S^{kd - d - 1}$: 

$$
\log_{\boldsymbol{X}}{\boldsymbol{Y}} = \frac{\theta}{\sin\theta}(\boldsymbol{Y} - \cos\theta\boldsymbol{X}), \hspace{0.25cm} \theta = \text{arccos}\left(\langle\boldsymbol{X}, \boldsymbol{Y}\rangle_F\right)
$$

Specifically, this gives a vector in $T_X$ pointing toward $\boldsymbol{Y}$. 

And conversely, the **exponential map** gives us a way to map from the tangent space back to the pre-shape manifold: 

$$
\exp_X(\boldsymbol{V}) = \cos\left(\Vert\boldsymbol{V}\Vert\right)\boldsymbol{X} + \sin\left(\Vert\boldsymbol{V}\Vert\right)\frac{\boldsymbol{V}}{\Vert\boldsymbol{V}\Vert}
$$

And this maps a vector $\boldsymbol{V}\in T_X$ back to our manifold or pre-shape space. 

## Statistics in shape space

We've established linearity with our tangent space $T_X$ on the pre-shape space $S^{kd-d-1}$; let's define a meaningful notion of the **mean** of all of our pre-shape configurations $\boldsymbol{X}_1, ..., \boldsymbol{X}_n$: 

$$
\mu = \arg\min_{X\in{S}^{kd-d-1}}\sum_{i=1}^nd^2\left(\boldsymbol{X}, \boldsymbol{X}_i\right)
$$

This $\mu$ is often called the **Karcher mean** or **Frechet mean**, representing an approximation of what the average configuration in our pre-shape space looks like. It's a shape that minimizes the total squared geodesic distance to all of our shapes of interest. 

We have a lot of letters going on ($k,d,n$), so I'll just remind you that:
- $k$: number of landmarks we have in each shape
- $d$: number of coordinates in (or the dimension of) each landmark (for example, 2 for planar shapes, 3 for 3D shapes...)
- $n$: the number of shapes over which we are averaging

To be explicit, for our non-normalized configurations (i.e., before scaling, translation, or rotation invariance is achieved), each observed configuration $\boldsymbol{X}_i\in\mathbb{R}^{k\times{d}}$ (where $i = 1, ..., n$) contains all $k$ landmarks:

$$
\boldsymbol{X}_i = \begin{bmatrix}x_i^{(i)} \\\ x_2^{(i)} \\\ \vdots \\\ x_k^{(i)}\end{bmatrix}, \hspace{0.25cm} \text{where each } x_j^{(i)}\in\mathbb{R}^d
$$

For example, if $d = 2$, then a single landmark $x_j^{(i)}$ is:

$$
x_j^{(i)} = \begin{bmatrix}c_1 \\\ c_2 \end{bmatrix} \in \mathbb{R}^2
$$



