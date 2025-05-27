# Kendall's shape space

## David G. Kendall - the man, the myth, the mathematician

For anyone new to the study of shape analysis, I'd like to introduce them to Professor David George Kendall (born 1918 in the UK)--a pioneer in applied probability and statistics, he was actually the first-ever Professor of Mathematical Statistics at Cambridge. He's referred to as the father of modern probability theory in Britain, and contributed to a number of diverse and esoteric applied fields in addition to his mastery of abstract mathematical theory. Kendall was initially interested in pursuing work related to his fascination with stars, becoming set on studying mathematics when he was advised it was a necessary first step to understanding astronomy. 

A fundamental piece in Kendall's growth as a mathematician was his experiences in the Projectile Development Establishment (PDE) in Wales during World War II, where he worked on rocketry. Kendall at the time (~1940) was a relatively new graduate and the team contained more seasoned mathematicians who could relay insights and experience to the younger Kendall. It was during Kendall's time with the PDE that he gained a sudden and deeper understanding of probability and statistics, filling a void left in the PDE when statisticians Frank Anscombe and Maurice Bartlette left for posts in London. Bartlett and Kendall reportedly began focusing on Markov processes governing random time-evolving systems around his time on the PDE project. 

After Kendall's time in the PDE, he was tenured at Magdelen College in Oxford starting in 1946, then in 1952 took on a year-long visiting position at Princeton University. Kendall was later appointed Director of the Statistical Laboratory at Cambridge University, where he was a fellow until his retirement in 1985 and after which was allowed to be a life fellow and contribute to academia for the rest of his life. Kendall passed away in 2007. He has been awarded the Guy Medal in silver and gold from the Royal Statistical Society (1955 and 1980), the Senior Whitehead Prize from the London Mathematical Society (1980), and the De Morgan Medal from the London Mathematical Society (1989). He was a fellow of the Royal Society from 1964 until his death and helped found the Bernoulli Society in 1975. 

Some of Kendall's most prominent works include his contributions to the understanding of stochastic processes and queueing theory; for example, there is a "Kendall notation" and for queues and imbedded Markov chain method introduced in his 1953 paper "Stochastic processes occurring in the theory of queues and their analysis by the method of the imbedded Markov chain". The A/S/c (for Arrival/Service/Servers) notation for describing queue types is still referred to as Kendall's notation. Kendall also developed early formal work in random sets, geometric probability, and spatial point processes, applying these concepts to models of river networks, biological shape, and topographic features. Out of this, Kendall became a pioneer in the modeling of spatial randomness, which is now key to geostatistics and ecology. A surprising outcome of Kendall's work was his debunking of some theories about ley lines; in his 1989 paper "Ley lines in question", Kendall showed how alignments of three or more points can arise purely by chance, meaning the appearance of ley lines in geographic data (like monument maps) doesn't inherently imply any meaningful pattern or intention. Now a classic example of mathematical rigor being used to debunk pseudoscientific claims, Kendall proposed quantitative tests for assessing whether alignments are statistically significant or not. I highly recommend reading the biography "David George Kendall" by Sir John Kingman FRS for more information on Kendall's early life and well known works (see [here](https://royalsocietypublishing.org/doi/10.1098/rsbm.2008.0017)); for now I'll turn the focus to Kendall's contributions to shape analysis. 

## Shape spaces 
I feel a good start to understanding shape in a mathematical way is to start with Kendall's framework (and any of my own personal interest in Kendall's life is mere coincidence...). I find that Kendall's formalization of shapes is intuitive even to the not-mathematically-inclined. We can think of a shape as a form or outline of an object--try to imagine the outline of a bird (an Antarctic petrel to be exact), for example:

![Antarctic Petrel](/assets/antarctic_petrel.JPG)

![Antarctic Petrel Mask](/assets/antarctic_petrel_mask.png)

![Antarctic Petrel Outline](/assets/antarctic_petrel_outline.png)

(If you're curious how to find this outline--or "extract contours"--in Python, I recommend [this blog post by Andrew Udell](https://towardsdatascience.com/background-removal-with-python-b61671d1508a/). I'll put the gist of it here): 

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
Let's formalize this for the mathematically-inclined (sorry for other readers... but I do blame you for reading a math blog if you don't want to see math). Say we have $k$ landmarks in \(\mathbb{R}^d\), where their coordinates form the configuration matrix:

$$
\boldsymbol{X} = \left[x_1, x_2, ..., x_k\right]^{\intercal} \in \mathbb{R}^{k\times{d}}
$$



