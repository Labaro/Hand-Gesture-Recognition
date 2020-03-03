from utils import *
import matplotlib.pyplot as plt

n = get_skeleton("skeletons_world_train.csv")
print(n.shape)

N = normalize_position(n)
V = get_vectors(N)

img = to_image(N[0])
img_vector = to_image(V[0])

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_vector)
plt.show()
