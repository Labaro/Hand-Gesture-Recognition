from utils import *
from load_and_display_data import *
import sys

"""
sample = n[3, :size[3], -1, :].T
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(*sample)
#print(sample)
plt.show()
tck, u = splprep([*sample], s=0, k=2)
X, Y, Z = splev(np.linspace(0, 1, 171), tck)
new_points = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)])
print(n[3, size[3] - 1, -1, :] - new_points[-1, :])
print(new_points)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(*sample, 'r-')
ax.plot(X, Y, Z, 'ro')
plt.show()

n_train = normalize_position(n_train)
n_test = normalize_position(n_test)

N_train = interpolate(n_train, size_train, 162)
N_test = interpolate(n_test, size_test, 162)

N = np.vstack([N_train, N_test])

min = np.array([np.min(N[:, :, :, 0]), np.min(N[:, :, :, 1]), np.min(N[:, :, :, 2])])
max = np.array([np.max(N[:, :, :, 0]), np.max(N[:, :, :, 1]), np.max(N[:, :, :, 2])])
print(min, max)


for i in range(n_train.shape[0]):
    im = to_image(N_train, i, min, max)
    plt.imsave("image/train/"+str(i)+".png", im)
    
for i in range(n_test.shape[0]):
    im = to_image(N_test, i, min, max)
    plt.imsave("image/test/"+str(i)+".png", im)"""

model = run_test_harness()

list_image = []
for i in range(840):
    im = mpimg.imread(f"image/test/{i}.png")
    list_image.append(np.copy(im)[:, :, :3])

id_test, _,  sequences_test = read_csv_infos('infos_test.csv', test=True)
pred = model.predict_classes(np.array(list_image))

pred = pred + 1
full_pred = np.vstack((np.array(id_test), pred))

full_pred_pd = pd.DataFrame(full_pred.T)

full_pred_pd.to_csv('full_pred.csv', index=False, header=['Id', 'prediction'], sep=',')
