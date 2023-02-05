from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint


def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Menbe görüntüsü', 'Yaradilan', 'Gözlenilen']
    for i in range(len(images)):
        pyplot.subplot(1, 3, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])
    pyplot.show()



    [X1, X2] = load_real_samples('maps_256.npz')
print('Loaded', X1.shape, X2.shape)
model = load_model('model_109600.h5')

ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

gen_image = model.predict(src_image)
plot_images(src_image, gen_image, tar_image)
