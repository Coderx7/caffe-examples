import numpy as np
import caffe
import yaml
from matplotlib import pyplot as plt

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.params = yaml.load(self.param_str)
        self.size = self.params['size']
        # generate random points
        self.x = np.random.random((self.size, 1)).astype(dtype=np.float32)
        # a simple linear relation ship links x and y
        self.y = 2.0 * self.x + 0.5

        # data
        top[0].reshape(*self.x.shape)
        top[0].data[...] = self.x
        # labels
        top[1].reshape(*self.y.shape)
        top[1].data[...] = self.y


    def reshape(self, bottom, top):
        top[0].reshape(*self.x.shape)
        top[1].reshape(*self.y.shape)

    def forward(self, bottom, top):
        # data
        top[0].data[...] = self.x
        # labels
        top[1].data[...] = self.y

    def backward(self, top, propagate_down, bottom):
        pass


class PlotLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # pyplot must be in interactive mode
        assert plt.isinteractive()
        self.ax.cla()
        plt.hold(True)
        self.ax.plot(bottom[0].data.ravel(), bottom[1].data.ravel(), 'b.')
        self.ax.plot(bottom[0].data.ravel(), bottom[2].data.ravel(), 'r.')
        plt.hold(False)
        plt.draw()
        # required so the plot can update
        plt.pause(0.001)

    def backward(self, top, propagate_down, bottom):
        pass
