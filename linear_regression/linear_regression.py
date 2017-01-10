import numpy as np
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import yaml
from matplotlib import pyplot as plt

SIZE = 100
NET_FILENAME = 'train.prototxt'
DEPLOY_FILENAME = 'deploy.prototxt'
SOLVER_FILENAME = 'solver.prototxt'
USE_PLOT_LAYER = True
USE_SOLVE = True

## Create the network layers with NetSpec

n = caffe.NetSpec()

# create 2d point data layer
n.data, n.label = L.Python(ntop=2,
                           python_param=dict(module='linreg',
                                             layer='DataLayer',
                                             param_str=yaml.dump(dict(size=SIZE))
                                             )
                           )

# linear regression weights
# the number of outputs is 1 since we want to find
# the slope of the line, caffe adds the bias (intercept) automatically
n.predicted_label = L.InnerProduct(n.data,
                                   num_output=1,
                                   weight_filler=dict(type='xavier'))

# l2 minimization
n.loss = L.EuclideanLoss(n.predicted_label,
                          n.label)

if USE_PLOT_LAYER:
    # create 2d point plot layer
    n.plot = L.Python(n.data,
                    n.label,
                    n.predicted_label,
                    ntop=0,
                    python_param=dict(module='linreg',
                                        layer='PlotLayer'))

# save net to file
with open(NET_FILENAME, 'w') as f:
    f.write(str(n.to_proto()))

# generate solver
solver_mode = P.Solver.CPU
device_id = 0
base_lr = 0.1
iter_size = 1

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'stepsize': 200,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 300,
    'snapshot': 300,
    'snapshot_prefix': '',
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
}

# save solver to file
solver_param = caffe_pb2.SolverParameter(
    train_net=NET_FILENAME,
    **solver_param)

with open(SOLVER_FILENAME, 'w') as f:
    f.write(str(solver_param))

# solve
caffe.set_mode_cpu()
solver = caffe.SGDSolver(SOLVER_FILENAME)

plt.ion()

# one can use this
if USE_SOLVE:
    solver.solve()
# or this
else:
    for i in range(1000):
        solver.step(1)
        W, b = solver.net.params['predicted_label']
        if i % 10 == 0:
            print('y = W * x + b -> y = {W} * x + {b}'.format(W=W.data[0], b=b.data[0]))

plt.ioff()

# Create inference network

n = caffe.NetSpec()

n.data = L.Input(shape=[dict(dim=[SIZE, 1]),])

# linear regression weights
# the number of outputs is 1 since we want to find
# the slope of the line, caffe adds the bias (intercept) automatically
n.predicted_label = L.InnerProduct(n.data,
                                   num_output=1,
                                   weight_filler=dict(type='xavier'))

# save net to file
with open(DEPLOY_FILENAME, 'w') as f:
    f.write(str(n.to_proto()))

net = caffe.Net(DEPLOY_FILENAME, '_iter_300.caffemodel', caffe.TEST)

x = np.random.random((SIZE, 1)).astype(dtype=np.float32)
# reshape if needed
net.blobs['data'].reshape(*x.shape)
net.blobs['data'].data[...] = x

out = net.forward()
py = out['predicted_label']

plt.figure()
plt.plot(x, py)
plt.show()