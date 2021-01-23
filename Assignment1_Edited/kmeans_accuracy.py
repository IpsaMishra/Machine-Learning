import numpy as np

true_center=np.array([[-3.08395337, -0.08533732],[-0.02371148,2.94511067],[3.04268933 ,0.02065249]])

estimated_center=np.array([[-3,0],[0,3],[3,0]])

error = 0

for i in range(3):

    error += np.linalg.norm(true_center[i]-estimated_center[i])

error /= 3
print(error)


import numpy as np

true_center=np.array([[-2.01727152,0.00628048],[ 0.23631406,2.23623991],[ 1.91298001,-0.07510182]])

estimated_center=np.array([[-2,0],[0,2],[2,0]])

error = 0

for i in range(3):

    error += np.linalg.norm(true_center[i]-estimated_center[i])

error /= 3
print(error)
