import jax
import time
from jax import jit, grad

key = jax.random.PRNGKey(0)


def xx(x, y1, y2):
	# A = jax.numpy.ones((y2,y1))
	# B = jax.numpy.ones((y1,x))

	A = jax.random.normal(key, (y2, y1))
	B = jax.random.normal(key, (y1, x))

	return jax.numpy.matmul(A, B)


yy = (xx(3, 4, 5))

print(type(yy), yy)
