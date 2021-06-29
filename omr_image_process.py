import os
import numpy as np
def order_points(approx):
	rect = np.zeros((4, 2), dtype="float32")
	s = approx.sum(axis=1)
	rect[0] = approx[np.argmin(s)]
	rect[2] = approx[np.argmax(s)]
	diff = np.diff(approx, axis=1)

	rect[1] = approx[np.argmin(diff)]
	rect[3] = approx[np.argmax(diff)]
	return rect