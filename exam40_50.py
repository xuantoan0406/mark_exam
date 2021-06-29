import cv2
import numpy as np
from os import listdir

def canny_edge_detection(image, blur_ksize=3, threshold1=50, threshold2=200):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
	img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
	# cv2.imwrite("black_whrite.jpg",img_canny)
	return img_canny
def order_points(approx):
	rect = np.zeros((4, 2), dtype="float32")
	s = approx.sum(axis=1)
	rect[0] = approx[np.argmin(s)]
	rect[2] = approx[np.argmax(s)]
	diff = np.diff(approx, axis=1)
	rect[1] = approx[np.argmin(diff)]
	rect[3] = approx[np.argmax(diff)]
	return rect


# lấy các table,chuyển sang tọa độ mới
def four_point_transform(image, approx):
	rect = order_points(approx)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped
def export_appproxs4050(image):
	thresh = canny_edge_detection(image)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	area_cnt = [cv2.contourArea(cnt) for cnt in contours]
	print(area_cnt[:6])
	approxs = []
	for contour in contours[:6]:
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
		if len(approx) == 4:
			approx = approx.reshape(4, 2)
			approxs.append(approx)
			# for i in approx:
			# 	approxs.append(i)
			# cv2.drawContours(image, approxs, -1, (0, 0, 255), 3)
	print(len(approxs))
	return approxs, area_cnt


def table_question4050(image_4050, num_question):
	height, width, depth = image_4050.shape
	height_grid = int(height / num_question)
	width_grid = int(width / 5)
	list_answers = []
	image_4050 = image_4050[int(height_grid / 4):-int(height_grid / 2.3), width_grid:-2]
	
	height, width, depth = image_4050.shape
	height_grid = int(height / num_question)
	for i in range(num_question):
		grid_i = image_4050[height_grid * i:height_grid * (i + 1)]
		# cv2.imwrite("40/image40__{}_{}_{}.jpg".format(num_img,num_table,i),grid_i)
		answers = []
	# cv2.imshow("A", imutils.resize(grid_i, height=100))
	# cv2.waitKey()
	
	return "ok"

def get_id_4050(image_id_student, num_column, num_img):
	height, width, depth = image_id_student.shape
	width_grid = int(width / num_column)
	list_id = []
	
	for i in range(num_column):
		grid_i = image_id_student[:, width_grid * i + 2:width_grid * (i + 1)]
		cv2.imwrite("dataId/id6/{}nut_img_{}_{}.jpg".format(num_img, num_column, i), grid_i)


def get_data_train(path_folder):
	k = 0
	for name in listdir(path_folder):
		img = cv2.imread(path_folder + name)
		approxs, area_cnt = export_appproxs4050(img)
		img = four_point_transform(img, approxs[0])
		
		approxs, area_cnt = export_appproxs4050(img)
		
		if len(approxs) >= 3:
			approxs_table_question_sorted = sorted(approxs[:], key=np.sum, reverse=False)
			# print(approxs_table_question_sorted)
			# a=four_point_transform(img, approxs_table_question_sorted[2])
			for i in range(3):
				a = four_point_transform(img, approxs_table_question_sorted[i])
				
				table_question4050(a, k, i, 14)
		k += 1


def get_id_train(path_folder):
	k = 0
	for name in listdir(path_folder):
		img = cv2.imread(path_folder + name)
		approxs, area_cnt = export_appproxs4050(img)
		img = four_point_transform(img, approxs[0])
		approxs, area_cnt = export_appproxs4050(img)
		if len(approxs) >= 5:
			a = four_point_transform(img, approxs[-2])
			get_id_4050(a, 6, k)
		k += 1

# get_id_train("40-50/50/")