import os

import cv2
import numpy as np
import imutils


def canny_edge_detection(image, blur_ksize=3, threshold1=50, threshold2=200):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
	img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
	return img_canny


# xuất ra list 4 điểm của các hình chữ nhật theo thứ tự diện tích
def export_appproxs120(image):
	thresh = canny_edge_detection(image)
	cv2.imwrite("candy.jpg", thresh)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	area_cnt = [cv2.contourArea(cnt) for cnt in contours]
	print(area_cnt[:8])
	approxs = []
	for contour in contours[:8]:
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
		if len(approx) == 4:
			approx = approx.reshape(4, 2)
			approxs.append(approx)
	# cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
	print(len(approxs))
	
	return approxs, area_cnt


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
		# cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
	print(len(approxs))
	return approxs, area_cnt


# sắp xếp 4 điểm theo đúng thứ tự top-left,top-right,bottom-right,bottom-left
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


def table_question120(image_table, num):
	height, width, depth = image_table.shape
	height_grid = int(height / 6)
	list_answers = []
	for i in range(6):
		grid_i = image_table[height_grid * i:height_grid * (i + 1)]
		height_b, width_b, depth2 = grid_i.shape
		height_box = int(height_b / 5)
		width_box = int(width_b / 5)
		
		grid_i = grid_i[int(height_box / 2):height_b - int(height_box / 3)]
		height_b, width_b, depth2 = grid_i.shape
		height_box = int(height_b / 5)
		width_box = int(width_b / 5)
		
		k = grid_i[:int(height_b / 5), width_box:width_box * 5 - 2]
		
		# cv2.imshow("a", k)
		# cv2.waitKey()
		
		for j in range(5):
			grid_ij = grid_i[height_box * j:height_box * (j + 1)]
			grid_ij = grid_ij[:, width_box:width_box * 5 - 2]
			cv2.imwrite("out/5/tm1{}{}{}.png".format(num, i, j), grid_ij)
	# cv2.imshow("a", grid_ij)
	# cv2.waitKey()
	print("ok")


# return grid_ij

# img=cv2.imread("dataTrain/z2523981002536_7a30d22504f7e833b9ae32cdcaf82f92.jpg")
# approxs, area_cnt=export_appproxs120(img)
# aa=four_point_transform(img, approxs[1])
# approxs, area_cnt = export_appproxs120(aa)
# aa=four_point_transform(aa, approxs[1])
# ak=table_question120(aa,0)

# img=cv2.imread("dataTrain/40-50/z2523980911153_8335f91f19bd7d4de1b475c91b504537.jpg")
# approxs, area_cnt=export_appproxs4050(img)
# aa=four_point_transform(img, approxs[0])
# approxs, area_cnt=export_appproxs4050(aa)
# aa=four_point_transform(aa, approxs[0])
# cv2.imshow("a",imutils.resize(aa,height=1000))
# cv2.waitKey()
def table_question40(image_tb, num):
	height, width, depth = image_tb.shape
	width_grid = int(width / 6)
	height_grid = int(height / 14)
	print(height, width, depth)
	image_tb = image_tb[2:-int(height_grid * 1.2), int(width_grid * 1.1):]
	height, width, depth = image_tb.shape
	print(height, width, depth)
	cv2.imshow("a", imutils.resize(image_tb, height=1000))
	cv2.waitKey()
	height_grid = int(height / 14)
	list_answers = []
	
	for i in range(14):
		grid_i = image_tb[int(height_grid * (i + 0.1)):height_grid * (i + 1)]
	# cv2.imwrite("out/4/tr{}{}.png".format(num,i),grid_i)
	
	cv2.imshow("a", imutils.resize(image_tb, height=1000))
	cv2.waitKey()


# table_question40(aa,0)

def process():
	img = cv2.imread("dataTrain/dataNone/4C555811-1AFE-4A25-8548-2FF0FDE76AD3.jpg")
	approxs, area_cnt = export_appproxs120(img)
	if len(approxs) >= 6:
		for i in range(5):
			
			aa = four_point_transform(img, approxs[i])
			height, width, depth = aa.shape
			if height / width > 3:
				table_question120(aa, i)
	else:
		print("zoo")
		img_n = four_point_transform(img, approxs[0])
		approxs, area_cnt = export_appproxs120(img_n)
		for i in range(5):
			aa = four_point_transform(img_n, approxs[i])
			height, width, depth = aa.shape
			if height / width > 3:
				table_question120(aa, i)
# process()
