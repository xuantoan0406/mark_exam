from os import listdir
import cv2
import numpy as np
from tensorflow.keras import models
import imutils

model_id = models.load_model("model/best__id-15-1.0.h5")
model2 = models.load_model("model/best__question-12--0.9978328347206116-1.0.h5")


# lọc ra các cạnh
# def canny_edge_detection(image, blur_ksize=3, threshold1=50, threshold2=200):
def canny_edge_detection(image, blur_ksize=5, threshold1=30, threshold2=80):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
	img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)

	#cv2.imwrite("black_whrite.jpg",img_canny)
	return img_canny

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

# xuất ra list 4 điểm của các hình chữ nhật theo thứ tự diện tích
def export_appproxs120(image):
	thresh = canny_edge_detection(image)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	area_cnt = [cv2.contourArea(cnt) for cnt in contours]
	print(area_cnt[:8])
	approxs = []
	
	i=0
	for contour in contours[:8]:
		# cv2.drawContours(image, contour, -1, (0, i, 255), 3)
		cnt=contour.reshape(len(contour),2)
		# approxs.append(cnt)
		conrers = order_points(cnt)
		approxs.append(conrers)

		
	approxs = sorted(approxs, key=cv2.contourArea, reverse=True)
	area_cnt2 = [cv2.contourArea(cnt) for cnt in approxs]
	print(area_cnt2)

	return approxs, area_cnt

	
	# for contour in contours[:8]:
	# 	# x, y, w, h = cv2.boundingRect(contour)
	# 	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# 	peri = cv2.arcLength(contour, True)
	# 	approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
	# 	if len(approx) == 4:
	# 		approx = approx.reshape(4, 2)
	# 		approxs2.append(approx)
	# 	cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
	# 	hull = cv2.convexHull(contour)
	# 	print(hull.shape)
	# cv2.drawContours(image, contours[:8], -1,(0,255,0),2)
	# cv2.imshow("A",imutils.resize(image,height=700))
	# cv2.waitKey()
	
	



def export_appproxs4050(image):
	thresh = canny_edge_detection(image)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	area_cnt = [cv2.contourArea(cnt) for cnt in contours]
	print(area_cnt[:6])
	approxs = []
	for contour in contours[:6]:
		cnt = contour.reshape(len(contour), 2)
		conrers=order_points(cnt)
		approxs.append(conrers)
	
	approxs=sorted(approxs, key=cv2.contourArea, reverse=True)
	area_cnt2 = [cv2.contourArea(cnt) for cnt in approxs]
	print(area_cnt2)
	
	return approxs, area_cnt






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


def return_answer(image_question):
	# import random
	# name_r=str(random.randint(2000000,3000000))
	# cv2.imwrite("answers_50/as_aawsaad{}.jpg".format(name_r),image_question)
	image_question = cv2.resize(image_question, (112, 28))
	
	image_question = (image_question / 255).astype('float32')
	image_question = np.expand_dims(image_question, axis=0)
	predict = model2.predict(image_question)
	
	answer = np.argmax(predict)
	switcher = {0: "?", 1: "A", 8: "B", 9: "C", 10: "D", 11: "AB", 12: "AC", 13: "AD", 14: "BC", 15: "BD", 2: "CD",
	            3: "ABC", 4: "ABD", 5: "ACD", 6: "BCD", 7: "ABCD"}
	answer = switcher.get(answer)
	
	return answer


def return_id_v2(image_question):
	# import random
	# name_r=str(random.randint(2000000,3000000))
	# cv2.imwrite("id_50/id_ffqwaasa{}.jpg".format(name_r),image_question)
	

	image_question = cv2.resize(image_question, (25, 400))
	image_question = (image_question / 255).astype('float32')
	image_question = np.expand_dims(image_question, axis=0)
	predict = model_id.predict(image_question)
	answer = np.argmax(predict)
	switcher = {0: "0", 1: "1", 3: "2", 4: "3", 5: "4", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9", 2: "?"}
	answer = switcher.get(answer)
	
	return answer


def test_predict(path):
	k = 0
	for name in listdir(path):
		if k == 20:
			break
		img = cv2.imread(path + name)
		return_answer(img)


# path="data_train/data_question_add/5/"
# test_predict(path)

# du doan
def examCode(imageEXamCode):
	height, width, depth = imageEXamCode.shape
	height_grid = int(height / 10)
	width_grid = int(width / 3)
	
	list_id = []
	for i in range(3):
		grid_i = imageEXamCode[:, width_grid * i + 2:width_grid * (i + 1) - 2]
		

		id = return_id_v2(grid_i)
		list_id.append(id)
	
	id = ""
	for i in range(3):
		if i == 2:
			id = id + list_id[i] + "|"
		else:
			id = id + list_id[i]
	
	# print("id_exam : {}".format(id))
	return id


def id_student(image_id_student):
	height, width, depth = image_id_student.shape
	height_grid = int(height / 10)
	width_grid = int(width / 6)
	list_id = []
	
	for i in range(6):
		grid_i = image_id_student[:, width_grid * i + 2:width_grid * (i + 1)]
		id = return_id_v2(grid_i)
		list_id.append(id)
	
	id = ""
	for i in range(6):
		if i == 5:
			id = id + list_id[i] + "|"
		else:
			id = id + list_id[i]
	# print("id_student : {}".format(id))
	return id


def table_question4050(image_4050, num_question, table_k):
	height, width, depth = image_4050.shape
	if num_question == 40:
		a_table_questions = 14
		check_question = 14 * table_k
		height_grid = int(height / a_table_questions)
	else:
		a_table_questions = 17
		check_question = 17 * table_k
		height_grid = int(height / a_table_questions)
	width_grid = int(width / 5)
	list_answers = []
	image_4050 = image_4050[int(height_grid / 4):-int(height_grid / 2.3), width_grid:-2]
	
	height, width, depth = image_4050.shape
	height_grid = int(height / a_table_questions)
	
	if check_question == 42:
		for i in range(a_table_questions - 2):
			grid_i = image_4050[height_grid * i:height_grid * (i + 1)]
			answers = return_answer(grid_i)
			list_answers.append(answers)
		questions = ""
		for i in range(a_table_questions - 2):
			questions = questions + list_answers[i] + "|"
		
	elif check_question == 51:
		for i in range(a_table_questions - 1):
			grid_i = image_4050[height_grid * i:height_grid * (i + 1)]
			answers = return_answer(grid_i)
			list_answers.append(answers)
		questions = ""
		for i in range(a_table_questions - 1):
			questions = questions + list_answers[i] + "|"
		
	else:
		for i in range(a_table_questions):
			grid_i = image_4050[height_grid * i:height_grid * (i + 1)]
			answers = return_answer(grid_i)
			list_answers.append(answers)
		questions = ""
		for i in range(a_table_questions):
			questions = questions + list_answers[i] + "|"
		
	return questions


def table_question120(image_table):
	height, width, depth = image_table.shape
	height_grid = int(height / 6)
	list_answers = []
	for i in range(6):
		grid_i = image_table[height_grid * i:height_grid * (i + 1)]
		height_b, width_b, depth2 = grid_i.shape
		height_box = int(height_b / 5)
		width_box = int(width_b / 5)
		
		grid_i = grid_i[int(height_box / 2):height_b - int(height_box / 3), width_box:width_box * 5 - 2]
		height_b, width_b, depth2 = grid_i.shape
		height_box = int(height_b / 5)
		# width_box = int(width_b / 5)
		
		for j in range(5):
			grid_ij = grid_i[height_box * j:height_box * (j + 1)]
			answers = return_answer(grid_ij)
			list_answers.append(answers)
	questions = ""
	for i in range(30):
		questions = questions + list_answers[i] + "|"
	# print("table : {}".format(questions))
	return questions


def get_id():
	import os
	k = 0
	
	for name in listdir("Anh_Nghieng/"):
		if os.path.isfile("dataTrain/" + name):
			image = cv2.imread("dataTrain/" + name)
			approxs, area_cnt = export_appproxs120(image)
			if len(approxs) == 8:
				image_exam = four_point_transform(image, approxs[-1])
				image_st = four_point_transform(image, approxs[-2])
				examCode(image_exam, k)
				id_student(image_st, k)
			elif len(approxs) == 7:
				image_st = four_point_transform(image, approxs[-1])
				id_student(image_st, k)
			else:
				continue
			k += 1


def get_id_in():
	import os
	path = "New folder/"
	k = 0
	for name in listdir(path):
		if os.path.isfile(path + name):
			image = cv2.imread(path + name)
			h, w, d = image.shape
			image = image[:int(h / 2.5), int(w * 3 / 5):]
			approxs, area_cnt = export_appproxs120(image)
			if len(approxs) == 1:
				image_st = four_point_transform(image, approxs[0])
				id_student(image_st, k)
				k += 1
			elif len(approxs) > 1:
				image_st = four_point_transform(image, approxs[0])
				image_exam = four_point_transform(image, approxs[1])
				examCode(image_exam, k)
				id_student(image_st, k)
				k += 1

# get_id_in()
