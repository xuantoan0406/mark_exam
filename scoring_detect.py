import process_image as pi
import cv2
import numpy as np
import sys
path_output = sys.argv[4]

def write_file(predict):
	with open(path_output+"/"+"out.csv", mode='a') as csv_file:
		csv_file.write(predict)

# return answer
def answer120(image, approxs_table_question_sorted):
	list_answers = ""
	for i in range(5):
		image_table = pi.four_point_transform(image, approxs_table_question_sorted[i])

		height, width, depth = image_table.shape
		if height / width > 3:
			tables = pi.table_question120(image_table)
			
			list_answers+=tables
	return list_answers


def answer4050(image, approxs_table_question_sorted, numberAnswer):
	list_answers = ""
	table_k=1
	for i in range(3):
		image_table = pi.four_point_transform(image, approxs_table_question_sorted[i])

		height, width, depth = image_table.shape
		if height / width > 3:
			tables = pi.table_question4050(image_table,numberAnswer,table_k)
			list_answers+=tables
			table_k+=1
	return list_answers


def predict_exam120(image, approxs):
	
	if len(approxs) == 8:
		image_id_exams=pi.four_point_transform(image, approxs[-1])
		height, width, depth=image_id_exams.shape
		if height/width>3:
			id_exams = pi.examCode(image_id_exams)
		else:
			id_exams="???|"
		image_id_student=pi.four_point_transform(image, approxs[-2])
		height, width, depth = image_id_student.shape
		if height/width>2:
			id_students = pi.id_student(image_id_student)
		else:
			id_students="???|"
			
	else:
		print("ID exam : ?")
		print("ID student : ?")
		id_exams = "???|"
		id_students = "??????|"
		
	
	# 4box 4 bang cau hoi
	if len(approxs) >= 5:
		approxs_sort=[]
		for i in range(5):
			approxs_sort.append(pi.order_points(approxs[i]))
		
		approxs_table_question_sorted = sorted(approxs_sort[:5], key=np.sum, reverse=False)
		#cv2.drawContours(image, approxs_table_question_sorted, -1, (0, 0, 255), 3)
		questions = answer120(image, approxs_table_question_sorted)
		
		
	else:
		questions=all_None(120)
		print("No Detect")
	return id_exams, id_students, questions


def all_None(num_question):
	question=""
	for i in range(num_question):
		question +="?|"
	return question

def predict_exam4050(image, approxs, number):
	if len(approxs) == 6:
		id_exams = pi.examCode(pi.four_point_transform(image, approxs[-1]))
		id_students = pi.id_student(pi.four_point_transform(image, approxs[-2]))
	else:
		print("ID exam : ?")
		print("ID student : ?")
		id_exams = "???|"
		id_students = "??????|"
		
	# 3box 3 bang cau hoi
	if len(approxs) >= 3:
		
		approxs_table_question_sorted = sorted(approxs[:3], key=np.sum, reverse=False)
	
		questions = answer4050(image, approxs_table_question_sorted, number)
		
	else:
		print("sorted")
		questions=all_None(number)
		
	return id_exams, id_students, questions


def example_exam120(image):
	height, width, depth = image.shape
	acreage = height * width
	approxs, area_cnt = pi.export_appproxs120(image)
	if area_cnt[0] / acreage < 0.2:
		id_exams, id_students, questions = predict_exam120(image, approxs)
		return id_exams, id_students, questions
	
	elif len(approxs) == 0:
		id_exams = "???|"
		id_students = "??????|"
		questions = all_None(120)
		print("No approxs")
		return id_exams, id_students, questions
	
	else:
		
		image_transform = pi.four_point_transform(image, approxs[0])
		print("detect paper")
		approxs, area_cnt = pi.export_appproxs120(image_transform)
		id_exams, id_students, questions = predict_exam120(image_transform, approxs)
		return id_exams, id_students, questions


def example_exam4050(image, number):
	height, width, depth = image.shape
	acreage = height * width
	approxs, area_cnt = pi.export_appproxs4050(image)
	if area_cnt[0] / acreage < 0.2:
		id_exams, id_students, questions = predict_exam4050(image, approxs, number)
		return id_exams, id_students, questions
	elif len(approxs) == 0:
		print("No Detect")
		id_exams = "???|"
		id_students = "??????|"
		questions = all_None(number)

		return id_exams, id_students, questions
	
	else:
		image_transform = pi.four_point_transform(image, approxs[0])
		approxs, area_cnt = pi.export_appproxs4050(image_transform)
		id_exams, id_students, questions = predict_exam4050(image_transform, approxs, number)
		return id_exams, id_students, questions




