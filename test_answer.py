import sys
import scoring_detect
from os import listdir
import cv2
import os
from sklearn.metrics import accuracy_score

def except_error(num):
	ls=""
	for i in range(num+9):
		ls += "?|"
	return ls


def read_file(path):
	with open(path, mode="r") as file:
		a = file.read()
	i = 0
	answer = ""
	ls_answers = []
	for j in a:
		
		if j == "|":
			if answer != "":
				ls_answers.append(answer)
				answer = ""
			i += 1
		else:
			answer += j
	print(len(ls_answers))
	return ls_answers


def read_questions(questions):
	answer = ""
	ls_answers = []
	for j in questions:
		
		if j == "|":
			if answer != "":
				ls_answers.append(answer)
				answer = ""
		else:
			answer += j
	return ls_answers

def predict_folder(path_input):
	if number == 120:
		for name in listdir(path_input):
			if os.path.isfile(path_input + "/" + name):
				try:
					image = cv2.imread(path_input + "/" + name)
					id_exams, id_students, questions = scoring_detect.example_exam120(image)
					print(path_input + "/" + name + "|1|" + id_students + id_exams + questions + "\n")
					answers=read_file("answer/exam_120.txt")
					answers_predict=read_questions(questions)
					if len(answers)==len(answers_predict):
						accuracy=accuracy_score(answers,answers_predict)
						print("accuracy :{}".format(accuracy))
					scoring_detect.write_file(
						path_input + "/" + name + "|1|" + id_students + id_exams + questions + "\n")
				except:
					ls=except_error(number)
					scoring_detect.write_file(
						path_input + "/" + name + "|1|" + ls + "\n")
					print("error yeah !!!!")
	elif number == 40 or number == 50:
		for name in listdir(path_input):
			if os.path.isfile(path_input + "/" + name):
				try:
					image = cv2.imread(path_input + "/" + name)
					id_exams, id_students, questions = scoring_detect.example_exam4050(image, number)
					print(path_input + "/" + name + "|1|" + id_students + id_exams + questions + "\n")
					answers = read_file("answer/exam_{}.txt".format(number))
					answers_predict = read_questions(questions)
					if len(answers) == len(answers_predict):
						accuracy = accuracy_score(answers, answers_predict)
						print("accuracy :{}".format(accuracy))
					scoring_detect.write_file(
						path_input + "/" + name + "|1|" + id_students + id_exams + questions + "\n")
				except:
					ls=except_error(number)
					scoring_detect.write_file(
						path_input + "/" + name + "|1|" + ls + "\n")
					print("error yeah !!!!")
	else:
		print("sai mau")
		return "sai mau"


if len(sys.argv) >= 7:
	path_input = sys.argv[2]
	path_output = sys.argv[4]
	number = int(sys.argv[6])
	if not os.path.isdir(path_output):
		os.mkdir(path_output)
	if sys.argv[1] == "--inDir" and sys.argv[3] == "--outDir" and sys.argv[5] == "--numOfQuestion":
		predict_folder(path_input)
		with open(path_output + "/" + "result.csv", mode='a') as csv_file:
			csv_file.write("done|")
	else:
		print("main.py + --inDir +path_inDir + --outDir + path_outDir+ --numOfQuestion +num + --multi +num_multi")
else:
	print("error syntax")

