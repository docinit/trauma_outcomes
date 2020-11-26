# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import pandas as pd
from flask import render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import numpy as np
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# initialize our Flask application and the model
app = flask.Flask(__name__)

# Классификатор
cur_dir = os.path.dirname(__file__)
# постоянное значение, вычисленное во время обучения модели
threshold = 0.5540086148648292
clf = pickle.load(open('/app/app/models/prognose_model10_2020_SVM.pkl','rb'))
scaler = pickle.load(open('/app/app/models/sc_scaler10_2020_SVM_mif.pkl','rb'))
file_saver = os.path.join(cur_dir,'outcome_saver.txt')
def classifier(document):
	dictionary = {True:'Выживет',False:'Смерть'}
	X = scaler.transform([document])
	y_coef = clf.predict_proba(X)
	y = y_coef[:,1]>threshold
	y = dictionary[y[0]]
	y_coef = y_coef[:,1]
	return y, y_coef[0]



class RequestForm(Form):
	fall = TextAreaField('',[validators.DataRequired()])
	gcs = TextAreaField('',[validators.DataRequired()])
	age = TextAreaField('',[validators.DataRequired()])
	sbp = TextAreaField('',[validators.DataRequired()])
	cr = TextAreaField('',[validators.DataRequired()])
	spo2 = TextAreaField('',[validators.DataRequired()])
	fio2 = TextAreaField('',[validators.DataRequired()])
	tr = TextAreaField('',[validators.DataRequired()])
	ast = TextAreaField('',[validators.DataRequired()])
	cl = TextAreaField('',[validators.DataRequired()])
	fen = TextAreaField('',[validators.DataRequired()])
	amyl = TextAreaField('',[validators.DataRequired()])
	prop = TextAreaField('',[validators.DataRequired()])
	pl = TextAreaField('',[validators.DataRequired()])
	neuro = TextAreaField('',[validators.DataRequired()])
	pin = TextAreaField('',[validators.DataRequired()])
	dofamin = TextAreaField('',[validators.DataRequired()])
	sex = TextAreaField('',[validators.DataRequired()])
	

@app.route('/')
def index():
	form = RequestForm(request.form)
	return render_template('trauma_app.html', form = form)
	
@app.route('/results',methods=['POST'])

def results():
	form = RequestForm(request.form)
	if request.method == 'POST' and form.validate():
		fall = float(request.form['fall'])
		gcs = float(request.form['gcs'])
		age = float(request.form['age'])
		sbp = float(request.form['sbp'])
		cr = float(request.form['cr'])
		spo2 = float(request.form['spo2'])
		fio2 = float(request.form['fio2'])
		tr = float(request.form['tr'])
		ast = float(request.form['ast'])
		cl = float(request.form['cl'])
		fen = float(request.form['fen'])
		amyl = float(request.form['amyl'])
		prop = float(request.form['prop'])
		pl = float(request.form['pl'])
		neuro = float(request.form['neuro'])
		pin = float(request.form['pin'])
		dofamin = float(request.form['dofamin'])
		sex = float(request.form['sex'])
		coma = gcs<8
		amyl_grade = amyl>=40
		cl_grade_2 = cl>=110
		x = 0
		# PEMOD, компонент GCS
		if gcs>=12:
			x+=0
		elif gcs in [10,11]:
			x+=1
		elif gcs in [8,9]:
			x+=2
		elif gcs in [6,7]:
			x+=3
		elif gcs <7:
			x+=4
		# PEMOD, компонент SBP
		if age<1/12:
			if sbp>73:
				x+=0
			elif sbp>67:
				x+=1
			elif sbp>58:
				x+=2
			elif sbp>44:
				x+=3
			else:
				x+=4
		elif age<12:
			if sbp>82:
				x+=0
			elif sbp>77:
				x+=1
			elif sbp>70:
				x+=2
			elif sbp<=57:
				x+=3
			else:
				x+=4
		else:
			if sbp>93:
				x+=0
			elif sbp>88:
				x+=1
			elif sbp>79:
				x+=2
			elif sbp>66:
				x+=3
			else:
				x+=4
		# PEMOD, компонент cr
		# перевод в др. единицы
		cr = cr/88.4173		
		if age<7/365:
			if cr<1.14:
				x+=0
			elif cr<=2.38:
				x+=1
			elif cr<=3.96:
				x+=2
			elif cr<=5.77:
				x+=3
			else:
				x+=4
		elif age<1:
			if cr<0.49:
				x+=0
			elif cr<=0.79:
				x+=1
			elif cr<=1.13:
				x+=2
			elif cr<=1.59:
				x+=3
			else:
				x+=4
		elif age<12:
			if cr<0.8:
				x+=0
			elif cr<=1.59:
				x+=1
			elif cr<=2.72:
				x+=2
			elif cr<=3.96:
				x+=3
			else:
				x+=4
		else:
			if cr<1.14:
				x+=0
			elif cr<=2.38:
				x+=1
			elif cr<=3.96:
				x+=2
			elif cr<=5.89:
				x+=3
			else:
				x+=4
		# PEMOD, компонент SPO2/FIO2
		dev = spo2 / fio2
		if dev > 300:
			x+=0
		elif dev >=226:
			x+=1
		elif dev >=151:
			x+=2
		elif dev >=76:
			x+=3
		else:
			x+=4
		# PEMOD, компонент Tr
		if tr >=120:
			x+=0
		elif tr >=81:
			x+=1
		elif tr >=51:
			x+=2
		elif tr >=26:
			x+=3
		else:
			x+=4
		# PEMOD, компонент AST
		if ast <=30:
			x+=0
		elif ast <=100:
			x+=1
		elif ast <=250:
			x+=2
		elif ast <=800:
			x+=3
		else:
			x+=4
		pemod = x
		
		prognose_requesting = [fall,pemod,cl_grade_2,fen,amyl_grade,coma,prop,pl,neuro,pin,dofamin,sex]
		print('Вы предоставили следующие данные:',prognose_requesting)
		y, y_coef = classifier(prognose_requesting)
		return render_template('results.html',
				content = prognose_requesting,
				prediction=y,
				probability=np.round(y_coef,4))
	return render_template('requestform.html',form=form)
	
	
@app.route('/thanks',methods=['POST'])

def feedback():
	feedback = request.form['feedback_button']
	prognose_requesting = request.form['prognose_requesting']
	prediction = request.form['prediction']
	dictionary = {'Выживет':1,'Смерть':0}
	y = dictionary[prediction]
	if feedback == 'Неверно':
		y=int(not(y))

	file_saver = 'outcome_saver.npy'
	try:
	    a = np.load(open(file_saver, 'rb'), allow_pickle=True)
	    np.save(file_saver, np.append(a.flatten(), prognose_requesting+str(y)))
	except Exception:
	    np.save(file_saver, prognose_requesting+str(y))
	return render_template('thanks.html')



if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
