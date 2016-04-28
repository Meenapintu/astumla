all:	run

run:
	python csv_split.py
	python scriptdatatrain2.py
	python scriptdatatest2.py
	python proj.py


