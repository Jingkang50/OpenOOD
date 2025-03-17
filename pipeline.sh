rm -r model_info 

cd utils
python convert.csv.to.json.py
python convert.json.to.html.py