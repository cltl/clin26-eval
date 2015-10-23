CLIN26, CONLL NERC and NED evaluation

Structure:
- `score-XXX.py` (Python scripts for evaluation)
- `key` (folder containing golden data)
- `response` (folder containing predicted data)

The script uses relative pathes. Do not change the structure and follow the instructions below:

1. Put golden files in `key` folder and your output in the `response` folder 
next to the key folder. (For factuality data, put them inside a subfolder called `factuality`. For NERC/NED data, put them inside a subfolder called `ne`.)
2. Make sure the files have the same name as the files in the key folder (otherwise, they will be ignored.)
3. Response and key files need to be in the CONLL 2011 format
4. `cd` into `clin26-eval` folder and run the score-XXX.py script from the command line 

The script write report to default `stdout` and `stderr`.
