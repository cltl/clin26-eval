CLIN26, CONLL coreference for entities

Structure:
- coref-scorer-v8.01 (CONLL2011 scorer for coreference)
- lib (Java code to collect the results)
- task (folder with the key and response files)
trial.score.sh (script to get the scores)

The script uses relative pathes. Do not change the structure since it breaks the relative paths.
Follow the instructions below:

1. Put your output in the response folder next to the key folder
2. Make sure the files have the same name as the files in the key folder
3. Response and key files need to be in the CONLL 2011 format

The script runs the coref scorer from CONLL2011 on the files in the key folder and the response folder with the same name.
Changing the names will break the script. The result is stored in a *.result file for each response file in the response folder.
We run the coref scorer with the BLANC metrics. The scorer reports the mentions and the reference sets, where the latter consists of coreferential mentions (coreference sets)
and mentions without coreference relations (noreference). The BLANC score averages the COREFERENCE and NOREFERENCE results.

The script creates a trial.result.csv file in the task folder with an overview based on the all the results file in the response folder.
We provide two overviews:

- microaverage over the sums of all mentions, reference results, coreference results and noreference results in each result file
- macroaverage over the BLANC results of each result file and the mentions



