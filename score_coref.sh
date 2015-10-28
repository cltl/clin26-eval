#check if enough arguments are passed, else print usage information
if [ $# -eq 0 ];
then
    echo
    echo "Usage:       : $0 task key"
    echo
    echo "task         : coref | coref_event | coref_ne"
    echo "key          : full path to folder containing system submission files. For every gold file, there needs to be a key file."
    echo "gold         : full path to folder containing gold files"
    echo "measurement  : blanc"
    exit -1;

fi

export cwd=/${PWD#*/}
export task=$1
export key=$2
export gold=$3
export measurement=$4


echo removing old result files from response folder
echo
rm -f $gold/*.result

echo running coref scorer on key and response files
echo
for gold_file in $gold/*.$task;
    do
        basename=${gold_file##*/}
        key_file=$key/$basename
        perl ./coref_scorer/coref-scorer-v8.01/scorer.pl $measurement $gold_file $key_file > $gold_file.result
    done

echo collecting the results from individual files
echo
java -cp ./coref_scorer/lib/collect-results.jar eu.newsreader.result.CollectResults --result-folder $gold --extension ".result" --label trial

echo sending overall results to stdout
echo
cat $(dirname "$gold")/trialresults.csv
