#check if enough arguments are passed, else print usage information
if [ $# -eq 0 ];
then
    echo
    echo "Usage:       : $0 task key response measurement"
    echo
    echo "task         : coref | coref_event | coref_ne"
    echo "key          : full path to folder containing key files."
    echo "response     : full path to folder containing response files. A response file needs to be provided for all key files"
    echo "measurement  : blanc"
    exit -1;

fi

export cwd=/${PWD#*/}
export task=$1
export key=$2
export response=$3
export measurement=$4


echo removing old result files from response folder
echo
rm -f $key/*.result

echo running coref scorer on key and response files
echo
for key_file in $key/*.$task;
    do
        basename=${key_file##*/}
        response_file=$response/$basename
        perl ./coref_scorer/coref-scorer-v8.01/scorer.pl $measurement $key_file $response_file > $key_file.result
    done

echo collecting the results from individual files
echo
java -cp ./coref_scorer/lib/collect-results.jar eu.newsreader.result.CollectResults --result-folder $key --extension ".result" --label trial

echo sending overall results to stdout
echo
cat $(dirname "$key")/trialresults.csv
