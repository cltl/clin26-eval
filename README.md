CLIN26, CONLL NERC, NED, and factuality evaluation

1. Put golden files and the response of your system in two separate, flat directories
2. Make sure the files in response directory have the same name as those in key directory (otherwise, they will be ignored)
3. Response and key files need to be in the CONLL 2011 format
4. Run the `score-XXX.py` or `score-XXX-sh` script from the command line, providing path to key and response dir.
Run each script without commands to see information about how to run the script.

The script write report to default `stdout` and `stderr`.

5. The development corpora can be found in 'dev_gold_standards'.
