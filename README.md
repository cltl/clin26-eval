CLIN26, CONLL NERC, NED, and factuality evaluation

1. Put golden files and the response of your system in two separate, flat directories
2. Make sure the files in response directory have the same name as those in key directory (otherwise, they will be ignored)
3. Response and key files need to be in the CONLL 2011 format
4. Run the `score-XXX.py` script from the command line, providing path to key and response dir 

The script write report to default `stdout` and `stderr`.
