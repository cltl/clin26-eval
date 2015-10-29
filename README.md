<h1>CLIN26, CONLL NERC, NED, COREF, and factuality evaluation</h1>

<h3>Terminology</h3>

Key and response:
<p>1. **key**: expected output</p>
<p>2. **response**: system output</p>


The folder **dev_corpora** contains six folders:
<p>1. **factuality**: factuality data</p>
<p>2. **ne**: named entity data</p>
<p>3. **e**: entity data</p>
<p>4. **coref_ne**: named entity coreference</p>
<p>5. **coref_event**: event coreference</p>
<p>6. **coref**: named entity and event coreference</p>

<h3>General instructions</h3>
<p>1. Put your response files in one flat directory (dev corpus and test corpus will be provided in this repository)</p>

<p>2. Make sure the files in the response directory have the same name as those in key directory (otherwise, they will be ignored)</p>

<p>3. Response and key files need to be in the CONLL 2011 format</p>

<p>4. A response file needs to be provided for each key file.</p>

<p>5. Run the `score-XXX.py` or `score-XXX-sh` script from the command line. Run each script without commands to see information about how to run the script.</p>


The script write report to default `stdout` and `stderr`.

<h3>Task-specific instructions</h3>

<h4> Entity coreference</h4>

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: /dev_corpora/coref_ne.

The measurement is always: blanc.


<h4> Event coreference</h4>

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: /dev_corpora/coref_event.

The measurement is always: blanc.

<h4> Entity and event coreference</h4>

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: /dev_corpora/coref.

The measurement is always: blanc.
