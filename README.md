## CLIN-26: CONLL NERC, NED, COREF, and Factuality evaluation

### Terminology

Key and response:

1. **key**: expected output
2. **response**: system output


The folder **dev_corpora** contains six folders:

1. **factuality**: factuality data
2. **ne**: named entity data
3. **e**: entity data
4. **coref_ne**: named entity coreference
5. **coref_event**: event coreference
6. **coref**: named entity and event coreference

## General instructions

1. Put your response files in one flat directory (dev corpus and test corpus will be provided in this repository)
2. Make sure the files in the response directory have the same name as those in key directory (otherwise, they will be ignored)
3. Response and key files need to be in the CONLL 2011 format
4. A response file needs to be provided for each key file.
5. Run the `score-XXX.py` or `score-XXX.sh` script from the command line. Run each script without commands to see information about how to run the script.

The script write report to default `stdout` and `stderr`.

### Task-specific instructions

#### Entity coreference

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: `/dev_corpora/coref_ne`.

The measurement is always: `blanc`.

#### Event coreference

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: `/dev_corpora/coref_event`.

The measurement is always: `blanc`.

#### Entity and event coreference

```bash
bash score_coref.sh task key response measurement
```

The development key folder can be found at: `/dev_corpora/coref`.

The measurement is always: `blanc`.

#### Named-Entity Recognition and Classification (NERC)

```bash
python score_nerc.py key response
```

#### Named-Entity Disambiguation (NED)

```bash
python score_nerc.py key response
```

#### Factuality

```bash
python score_factuality.py key response
```
