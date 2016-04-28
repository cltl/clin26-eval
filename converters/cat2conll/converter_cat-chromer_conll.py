import xml.etree.ElementTree as etree
import argparse
from argparse import RawTextHelpFormatter
import collections
import logging
import os
'''
goal:

(t_id) ->
    'token_number'             -> 1. Token number # done
    'word'                     -> 2. word # done
    'sent_id'                  -> sent_id # done
    'ne_coreference'
        'ent_type'                 -> 3. named entity label # done
        'position'                 -> first | in_between | last # done
        'external_ref'             -> 4. ned link # done
        'target_m_id'              -> target m_id of entity # done
    'event_coreference'
        'position'                 -> first | in_between | last # done
        'target_m_id'              -> m_id of target event # done
    #factuality
    'event_mark'               -> 3. E | _ # done
    'certainty'                -> certainty # done
    'polarity'                 -> polarity # done
    #ne
    'ent_type'                 -> ent_type #done
    'exterenal_ref'            -> external_ref #done
    #coreference
    'coref_string\             -> string #done

preloaded dicts needed:
information where to find what:

## NER NED
#Chromer: Markables/ENTITY_MENTION
<ENTITY_MENTION comment="" head="" m_id="22" syntactic_type="NAM">
    <token_anchor t_id="17"/>
    <token_anchor t_id="16"/>
</ENTITY_MENTION>

#Chromer: Markables/ENTITY
<ENTITY
TAG_DESCRIPTOR="Steve Jobs"
ent_type="PER"
external_ref="http://dbpedia.org/resource/Steve_Jobs "
instance_id="PER19274271332322852" m_id="105"
/>

#Chromer: Relations/REFERS_TO
<REFERS_TO comment="" r_id="263541">
  <source m_id="22"/>
  <source m_id="24"/>
  <source m_id="58"/>
  <target m_id="105"
/>

## FACTUALITY
#Chromer: Markables/EVENT_MENTION
<EVENT_MENTION
aspect=""
certainty="CERTAIN"
comment=""
m_id="46"
modality=""
polarity="POS"
pos="VERB"
pred="aankondigen"
special_cases="NONE"
tense="PAST"
time="NON_FUTURE">
    <token_anchor t_id="25"/>
    <token_anchor t_id="28"
/>

## ENTITY/EVENT COREFERENCE
<REFERS_TO r_id="258912">
   <source m_id="49"/>
   <target m_id="54"
/>


'''
#parse user input
message = '''Converter from Cromer to conll format for:
ner_ned, factuality, and entity and event coreference.
It should work with python2.7 and python3.4 without external libraries.

WARNING: only sentences 0 till 6 are converted!
WARNING: for factuality: all polarity seems to POS and all certainty \
seems to be CERTAIN.

information is logged if (written to args.output_path.log.task):
(1) if an entity_mention has no annotation
(2) no polarity or certainty value in event mention
(3) if one t_id refers to different external_refs
'''

parser = argparse.ArgumentParser(description=message,
                                 formatter_class=RawTextHelpFormatter)
for identifier,help_message in [('cromer','path to Chromer file'),
                                ('output_path','path to output'),
                                ('task','e | ne | factuality | coref | coref_event | coref_ne')]:
    parser.add_argument(identifier, help=help_message)
parser.add_argument('release', help='release | gold',)

args = parser.parse_args()

#flags to extract coref info for ne and event
perform_coref_ne = True
perform_coref_event = True

if args.task == 'coref_event':
    perform_coref_ne = False
if args.task == 'coref_ne':
    perform_coref_event = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a file handler
handler = logging.FileHandler(args.output_path+'.log.'+args.task,mode="w")
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s - %(name)s  - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

#parse document
doc = etree.parse(args.cromer)
data = collections.defaultdict(dict)
Identifier = collections.namedtuple('Identifier', ['token_id'])

#preload dicts
entity_dict = {entity_el.get('m_id') : {'ent_type' : entity_el.get('ent_type',
                                                     default="_"),
                                 'external_ref' : entity_el.get('external_ref',
                                                                default="_")}
               for entity_el in doc.iterfind('Markables/ENTITY')
                }

refers_to_dict = { source_el.get('m_id') : refers_el.find('target').get('m_id')
                  for refers_el in doc.iterfind('Relations/REFERS_TO')
                  for source_el in refers_el.iterfind('source')}

coreference_dict = {}

#loop cromer xml and add general info about t_id
for token_el in doc.iterfind('token'):
    identifier = Identifier(int(token_el.get('t_id')))
    data[identifier]['sent_id'] = int(token_el.get('sentence'))
    data[identifier]['word'] = token_el.text
    data[identifier]['number'] = int(token_el.get('number'))

    data[identifier]['event_mark'] = '_'
    data[identifier]['event_mark_updated'] = '_'
    data[identifier]['random'] = '_'
    data[identifier]['factuality'] = '_'
    data[identifier]['certainty'] = '_'
    data[identifier]['certainty_updated'] = '_'
    data[identifier]['polarity'] = '_'
    data[identifier]['polarity_updated'] = '_'
    data[identifier]['ent_type'] = '_'
    data[identifier]['external_ref'] = '_'

    data[identifier]['event_coreference'] = []
    data[identifier]['ne_coreference'] = []

#add ne information
for category,xml_path in [('event_coreference','Markables/EVENT_MENTION'),
                          ('ne_coreference','Markables/ENTITY_MENTION')]:
    for el in doc.iterfind(xml_path):
        source_m_id = el.get('m_id')
        if source_m_id in refers_to_dict:
            refers_to = refers_to_dict[source_m_id]

            #check if args.task == 'ne' and filter accordingly
            syn_type = el.get('syntactic_type')
            if all([args.task == 'ne',
                    syn_type not in ['NAM','PRE.NAM']]):
                continue

            #position + mw
            anchor_els = el.findall('token_anchor')
            num_els = len(anchor_els)
            t_ids = sorted([int(anchor_el.get('t_id'))
                     for anchor_el in anchor_els])

            #check here if entity or event reference is already in dict
            event_identifier = ()
            for t_id in t_ids:
                event_identifier += (t_id,)
            event_identifier = (category,event_identifier)

            if event_identifier in coreference_dict:
                basename = os.path.basename(args.cromer)
                logger.debug('the t_ids %s from file %s  were annotated twice with the same annotation' % (event_identifier,basename))
                continue
            else:
                coreference_dict[event_identifier] = ''

            #factuality data
            factuality = {}
            if xml_path == 'Markables/EVENT_MENTION':
                factuality = {key : el.get(key)
                              for key in ['certainty','polarity']}
                factuality['event_mark'] = "E"

            for counter,t_id in enumerate(t_ids,1):
                if num_els == 1:
                    position = '(%s)'
                elif counter == 1:
                    position = '(%s'
                elif counter == num_els:
                    position = '%s)'
                else:
                    position = '%s'
                t_id = Identifier(t_id)


                info = {'position' : position,
                        'source_m_id' : source_m_id,
                        'target_m_id' : refers_to}
                info.update(factuality)

                #entity
                if category == 'ne_coreference':
                    if refers_to in entity_dict:
                        entity_info = entity_dict[refers_to]

                        if entity_info['ent_type'] == '':
                            entity_info['ent_type'] = '_'

                        if entity_info['external_ref'] == '':
                            entity_info['external_ref'] = '_'
                        info.update(entity_info)
                        data[t_id]['ne_coreference'].append(info)
                else:
                    data[t_id]['event_coreference'].append(info)

        else:
            logger.debug('m_id '+ source_m_id +' has no annotation:')
            logger.debug(etree.tostring(el))

for t_id,info in sorted(data.items()):

    #add ne data to dict
    ne_coref = info['ne_coreference']
    if len(ne_coref) >= 1:
        set_external_refs = set([value['external_ref']
                            for value in ne_coref])
        if '_' in set_external_refs:
            set_external_refs.remove('_')

        ent_types = []
        for value in ne_coref:
            if value['ent_type'] not in ['','_']:
                ent_types.append(value)

        if ent_types:
            ent_type = '|'.join([ne_info['position'] % ne_info['ent_type']
                                     for ne_info in ent_types])
            data[t_id]['ent_type'] = ent_type

        if len(set_external_refs) == 1:
            for external_ref in set_external_refs:
                pass
            data[t_id]['external_ref'] = external_ref

        elif len(set_external_refs) >= 2:
            logger.debug('different or external refs for %s:' % t_id)
            logger.debug(set_external_refs)
            logger.debug(info['ne_coreference'])
            data[t_id]['external_ref'] = '_'

    #add coreference data to dict
    target_m_ids = []
    for coref,perform in [(info['ne_coreference'],perform_coref_ne),
                          (info['event_coreference'],perform_coref_event)
                           ]:
        if all([perform,
                len(coref) >= 1]):
            target_m_ids += [value['position'] % value['target_m_id']
                             for value in coref]

    coref_string = "|".join(target_m_ids)
    if not coref_string:
        coref_string = '_'
    data[t_id]['coref_string'] = coref_string


#change factuality tags
for t_id,info in data.items():
    event_coreferences = info['event_coreference']
    if event_coreferences:

        factuality = collections.defaultdict(list)
        for event_coreference in event_coreferences:
            template = 'B-%s'
            if event_coreference['position'] in ['%s','%s)']:
                template = 'I-%s'
            for key in ['event_mark','certainty','polarity']:
                value = event_coreference[key]
                if value:
                    value = template % value
                else:
                    value = '_'
                factuality[key].append(value)

        event_mark_string = ' '.join(factuality['event_mark'])
        certainty_string = ' '.join(factuality['certainty'])
        polarity_string = ' '.join(factuality['polarity'])
        data[t_id]['event_mark'] = event_mark_string
        data[t_id]['certainty'] = certainty_string
        data[t_id]['polarity'] = polarity_string
        
        source_m_id = '-' + event_coreferences[0]['source_m_id']
        data[t_id]['event_mark_updated'] = event_mark_string + source_m_id
        data[t_id]['certainty_updated'] = certainty_string + source_m_id
        data[t_id]['polarity_updated'] = polarity_string + source_m_id

        #if len(factuality['certainty']) >= 2:
        #    print(data[t_id])
        #    input('continue?')

#export function
def export(output_path,d,keys,max_sent_id=7):
    '''
    export function

    WARNING: for all coreference tasks:
    (1) the first line of the output is: #begin document (basename);
    (2) the last line of the output is: #end document

    @type  output_path: str
    @param output_path: output path (.ne , .coref, .factuality)

    @type  d: collections.defaultdict
    @param d: all annotation info

    @type  keys: list
    @param keys: keys to export (in correct order)
    starting with the third columns (first two are always token id and word

    @type max_sent_id: int
    @param max_sent_id: default print sentences 0 till 5.
    '''
    cur_sent_id = 0
    basename = os.path.basename(args.output_path)

    with open(output_path,'w') as outfile:

        if args.task.startswith('coref'):
            first_line = '#begin document ({basename});\n'.format(**locals())
            outfile.write(first_line)

        for t_id,info in sorted(d.items()):

            prefix = ''
            sent_id = info['sent_id']

            if sent_id > max_sent_id:
                continue

            if sent_id != cur_sent_id:
                prefix = '\n'

            export = []
            if args.task.startswith('coref'):
                export.append(basename)
            export += [str(t_id.token_id),info['word']]

            if args.release == 'release':

                if args.task == 'factuality':
                    event_mark = info['event_mark']
                    export.append(event_mark)
                    export += ['_' for key in keys[1:] ]

                else:
                    export += ['_' for key in keys]

            else:
                export += [info[key] if info[key] else '_'
                           for key in keys]
            outfile.write(prefix+'\t'.join(export)+'\n')

            cur_sent_id = sent_id

        if args.task.startswith('coref'):
            outfile.write('#end document')


if args.task.startswith('coref'):
    keys = ['coref_string']
elif args.task == 'factuality':
    keys = ['event_mark','certainty','polarity']
elif args.task == 'factuality_updated':
    keys = ['event_mark_updated','certainty_updated','polarity_updated']
elif args.task == 'e':
    keys = ['ent_type','external_ref']
elif args.task == 'ne':
    keys = ['ent_type','external_ref']
elif args.task == 'event_detection':
    keys = ['event_mark','random','random']
export(args.output_path,data,keys)
