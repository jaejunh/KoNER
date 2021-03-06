#!../konerenv/bin/python
# -- coding: utf-8 --

#STDERR Printing
from __future__ import print_function
import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import io
import time
from datetime import datetime
from uuid import uuid4
from loader import make_gazette_to_dic, prepare_sentence

from utils import evaluate_lexicon_tagger, zero_digits
from loader import load_sentences
from model import Model
from konlpy.tag import Kkma, Mecab
from konlpy.utils import pprint as pp

import re
import optparse

import copy

from operator import itemgetter

import json
from functools import wraps
from flask import Flask,jsonify,abort,make_response,request,Response

import math
import redis


#############################################################################################
# GLOBAL VARIABLES
#############################################################################################

REDIS_DB=10
HOST='localhost'
POOL = redis.ConnectionPool(host=HOST, port=6379, db=REDIS_DB)

NER={}
VETO={}
FREQ={}
MAX_KEEP=30 # num of tag type *  num of page * 2 = 5 * 3 * 2 = 30
MAX_RETURN=5

#############################################################################################
# Redis DB or Standalone for session 
#############################################################################################
def init_ner_redis():
    eprint("...............  Init Redis({}) DB({})".format(HOST,REDIS_DB))
    try:
        r = redis.Redis(connection_pool=POOL)
        r.flushdb()
    except:
        eprint("---->Error:  Flushing Redis data")


def store_ner_redis(sessionkey,ner):
    s_ner = json.dumps(ner)
    try:
        r = redis.Redis(connection_pool=POOL)
        r.set(sessionkey,s_ner)
    except:
        eprint("Error:  Storing Redis data")
    return ner

def load_ner_redis(sessionkey):
    ner=[]
    try:
        r=redis.Redis(connection_pool=POOL)
        r_ner=r.get(sessionkey)
        if r_ner is not None and len(r_ner) > 0: ner=json.loads(r_ner)
    except:
        eprint("Error:  Fetching Redis data")
    #eprint("load_ner_redis")
    #pp(ner,stream=sys.stderr)
    return ner

def init_ner_dic(): pass

def store_ner_dic(sessionkey,ner):
    NER[sessionkey]=ner
    return ner

def load_ner_dic(sessionkey): return NER.get(sessionkey,[])

# Map Definition of db func 
load_NER=load_ner_redis
store_NER=store_ner_redis
init_NER=init_ner_redis


#############################################################################################

def read_VETO(filename):
    with io.open(filename,mode="r", encoding='utf-8') as f:
        VETO = { v[1]: 1 for line in f.read().splitlines() for v in [ line.split(',')[0:2] ] }
    return VETO 

def read_FREQ(filename): 
    with io.open(filename,mode="r", encoding='utf-8') as f:
        FREQ = { v[1]: int(v[0]) for line in f.read().splitlines() for v in [ line.split(',')[0:2] ] }
    #pp(FREQ.keys())
    #eprint("number of FREQ: {}".format(len(FREQ.keys())))
    return FREQ 



def as_json(status=200):
    def real_as_json(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            res = f(*args, **kwargs)
            res = json.dumps(res, ensure_ascii=False).encode('utf8')
            return Response(res, content_type='application/json; charset=utf-8', status=status)
        return wrapper
    return real_as_json 

#############################################################################################

app = Flask(__name__)


optparser = optparse.OptionParser()
optparser.add_option("-s", "--server", action="store_true", dest="server", default=False, help="Run Server")
optparser.add_option("-i", "--input", default="", help="Input file location")
optparser.add_option("-o", "--output", default="", help="Output file location")
optparser.add_option("-m", "--model", default="./model", help="Model file location")
optparser.add_option("-d", "--dictionary", default="./data/gazette", help="Dictionary file location")
optparser.add_option("-p", "--preprocess", default="0", help="Check input file format"
                                                            "PoS tagged text(=1) or raw text(=0)")

opts = optparser.parse_args()[0]
eprint('input file path : ', opts.input)
eprint('output file path : ', opts.output)

server_mode = opts.server
input_filename = opts.input
output_filename = opts.output
model_path = opts.model
dict_path = opts.dictionary
is_preprocess = opts.preprocess

assert server_mode or os.path.isfile(opts.input)
assert len(opts.preprocess) > 0


def split_sentence(input_filename,zeros):
    """
    Split raw document into sentences
    :return: list of sentences
    """
    sentences = []
    with open(input_filename, 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            if zeros:
                line=zero_digits(line)
            if line != '':
                line = re.split(r' *[\.\?!\|][\'"\)\]]* *', line)
                for l in line:
                    if len(l) > 0:
                        sentences.append(l)
    return sentences

def split_sentence_from_json(lines,zeros):
    """
    Split raw document into sentences
    :return: list of sentences
    """
    sentences = []
    if lines:
        for line in re.split(r'\n',lines):
            line = line.strip()
            if zeros:
                line=zero_digits(line)
            if line != '':
                line = re.split(r' *[\.\?!\|][\'"\)\]]* *', line)
                for l in line:
                    if len(l) > 0:
                        sentences.append(l)
    return sentences



def transform_pos(value, tagger='kkma'):
    """
    Transformation rule from KKMA to Sejong pos tag
    :param value: predicted pos tag (KKMA)
    :return: transformed pos tag (Sejong)
    """
    if tagger == 'kkma':
        if value == 'NNM':
            return 'NNB'
        elif value == 'VXV' or value == 'VXA':
            return 'VX'
        elif value == 'MDT' or value == 'MDN':
            return 'MM'
        elif value == 'MAC':
            return 'MAJ'
        elif value == 'JKM':
            return 'JKB'
        elif value == 'JKI':
            return 'JKV'
        elif value == 'EPH' or value == 'EPT' or value == 'EPP':
            return 'EP'
        elif value == 'EFN' or value == 'EFQ' or value == 'EFO' or value == 'EFA' or value == 'EFI' or value == 'EFR':
            return 'EF'
        elif value == 'ECE' or value == 'ECD' or value == 'ECS':
            return 'EC'
        elif value == 'ETD':
            return 'ETM'
        elif value == 'UN':
            return 'NF'
        elif value == 'UV':
            return 'NV'
        elif value == 'UE':
            return 'NA'
        elif value == 'OL':
            return 'SL'
        elif value == 'OH':
            return 'SH'
        elif value == 'ON':
            return 'SN'
        elif value == 'XPV':
            return 'XPN'
        else:
            return value
    elif tagger == 'mecab':
        if value == 'NNBC':
            return 'NNB'
        elif value == 'SSO' or value == 'SSC':
            return 'SS'
        elif value == 'SC':
            return 'SP'
        # elif value == 'SY':
        else:
            return value
          

def tag_pos(sentences, tagger='kkma'):
    """
    Predict Part-of-Speech tag of input sentences
    PoS tagger: KKMA
    :param sentences: list of input sentences
    :return: tagged sentences
    """
    if tagger == 'kkma':
        kkma = Kkma()
    elif tagger == 'mecab':
        mecab = Mecab()
                
    morph_lists = []
    for sent in sentences:
        morph_list = []
        if tagger == 'kkma':
            pos_tagged_sentences = kkma.pos(sent)
        elif tagger == 'mecab':
            pos_tagged_sentences = mecab.pos(sent)
            
        for (key, value) in pos_tagged_sentences:
            value = transform_pos(value, tagger)
            morph_list.append([key, value])
        morph_lists.append(morph_list)

    return morph_lists

def is_flush_tag(bioes):
    if bioes == "0": return True
    if bioes == "O": return True
    if bioes == "B": return True
    if bioes == "S": return True
    if bioes == "NOK": return True
    return False

def is_continue_tag(bioes):
    if bioes == "I": return True
    if bioes == "E": return True
    return False

def is_other_flush_tag(bioes):
    if bioes == "B": return True
    if bioes == "I": return True
    if bioes == "E": return True
    if bioes == "S": return True
    return False

    

def token_exist(sessionkey,token):
    if len(NER.get('sessionkey',[])) == 0:
        return False
    for i in NER['sessionkey']:
        if i['value'] == token:
            return True
    return False



def add_NER(sessionkey,ner,old_ner):
    #pp(ner,stream=sys.stderr)
    result=ner

    if len(old_ner) > 0:
        ner_unmatched = [ x for x in ner if x['value'] not in [ y['value'] for y in old_ner ]] 
        ner_matched = [ x for x in old_ner if x['value'] in [ y['value'] for y in ner ]]
        ner_leftout = [ x for x in old_ner if x['value'] not in [y['value'] for y in ner ]]   
        
        for i in ner_matched:
            i['count'] += 1
            i['new'] = 1

        for i in ner_leftout:
            i['new'] = 0

        #eprint("ner_unmatched"); pp(ner_unmatched,stream=sys.stderr)
        #eprint("ner_matched"); pp(ner_matched,stream=sys.stderr)
        #eprint("ner_leftout"); pp(ner_leftout,stream=sys.stderr)

        result = ner_unmatched + ner_matched + ner_leftout

    #NER[sessionkey]=result[0:MAX_KEEP]
    #return NER[sessionkey]
    return store_NER(sessionkey,result[0:MAX_KEEP])




def is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False

    return True

def is_veto(word):
    return VETO.get(word,0) == 1 

def get_word_freq(word):
    return FREQ.get(word,0)

def add_score_OTHER(current_OTHER):
    collection=[]
    for i in current_OTHER:
        #eprint("..... other: {}".format(i['value']))
        if is_number(i['value']):
            pass
        elif len(i['value']) > 1 and not is_veto(i['value']):      
            # score <= inverse frequency 
            freq=get_word_freq(i['value'])
            i['score'] = 1/freq if freq > 0 else 0.9
            i['score'] = float("%0.2f" % i['score'])
            collection.append(i)
    return collection
    


def recalc_score(x):
    word_list = x['value'].split(' ')
    r = len(re.sub(' ','',x['value'])) 

    if r == 0: return x['score']

    count = 0
    for i in word_list:
        if is_veto(i):
            count += len(i)

    assert count <= r

    # prefered length for keyword is |5|
    favor_len = 1/math.sqrt(math.fabs(5.0*5.0-r*r)) if r != 5 else 1

    x['score'] = (x['score'] * (1.0 - count / r)) * favor_len
    return x['score']
    
    
def adjust_veto_score(new_ner):
    for i in new_ner:
        i['score'] = recalc_score(i)
    return new_ner

def collect_NER(sentence_lists):
    result=[]
    for sentence in sentence_lists:
        score=0
        cur=""
        collection=[]

        for tagged_word in sentence:
            taglist=tagged_word.split('\t') 
            tag=taglist[2].split('-')
            if is_flush_tag(tag[0]):
                # do flush collectiona -> result
                if len(collection) > 0: 
                    result.append({ 'tag' : cur, 'value' : ' '.join(collection).decode('utf-8') , 'score': float("%0.2f" % score), 'count': 1 })
                    collection=[]
                    score=0
                # now start new tag
                if tag[0] == 'B' or tag[0] == 'S':
                    cur=tag[1]
                    collection=[taglist[0]]
                    score=float(taglist[3])
            elif is_continue_tag(tag[0]):
                # no flush tag.  keep adding
                collection.append(taglist[0])
                score = (score+float(taglist[3]))*0.5
            else:
                # invalid tag
                pass
        
        # flush if collection exist
        if len(collection) > 0: result.append({ 'tag' : cur, 'value' : ' '.join(collection).decode('utf-8'), 'score': float("%0.2f" % score), 'count': 1 })
    result=adjust_veto_score(result)
    return result

def collect_OTHER(sentence_lists):
    result=[]
    for sentence in sentence_lists:
        collection=[]

        for tagged_word in sentence:
            taglist=tagged_word.split('\t') 
            tag=taglist[2].split('-')
            if is_other_flush_tag(tag[0]) or not taglist[1].startswith('NN'):
                if len(collection) > 0: 
                    result.append({ 'value': ' '.join(collection).decode('utf-8'), 'count' : 1 })
                    collection=[]
            else:   
                # no flush tag.  keep adding
                collection.append(taglist[0])
        
        # flush if collection exist
        if len(collection) > 0: result.append({ 'value': ' '.join(collection).decode('utf-8'), 'count': 1})
    result=add_score_OTHER(result)
    return result


def min(a,b): return a if a < b else b

def dedup(ner,cutoff=True):
    unique_list=[]
    [unique_list.append(x) for x in ner if "{}/{}".format(x['value'],x.get('tag','')) not in [ "{}/{}".format(y['value'],y.get('tag',''))  for y in unique_list]]
    unique_list=[ x for x in unique_list if x.get("score",0.0) >= 0.001 and x.get("tag","") not in [ "TI", "DT"] and len(x.get("value","")) > 1 ]
    for i in unique_list:
        i['count']=0
        for j in ner:
            if j['value'] == i['value']:
                i['score'] = min(i['score'],j['score']) 
                i['count'] += 1
    if cutoff:
        unique_list=[ x for x in unique_list if x.get("score",0.0) >= 2.0 ]
    return unique_list

def choose_best(sessionkey,new_ner,old_ner):
    # dedup new_ner  i.e. (one, one, four) => (one,four)
    cutoff=len(old_ner) > 0 
    new_ner=dedup(new_ner,cutoff)
    
    ts=time.time()
    sorted_NER=sorted(new_ner,key=lambda x: x.get('score',0),reverse=True)[0:MAX_RETURN]
    for i in sorted_NER:
        i['timestamp']=ts
        i['new']=1

    # oldify NER
    if len(old_ner) > 0:
        for i in old_ner:
            i['new']=0
    
    

    if len(sorted_NER) == 0:
        # Fallback to stored values if sessionkey is given
        if len(sessionkey) > 0:
            sorted_NER=sorted(old_ner,key=lambda x: (x.get('new',0),x.get('score',0),x.get('timestamp',0)),reverse=True)[0:MAX_RETURN]
        return sorted_NER
    else:
        # We found answer.  save them for later if sessionkey is given
        if len(sessionkey) > 0:
            #eprint("DEBUG:saving sessionkey ... " + sessionkey)
            return add_NER(sessionkey,sorted_NER,old_ner)
    return sorted_NER

##################################################################################################################
##################################################################################################################

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"


@app.route('/koner/api/v1.0/ner', methods=['POST'])
@as_json(200)
def get_sentence():
    if not request.json or not 'sessionkey' in request.json:
        abort(400)

    sessionkey=request.json.get('sessionkey',"") 
    return { 'sessionkey': sessionkey, 'result' : NER.get(sessionkey,[]) }
    #return { 'sessionkey': sessionkey, 'result' : NER }
    

@app.route('/koner/api/v1.0/tag', methods=['POST'])
@as_json(202)
def evaluate_sentence():
    start = time.time()
    if not request.json or not 'text' in request.json:
        abort(400)
    
    sentences = split_sentence_from_json(request.json['text'],zeros=0)
    #end=time.time() ; eprint("... split time: {:.3f}s".format(end-start)) ; start=end
    test_sentences = tag_pos(sentences, tagger='mecab')
    #end=time.time() ; eprint("... tag time: {:.3f}s".format(end-start)) ; start=end
    test_data = prepare_sentence(test_sentences, word_to_id, slb_to_id, char_to_id, pos_to_id)
    sentence_lists = evaluate_lexicon_tagger(parameters, f_eval, test_sentences, test_data,
                                            id_to_tag, gazette_dict, max_label_len=parameters['lexicon_dim'])
    
    end=time.time() ; eval_time="{:.3f}s".format(end-start); eprint("... eval time: "+eval_time) ; start=end
    sessionkey=request.json.get('sessionkey',"")[0:40] 
    note=request.json.get('note',"")[0:40] 
    old_ner=load_NER(sessionkey)
    new_ner=collect_NER(sentence_lists)
    result_NER=choose_best(sessionkey,new_ner,old_ner)[0:MAX_RETURN]
    #end=time.time() ; eprint("... choosebest time: {:.3f}s".format(end-start)) ; start=end

    result_OTHER=[][0:MAX_RETURN]
    if 'other' in request.json:
        current_OTHER=collect_OTHER(sentence_lists)
        result_OTHER=choose_best("",current_OTHER,[])[0:MAX_RETURN]
    if request.json.get('type',"") == "debug":
        return { 'NER': result_NER, 'OTHER': result_OTHER, 
                'debug': sentence_lists, 'note': note, 'eval-time': eval_time}

    #end=time.time() ; eprint("... final time: {:.3f}s".format(end-start)) ; start=end
    return { 'NER': result_NER, 'OTHER': result_OTHER, 'note': note, 'eval-time': eval_time}



start = time.time()
VETO=read_VETO('./testdata/fdist.csv.veto')
FREQ=read_FREQ('./testdata/fdist.csv')

if __name__=="__main__":

    eprint("number of FREQ: {}".format(len(FREQ.keys())))
    eprint("number of VETO: {}".format(len(VETO.keys()))) #pp(VETO,stream=sys.stderr)
    
    
    eprint("Loading...NER model")

    model = Model(model_path=model_path)
    parameters = model.parameters
    #parameters['crf'] = False
    if not server_mode: pp(parameters,stream=sys.stderr)

    # Load the mappings
    word_to_id, slb_to_id, char_to_id, tag_to_id, pos_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_slb, model.id_to_char, model.id_to_tag, model.id_to_pos]
    ]
    id_to_tag = model.id_to_tag
    pp(id_to_tag)
    end=time.time() ; eprint("... Loading Model: {:.3f}s".format(end-start)) ; start=end

    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()
    end=time.time() ; eprint("... Build Model: {:.3f}s".format(end-start)) ; start=end

    ############################
    # Load Gazette 
    ############################
    gazette_dict = make_gazette_to_dic(dict_path)

    gazette_dict_for, gazette_dict_len = dict(), dict()
    with open(dict_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            words, tag = line[0], line[1]

            if len(words) > 3:
                gazette_dict_len[words] = len(words)
                gazette_dict_for[words] = tag

    gazette_dict_len = sorted(gazette_dict_len.iteritems(), key=itemgetter(1), reverse=True)
    end=time.time() ; eprint("... Building Gazette Dictionary: {:.3f}s".format(end-start)) ; start=end

    ############################
    # Load input text
    ############################
    if server_mode:
        init_NER()
        app.run(debug=False, host='0.0.0.0')
    else:
        if is_preprocess == 1:
            test_sentences = load_sentences(input_filename, zeros=1)
        else:
            sentences = split_sentence(input_filename,zeros=0)
            test_sentences = tag_pos(sentences, tagger='mecab')
            #pp(test_sentences,stream=sys.stderr)

            end=time.time() ; eprint("... Morphological Analysis: {:.3f}s".format(end-start)) ; start=end


            eprint('Running...NER')
            test_data = prepare_sentence(test_sentences, word_to_id, slb_to_id, char_to_id, pos_to_id)
            sentence_lists = evaluate_lexicon_tagger(parameters, f_eval, test_sentences, test_data,
                                            id_to_tag, gazette_dict, max_label_len=parameters['lexicon_dim'],
                                            output_path=output_filename)

            end=time.time() ; eprint("... Evaluate Sentence: {:.3f}s".format(end-start)) ; start=end
