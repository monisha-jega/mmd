# Simple usage
from stanfordcorenlp import StanfordCoreNLP
import sys, os, json
import cPickle as pl

#nlp = StanfordCoreNLP("../../corenlp_new")
#java -mx4g -cp "../../corenlp_new/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8000 -timeout 15000 -annotators "tokenize,ssplit,dep"
nlp = StanfordCoreNLP('http://localhost', port=8000)

sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
print(nlp.pos_tag(sentence))
print 'Tokenize:', nlp.word_tokenize(sentence)
print 'Part of Speech:', nlp.pos_tag(sentence)
# print 'Named Entities:', nlp.ner(sentence)
# print 'Constituency Parsing:', nlp.parse(sentence)
print 'Dependency Parsing:', nlp.dependency_parse(sentence)





from parameters import *

data_type = sys.argv[1]
if data_type == "train":
	dir = train_dir
elif data_type == "val":
	dir = val_dir
else:
	dir = test_dir

dep_parses = {}

dirfiles =  os.listdir(dir)
print("num files ", len(dirfiles))
count = 0

#For each dialogue 
for json_file in dirfiles[:]:
	#Load dialogue
	dialogue_json = json.load(open(dir+json_file))
	# print(len(dialogue_json))
	
	#For each utterance in dialogue
	for i, utterance_ in enumerate(dialogue_json[:]):
		
		utterance = utterance_['utterance'] #The textual part
		if utterance is None:
			continue	

		#Tokenize text and update vocabulary
		if utterance['nlg'] in ["", None]:
			refined_parse =  []
		else:
			parse = nlp.dependency_parse(utterance['nlg'].encode('utf-8'))
			refined_parse = [parse[0][2]]
			for triplet in parse[:]:
				refined_parse.append((triplet[1], triplet[2]))

		dep_parses[json_file + "_" + str(i)] = refined_parse

	count += 1
	if count % 1000 == 0:
		print(count)

pl.dump(dep_parses, open(data_dump_dir + data_type +"_depparses.pkl", 'wb'))
#print(dep_parses)
		
nlp.close() # Do not forget to close! The backend server will consume a lot memery.