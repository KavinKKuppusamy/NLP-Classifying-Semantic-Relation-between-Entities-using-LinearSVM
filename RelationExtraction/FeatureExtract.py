import os
import re
from collections import Iterable

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
from nltk.tag.stanford import StanfordNERTagger
import spacy
import networkx as nx

nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Sanford Info
# os.environ["JAVAHOME"] = 'C:/Program Files/Java/jdk-15.0.1/bin/java.exe'
# jar = './stanford-ner-tagger/stanford-ner.jar'
# model = './stanford-ner-tagger/english.all.3class.distsim.crf.ser.gz'

# Prepare NER tagger with english model
# ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
all_relations = []


def flatten(in_list):
    for item in in_list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class FeatureExtract:

    def __init__(self, input_sentence, input_relation):
        # Read line and store as string
        self.input_sentence = input_sentence
        self.input_relation = input_relation
        self.e1_or_e2_hyper = None
        self.e1, self.e2, self.words, self.before_e1, self.between_e1_e2, self.after_e2, self.e1_or_e2_words = None, None, None, None, None, None, None
        self.sentence_string, self.lemma, self.lemma_pos, self.e1_ner, self.e2_ner = None, None, None, None, None
        self.word_outside, self.prefix5between, self.distance_between, self.e1_index, self.e2_index, self.pos_tags, self.e1_tag, self.e2_tag, self.e1_or_e2_tag, self.pos_tags_between = None, None, None, None, None, None, None, None, None, None
        self.SynsetE1, self.SynsetE2, self.HyperE1, self.HyperE2, self.HypoE1, self.HypoE2, self.HoloE1, self.HoloE2, self.MeroE1, self.MeroE2 = None, None, None, None, None, None, None, None, None, None
        self.relation, self.direction = None, None
        self.lc_hyper = None
        self.parse_list = None
        self.root_word_lemma, self.verb_class, self.only_classids = None, None, None
        self.dep_path_len2_location, self.dep_path_len1, self.sdp_root_lemma, self.sdp_root, self.sdp_root_tag, self.connecting_path = None, None, None, None, None, None
        self.e1_dep, self.e1_postag, self.e2_dep, self.e2_postag = None, None, None, None
        self.root_word = None
        self.shortest_path_len = None
        self.shortest_path = None
        self.root_word_location = None
        self.max_entity_sim = None
        self.pre_process_and_tokenize()
        self.wordnet_features()
        self.reveal_relation_direction(self.input_relation)
        self.lowest_common_hypyernyms()
        self.perform_NER()
        self.find_root_word_location()
        self.find_shortest_path_dependency()
        self.max_entity_similarity()
        self.root_word_verbnet_features()
        self.dependency_parsing()

    def pre_process_and_tokenize(self):
        self.input_sentence = re.findall('".*"', self.input_sentence)[0].strip("\"")
        self.e1 = re.findall('(?<=<e1>).*(?=</e1>)', self.input_sentence)[0].strip()
        if len(self.e1.strip().split()) > 1:
            old_e1 = self.e1
            self.e1, temp_tag = self.find_head_word(self.e1)
            self.input_sentence = self.input_sentence.replace(old_e1, self.e1)
        self.e2 = re.findall('(?<=<e2>).*(?=</e2>)', self.input_sentence)[0].strip()
        if len(self.e2.strip().split()) > 1:
            old_e2 = self.e2
            self.e2, temp_tag = self.find_head_word(self.e2)
            self.input_sentence = self.input_sentence.replace(old_e2, self.e2)
        self.before_e1 = re.findall('.*(?=<e1>)', self.input_sentence)[0].strip()
        self.between_e1_e2 = re.findall('(?<=</e1>).*(?=<e2>)', self.input_sentence)[0].strip()
        self.after_e2 = re.findall('(?<=</e2>).*', self.input_sentence)[0].strip()
        self.sentence_string = self.input_sentence.replace('<e1>', ' ').replace('</e1>', ' ').replace('<e2>',
                                                                                                      ' ').replace(
            '</e2>', ' ')
        self.sentence_string = re.sub('\s+', ' ', self.sentence_string)
        # convert strings to token list

        self.words = word_tokenize(self.sentence_string)
        self.before_e1 = word_tokenize(self.before_e1)
        self.between_e1_e2 = word_tokenize(self.between_e1_e2)
        self.after_e2 = word_tokenize(self.after_e2)
        self.e1_or_e2_words = list({self.e1, self.e2})
        before_e1 = [self.before_e1[-1]] if self.before_e1 else []
        self.word_outside = before_e1 + self.after_e2

        # Stemming with Prefix value 5
        self.prefix5between = [word[:5] for word in self.between_e1_e2]
        self.prefix5between = ' '.join(self.prefix5between) if self.prefix5between is not None else None
        self.distance_between = len(self.between_e1_e2)
        self.e1_index = len(self.before_e1)
        self.e2_index = len(self.before_e1) + 1 + len(self.between_e1_e2)
        self.pos_tags = nltk.pos_tag(self.words)
        self.e1_tag = self.pos_tags[self.e1_index][1]
        self.e2_tag = self.pos_tags[self.e2_index][1]
        self.e1_or_e2_tag = list({self.e1_tag, self.e2_tag})
        self.pos_tags_between = '_'.join([self.pos_tags[index][1][0] for index in
                                          range(self.e1_index + 1, self.e2_index)])
        self.lemma = [lemmatizer.lemmatize(w) for w in self.words]
        self.lemma_pos = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in self.words]
        self.root_word, temp_tag = self.find_head_word(self.sentence_string)
        self.before_e1 = ' '.join(self.before_e1)
        self.after_e2 = ' '.join(self.after_e2)
        self.between_e1_e2 = ' '.join(self.between_e1_e2)
        self.e1_or_e2_words = ' '.join(self.e1_or_e2_words)
        self.lemma_pos = ' '.join(self.lemma_pos)
        self.word_outside = ' '.join(self.word_outside)
        self.e1_or_e2_tag = ' '.join(self.e1_or_e2_tag)

    def find_head_word(self, noun_phrase):
        head_word, head_tag = [[token.head.text, token.tag_] for token in nlp(noun_phrase) if token.dep_ == 'ROOT'][0]
        return [head_word, head_tag]

    def preprocess_wn_list(self, wn_list, single_list=False, nested_list=False):
        ret_list = []
        if single_list:
            ret_list = list(set([str(entry)[8:-2] for entry in wn_list]))
        if nested_list:
            temp_list = [entry[0] for entry in wn_list if len(entry) > 0]
            ret_list = list(set([str(entry)[8:-2] for entry in temp_list]))
        return ret_list if ret_list else [None]

    def wordnet_features(self):
        self.SynsetE1 = wn.synsets(self.e1)
        self.SynsetE2 = wn.synsets(self.e2)
        self.HyperE1 = self.preprocess_wn_list([synset.hypernyms() for synset in self.SynsetE1], nested_list=True)
        self.HyperE2 = self.preprocess_wn_list([synset.hypernyms() for synset in self.SynsetE2], nested_list=True)
        self.HypoE1 = self.preprocess_wn_list([synset.hyponyms() for synset in self.SynsetE1], nested_list=True)
        self.HypoE2 = self.preprocess_wn_list([synset.hyponyms() for synset in self.SynsetE2], nested_list=True)
        self.HoloE1 = self.preprocess_wn_list([synset.part_holonyms() for synset in self.SynsetE1], nested_list=True)
        self.HoloE2 = self.preprocess_wn_list([synset.part_holonyms() for synset in self.SynsetE2], nested_list=True)
        self.MeroE1 = self.preprocess_wn_list([synset.part_meronyms() for synset in self.SynsetE1], nested_list=True)
        self.MeroE2 = self.preprocess_wn_list([synset.part_meronyms() for synset in self.SynsetE2], nested_list=True)

        self.e1_or_e2_hyper = list(set(self.HyperE1 + self.HyperE1))
        self.e1_or_e2_hyper = list(filter(None, self.e1_or_e2_hyper))
        self.e1_or_e2_hyper = ' '.join(self.e1_or_e2_hyper) if self.e1_or_e2_hyper is not None else None

        self.HyperE1 = list(filter(None, self.HyperE1))
        self.HyperE1 = ' '.join(self.HyperE1)
        self.HyperE2 = list(filter(None, self.HyperE2))
        self.HyperE2 = ' '.join(self.HyperE2)

        self.HypoE1 = list(filter(None, self.HypoE1))
        self.HypoE1 = ' '.join(self.HypoE1)
        self.HypoE2 = list(filter(None, self.HypoE2))
        self.HypoE2 = ' '.join(self.HypoE2)

        self.HoloE1 = list(filter(None, self.HoloE1))
        self.HoloE1 = ' '.join(self.HoloE1)
        self.HoloE2 = list(filter(None, self.HoloE2))
        self.HoloE2 = ' '.join(self.HoloE2)

        self.MeroE1 = list(filter(None, self.MeroE1))
        self.MeroE1 = ' '.join(self.MeroE1)
        self.MeroE2 = list(filter(None, self.MeroE2))
        self.MeroE2 = ' '.join(self.MeroE2)

        self.SynsetE1 = self.preprocess_wn_list(self.SynsetE1, single_list=True)
        self.SynsetE2 = self.preprocess_wn_list(self.SynsetE2, single_list=True)

        self.SynsetE1 = list(filter(None, self.SynsetE1))
        self.SynsetE1 = ' '.join(self.SynsetE1)
        self.SynsetE2 = list(filter(None, self.SynsetE2))
        self.SynsetE2 = ' '.join(self.SynsetE2)

    def lowest_common_hypyernyms(self):
        try:
            syn_e1 = wn.synsets(self.e1)[0]
            syn_e2 = wn.synsets(self.e2)[0]
            self.lc_hyper = str(syn_e1.lowest_common_hypernyms(syn_e2)[0])[8:-2]
        except IndexError as e:
            self.lc_hyper = None

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wn.ADJ,
                    "N": wn.NOUN,
                    "V": wn.VERB,
                    "R": wn.ADV}
        return tag_dict.get(tag, wn.NOUN)

    # Set relation function, just reading from the text file
    def reveal_relation_direction(self, line):
        line = line.replace('(', '**').replace(')', '')
        line_split = line.split('**')
        self.relation = line_split[0].strip()
        if len(line_split) > 1:
            self.direction = line_split[1].strip()
        all_relations.append(self.relation)

    def perform_NER(self):
        doc = nlp(self.sentence_string)
        spacy_nlp = []
        for ent in doc.ents:
            spacy_nlp.append([ent.text, ent.start_char, ent.end_char, ent.label_])
        for tag in spacy_nlp:
            if tag[0] == self.e1:
                self.e1_ner = tag[3]
            if tag[0] == self.e2:
                self.e2_ner = tag[3]

        # words_token = nltk.word_tokenize(self.sentence_string)
        # # Run NER tagger on words
        # word_tagged = ner_tagger.tag(words_token)
        # for word_tag in word_tagged:
        #     if self.e1_tag in word_tag[0]:
        #         self.e1_ner = word_tag[1]
        #     if self.e2_tag in word_tag[0]:
        #         self.e2_ner = word_tag[1]
        return

    def dependency_parsing(self):
        doc = nlp(self.sentence_string)
        parse_list = []
        for token in doc:
            parse_list.append([token.text, token.tag_, token.head.text, token.dep_])
        self.parse_list = parse_list

    def find_shortest_path_dependency(self):
        e1 = re.sub(r'[^\w\s]', '', self.e1)
        e2 = re.sub(r'[^\w\s]', '', self.e2)
        sent_string = self.sentence_string.replace(self.e1, e1).replace(self.e2, e2)
        doc = nlp(sent_string)

        for tok in doc:
            if e1 in tok.text:
                self.e1_dep, self.e1_postag = tok.dep_, tok.pos_
            elif e2 in tok.text:
                self.e2_dep, self.e2_postag = tok.dep_, tok.pos_

        edges = []
        for token in doc:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        entity1 = e1.lower()
        entity2 = e2.lower()
        self.shortest_path_len = nx.shortest_path_length(graph, source=entity1, target=entity2)
        self.shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
        self.connecting_path = ' '.join(nx.shortest_path(graph, source=entity1, target=entity2)[1:-1])
        self.shortest_path = ' '.join(self.shortest_path)
        self.sdp_root, self.sdp_root_tag = self.find_head_word(
            self.connecting_path) if self.connecting_path else (None, None)
        self.sdp_root_lemma = lemmatizer.lemmatize(self.sdp_root, 'v') if self.sdp_root is not None else None
        self.dep_path_len1 = ' '.join(
            [f'{self.sdp_root_lemma}->{self.e1_dep}->E1', f'{self.sdp_root_lemma}->{self.e2_dep}->E2'])
        self.dep_path_len2_location = f"E1_{self.e1_dep}_{self.root_word_location}_{self.e2_dep}_E2"
        return

    def find_root_word_location(self):
        e1_index = self.e1_index
        e2_index = self.e2_index
        try:
            root_index = self.words.index(self.root_word)
        except Exception as e:
            for i in range(len(self.words)):
                if self.root_word in self.words[i]:
                    root_index = i
        loc = None
        if root_index < e1_index:
            loc = 'BEFORE'
        elif root_index > e2_index:
            loc = 'AFTER'
        else:
            loc = 'BETWEEN'
        self.root_word_location = loc

    def max_entity_similarity(self):
        from itertools import product
        allsyns1 = set(ss for ss in wn.synsets(self.e1))
        allsyns2 = set(ss for ss in wn.synsets(self.e2))
        sim_values = [wn.wup_similarity(s1, s2) for s1, s2 in product(allsyns1, allsyns2)]
        sim_values = list(filter(None, sim_values))
        self.max_entity_sim = max(sim_values) if len(sim_values) > 0 else 0

    def root_word_verbnet_features(self):
        self.root_word_lemma = lemmatizer.lemmatize(self.root_word, 'v')
        all_classids = vn.classids(lemma=self.root_word_lemma)
        self.verb_class = ' '.join([c_id.split('-')[0] for c_id in all_classids])
        self.only_classids = ' '.join([vn.shortid(c_id) for c_id in all_classids])
