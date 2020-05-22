import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import pickle
# Import Heapq 
from heapq import nlargest
class Summary:
    # Place All As A Function For Reuseability
    def text_summarizer(self,raw_docx):
        raw_text = raw_docx
        nlp = spacy.load('en_core_web_sm')
        docx = nlp(raw_text)
        stopwords = list(STOP_WORDS)
        # Build Word Frequency
    # word.text is tokenization in spacy
        word_frequencies = {}  
        for word in docx:  
            if word.text not in stopwords:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    
    
        maximum_frequncy = max(word_frequencies.values())
    
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        # Sentence Tokens
        sentence_list = [ sentence for sentence in docx.sents ]
    
        # Calculate Sentence Score and Ranking
        sentence_scores = {}  
        for sent in sentence_list:  
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if len(sent.text.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]
    
        # Find N Largest
        summary_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
        final_sentences = [ w.text for w in summary_sentences ]
        summary = ' '.join(final_sentences)
        #print("Original Document\n")
        #print(raw_docx)
        #print("Total Length:",len(raw_docx))
        #print('\n\nSummarized Document\n')
        #print(summary)
        #print("Total Length:",len(summary))
        return summary
o=Summary()
with open('biomedical.txt', 'r') as file:
    document1 = file.read().replace('\n', '')
pickle.dump(o, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
#print(model.text_summarizer(document1))