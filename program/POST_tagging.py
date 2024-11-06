import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sentence = "Part-of-speech tagging is important for natural language processing."
words = word_tokenize(sentence)
pos_tags = pos_tag(words)
print(pos_tags)