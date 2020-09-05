import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import demoji
from textblob import TextBlob
from lime.lime_text import LimeTextExplainer

import spacy
nlp = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
stop = set(stopwords.words('english'))

import re
import string
from collections import Counter

class EDA:
	def __init__(self):
		path = './Tweets.csv'
		self.df = pd.read_csv(path)
		self.colors=['blue','red','green']
		self.categories = ['positive', 'negative', 'neutral']
		self.all_words = self.df['text'].str.lower().str.cat(sep=' ')
		self.tweets = [self.df[self.df['airline_sentiment'] == cat]['text'] for cat in self.categories]

	def show_target_field(self):
		counts = self.df.airline_sentiment.value_counts()
		# print(counts, '\n')
		plt.figure(figsize=(8,5))
		sns.barplot(counts.index, counts)
		plt.title('Number of Instances for Different Tweet Sentiment Categories')
		plt.gca().set_ylabel('Counts')

	def word_counts(self):
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))

		content = [tweet.map(lambda t: len(t.split())) for tweet in self.tweets]

		for text, ax, c, cat in zip(content, (ax1, ax2, ax3), self.colors, self.categories):
			sns.distplot(text, ax=ax, color=c)
			ax.set_title("'{} Tweets'".format(cat))

		fig.suptitle('Word Count Distribution Per Sentiment Category')
		# plt.show()

	def most_freq_words(self, c=10, remove_at=False, remove_stop=False):
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
		axis = [ax1, ax2, ax3]
		results = []
		stop_ = stop if remove_stop else ()
		
		for i in range(3):
			if remove_at:
				all_words = self.tweets[i].str.replace(r'@\w+|#\w*', '').str.lower().str.cat(sep=' ').split()
			else:
				all_words = self.tweets[i].str.lower().str.cat(sep=' ').split()
			
			ax, cat, color = axis[i], self.categories[i], self.colors[i]
			
			freq = Counter([w for w in all_words if w not in stop_]).most_common(c)
			freq_df = pd.DataFrame(freq, columns=['top_words', 'counts']).set_index('top_words')
			results.append((cat, freq_df))
			freq_df.plot.bar(rot=75, ax=ax, color=color)

		return results

	def find_at_hash(self, at=True):
		pattern = r'@\w+' if at else r'#\w+'
		words = re.findall(pattern, self.all_words)
		counts = Counter(words)	
		
		top_tags = pd.DataFrame(counts.most_common(10), columns=['tags', 'counts']).set_index('tags')
		top_tags.plot.bar(rot=55, color='tomato')

	def find_emojis(self):
		pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)

		emos = re.findall(pattern, self.all_words)
		emos = Counter(''.join(emos)).most_common()[1:]
		emos = pd.DataFrame(emos, columns=['emo', 'counts'])
		return emos


class Cleaner:
	def __init__(self, text):
		self.text = text
		self.tokenizer = TweetTokenizer()

	def lowercasing(self):
		self.text = self.text.lower()
		return self.text, False

	def remove_html(self):
		"""html tags in the text: """
		pattern = r'<.*?>'
		tags = re.findall(pattern, self.text)
		if tags == []:
			return self.text, False
		self.text = re.sub(pattern, '', self.text)
		return self.text, tags

	def remove_punc(self):
		"""punctuations found: """
		pattern = r'[^\w ]'
		all_symbols = re.findall(pattern, self.text)
		if all_symbols == []:
			return self.text, False

		punc = string.punctuation
		table = str.maketrans(' ', ' ', punc)
		self.text = self.text.translate(table)
		return self.text, all_symbols

	def replace_links(self):
		"""links found: """
		pattern = r'https?://\S+|www\.\S+'
		all_links = re.findall(pattern, self.text)
		if all_links == []:
			return self.text, False
		self.text = re.sub(pattern, ' constanturl', self.text)
		return self.text, all_links

	def replace_emojis(self):
		"""emojis/emoticons found:  """
		emojis = demoji.findall(self.text)
		if not emojis:
			return self.text, False

		tokenized_text = tokenizer.tokenize(text)
		for i, s in enumerate(tokenized_text):
			if s in emojis.keys():
				tokenized_text[i] = emojis[s]
		self.text = ' '.join(tokenized_text)
		self.text = self.erase_emojis()
		return self.text, list(emojis.keys())

	def erase_emojis(self):
		return self.text.encode('ascii', 'ignore').decode('utf-8')

	def replace_mentions(self):
		"""mentions found in text: """
		found = re.findall(r'@\w+', self.text)
		if not found:
			return self.text, False
		self.text = re.sub(r'@\w+', ' constantmention ', self.text)
		return self.text, found

	def replace_hashtags(self):
		"""hashtags found in text: """
		found = re.findall(r'#\w+', self.text)
		if not found:
			return self.text, False
		self.text = re.sub(r'#\w+', ' constantmention ', self.text)
		return self.text, found

	def remove_stop(self):
		"""stopwords found: """
		stops = []
		good_words = []
		for w in self.tokenizer.tokenize(self.text):
			if w in stop:
				stops.append(w)
			else:
				good_words.append(w)

		if stops == []:
			return self.text, False
		self.text = ' '.join(good_words)
		return self.text, stops

	def lemmatize(self):
		doc = nlp(self.text)
		lemmas = []
		for tok in doc:
			lem = tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text
			lemmas.append(lem)
		self.text = ' '.join(lemmas)
		return self.text, False

	def sepll_correct(self):
		self.text = str(TextBlob(self.text).correct())
		return self.text, False


def get_sentiment(text, model):
	cleaner = Cleaner(text)

	cleaner.remove_stop()
	cleaner.lowercasing()
	cleaner.remove_html()
	cleaner.replace_links()
	cleaner.replace_hashtags()
	cleaner.replace_mentions()
	cleaner.replace_emojis()
	cleaner.lemmatize()
	cleaner.remove_punc()
	# cleaner.sepll_correct()
	
	text = cleaner.text
	sentiment = model.predict([text])[0]
	return text, sentiment





sent_dict = {0:'Negative T_T', 1:'Neutral -.-', 2:'Positive :)'}
# responses = {
# 			0:["We are sorry to hear that. In xxx airline we are doing everything we can to improve the \
# 				quality of our service. If you have any complains to make, please contact..."],
# 			1: ["We wish you a good stay in Barcelona, and we hope to see you soon."], 
# 			2: ["Thank you for the kind word, our mission is to provide you with our best services.",
# 					"Thank you for your support and we hope to see you soon!"]
# 			}