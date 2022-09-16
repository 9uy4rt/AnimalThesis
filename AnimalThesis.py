import pandas as pd
import glob
import re
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

all_files = glob.glob('D:\Code\Python\AnimalThesis\AnimalThesis.csv')
all_files_data = []

for file in all_files: 
 data_frame = pd.read_csv(file) 
 all_files_data.append(data_frame)

all_files_data_concat = pd.concat(all_files_data, axis = 0, ignore_index = True) 
all_files_data_concat.to_csv('D:\Code\Python\AnimalThesis\AnimalThesis.csv', encoding = 'utf-8', index = False)

all_title = all_files_data_concat['제목']

stopWords = set(stopwords.words("english"))

lemma = WordNetLemmatizer()
words = []

for title in all_title: 
	EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title)) 
	EnWordsToken = word_tokenize(EnWords.lower()) 
	EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords] 
	EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop] 
	words.append(EnWordsTokenStopLemma)

words2 = list(reduce(lambda x, y: x+y, words))
while 'animal' in words2:
	words2.remove('animal')

count = Counter(words2)
word_count = dict()

for tag, counts in count.most_common(50):
	if(len(str(tag))>1):
		word_count[tag] = counts
		print("%s : %d" % (tag, counts))

sorted_Keys = sorted(word_count, key = word_count.get, reverse = True)
sorted_Values = sorted(word_count.values(), reverse = True)
plt.bar(range(len(word_count)), sorted_Values, align = 'center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation = '85')
plt.show()

all_files_data_concat['doc_count'] = 0 
summary_year = all_files_data_concat.groupby('출판일', as_index = False)['doc_count'].count() 
summary_year

plt.figure(figsize = (12, 5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)
plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])
plt.show()

stopwords = set(STOPWORDS)
wc = WordCloud(background_color = 'ivory', stopwords = stopwords, width = 800, height = 600)
cloud = wc.generate_from_frequencies(word_count)
plt.figure(figsize = (8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
ratio = [12.7, 10.5, 10.1, 9.8, 6.3, 3.9, 2.8, 1.8, 0.8, 45.5]

labels = ['PJB PUBLICATIONS LTD', 'Elsevier Science B.V., Amsterdam', 'Elsevier', 'UNIVERSITIES FEDERATION FOR ANIMAL WELFARE', 'Cambridge University Press', 'American Society of Animal Science [etc.]', 'American Society of Animal Science', '[University of Illinois Press, Ferrater Mora Oxford Centre for Animal Ethics]', 'SCAS', 'Else']

plt.pie(ratio, labels=labels, autopct='%.1f%%')
plt.show()