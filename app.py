import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import load

from PIL import Image

from eda import *



def main():
	st.title('Trying out Sentiment Analysis with Streamlit!')

	st.subheader("EDA, Data Cleaning, & Modeling with Kaggle's \
		Twitter US Ariline Sentiment Dataset.")

	main_image = Image.open('./Images/nlp-pipe.jpg')
	st.image(main_image, use_column_width=True)

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white; font-size:18px; text-align:center">Choose what to do:</p></div>
	"""
	st.markdown(html_temp, unsafe_allow_html=True)
	


	if st.checkbox('Exploratory Data Analysis'):
		explorer = EDA()
		n_rows = st.sidebar.slider('Displaying dataset, select number of rows', 10, 20)
		
		all_cols = explorer.df.columns.tolist()
		select_cols = st.sidebar.multiselect('Select column(s) to display:', all_cols, ['airline_sentiment', 'text'])

		'Number of rows:', n_rows, #
		explorer.df[select_cols].head(n_rows), #

		
		if st.sidebar.checkbox('Most Frequent Words Per Category'):
			'---------------------------------------------', #
			st.info("Try with removing stopwords and/or tags('@'/'#')")
			st.write('Most Frequent Words for Positive(Blue), Negative(Red), and Neutral(Green) Tweets:')
			c = st.sidebar.slider(
				'Select a number for the top frequent words to display',
				10, 15, 10)
			c = int(c)

			remove_stop = False
			if st.sidebar.checkbox('Remove stop words'):
				remove_stop = True
			
			remove_at = False
			if st.sidebar.checkbox('Remove @ and #'):
				remove_at = True
			
			
			freqs = explorer.most_freq_words(c, remove_at, remove_stop)
			plt.show()
			st.pyplot()
			

			cat = st.sidebar.selectbox(
				"To view word counts, select a sentiment category",
				('Positive', 'Negative', 'Neutral'))
			
			if cat == 'Positive':
				'Top words in ', freqs[0][0], ' tweets', #
				freqs[0][1].head(c), #
			elif cat == 'Negative':
				'Top words in ', freqs[1][0], ' tweets', #
				freqs[1][1].head(c), #
			else:
				'Top words in ', freqs[2][0], ' tweets', #
				freqs[2][1].head(c), #

		
		if st.sidebar.checkbox('Word Counts'):
			'---------------------------------------------', #
			explorer.word_counts()
			st.pyplot()
		

		if st.sidebar.checkbox("View most frequent @'s and #'s"):
			'---------------------------------------------', #
			char = st.sidebar.radio('', ('@', '#'))
			if char == '@':
				explorer.find_at_hash()
			else:
				explorer.find_at_hash(at=False)
			st.pyplot()

		
		if st.sidebar.checkbox("View most frequent emojis and emoticons"):
			'---------------------------------------------', #
			c = st.sidebar.slider('Choose the number of top emojis to view',
				10, 20)
			emojis = explorer.find_emojis()
			emojis.head(c), #
			st.balloons()

		
		if st.sidebar.checkbox('Target Field'):
			'---------------------------------------------', #
			explorer.show_target_field()
			st.pyplot()



	if st.checkbox("Text Preprocessing And Sentiment Analysis"):
		text = st.text_area("Enter your text to analize:",
				"@americanairline Thanks for the #amazing flying experience!")
		cleaner = Cleaner(text)
		operations = st.sidebar.multiselect("Choose the preprocessing steps to perform",
			['Lowercasing', 'Remove html tags', 'Remove punctuations', 'Replace links',
			'Replace emojis', 'Replace Mentions(@)', 'Replace Hashtags(#)', 'Remove stop words',
			'Lemmatization', 'Spell correction'], ['Remove stop words'])

		str_to_func = {'Lowercasing': cleaner.lowercasing, 'Remove html tags': cleaner.remove_html,
				'Remove punctuations': cleaner.remove_punc, 'Replace links': cleaner.replace_links,
				'Replace Mentions(@)': cleaner.replace_mentions, 'Replace Hashtags(#)': cleaner.replace_hashtags,
				'Replace emojis': cleaner.replace_emojis,'Remove stop words': cleaner.remove_stop,
				'Lemmatization': cleaner.lemmatize, 'Spell correction': cleaner.sepll_correct}
		
		if not operations:
			st.info('### No preprocessing steps selected')
		else:
			for op in operations:
				op = str_to_func[op]
				sample_text, findings = op()
				
				if findings:
					st.info(op.__doc__ + ', '.join(findings).strip())

			st.write('#### Preprocessed text: ', sample_text)
			
	

		if st.button("Analyze Text Sentiment"):
			model = load('./Model/lr_clf.joblib')
			# confusion_matrix = Image.open('./Images/confusion_matrix.jpg')
			# 'Model Performance on the Test set:', #
			# st.image(confusion_matrix)

			class_names = ['negative', 'neutral', 'positive']
			explainer = LimeTextExplainer(class_names=class_names)

			if text:
				model = load('./lr_clf.joblib')
				processed_text, sentiment = get_sentiment(text, model)
				'Original text ---> ', text, #
				'Processed text --> ', processed_text, #
				'Text Sentiment --> {}'.format(sent_dict[sentiment]), #

				exp = explainer.explain_instance(processed_text, model.predict_proba)
				# exp.show_in_notebook()
				exp.as_pyplot_figure()
				st.pyplot()




if __name__ == '__main__':
	main()


