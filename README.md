# Multilingual Emoticon Prediction of Tweets about COVID-19😷

Emojis are a widely used tool for encoding emotional content in informal messages such as
tweets, and predicting which emoji corresponds to a piece of text can be used as a proxy for
measuring the emotional content in the text. This paper presents the first model for predicting
emojis in highly multilingual text. Our BERTmoticon model is a fine-tuned version of the
multilingual BERT model (Devlin et al., 2018), and it can predict emojis for text written in 102
different languages. We trained our BERTmoticon model on 54.2 million geolocated tweets
sent in the first 6 months of 2020, and we apply the model to a case study analyzing the emotional
reaction of Twitter users to news about the coronavirus. Example findings include a spike in
sadness when the World Health Organization (WHO) declared that coronavirus was a global
pandemic, and a spike in anger and disgust when the number of COVID-19 related deaths in
the United States surpassed one hundred thousand. We provide an easy-to-use and open source
python library for predicting emojis with BERTmoticon so that the model can easily be applied
to other data mining tasks.

The table below shows what emojis Bertmoticon can predict across 10 different languages. (Translation: Google Translate) 
![GitHub Logo](/paper/png_figures/languages.png)

The F1-Perfomance of the model is shown here:
![GitHub Logo](/paper/png_figures/f1perf.png)

Using the emoji definitions and their use in our model we were able to map them to the plutchik wheel of sentiment. (Mapping can be further improved)
![GitHub Logo](/paper/png_figures/emojimap.png)

We then applied this model and this mapping to 16 million tweets from the first half of 2020. The mask emoji is considered as a seperate category, since we want to observe any earlier occurences of it before the outbreak.

![GitHub Logo](/paper/png_figures/graph8.png)
