# Bertmoticon 

The Bertmoticon package is fine-tuned from the [BERT](https://github.com/google-research/bert) model, to the emoji prediction task. It can predict emojis in 102 languages. In this package we include two functions that enable the use of it: [bertmoticon.infer](#bertmoticon.infer) and [bertmoticon.infer_mappings](#bertmoticon.infer_mappings). The number of emojis available for this model are 80; and are listed in [bertmoticon.emojis](#bertmoticon.emojis). 

## Paper 

[link to paper](https://aclanthology.org/2020.peoples-1.11.pdf)

## TwitterEmoticon Dataset

You can download the TwitterEmoticon Dataset here: [link](https://izbicki.me/public/cs/tweets/). Abiding by the TwitterAPI's guidelines we are only allowed to share the IDs of the tweets. The link contains the test,train and valiadation splits that we used for our paper. The duplicates that exist signal that the tweet had more than one emoji.

## TwitterCovid Dataset

You can download the TwitterCovid dataset that is mentioned in the paper through here: (5 parts) [a](https://github.com/Stefanos-stk/Bertmoticon/blob/master/bertmoticon_covid_dataset_part-aa.gz), [b](https://github.com/Stefanos-stk/Bertmoticon/blob/master/bertmoticon_covid_dataset_part-ab.gz) , [c](https://github.com/Stefanos-stk/Bertmoticon/blob/master/bertmoticon_covid_dataset_part-ac.gz) , [d](https://github.com/Stefanos-stk/Bertmoticon/blob/master/bertmoticon_covid_dataset_part-ad.gz) , [e](https://github.com/Stefanos-stk/Bertmoticon/blob/master/bertmoticon_covid_dataset_part-ae.gz).
Abiding by TwitterAPI's guidelines we are only allowed to provide the IDs of the tweets.

## Installation

Installing the Bertmoticon package from [PyPI](https://pypi.org/) using:

```
pip3 install bertmoticon
```
## Importing in python
Importing the package can be done as:
```
import bertmoticon
```
If the model is not already downloaded; upon first run it will download and extract the model automatically as such:
```
Downloading bermoticon model
[=                                                          ]
...
[==================                                         ]
...
[===========================================================]
Extracting the model
```
The model is not included with the pypi installation. It requires 1.34 GB. Loads it either into CUDA or CPU based on CUDA availability.
## Usage


## bertmoticon.emojis
The model can predict up to 80 emojis. Acceessing the emojis can be done by calling the global variable ```emojis``` called as ```bertmoticon.emojis```. 
```
>>> print(bertmoticon.emojis)
['😂', '😭', '😍', '😊', '🙏', '😅', '😁', '🙄', '😘', '😔', '😩', '😉', '😎', '😢', '😆', '😋', '😌', '😳', '😏', '🙂', '😃', '🙃', '😒', '😜', '😀', '😱', '🙈', '😄', '😡', '😬', '🙌', '😴', '😫', '😪', '😤', '😇', '😈', '😞', '😷', '😣', '😥', '😝', '😑', '😓', '😕', '😹', '😐', '😻', '😖', '😛', '😠', '🙊', '😰', '😚', '😲', '😶', '😮', '🙁', '😵', '😗', '😟', '😨', '🙇', '🙋', '😙', '😯', '🙆', '🙉', '😧', '😿', '😸', '🙀', '😦', '😽', '😺', '😼', '🙅', '😾', '🙍', '🙎']
```

## bertmoticon.infer

Takes in a ```list``` of ```strings``` and an ```int``` number of guesses. It returns a list of dictionaries, where each dictionary contains an emoji and a corresponding percentage.

```
>>> ls_of_strings =  ["Vote #TRUMP2020ToSaveAmerica from corrupt Joe Biden and the radical left.","Je veux aller dormir. #fatigué"]
>>> print(bertmoticon.infer(ls_of_strings,3))
    [{'😂': '0.1938', '😡': '0.1866', '🙄': '0.0847'}, {'😴': '0.1547', '😭': '0.1507', '😩': '0.0892'}]
```
## bertmoticon.infer_mappings
Takes in a ```list``` of ```strings```, a dictionary ```dict``` of the emoji mappings, and an ```int``` number of guesses. It returns the number of occurences of each key value. We define the dictionary and the list as follows:

```
>>> mappings = {"Anger":['😡'], "Other":['😂','😭']}
>>> ls_of_strings =  ["Vote #TRUMP2020ToSaveAmerica from corrupt Joe Biden and the radical left.","Je veux aller dormir. #fatigué"]
```
The key values are the category names and the values are lists of the emojis contained in that category. Then parsed into the ```bertmoticon.infer_mappings``` returns:
```
>>>print(bertmoticon.infer_mappings(ls_of_strings,mappings,3))
{'Anger': 1, 'Other': 2}
```

