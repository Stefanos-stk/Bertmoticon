# Bertmoticon 

The Bertmoticon package is fine-tuned from the [BERT](https://github.com/google-research/bert) model, to the emoji prediction task. It can predict emojis in 102 languages. 

## Instalation

You can install the Bertmoticon package from [PyPI](https://pypi.org/):

```
pip3 install bertmoticon
```

## How to use

```
import bertmoticon.predict

print(all_categories)
```
prints the emojis supported by the library:
```
 ['😂', '😭', '😍', '😊', '🙏', '😅', '😁', '🙄', '😘', '😔', '😩', '😉', '😎', '😢', '😆', '😋', '😌', '😳', '😏', '🙂', '😃', '🙃', '😒', '😜', '😀', '😱', '🙈', '😄', '😡', '😬', '🙌', '😴', '😫', '😪', '😤', '😇', '😈', '😞', '😷', '😣', '😥', '😝', '😑', '😓', '😕', '😹', '😐', '😻', '😖', '😛', '😠', '🙊', '😰', '😚', '😲', '😶', '😮', '🙁', '😵', '😗', '😟', '😨', '🙇', '🙋', '😙', '😯', '🙆', '🙉', '😧', '😿', '😸', '🙀', '😦', '😽', '😺', '😼', '🙅', '😾', '🙍', '🙎']
```

The first function is:
```
dict_results = infer_mappings(["This covid pandemic is getting rougher. WEAR YOUR MASK","Je veux dormir... et tout oublier."],{"Category_1":["😂", "😭"],"Category_2":["😷"]},5)
print(dict_results)
```
and it returns:
```
{'Category_1': 4, 'Category_2': 1}
```
The arguments of the infer_mappings function are:

- list of strings : ```["This covid pandemic is getting rougher. WEAR YOUR MASK","Je veux dormir... et tout oublier."]```
- dictionary categorizing emojis: ```{"Category_1":["😂", "😭"],"Category_2":["😷"]}```
- number of guesses per sentence: ```5```

It returns the number of occurences of each emoji and places it under the categories in the given dictionary.

The other function is:
```
list_results = infer(["This covid pandemic is getting rougher. WEAR YOUR MASK","Je veux dormir... et tout oublier."],5)
print(list_results)
```
returning a list of dictionaries containing the guesses and the percentages for each given string:
```
[{'😷': '0.3458', '😂': '0.0502', '😭': '0.0415', '😔': '0.0414', '😢': '0.0403'}, 
{'😴': '0.0966', '😭': '0.0888', '😂': '0.0885', '😔': '0.0719', '🙄': '0.0640'}]
```

The fine-tuned model does is not inside of the pypi package; instead it is downloaded upon first usage of the ```infer``` function. Total size of the mode is 1.34GB.


## How to download pre-trained model

Using the function ```download()``` you can download the fine-tuned model. The fine-tuned is also downloaded when using the ```infer``` or ```infer_mappings```.