'''
FIXME:
add an appropriate docstring for this file
'''


import os


__version__ = "0.1.0"


emojis = ['😂','😭','😍','😊','🙏','😅','😁','🙄','😘','😔','😩','😉','😎','😢','😆','😋','😌','😳','😏','🙂','😃','🙃','😒','😜','😀','😱','🙈','😄','😡','😬','🙌','😴','😫','😪','😤','😇','😈','😞','😷','😣','😥','😝','😑','😓','😕','😹','😐','😻','😖','😛','😠','🙊','😰','😚','😲','😶','😮','🙁','😵','😗','😟','😨','🙇','🙋','😙','😯','🙆','🙉','😧','😿','😸','🙀','😦','😽','😺','😼','🙅','😾','🙍','🙎']


# FIXME:
# this should be set to the data directory for the package
# see https://stackoverflow.com/a/5601839/1241368
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'bertmoticon')


def is_model_downloaded():
    '''
    Check whether the model has been downloaded or not.
    '''
    return os.path.exists(model_path)


def download_model(force_redownload=False):
    '''
    Download the pretrained weights for the bertmoticon model if they are not already downloaded.

    This function only needs to be called once,
    and is automatically called when the infer function is called for the first time.
    '''
    import requests
    import sys
    import tarfile
    import tempfile

    url = "https://izbicki.me/public/cs/bertmoticon.tgz"

    if is_model_downloaded() and not force_redownload:
        return

    # download the model tgz file
    print("Downloading bermoticon model")
    fd, temp_path = tempfile.mkstemp()
    with open(temp_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()
    os.close(fd)
    print()

    # extract the model
    print("Extracting the model")
    tf = tarfile.open(temp_path)
    tf.extractall(model_path)

    # delete the temporary file
    os.remove(temp_path)


def infer(lines:list, guesses=80):
    """
    Given list of strings, returns a list of dictionaries providing emoji and percent for each line in lines

    Parameters:
        lines (list): list of strings

        guesses (int): number of returned emojis for each string in lines

    Returns:
        (list): list of dictionaries for each sentece provided


    >>> infer(["Vote #TRUMP2020ToSaveAmerica from corrupt Joe Biden and the radical left.","MASKS, YOU NEED TO WEAR THEM PEOPLE! ","je suis fatigue, dormir","god i love macdonalds ughhh"],3)
    [{'😂': '0.1938', '😡': '0.1866', '🙄': '0.0847'}, {'😷': '0.4744', '😂': '0.0824', '🙄': '0.0397'}, {'😴': '0.1853', '😭': '0.1700', '😂': '0.0712'}, {'😭': '0.3475', '😩': '0.1859', '😍': '0.1412'}]
    """
    import torch
    import torch.nn as nn
    import transformers
 
    # validate input
    if type(lines) is not list:
        raise ValueError("Input arguments are of type (list) and (int) in this order")
    elif len(lines)==0:
        raise ValueError("The input list is empty")

    # download the model weights if not already downloaded
    download_model()

    # check CUDA availability 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def str_to_tensor_bert(lines:str):

        max_length = 64
        encodings = []
        for line in lines:
            encoding = tokenizer.encode_plus(
                line,
                #add_special_tokens = True,
                max_length = max_length,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt',
                )
            encodings.append(encoding)
        input_ids = torch.cat([ encoding['input_ids'] for encoding in encodings ],dim=0).to(device)
        attention_mask = torch.cat([ encoding['attention_mask'] for encoding in encodings ],dim=0).to(device)
        return input_ids,attention_mask

    # include all layer model 
    model_name = 'bert-base-multilingual-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # FIXME:
    # you have a LOT of junk code in here that needs to be cleaned up
    all_layers = True
    if all_layers == False:
        #Include last layer model
        bert = transformers.BertModel.from_pretrained(model_name)
    class BertFineTuning(nn.Module):
        def __init__(self):
            super().__init__()
            if all_layers:
                self.bert =  transformers.BertModel.from_pretrained(model_name)
            embedding_size = 128
            self.fc_class = nn.Linear(768,len(emojis))

        def forward(self,x):
            input_ids, attention_mask = x
            if all_layers:
                last_layer,embedding = self.bert(input_ids)
            else:
                last_layer,embedding = bert(input_ids)
            embedding = torch.mean(last_layer,dim=1)
            out = self.fc_class(embedding)
            return out, None

    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # FIXME:
    # do you know what this line is supposed to do?
    # as-is, it doesn't do anything but mess up settings for advanced users
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = BertFineTuning()
    model.to(device)
    softmax = torch.nn.Softmax(dim=1)
    model.eval()

    # load the pretrained model
    # FIXME:
    # every time you call the infer function, you are reloading the model, which is incredibly slow
    # you need to make your model variable static function and only load it on the first call to infer
    # see: https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    model_dict = torch.load(os.path.join(model_path,'babel/model'))
    model.load_state_dict(model_dict['model_state_dict'], strict=False)

    k = guesses
    return_lst = []
    lst_emoji = [] 
    lst_perc = []

    # catch out of bounds requests
    # FIXME:
    # all input validation should be done at the beginning of the function
    k = 80 if guesses>80 else 1 if guesses<0 else guesses

    opt_line_tensors = str_to_tensor_bert(lines)
    output_class,output_nextchars = model(opt_line_tensors)
    probs = softmax(output_class)
    top_n, top_i = probs.topk(k)

    for xs in range(len(lines)):
        for i in range(k):
            #get results
            guess_i = top_i[xs,i].item()
            guess = emojis[guess_i]
            percent = '%0.4f'%top_n[xs,i].item()

            #append to lists
            lst_emoji.append(guess)
            lst_perc.append(percent)
        #append dictionary to list
        return_lst.append(dict(zip(lst_emoji, lst_perc)))
        
        # FIXME:
        # the comments in this section are not particularly useful,
        # for example, "clear lists" doesn't tell me anything that I couldn't more easily get
        # from just reading the code that says list.clear()
        # a comment should either:
        # 1) summarize a whole "paragraph" of code into a single line, or
        # 2) tell my *why* you are doing something

        #clear lists 
        lst_emoji.clear()
        lst_perc.clear()
    return return_lst


def infer_mappings(lines:list,mappings:dict,guesses =80):
    """
    
    Given dictionary mappings of the given 80 emojis & list of strings, returns a dictionary of the mappings accompanied by the predicted 
    emoji occurence from the list of lines. 

    Parameters:
        lines (list): list of strings

        mappings (dict): dictionary with keys being the name of the mappings and the values being the list of emojis in that category 

        guesses (int): number of returned emojis for each string in lines

    Returns:
        (dict): dictionary of the mappings accompanied by the predicted emoji occurence from the list of lines

    doc testing 
    >>> infer_mappings(["testing the outputs","hope it works xd"],{"category_1":['😂'],"category_2":['😭']},5)
    {'category_1': 2, 'category_2': 2}


    >>> infer_mappings(["testing the outputs","hope it works xd"],5)
    {'emojis': 10}
    """
    if type(mappings) is int:
        guesses = mappings
        mappings={"emojis":emojis}
        
    # FIXME:
    # this function has a LOT of repeated code from your infer function;
    # you need to remove the duplicated code from this function,
    # and call the infer function within this function

    #Check CUDA availability 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def str_to_tensor_bert(lines:str):

        max_length = 64
        encodings = []
        for line in lines:
            encoding = tokenizer.encode_plus(
                line,
                #add_special_tokens = True,
                max_length = max_length,
                pad_to_max_length = True,
                return_attention_mask = True,
                return_tensors = 'pt',
                )
            encodings.append(encoding)
        input_ids = torch.cat([ encoding['input_ids'] for encoding in encodings ],dim=0).to(device)
        attention_mask = torch.cat([ encoding['attention_mask'] for encoding in encodings ],dim=0).to(device)
        return input_ids,attention_mask


    #Include all layer model 
    model_name = 'bert-base-multilingual-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    all_layers = True
    if all_layers == False:
        #Include last layer model
        bert = transformers.BertModel.from_pretrained(model_name)
    class BertFineTuning(nn.Module):
        def __init__(self):
            super().__init__()
            if all_layers:
                self.bert =  transformers.BertModel.from_pretrained(model_name)
            embedding_size = 128
            self.fc_class = nn.Linear(768,len(emojis))

        def forward(self,x):
            input_ids, attention_mask = x
            if all_layers:
                last_layer,embedding = self.bert(input_ids)
            else:
                last_layer,embedding = bert(input_ids)
            embedding = torch.mean(last_layer,dim=1)
            out = self.fc_class(embedding)
            return out, None

    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = BertFineTuning()
    model.to(device)
    softmax = torch.nn.Softmax(dim=1)
    model.eval()

    #import the pretrained model
    model_dict = torch.load('model/babel/model')
    model.load_state_dict(model_dict['model_state_dict'],strict = False)

    
    k = guesses
    return_lst = []
    lst_emoji = [] 
    lst_perc = []

    #Catch out of bounds requests
    k = 80 if guesses>80 else 1 if guesses<0 else guesses

    opt_line_tensors = str_to_tensor_bert(lines)
    output_class,output_nextchars = model(opt_line_tensors)
    probs = softmax(output_class)
    top_n, top_i = probs.topk(k)

    #print(top_n,len(top_n))
    for xs in range(len(lines)):
        for i in range(k):
            #get results
            guess_i = top_i[xs,i].item()
            guess = emojis[guess_i]
            percent = '%0.4f'%top_n[xs,i].item()

            #append to lists
            lst_emoji.append(guess)
            lst_perc.append(percent)
        #append dictionary to list
        return_lst.append(dict(zip(lst_emoji, lst_perc)))

        #clear lists 
        lst_emoji.clear()
        lst_perc.clear()

    #map the emojis to the user's dictionary mappings
    return_dict = {}
    for key, value in mappings.items():
        return_dict[key] = 0

    for i in range(len(return_lst)):
        for key, value in return_lst[i].items():
            for category, emoji_cat in mappings.items():
                if key in emoji_cat:
                    return_dict[category] += 1       
    return return_dict

