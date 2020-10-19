import torch
import torch.nn as nn
import transformers
import tarfile
import os,requests,sys
 

all_categories = ['😂','😭','😍','😊','🙏','😅','😁','🙄','😘','😔','😩','😉','😎','😢','😆','😋','😌','😳','😏','🙂','😃','🙃','😒','😜','😀','😱','🙈','😄','😡','😬','🙌','😴','😫','😪','😤','😇','😈','😞','😷','😣','😥','😝','😑','😓','😕','😹','😐','😻','😖','😛','😠','🙊','😰','😚','😲','😶','😮','🙁','😵','😗','😟','😨','🙇','🙋','😙','😯','🙆',
    '🙉','😧','😿','😸','🙀','😦','😽','😺','😼','🙅','😾','🙍','🙎']

def download():
    link = "http://izbicki.me/public/cs/bertmoticon.tgz"
    file_name = "download_model.tgz"
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
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
    _extract()
    _remove()

def _extract():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = "download_model.tgz"
    tf = tarfile.open(filename)
    print("Excracting...")
    tf.extractall(dir_path + "/model/")

def _remove():
    print("Removing zip file...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.remove(dir_path + "/download_model.tgz")


def infer(lines:list,guesses =80):
    """
    Given list of strings, returns a list of dictionaries providing emoji and percent for each line in lines

    Parameters:
        lines (list): list of strings


        guesses (int): number of returned emojis for each string in lines

    Returns:
        (list): list of dictionaries for each sentece provided


    >>> infer(["Vote #TRUMP2020ToSaveAmerica from corrupt Joe Biden and the radical left.","MASKS, YOU NEED TO WEAR THEM PEOPLE! ","je suis fatigue, dormir","god i love macdonalds ughhh"],3)
    [{'😂': '0.1938', '😡': '0.1866', '🙄': '0.0847'}, {'😷': '0.4744', '😂': '0.0824', '🙄': '0.0397'}, {'😴': '0.1853', '😭': '0.1700', '😂': '0.0712'}, {'😭': '0.3475', '😩': '0.1859', '😍': '0.1412'}]
    
    >>> infer([],3)
    Traceback (most recent call last):
      File "/usr/lib/python3.6/doctest.py", line 1330, in __run
        compileflags, 1), test.globs)
      File "<doctest __main__.infer[1]>", line 1, in <module>
        infer([],3)
      File "predict.py", line 74, in infer
        raise ValueError("The input list is empty")
    ValueError: The input list is empty
    """
    if len(lines)==0:
        raise ValueError("The input list is empty")
    elif type(lines) is int:
        raise ValueError("Input arguments are of type (list) and (int) in this order")

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
            self.fc_class = nn.Linear(768,len(all_categories))

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.listdir(dir_path +'/model/') :
        print("Downloading pre-trained model...")
        download()
        

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

    for xs in range(len(lines)):
        for i in range(k):
            #get results
            guess_i = top_i[xs,i].item()
            guess = all_categories[guess_i]
            percent = '%0.4f'%top_n[xs,i].item()

            #append to lists
            lst_emoji.append(guess)
            lst_perc.append(percent)
        #append dictionary to list
        return_lst.append(dict(zip(lst_emoji, lst_perc)))

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
    {'all_categories': 10}
    """
    if type(mappings) is int:
        guesses = mappings
        mappings={"all_categories":all_categories}
        

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
            self.fc_class = nn.Linear(768,len(all_categories))

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.listdir(dir_path +'/model/') :
        print("Downloading pre-trained model...")
        download()
   

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
            guess = all_categories[guess_i]
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

