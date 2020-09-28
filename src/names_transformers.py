#!/usr/bin/env python3

# process command line args
import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser_control = parser.add_argument_group('control options')
parser_control.add_argument('--infer',action='store_true')
parser_control.add_argument('--train',action='store_true')
parser_control.add_argument('--generate',action='store_true')
parser_control.add_argument('--evaluate',action='store_true')

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data',default='names')
parser_data.add_argument('--data_format',choices=['names','headlines','emoji'],default='emoji')
parser_data.add_argument('--sample_strategy',choices=['uniform_line','uniform_category','linear_choice'],default='linear_choice')
parser_data.add_argument('--case_insensitive',action='store_true')
parser_data.add_argument('--dropout',type=float,default=0.0)

parser_model = parser.add_argument_group('model options')
parser_model.add_argument('--model',choices=['cnn','rnn','gru','lstm','bert'],default='bert')
parser_model.add_argument('--resnet',action='store_true')
parser_model.add_argument('--hidden_layer_size',type=int,default=128)
parser_model.add_argument('--num_layers',type=int,default=1)
parser_model.add_argument('--conditional_model',action='store_true')

parser_opt = parser.add_argument_group('optimization options')
parser_opt.add_argument('--batch_size',type=int,default=1)
parser_opt.add_argument('--learning_rate',type=float,default=1e-1)
parser_opt.add_argument('--optimizer',choices=['sgd','adam'],default='sgd')
parser_opt.add_argument('--gradient_clipping',action='store_true')
parser_opt.add_argument('--momentum',type=float,default=0.9)
parser_opt.add_argument('--weight_decay',type=float,default=1e-4)
parser_opt.add_argument('--samples',type=int,default=100000000)
parser_opt.add_argument('--input_length',type=int)
parser_opt.add_argument('--warm_start')
parser_opt.add_argument('--disable_categories',action='store_true')

parser_infer = parser.add_argument_group('inference options')
parser_infer.add_argument('--infer_path',default='explain_outputs')

parser_generate = parser.add_argument_group('generate options')
parser_generate.add_argument('--temperature',type=float,default=1.0)
parser_generate.add_argument('--max_sample_length',type=int,default=100)
parser_generate.add_argument('--category',nargs='*')

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
parser_debug.add_argument('--print_delay',type=int,default=5)
parser_debug.add_argument('--log_dir_base',type=str,default='log')
parser_debug.add_argument('--log_dir',type=str)
parser_debug.add_argument('--save_every',type=int,default=1000)
parser_debug.add_argument('--print_every',type=int,default=100)

parser_bert = parser.add_argument_group('bert specifics')
parser_bert = parser.add_argument('--train_where',choices =['inside','outside'],default = 'outside')


parser_eval = parser.add_argument_group('eval options')
parser_eval.add_argument('--csv_name',type=str,default='eval.csv')
args = parser.parse_args()

if args.model=='cnn' and args.input_length is None:
    raise ValueError('if --model=cnn, then you must specify --input_length')

# load args from file if warm starting
if args.warm_start is not None:
    import sys
    import os
    args_orig = args
    args = parser.parse_args(['@'+os.path.join(args.warm_start,'args')]+sys.argv[1:])
    args.train = args_orig.train

# supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load modules
import datetime
import glob
import os
import math
import random
import string
import sys
import time
import unicodedata
from unidecode import unidecode

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import transformers

torch.cuda.empty_cache()

from sklearn import metrics

import demoji
import re
import spacy
#demoji.download_codes()
# set device to cpu/gpu

with torch.no_grad():
    if args.device=='gpu' or (torch.cuda.is_available() and args.device=='auto'):
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    else:
        device = torch.device('cpu')
    print('device=',device)

    # import the training data
    BOL = '\x00'
    EOL = '\x01'
    OOV = '\x02'
    if args.case_insensitive:
        vocabulary = string.ascii_lowercase
    else:
        vocabulary = string.ascii_letters
    vocabulary += " .,;'" + '1234567890:-/#$%' + OOV + BOL + EOL
    print('len(vocabulary)=',len(vocabulary))

    def unicode_to_ascii(s):
        '''
        Removes diacritics from unicode characters.
        See: https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in vocabulary
        )

    def format_line(line):
        line = demoji.replace(line)
        line = re.sub(r"(@\S*)", "@", line)
        line = re.sub(r"http://\S*", "url",line)
        return line

    # Build the category_lines dictionary, a list of names per language
    if args.data_format == 'names':
        category_lines = {}
        all_categories = []
        #for filename in glob.glob(os.path.join(args.data,'*.txt')):
        for filename in glob.glob(os.path.join(args.data,'*')):
            print('filename=',filename)
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            lines = [format_line(line) for line in lines]
            category_lines[category] = lines


    elif args.data_format == 'emoji':
        import numpy as np
        all_categories = []
        emojis = open("list.txt", "rt").read()
        all_categories = list(emojis)
        #all_emojis = np.array(all_emojis)
        print("Number of categories = ", len(all_categories))
        specs = open("specifics_emoji.txt","rt").read()
        spcefic_emojis = list(specs)
        specs_array = np.array(spcefic_emojis)
    


    def str_to_tensor(ss,input_length=None):
        '''
        Converts a list of strings into a tensor of shape <max_length, len(ss), len(vocabulary)>.
        This is used to convert text into a form suitable for input into a RNN/CNN.
        '''
        max_length = max([len(s) for s in ss]) + 2
        if input_length:
            max_length = input_length
        tensor = torch.zeros(max_length, len(ss), len(vocabulary)).to(device)
        for j,s in enumerate(ss):
            s = BOL + s + EOL
            for i, letter in enumerate(s):
                if i<max_length:
                    vocabulary_i = vocabulary.find(letter)
                    if vocabulary_i==-1:
                        vocabulary_i = vocabulary.find(OOV)
                    tensor[i,j,vocabulary_i] = 1
        return tensor

    def str_to_tensor_bert(lines):
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
        input_ids = torch.cat([ encoding['input_ids'] for encoding in encodings ],dim=0)
        attention_mask = torch.cat([ encoding['attention_mask'] for encoding in encodings ],dim=0)
        return input_ids,attention_mask

    # define the model
    input_size = len(vocabulary)
    if args.conditional_model:
        input_size += len(all_categories)

    class RNNModel(nn.Module):
        def __init__(self):
            super(RNNModel,self).__init__()
            if args.model=='rnn':
                mk_rnn = nn.RNN
            if args.model=='gru':
                mk_rnn = nn.GRU
            if args.model=='lstm':
                mk_rnn = nn.LSTM
            self.rnn = mk_rnn(
                    input_size,
                    args.hidden_layer_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                    )
            self.fc_class = nn.Linear(args.hidden_layer_size,len(all_categories))
            self.dropout = nn.Dropout(args.dropout)
            self.fc_nextchars = nn.Linear(args.hidden_layer_size,len(vocabulary))

        def forward(self, x):
            # out is 3rd order: < len(line) x batch size x hidden_layer_size >
            out,h_n = self.rnn(x)
            out = self.dropout(out)
            out_class = self.fc_class(out[out.shape[0]-1,:,:])
            out_nextchars = torch.zeros(out.shape[0] , out.shape[1], len(vocabulary) ).to(device)
            for i in range(out.shape[0]):
                out_nextchars[i,:,:] = self.fc_nextchars(out[i,:,:])
            return out_class, out_nextchars

    class ResnetRNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            if args.model=='rnn':
                mk_rnn = nn.RNN
            if args.model=='gru':
                mk_rnn = nn.GRU
            if args.model=='lstm':
                mk_rnn = nn.LSTM
            rnn_input_size = input_size
            self.rnns = []
            for layer in range(args.num_layers):
                rnn = mk_rnn(
                        rnn_input_size,
                        args.hidden_layer_size,
                        num_layers=1,
                        )
                self.add_module('rnn'+str(layer),rnn)
                self.rnns.append(rnn)
                rnn_input_size = args.hidden_layer_size
            self.fc_class = nn.Linear(args.hidden_layer_size,len(all_categories))
            self.dropout = nn.Dropout(args.dropout)
            self.fc_nextchars = nn.Linear(args.hidden_layer_size,len(vocabulary))

        def forward(self, x):
            # out is 3rd order: < len(line) x batch size x hidden_layer_size >
            out = x
            for layer,rnn in enumerate(self.rnns):
                out_prev = out
                out,_ = rnn(out)
                if layer>0 and args.resnet:
                    out = out + out_prev
                out = self.dropout(out)
            out_class = self.fc_class(out[out.shape[0]-1,:,:])
            out_nextchars = torch.zeros(out.shape[0] , out.shape[1], len(vocabulary) ).to(device)
            for i in range(out.shape[0]):
                out_nextchars[i,:,:] = self.fc_nextchars(out[i,:,:])
            return out_class, out_nextchars


    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel,self).__init__()
            self.relu = nn.ReLU()
            self.cnn = \
                nn.Conv1d(input_size,args.hidden_layer_size,3,padding=1)
            self.cnns = (args.num_layers-1)*[
                nn.Conv1d(args.hidden_layer_size,args.hidden_layer_size,3,padding=1)
                ]
            self.dropout = nn.Dropout(args.dropout)
            self.fc_class = nn.Linear(args.hidden_layer_size*args.input_length,len(all_categories))
            self.fc_nextchars = nn.Linear(args.hidden_layer_size,input_size)

        def forward(self,x):
            out = torch.einsum('lbv->bvl',x)
            out = self.cnn(out)
            out = self.relu(out)
            for cnn in self.cnns:
                out = cnn(out)
                out = self.relu(out)
                out = self.dropout(out)
            out_class = out.view(args.batch_size,args.hidden_layer_size*args.input_length)
            out_class = self.fc_class(out_class)
            out = torch.einsum('ijk->kij',out)
            out_nextchars = torch.zeros([out.shape[0],out.shape[1],input_size])
            for i in range(out.shape[0]):
                out_nextchars[i,:,:] = self.fc_nextchars(out[i,:])
            return out_class, out_nextchars


    model_name = 'bert-base-multilingual-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    if args.train_where == 'outside':
        bert = transformers.BertModel.from_pretrained(model_name)
        print('bert.config.vocab_size=',bert.config.vocab_size)
    class BertFineTuning(nn.Module):
        def __init__(self):
            super().__init__()
            if args.train_where == 'inside':
                self.bert =  transformers.BertModel.from_pretrained(model_name)
            embedding_size = args.hidden_layer_size
            self.fc_class = nn.Linear(768,len(all_categories))

        def forward(self,x):
            input_ids, attention_mask = x
            if args.train_where == 'inside':
                last_layer,embedding = self.bert(input_ids)
            else:
                last_layer,embedding = bert(input_ids)
            embedding = torch.mean(last_layer,dim=1)
            out = self.fc_class(embedding)
            return out, None

    # load the model
    if args.model=='bert':
        model = BertFineTuning()
    elif args.model=='cnn':
        model = CNNModel()
    else:
        if args.resnet:
            model = ResnetRNNModel()
        else:
            model = RNNModel()
    model.to(device)

    import pprint
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib
    import matplotlib.font_manager as font_manager
    import os



    if args.warm_start:
        print('warm starting model from',args.warm_start)
        model_dict = torch.load(os.path.join(args.warm_start,'model'))
        model.load_state_dict(model_dict['model_state_dict'],strict = False)
            
        def metrics_report_to_df(ytrue, ypred, specs):
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred,labels = specs)
            classification_report = pd.concat(map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
            classification_report.columns = ["precision", "recall", "f1-score", "support"]
            classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, ypred, average='weighted')
            classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum() 
            return(classification_report)
        
        def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
            """
            Generate matrix plot of confusion matrix with pretty annotations.
            The plot image is saved to disk.
            args: 
            y_true:    true label of the data, with shape (nsamples,)
            y_pred:    prediction of the data, with shape (nsamples,)
            filename:  filename of figure file to save
            labels:    string array, name the order of class labels in the confusion matrix.
                        use `clf.classes_` if using scikit-learn models.
                        with shape (nclass,).
            ymap:      dict: any -> string, length == nclass.
                        if not None, map the labels & ys to more understandable strings.
                        Caution: original y_true, y_pred and labels must align.
            figsize:   the size of the figure plotted.
            """
            #plt.rcParams['font.family'] = 'Segoe UI Emoji'

            if ymap is not None:
                y_pred = [ymap[yi] for yi in y_pred]
                y_true = [ymap[yi] for yi in y_true]
                labels = [ymap[yi] for yi in labels]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            print(cm)
            print(type(cm))
            file_name = filename  + ".txt"
            a_file = open(file_name, "w")
            for row in cm:
                np.savetxt(a_file, row)
            a_file.close()

            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            annot = np.empty_like(cm).astype(str)
            nrows, ncols = cm.shape
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    p = cm_perc[i, j]
                    if i == j:
                        s = cm_sum[i]
                        annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                    elif c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm = pd.DataFrame(cm, index=labels, columns=labels)
            cm.index.name = 'Actual'
            cm.columns.name = 'Predicted'
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cm, annot=annot, fmt='', ax=ax,cmap="YlGnBu",robust=True)
            plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)


    # training
    if args.train or args.evaluate:
        from collections import defaultdict

        langs = defaultdict(lambda: {})
        langs = {
            'true': defaultdict(lambda: []),
            'pred': defaultdict(lambda: []),
            }
        countries =defaultdict(lambda: [])
        countries ={
            'true': defaultdict(lambda: []),
            'pred': defaultdict(lambda: []),
        }
        y_true = []
        y_pred = []
        # create log_dir
        log_dir = args.log_dir
        if log_dir is None:
            log_dir = os.path.join(args.log_dir_base,(
                'model='+args.model+
                '_hidden='+str(args.hidden_layer_size)+
                '_layers='+str(args.num_layers)+
                '_cond='+str(args.conditional_model)+
                '_resnet='+str(args.resnet)+
                '_lr='+str(args.learning_rate)+
                '_optim='+args.optimizer+
                '_clip='+str(args.gradient_clipping)+
                '_'+str(datetime.datetime.now())
                ))
        try:
            os.makedirs(log_dir)
            with open(os.path.join(log_dir,'args'), 'w') as f:
                f.write('\n'.join(sys.argv[1:]))
        except FileExistsError:
            print('cannot create log dir,',log_dir,'already exists')
            sys.exit(1)
        writer = SummaryWriter(log_dir=log_dir)

        # prepare model for training
        criterion = nn.CrossEntropyLoss()
        print('model.parameters()=',list(model.parameters()))
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                    )
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                    )
        model.train()
        import gzip
        import json

        # training loop
        start_time = time.time()
        step = 1
        flag_completed = False
        with gzip.open(args.data,'rb') as file:
            while True:
                    
                    categories = []
                    lines = []
                    country_codes = []
                    languages = []
                    while (len(lines) != args.batch_size):
                        try:
                            tweet = json.loads(next(file))
                            if tweet is None:
                                flag_completed = True
                                break
                        except json.decoder.JSONDecodeError:
                            continue
                        except StopIteration:
                            flag_completed = True
                            break
                        text = tweet['text']
                        line = format_line(text)  
                        category = emoji = tweet['emoji']
                        lang = tweet['lang']
                        languages.append(lang)
                        countrycode = tweet['country_code']
                        country_codes.append(countrycode)
                        categories.append(all_categories.index(category))
                        lines.append(line)
                    if flag_completed:
                        break 
                    step = step + 1
                    category_tensor = torch.tensor(categories, dtype=torch.long).to(device)
                    if args.model=='bert':
                        input_tensor = str_to_tensor_bert(lines)
                    else:
                        line_tensor = str_to_tensor(lines,args.input_length)

                        if args.conditional_model:
                            category_onehot = torch.nn.functional.one_hot(category_tensor, len(all_categories)).float()
                            category_onehot = torch.unsqueeze(category_onehot,0)
                            category_onehot = torch.cat(line_tensor.shape[0]*[category_onehot],dim=0)
                            input_tensor = torch.cat([line_tensor,category_onehot],dim=2)
                        else:
                            input_tensor = line_tensor

                        input_tensor = input_tensor.to(device)
                    category_tensor = category_tensor.to(device)

                    # perform training step
                    output_class,output_nextchars = model(input_tensor)
                    loss_class = criterion(output_class, category_tensor)
                    if args.model=='bert':
                        loss_nextchars = torch.tensor(0.0)
                    else:
                        loss_nextchars_perchar = torch.zeros(output_nextchars.shape[0]).to(device)
                        for i in range(output_nextchars.shape[0]-1):
                            _, nextchar_i = line_tensor[i+1,:].topk(1)
                            nextchar_i = nextchar_i.view([-1])
                            loss_nextchars_perchar[i] = criterion(output_nextchars[i,:], nextchar_i)
                        loss_nextchars = torch.mean(loss_nextchars_perchar)

                    if not args.evaluate:
                        if args.conditional_model or args.disable_categories:
                            loss = loss_nextchars
                        else:
                            loss = loss_class + loss_nextchars
                        loss.backward()
                        grad_norm = sum([ torch.norm(p.grad)**2 for p in model.parameters() if p.grad is not None])**(1/2)
                        if args.gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        optimizer.step()

                    #2*((precision*recall)/(precision+recall))
                    # log optimization information
                    writer.add_scalar('train/loss_class', loss_class.item(), step)
                    writer.add_scalar('train/loss_nextchars', loss_nextchars.item(), step)
                    if not args.evaluate:
                        writer.add_scalar('train/loss', loss.item(), step)
                        writer.add_scalar('train/grad_norm', grad_norm.item(), step)

                    # get accuracy@k
                    ks = [1,5]
                    k = max(ks)
                    top_n, top_i = output_class.topk(k)
                    category_tensor_k = torch.cat(k*[torch.unsqueeze(category_tensor,dim=1)],dim=1)
                    accuracies = torch.where(
                            top_i[:,:]==category_tensor_k,
                            torch.ones([args.batch_size,k]).to(device),
                            torch.zeros([args.batch_size,k]).to(device)
                            )

                    for k in ks:
                        accuracies_k,_ = torch.max(accuracies[:,:k], dim=1)
                        accuracy_k = torch.mean(accuracies_k).item()
                        writer.add_scalar('accuracy/@'+str(k), accuracy_k, step)

                    # print status update
                    if step % args.print_every == 0 or True:
                        
                        # get category from output
                        for xs in range(args.batch_size):
                            top_n, top_i = output_class.topk(1)
                            guess_i = top_i[xs].item()
                            category_i = category_tensor[xs]
                            guess = all_categories[guess_i]
                            category = all_categories[category_i]
                            y_true.append(category)
                            y_pred.append(guess)

                            langs['true'][languages[xs]].append(category)
                            langs['pred'][languages[xs]].append(guess)

                            countries['true'][country_codes[xs]].append(category)
                            countries['pred'][country_codes[xs]].append(guess)

                            

                
                    # save model
                    if (step)%args.save_every == 0 or (step)==args.samples:
                        print('saving model checkpoint')
                        if not args.evaluate:
                            torch.save({
                                    'step':step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss':loss
                                }, os.path.join(log_dir,'model'))
                        else:
                            torch.save({
                                    'step':step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                }, os.path.join(log_dir,'model'))

            from sklearn import metrics
            import pandas as pd
            import numpy as np
            import pprint
            
            lst = []
            if args.evaluate:
                print("---------------top_15----------------------")
                cm_analysis(y_true,y_pred,'top15_confusion.pdf',labels = list(specs),figsize=(40,40))
                #def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
                print("---------------all-80----------------------")
                cm_analysis(y_true,y_pred,'all80_confusion-testin.pdf',labels = list(emojis),figsize=(100,100))
                #print("Accuracy ",metrics.accuracy_score(y_true, y_pred))
                #print("F1-Score ",metrics.f1_score(y_true, y_pred,average='weighted'))
                #rint("Classification Report :")
                '''
                
                classification_report = metrics_report_to_df(y_true, y_pred,specs_array)
                #pprint.pprint(classification_report)
                tmp0 = 'general_report_' + args.csv_name
                classification_report.to_csv(tmp0)

                complete_country_report = pd.DataFrame(columns = ["precision" , "recall",  "f1-score"  ,"support" ,"country"])
                for key in countries['true'].keys():
                    classification_report = metrics_report_to_df(countries['true'][key],countries['pred'][key],specs_array)

                    classification_report.insert(4,"country",key,True)
                    complete_country_report = complete_country_report.append(classification_report)

                    #copy = classification_report.tail(1).copy()
                    #copy.insert(4,"country",key,True)
                    #complete_country_report = complete_country_report.append(copy)
                
                
                tmp1 = 'country_report_' + args.csv_name
                complete_country_report.to_csv(tmp1)

                complete_lang_report = pd.DataFrame(columns = ["precision" , "recall",  "f1-score"  ,"support" ,"lang"])
                for key in langs['true'].keys():
                    classification_report = metrics_report_to_df(langs['true'][key],langs['pred'][key],specs_array)

                    
                    classification_report.insert(4,"lang",key,True)
                    complete_lang_report = complete_lang_report.append(classification_report)

                    #copy = classification_report.tail(1).copy()
                    #copy.insert(4,"lang",key,True)
                    #complete_lang_report = complete_lang_report.append(copy)

                tmp2 = 'language_report_' + args.csv_name
                complete_lang_report.to_csv(tmp2)
                '''

    # infer
    def infer(line):
        line = line.strip()
        if args.case_insensitive:
            line = line.lower()
        line_tensor = str_to_tensor_bert([line])
        output_class,output_nextchars = model(line_tensor)
        probs = softmax(output_class)
        k=5
        top_n, top_i = probs.topk(k)
        print('line=',line)
        for i in range(k):
            guess = all_categories[top_i[0,i].item()]
            print('  ',i,guess, '(%0.2f)'%top_n[0,i].item())
        if args.infer_path is not None:
            i = 0
            while os.path.exists(os.path.join(args.infer_path,"line%s.char.png" % str(i).zfill(4))):
                i += 1
            path_base = os.path.join(args.infer_path,'line'+str(i).zfill(4))
            print('path_base=',path_base)
            explain(line, path_base+'.char.png', 'char') 
            #explain(line, path_base+'.word.png', 'word') 

    def explain(line,filename,explain_type):
        if True:
            from janome.tokenizer import Tokenizer as janome_tokenizer
            token_object = janome_tokenizer()


        formated_line  = format_line(line)
        input_tensor = str_to_tensor_bert(list(formated_line))
        x,_ = model(input_tensor)
        full_line_tensor = softmax(x)
        scores = torch.zeros([len(formated_line)])
        if explain_type == 'char':
            length_word = len(formated_line)
            list_of_lines = []
            for i in range(length_word):
                text = formated_line
                list_of_lines.append(manual_replace(text,' ',i))
            for indx,line in enumerate(list_of_lines):
                input_tensor = str_to_tensor_bert(list(line))
                y,_ = model(input_tensor)
                part_line_tensor = softmax(y)
                l2 = (torch.dist(full_line_tensor, part_line_tensor,2))**3
                scores[indx] = float(l2)
            line2img(formated_line,scores,filename)   

        else:
            list_of_words = []
            ja = spacy.blank('ja')
            for word in ja(formated_line):
                list_of_words.append(str(word))
            list_of_words = formated_line.split()
            list_of_lines = []
            for indx,word in enumerate(list_of_words):
                blank = ' '*(len(word)+1)
                if indx == 0:
                    text = blank + ' '.join(list_of_words[0:indx]) + ' '.join(list_of_words[indx+1:])
                elif indx == len(list_of_words)-1:
                    text =  ' '.join(list_of_words[0:indx]) + ' '.join(list_of_words[indx+1:]) + blank
                else: 
                    text = ' '.join(list_of_words[0:indx]) + blank + ' ' +  ' '.join(list_of_words[indx+1:])
                list_of_lines.append(text)

            tmp_list = []
            length = len(formated_line)
            real_count = 0
            for indx,line in enumerate(list_of_lines):
                input_tensor = str_to_tensor_bert(list(line))
                y,_ = model(input_tensor)
                part_line_tensor = softmax(y)
                l2 = (torch.dist(full_line_tensor, part_line_tensor,2))**3
                tmp_list.append(l2)
            
            real_count = 0
            for index,word in enumerate(list_of_words):
                for i in range(len(word)):
                    scores[real_count + i] = tmp_list[index]
                if index != len(list_of_words):
                    real_count = real_count + len(word) + 1

            line2img(formated_line,scores,filename) 



    def manual_replace(s, char, index):
        return s[:index] + char + s[index +1:]


 
    def line2img(
            line,
            scores,
            filename,
            maxwidth=50,
            img_width=800
            ):
        '''
        Outputs an image containing text with green/red background highlights to indicate the importance of words in the text.
        Arguments:
            line (str): the text that should be printed
            scores (Tensor): a vector of size len(line), where each index contains the "weight" of the corresponding letter in the line string; positive values will be colored green, and negative values red.
            filename (str): the name of the output file
        '''
        import matplotlib
        import matplotlib.colors as colors
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt    
        import numpy as np
        import math
        import matplotlib.font_manager as font_manager
        import os

        font_manager.FontEntry(fname = os.path.expanduser('~/project/fonts/ipag.ttf'))
        fp = matplotlib.font_manager.FontProperties(fname = os.path.expanduser('~/project/fonts/ipag.ttf'))
        plt.rcParams['font.family'] = fp.get_name()

        im_height=1+len(line)//maxwidth
        im=np.zeros([maxwidth,im_height])
        for i in range(scores.shape[0]):
            im[i%maxwidth,im_height-i//maxwidth-1] = scores[i]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","green"])
        scores_max=torch.max(scores)
        norm=plt.Normalize(-scores_max,scores_max)

        dpi=96
        fig, ax = plt.subplots(figsize=(img_width/dpi, 300/dpi), dpi=dpi)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.5,-0.5+maxwidth)
        ax.set_ylim(-0.5, 0.5+i//maxwidth)
        ax.imshow(im.transpose(),cmap=cmap,norm=norm)
        #,fontproperties= fp
        for i,c in enumerate(line):
            ax.text(i%maxwidth-0.25,im_height-i//maxwidth-0.25-1,c,fontsize=12,fontproperties= fp)
        plt.tight_layout()
        plt.savefig(filename,bbox_inches='tight')



    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    if args.infer:
        for line in sys.stdin:
            infer(line)

    if args.generate:
        import random
        line = ''
        for i in range(args.max_sample_length):
            line_tensor = str_to_tensor([line],args.input_length)
            if args.conditional_model:
                category_onehot = torch.zeros([line_tensor.shape[1], len(all_categories)]).to(device)
                for category in args.category:
                    category_i = all_categories.index(category)
                    category_onehot[0, category_i] = 1
                category_onehot = torch.unsqueeze(category_onehot,0)
                category_onehot = torch.cat(line_tensor.shape[0]*[category_onehot],dim=0)
                input_tensor = torch.cat([line_tensor,category_onehot],dim=2)
            else:
                input_tensor = line_tensor
            _,output_nextchars = model(input_tensor)
            # 3rd order tensor < len(line) x batch_size x len(vocabulary) >
            probs = softmax(args.temperature*output_nextchars[i,:,:])
            dist = torch.distributions.categorical.Categorical(probs)
            nextchar_i = dist.sample()
            nextchar = vocabulary[nextchar_i]
            if nextchar == EOL:
                break
            if nextchar == OOV:
                nextchar='~'
            line += nextchar
        if args.conditional_model:
            print('name=',line)
        else:
            infer(line)