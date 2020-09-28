import sys
import os
import torch
import torch.nn as nn
import transformers

all_layers = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#importing the categories
all_categories = []
emojis = open("list.txt", "rt").read()
all_categories = list(emojis)
print("Number of categories = ", len(all_categories))

model_name = 'bert-base-multilingual-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
if all_layers == False:
    bert = transformers.BertModel.from_pretrained(model_name)
    print('bert.config.vocab_size=',bert.config.vocab_size)
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
    input_ids = torch.cat([ encoding['input_ids'] for encoding in encodings ],dim=0).to(device)
    attention_mask = torch.cat([ encoding['attention_mask'] for encoding in encodings ],dim=0).to(device)
    return input_ids,attention_mask


#torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = BertFineTuning()
model.to(device)
softmax = torch.nn.Softmax(dim=1)
model.eval()
model_dict = torch.load('log_correct/inside/model849946/model')
model.load_state_dict(model_dict['model_state_dict'],strict = False)

#infer list takes in a list of strings, and a number of guesses for the model. 
#It returns a list of tuples each containing the emoji and the percent that the model predicts them for the given set of sentences
#it also returns the count of lines it received
def infer_list(lines,k = 80):
    return_lst = []
    lst_emoji = [] 
    lst_perc = []
    if k>80:
        k=80
    if k<0:
        k=1
    for line in lines:
        #print(line)
        line = line.strip()
        line_tensor = str_to_tensor_bert([line])
        output_class,output_nextchars = model(line_tensor)
        probs = softmax(output_class)
        top_n, top_i = probs.topk(k)
        for i in range(k):
            guess = all_categories[top_i[0,i].item()]
            percent = '%0.04f'%top_n[0,i].item()
            lst_emoji.append(guess)
            lst_perc.append(percent)
        return_lst.append(dict(zip(lst_emoji, lst_perc)))
        lst_emoji.clear()
        lst_perc.clear()
        
    return return_lst

'''
    from collections import Counter
    c = Counter(lst_occur)
    lst_return =[(i, c[i] / len(lst_occur) * 100.0) for i in c]
    return lst_return, len(lines)
'''


def infer_line(line,k):
    return_list = []
    if k is None:
        k=80
    if k>80:
        k=80
    if k<0:
        k=1
    line = line.strip()
    line_tensor = str_to_tensor_bert([line])
    output_class,output_nextchars = model(line_tensor)
    probs = softmax(output_class)
    top_n, top_i = probs.topk(k)
    for i in range(k):
        guess = all_categories[top_i[0,i].item()]
        lst_emoji.append(guess) 
        lst_perc.append('%0.04f'%top_n[0,i].item())
        
    return lst_emoji,lst_perc

#sentence = 'this day sucks'
#x,y = infer_line(sentence,)

#print(x,y)
'''
import pprint
x = infer_list(['I think I am sick, I am coughing and feeling tired. I hope it is not coronavirus. Wear your mask. #pandemic #sick #covid',
'Acho que estou doente, tossindo e me sentindo cansada. Espero que não seja coronavírus. Use sua máscara. #pandemic #sick #covid',
'Creo que estoy enfermo, tengo tos y me siento cansado. Espero que no sea coronavirus. Use su máscara. #pandemia #enfermos #covid',
'私は病気だと思います、私は咳をしていて疲れています。コロナウイルスでないことを願っています。マスクを着用してください。 ＃パンデミック＃病気#covid',
'Saya pikir saya sakit, saya batuk dan merasa lelah. Saya harap ini bukan virus corona. Kenakan topeng Anda. #pandemi #sick #covid',
'أعتقد أنني مريض ، أسعل وأشعر بالتعب. آمل ألا يكون فيروس كورونا. ارتدي قناعك. # وباء # مريض # كوفيد',
'Sa palagay ko ay may sakit ako, umuubo ako at nakakapagod. Sana hindi ito coronavirus. Isuot ang iyong maskara. #pandemic #sick #covid',
'Sanırım hastayım, öksürüyorum ve yorgun hissediyorum. Umarım koronavirüs değildir. Maskeni tak. #pandemik #hasta #covid',
'Je pense que je suis malade, je tousse et je me sens fatigué. J espère que ce ne est pas un coronavirus. Portez votre masque. #pandémique #sick #covid',
'मुझे लगता है कि मैं बीमार हूं, मुझे खांसी हो रही है और थकान महसूस हो रही है। मुझे आशा है कि यह कोरोनावायरस नहीं है। अपना मास्क पहनें। #महामारी #लड़की #समाज',
'ฉันคิดว่าฉันไม่สบายฉันไอและรู้สึกเหนื่อย ฉันหวังว่ามันจะไม่ใช่ coronavirus สวมหน้ากาก. #โรคระบาด #ป่วย #โควิด',
'Penso di essere malato, sto tossendo e mi sento stanco. Spero non sia il coronavirus. Indossa la tua maschera. #pandemia #malato #covido',
'Ik denk dat ik ziek ben, ik hoest en voel me moe. Ik hoop dat het geen coronavirus is. Draag je masker. #pandemisch #ziek #covid',
'Мне кажется, что я болен, кашляю и чувствую усталость. Надеюсь, это не коронавирус. Наденьте маску. #pandemic #sick #covid',
'Νομίζω ότι είμαι άρρωστος, βήχα και αισθάνομαι κουρασμένος. Ελπίζω ότι δεν είναι κοροναϊός. Φορέστε τη μάσκα σας. #πανδημία #άρρωστο #covid'],7)
'''
import pprint

y = ['Washington man Is 1st in US to Catch Newly Discovered Dangerous Pneumonia. Get out your face masks folks! #coronavirus #wuhan #mask',
'Washington Man é o primeiro nos EUA a pegar pneumonia perigosa recém-descoberta. Tirem suas máscaras, pessoal! #coronavirus #wuhan #mask',
'El hombre de Washington es el primero en EE. UU. En contraer una neumonía peligrosa recién descubierta. ¡Saquen sus mascarillas, amigos! #coronavirus #wuhan #mascara',
'ワシントンマンは、新しく発見された危険な肺炎をキャッチするために米国で最初です。フェイスマスクの人を出してください！ ＃コロナウイルス＃武漢＃マスク',
'واشنطن مان هو الأول في الولايات المتحدة للإصابة بالالتهاب الرئوي الخطير المكتشف حديثًا. اخرجوا أقنعة الوجه يا جماعة! # فيروس_كورونا # ووهان # قناع',
'Ang Washington Man ay Ika-1 sa US upang Makibalita ng Bagong Nakatuklas na Mapanganib na pneumonia. Lumabas ang iyong mga maskara sa mukha mga kamag-anak! #coronavirus #wuhan #mask',
'Washington adamı, Yeni Keşfedilen Tehlikeli Zatürree\'yi ABD\'de Yakalayan İlk Kişi. Yüz maskelerinizi çıkarın millet! #coronavirus #wuhan #maske',
'L\'homme de Washington est le premier aux États-Unis à attraper une pneumonie dangereuse nouvellement découverte. Sortez vos masques! #coronavirus #wuhan #mask',
'ชายชาววอชิงตันเป็นคนแรกในสหรัฐฯที่จับโรคปอดบวมอันตรายที่เพิ่งค้นพบ กำจัดคนมาสก์หน้าของคุณ! #โคโรนาไวรัส #วูฮาน #หน้ากาก',
'वॉशिंगटन मैन इज न्यू फर्स्ट डिसाइडेड डेंजरस न्यूमोनिया को पकड़ने के लिए अमेरिका में प्रथम स्थान पर है। अपने चेहरे मास्क लोगों को बाहर निकालो! #कोरोनोवायरस #वुहान #मास्क',
'L\'uomo di Washington è il primo negli Stati Uniti a contrarre una polmonite pericolosa scoperta di recente. Tira fuori le maschere per il viso, gente! #coronavirus #wuhan #mask'
'Washington-man is de eerste in de VS die nieuw ontdekte gevaarlijke longontsteking oploopt. Haal je gezichtsmaskers tevoorschijn, mensen! #coronavirus #wuhan #masker',
'Мужчина из Вашингтона первым в США заразился недавно обнаруженной опасной пневмонией. Убирайтесь, ребята, маски! #коронавирус #ухань #маска',
'Ο Ουάσινγκτον είναι 1ος στις ΗΠΑ για να πιάσει πρόσφατα ανακάλυψη επικίνδυνη πνευμονία. Βγείτε τους μάσκες προσώπου σας! #coronavirus #wuhan #mask',
'Der Mann aus Washington ist der erste in den USA, der eine neu entdeckte gefährliche Lungenentzündung bekommt. Holen Sie sich Ihre Gesichtsmasken Leute! #coronavirus #wuhan #mask',
'איש וושינגטון הוא הראשון בארה"ב לתפוס דלקת ריאות מסוכנת שהתגלתה לאחרונה. צא מסכות הפנים שלך אנשים! #coronavirus #wuhan # מסכה',
'워싱턴 남자는 새로 발견 된 위험한 폐렴을 미국에서 처음으로 잡았습니다. 당신의 얼굴 마스크 사람들을 꺼내십시오! # 코로나 바이러스 # 무한 # 마스크',
'Człowiek z Waszyngtonu jest pierwszym w USA, który złapał nowo odkryte niebezpieczne zapalenie płuc. Zdejmijcie maski na twarz! #coronavirus #wuhan #mask',
'Pria Washington adalah yang pertama di AS untuk Menangkap Pneumonia Berbahaya yang Baru Ditemukan. Keluarkan masker wajah kalian! #coronavirus #wuhan #mask' ]


z = ['Washington sick man Is 1st in US to Catch Newly Discovered Dangerous Pneumonia. Get out your face masks folks! #coronavirus #wuhan #mask',
'Homem doente em Washington é o primeiro nos Estados Unidos a pegar pneumonia perigosa recém-descoberta. Tirem suas máscaras, pessoal! #coronavirus #wuhan #mask',
'Un enfermo de Washington es el primero en los Estados Unidos en contraer una neumonía peligrosa recién descubierta. ¡Saquen sus mascarillas, amigos! #coronavirus #wuhan #mascara',
'ワシントンの病人は新しく発見された危険な肺炎を捕まえるために米国で最初です。フェイスマスクの人を出してください！ ＃コロナウイルス＃武漢＃マスク',
'رجل مريض بواشنطن هو الأول في الولايات المتحدة الذي يصاب بالالتهاب الرئوي الخطير المكتشف حديثًا. اخرجوا أقنعة الوجه يا جماعة! #فيروس_كورونا #ووهان #قناع',
'Ang taong may sakit sa Washington ay Ika-1 sa Estados Unidos upang Makuha ang Bagong Nakatuklas na Mapanganib na pneumonia. Lumabas ang iyong mga maskara sa mukha mga kamag-anak! #coronavirus #wuhan #mask',
'Washingto\'daki hasta adam, ABD\'de Yeni Keşfedilen Tehlikeli Zatürreyi Yakalayan İlk Kişi. Yüz maskelerinizi çıkarın millet! #coronavirus #wuhan #maske',
'Un homme malade de Washington est le premier aux États-Unis à attraper une pneumonie dangereuse nouvellement découverte. Sortez vos masques! #coronavirus #wuhan #mask',
'หนุ่มป่วยในวอชิงตันเป็นคนแรกในสหรัฐฯที่จับโรคปอดบวมอันตรายที่เพิ่งค้นพบ กำจัดคนมาสก์หน้าของคุณ! #โคโรนาไวรัส #วูฮาน #หน้ากาก',
'वाशिंगटन बीमार आदमी अमेरिका में 1 है। न्यूली डिस्कवरी खतरनाक न्यूमोनिया को पकड़ने के लिए। अपने चेहरे मास्क लोगों को बाहर निकालो! #कोरोनोवायरस #वुहान #मास्क',
'Un malato di Washington è il primo negli Stati Uniti a contrarre una polmonite pericolosa scoperta di recente. Tira fuori le tue maschere per il viso, gente! #coronavirus #wuhan #mask',
'Больной из Вашингтона первым в США заразился недавно обнаруженной опасной пневмонией. Убирайтесь, ребята, маски! #коронавирус #ухань #маска',
'Ο άντρας της Ουάσινγκτον είναι ο 1ος στις ΗΠΑ για να πιάσει πρόσφατα ανακάλυψη επικίνδυνη πνευμονία. Βγείτε τους μάσκες προσώπου σας! #coronavirus #wuhan #μάσκα',
'Der kranke Mann aus Washington ist der erste in den USA, der an einer neu entdeckten gefährlichen Lungenentzündung erkrankt. Holen Sie sich Ihre Gesichtsmasken Leute! #coronavirus #wuhan #mask',
'איש חולה בוושינגטון הוא הראשון בארה"ב לתפוס דלקת ריאות שהתגלתה לאחרונה. צא מסכות הפנים שלך אנשים! #coronavirus #wuhan #מסכה',
'워싱턴의 아픈 남자는 미국에서 처음으로 새로 발견 된 위험한 폐렴을 잡았습니다. 당신의 얼굴 마스크 사람들을 꺼내십시오! #코로나 바이러스 #무한 #마스크',
'Chory z Waszyngtonu jest pierwszym w USA, który złapał nowo odkryte niebezpieczne zapalenie płuc. Zdejmijcie maski na twarz! #coronavirus #wuhan #mask',
'Orang sakit Washington adalah yang pertama di AS untuk Menangkap Pneumonia Berbahaya yang Baru Ditemukan. Keluarkan masker wajah kalian! #coronavirus #wuhan #mask'
]

j = ['How do we tolerate 3000 Americans dying everyday from #COVID19? THREE. THOUSAND. EVERY. DAY.']
for i in infer_list(j,10):
    print(i)
'''
lsttt = []
for i in x:
    for key in i.keys():
        lsttt.append(key)
    print(lsttt)
    lsttt.clear()
'''