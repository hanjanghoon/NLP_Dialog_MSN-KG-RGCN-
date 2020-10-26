import spotlight
import pickle
#dd=spotlight.annotate('http://163.239.27.43:2222/rest/annotate','president obama')
#print(dd)
utter,resp= pickle.load(file=open("data/dev_ori.pkl", 'rb'))
utters_entity=[]
resp_entity=[]
for i,(utters,res) in enumerate(zip(utter,resp)):

    if(i%100000==0):
        print(i)
    try:
        sent2entity=spotlight.annotate('http://163.239.27.43:2222/rest/annotate',utters)
        entitis=[]
        if(sent2entity is not None):
            for entity in sent2entity:
                entitis.append('<'+entity["URI"]+'>')
        #print(sent2entity)
        utters_entity.append(entitis)
    except:
        utters_entity.append([])
    try:
        sent2entity = spotlight.annotate('http://163.239.27.43:2222/rest/annotate', res)
        entitis = []
        if (sent2entity is not None):
            for entity in sent2entity:
                entitis.append('<'+entity["URI"]+'>')
        # print(sent2entity)
        resp_entity.append(entitis)
    except:
        resp_entity.append([])


pickle.dump([utters_entity,resp_entity], file=open("data/dev_entity.pkl", 'wb'))