import pickle as pkl

subkg = pkl.load(open('data/e-commerce/subkg.pkl', 'rb'))
entity2entityId = pkl.load(open('data/e-commerce/entity2entityId.pkl', 'rb'))
entityId2entity = dict([(entity2entityId[k], k) for k in entity2entityId])
relation2relationId = pkl.load(open('data/e-commerce/relation2relationId.pkl', 'rb'))
relationId2relation = dict([(relation2relationId[k], k) for k in relation2relationId])
entire_utter_entities,entire_resp_entities = pkl.load(file=open("data/e-commerce/dev_entity_new.pkl", 'rb'))

new_entire_utter_entities_id=[]
for i,utters_entities in enumerate(entire_utter_entities):
    if (i % 100000 == 0):
        print(i)
    new_utters_entities_id=[]
    new_utters_entities_id = [str(entity2entityId[x]) for x in utters_entities if x in entity2entityId]
    new_entire_utter_entities_id.append(new_utters_entities_id)

new_entire_resp_entities_id=[]
for i,resp_entities in enumerate(entire_resp_entities):
    if (i % 100000 == 0):
        print(i)
    new_resp_entities_id=[]
    new_resp_entities_id = [str(entity2entityId[x]) for x in resp_entities if x in entity2entityId]
    new_entire_resp_entities_id.append(new_resp_entities_id)

pkl.dump([new_entire_utter_entities_id,new_entire_resp_entities_id], file=open("data/e-commerce/dev_entity_id.pkl", 'wb'))