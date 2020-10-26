import pickle
from collections import defaultdict

def _load_kg(path):
    kg = defaultdict(list)
    with open(path) as f:
        for line in f.readlines():
            tuples = line.split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                # TODO: include property/publisher and subject/year, etc
                if "ontology" in r:
                    kg[h].append((r, t))
    return kg


def _extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)
    subkg_hrt = set()
    #전체 kg 중에서 movie item list와 관련있는것만 뽑아서 만든다.
    ripple_set = []
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = seed_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                h, r, t = entity, tail_and_relation[0], tail_and_relation[1]
                if (h, r, t) not in subkg_hrt:
                    subkg[h].append((r, t))
                    subkg_hrt.add((h, r, t))
                memories_h.append(h)
                memories_r.append(r)
                memories_t.append(t)

        ripple_set.append((memories_h, memories_r, memories_t))

    return subkg


def build(filename):
    entire_utter_entities,entire_resp_entities = pickle.load(file=open(filename, 'rb'))

    id2entity={}
    i=1
    for resp_entities in entire_resp_entities:
        for resp_entity in resp_entities:
            if resp_entity not in id2entity.values():
                id2entity[i]=resp_entity
                i+=1
    kg = _load_kg("data/mappingbased_objects_en.ttl")
    subkg = _extract_subkg(
        kg,
        [
            id2entity[k]
            for k in id2entity
            if id2entity[k] is not None and kg[id2entity[k]] != []
        ],
        2,
    )
    entities = set([k for k in subkg]) | set([x[1] for k in subkg for x in subkg[k]])
    entity2entityId = dict([(k, i) for i, k in enumerate(entities)])
    relations = set([x[0] for k in subkg for x in subkg[k]])
    relation2relationId = dict([(k, i) for i, k in enumerate(relations)])
    subkg_idx = defaultdict(list)
    for h in subkg:
        for r, t in subkg[h]:
            subkg_idx[entity2entityId[h]].append((relation2relationId[r], entity2entityId[t]))

    pickle.dump(id2entity, open("data/e-commerce/id2entity.pkl", "wb"))
    pickle.dump(subkg_idx, open("data/e-commerce/subkg.pkl", "wb"))
    pickle.dump(entity2entityId, open("data/e-commerce/entity2entityId.pkl", "wb"))
    pickle.dump(relation2relationId, open("data/e-commerce/relation2relationId.pkl", "wb"))
    print('end')

build("data/e-commerce/train_entity_new.pkl")