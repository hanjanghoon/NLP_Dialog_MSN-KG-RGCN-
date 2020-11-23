# NLP_Dialog_MSN-KG_RGCN

Mutihop selector + DBpedia(knowledge graph) + RGCN

### Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots
https://www.aclweb.org/anthology/D19-1011.pdf

### Towards Knowledge-Based Recommender Dialog System
https://arxiv.org/pdf/1908.05391.pdf

### DBpedia spotlight
https://www.dbpedia-spotlight.org/

### Modeling Relational Data with Graph Convolutional Networks
https://arxiv.org/pdf/1703.06103.pdf

Ubuntu data V1, Response selection

Multi-hop selector에 knoweledg graph entity 정보를 융합하여 성능 향상시킴

kg로 dbpedia 사용

#### dbpedia_spotlight.py 
ori.pkl에서 utterance 와 response의 문장마다 관련된 entitiy를 annotate api를 통해 가져와서, entity 파일인 train_entity.pkl 파일을 만들어줌.
#### makekg.py
dbpedia의 triple 파일인 mappingbased_objects_en.ttl 을 사용하여 subkg를 만듬. 이때 앞서 구축한 train_entity.pkl 파일을 통해 주어진 entity에 대해서만 subkg를 구축 이때 subkg는 id로 구축되기 때문에 id 파일인 (id2entity,entity2entityId,relation2relationId)와 subkg 파일을 만들어 줌. 이때 subkg는 2번 까지 보게 됨.(2 hop) 구축한 파일은 entity가 약 25000정도 됨. 
#### entitiy2id.py
이 파일은 각각의 utterance 의 entitiy를 id로 변환시켜서 저장해줌. 이는 나중에 id를 통해 관련 entity를 찾아 rgcn 해주기 위한 과정. train_entity_id.pkl은 결국 utterance 와 response에 대한 관련 entity id를 가지게 됨.

#MSN.py
본격적인 알고리즘 구축. torch_geometric.nn.conv.rgcn_con 제공하는 rgcn을 활용하여 구축 이때 edgelist를 통해 relation 중 특정 갯수이상 나타나는 relation을 가진 트리플에 대해서만 구축.
이렇게 구한 entity feature들은 문장 정보와 함께 활용됨. 
Towards Knowledge-Based Recommender Dialog System에서 사용된 Relational Graph Convolutional Networks (R-GCNs)를 적용하여 embedding을 만들어서 기존 score fucntion 과 융합.
