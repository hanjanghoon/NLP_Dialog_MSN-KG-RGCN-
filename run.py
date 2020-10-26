import time
import argparse
import pickle
from MSN import MSN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

task_dic = {
    'ubuntu':'./dataset/ubuntu_data/',
    'douban':'./dataset/DoubanConversaionCorpus/',
    'alime':'./dataset/E_commerce/'
}
data_batch_size = {
    "ubuntu": 200,
    "douban": 150,
    "alime":  200
}

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--batch_size",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
args = parser.parse_args()
args.batch_size = data_batch_size[args.task]
args.save_path += args.task + '.' + MSN.__name__ + ".pt"
args.score_file_path = task_dic[args.task] + args.score_file_path

print(args)
print("Task: ", args.task)


def train_model():
    path = task_dic[args.task]
    X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+"train.pkl", 'rb'))
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    #idx2sentnece(X_train_utterances, X_train_responses, X_dev_utterances, X_dev_responses, vocab, y_train)


    #convert_entityidx_str2int_withlen(path)


    X_train_utterances_entites_id,X_train_responses_entites_id,X_train_utterances_entities_length,X_train_responses_entities_length = pickle.load(file=open(path + "KB/train_entity_final.pkl", 'rb'))
    X_dev_utterances_entites_id, X_dev_responses_entites_id, X_dev_utterances_entities_length, X_dev_responses_entities_length = pickle.load(file=open(path + "KB/dev_entity_final.pkl", 'rb'))
    '''

    train_toy=10000
    dev_toy=10000

    X_train_utterances = X_train_utterances[:train_toy]
    X_train_responses = X_train_responses[:train_toy]
    y_train = y_train[:train_toy]
    X_train_utterances_entites_id = X_train_utterances_entites_id[:train_toy]
    X_train_responses_entites_id = X_train_responses_entites_id[:train_toy]
    X_train_utterances_entities_length = X_train_utterances_entities_length[:train_toy]
    X_train_responses_entities_length = X_train_responses_entities_length[:train_toy]

    X_dev_utterances= X_dev_utterances[:dev_toy]
    X_dev_responses=X_dev_responses[:dev_toy]
    y_dev=y_dev[:dev_toy]
    X_dev_utterances_entites_id = X_dev_utterances_entites_id[:dev_toy]
    X_dev_responses_entites_id = X_dev_responses_entites_id[:dev_toy]
    X_dev_utterances_entities_length = X_dev_utterances_entities_length[:dev_toy]
    X_dev_responses_entities_length = X_dev_responses_entities_length[:dev_toy]

    '''

    model = MSN(word_embeddings, args=args)

    model.fit(
        X_train_utterances, X_train_responses, y_train,
        X_train_utterances_entites_id,X_train_responses_entites_id,X_train_utterances_entities_length,X_train_responses_entities_length,
        #X_dev_utterances, X_dev_responses, y_dev,
        #X_dev_utterances_entites_id, X_dev_responses_entites_id, X_dev_utterances_entities_length,X_dev_responses_entities_length,

        X_dev_utterances_entites_id, X_dev_responses_entites_id, X_dev_utterances_entities_length,X_dev_responses_entities_length,
        X_dev_utterances, X_dev_responses, y_dev
    )



def idx2sentnece(train_u, train_r, dev_u, dev_r, vocab, y_train):

    # tokenized_texts=tokenized_texts = [bert_tokenizer.tokenize("i am hppy")]
    # print (tokenized_texts[0])

    reverse_vocab = {v: k for k, v in vocab.items()}

    train_bu = []  # 총 백만.
    for i, context in enumerate(train_u):  # context len =10
        context_b = []
        if (i % 100000 == 0):
            print(i)

        for utterance in context:  # utterance max =50
            utterance_b = ""
            for word_idx in utterance:
                if (word_idx == 0): continue
                utterance_b += reverse_vocab[word_idx] + " "
            if (len(utterance_b) == 0):
                continue

            utterance_b = utterance_b[:-1]
            # print(utterance_b)


            # utterance_t+= [0 for i in range(50-len(utterance_t))]#맥스 단어가 50임 빠끄
            context_b.append(utterance_b)
        train_bu.append(context_b)

    train_br = []

    for utterance, y in zip(train_r, y_train):  # utterance max =1문장
        utterance_b = ""
        for word_idx in utterance:
            if (word_idx == 0): continue
            utterance_b += reverse_vocab[word_idx] + " "
        '''
        if (len(utterance_b) == 0):#백만개에서 줄어듬......
            print("response missing!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        '''
        utterance_b = utterance_b[:-1]
        # utterance_t += [0 for i in range(50 - len(utterance_t))]
        train_br.append(utterance_b)
        # print(utterance_t)
    print("end")
    pickle.dump([train_bu, train_br], file=open("sentence/e_commerce/train_ori.pkl", 'wb'))

    dev_bu = []  # 총 백만.
    for context in dev_u:  # context len =10
        context_b = []
        for utterance in context:  # utterance max =50
            utterance_b = ""
            for word_idx in utterance:
                if (word_idx == 0): continue
                utterance_b += reverse_vocab[word_idx] + " "

            if (len(utterance_b) == 0):
                continue
            utterance_b = utterance_b[:-1]
            # print(utterance_b)

            # utterance_t += [0 for i in range(50 - len(utterance_t))]
            context_b.append(utterance_b)
        dev_bu.append(context_b)

    dev_br = []
    for utterance in dev_r:  # utterance max =1문장
        utterance_b = ""
        for word_idx in utterance:
            if (word_idx == 0): continue
            utterance_b += reverse_vocab[word_idx] + " "
        '''
        if (len(utterance_b) == 0):
            continue
        '''
        utterance_b = utterance_b[:-1]
        # utterance_t += [0 for i in range(50 - len(utterance_t))]
        dev_br.append(utterance_b)

    pickle.dump([dev_bu, dev_br], file=open("sentence/e_commerce/dev_ori.pkl", 'wb'))

def convert_entityidx_str2int_withlen(path):
    X_train_utterances_entites_id, X_train_responses_entites_id = pickle.load(file=open(path + "KB/entity_id.pkl", 'rb'))
    X_train_utterances_entities_length = []
    X_train_responses_entities_length = []
    for i in range(1000000):
        if (len(X_train_utterances_entites_id[i]) > 20):
            X_train_utterances_entites_id[i] = X_train_utterances_entites_id[i][:20]
            X_train_utterances_entities_length.append(20)
        else:
            X_train_utterances_entities_length.append(len(X_train_utterances_entites_id[i]))
            X_train_utterances_entites_id[i].extend([0] * (20 - len(X_train_utterances_entites_id[i])))

        X_train_utterances_entites_id[i] = list(map(int, X_train_utterances_entites_id[i]))
        if (len(X_train_responses_entites_id[i]) > 10):
            X_train_responses_entites_id[i] = X_train_responses_entites_id[i][:10]
            X_train_responses_entities_length.append(10)
        else:
            X_train_responses_entities_length.append(len(X_train_responses_entites_id[i]))
            X_train_responses_entites_id[i].extend([0] * (10 - len(X_train_responses_entites_id[i])))

        X_train_responses_entites_id[i] = list(map(int, X_train_responses_entites_id[i]))

    X_dev_utterances_entites_id, X_dev_responses_entites_id = pickle.load(file=open(path + "KB/dev_entity_id.pkl", 'rb'))
    X_dev_utterances_entities_length = []
    X_dev_responses_entities_length = []
    for i in range(500000):
        if (len(X_dev_utterances_entites_id[i]) > 20):
            X_dev_utterances_entites_id[i] = X_dev_utterances_entites_id[i][:20]
            X_dev_utterances_entities_length.append(20)
        else:
            X_dev_utterances_entities_length.append(len(X_dev_utterances_entites_id[i]))
            X_dev_utterances_entites_id[i].extend([0] * (20 - len(X_dev_utterances_entites_id[i])))

        X_dev_utterances_entites_id[i] = list(map(int, X_dev_utterances_entites_id[i]))
        if (len(X_dev_responses_entites_id[i]) > 10):
            X_dev_responses_entites_id[i] = X_dev_responses_entites_id[i][:10]
            X_dev_responses_entities_length.append(10)
        else:
            X_dev_responses_entities_length.append(len(X_dev_responses_entites_id[i]))
            X_dev_responses_entites_id[i].extend([0] * (10 - len(X_dev_responses_entites_id[i])))

        X_dev_responses_entites_id[i] = list(map(int, X_dev_responses_entites_id[i]))
    pickle.dump([ X_train_utterances_entites_id,X_train_responses_entites_id,X_train_utterances_entities_length,X_train_responses_entities_length], file=open(path + "KB/train_entity_final.pkl", 'wb'))
    pickle.dump([ X_dev_utterances_entites_id,X_dev_responses_entites_id,X_dev_utterances_entities_length,X_dev_responses_entities_length], file=open(path + "KB/dev_entity_final.pkl", 'wb'))

def test_model():
    path = task_dic[args.task]
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

def test_adversarial():
    path = task_dic[args.task]
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    print("adversarial test set (k=1): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_1.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=2): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_2.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=3): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_3.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end-start)/60, " min")




