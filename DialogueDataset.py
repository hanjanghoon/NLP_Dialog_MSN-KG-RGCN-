import torch
from torch.utils.data import TensorDataset



class DialogueDataset(TensorDataset):

    def __init__(self, X_utterances, X_responses,X_utterances_entites_id,X_reponse_entites_id,X_utterances_entites_len,X_utterances_response_len ,y_labels=None):
        super(DialogueDataset, self).__init__()
        X_utterances = torch.LongTensor(X_utterances)
        X_responses = torch.LongTensor(X_responses)
        X_utterances_entites_id=torch.LongTensor(X_utterances_entites_id)
        X_reponse_entites_id=torch.LongTensor(X_reponse_entites_id)
        X_utterances_entites_len = torch.LongTensor(X_utterances_entites_len)
        X_utterances_response_len = torch.LongTensor(X_utterances_response_len)
        print("X_utterances: ", X_utterances.size())
        print("X_responses: ", X_responses.size())
        print("X_utterances: ", X_utterances_entites_id.size())
        print("X_responses: ", X_reponse_entites_id.size())
        print("X_utterances: ", X_utterances_entites_len.size())
        print("X_responses: ", X_utterances_response_len.size())
        #print("X_utterances: ", len(X_utterances_entites_id))
        #print("X_responses: ", len(X_reponse_entites_id))

        if y_labels is not None:
            y_labels = torch.FloatTensor(y_labels)
            print("y_labels: ", y_labels.size())
            self.tensors = [X_utterances, X_responses,X_utterances_entites_id,X_reponse_entites_id,X_utterances_entites_len,X_utterances_response_len, y_labels]
        else:
            self.tensors = [X_utterances, X_responses,X_utterances_entites_id,X_reponse_entites_id,X_utterances_entites_len,X_utterances_response_len]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])

