import pickle
def read_list(list):
    # for reading also binary mode is important
    with open(list, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

gottenWeights = read_list("FromScratch\weights.txt")
gottenBiases = read_list("FromScratch\\biases.txt")
# print(gottenWeights)
# print(gottenBiases)