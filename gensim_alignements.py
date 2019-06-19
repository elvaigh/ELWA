from gensim.models import TranslationMatrix
from gensim.models.keyedvectors import KeyedVectors


def train_translation(model_source_path, model_target_path, transmat_outpath):    
    word_pairs = []
    source_model = KeyedVectors.load(model_source_path)
    target_model = KeyedVectors.load(model_target_path)

    for word in target_model.wv.vocab:
        if word in source_model.wv.vocab:
            word_pairs.append((word, word))

    trans_model = TranslationMatrix(source_model.wv,
                                    target_model.wv,
                                    word_pairs=word_pairs)

    trans_model.save(transmat_path)
    return trans_model


def predict_translation(emb_path, transmat_path):
    model = KeyedVectors.load(emb_path) # load embedings to translate
    tm = TranslationMatrix.load(transmat_path)  # load translation model
    translated_entities = np.dot(np.array(model.wv), tm.translation_matrix)
    return translated_entities