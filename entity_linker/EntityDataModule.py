import msgpack
import msgpack_numpy as m
import pandas as pd
import numpy as np
from DataGenerator import generate
from loguru import logger

import torch as th
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader


class EntityLinkingDataModule(LightningDataModule):
    def __init__(self, extractor, batch_size):
        super().__init__()
        self.entity_extractor = extractor
        self.batch_size = batch_size

    def prepare_data(self, *args, **kwargs):
        context = self.prepare_dataset(self.doc_embeddings_table)
        self.data = []
        for df, en_positive, doc_vec in context:
            for en_seqs, x, y in self.prepare_doc(df, en_positive, doc_vec):
                if y != []:
                    self.data.append((x, y))
        #             X.append(x)
        #             Y.append(y)
        # return th.tensor(X), th.tensor(Y, dtype=th.float32)
        # return super().prepare_data(*args, **kwargs)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train, self.val = random_split(self.data, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.data
            # MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def prepare_dataset(self, embedding_name):
        yield from self.prepare_relation_dataset()
        yield from self.prepare_corpus_dataset(embedding_name)

    def prepare_relation_dataset(self):
        """generation of relations having ambiguous entities"""
        logger.info("generation of relations having ambiguous entities")
        # pour chaque entité associée à une forme ambigue
        # on récupère tous les vecteurs-entités associés a la même formes
        # on récupère les vecteurs de ses entités en relation
        # X, Y = [], []

        for en_id, ambiguities, linked, _formes in generate("ambiguity"):
            linked = linked.split()
            ambiguities = ambiguities.split()
            # extracted_entities = [en_id] + ambiguities + linked
            extracted_entities = [{"start": 0, "end": 1, "id": en_id}]
            extracted_entities += [
                {"start": 0, "end": 1, "id": e_id} for e_id in ambiguities
            ]
            extracted_entities += [
                {"start": i + 1, "end": i + 2, "id": e_id}
                for i, e_id in enumerate(linked)
            ]
            df = pd.DataFrame(extracted_entities)
            doc_vec = np.zeros(100)
            # il faut générer ici un doc_vecteur pour cette entité
            # une moyenne des documents en lien avec l'entité ambigue ?
            # ou un doc vecteur des formes entités
            # UTILISER LES FORMES DU TRIE + WORD EMBEDDINGS ICI
            en_positive = [en_id] + linked
            yield df, en_positive, doc_vec

            # for en_seqs, x, y in self.prepare_doc(df, en_positive, doc_vec):
            #     if y != []:
            #         yield x, y

        # Gérer ici le cas des documents/contexte vide
        # doc_emb = ctx_emb = wemb(en), avg(wemb(en_i))
        # ent_emb = pemb(en), avg(wemb(en_i))

    def prepare_corpus_dataset_new(self, embedding_name):
        """Transforme un ensemble de documents+entités en une matrice donnée en entré d'un algorithme de classification"""
        # x = np.concatenate(Parallel(n_jobs=3)(delayed(self.doc_encoder.transform)([doc]) for doc in x))
        from concurrent.futures import ThreadPoolExecutor

        def get_doc_example(data):
            doc_id, doc, doc_vec, en_positive = data
            en_positive = en_positive.split()
            doc_vec = msgpack.unpackb(doc_vec, object_hook=m.decode)
            extracted_ens = self.entity_extractor.get_entities_dataframe(doc)
            return extracted_ens, en_positive, doc_vec

        table_mapping = {"doc_embeddings": embedding_name}
        with ThreadPoolExecutor(4) as ex:
            yield from ex.map(get_doc_example, generate("annotation", table_mapping))

    def prepare_corpus_dataset(self, embedding_name):
        """Transforme un ensemble de documents+entités en une matrice donnée en entré d'un algorithme de classification"""
        table_mapping = {"doc_embeddings": embedding_name}
        for _doc_id, doc, doc_vec, en_positive in generate("annotation", table_mapping):
            en_positive = en_positive.split()
            doc_vec = msgpack.unpackb(doc_vec, object_hook=m.decode)
            extracted_ens = self.entity_extractor.get_entities_dataframe(doc)
            yield extracted_ens, en_positive, doc_vec

    def prepare_wiki_dataset(self):
        """Transforme un ensemble de documents+entités en une matrice donnée en entrée d'un algorithme de classification"""

        nb_examples = 0
        nb_docs = 0
        X, Y = [], []
        for source, entities, formes in generate("annotation_wiki"):
            if entities is None:
                continue
            en_positive = entities.split("|")
            en_positive.append(source)
            extracted_ens = self.entity_extractor.get_entities_dataframe(formes)

            for en_seqs, x, y in self.prepare_doc(
                extracted_ens, en_positive, np.array([])
            ):
                if y != []:
                    X.append(x)
                    Y.append(y)
                    nb_examples += 1
            nb_docs += 1
            if nb_docs > 20000:
                break

        return np.array(X), np.array(Y, dtype=np.float32)


if __name__ == "__main__":
    import fire

    fire.Fire(EntityLinkingDataModule)
